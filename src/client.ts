import * as https from 'https';
import * as http from 'http';
import { URL } from 'url';
import { promisify } from 'util';
import { gzip, gunzip, brotliCompress, brotliDecompress } from 'zlib';
import { performance } from 'perf_hooks';
import { EventEmitter } from 'events';
import { cpus } from 'os';

import {
	ClientConfig,
	VectorRecord,
	CreateCollectionParams,
	CollectionInfo,
	ApiResponse,
	UpsertResponse,
	HealthResponse,
	InfoResponse,
	RequestOptions,
	Logger
} from '../shared/types';

import { VectorOperations } from "./vector_ops";
import { QueryCache, BufferPool } from './caching';

const gzipAsync = promisify(gzip);
const gunzipAsync = promisify(gunzip);
const brotliCompressAsync = promisify(brotliCompress);
const brotliDecompressAsync = promisify(brotliDecompress);

class ModuleLogger implements Logger {
	constructor(private enabled: boolean = false) { }

	debug(message: string, ...args: any[]): void {
		if (this.enabled) console.debug(`[DEBUG] ${message}`, ...args);
	}

	info(message: string, ...args: any[]): void {
		if (this.enabled) console.info(`[INFO] ${message}`, ...args);
	}

	warn(message: string, ...args: any[]): void {
		if (this.enabled) console.warn(`[WARN] ${message}`, ...args);
	}

	error(message: string, ...args: any[]): void {
		if (this.enabled) console.error(`[ERROR] ${message}`, ...args);
	}
}

const globalClients = new Set<Client>();
let globalSigintHandlerInstalled = false;

function installGlobalSigintHandler() {
	if (globalSigintHandlerInstalled) return;
	globalSigintHandlerInstalled = true;

	const cleanup = async () => {
		const cleanupPromises = Array.from(globalClients).map(async (client) => {
			try {
				await client.close();
			} catch (error) { }
		});

		try {
			await Promise.race([
				Promise.all(cleanupPromises),
				new Promise(resolve => setTimeout(resolve, 2000))
			]);
		} catch (error) { }

		process.exit(0);
	};

	process.on('SIGINT', cleanup);
	process.on('SIGTERM', cleanup);
}

export class Client extends EventEmitter {
	private readonly host: string;
	private readonly config: Required<ClientConfig>;
	private readonly logger: Logger;
	private readonly queryCache: QueryCache;
	private readonly bufferPool: BufferPool;
	private readonly collectionCache = new Map<string, CollectionInfo>();
	private requestCount = 0;
	private requestTimes: number[] = [];
	private readonly startTime = Date.now();
	private cleanupInterval?: NodeJS.Timeout;
	private isDestroyed = false;
	private pendingRequests = new Set<AbortController>();

	constructor(host: string, config: ClientConfig = {}) {
		super();

		this.host = host.replace(/\/$/, '');
		const cpuCount = cpus().length;

		this.config = {
			timeout: config.timeout ?? 120,
			connectionPoolSize: config.connectionPoolSize ?? Math.max(cpuCount * 5, 20),
			retryAttempts: config.retryAttempts ?? 5,
			compression: config.compression ?? true,
			debug: config.debug ?? false,
			cacheSize: config.cacheSize ?? 1000,
			threadPoolSize: config.threadPoolSize ?? Math.max(cpuCount * 2, 8),
			cacheTtl: config.cacheTtl ?? 300
		};

		this.logger = new ModuleLogger(this.config.debug);
		this.queryCache = new QueryCache(this.config.cacheSize, this.config.cacheTtl * 1000);
		this.bufferPool = new BufferPool(50);
		this.cleanupInterval = setInterval(() => this.cleanup(), 30000);

		globalClients.add(this);
		installGlobalSigintHandler();

		this.on('error', (error) => {
			this.logger.error('Client error:', error);
		});
	}

	async createCollection(params: CreateCollectionParams): Promise<ApiResponse> {
		const payload = {
			name: params.name,
			dimensions: params.dimensions,
			enable_hnsw: params.enableHnsw ?? true,
			shards: params.shards ?? 16,
			m: params.m ?? 16,
			ef_construction: params.efConstruction ?? 200
		};

		const response = await this.request<ApiResponse>({
			method: 'POST',
			path: '/collections',
			body: JSON.stringify(payload)
		});

		if (response.success) {
			this.collectionCache.set(params.name, {
				name: params.name,
				dimensions: params.dimensions,
				enableHnsw: params.enableHnsw ?? true,
				shards: params.shards ?? 16,
				m: params.m ?? 16,
				efConstruction: params.efConstruction ?? 200
			});
		}

		return response;
	}

	private async getCollectionDimensions(collectionName: string): Promise<number | undefined> {
		const cached = this.collectionCache.get(collectionName);
		if (cached?.dimensions) {
			return cached.dimensions;
		}

		try {
			const info = await this.request<CollectionInfo>({
				method: 'GET',
				path: `/collections/${collectionName}`
			});

			if (info.dimensions) {
				this.collectionCache.set(collectionName, info);
				return info.dimensions;
			}
		} catch (error) {
			this.logger.warn(`Could not get dimensions for collection ${collectionName}:`, error);
		}

		return undefined;
	}

	private async validateVectorDimensions(collectionName: string, vector: number[]): Promise<boolean> {
		const expectedDims = await this.getCollectionDimensions(collectionName);
		if (expectedDims === undefined) return true;

		if (vector.length !== expectedDims) {
			throw new Error(
				`Vector dimension mismatch: got ${vector.length}, expected ${expectedDims} for collection '${collectionName}'`
			);
		}

		return true;
	}

	async upsert(
		collectionName: string,
		records: VectorRecord[],
		options: {
			batchSize?: number;
			showProgress?: boolean;
			parallelism?: number;
			validateDimensions?: boolean;
		} = {}
	): Promise<UpsertResponse> {
		if (!records.length) {
			return { upserted_count: 0 };
		}

		const { validateDimensions = true } = options;

		if (validateDimensions && records.length > 0) {
			const expectedDims = await this.getCollectionDimensions(collectionName);
			if (expectedDims !== undefined) {
				for (let i = 0; i < Math.min(records.length, 10); i++) {
					const vector = records[i].values || records[i].vector || [];
					if (vector.length !== expectedDims) {
						throw new Error(
							`Vector dimension mismatch in record ${i}: got ${vector.length}, expected ${expectedDims} for collection '${collectionName}'`
						);
					}
				}
			}
		}

		const vectorOps = this.vectorOperations(collectionName);
		return vectorOps.upsert(records, options);
	}

	vectorOperations(collectionName: string = 'default'): VectorOperations {
		return new VectorOperations(
			this.host,
			collectionName,
			{
				request: this.request.bind(this),
				queryCache: this.queryCache,
				bufferPool: this.bufferPool,
				logger: this.logger,
				collectionCache: this.collectionCache,
				getCollectionDimensions: this.getCollectionDimensions.bind(this),
				validateVectorDimensions: this.validateVectorDimensions.bind(this)
			}
		);
	}

	vector_operations(collectionName: string = 'default'): VectorOperations {
		return this.vectorOperations(collectionName);
	}

	async commit(): Promise<ApiResponse> {
		return this.request<ApiResponse>({
			method: 'POST',
			path: '/admin/commit'
		});
	}

	async health(): Promise<HealthResponse> {
		return this.request<HealthResponse>({
			method: 'GET',
			path: '/health',
			timeout: 10
		});
	}

	async info(): Promise<InfoResponse> {
		const cacheKey = 'server_info';
		const cached = this.queryCache.get(cacheKey);
		if (cached?.matches) {
			return cached.matches as any;
		}

		return this.request<InfoResponse>({
			method: 'GET',
			path: '/info',
			timeout: 10
		});
	}

	async clearCache(): Promise<{ success: boolean; message: string }> {
		this.queryCache.clear();
		this.collectionCache.clear();
		return { success: true, message: 'All caches cleared' };
	}

	async request<T = any>(options: RequestOptions): Promise<T> {
		if (this.isDestroyed) {
			throw new Error('Client has been destroyed');
		}

		const { method, path, headers = {}, body, timeout = this.config.timeout } = options;
		let lastError: Error;

		for (let attempt = 0; attempt < this.config.retryAttempts; attempt++) {
			const controller = new AbortController();
			this.pendingRequests.add(controller);

			try {
				const startTime = performance.now();

				const result = await this.performRequest<T>({
					method,
					path,
					headers: {
						'Content-Type': 'application/json',
						'Accept': 'application/json',
						...(this.config.compression && { 'Accept-Encoding': 'gzip, br' }),
						...headers
					},
					body,
					timeout: timeout * 1000
				});

				const duration = performance.now() - startTime;
				this.requestCount++;
				this.requestTimes.push(duration);

				if (this.requestTimes.length > 1000) {
					this.requestTimes = this.requestTimes.slice(-500);
				}

				this.pendingRequests.delete(controller);
				return result;

			} catch (error) {
				this.pendingRequests.delete(controller);
				lastError = error as Error;
				this.logger.debug(`Request attempt ${attempt + 1} failed:`, error);

				if (attempt < this.config.retryAttempts - 1) {
					const delay = Math.min(1000 * Math.pow(2, attempt) + Math.random() * 100, 10000);
					await new Promise(resolve => setTimeout(resolve, delay));
				}
			}
		}

		throw lastError!;
	}

	private async performRequest<T>(options: RequestOptions): Promise<T> {
		const url = new URL(this.host + options.path);
		const requestModule = url.protocol === 'https:' ? https : http;

		return new Promise<T>((resolve, reject) => {
			const timeoutId = setTimeout(() => {
				reject(new Error(`Request timeout after ${options.timeout}ms`));
			}, options.timeout);

			const req = requestModule.request({
				hostname: url.hostname,
				port: url.port,
				path: url.pathname + url.search,
				method: options.method,
				headers: options.headers,
				timeout: options.timeout
			}, async (res) => {
				clearTimeout(timeoutId);

				let responseData = Buffer.alloc(0);

				res.on('data', (chunk: Buffer) => {
					responseData = Buffer.concat([responseData, chunk]);
				});

				res.on('end', async () => {
					try {
						let finalData = responseData;
						const contentEncoding = res.headers['content-encoding'];

						if (contentEncoding === 'gzip') {
							finalData = await gunzipAsync(responseData);
						} else if (contentEncoding === 'br') {
							finalData = await brotliDecompressAsync(responseData);
						}

						if (res.statusCode! >= 400) {
							let errorMsg = `HTTP ${res.statusCode}`;
							try {
								const errorData = JSON.parse(finalData.toString());
								if (errorData.error) {
									errorMsg += ` - ${errorData.error}`;
								}
							} catch {
								errorMsg += ` - ${finalData.toString().substring(0, 100)}`;
							}
							reject(new Error(errorMsg));
							return;
						}

						const result = JSON.parse(finalData.toString());
						resolve(result);
					} catch (parseError) {
						reject(parseError);
					}
				});
			});

			req.on('error', (error) => {
				clearTimeout(timeoutId);
				reject(error);
			});

			req.on('timeout', () => {
				clearTimeout(timeoutId);
				req.destroy();
				reject(new Error('Request timeout'));
			});

			if (options.body) {
				req.write(options.body);
			}

			req.end();
		});
	}

	private cleanup(): void {
		if (this.isDestroyed) return;

		this.emit('metrics', {
			requestCount: this.requestCount,
			avgRequestTime: this.requestTimes.length > 0
				? this.requestTimes.reduce((a, b) => a + b, 0) / this.requestTimes.length
				: 0,
			cacheStats: this.queryCache.getStats(),
			uptime: Date.now() - this.startTime
		});
	}

	getStats(): Record<string, any> {
		const now = Date.now();
		return {
			requests: {
				total: this.requestCount,
				avgTime: this.requestTimes.length > 0
					? this.requestTimes.reduce((a, b) => a + b, 0) / this.requestTimes.length
					: 0,
				recentTimes: this.requestTimes.slice(-10)
			},
			cache: this.queryCache.getStats(),
			bufferPool: this.bufferPool.getStats(),
			uptime: now - this.startTime,
			config: this.config
		};
	}

	async close(): Promise<void> {
		if (this.isDestroyed) return;

		this.isDestroyed = true;
		globalClients.delete(this);

		for (const controller of this.pendingRequests) {
			try {
				controller.abort();
			} catch (error) { }
		}
		this.pendingRequests.clear();

		if (this.cleanupInterval) {
			clearInterval(this.cleanupInterval);
			this.cleanupInterval = undefined;
		}

		try {
			this.queryCache.destroy();
			this.bufferPool.clear();
			this.collectionCache.clear();
		} catch (error) { }

		this.removeAllListeners();
	}
}

export function createClient(host: string, config: ClientConfig = {}): Client {
	return new Client(host, config);
}
