import * as http2 from 'http2';
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
	PooledConnection,
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

function cleanJsonResponse(buffer: Buffer): Buffer {
	let data = buffer.toString();
	data = data.replace(/\x1b\[[0-9;]*m/g, '');
	const jsonStart = data.indexOf('{');
	const jsonEnd = data.lastIndexOf('}');

	if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
		data = data.substring(jsonStart, jsonEnd + 1);
	}

	data = data.replace(/[\x00-\x1f\x7f-\x9f]/g, '');

	return Buffer.from(data, 'utf8');
}

export class Client extends EventEmitter {
	private readonly host: string;
	private readonly config: Required<ClientConfig>;
	private readonly logger: Logger;

	// Connection management
	private connectionPool = new Map<string, PooledConnection>();
	private readonly maxConnections: number;
	private readonly maxStreamsPerConnection: number = 100;

	// Performance optimizations
	private readonly queryCache: QueryCache;
	private readonly bufferPool: BufferPool;
	private readonly collectionCache = new Map<string, CollectionInfo>();

	// Metrics and monitoring
	private requestCount = 0;
	private requestTimes: number[] = [];
	private readonly startTime = Date.now();

	// Cleanup handling
	private cleanupInterval?: NodeJS.Timeout;
	private isDestroyed = false;

	constructor(host: string, config: ClientConfig = {}) {
		super();

		this.host = host.replace(/\/$/, ''); // Remove trailing slash

		const cpuCount = cpus().length;

		// Set intelligent defaults based on system capabilities
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
		this.maxConnections = this.config.connectionPoolSize;

		// Initialize performance components
		this.queryCache = new QueryCache(this.config.cacheSize, this.config.cacheTtl * 1000);
		this.bufferPool = new BufferPool(50);

		// Setup periodic cleanup
		this.cleanupInterval = setInterval(() => this.cleanup(), 30000); // Every 30 seconds

		// Handle process cleanup
		process.on('beforeExit', () => this.close());
		process.on('SIGINT', () => this.close());
		process.on('SIGTERM', () => this.close());
	}

	/**
	 * Create a new vector collection
	 */
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

		// Cache collection info on successful creation
		if (response.success) {
			this.collectionCache.set(params.name, {
				name: params.name,
				dimensions: params.dimensions,
				enableHnsw: params.enableHnsw ?? true,
				shards: params.shards ?? 16,
				m: params.m ?? 16,
				efConstruction: params.efConstruction ?? 200,
				createdAt: new Date().toISOString()
			});
		}

		return response;
	}

	/**
	 * List all collections
	 */
	async listCollections(): Promise<CollectionInfo[]> {
		const cacheKey = 'list_collections';

		// Try cache first for metadata operations
		const cached = this.queryCache.get(cacheKey);
		if (cached && cached.matches) {
			return cached.matches as any; // Type assertion for collections list
		}

		const response = await this.request<{ collections: CollectionInfo[] }>({
			method: 'GET',
			path: '/collections'
		});

		const collections = response.collections || [];

		// Update collection cache
		for (const collection of collections) {
			this.collectionCache.set(collection.name, collection);
		}

		return collections;
	}

	/**
	 * Get collection information
	 */
	async describeCollection(name: string): Promise<CollectionInfo> {
		// Check local cache first
		const cached = this.collectionCache.get(name);
		if (cached) {
			return cached;
		}

		const response = await this.request<CollectionInfo>({
			method: 'GET',
			path: `/collections/${encodeURIComponent(name)}`
		});

		// Cache the result
		this.collectionCache.set(name, response);

		return response;
	}

	/**
	 * Delete a collection
	 */
	async deleteCollection(name: string): Promise<ApiResponse> {
		const response = await this.request<ApiResponse>({
			method: 'DELETE',
			path: `/collections/${encodeURIComponent(name)}`
		});

		// Clear caches
		this.collectionCache.delete(name);
		this.queryCache.clear(); // Simple approach: clear all query cache

		return response;
	}

	/**
	 * Get collection dimensions (cached)
	 */
	async getCollectionDimensions(collectionName: string): Promise<number | undefined> {
		const cached = this.collectionCache.get(collectionName);
		if (cached?.dimensions) {
			return cached.dimensions;
		}

		try {
			const info = await this.describeCollection(collectionName);
			return info.dimensions;
		} catch (error) {
			this.logger.warn(`Failed to get dimensions for collection ${collectionName}:`, error);
			return undefined;
		}
	}

	/**
	 * Validate vector dimensions against collection
	 */
	async validateVectorDimensions(collectionName: string, vector: number[]): Promise<boolean> {
		const expectedDims = await this.getCollectionDimensions(collectionName);
		if (expectedDims === undefined) {
			return true; // Can't validate without expected dimensions
		}
		return vector.length === expectedDims;
	}

	/**
	 * Batch insert with performance optimizations
	 */
	async batchInsert(
		records: VectorRecord[],
		collectionName: string = 'default',
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

		const {
			validateDimensions = true,
			parallelism = Math.min(cpus().length, 8)
		} = options;

		// Validate dimensions if requested
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

		// Use vector operations for optimized batching
		const vectorOps = this.vectorOperations(collectionName);
		return vectorOps.upsert(records, options);
	}

	/**
	 * Get vector operations interface
	 */
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

	/**
	 * Alias for vectorOperations for Python API compatibility
	 */
	vector_operations(collectionName: string = 'default'): VectorOperations {
		return this.vectorOperations(collectionName);
	}

	/**
	 * Force commit to disk
	 */
	async commit(): Promise<ApiResponse> {
		return this.request<ApiResponse>({
			method: 'POST',
			path: '/admin/commit'
		});
	}

	/**
	 * Check server health
	 */
	async health(): Promise<HealthResponse> {
		return this.request<HealthResponse>({
			method: 'GET',
			path: '/health',
			timeout: 10
		});
	}

	/**
	 * Get server information
	 */
	async info(): Promise<InfoResponse> {
		const cacheKey = 'server_info';
		const cached = this.queryCache.get(cacheKey);
		if (cached?.matches) {
			return cached.matches as any;
		}

		const response = await this.request<InfoResponse>({
			method: 'GET',
			path: '/info',
			timeout: 10
		});

		return response;
	}

	/**
	 * Clear all caches
	 */
	async clearCache(): Promise<{ success: boolean; message: string }> {
		this.queryCache.clear();
		this.collectionCache.clear();
		return { success: true, message: 'All caches cleared' };
	}

	/**
	 * High-performance HTTP request with connection pooling and retries
	 */
	async request<T = any>(options: RequestOptions): Promise<T> {
		const { method, path, headers = {}, body, timeout = this.config.timeout } = options;

		let lastError: Error;

		for (let attempt = 0; attempt < this.config.retryAttempts; attempt++) {
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
					timeout: timeout * 1000 // Convert to milliseconds
				});

				// Track performance metrics
				const duration = performance.now() - startTime;
				this.requestCount++;
				this.requestTimes.push(duration);

				// Keep only last 1000 request times for memory efficiency
				if (this.requestTimes.length > 1000) {
					this.requestTimes = this.requestTimes.slice(-500);
				}

				return result;

			} catch (error) {
				lastError = error as Error;
				this.logger.debug(`Request attempt ${attempt + 1} failed:`, error);

				if (attempt < this.config.retryAttempts - 1) {
					// Exponential backoff with jitter
					const delay = Math.min(1000 * Math.pow(2, attempt) + Math.random() * 100, 10000);
					await new Promise(resolve => setTimeout(resolve, delay));
				}
			}
		}

		throw lastError!;
	}

	/**
	 * Perform actual HTTP request with connection pooling
	 */
	private async performRequest<T>(options: RequestOptions): Promise<T> {
		const url = new URL(this.host);
		const isHttps = url.protocol === 'https:';

		return this.performHttp1Request<T>(options, isHttps);
	}

	/**
	 * HTTP/1.1 fallback request
	 */
	private async performHttp1Request<T>(options: RequestOptions, isHttps: boolean): Promise<T> {
		const url = new URL(this.host + options.path);
		const requestModule = isHttps ? https : http;

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
						// Handle compression
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

			// Send request body
			if (options.body) {
				req.write(options.body);
			}

			req.end();
		});
	}

	/**
	 * Get or create HTTP/2 connection with connection pooling
	 */
	private async getOrCreateHttp2Connection(): Promise<PooledConnection> {
		const url = new URL(this.host);
		const connectionKey = `${url.protocol}//${url.host}`;

		// Find existing connection with available capacity
		const existing = this.connectionPool.get(connectionKey);
		if (existing &&
			existing.activeStreams < this.maxStreamsPerConnection &&
			!existing.client.destroyed &&
			!existing.client.closed) {
			return existing;
		}

		// Create new connection
		const client = http2.connect(this.host, {
			maxSessionMemory: 100 // 100 MB
		});

		const connection: PooledConnection = {
			client,
			activeStreams: 0,
			lastUsed: Date.now(),
			host: connectionKey
		};

		// Handle connection events
		client.on('error', (error) => {
			this.logger.error('HTTP/2 connection error:', error);
			this.connectionPool.delete(connectionKey);
		});

		client.on('close', () => {
			this.logger.debug('HTTP/2 connection closed');
			this.connectionPool.delete(connectionKey);
		});

		this.connectionPool.set(connectionKey, connection);
		this.logger.debug(`Created new HTTP/2 connection to ${connectionKey}`);

		return connection;
	}

	/**
	 * Periodic cleanup of stale connections and caches
	 */
	private cleanup(): void {
		if (this.isDestroyed) return;

		const now = Date.now();
		const staleTimeout = 60000; // 1 minute

		// Clean up stale connections
		for (const [key, connection] of this.connectionPool) {
			if (now - connection.lastUsed > staleTimeout && connection.activeStreams === 0) {
				this.logger.debug(`Closing stale connection to ${key}`);
				connection.client.close();
				this.connectionPool.delete(key);
			}
		}

		// Emit metrics for monitoring
		this.emit('metrics', {
			connections: this.connectionPool.size,
			requestCount: this.requestCount,
			avgRequestTime: this.requestTimes.length > 0
				? this.requestTimes.reduce((a, b) => a + b, 0) / this.requestTimes.length
				: 0,
			cacheStats: this.queryCache.getStats(),
			uptime: now - this.startTime
		});
	}

	/**
	 * Get performance statistics
	 */
	getStats(): Record<string, any> {
		const now = Date.now();
		return {
			connections: {
				active: this.connectionPool.size,
				maxConnections: this.maxConnections
			},
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

	/**
	 * Graceful shutdown and resource cleanup
	 */
	async close(): Promise<void> {
		if (this.isDestroyed) return;

		this.isDestroyed = true;
		this.logger.info('Closing Antarys client...');

		// Clear cleanup interval
		if (this.cleanupInterval) {
			clearInterval(this.cleanupInterval);
		}

		// Close all HTTP/2 connections
		const closePromises: Promise<void>[] = [];
		for (const [key, connection] of this.connectionPool) {
			closePromises.push(
				new Promise<void>((resolve) => {
					connection.client.close(() => {
						this.logger.debug(`Closed connection to ${key}`);
						resolve();
					});
				})
			);
		}

		// Wait for all connections to close (with timeout)
		try {
			await Promise.race([
				Promise.all(closePromises),
				new Promise(resolve => setTimeout(resolve, 5000)) // 5 second timeout
			]);
		} catch (error) {
			this.logger.warn('Some connections may not have closed cleanly:', error);
		}

		// Clean up caches and pools
		this.queryCache.destroy();
		this.bufferPool.clear();
		this.collectionCache.clear();
		this.connectionPool.clear();

		this.logger.info('Antarys client closed successfully');
		this.removeAllListeners();
	}
}

/**
 * Create a new Antarys client with optimized defaults
 */
export function createClient(host: string, config: ClientConfig = {}): Client {
	return new Client(host, config);
}
