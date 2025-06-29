export interface ClientConfig {
	timeout?: number;
	connectionPoolSize?: number;
	retryAttempts?: number;
	compression?: boolean;
	debug?: boolean;
	cacheSize?: number;
	threadPoolSize?: number;
	cacheTtl?: number;
}

export interface VectorRecord {
	id: string;
	values?: number[];
	vector?: number[];
	metadata?: Record<string, any>;
}

export interface SearchResult {
	id: string;
	score: number;
	values?: number[];
	metadata?: Record<string, any>;
}

export interface SearchResults {
	matches: SearchResult[];
}

export interface BatchSearchResults {
	results: SearchResults[];
}

export interface SearchParams {
	vector?: number[];
	topK?: number;
	includeValues?: boolean;
	includeMetadata?: boolean;
	filter?: Record<string, any>;
	useAnn?: boolean;
	efSearch?: number;
	threshold?: number;
	skipCache?: boolean;
	validateDimensions?: boolean;
}

export interface UpsertParams {
	vectors: VectorRecord[];
	batchSize?: number;
	showProgress?: boolean;
	parallelWorkers?: number;
	validateDimensions?: boolean;
}

export interface CreateCollectionParams {
	name: string;
	dimensions: number;
	enableHnsw?: boolean;
	shards?: number;
	m?: number;
	efConstruction?: number;
}

export interface CollectionInfo {
	name: string;
	dimensions?: number;
	vectorCount?: number;
	enableHnsw?: boolean;
	shards?: number;
	m?: number;
	efConstruction?: number;
	createdAt?: string;
}

export interface ApiResponse<T = any> {
	success?: boolean;
	error?: string;
	message?: string;
	data?: T;
}

export interface UpsertResponse {
	upserted_count?: number;
	count?: number;
	success?: boolean;
}

export interface QueryResponse {
	results?: SearchResult[];
	matches?: SearchResult[];
}

export interface BatchQueryResponse {
	results?: QueryResponse[];
}

export interface DeleteResponse {
	deleted?: string[];
	failed?: string[];
}

export interface CacheStats {
	cacheEnabled: boolean;
	cacheHits?: number;
	cacheMisses?: number;
	hitRate?: number;
	cacheSize?: number;
}

export interface HealthResponse {
	status: string;
	uptime?: number;
	version?: string;
}

export interface InfoResponse {
	version: string;
	buildDate?: string;
	features?: string[];
}

// Internal types for HTTP operations
export interface RequestOptions {
	method: 'GET' | 'POST' | 'PUT' | 'DELETE';
	path: string;
	headers?: Record<string, string>;
	body?: Buffer | string;
	timeout?: number;
}

export interface PooledConnection {
	client: any;
	activeStreams: number;
	lastUsed: number;
	host: string;
}

export interface BatchOperation<T> {
	items: T[];
	resolve: (result: any) => void;
	reject: (error: Error) => void;
	timestamp: number;
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export interface Logger {
	debug(message: string, ...args: any[]): void;
	info(message: string, ...args: any[]): void;
	warn(message: string, ...args: any[]): void;
	error(message: string, ...args: any[]): void;
}
