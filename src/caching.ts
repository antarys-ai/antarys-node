import { createHash } from 'crypto';
import { SearchResults, CacheStats } from '../shared/types';

interface CacheEntry<T> {
	value: T;
	timestamp: number;
	hits: number;
}

interface LRUNode<T> {
	key: string;
	value: CacheEntry<T>;
	prev?: LRUNode<T>;
	next?: LRUNode<T>;
}

export class HighPerformanceCache<T> {
	private cache = new Map<string, LRUNode<T>>();
	private head?: LRUNode<T>;
	private tail?: LRUNode<T>;
	private readonly maxSize: number;
	private readonly ttl: number;
	private hits = 0;
	private misses = 0;
	private cleanupInterval?: NodeJS.Timeout;

	// 5 minutes default TTL
	constructor(maxSize: number = 1000, ttl: number = 300000) {
		this.maxSize = maxSize;
		this.ttl = ttl;
		this.cleanupInterval = setInterval(() => this.cleanup(), 60000);
	}

	get(key: string): T | undefined {
		const node = this.cache.get(key);

		if (!node) {
			this.misses++;
			return undefined;
		}

		if (Date.now() - node.value.timestamp > this.ttl) {
			this.delete(key);
			this.misses++;
			return undefined;
		}

		this.moveToFront(node);
		node.value.hits++;
		this.hits++;

		return node.value.value;
	}

	set(key: string, value: T): void {
		const existing = this.cache.get(key);

		if (existing) {
			existing.value.value = value;
			existing.value.timestamp = Date.now();
			this.moveToFront(existing);
			return;
		}

		const entry: CacheEntry<T> = {
			value,
			timestamp: Date.now(),
			hits: 0
		};

		const node: LRUNode<T> = { key, value: entry };

		this.addToFront(node);
		this.cache.set(key, node);

		if (this.cache.size > this.maxSize) {
			this.evictLRU();
		}
	}

	delete(key: string): boolean {
		const node = this.cache.get(key);
		if (!node) return false;

		this.removeNode(node);
		this.cache.delete(key);
		return true;
	}

	clear(): void {
		this.cache.clear();
		this.head = undefined;
		this.tail = undefined;
		this.hits = 0;
		this.misses = 0;
	}

	getStats(): CacheStats {
		const totalRequests = this.hits + this.misses;
		return {
			cacheEnabled: true,
			cacheHits: this.hits,
			cacheMisses: this.misses,
			hitRate: totalRequests > 0 ? this.hits / totalRequests : 0,
			cacheSize: this.cache.size
		};
	}

	private moveToFront(node: LRUNode<T>): void {
		this.removeNode(node);
		this.addToFront(node);
	}

	private addToFront(node: LRUNode<T>): void {
		node.prev = undefined;
		node.next = this.head;

		if (this.head) {
			this.head.prev = node;
		}

		this.head = node;

		if (!this.tail) {
			this.tail = node;
		}
	}

	private removeNode(node: LRUNode<T>): void {
		if (node.prev) {
			node.prev.next = node.next;
		} else {
			this.head = node.next;
		}

		if (node.next) {
			node.next.prev = node.prev;
		} else {
			this.tail = node.prev;
		}
	}

	private evictLRU(): void {
		if (this.tail) {
			this.cache.delete(this.tail.key);
			this.removeNode(this.tail);
		}
	}

	private cleanup(): void {
		const now = Date.now();
		const expiredKeys: string[] = [];

		for (const [key, node] of this.cache.entries()) {
			if (now - node.value.timestamp > this.ttl) {
				expiredKeys.push(key);
			}
		}

		for (const key of expiredKeys) {
			this.delete(key);
		}
	}

	destroy(): void {
		if (this.cleanupInterval) {
			clearInterval(this.cleanupInterval);
		}
		this.clear();
	}
}

export class QueryCache {
	private cache: HighPerformanceCache<SearchResults>;

	constructor(maxSize: number = 1000, ttl: number = 300000) {
		this.cache = new HighPerformanceCache<SearchResults>(maxSize, ttl);
	}

	computeCacheKey(
		collection: string,
		vector: number[],
		topK: number,
		params: Record<string, any> = {}
	): string {
		const queryData = {
			collection,
			vector: this.hashVector(vector),
			topK,
			...params
		};

		const hash = createHash('sha256')
			.update(JSON.stringify(queryData))
			.digest('hex');

		return `${collection}:${hash.substring(0, 16)}`;
	}

	private hashVector(vector: number[]): string {
		if (vector.length <= 100) {
			return createHash('md5').update(vector.join(',')).digest('hex');
		}

		// For large vectors, sample key points
		const samples = [];
		const step = Math.floor(vector.length / 20);

		for (let i = 0; i < vector.length; i += step) {
			samples.push(vector[i]);
		}

		return createHash('md5').update(samples.join(',')).digest('hex');
	}

	get(key: string): SearchResults | undefined {
		return this.cache.get(key);
	}

	set(key: string, value: SearchResults): void {
		this.cache.set(key, value);
	}

	clear(): void {
		this.cache.clear();
	}

	getStats(): CacheStats {
		return this.cache.getStats();
	}

	destroy(): void {
		this.cache.destroy();
	}
}

export class BufferPool {
	private pools = new Map<number, Buffer[]>();
	private readonly maxPoolSize: number;
	private readonly sizes: number[];

	constructor(maxPoolSize: number = 50) {
		this.maxPoolSize = maxPoolSize;
		this.sizes = [1024, 4096, 16384, 65536, 262144];

		// pre-allocate initial buffers
		for (const size of this.sizes) {
			this.pools.set(size, []);
			for (let i = 0; i < 10; i++) {
				this.pools.get(size)!.push(Buffer.allocUnsafe(size));
			}
		}
	}

	acquire(size: number): Buffer {
		const poolSize = this.sizes.find(s => s >= size);

		if (!poolSize) {
			return Buffer.allocUnsafe(size);
		}

		const pool = this.pools.get(poolSize);
		if (pool && pool.length > 0) {
			const buffer = pool.pop()!;
			return buffer.subarray(0, size);
		}

		return Buffer.allocUnsafe(size);
	}

	release(buffer: Buffer): void {
		const originalSize = this.getOriginalSize(buffer);
		if (!originalSize) return;

		const pool = this.pools.get(originalSize);
		if (pool && pool.length < this.maxPoolSize) {
			buffer.fill(0);
			pool.push(buffer);
		}
	}

	private getOriginalSize(buffer: Buffer): number | undefined {
		return this.sizes.find(size => buffer.length <= size);
	}

	clear(): void {
		this.pools.clear();
	}

	getStats(): Record<string, number> {
		const stats: Record<string, number> = {};
		for (const [size, pool] of this.pools) {
			stats[`pool_${size}`] = pool.length;
		}
		return stats;
	}
}
