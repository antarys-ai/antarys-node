import { createHash } from 'crypto';
export class HighPerformanceCache {
    // 5 minutes default TTL
    constructor(maxSize = 1000, ttl = 300000) {
        this.cache = new Map();
        this.hits = 0;
        this.misses = 0;
        this.maxSize = maxSize;
        this.ttl = ttl;
        this.cleanupInterval = setInterval(() => this.cleanup(), 60000);
    }
    get(key) {
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
    set(key, value) {
        const existing = this.cache.get(key);
        if (existing) {
            existing.value.value = value;
            existing.value.timestamp = Date.now();
            this.moveToFront(existing);
            return;
        }
        const entry = {
            value,
            timestamp: Date.now(),
            hits: 0
        };
        const node = { key, value: entry };
        this.addToFront(node);
        this.cache.set(key, node);
        if (this.cache.size > this.maxSize) {
            this.evictLRU();
        }
    }
    delete(key) {
        const node = this.cache.get(key);
        if (!node)
            return false;
        this.removeNode(node);
        this.cache.delete(key);
        return true;
    }
    clear() {
        this.cache.clear();
        this.head = undefined;
        this.tail = undefined;
        this.hits = 0;
        this.misses = 0;
    }
    getStats() {
        const totalRequests = this.hits + this.misses;
        return {
            cacheEnabled: true,
            cacheHits: this.hits,
            cacheMisses: this.misses,
            hitRate: totalRequests > 0 ? this.hits / totalRequests : 0,
            cacheSize: this.cache.size
        };
    }
    moveToFront(node) {
        this.removeNode(node);
        this.addToFront(node);
    }
    addToFront(node) {
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
    removeNode(node) {
        if (node.prev) {
            node.prev.next = node.next;
        }
        else {
            this.head = node.next;
        }
        if (node.next) {
            node.next.prev = node.prev;
        }
        else {
            this.tail = node.prev;
        }
    }
    evictLRU() {
        if (this.tail) {
            this.cache.delete(this.tail.key);
            this.removeNode(this.tail);
        }
    }
    cleanup() {
        const now = Date.now();
        const expiredKeys = [];
        for (const [key, node] of this.cache.entries()) {
            if (now - node.value.timestamp > this.ttl) {
                expiredKeys.push(key);
            }
        }
        for (const key of expiredKeys) {
            this.delete(key);
        }
    }
    destroy() {
        if (this.cleanupInterval) {
            clearInterval(this.cleanupInterval);
        }
        this.clear();
    }
}
export class QueryCache {
    constructor(maxSize = 1000, ttl = 300000) {
        this.cache = new HighPerformanceCache(maxSize, ttl);
    }
    computeCacheKey(collection, vector, topK, params = {}) {
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
    hashVector(vector) {
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
    get(key) {
        return this.cache.get(key);
    }
    set(key, value) {
        this.cache.set(key, value);
    }
    clear() {
        this.cache.clear();
    }
    getStats() {
        return this.cache.getStats();
    }
    destroy() {
        this.cache.destroy();
    }
}
export class BufferPool {
    constructor(maxPoolSize = 50) {
        this.pools = new Map();
        this.maxPoolSize = maxPoolSize;
        this.sizes = [1024, 4096, 16384, 65536, 262144];
        // pre-allocate initial buffers
        for (const size of this.sizes) {
            this.pools.set(size, []);
            for (let i = 0; i < 10; i++) {
                this.pools.get(size).push(Buffer.allocUnsafe(size));
            }
        }
    }
    acquire(size) {
        const poolSize = this.sizes.find(s => s >= size);
        if (!poolSize) {
            return Buffer.allocUnsafe(size);
        }
        const pool = this.pools.get(poolSize);
        if (pool && pool.length > 0) {
            const buffer = pool.pop();
            return buffer.subarray(0, size);
        }
        return Buffer.allocUnsafe(size);
    }
    release(buffer) {
        const originalSize = this.getOriginalSize(buffer);
        if (!originalSize)
            return;
        const pool = this.pools.get(originalSize);
        if (pool && pool.length < this.maxPoolSize) {
            buffer.fill(0);
            pool.push(buffer);
        }
    }
    getOriginalSize(buffer) {
        return this.sizes.find(size => buffer.length <= size);
    }
    clear() {
        this.pools.clear();
    }
    getStats() {
        const stats = {};
        for (const [size, pool] of this.pools) {
            stats[`pool_${size}`] = pool.length;
        }
        return stats;
    }
}
//# sourceMappingURL=caching.js.map