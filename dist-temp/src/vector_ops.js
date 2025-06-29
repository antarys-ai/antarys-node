import { cpus } from 'os';
import { existsSync } from 'fs';
import { join, resolve } from 'path';
import { Worker } from 'worker_threads';
export class VectorOperations {
    constructor(host, collectionName, context) {
        // Performance tracking
        this.cacheHits = 0;
        this.cacheMisses = 0;
        // Batch operation queues for optimization
        this.upsertQueue = [];
        this.batchDelay = 50; // 50ms batch delay for auto-batching
        this.host = host.replace(/\/$/, '');
        this.collectionName = collectionName;
        this.context = context;
        // Initialize worker pool with CPU-based sizing
        this.maxWorkers = Math.min(cpus().length * 2, 16); // Max 16 workers
        this.workerPool = new WorkerPool(this.maxWorkers, context.logger);
        this.context.logger.debug(`Created VectorOperations for collection '${collectionName}' with ${this.maxWorkers} workers`);
    }
    /**
     * Upsert vectors with intelligent batching and worker thread parallelization
     */
    async upsert(vectors, options = {}) {
        if (!vectors.length) {
            return { upserted_count: 0 };
        }
        const { batchSize = 5000, showProgress = false, parallelWorkers = Math.min(this.maxWorkers, 8), validateDimensions = true } = options;
        // Validate dimensions if requested
        if (validateDimensions) {
            await this.validateBatchDimensions(vectors.slice(0, 10)); // Sample validation
        }
        this.context.logger.debug(`Upserting ${vectors.length} vectors with ${parallelWorkers} workers`);
        // Use worker threads for CPU-intensive preprocessing if available and beneficial
        const processedVectors = await this.preprocessVectorsParallel(vectors, parallelWorkers);
        // Create batches
        const batches = [];
        for (let i = 0; i < processedVectors.length; i += batchSize) {
            batches.push(processedVectors.slice(i, i + batchSize));
        }
        // Process batches in parallel with worker thread pool
        let totalUpserted = 0;
        const semaphore = new Semaphore(parallelWorkers);
        const batchPromises = batches.map(async (batch, index) => {
            await semaphore.acquire();
            try {
                // Use worker thread for batch processing if available and batch is large enough
                let processedBatch = batch;
                if (batch.length > 500 && this.workerPool.isHealthy()) {
                    try {
                        processedBatch = await this.workerPool.execute('processBatch', {
                            batch,
                            collectionName: this.collectionName
                        });
                    }
                    catch (workerError) {
                        if (workerError.message) {
                            this.context.logger.debug('Worker processing failed, using main thread:', workerError.message);
                        }
                        processedBatch = batch;
                    }
                }
                const result = await this.upsertBatch(processedBatch);
                const count = result.count || result.upserted_count || batch.length;
                totalUpserted += count;
                if (showProgress) {
                    this.context.logger.info(`Batch ${index + 1}/${batches.length} completed: ${count} vectors`);
                }
                return count;
            }
            catch (error) {
                this.context.logger.error(`Batch ${index + 1} failed:`, error);
                throw error;
            }
            finally {
                semaphore.release();
            }
        });
        // Wait for all batches to complete
        await Promise.all(batchPromises);
        this.context.logger.debug(`Upsert complete: ${totalUpserted}/${vectors.length} vectors inserted`);
        return { upserted_count: totalUpserted };
    }
    /**
     * Optimized batch upsert with retry logic
     */
    async upsertBatch(batch) {
        // Format vectors for Go server compatibility
        const formattedVectors = batch.map(vec => ({
            id: String(vec.id),
            vector: vec.vector || vec.values || [],
            ...(vec.metadata && { metadata: vec.metadata })
        }));
        const payload = {
            collection: this.collectionName,
            vectors: formattedVectors
        };
        // Use buffer pool for large payloads
        const payloadStr = JSON.stringify(payload);
        const payloadBuffer = this.context.bufferPool.acquire(Buffer.byteLength(payloadStr));
        payloadBuffer.write(payloadStr);
        try {
            const response = await this.context.request({
                method: 'POST',
                path: '/vectors/upsert',
                body: payloadBuffer.toString(),
                timeout: 120
            });
            return response;
        }
        finally {
            this.context.bufferPool.release(payloadBuffer);
        }
    }
    /**
     * High-performance vector similarity search with caching
     */
    async query(params = {}) {
        const { vector, topK = 10, includeValues = false, includeMetadata = true, filter, useAnn = true, efSearch = 100, threshold = 0.0, skipCache = false, validateDimensions = true } = params;
        if (!vector || !vector.length) {
            throw new Error('Query vector is required');
        }
        // Validate dimensions if requested
        if (validateDimensions) {
            const isValid = await this.context.validateVectorDimensions(this.collectionName, vector);
            if (!isValid) {
                const expectedDims = await this.context.getCollectionDimensions(this.collectionName);
                throw new Error(`Query vector dimension mismatch: got ${vector.length}, expected ${expectedDims} for collection '${this.collectionName}'`);
            }
        }
        // Check cache if enabled and not skipped
        let cacheKey;
        if (!skipCache) {
            cacheKey = this.context.queryCache.computeCacheKey(this.collectionName, vector, topK, {
                includeValues,
                includeMetadata,
                useAnn,
                threshold,
                filter: filter ? JSON.stringify(filter) : undefined
            });
            const cached = this.context.queryCache.get(cacheKey);
            if (cached) {
                this.cacheHits++;
                this.context.logger.debug(`Cache hit for query ${cacheKey.substring(0, 16)}...`);
                return cached;
            }
            this.cacheMisses++;
        }
        // Prepare query payload
        const queryPayload = {
            collection: this.collectionName,
            vector: this.optimizeVector(vector),
            top_k: topK,
            include_vectors: includeValues,
            include_metadata: includeMetadata,
            use_ann: useAnn,
            ef_search: efSearch,
            threshold: Number(threshold),
            ...(filter && { filter })
        };
        try {
            const response = await this.context.request({
                method: 'POST',
                path: '/vectors/query',
                body: JSON.stringify(queryPayload),
                timeout: 60
            });
            // Format results
            const matches = (response.results || []).map(match => ({
                id: match.id,
                score: match.score,
                ...(includeValues && match.values && { values: match.values }),
                ...(includeMetadata && match.metadata && { metadata: match.metadata })
            }));
            const result = { matches };
            // Cache the result
            if (cacheKey && !skipCache) {
                this.context.queryCache.set(cacheKey, result);
            }
            return result;
        }
        catch (error) {
            this.context.logger.error('Query failed:', error);
            throw error;
        }
    }
    /**
     * Batch query for multiple vectors with worker thread parallelization
     */
    async batchQuery(vectors, options = {}) {
        const { topK = 10, includeValues = false, includeMetadata = true, filter, useAnn = true, efSearch = 100, threshold = 0.0, validateDimensions = true } = options;
        if (!vectors.length) {
            throw new Error('Query vectors are required');
        }
        // Validate dimensions if requested
        if (validateDimensions) {
            const expectedDims = await this.context.getCollectionDimensions(this.collectionName);
            if (expectedDims !== undefined) {
                for (let i = 0; i < Math.min(vectors.length, 5); i++) {
                    if (vectors[i].length !== expectedDims) {
                        throw new Error(`Query vector ${i} dimension mismatch: got ${vectors[i].length}, expected ${expectedDims} for collection '${this.collectionName}'`);
                    }
                }
            }
        }
        // For large batches, split them up
        const maxBatchSize = 50;
        if (vectors.length > maxBatchSize) {
            const results = [];
            for (let i = 0; i < vectors.length; i += maxBatchSize) {
                const batch = vectors.slice(i, i + maxBatchSize);
                const batchResult = await this.batchQuery(batch, {
                    ...options,
                    validateDimensions: false // Already validated above
                });
                results.push(...batchResult.results);
            }
            return { results };
        }
        // Use worker threads for parallel vector processing if available and beneficial
        let processedVectors;
        if (vectors.length > 10 && this.workerPool.isHealthy()) {
            try {
                processedVectors = await this.workerPool.execute('processQueryVectors', {
                    vectors,
                    collectionName: this.collectionName
                });
            }
            catch (workerError) {
                if (workerError.message) {
                    this.context.logger.debug('Worker query processing failed, using main thread:', workerError.message);
                }
                processedVectors = await Promise.all(vectors.map(vec => Promise.resolve(this.optimizeVector(vec))));
            }
        }
        else {
            // Process vectors in parallel for CPU-bound operations (small batches or no workers)
            processedVectors = await Promise.all(vectors.map(vec => Promise.resolve(this.optimizeVector(vec))));
        }
        // Create batch query payload
        const queries = processedVectors.map((vec, index) => ({
            vector: vec,
            top_k: topK,
            include_values: includeValues,
            include_metadata: includeMetadata,
            use_ann: useAnn,
            ef_search: efSearch,
            threshold: Number(threshold),
            query_id: `query_${index}`,
            ...(filter && { filter })
        }));
        const payload = {
            collection: this.collectionName,
            queries,
            include_vectors: includeValues,
            include_metadata: includeMetadata
        };
        try {
            const response = await this.context.request({
                method: 'POST',
                path: '/vectors/batch_query',
                body: JSON.stringify(payload),
                timeout: 120
            });
            // Format results
            const formattedResults = (response.results || []).map(queryResult => {
                const matches = (queryResult.results || []).map(match => ({
                    id: match.id,
                    score: match.score,
                    ...(includeValues && match.values && { values: match.values }),
                    ...(includeMetadata && match.metadata && { metadata: match.metadata })
                }));
                return { matches };
            });
            return { results: formattedResults };
        }
        catch (error) {
            this.context.logger.error('Batch query failed:', error);
            throw error;
        }
    }
    /**
     * Delete vectors by ID
     */
    async delete(ids) {
        if (!ids.length) {
            return { deleted: [], failed: [] };
        }
        const payload = {
            collection: this.collectionName,
            ids: ids.map(id => String(id))
        };
        try {
            const response = await this.context.request({
                method: 'POST',
                path: '/vectors/delete',
                body: JSON.stringify(payload),
                timeout: 30
            });
            // Invalidate cache for bulk deletions
            if (ids.length > 10) {
                this.context.queryCache.clear();
            }
            return response;
        }
        catch (error) {
            this.context.logger.error('Delete operation failed:', error);
            return { deleted: [], failed: ids };
        }
    }
    /**
     * Get a specific vector by ID
     */
    async getVector(vectorId) {
        try {
            const response = await this.context.request({
                method: 'GET',
                path: `/vectors/${encodeURIComponent(vectorId)}?collection=${encodeURIComponent(this.collectionName)}`,
                timeout: 30
            });
            return response;
        }
        catch (error) {
            if (error && error.message) {
                return null;
            }
            this.context.logger.error(`Error retrieving vector ${vectorId}:`, error);
            throw error;
        }
    }
    /**
     * Count vectors in collection
     */
    async countVectors() {
        try {
            const collectionInfo = await this.context.request({
                method: 'GET',
                path: `/collections/${encodeURIComponent(this.collectionName)}`,
                timeout: 30
            });
            return collectionInfo.vectorCount || 0;
        }
        catch (error) {
            this.context.logger.error('Error getting vector count:', error);
            return 0;
        }
    }
    /**
     * Get cache performance statistics
     */
    getCacheStats() {
        const totalRequests = this.cacheHits + this.cacheMisses;
        const baseStats = this.context.queryCache.getStats();
        return {
            ...baseStats,
            cacheHits: this.cacheHits,
            cacheMisses: this.cacheMisses,
            hitRate: totalRequests > 0 ? this.cacheHits / totalRequests : 0
        };
    }
    /**
     * Clear query cache for this collection
     */
    async clearCache() {
        this.context.queryCache.clear();
        this.cacheHits = 0;
        this.cacheMisses = 0;
        return {
            success: true,
            message: `Cache cleared for collection '${this.collectionName}'`
        };
    }
    /**
     * Get collection dimensions (cached)
     */
    async getCollectionDimensions() {
        return this.context.getCollectionDimensions(this.collectionName);
    }
    /**
     * Validate vector dimensions against collection
     */
    async validateVectorDimensions(vector) {
        return this.context.validateVectorDimensions(this.collectionName, vector);
    }
    // Private helper methods
    /**
     * Validate dimensions for a batch of vectors
     */
    async validateBatchDimensions(vectors) {
        const expectedDims = await this.context.getCollectionDimensions(this.collectionName);
        if (expectedDims === undefined)
            return;
        const errors = [];
        for (let i = 0; i < vectors.length; i++) {
            const vector = vectors[i].vector || vectors[i].values || [];
            if (vector.length !== expectedDims) {
                errors.push(`Vector ${i} dimension mismatch: got ${vector.length}, expected ${expectedDims}`);
            }
        }
        if (errors.length > 0) {
            const maxErrors = Math.min(5, errors.length);
            let errorMsg = errors.slice(0, maxErrors).join('; ');
            if (errors.length > maxErrors) {
                errorMsg += ` (and ${errors.length - maxErrors} more errors)`;
            }
            throw new Error(`Dimension validation failed: ${errorMsg}`);
        }
    }
    /**
     * Preprocess vectors using worker threads for CPU-intensive operations
     */
    async preprocessVectorsParallel(vectors, workerCount) {
        // For small datasets or if workers aren't available, process directly
        if (vectors.length < 1000 || !this.workerPool.isHealthy()) {
            return this.preprocessVectorsDirect(vectors);
        }
        try {
            // Split vectors into chunks for parallel processing
            const chunkSize = Math.ceil(vectors.length / workerCount);
            const chunks = [];
            for (let i = 0; i < vectors.length; i += chunkSize) {
                chunks.push(vectors.slice(i, i + chunkSize));
            }
            // Process chunks in parallel using worker threads
            const processedChunks = await Promise.all(chunks.map(chunk => this.workerPool.execute('preprocessVectors', {
                vectors: chunk,
                collectionName: this.collectionName
            })));
            // Flatten results
            return processedChunks.flat();
        }
        catch (error) {
            if (error.message) {
                this.context.logger.debug('Worker preprocessing failed, using main thread:', error.message);
            }
            return this.preprocessVectorsDirect(vectors);
        }
    }
    /**
     * Direct preprocessing for small datasets or fallback
     */
    preprocessVectorsDirect(vectors) {
        return vectors.map(vec => {
            const vectorValues = vec.vector || vec.values;
            if (!vectorValues) {
                throw new Error(`Vector missing 'values' or 'vector' field for ID: ${vec.id}`);
            }
            // Ensure all values are numbers and optimize for JSON serialization
            const optimizedVector = this.optimizeVector(vectorValues);
            return {
                id: String(vec.id),
                vector: optimizedVector,
                ...(vec.metadata && { metadata: vec.metadata })
            };
        });
    }
    /**
     * Optimize vector for JSON serialization and network transmission
     */
    optimizeVector(vector) {
        // Ensure all values are native JavaScript numbers (not numpy types)
        // Round to reasonable precision to reduce payload size
        return vector.map(v => {
            const num = Number(v);
            // Round to 6 decimal places for reasonable precision vs size tradeoff
            return Math.round(num * 1000000) / 1000000;
        });
    }
    /**
     * Clean up resources including worker pool
     */
    async close() {
        this.context.logger.debug(`Closing VectorOperations for collection '${this.collectionName}'`);
        // Terminate worker pool
        await this.workerPool.terminate();
        // Clear any pending timeouts
        if (this.upsertTimeout) {
            clearTimeout(this.upsertTimeout);
        }
    }
}
/**
 * Simple semaphore for controlling concurrency
 */
class Semaphore {
    constructor(permits) {
        this.waitQueue = [];
        this.permits = permits;
    }
    async acquire() {
        if (this.permits > 0) {
            this.permits--;
            return Promise.resolve();
        }
        return new Promise(resolve => {
            this.waitQueue.push(resolve);
        });
    }
    release() {
        this.permits++;
        if (this.waitQueue.length > 0) {
            this.permits--;
            const resolve = this.waitQueue.shift();
            resolve();
        }
    }
}
/**
 * Worker Pool for managing CPU-intensive tasks in separate threads
 * Now with proper fallback handling and path resolution
 */
class WorkerPool {
    constructor(maxWorkers, logger) {
        this.maxWorkers = maxWorkers;
        this.workers = [];
        this.availableWorkers = [];
        this.taskQueue = [];
        this.isTerminated = false;
        this.logger = logger;
        this.initializeWorkers();
    }
    initializeWorkers() {
        const currentDir = process.cwd();
        const possiblePaths = [
            join(currentDir, 'worker.cjs'),
            join(currentDir, 'worker.js'),
            join(currentDir, 'worker.ts'),
            resolve(currentDir, 'worker.cjs'),
            resolve(currentDir, 'worker.js'),
            resolve(currentDir, '..', 'src', 'worker.cjs'),
            resolve(currentDir, '..', 'src', 'worker.js'),
            resolve(currentDir, '..', 'dist', 'worker.cjs'),
            resolve(currentDir, '..', 'dist', 'worker.js'),
            resolve(process.cwd(), 'src', 'worker.cjs'),
            resolve(process.cwd(), 'src', 'worker.js'),
            resolve(process.cwd(), 'dist', 'worker.cjs'),
            resolve(process.cwd(), 'dist', 'worker.js'),
            resolve(process.cwd(), 'worker.cjs'),
            resolve(process.cwd(), 'worker.js')
        ];
        let workerPath = null;
        for (const path of possiblePaths) {
            console.log(path);
            if (existsSync(path)) {
                workerPath = path;
                break;
            }
        }
        if (!workerPath) {
            this.logger.debug('Worker file not found, falling back to main thread processing');
            return;
        }
        for (let i = 0; i < this.maxWorkers; i++) {
            try {
                const worker = new Worker(workerPath);
                worker.on('message', ({ taskId, result, error }) => {
                    if (taskId === 'worker_ready') {
                        return; // Worker initialization complete
                    }
                    // Find and resolve the corresponding task
                    const taskIndex = this.taskQueue.findIndex(t => t.taskId === taskId);
                    if (taskIndex !== -1) {
                        const task = this.taskQueue.splice(taskIndex, 1)[0];
                        if (error) {
                            task.reject(new Error(error));
                        }
                        else {
                            task.resolve(result);
                        }
                    }
                    // Return worker to available pool
                    if (!this.isTerminated) {
                        this.availableWorkers.push(worker);
                        this.processQueue();
                    }
                });
                worker.on('error', (error) => {
                    this.logger.debug('Worker error:', error.message);
                    // Remove failed worker from pools
                    const workerIndex = this.workers.indexOf(worker);
                    if (workerIndex > -1) {
                        this.workers.splice(workerIndex, 1);
                    }
                    const availableIndex = this.availableWorkers.indexOf(worker);
                    if (availableIndex > -1) {
                        this.availableWorkers.splice(availableIndex, 1);
                    }
                });
                worker.on('exit', (code) => {
                    if (code !== 0) {
                        this.logger.debug(`Worker stopped with exit code ${code}`);
                    }
                });
                this.workers.push(worker);
                this.availableWorkers.push(worker);
            }
            catch (error) {
                if (error.message) {
                    this.logger.debug('Failed to create worker:', error.message);
                }
            }
        }
        if (this.workers.length > 0) {
            this.logger.debug(`Initialized ${this.workers.length} worker threads`);
        }
        else {
            this.logger.debug('No worker threads available, all processing will run on main thread');
        }
    }
    async execute(task, data) {
        // If no workers available, fall back to main thread processing
        if (this.workers.length === 0 || this.isTerminated) {
            return this.executeMainThread(task, data);
        }
        return new Promise((resolve, reject) => {
            const taskId = `${task}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
            this.taskQueue.push({
                taskId,
                task,
                data,
                resolve,
                reject
            });
            this.processQueue();
            // Set timeout for long-running tasks
            setTimeout(() => {
                const taskIndex = this.taskQueue.findIndex(t => t.taskId === taskId);
                if (taskIndex !== -1) {
                    const failedTask = this.taskQueue.splice(taskIndex, 1)[0];
                    failedTask.reject(new Error(`Worker task timeout: ${task}`));
                }
            }, 30000); // 30 second timeout
        });
    }
    executeMainThread(task, data) {
        // Fallback to main thread processing
        return new Promise((resolve, reject) => {
            try {
                let result;
                switch (task) {
                    case 'preprocessVectors':
                        result = this.preprocessVectorsMainThread(data.vectors);
                        break;
                    case 'processQueryVectors':
                        result = this.processQueryVectorsMainThread(data.vectors);
                        break;
                    case 'processBatch':
                        result = this.processBatchMainThread(data.batch);
                        break;
                    default:
                        throw new Error(`Unsupported main thread task: ${task}`);
                }
                resolve(result);
            }
            catch (error) {
                reject(error);
            }
        });
    }
    preprocessVectorsMainThread(vectors) {
        return vectors.map(vec => {
            const vectorValues = vec.vector || vec.values;
            if (!vectorValues) {
                throw new Error(`Vector missing 'values' or 'vector' field for ID: ${vec.id}`);
            }
            const optimizedVector = this.optimizeVectorMainThread(vectorValues);
            return {
                id: String(vec.id),
                vector: optimizedVector,
                ...(vec.metadata && { metadata: vec.metadata })
            };
        });
    }
    processQueryVectorsMainThread(vectors) {
        return vectors.map(vec => this.optimizeVectorMainThread(vec));
    }
    processBatchMainThread(batch) {
        return batch.map(vec => {
            const vectorValues = vec.vector || vec.values || [];
            return {
                ...vec,
                vector: this.optimizeVectorMainThread(vectorValues)
            };
        });
    }
    optimizeVectorMainThread(vector) {
        return vector.map(v => {
            const num = Number(v);
            return Math.round(num * 1000000) / 1000000;
        });
    }
    processQueue() {
        while (this.taskQueue.length > 0 && this.availableWorkers.length > 0) {
            const task = this.taskQueue.shift();
            const worker = this.availableWorkers.shift();
            // Send task to worker
            worker.postMessage({
                task: task.task,
                data: task.data,
                taskId: task.taskId
            });
        }
    }
    async terminate() {
        this.isTerminated = true;
        // Reject all pending tasks
        this.taskQueue.forEach(task => {
            task.reject(new Error('Worker pool terminated'));
        });
        this.taskQueue = [];
        // Terminate all workers
        await Promise.all(this.workers.map(worker => worker.terminate()));
        this.workers = [];
        this.availableWorkers = [];
    }
    getStats() {
        return {
            totalWorkers: this.workers.length,
            availableWorkers: this.availableWorkers.length,
            queueLength: this.taskQueue.length
        };
    }
    isHealthy() {
        return this.workers.length > 0 && !this.isTerminated;
    }
}
//# sourceMappingURL=vector_ops.js.map