import { parentPort } from 'worker_threads';
/**
 * Optimize vector for JSON serialization and network transmission
 * Round to reasonable precision to reduce payload size
 */
function optimizeVector(vector) {
    return vector.map(v => {
        const num = Number(v);
        // Round to 6 decimal places for reasonable precision vs size tradeoff
        return Math.round(num * 1000000) / 1000000;
    });
}
/**
 * Preprocess vectors for optimal serialization
 */
function preprocessVectors(vectors) {
    return vectors.map(vec => {
        const vectorValues = vec.vector || vec.values;
        if (!vectorValues) {
            throw new Error(`Vector missing 'values' or 'vector' field for ID: ${vec.id}`);
        }
        // Ensure all values are numbers and optimize for JSON serialization
        const optimizedVector = optimizeVector(vectorValues);
        return {
            id: String(vec.id),
            vector: optimizedVector,
            ...(vec.metadata && { metadata: vec.metadata })
        };
    });
}
/**
 * Process query vectors for optimal performance
 */
function processQueryVectors(vectors) {
    return vectors.map(vec => optimizeVector(vec));
}
/**
 * Process batch data with additional optimizations
 */
function processBatch(batch) {
    return batch.map(vec => {
        const vectorValues = vec.vector || vec.values || [];
        // Additional validation and processing
        if (!Array.isArray(vectorValues)) {
            throw new Error(`Invalid vector format for ID: ${vec.id}`);
        }
        return {
            ...vec,
            vector: optimizeVector(vectorValues)
        };
    });
}
/**
 * Calculate vector similarity using cosine similarity
 */
function calculateCosineSimilarity(vectorA, vectorB) {
    if (vectorA.length !== vectorB.length) {
        throw new Error('Vectors must have the same dimensions');
    }
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vectorA.length; i++) {
        dotProduct += vectorA[i] * vectorB[i];
        normA += vectorA[i] * vectorA[i];
        normB += vectorB[i] * vectorB[i];
    }
    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude === 0 ? 0 : dotProduct / magnitude;
}
/**
 * Batch similarity calculation for local filtering/ranking
 */
function batchSimilarityCalculation(queryVector, targetVectors) {
    return targetVectors.map(targetVector => calculateCosineSimilarity(queryVector, targetVector));
}
/**
 * Normalize vectors to unit length for cosine similarity optimization
 */
function normalizeVectors(vectors) {
    return vectors.map(vector => {
        const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
        return norm === 0 ? vector : vector.map(val => val / norm);
    });
}
/**
 * Validate vector dimensions against expected dimensions
 */
function validateVectorDimensions(vectors, expectedDimensions) {
    const errors = [];
    for (let i = 0; i < vectors.length; i++) {
        const vector = vectors[i].vector || vectors[i].values || vectors[i];
        if (!Array.isArray(vector)) {
            errors.push(`Vector ${i}: Not an array`);
            continue;
        }
        if (vector.length !== expectedDimensions) {
            errors.push(`Vector ${i}: Expected ${expectedDimensions} dimensions, got ${vector.length}`);
        }
        // Check for invalid values
        const hasInvalidValues = vector.some(val => typeof val !== 'number' || !isFinite(val));
        if (hasInvalidValues) {
            errors.push(`Vector ${i}: Contains invalid numeric values`);
        }
    }
    return {
        valid: errors.length === 0,
        errors
    };
}
/**
 * Convert vectors to Float32Array for better memory efficiency and performance
 */
function convertToFloat32Arrays(vectors) {
    return vectors.map(vector => new Float32Array(vector));
}
/**
 * Compress vectors using simple quantization (for storage optimization)
 */
function quantizeVectors(vectors, precision = 1000) {
    return vectors.map(vector => vector.map(value => Math.round(value * precision) / precision));
}
/**
 * Main worker message handler
 */
function handleWorkerMessage(message) {
    const { task, data, taskId } = message;
    try {
        let result;
        switch (task) {
            case 'preprocessVectors':
                result = preprocessVectors(data.vectors);
                break;
            case 'processQueryVectors':
                result = processQueryVectors(data.vectors);
                break;
            case 'processBatch':
                result = processBatch(data.batch);
                break;
            case 'calculateSimilarity':
                result = calculateCosineSimilarity(data.vectorA, data.vectorB);
                break;
            case 'batchSimilarity':
                result = batchSimilarityCalculation(data.queryVector, data.targetVectors);
                break;
            case 'normalizeVectors':
                result = normalizeVectors(data.vectors);
                break;
            case 'validateDimensions':
                result = validateVectorDimensions(data.vectors, data.expectedDimensions);
                break;
            case 'convertToFloat32':
                result = convertToFloat32Arrays(data.vectors);
                break;
            case 'quantizeVectors':
                result = quantizeVectors(data.vectors, data.precision);
                break;
            case 'optimizeVector':
                result = optimizeVector(data.vector);
                break;
            default:
                throw new Error(`Unknown task: ${task}`);
        }
        return { taskId, result };
    }
    catch (error) {
        return {
            taskId,
            error: error instanceof Error ? error.message : String(error)
        };
    }
}
// Set up message listener if this is running as a worker thread
if (parentPort) {
    parentPort.on('message', (message) => {
        const response = handleWorkerMessage(message);
        parentPort.postMessage(response);
    });
    // Signal that worker is ready
    parentPort.postMessage({ taskId: 'worker_ready', result: true });
}
// Export functions for testing or direct use
export { optimizeVector, preprocessVectors, processQueryVectors, processBatch, calculateCosineSimilarity, batchSimilarityCalculation, normalizeVectors, validateVectorDimensions, convertToFloat32Arrays, quantizeVectors, handleWorkerMessage };
//# sourceMappingURL=worker.js.map