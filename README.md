# Antarys Vector Database Node.js Client

High-performance TypeScript/Node.js client for Antarys vector database with HTTP/2, connection pooling, worker thread parallelization, and built in caching.

## Installation

Install via [npm package](https://www.npmjs.com/package/antarys)

```bash
npm install antarys
```

```bash
yarn add antarys
```

```bash
pnpm add antarys
```

## Quick Start

```typescript
import { createClient } from 'antarys';

async function main() {
    // Initialize client with performance optimizations
    const client = createClient('http://localhost:8080', {
        connectionPoolSize: 100,  // Auto-sized based on CPU count
        compression: true,
        cacheSize: 1000,
        threadPoolSize: 16,
        debug: true
    });

    // Create collection
    await client.createCollection({
        name: 'my_vectors',
        dimensions: 1536,
        enableHnsw: true,
        shards: 16
    });

    const vectors = client.vectorOperations('my_vectors');

    // Upsert vectors
    await vectors.upsert([
        { id: '1', values: Array(1536).fill(0.1), metadata: { category: 'A' } },
        { id: '2', values: Array(1536).fill(0.2), metadata: { category: 'B' } }
    ]);

    // Query similar vectors
    const results = await vectors.query({
        vector: Array(1536).fill(0.1),
        topK: 10,
        includeMetadata: true
    });

    await client.close();
}

main().catch(console.error);
```

## Simple RAG Example

Here's a complete example showing how to build a Retrieval-Augmented Generation (RAG) system with Antarys and OpenAI:

```typescript
import OpenAI from 'openai';
import { createClient } from "antarys";

class SimpleRAG {
    private openai: OpenAI;
    private antarys: any;
    private vectors: any;

    constructor() {
        this.openai = new OpenAI();
        this.antarys = null;
        this.vectors = null;
    }

    async init(): Promise<void> {
        this.antarys = createClient("http://localhost:8080");

        // Try to create collection, ignore if exists
        try {
            await this.antarys.createCollection({
                name: "docs",
                dimensions: 1536
            });
        } catch (error: any) {
            if (!error.message.includes('already exists')) {
                throw error;
            }
        }

        this.vectors = this.antarys.vectorOperations("docs");
    }

    async embed(text: string): Promise<number[]> {
        const response = await this.openai.embeddings.create({
            model: "text-embedding-3-small",
            input: text
        });
        return response.data[0].embedding;
    }

    async add(docId: string, content: string): Promise<void> {
        const embedding = await this.embed(content);
        await this.vectors.upsert([{
            id: docId,
            values: embedding,
            metadata: { content }
        }]);
    }

    async search(query: string, topK: number = 3): Promise<any[]> {
        const embedding = await this.embed(query);
        const results = await this.vectors.query({
            vector: embedding,
            topK,
            includeMetadata: true
        });
        return results.matches;
    }

    async generate(query: string, docs: any[]): Promise<string> {
        const context = docs.map(doc => doc.metadata.content).join("\n");
        const response = await this.openai.chat.completions.create({
            model: "gpt-4",
            messages: [{
                role: "user",
                content: `Context: ${context}\n\nQuestion: ${query}`
            }]
        });
        return response.choices[0].message.content || '';
    }

    async query(question: string, verbose: boolean = false): Promise<[string, any[]]> {
        const docs = await this.search(question);
        const answer = await this.generate(question, docs);

        if (verbose) {
            console.log(`Q: ${question}`);
            console.log(`A: ${answer}`);
            docs.forEach(doc => {
                console.log(`Source: ${doc.id} (${doc.score.toFixed(3)})`);
            });
        }

        return [answer, docs];
    }

    async close(): Promise<void> {
        if (this.antarys) {
            await this.antarys.close();
        }
    }
}

async function main() {
    const rag = new SimpleRAG();

    await rag.init();

    await rag.add("AHNSW",
        "Unlike traditional sequential HNSW, we are using a different asynchronous approach to HNSW and eliminating thread locks with the help of architectural fine tuning. We will soon release more technical details on the Async HNSW algorithmic approach.");
    await rag.add("Antarys",
        "Antarys is a multi-modal vector database and it uses the AHNSW algorithm to enhance its performance to perform semantic searching based on similarity");

    await rag.query("what is Antarys?", true);

    await rag.close();
}

main().catch(console.error);
```

## Core Concepts

### Collections

```typescript
import { Client, createClient } from 'antarys';

const client = createClient('http://localhost:8080');

// Create collection with optimized parameters
await client.createCollection({
    name: 'vectors',
    dimensions: 1536,        // Required: vector dimensions
    enableHnsw: true,        // Enable HNSW indexing for fast ANN
    shards: 16,              // Parallel processing shards
    m: 16,                   // HNSW connectivity parameter
    efConstruction: 200      // HNSW construction quality
});

// List collections
const collections = await client.listCollections();

// Get collection info
const info = await client.describeCollection('vectors');

// Delete collection
await client.deleteCollection('vectors');
```

### Vector Operations

#### Single Vector Upsert

```typescript
const vectors = client.vectorOperations('my_collection');

const record = {
    id: '1',
    values: [0.1, 0.2, 0.3],  // Must match collection dimensions
    metadata: { category: 'example', score: 0.95 }
};

await vectors.upsert([record]);
```

#### Batch Vector Upsert

```typescript
const largeDataset = [
    { id: '1', values: Array(1536).fill(0.1), metadata: { type: 'doc' } },
    { id: '2', values: Array(1536).fill(0.2), metadata: { type: 'image' } },
    // ... thousands more
];

await vectors.upsert(largeDataset, {
    batchSize: 5000,          // Optimal batch size
    parallelWorkers: 8,       // Worker threads for preprocessing
    validateDimensions: true, // Ensure vector dimensions match
    showProgress: true        // Display progress bar
});
```

#### Query Similar Vectors

```typescript
// Basic similarity search
const results = await vectors.query({
    vector: Array(1536).fill(0.1),
    topK: 10,
    includeMetadata: true,
    includeValues: false      // Reduce response size
});

// Advanced search with filtering
const filteredResults = await vectors.query({
    vector: queryVector,
    topK: 20,
    includeMetadata: true,
    filter: { category: 'documents' },  // Metadata filtering
    threshold: 0.8,                     // Minimum similarity score
    useAnn: true,                       // Use approximate search
    efSearch: 200                       // HNSW search quality
});
```

#### Batch Query Operations

```typescript
const queries = [
    { vector: Array(1536).fill(0.1), topK: 5 },
    { vector: Array(1536).fill(0.2), topK: 5 },
    { vector: Array(1536).fill(0.3), topK: 5 }
];

const batchResults = await vectors.batchQuery(queries);
```

#### Delete Vectors

```typescript
// Delete single vector
await vectors.deleteVector('vector_id_1');

// Delete multiple vectors
await vectors.deleteVectors(['id1', 'id2', 'id3']);

// Delete with filtering
await vectors.deleteByFilter({ category: 'outdated' });
```

#### Get Vector by ID

```typescript
const vector = await vectors.getVector('vector_id_1');
if (vector) {
    console.log('Found:', vector.metadata);
}
```

### Memory and Resource Management

```typescript
// Force commit for persistence
await client.commit();

// Clear client-side caches
await client.clearCache();
await vectors.clearCache();

// Get performance statistics
const stats = client.getStats();
console.log('Cache hit rate:', stats.cache.hitRate);
console.log('Average request time:', stats.requests.avgTime);

// Proper resource cleanup
await client.close();
```

## Advanced Features

### Client Configuration

```typescript
import { createClient } from 'antarys';

const client = createClient('http://localhost:8080', {
    // Connection Pool Optimization
    connectionPoolSize: 100,     // High concurrency (auto: CPU_COUNT * 5)
    timeout: 120,                // Extended timeout for large operations

    // HTTP/2 and Compression
    compression: true,           // Enable response compression

    // Caching Configuration
    cacheSize: 1000,            // Client-side query cache
    cacheTtl: 300,              // Cache TTL in seconds

    // Threading and Parallelism
    threadPoolSize: 16,         // CPU-bound operations (auto: CPU_COUNT * 2)

    // Retry Configuration
    retryAttempts: 5,           // Network resilience

    // Debug Mode
    debug: true                 // Performance monitoring
});
```

### Dimension Validation

```typescript
// Automatic dimension validation
const isValid = await vectors.validateVectorDimensions([0.1, 0.2, 0.3]);

// Get collection dimensions
const dims = await vectors.getCollectionDimensions();
```

### Cache Performance Monitoring

```typescript
// Get cache statistics
const stats = vectors.getCacheStats();
console.log(`Cache hit rate: ${(stats.hitRate * 100).toFixed(2)}%`);
console.log(`Cache size: ${stats.cacheSize}`);
```

## Performance Optimization

### Recommended Settings by Scale

#### Small Scale (< 1M vectors)

```typescript
const client = createClient('http://localhost:8080', {
    connectionPoolSize: 20,
    cacheSize: 500,
    threadPoolSize: 4
});

const batchSize = 1000;
const parallelWorkers = 2;
```

#### Medium Scale (1M - 10M vectors)

```typescript
const client = createClient('http://localhost:8080', {
    connectionPoolSize: 50,
    cacheSize: 2000,
    threadPoolSize: 8
});

const batchSize = 3000;
const parallelWorkers = 4;
```

#### Large Scale (10M+ vectors)

```typescript
const client = createClient('http://localhost:8080', {
    connectionPoolSize: 100,
    cacheSize: 5000,
    threadPoolSize: 16
});

const batchSize = 5000;
const parallelWorkers = 8;
```

### Batch Operation Tuning

```typescript
// Optimal batch upsert parameters
await vectors.upsert(largeDataset, {
    batchSize: 5000,           // Optimal for network efficiency
    parallelWorkers: 8,        // Match server capability
    validateDimensions: true,  // Prevent dimension errors
    showProgress: true
});

// High-throughput query configuration
const results = await vectors.query({
    vector: queryVector,
    topK: 100,
    includeValues: false,      // Reduce response size
    includeMetadata: true,
    useAnn: true,              // Fast approximate search
    efSearch: 200,             // Higher quality (vs speed)
    skipCache: false           // Leverage cache
});
```

### Server-Side Optimization

#### HNSW Index Parameters

```typescript
await client.createCollection({
    name: 'high_performance',
    dimensions: 1536,
    enableHnsw: true,

    // HNSW Tuning
    m: 16,                     // Connectivity (16-64 for high recall)
    efConstruction: 200,       // Graph construction quality (200-800)
    shards: 32                 // Parallel processing (match CPU cores)
});

// Query-time HNSW parameters
const results = await vectors.query({
    vector: queryVector,
    efSearch: 200,             // Search quality (100-800)
    useAnn: true               // Enable HNSW acceleration
});
```

## TypeScript Support

The client is fully typed with TypeScript:

```typescript
import {
    Client,
    VectorRecord,
    SearchParams,
    SearchResults,
    CreateCollectionParams,
    ClientConfig
} from 'antarys';

// Type-safe vector record
const record: VectorRecord = {
    id: 'example',
    values: [0.1, 0.2, 0.3],
    metadata: { key: 'value' }
};

// Search parameters with full type safety
const params: SearchParams = {
    vector: [0.1, 0.2, 0.3],
    topK: 10,
    includeMetadata: true,
    threshold: 0.8
};
```

## Health Monitoring

```typescript
// Check server health
const health = await client.health();
console.log('Server status:', health.status);

// Get server information
const info = await client.info();
console.log('Server version:', info.version);

// Collection statistics
const collectionInfo = await client.describeCollection('vectors');
console.log('Vector count:', collectionInfo.vectorCount || 0);
```

## Error Handling

```typescript
try {
    await vectors.upsert(records);
} catch (error) {
    if (error.message.includes('dimension mismatch')) {
        console.error('Vector dimensions do not match collection schema');
    } else if (error.message.includes('connection')) {
        console.error('Network connectivity issue');
    } else {
        console.error('Unexpected error:', error);
    }
}
```

## License

MIT
