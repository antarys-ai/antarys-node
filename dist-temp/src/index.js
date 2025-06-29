export { Client, createClient } from './client';
export { VectorOperations } from './vector_ops';
export { QueryCache, BufferPool, HighPerformanceCache } from './caching';
export * from '../shared/types';
export async function createCollection(client, params) {
    return client.createCollection(params);
}
import { Client, createClient } from './client';
export default { Client, createClient };
//# sourceMappingURL=index.js.map