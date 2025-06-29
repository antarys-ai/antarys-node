export { Client, createClient } from './client';
export { VectorOperations } from './vector_ops';
export { QueryCache, BufferPool, HighPerformanceCache } from './caching';

export * from '../shared/types';


export async function createCollection(
	client: Client,
	params: CreateCollectionParams
): Promise<ApiResponse> {
	return client.createCollection(params);
}

import { ApiResponse, CreateCollectionParams } from '../shared/types';
import { Client, createClient } from './client';
export default { Client, createClient };
