import { IDBHelper } from '../utils/idb.js';
import { embedder } from './embedder.js';
import logger from '../utils/logger.js';

class VectorStore {
    constructor() {
        this.dbName = 'llm-web-vectors';
        this.version = 1;
        this.db = null;
        this.helper = null;
        this.stores = {
            documents: 'documents',
            chunks: 'chunks',
            vectors: 'vectors',
            metadata: 'metadata'
        };
    }

    async initialize() {
        this.helper = new IDBHelper(this.dbName, this.version);

        await this.helper.open((db, oldVersion, newVersion) => {
            // Documents store
            if (!db.objectStoreNames.contains(this.stores.documents)) {
                const docStore = db.createObjectStore(this.stores.documents, {
                    keyPath: 'id',
                    autoIncrement: true
                });
                docStore.createIndex('name', 'name', { unique: false });
                docStore.createIndex('type', 'type', { unique: false });
                docStore.createIndex('createdAt', 'createdAt', { unique: false });
            }

            // Chunks store
            if (!db.objectStoreNames.contains(this.stores.chunks)) {
                const chunkStore = db.createObjectStore(this.stores.chunks, {
                    keyPath: 'id',
                    autoIncrement: true
                });
                chunkStore.createIndex('docId', 'docId', { unique: false });
                chunkStore.createIndex('position', 'position', { unique: false });
            }

            // Vectors store (stores Float32Array as Blob)
            if (!db.objectStoreNames.contains(this.stores.vectors)) {
                const vectorStore = db.createObjectStore(this.stores.vectors, {
                    keyPath: 'id'
                });
                vectorStore.createIndex('chunkId', 'chunkId', { unique: true });
            }

            // Metadata store
            if (!db.objectStoreNames.contains(this.stores.metadata)) {
                const metaStore = db.createObjectStore(this.stores.metadata, {
                    keyPath: 'key'
                });
            }
        });

        // Initialize metadata
        const meta = await this.helper.get(this.stores.metadata, 'config');
        if (!meta) {
            await this.helper.put(this.stores.metadata, {
                key: 'config',
                chunkSize: 800,
                chunkOverlap: 200,
                dimensions: null,
                totalDocuments: 0,
                totalChunks: 0,
                createdAt: Date.now()
            });
        }

        return true;
    }

    async addDocument(content, metadata = {}) {
        logger.log(`[VectorStore] Adding document: ${metadata.name || 'Untitled'} (${content.length} chars`);

        const doc = {
            name: metadata.name || 'Untitled',
            type: metadata.type || 'text',
            content: content,
            size: content.length,
            createdAt: Date.now(),
            ...metadata
        };

        const docId = await this.helper.add(this.stores.documents, doc);

        // Chunk the document
        const chunks = this.chunkText(content, docId);
        logger.log(`[VectorStore] Created ${chunks.length} chunks for document ${docId}`);

        // Add chunks and generate embeddings
        const chunkIds = [];
        for (let i = 0; i < chunks.length; i++) {
            const chunk = chunks[i];
            const chunkId = await this.helper.add(this.stores.chunks, chunk);
            chunkIds.push(chunkId);

            // Generate embedding
            const embedding = await embedder.embed(chunk.text);
            await this.storeVector(chunkId, embedding);

            if (i === 0) {
                logger.log(`[VectorStore] First chunk embedding dims: ${embedding.length}`);
            }
        }

        logger.log(`[VectorStore] Successfully stored ${chunkIds.length} embeddings for document ${docId}`);

        // Update metadata
        const config = await this.helper.get(this.stores.metadata, 'config');
        config.totalDocuments++;
        config.totalChunks += chunks.length;
        if (!config.dimensions && chunks.length > 0) {
            const firstVector = await this.getVector(chunkIds[0]);
            config.dimensions = firstVector.length;
        }
        await this.helper.put(this.stores.metadata, config);

        // Verify embeddings were created
        let verifiedCount = 0;
        for (const chunkId of chunkIds) {
            const vector = await this.helper.get(this.stores.vectors, chunkId);
            if (vector) {
                verifiedCount++;
            }
        }

        if (verifiedCount !== chunkIds.length) {
            logger.warn(`[VectorStore] Only ${verifiedCount}/${chunkIds.length} embeddings were verified`);
        }

        return {
            docId,
            chunksCreated: chunks.length,
            chunkCount: chunks.length,
            embeddingCount: verifiedCount,
            success: verifiedCount === chunkIds.length,
            chunkIds
        };
    }

    async addDocuments(documents, onProgress) {
        const results = [];
        const total = documents.length;

        for (let i = 0; i < total; i++) {
            const doc = documents[i];
            const result = await this.addDocument(
                doc.content,
                doc.metadata || {}
            );
            results.push(result);

            if (onProgress) {
                onProgress({
                    current: i + 1,
                    total,
                    percent: ((i + 1) / total) * 100,
                    document: doc.metadata?.name || `Document ${i + 1}`
                });
            }
        }

        return results;
    }

    chunkText(text, docId, chunkSize = 800, overlap = 200) {
        const chunks = [];
        const words = text.split(/\s+/);

        // Approximate token count (rough estimate: 1 token ≈ 0.75 words)
        const approximateTokens = (text) => Math.ceil(text.split(/\s+/).length / 0.75);

        let currentChunk = [];
        let currentTokenCount = 0;
        let position = 0;

        for (let i = 0; i < words.length; i++) {
            const word = words[i];
            const wordTokens = approximateTokens(word);

            if (currentTokenCount + wordTokens > chunkSize && currentChunk.length > 0) {
                // Save current chunk
                chunks.push({
                    docId,
                    text: currentChunk.join(' '),
                    position: position++,
                    startIndex: i - currentChunk.length,
                    endIndex: i,
                    tokenCount: currentTokenCount
                });

                // Start new chunk with overlap
                const overlapWords = Math.floor(currentChunk.length * (overlap / chunkSize));
                currentChunk = currentChunk.slice(-overlapWords);
                currentTokenCount = approximateTokens(currentChunk.join(' '));
            }

            currentChunk.push(word);
            currentTokenCount += wordTokens;
        }

        // Save the last chunk
        if (currentChunk.length > 0) {
            chunks.push({
                docId,
                text: currentChunk.join(' '),
                position: position,
                startIndex: words.length - currentChunk.length,
                endIndex: words.length,
                tokenCount: currentTokenCount
            });
        }

        return chunks;
    }

    async storeVector(chunkId, vector) {
        // Convert to Float32Array if needed
        const float32Vector = vector instanceof Float32Array
            ? vector
            : new Float32Array(vector);

        // Store as blob for efficiency
        const blob = new Blob([float32Vector.buffer]);

        await this.helper.put(this.stores.vectors, {
            id: chunkId,
            chunkId,
            blob,
            dimensions: float32Vector.length
        });
    }

    async getVector(chunkId) {
        const record = await this.helper.get(this.stores.vectors, chunkId);
        if (!record) return null;

        const arrayBuffer = await record.blob.arrayBuffer();
        return Array.from(new Float32Array(arrayBuffer));
    }

    async queryVector(queryText, options = {}) {
        const {
            topK = 10,
            threshold = 0.0,
            useMMR = false,
            mmrLambda = 0.5
        } = options;

        logger.log(`[VectorStore] Searching for: "${queryText.substring(0, 50)}..." (threshold: ${threshold}, topK: ${topK}`);

        // Generate query embedding
        const queryVector = await embedder.embed(queryText);

        // Get all vectors
        const allVectors = await this.helper.getAll(this.stores.vectors);
        logger.log(`[VectorStore] Searching through ${allVectors.length} vectors`);
        const vectorsWithScores = [];

        for (const record of allVectors) {
            const arrayBuffer = await record.blob.arrayBuffer();
            const vector = Array.from(new Float32Array(arrayBuffer));

            const score = embedder.cosineSimilarity(queryVector, vector);

            if (score >= threshold) {
                const chunk = await this.helper.get(this.stores.chunks, record.chunkId);
                const doc = await this.helper.get(this.stores.documents, chunk.docId);

                vectorsWithScores.push({
                    vector,
                    score,
                    chunk,
                    document: doc,
                    chunkId: record.chunkId
                });
            }
        }

        logger.log(`[VectorStore] Found ${vectorsWithScores.length} results above threshold`);
        if (vectorsWithScores.length > 0) {
            logger.log(`[VectorStore] Top result score: ${vectorsWithScores[0]?.score?.toFixed(3)} from doc: ${vectorsWithScores[0]?.document?.name}`);
        }

        // Apply ranking strategy
        let results;
        if (useMMR && vectorsWithScores.length > 0) {
            results = embedder.mmrRerank(queryVector, vectorsWithScores, mmrLambda, topK);
            logger.log(`[VectorStore] Applied MMR reranking`);
        } else {
            results = embedder.findTopK(queryVector, vectorsWithScores, topK);
        }

        logger.log(`[VectorStore] Returning ${results.length} results`);
        return results;
    }

    async search(query, options = {}) {
        const startTime = performance.now();
        const results = await this.queryVector(query, options);
        const searchTime = performance.now() - startTime;

        return {
            query,
            results,
            searchTime,
            totalResults: results.length
        };
    }

    async deleteDocument(docId) {
        // Delete chunks
        const chunks = await this.helper.getAll(this.stores.chunks);
        const chunkIds = chunks
            .filter(chunk => chunk.docId === docId)
            .map(chunk => chunk.id);

        for (const chunkId of chunkIds) {
            await this.helper.delete(this.stores.chunks, chunkId);
            await this.helper.delete(this.stores.vectors, chunkId);
        }

        // Delete document
        await this.helper.delete(this.stores.documents, docId);

        // Update metadata
        const config = await this.helper.get(this.stores.metadata, 'config');
        config.totalDocuments--;
        config.totalChunks -= chunkIds.length;
        await this.helper.put(this.stores.metadata, config);

        return {
            deletedChunks: chunkIds.length
        };
    }

    async clear() {
        await this.helper.clear(this.stores.documents);
        await this.helper.clear(this.stores.chunks);
        await this.helper.clear(this.stores.vectors);

        // Reset metadata
        const config = await this.helper.get(this.stores.metadata, 'config');
        config.totalDocuments = 0;
        config.totalChunks = 0;
        await this.helper.put(this.stores.metadata, config);
    }

    async getAllDocuments() {
        const documents = await this.helper.getAll(this.stores.documents);

        // Add chunk count for each document
        const chunks = await this.helper.getAll(this.stores.chunks);
        const chunkCountByDoc = {};

        chunks.forEach(chunk => {
            chunkCountByDoc[chunk.docId] = (chunkCountByDoc[chunk.docId] || 0) + 1;
        });

        return documents.map(doc => ({
            ...doc,
            chunkCount: chunkCountByDoc[doc.id] || 0,
            sizeKB: (doc.size / 1024).toFixed(2)
        }));
    }

    async getStats() {
        const config = await this.helper.get(this.stores.metadata, 'config');
        const docCount = await this.helper.count(this.stores.documents);
        const chunkCount = await this.helper.count(this.stores.chunks);
        const vectorCount = await this.helper.count(this.stores.vectors);

        // Calculate storage size (rough estimate)
        const estimatedSize = vectorCount * (config.dimensions || 384) * 4; // 4 bytes per float

        return {
            documents: docCount,
            chunks: chunkCount,
            vectors: vectorCount,
            dimensions: config.dimensions,
            estimatedSizeMB: (estimatedSize / (1024 * 1024)).toFixed(2),
            createdAt: config.createdAt
        };
    }

    async compact() {
        // Remove orphaned chunks and vectors
        const allDocs = await this.helper.getAll(this.stores.documents);
        const validDocIds = new Set(allDocs.map(doc => doc.id));

        const allChunks = await this.helper.getAll(this.stores.chunks);
        let removedChunks = 0;

        for (const chunk of allChunks) {
            if (!validDocIds.has(chunk.docId)) {
                await this.helper.delete(this.stores.chunks, chunk.id);
                await this.helper.delete(this.stores.vectors, chunk.id);
                removedChunks++;
            }
        }

        return {
            removedChunks,
            message: `Compaction complete. Removed ${removedChunks} orphaned chunks.`
        };
    }

    async close() {
        if (this.helper) {
            this.helper.close();
        }
    }

    async destroy() {
        if (this.helper) {
            await this.helper.deleteDatabase();
        }
    }
}

// Create singleton instance
const vectorStore = new VectorStore();

export { vectorStore, VectorStore };

export async function initializeVectorStore() {
    return vectorStore.initialize();
}

export async function addDocument(content, metadata) {
    return vectorStore.addDocument(content, metadata);
}

export async function searchDocuments(query, options) {
    return vectorStore.search(query, options);
}

export async function getVectorStoreStats() {
    return vectorStore.getStats();
}

export async function getAllDocuments() {
    return vectorStore.getAllDocuments();
}

export async function deleteDocument(docId) {
    return vectorStore.deleteDocument(docId);
}

export async function clearAllDocuments() {
    return vectorStore.clear();
}