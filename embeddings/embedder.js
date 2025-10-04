import { modelLoader } from '../llm/loader.js';
import logger from '../utils/logger.js';

class Embedder {
    constructor() {
        this.pipeline = null;
        this.modelId = null;
        this.dimensions = null;
    }

    async initialize(modelId = 'Xenova/all-MiniLM-L6-v2', options = {}) {
        if (this.pipeline && this.modelId === modelId) {
            return this.pipeline;
        }

        this.modelId = modelId;
        this.pipeline = await modelLoader.loadEmbedder(modelId, options);

        // Get embedding dimensions by generating a test embedding
        const testEmbedding = await this.embed('test', { pooling: 'mean', normalize: true });
        this.dimensions = testEmbedding.length;
        logger.log(`[Embedder] Initialized with ${this.dimensions} dimensions`);

        return this.pipeline;
    }

    async embed(text, options = {}) {
        if (!this.pipeline) {
            throw new Error('Embedder not initialized. Call initialize() first.');
        }

        const defaultOptions = {
            pooling: 'mean',
            normalize: true
        };

        const embedOptions = { ...defaultOptions, ...options };

        const result = await this.pipeline(text, embedOptions);

        // Convert to regular array if it's a tensor
        if (result.data instanceof Float32Array) {
            return Array.from(result.data);
        }

        return result;
    }

    async embedBatch(texts, options = {}) {
        if (!Array.isArray(texts)) {
            throw new Error('embedBatch expects an array of texts');
        }

        const embeddings = [];

        // Process in smaller batches to avoid memory issues
        const batchSize = options.batchSize || 10;
        for (let i = 0; i < texts.length; i += batchSize) {
            const batch = texts.slice(i, i + batchSize);
            const batchEmbeddings = await Promise.all(
                batch.map(text => this.embed(text, options))
            );
            embeddings.push(...batchEmbeddings);
        }

        return embeddings;
    }

    cosineSimilarity(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have the same dimensions');
        }

        let dotProduct = 0;
        let norm1 = 0;
        let norm2 = 0;

        for (let i = 0; i < vec1.length; i++) {
            dotProduct += vec1[i] * vec2[i];
            norm1 += vec1[i] * vec1[i];
            norm2 += vec2[i] * vec2[i];
        }

        norm1 = Math.sqrt(norm1);
        norm2 = Math.sqrt(norm2);

        if (norm1 === 0 || norm2 === 0) {
            return 0;
        }

        return dotProduct / (norm1 * norm2);
    }

    euclideanDistance(vec1, vec2) {
        if (vec1.length !== vec2.length) {
            throw new Error('Vectors must have the same dimensions');
        }

        let sum = 0;
        for (let i = 0; i < vec1.length; i++) {
            const diff = vec1[i] - vec2[i];
            sum += diff * diff;
        }

        return Math.sqrt(sum);
    }

    findTopK(queryVector, vectors, k = 10, metric = 'cosine') {
        const scores = vectors.map((vec, index) => {
            let score;
            if (metric === 'cosine') {
                score = this.cosineSimilarity(queryVector, vec.vector);
            } else if (metric === 'euclidean') {
                score = -this.euclideanDistance(queryVector, vec.vector); // Negative for descending sort
            } else {
                throw new Error(`Unknown metric: ${metric}`);
            }

            return {
                ...vec,
                index,
                score
            };
        });

        // Sort by score (descending)
        scores.sort((a, b) => b.score - a.score);

        return scores.slice(0, k);
    }

    // Maximum Marginal Relevance (MMR) for diversity
    mmrRerank(queryVector, vectors, lambda = 0.5, k = 10) {
        if (vectors.length === 0) return [];

        const selected = [];
        const remaining = [...vectors];

        // First, select the most relevant document
        let maxScore = -Infinity;
        let maxIndex = -1;

        remaining.forEach((vec, index) => {
            const score = this.cosineSimilarity(queryVector, vec.vector);
            if (score > maxScore) {
                maxScore = score;
                maxIndex = index;
            }
        });

        if (maxIndex >= 0) {
            selected.push({
                ...remaining[maxIndex],
                score: maxScore
            });
            remaining.splice(maxIndex, 1);
        }

        // Select remaining documents based on MMR
        while (selected.length < k && remaining.length > 0) {
            maxScore = -Infinity;
            maxIndex = -1;

            remaining.forEach((candidate, index) => {
                const relevance = this.cosineSimilarity(queryVector, candidate.vector);

                // Calculate max similarity to already selected documents
                let maxSim = 0;
                selected.forEach(selectedDoc => {
                    const sim = this.cosineSimilarity(candidate.vector, selectedDoc.vector);
                    if (sim > maxSim) {
                        maxSim = sim;
                    }
                });

                // MMR score
                const mmrScore = lambda * relevance - (1 - lambda) * maxSim;

                if (mmrScore > maxScore) {
                    maxScore = mmrScore;
                    maxIndex = index;
                }
            });

            if (maxIndex >= 0) {
                selected.push({
                    ...remaining[maxIndex],
                    score: maxScore
                });
                remaining.splice(maxIndex, 1);
            } else {
                break;
            }
        }

        return selected;
    }

    getDimensions() {
        return this.dimensions;
    }

    isInitialized() {
        return this.pipeline !== null;
    }
}

// Create singleton instance
const embedder = new Embedder();

export { embedder, Embedder };

export async function initializeEmbedder(modelId, options) {
    return embedder.initialize(modelId, options);
}

export async function embedText(text, options) {
    return embedder.embed(text, options);
}

export async function embedTexts(texts, options) {
    return embedder.embedBatch(texts, options);
}

export function cosineSimilarity(vec1, vec2) {
    return embedder.cosineSimilarity(vec1, vec2);
}

export function findTopK(queryVector, vectors, k, metric) {
    return embedder.findTopK(queryVector, vectors, k, metric);
}