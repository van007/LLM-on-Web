import { vectorStore } from '../embeddings/store.js';
import logger from '../utils/logger.js';

class RAGPipeline {
    constructor() {
        this.defaultOptions = {
            maxContextTokens: 2000,
            topK: 5,
            threshold: 0.3,
            useMMR: true,
            mmrLambda: 0.5,
            includeMetadata: true
        };
    }

    async retrieveContext(query, options = {}) {
        const config = { ...this.defaultOptions, ...options };
        logger.log(`[RAG] Retrieving context for query: "${query.substring(0, 50)}..."`);
        logger.log(`[RAG] Config: topK=${config.topK}, threshold=${config.threshold}, maxTokens=${config.maxContextTokens}`);

        // Search for relevant chunks
        const searchResults = await vectorStore.search(query, {
            topK: config.topK * 2, // Get more candidates for filtering
            threshold: config.threshold,
            useMMR: config.useMMR,
            mmrLambda: config.mmrLambda
        });

        if (!searchResults.results || searchResults.results.length === 0) {
            logger.warn('[RAG] No relevant documents found for query!');
            return {
                contextText: '',
                sources: [],
                tokensUsed: 0,
                chunksUsed: 0
            };
        }

        logger.log(`[RAG] Found ${searchResults.results.length} potential chunks`);

        // Build context with token budget
        const contextChunks = [];
        const sources = new Map();
        let currentTokens = 0;

        for (const result of searchResults.results) {
            const chunk = result.chunk;
            const doc = result.document;

            // Approximate token count
            const chunkTokens = this.estimateTokens(chunk.text);

            if (currentTokens + chunkTokens > config.maxContextTokens) {
                break; // Token budget exceeded
            }

            contextChunks.push({
                text: chunk.text,
                score: result.score,
                position: chunk.position,
                docId: doc.id,
                docName: doc.name
            });

            // Track unique sources
            if (!sources.has(doc.id)) {
                sources.set(doc.id, {
                    id: doc.id,
                    name: doc.name,
                    type: doc.type,
                    chunks: []
                });
            }
            sources.get(doc.id).chunks.push({
                position: chunk.position,
                score: result.score
            });

            currentTokens += chunkTokens;

            if (contextChunks.length >= config.topK) {
                break; // Reached desired number of chunks
            }
        }

        // Format context text
        const contextText = this.formatContext(contextChunks, config.includeMetadata);

        logger.log(`[RAG] Built context with ${contextChunks.length} chunks from ${sources.size} documents`);
        logger.log(`[RAG] Context length: ${contextText.length} chars, ~${currentTokens} tokens`);

        // Convert sources map to array
        const sourcesList = Array.from(sources.values()).map(source => ({
            ...source,
            relevance: Math.max(...source.chunks.map(c => c.score))
        }));

        // Sort sources by relevance
        sourcesList.sort((a, b) => b.relevance - a.relevance);

        if (sourcesList.length > 0) {
            logger.log(`[RAG] Top source: ${sourcesList[0].name} (relevance: ${(sourcesList[0].relevance * 100).toFixed(1)}%)`);
        }

        return {
            contextText,
            sources: sourcesList,
            tokensUsed: currentTokens,
            chunksUsed: contextChunks.length,
            searchTime: searchResults.searchTime
        };
    }

    formatContext(chunks, includeMetadata = true) {
        if (chunks.length === 0) {
            return '';
        }

        let context = 'CONTEXT:\n';
        context += '========\n\n';

        // Group chunks by document
        const docGroups = new Map();
        chunks.forEach(chunk => {
            if (!docGroups.has(chunk.docId)) {
                docGroups.set(chunk.docId, []);
            }
            docGroups.get(chunk.docId).push(chunk);
        });

        // Format each document's chunks
        let docIndex = 1;
        for (const [, docChunks] of docGroups) {
            const docName = docChunks[0].docName;

            if (includeMetadata) {
                context += `[Document ${docIndex}: ${docName}]\n`;
            }

            // Sort chunks by position to maintain reading order
            docChunks.sort((a, b) => a.position - b.position);

            docChunks.forEach(chunk => {
                context += chunk.text;
                if (!chunk.text.endsWith('\n')) {
                    context += '\n';
                }
                context += '\n';
            });

            docIndex++;
        }

        context += '========\n';
        return context;
    }

    buildPromptWithContext(userQuery, context, systemPrompt = null) {
        const defaultSystemPrompt = `You are a helpful assistant with access to a knowledge base.
Use the provided context to answer questions accurately.
If the context doesn't contain relevant information, say so clearly.
Always cite which document(s) you're referencing when using the context.
Provide plain text responses suitable for text-to-speech conversion.
Use simple punctuation and clear sentence structure.`;

        const fullSystemPrompt = systemPrompt || defaultSystemPrompt;

        const messages = [
            {
                role: 'system',
                content: `${fullSystemPrompt}\n\n${context}`
            },
            {
                role: 'user',
                content: userQuery
            }
        ];

        return messages;
    }

    estimateTokens(text) {
        // Rough approximation: 1 token ≈ 4 characters or 0.75 words
        const charCount = text.length;
        const wordCount = text.split(/\s+/).length;

        // Use average of both estimates
        const charEstimate = charCount / 4;
        const wordEstimate = wordCount / 0.75;

        return Math.ceil((charEstimate + wordEstimate) / 2);
    }

    async processQuery(query, options = {}) {
        const startTime = performance.now();
        logger.log(`[RAG] Processing query: "${query}"`);

        // Retrieve relevant context
        const contextResult = await this.retrieveContext(query, options);

        // Build augmented prompt
        const messages = this.buildPromptWithContext(
            query,
            contextResult.contextText,
            options.systemPrompt
        );

        const processingTime = performance.now() - startTime;

        logger.log(`[RAG] Query processed in ${processingTime.toFixed(2)}ms`);
        logger.log(`[RAG] Context provided: ${contextResult.contextText ? 'Yes' : 'No'} (${contextResult.sources.length} sources)`);

        return {
            messages,
            context: contextResult.contextText,
            sources: contextResult.sources,
            metadata: {
                tokensUsed: contextResult.tokensUsed,
                chunksUsed: contextResult.chunksUsed,
                processingTime,
                searchTime: contextResult.searchTime
            }
        };
    }

    formatSources(sources) {
        if (!sources || sources.length === 0) {
            return '';
        }

        let formatted = '\n\nSources:\n';
        sources.forEach((source, index) => {
            formatted += `${index + 1}. ${source.name}`;
            if (source.relevance) {
                formatted += ` (relevance: ${(source.relevance * 100).toFixed(1)}%)`;
            }
            formatted += '\n';
        });

        return formatted;
    }

    async analyzeQuery(query) {
        // Simple query analysis to determine if RAG is needed
        const questionWords = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'whose'];
        const lowerQuery = query.toLowerCase();

        // Check for question indicators
        const isQuestion = questionWords.some(word =>
            lowerQuery.startsWith(word) || lowerQuery.includes(` ${word} `)
        ) || query.includes('?');

        // Check for specific information requests
        const needsSpecificInfo = [
            'tell me about',
            'explain',
            'describe',
            'what is',
            'what are',
            'define',
            'summarize',
            'list'
        ].some(phrase => lowerQuery.includes(phrase));

        // Check for document references
        const referencesDocuments = [
            'document',
            'file',
            'paper',
            'article',
            'according to',
            'based on'
        ].some(term => lowerQuery.includes(term));

        return {
            shouldUseRAG: isQuestion || needsSpecificInfo || referencesDocuments,
            queryType: isQuestion ? 'question' : 'statement',
            confidence: (isQuestion ? 0.4 : 0) +
                       (needsSpecificInfo ? 0.4 : 0) +
                       (referencesDocuments ? 0.2 : 0)
        };
    }
}

// Create singleton instance
const ragPipeline = new RAGPipeline();

export { ragPipeline, RAGPipeline };

export async function retrieveContext(query, options) {
    return ragPipeline.retrieveContext(query, options);
}

export async function processRAGQuery(query, options) {
    return ragPipeline.processQuery(query, options);
}

export function formatSources(sources) {
    return ragPipeline.formatSources(sources);
}

export async function analyzeQuery(query) {
    return ragPipeline.analyzeQuery(query);
}