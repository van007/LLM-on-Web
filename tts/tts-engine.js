import logger from '../utils/logger.js';

export class TTSEngine {
    constructor() {
        this.model = null;
        this.isLoading = false;
        this.device = null;
        this.dtype = null;
        this.abortController = null;
    }

    async initialize(progressCallback = null) {
        if (this.model || this.isLoading) return;

        this.isLoading = true;

        try {
            const { KokoroTTS, TextSplitterStream } = await import('https://cdn.jsdelivr.net/npm/kokoro-js@1.2.1/+esm');
            this.TextSplitterStream = TextSplitterStream;

            const hasWebGPU = 'gpu' in navigator && navigator.gpu;
            this.device = hasWebGPU ? 'webgpu' : 'wasm';
            this.dtype = hasWebGPU ? 'fp32' : 'q8';

            logger.log(`Initializing TTS with device: ${this.device}, dtype: ${this.dtype}`);

            const modelConfig = {
                dtype: this.dtype,
                device: this.device,
                progress_callback: progressCallback
            };

            this.model = await KokoroTTS.from_pretrained(
                'onnx-community/Kokoro-82M-ONNX',
                modelConfig
            );

            logger.log('TTS model loaded successfully');
        } catch (error) {
            logger.error('Failed to initialize TTS:', error);
            throw error;
        } finally {
            this.isLoading = false;
        }
    }

    estimateTokenCount(text) {
        // Approximate token count: 1 token ≈ 4 characters
        return Math.ceil(text.length / 4);
    }

    async generateSpeech(text, options = {}) {
        if (!this.model) {
            throw new Error('TTS model not initialized');
        }

        const tokenCount = this.estimateTokenCount(text);
        logger.log(`[TTS] Text length: ${text.length} chars, estimated tokens: ${tokenCount}`);

        // Use streaming for messages > 100 tokens
        if (tokenCount > 100) {
            logger.log('[TTS] Using streaming generation for long text');
            return this.generateStreamingSpeech(text, options);
        }

        const {
            voice = 'af_heart',
            speed = 1.0
        } = options;

        try {
            this.abortController = new AbortController();

            const generateOptions = {
                voice,
                speed,
                signal: this.abortController.signal
            };

            logger.log('[TTS] Using direct generation for short text');
            const audio = await this.model.generate(text, generateOptions);
            return [audio];
        } catch (error) {
            if (error.name === 'AbortError') {
                logger.log('TTS generation aborted');
                return null;
            }
            logger.error('TTS generation failed:', error);
            throw error;
        }
    }

    async generateStreamingSpeech(text, options = {}) {
        if (!this.model) {
            throw new Error('TTS model not initialized');
        }

        if (!this.TextSplitterStream) {
            throw new Error('TextSplitterStream not available');
        }

        const {
            voice = 'af_heart',
            speed = 1.0
        } = options;

        try {
            this.abortController = new AbortController();

            const generateOptions = {
                voice,
                speed,
                signal: this.abortController.signal
            };

            // Create text splitter and stream
            const splitter = new this.TextSplitterStream();
            const stream = this.model.stream(splitter, generateOptions);

            // Create async generator for audio chunks
            const audioGenerator = async function* () {
                // Add text to splitter immediately
                splitter.push(text);
                splitter.close();

                // Process the stream and yield audio chunks
                for await (const { text: chunkText, phonemes, audio } of stream) {
                    if (this.abortController && this.abortController.signal.aborted) {
                        logger.log('[TTS] Streaming generation aborted');
                        break;
                    }

                    logger.log(`[TTS] Generated streaming chunk: "${chunkText.substring(0, 50)}..."`);
                    // Yield the audio chunk immediately for streaming playback
                    yield audio;
                }
            }.bind(this);

            // Return the generator for streaming consumption
            return audioGenerator();

        } catch (error) {
            if (error.name === 'AbortError') {
                logger.log('TTS streaming generation aborted');
                return null;
            }
            logger.error('TTS streaming generation failed:', error);
            throw error;
        }
    }

    splitTextIntoChunks(text, maxChunkSize = 500) {
        // Split by sentence-ending punctuation while keeping the punctuation
        // This regex splits on .!? followed by space or end of string, but keeps the delimiter
        const sentenceRegex = /([.!?]+\s*)/;
        const parts = text.split(sentenceRegex).filter(part => part.trim());

        // Reconstruct sentences by combining text with its following punctuation
        const sentences = [];
        for (let i = 0; i < parts.length; i++) {
            if (i < parts.length - 1 && /^[.!?]+\s*$/.test(parts[i + 1])) {
                // Current part is text, next part is punctuation
                sentences.push(parts[i] + parts[i + 1]);
                i++; // Skip the punctuation part as we've already included it
            } else if (!/^[.!?]+\s*$/.test(parts[i])) {
                // Current part is text without following punctuation
                sentences.push(parts[i]);
            }
        }

        // If no sentences were found (no punctuation in text), treat entire text as one sentence
        if (sentences.length === 0) {
            sentences.push(text);
        }

        const chunks = [];
        let currentChunk = '';

        for (const sentence of sentences) {
            const trimmedSentence = sentence.trim();

            // If a single sentence is longer than max size, split it further
            if (trimmedSentence.length > maxChunkSize) {
                // Save current chunk if any
                if (currentChunk.trim()) {
                    chunks.push(currentChunk.trim());
                    currentChunk = '';
                }

                // Split long sentence by words
                const words = trimmedSentence.split(/\s+/);
                let tempChunk = '';

                for (const word of words) {
                    if ((tempChunk + ' ' + word).length > maxChunkSize && tempChunk) {
                        chunks.push(tempChunk.trim());
                        tempChunk = word;
                    } else {
                        tempChunk = tempChunk ? tempChunk + ' ' + word : word;
                    }
                }

                // Add remaining words as the next chunk
                if (tempChunk.trim()) {
                    currentChunk = tempChunk.trim();
                }
            } else if ((currentChunk + ' ' + trimmedSentence).length > maxChunkSize && currentChunk) {
                // Current chunk + new sentence exceeds limit
                chunks.push(currentChunk.trim());
                currentChunk = trimmedSentence;
            } else {
                // Add sentence to current chunk
                currentChunk = currentChunk ? currentChunk + ' ' + trimmedSentence : trimmedSentence;
            }
        }

        // Don't forget the last chunk
        if (currentChunk.trim()) {
            chunks.push(currentChunk.trim());
        }

        // Log chunk information for debugging
        logger.log(`[TTS] Text chunking: ${text.length} chars -> ${chunks.length} chunks`);
        chunks.forEach((chunk, i) => {
            logger.log(`[TTS] Chunk ${i + 1}: ${chunk.length} chars, starts with: "${chunk.substring(0, 30)}..."`);
        });

        return chunks;
    }

    abort() {
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
    }

    isReady() {
        return this.model !== null;
    }

    getDeviceInfo() {
        return {
            device: this.device,
            dtype: this.dtype
        };
    }
}

const ttsEngine = new TTSEngine();
export default ttsEngine;