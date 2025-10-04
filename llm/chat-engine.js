import { TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3';
import { modelLoader } from './loader.js';
import logger from '../utils/logger.js';

class ChatEngine {
    constructor() {
        this.abortController = null;
        this.ownAbortController = false;
        this.isGenerating = false;
        this.stopSequences = ['</s>', '\n\nUser:', '\n\nHuman:', '[END]'];
        this.defaultSystemPrompt = 'You are a helpful, factual assistant running locally in the user\'s browser. Provide plain text responses suitable for text-to-speech conversion. Use simple punctuation and clear sentence structure.';
        this.metrics = {
            tokensGenerated: 0,
            generationTime: 0,
            timeToFirstToken: 0
        };
    }

    async generateStream(options = {}) {
        const {
            messages,
            params = {},
            onToken,
            onDone,
            onError,
            signal,
            systemPrompt = this.defaultSystemPrompt
        } = options;

        if (this.isGenerating) {
            throw new Error('Generation already in progress');
        }

        const pipeline = modelLoader.getTextGenerationPipeline();
        if (!pipeline) {
            throw new Error('Text generation pipeline not loaded');
        }

        this.isGenerating = true;
        // Track whether we created the AbortController or it was provided externally
        this.ownAbortController = !signal;
        if (signal) {
            // External signal provided, wrap it
            this.abortController = { signal };
        } else {
            // Create our own AbortController
            this.abortController = new AbortController();
        }

        const startTime = performance.now();
        let firstTokenTime = null;
        let generatedText = '';
        let tokenCount = 0;

        const formattedMessages = this.formatMessages(messages, systemPrompt);

        const generationParams = {
            max_new_tokens: params.maxNewTokens || 256,
            temperature: params.temperature ?? 1.0,
            top_p: params.topP ?? 0.9,
            do_sample: params.temperature > 0,
            repetition_penalty: params.repetitionPenalty || 1.1,
            pad_token_id: pipeline.tokenizer.pad_token_id,
            eos_token_id: pipeline.tokenizer.eos_token_id
        };

        const maxTime = params.maxTime || 120000;
        const timeoutId = setTimeout(() => {
            if (this.isGenerating) {
                this.stop();
                if (onError) {
                    onError(new Error('Generation timeout'));
                }
            }
        }, maxTime);

        try {
            const streamer = new TextStreamer(pipeline.tokenizer, {
                skip_prompt: true,
                skip_special_tokens: true,
                callback_function: (text) => {
                    if (this.abortController?.signal?.aborted) {
                        return;
                    }

                    if (!firstTokenTime) {
                        firstTokenTime = performance.now();
                        this.metrics.timeToFirstToken = firstTokenTime - startTime;
                    }

                    generatedText += text;
                    tokenCount++;

                    if (this.shouldStop(generatedText)) {
                        generatedText = this.trimStopSequence(generatedText);
                        this.stop();
                        return;
                    }

                    if (generatedText.length > (params.maxLength || 4096)) {
                        this.stop();
                        return;
                    }

                    if (onToken) {
                        onToken(text, {
                            totalTokens: tokenCount,
                            text: generatedText,
                            timeElapsed: performance.now() - startTime
                        });
                    }
                }
            });

            generationParams.streamer = streamer;

            const result = await pipeline(formattedMessages, generationParams);

            clearTimeout(timeoutId);

            const endTime = performance.now();
            this.metrics.generationTime = endTime - startTime;
            this.metrics.tokensGenerated = tokenCount;

            const finalText = result[0].generated_text.at(-1).content;

            if (onDone) {
                onDone({
                    text: finalText,
                    tokens: tokenCount,
                    time: this.metrics.generationTime,
                    timeToFirstToken: this.metrics.timeToFirstToken,
                    tokensPerSecond: tokenCount / (this.metrics.generationTime / 1000)
                });
            }

            return finalText;

        } catch (error) {
            clearTimeout(timeoutId);

            if (error.name === 'AbortError' || this.abortController?.signal?.aborted) {
                logger.log('[ChatEngine] Generation aborted');
                if (onDone) {
                    onDone({
                        text: generatedText,
                        tokens: tokenCount,
                        time: performance.now() - startTime,
                        aborted: true
                    });
                }
                return generatedText;
            }

            logger.error('[ChatEngine] Generation error:', error);
            if (onError) {
                onError(error);
            }
            throw error;

        } finally {
            this.isGenerating = false;
            this.abortController = null;
            this.ownAbortController = false;
        }
    }

    async generate(messages, params = {}) {
        return new Promise((resolve, reject) => {
            let result = '';

            this.generateStream({
                messages,
                params,
                onToken: (token) => {
                    result += token;
                },
                onDone: (stats) => {
                    resolve({
                        text: stats.text || result,
                        ...stats
                    });
                },
                onError: reject
            });
        });
    }

    formatMessages(messages, systemPrompt) {
        const formattedMessages = [];

        if (systemPrompt) {
            formattedMessages.push({
                role: 'system',
                content: systemPrompt
            });
        }

        if (Array.isArray(messages)) {
            formattedMessages.push(...messages.map(msg => ({
                role: msg.role || 'user',
                content: msg.content
            })));
        } else if (typeof messages === 'string') {
            formattedMessages.push({
                role: 'user',
                content: messages
            });
        } else if (messages && typeof messages === 'object') {
            formattedMessages.push(messages);
        }

        return formattedMessages;
    }

    shouldStop(text) {
        return this.stopSequences.some(seq => text.includes(seq));
    }

    trimStopSequence(text) {
        for (const seq of this.stopSequences) {
            const index = text.indexOf(seq);
            if (index !== -1) {
                return text.substring(0, index);
            }
        }
        return text;
    }

    stop() {
        // Only abort if we own the controller and it has an abort method
        if (this.ownAbortController && this.abortController && typeof this.abortController.abort === 'function') {
            if (!this.abortController.signal.aborted) {
                this.abortController.abort();
            }
        }
        this.isGenerating = false;
        this.ownAbortController = false;
    }

    isActive() {
        return this.isGenerating;
    }

    getMetrics() {
        return { ...this.metrics };
    }

    setSystemPrompt(prompt) {
        this.defaultSystemPrompt = prompt;
    }

    setStopSequences(sequences) {
        this.stopSequences = sequences;
    }

    addStopSequence(sequence) {
        if (!this.stopSequences.includes(sequence)) {
            this.stopSequences.push(sequence);
        }
    }

    removeStopSequence(sequence) {
        const index = this.stopSequences.indexOf(sequence);
        if (index > -1) {
            this.stopSequences.splice(index, 1);
        }
    }
}

const chatEngine = new ChatEngine();

export {
    chatEngine,
    ChatEngine
};

export async function generateStream(options) {
    return chatEngine.generateStream(options);
}

export async function generate(messages, params) {
    return chatEngine.generate(messages, params);
}

export function stopGeneration() {
    return chatEngine.stop();
}

export function isGenerating() {
    return chatEngine.isActive();
}

export function getChatMetrics() {
    return chatEngine.getMetrics();
}