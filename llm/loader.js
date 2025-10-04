import { pipeline, env, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3';
import logger from '../utils/logger.js';

class ModelLoader {
    constructor() {
        this.pipelines = {
            textGeneration: null,
            featureExtraction: null
        };
        this.tokenizers = {};
        this.progressCallbacks = [];
        this.isInitialized = false;
    }

    async initTransformers(options = {}) {
        const { backendPreference = 'auto' } = options;

        if (backendPreference === 'auto') {
            if (navigator.gpu) {
                logger.log('[ModelLoader] WebGPU detected, using WebGPU backend');
            } else {
                logger.log('[ModelLoader] WebGPU not available, falling back to WASM');
            }
        }

        env.allowRemoteModels = true;
        env.allowLocalModels = options.allowLocalModels ?? true;
        env.useBrowserCache = true;

        if (options.localModelPath) {
            env.localModelPath = options.localModelPath;
        }

        if (self.crossOriginIsolated) {
            logger.log('[ModelLoader] Cross-origin isolated, WASM threads enabled');
        } else {
            logger.log('[ModelLoader] Not cross-origin isolated, WASM threads disabled');
        }

        this.isInitialized = true;
        return true;
    }

    async loadLLM(modelId = 'onnx-community/Qwen2.5-0.5B-Instruct', options = {}) {
        if (!this.isInitialized) {
            await this.initTransformers(options);
        }

        const device = this.detectDevice(options.device);
        const dtype = options.dtype || 'q4';

        logger.log(`[ModelLoader] Loading LLM: ${modelId} with dtype: ${dtype} on device: ${device}`);

        const progressCallback = this.createProgressCallback(modelId, 'llm');

        try {
            this.pipelines.textGeneration = await pipeline(
                'text-generation',
                modelId,
                {
                    dtype,
                    device,
                    progress_callback: progressCallback
                }
            );

            this.tokenizers[modelId] = await AutoTokenizer.from_pretrained(modelId);

            logger.log('[ModelLoader] LLM loaded successfully');
            this.notifyProgress({ status: 'ready', name: modelId, type: 'llm' });

            return this.pipelines.textGeneration;
        } catch (error) {
            logger.error('[ModelLoader] Failed to load LLM:', error);
            this.notifyProgress({
                status: 'error',
                name: modelId,
                type: 'llm',
                error: error.message
            });
            throw error;
        }
    }

    async loadEmbedder(modelId = 'Xenova/all-MiniLM-L6-v2', options = {}) {
        if (!this.isInitialized) {
            await this.initTransformers(options);
        }

        const device = this.detectDevice(options.device);
        const dtype = options.dtype || 'fp32';

        logger.log(`[ModelLoader] Loading embedder: ${modelId} with dtype: ${dtype} on device: ${device}`);

        const progressCallback = this.createProgressCallback(modelId, 'embedder');

        try {
            this.pipelines.featureExtraction = await pipeline(
                'feature-extraction',
                modelId,
                {
                    dtype,
                    device,
                    progress_callback: progressCallback
                }
            );

            logger.log('[ModelLoader] Embedder loaded successfully');
            this.notifyProgress({ status: 'ready', name: modelId, type: 'embedder' });

            return this.pipelines.featureExtraction;
        } catch (error) {
            logger.error('[ModelLoader] Failed to load embedder:', error);
            this.notifyProgress({
                status: 'error',
                name: modelId,
                type: 'embedder',
                error: error.message
            });
            throw error;
        }
    }

    detectDevice(preferredDevice) {
        if (preferredDevice && ['webgpu', 'wasm'].includes(preferredDevice)) {
            return preferredDevice;
        }

        if (navigator.gpu) {
            return 'webgpu';
        }

        return 'wasm';
    }

    createProgressCallback(modelId, modelType) {
        const progressTracker = {
            totalFiles: 0,
            loadedFiles: 0,
            totalBytes: 0,
            loadedBytes: 0,
            files: {}
        };

        return (progress) => {
            const { status, file, progress: progressValue, loaded, total } = progress;

            if (status === 'initiate') {
                progressTracker.totalFiles++;
                progressTracker.files[file] = { loaded: 0, total: 0 };
                this.notifyProgress({
                    status: 'initiate',
                    name: modelId,
                    type: modelType,
                    file,
                    totalFiles: progressTracker.totalFiles
                });
            } else if (status === 'download') {
                this.notifyProgress({
                    status: 'download',
                    name: modelId,
                    type: modelType,
                    file
                });
            } else if (status === 'progress') {
                if (file && progressTracker.files[file]) {
                    progressTracker.files[file].loaded = loaded || 0;
                    progressTracker.files[file].total = total || 0;
                }

                progressTracker.loadedBytes = Object.values(progressTracker.files)
                    .reduce((sum, f) => sum + f.loaded, 0);
                progressTracker.totalBytes = Object.values(progressTracker.files)
                    .reduce((sum, f) => sum + f.total, 0);

                const overallProgress = progressTracker.totalBytes > 0
                    ? (progressTracker.loadedBytes / progressTracker.totalBytes) * 100
                    : 0;

                this.notifyProgress({
                    status: 'progress',
                    name: modelId,
                    type: modelType,
                    file,
                    progress: overallProgress,
                    loaded: progressTracker.loadedBytes,
                    total: progressTracker.totalBytes,
                    fileProgress: progressValue
                });
            } else if (status === 'done') {
                progressTracker.loadedFiles++;
                this.notifyProgress({
                    status: 'done',
                    name: modelId,
                    type: modelType,
                    file,
                    loadedFiles: progressTracker.loadedFiles,
                    totalFiles: progressTracker.totalFiles
                });
            } else if (status === 'ready') {
                this.notifyProgress({
                    status: 'ready',
                    name: modelId,
                    type: modelType
                });
            }
        };
    }

    onProgress(callback) {
        this.progressCallbacks.push(callback);
        return () => {
            const index = this.progressCallbacks.indexOf(callback);
            if (index > -1) {
                this.progressCallbacks.splice(index, 1);
            }
        };
    }

    notifyProgress(event) {
        this.progressCallbacks.forEach(callback => {
            try {
                callback(event);
            } catch (error) {
                logger.error('[ModelLoader] Progress callback error:', error);
            }
        });
    }

    getPipelines() {
        return this.pipelines;
    }

    getTextGenerationPipeline() {
        return this.pipelines.textGeneration;
    }

    getFeatureExtractionPipeline() {
        return this.pipelines.featureExtraction;
    }

    getTokenizer(modelId) {
        return this.tokenizers[modelId];
    }

    async warmup(pipeline, prompt = 'Hello') {
        if (!pipeline) {
            throw new Error('Pipeline not loaded');
        }

        logger.log('[ModelLoader] Warming up model...');

        try {
            if (pipeline.task === 'text-generation') {
                await pipeline([{ role: 'user', content: prompt }], {
                    max_new_tokens: 1
                });
            } else if (pipeline.task === 'feature-extraction') {
                await pipeline(prompt, {
                    pooling: 'mean',
                    normalize: true
                });
            }

            logger.log('[ModelLoader] Warmup complete');
            return true;
        } catch (error) {
            logger.error('[ModelLoader] Warmup failed:', error);
            return false;
        }
    }

    async clearCache() {
        try {
            const databases = await indexedDB.databases();
            for (const db of databases) {
                if (db.name && db.name.includes('transformers')) {
                    indexedDB.deleteDatabase(db.name);
                    logger.log(`[ModelLoader] Deleted database: ${db.name}`);
                }
            }
            return true;
        } catch (error) {
            logger.error('[ModelLoader] Failed to clear cache:', error);
            return false;
        }
    }

    dispose() {
        this.pipelines.textGeneration = null;
        this.pipelines.featureExtraction = null;
        this.tokenizers = {};
        this.progressCallbacks = [];
        this.isInitialized = false;
    }
}

const modelLoader = new ModelLoader();

export {
    modelLoader,
    ModelLoader
};

export async function initTransformers(options) {
    return modelLoader.initTransformers(options);
}

export async function loadLLM(modelId, options) {
    return modelLoader.loadLLM(modelId, options);
}

export async function loadEmbedder(modelId, options) {
    return modelLoader.loadEmbedder(modelId, options);
}

export function getPipelines() {
    return modelLoader.getPipelines();
}

export function onProgress(callback) {
    return modelLoader.onProgress(callback);
}