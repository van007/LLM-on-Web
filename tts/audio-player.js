import logger from '../utils/logger.js';

export class AudioPlayer {
    constructor() {
        this.audioContext = null;
        this.currentSource = null;
        this.audioQueue = [];
        this.isPlaying = false;
        this.isPaused = false;
        this.startTime = 0;
        this.pauseTime = 0;
        this.currentBuffer = null;
        this.onEndCallback = null;
        this.onProgressCallback = null;
        this.progressInterval = null;
        this.playbackRate = 1.0;
        this.volume = 1.0;
        this.gainNode = null;
        // Cumulative tracking for chunked audio
        this.totalDuration = 0;
        this.completedDuration = 0;
        this.chunkDurations = [];
        this.currentChunkIndex = 0;
        // Streaming mode flag
        this.isStreaming = false;
    }

    async initialize() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.gainNode = this.audioContext.createGain();
            this.gainNode.connect(this.audioContext.destination);
            this.gainNode.gain.value = this.volume;
        }

        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    async initializeStreamingPlayback(audioGenerator) {
        // Initialize audio context first
        await this.initialize();

        // Reset cumulative tracking
        this.totalDuration = 0;
        this.completedDuration = 0;
        this.chunkDurations = [];
        this.currentChunkIndex = 0;
        this.isStreaming = true;

        logger.log('[AudioPlayer] Starting streaming playback');

        try {
            let chunkIndex = 0;

            // Consume the audio stream
            for await (const audioData of audioGenerator) {
                if (this.abortController && this.abortController.signal.aborted) {
                    logger.log('[AudioPlayer] Streaming aborted');
                    break;
                }

                logger.log(`[AudioPlayer] Received streaming chunk ${chunkIndex + 1}`);

                // Calculate and store chunk duration
                let duration = 0;
                try {
                    if (audioData.toWav) {
                        const wavArrayBuffer = audioData.toWav();
                        const audioBuffer = await this.audioContext.decodeAudioData(wavArrayBuffer.slice(0));
                        duration = audioBuffer.duration;
                    } else if (audioData instanceof Float32Array) {
                        duration = audioData.length / this.audioContext.sampleRate;
                    } else if (audioData instanceof ArrayBuffer) {
                        const audioBuffer = await this.audioContext.decodeAudioData(audioData.slice(0));
                        duration = audioBuffer.duration;
                    }
                } catch (error) {
                    logger.error('Failed to calculate chunk duration:', error);
                }

                this.chunkDurations.push(duration);
                this.totalDuration += duration;

                // Add to queue
                this.audioQueue.push(audioData);

                // Start playing first chunk immediately
                if (!this.isPlaying && this.audioQueue.length === 1) {
                    logger.log('[AudioPlayer] Starting playback of first streaming chunk');
                    await this.playNext();
                }

                chunkIndex++;
            }

            logger.log(`[AudioPlayer] Streaming complete: ${chunkIndex} chunks, total duration: ${this.totalDuration.toFixed(2)}s`);
        } catch (error) {
            logger.error('[AudioPlayer] Streaming playback error:', error);
            throw error;
        } finally {
            this.isStreaming = false;
        }
    }

    async initializeChunkedPlayback(audioChunks) {
        // Initialize audio context first
        await this.initialize();

        // Reset cumulative tracking
        this.totalDuration = 0;
        this.completedDuration = 0;
        this.chunkDurations = [];
        this.currentChunkIndex = 0;

        // Calculate duration for each chunk
        for (const audioData of audioChunks) {
            try {
                let duration = 0;

                if (audioData.toWav) {
                    const wavArrayBuffer = audioData.toWav();
                    const audioBuffer = await this.audioContext.decodeAudioData(wavArrayBuffer.slice(0));
                    duration = audioBuffer.duration;
                } else if (audioData instanceof Float32Array) {
                    duration = audioData.length / this.audioContext.sampleRate;
                } else if (audioData instanceof ArrayBuffer) {
                    const audioBuffer = await this.audioContext.decodeAudioData(audioData.slice(0));
                    duration = audioBuffer.duration;
                }

                this.chunkDurations.push(duration);
                this.totalDuration += duration;
            } catch (error) {
                logger.error('Failed to calculate chunk duration:', error);
                this.chunkDurations.push(0);
            }
        }

        logger.log(`[AudioPlayer] Initialized chunked playback: ${audioChunks.length} chunks, total duration: ${this.totalDuration.toFixed(2)}s`);

        // Add all chunks to queue
        for (const audioData of audioChunks) {
            this.audioQueue.push(audioData);
        }

        // Start playing the first chunk
        if (this.audioQueue.length > 0) {
            await this.playNext();
        }
    }

    async playAudioData(audioData) {
        await this.initialize();

        try {
            let audioBuffer;

            if (audioData.toWav) {
                const wavArrayBuffer = audioData.toWav();
                audioBuffer = await this.audioContext.decodeAudioData(wavArrayBuffer);
            } else if (audioData instanceof Float32Array) {
                audioBuffer = this.audioContext.createBuffer(
                    1,
                    audioData.length,
                    this.audioContext.sampleRate
                );
                audioBuffer.copyToChannel(audioData, 0);
            } else if (audioData instanceof ArrayBuffer) {
                audioBuffer = await this.audioContext.decodeAudioData(audioData);
            } else {
                throw new Error('Unsupported audio data format');
            }

            this.currentBuffer = audioBuffer;
            await this.playBuffer(audioBuffer);
        } catch (error) {
            logger.error('Failed to play audio:', error);
            throw error;
        }
    }

    async playBuffer(buffer) {
        if (this.currentSource) {
            this.stop();
        }

        this.currentSource = this.audioContext.createBufferSource();
        this.currentSource.buffer = buffer;
        this.currentSource.playbackRate.value = this.playbackRate;
        this.currentSource.connect(this.gainNode);

        this.currentSource.onended = () => {
            this.handlePlaybackEnd();
        };

        this.startTime = this.audioContext.currentTime;
        this.currentSource.start(0);
        this.isPlaying = true;
        this.isPaused = false;

        this.startProgressTracking();
    }

    addToQueue(audioData) {
        this.audioQueue.push(audioData);

        if (!this.isPlaying && this.audioQueue.length === 1) {
            this.playNext();
        }
    }

    async playNext() {
        if (this.audioQueue.length === 0) {
            this.isPlaying = false;
            return;
        }

        const nextAudio = this.audioQueue.shift();

        // If we're in chunked playback mode, use special handling
        if (this.totalDuration > 0) {
            await this.playNextChunk(nextAudio);
        } else {
            await this.playAudioData(nextAudio);
        }
    }

    async playNextChunk(audioData) {
        await this.initialize();

        try {
            let audioBuffer;

            if (audioData.toWav) {
                const wavArrayBuffer = audioData.toWav();
                audioBuffer = await this.audioContext.decodeAudioData(wavArrayBuffer);
            } else if (audioData instanceof Float32Array) {
                audioBuffer = this.audioContext.createBuffer(
                    1,
                    audioData.length,
                    this.audioContext.sampleRate
                );
                audioBuffer.copyToChannel(audioData, 0);
            } else if (audioData instanceof ArrayBuffer) {
                audioBuffer = await this.audioContext.decodeAudioData(audioData);
            } else {
                throw new Error('Unsupported audio data format');
            }

            this.currentBuffer = audioBuffer;

            // Play the buffer while maintaining chunked playback state
            if (this.currentSource) {
                this.currentSource.disconnect();
                this.currentSource.stop();
                this.currentSource = null;
            }

            this.currentSource = this.audioContext.createBufferSource();
            this.currentSource.buffer = audioBuffer;
            this.currentSource.playbackRate.value = this.playbackRate;
            this.currentSource.connect(this.gainNode);

            this.currentSource.onended = () => {
                this.handlePlaybackEnd();
            };

            this.startTime = this.audioContext.currentTime;
            this.currentSource.start(0);
            this.isPlaying = true;
            this.isPaused = false;

            this.startProgressTracking();
        } catch (error) {
            logger.error('Failed to play chunk:', error);
            throw error;
        }
    }

    pause() {
        if (this.isPlaying && !this.isPaused) {
            this.pauseTime = this.audioContext.currentTime;

            // Disconnect without stopping to avoid triggering onended
            if (this.currentSource) {
                this.currentSource.onended = null; // Remove the callback
                this.currentSource.disconnect();
                this.currentSource.stop();
            }

            this.isPaused = true;
            this.isPlaying = false;
            this.stopProgressTracking();
        }
    }

    async resume() {
        if (this.isPaused && this.currentBuffer) {
            await this.initialize();

            const elapsed = this.pauseTime - this.startTime;
            const offset = elapsed * this.playbackRate;

            this.currentSource = this.audioContext.createBufferSource();
            this.currentSource.buffer = this.currentBuffer;
            this.currentSource.playbackRate.value = this.playbackRate;
            this.currentSource.connect(this.gainNode);

            this.currentSource.onended = () => {
                this.handlePlaybackEnd();
            };

            this.startTime = this.audioContext.currentTime - elapsed;
            this.currentSource.start(0, offset);

            this.isPlaying = true;
            this.isPaused = false;
            this.startProgressTracking();
        }
    }

    stop() {
        if (this.currentSource) {
            this.currentSource.disconnect();
            this.currentSource.stop();
            this.currentSource = null;
        }

        this.isPlaying = false;
        this.isPaused = false;
        this.isStreaming = false;
        this.audioQueue = [];
        this.currentBuffer = null;
        this.stopProgressTracking();

        // Reset cumulative tracking
        this.totalDuration = 0;
        this.completedDuration = 0;
        this.chunkDurations = [];
        this.currentChunkIndex = 0;

        if (this.onEndCallback) {
            this.onEndCallback();
        }
    }

    handlePlaybackEnd() {
        this.isPlaying = false;
        this.stopProgressTracking();

        // Update completed duration for chunked playback
        if (this.totalDuration > 0 && this.currentChunkIndex < this.chunkDurations.length) {
            this.completedDuration += this.chunkDurations[this.currentChunkIndex];
            this.currentChunkIndex++;
        }

        if (this.audioQueue.length > 0) {
            this.playNext();
        } else {
            // Reset cumulative tracking when all chunks are done
            this.totalDuration = 0;
            this.completedDuration = 0;
            this.chunkDurations = [];
            this.currentChunkIndex = 0;

            if (this.onEndCallback) {
                this.onEndCallback();
            }
        }
    }

    startProgressTracking() {
        this.stopProgressTracking();

        if (this.onProgressCallback && this.currentBuffer) {
            this.progressInterval = setInterval(() => {
                const currentChunkElapsed = (this.audioContext.currentTime - this.startTime) * this.playbackRate;

                // If we're tracking multiple chunks, calculate cumulative progress
                if (this.totalDuration > 0) {
                    const totalElapsed = this.completedDuration + currentChunkElapsed;
                    const progress = Math.min(totalElapsed / this.totalDuration, 1);

                    // Pass cumulative values to callback
                    this.onProgressCallback(progress, totalElapsed, this.totalDuration);
                } else {
                    // Single chunk or no chunked playback initialized
                    const duration = this.currentBuffer.duration;
                    const progress = Math.min(currentChunkElapsed / duration, 1);
                    this.onProgressCallback(progress, currentChunkElapsed, duration);
                }

                if (currentChunkElapsed >= this.currentBuffer.duration) {
                    this.stopProgressTracking();
                }
            }, 100);
        }
    }

    stopProgressTracking() {
        if (this.progressInterval) {
            clearInterval(this.progressInterval);
            this.progressInterval = null;
        }
    }

    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        if (this.gainNode) {
            this.gainNode.gain.value = this.volume;
        }
    }

    setPlaybackRate(rate) {
        this.playbackRate = Math.max(0.5, Math.min(2, rate));
        if (this.currentSource) {
            this.currentSource.playbackRate.value = this.playbackRate;
        }
    }

    onEnd(callback) {
        this.onEndCallback = callback;
    }

    onProgress(callback) {
        this.onProgressCallback = callback;
    }

    getState() {
        return {
            isPlaying: this.isPlaying,
            isPaused: this.isPaused,
            hasQueue: this.audioQueue.length > 0,
            isStreaming: this.isStreaming,
            volume: this.volume,
            playbackRate: this.playbackRate
        };
    }

    destroy() {
        this.stop();
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close();
        }
        this.audioContext = null;
        this.gainNode = null;
    }
}

export default AudioPlayer;