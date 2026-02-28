import ttsEngine from './tts-engine.js';
import AudioPlayer from './audio-player.js';
import logger from '../utils/logger.js';

class TTSUI {
    constructor() {
        this.activePlayers = new Map();
        this.activeButtons = new Map();
        this.audioData = new Map();
        this.downloadAudioBtns = new Map();
        this.isModelLoading = false;
    }

    createTTSButton(messageContent, messageId) {
        const container = document.createElement('div');
        container.className = 'tts-controls';

        // Button row: [Download Chat] [TTS Play/Pause] [Download Audio]
        const buttonRow = document.createElement('div');
        buttonRow.className = 'tts-button-row';

        // Download Chat button
        const downloadChatBtn = document.createElement('button');
        downloadChatBtn.className = 'tts-action-btn download-chat-btn';
        downloadChatBtn.setAttribute('aria-label', 'Download message as markdown');
        downloadChatBtn.title = 'Download message';
        downloadChatBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
            <line x1="16" y1="13" x2="8" y2="13"/>
            <line x1="16" y1="17" x2="8" y2="17"/>
            <polyline points="10 9 9 9 8 9"/>
        </svg>`;
        downloadChatBtn.addEventListener('click', () => {
            this.handleDownloadChat(messageContent, messageId);
        });

        // TTS Play/Pause button (existing)
        const button = document.createElement('button');
        button.className = 'tts-button';
        button.setAttribute('aria-label', 'Play text-to-speech');
        button.dataset.messageId = messageId;

        const playIcon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z"/>
        </svg>`;

        button.innerHTML = playIcon;

        // Download Audio button (hidden initially)
        const downloadAudioBtn = document.createElement('button');
        downloadAudioBtn.className = 'tts-action-btn download-audio-btn';
        downloadAudioBtn.setAttribute('aria-label', 'Download audio as WAV');
        downloadAudioBtn.title = 'Download audio';
        downloadAudioBtn.style.display = 'none';
        downloadAudioBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>`;
        downloadAudioBtn.addEventListener('click', () => {
            this.handleDownloadAudio(messageId);
        });

        this.downloadAudioBtns.set(messageId, downloadAudioBtn);

        buttonRow.appendChild(downloadChatBtn);
        buttonRow.appendChild(button);
        buttonRow.appendChild(downloadAudioBtn);

        // Create progress container
        const progressContainer = document.createElement('div');
        progressContainer.className = 'tts-progress-container';

        const progressBar = document.createElement('div');
        progressBar.className = 'tts-progress-bar';
        const progressFill = document.createElement('div');
        progressFill.className = 'tts-progress-fill';
        progressBar.appendChild(progressFill);

        const progressText = document.createElement('span');
        progressText.className = 'tts-progress-text';
        progressText.textContent = '0:00';

        progressContainer.appendChild(progressBar);
        progressContainer.appendChild(progressText);

        container.appendChild(buttonRow);
        container.appendChild(progressContainer);

        button.addEventListener('click', async () => {
            await this.handleButtonClick(button, messageContent, messageId, progressFill, progressText);
        });

        return container;
    }

    handleDownloadChat(messageContent, messageId) {
        const blob = new Blob([messageContent], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `message-${messageId}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    handleDownloadAudio(messageId) {
        const data = this.audioData.get(messageId);
        if (!data || !data.complete || data.wavBuffers.length === 0) return;

        let wavBuffer;
        if (data.wavBuffers.length === 1) {
            wavBuffer = data.wavBuffers[0];
        } else {
            wavBuffer = combineWavBuffers(data.wavBuffers);
        }

        const blob = new Blob([wavBuffer], { type: 'audio/wav' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `audio-${messageId}.wav`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    showDownloadAudioButton(messageId) {
        const btn = this.downloadAudioBtns.get(messageId);
        if (btn) {
            btn.style.display = 'inline-flex';
        }
    }

    async handleButtonClick(button, messageContent, messageId, progressFill, progressText) {
        const existingPlayer = this.activePlayers.get(messageId);

        if (existingPlayer) {
            const state = existingPlayer.getState();

            if (state.isPlaying) {
                existingPlayer.pause();
                this.updateButtonIcon(button, 'play');
            } else if (state.isPaused) {
                await existingPlayer.resume();
                this.updateButtonIcon(button, 'pause');
            } else {
                await this.startTTS(button, messageContent, messageId, progressFill, progressText);
            }
        } else {
            await this.startTTS(button, messageContent, messageId, progressFill, progressText);
        }
    }

    async startTTS(button, messageContent, messageId, progressFill, progressText) {
        try {
            const textContent = this.prepareTextForTTS(messageContent);

            // Clear previous audio data and hide download button for fresh generation
            this.audioData.set(messageId, { wavBuffers: [], complete: false });
            const downloadBtn = this.downloadAudioBtns.get(messageId);
            if (downloadBtn) {
                downloadBtn.style.display = 'none';
            }

            if (!ttsEngine.isReady()) {
                this.updateButtonIcon(button, 'loading');

                const progressHandler = (progress) => {
                    if (progress.status === 'progress' && progress.file) {
                        const percent = Math.round((progress.loaded / progress.total) * 100);
                        button.title = `Loading model: ${percent}%`;
                    }
                };

                await ttsEngine.initialize(progressHandler);
                button.title = '';
            }

            this.updateButtonIcon(button, 'loading');

            this.stopAllOtherPlayers(messageId);

            const player = new AudioPlayer();
            this.activePlayers.set(messageId, player);
            this.activeButtons.set(messageId, button);

            player.onProgress((progress, elapsed, duration) => {
                progressFill.style.width = `${progress * 100}%`;
                const elapsedFormatted = this.formatTime(elapsed);
                const durationFormatted = this.formatTime(duration);

                // Show streaming indicator if still streaming
                const state = player.getState();
                if (state.isStreaming) {
                    progressText.textContent = `${elapsedFormatted} (streaming...)`;
                    button.title = `${elapsedFormatted} (generating more audio...)`;
                } else {
                    progressText.textContent = `${elapsedFormatted}`;
                    button.title = `${elapsedFormatted} / ${durationFormatted}`;
                }
            });

            player.onEnd(() => {
                this.updateButtonIcon(button, 'play');
                progressFill.style.width = '0%';
                progressText.textContent = '0:00';
                button.title = '';
                this.activePlayers.delete(messageId);
                this.activeButtons.delete(messageId);
            });

            // Generate speech - returns either array or async generator
            const result = await ttsEngine.generateSpeech(textContent);

            if (!result) {
                throw new Error('No audio generated');
            }

            // Check if result is an async generator (streaming mode)
            if (result && typeof result[Symbol.asyncIterator] === 'function') {
                logger.log('[TTS UI] Using streaming playback mode');
                this.updateButtonIcon(button, 'pause');

                // Wrap the generator to intercept audio chunks for download
                const audioDataEntry = this.audioData.get(messageId);
                const self = this;
                const wrappedGenerator = async function* () {
                    try {
                        for await (const audioChunk of result) {
                            // Capture WAV data for download
                            try {
                                if (audioChunk.toWav) {
                                    const wavBuffer = audioChunk.toWav();
                                    audioDataEntry.wavBuffers.push(wavBuffer);
                                }
                            } catch (err) {
                                logger.error('[TTS UI] Failed to capture WAV chunk:', err);
                            }
                            yield audioChunk;
                        }
                        // Generator exhausted — all chunks captured
                        audioDataEntry.complete = true;
                        self.showDownloadAudioButton(messageId);
                    } catch (err) {
                        // If generator breaks (abort), don't mark complete
                        logger.log('[TTS UI] Streaming generator interrupted:', err);
                    }
                }();

                await player.initializeStreamingPlayback(wrappedGenerator);
            } else if (Array.isArray(result)) {
                // Handle array of audio chunks (non-streaming mode)
                if (result.length === 0) {
                    throw new Error('No audio chunks generated');
                }

                // Capture WAV data from all chunks
                const audioDataEntry = this.audioData.get(messageId);
                for (const chunk of result) {
                    try {
                        if (chunk.toWav) {
                            const wavBuffer = chunk.toWav();
                            audioDataEntry.wavBuffers.push(wavBuffer);
                        }
                    } catch (err) {
                        logger.error('[TTS UI] Failed to capture WAV chunk:', err);
                    }
                }
                audioDataEntry.complete = true;
                this.showDownloadAudioButton(messageId);

                if (result.length === 1) {
                    logger.log('[TTS UI] Playing single audio chunk');
                    await player.playAudioData(result[0]);
                } else {
                    // Multiple chunks - use chunked playback with cumulative progress
                    logger.log(`[TTS UI] Playing ${result.length} audio chunks with cumulative progress`);
                    await player.initializeChunkedPlayback(result);
                }

                this.updateButtonIcon(button, 'pause');
            } else {
                throw new Error('Unexpected result from generateSpeech');
            }

        } catch (error) {
            logger.error('TTS failed:', error);
            this.updateButtonIcon(button, 'play');
            button.title = 'TTS failed. Click to retry.';
            progressFill.style.width = '0%';
            progressText.textContent = '0:00';

            const player = this.activePlayers.get(messageId);
            if (player) {
                player.destroy();
                this.activePlayers.delete(messageId);
            }
            this.activeButtons.delete(messageId);
        }
    }

    stopAllOtherPlayers(exceptMessageId = null) {
        for (const [messageId, player] of this.activePlayers.entries()) {
            if (messageId !== exceptMessageId) {
                player.stop();
                const button = this.activeButtons.get(messageId);
                if (button) {
                    this.updateButtonIcon(button, 'play');
                }
                this.activePlayers.delete(messageId);
                this.activeButtons.delete(messageId);
            }
        }
    }

    updateButtonIcon(button, state) {
        const playIcon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z"/>
        </svg>`;

        const pauseIcon = `<svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
        </svg>`;

        const loadingIcon = `<svg class="tts-loading-spinner" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10" stroke-dasharray="32" stroke-dashoffset="32">
                <animate attributeName="stroke-dashoffset" dur="1.5s" repeatCount="indefinite" from="32" to="0"/>
            </circle>
        </svg>`;

        switch (state) {
            case 'play':
                button.innerHTML = playIcon;
                button.setAttribute('aria-label', 'Play text-to-speech');
                break;
            case 'pause':
                button.innerHTML = pauseIcon;
                button.setAttribute('aria-label', 'Pause text-to-speech');
                break;
            case 'loading':
                button.innerHTML = loadingIcon;
                button.setAttribute('aria-label', 'Loading text-to-speech');
                break;
        }
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }

    /**
     * Mirrors ChatUI.renderMarkdown but outputs link text only (no <a> tags).
     * Keeps TTS decoupled from chat UI while using the same markdown patterns.
     */
    _renderMarkdownForTTS(text) {
        let html = text;
        // Escape HTML (matches renderMarkdown — makes DOM insertion XSS-safe)
        html = html.replace(/</g, '&lt;').replace(/>/g, '&gt;');
        // Code blocks → <pre><code>
        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, _lang, code) => {
            return `<pre><code>${code.trim()}</code></pre>`;
        });
        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
        // Bold
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        // Italic
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
        // Links → plain text only (no URL spoken)
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1');
        // NOTE: Unlike renderMarkdown, we do NOT convert \n to <br> here.
        // DOM .textContent ignores <br> elements, so preserving \n as-is
        // ensures line breaks survive the DOM extraction step.
        return html;
    }

    /**
     * Convert raw LLM markdown text to clean plain text for TTS.
     * Pre-processes patterns renderMarkdown doesn't handle, then uses
     * DOM .textContent to strip exactly what the app recognises as markup.
     */
    prepareTextForTTS(rawText) {
        let text = rawText;

        // --- Pre-process patterns not handled by renderMarkdown ---
        // Remove code blocks entirely (we don't want code read aloud)
        text = text.replace(/```[\s\S]*?```/g, '');
        // Remove images ![alt](url)
        text = text.replace(/!\[([^\]]*)\]\([^)]+\)/g, '');
        // Remove headers (# ## ### etc.)
        text = text.replace(/^#{1,6}\s+/gm, '');
        // Remove blockquotes
        text = text.replace(/^>\s+/gm, '');
        // Remove unordered list markers (-, *, +)
        text = text.replace(/^[\s]*[-*+]\s+/gm, '');
        // Remove ordered list markers (1., 2., etc.)
        text = text.replace(/^[\s]*\d+\.\s+/gm, '');
        // Remove horizontal rules (---, ***, ___)
        text = text.replace(/^[\s]*[-*_]{3,}[\s]*$/gm, '');
        // Remove strikethrough
        text = text.replace(/~~(.+?)~~/g, '$1');
        // Remove HTML comments
        text = text.replace(/<!--[\s\S]*?-->/g, '');

        // --- Convert remaining markdown (bold, italic, inline code, links) via DOM ---
        const html = this._renderMarkdownForTTS(text);
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;

        // Strip any <pre><code> blocks that survived (e.g. unclosed fences)
        tempDiv.querySelectorAll('pre code').forEach(el => el.textContent = '');

        let cleaned = tempDiv.textContent || '';

        // --- Whitespace cleanup (TextSplitterStream compatibility) ---
        cleaned = cleaned.replace(/ {2,}/g, ' ');
        cleaned = cleaned.split('\n').map(line => line.trimEnd()).join('\n');
        cleaned = cleaned.replace(/\n{3,}/g, '\n\n').trim();

        return cleaned;
    }

    stopAll() {
        for (const [messageId, player] of this.activePlayers.entries()) {
            player.stop();
            const button = this.activeButtons.get(messageId);
            if (button) {
                this.updateButtonIcon(button, 'play');
            }
        }
        this.activePlayers.clear();
        this.activeButtons.clear();
        this.audioData.clear();
        this.downloadAudioBtns.clear();
    }

    destroy() {
        this.stopAll();
        ttsEngine.abort();
    }
}

/**
 * Combine multiple WAV ArrayBuffers into a single WAV file.
 * All chunks must share the same audio format (sample rate, channels, bit depth).
 */
function combineWavBuffers(wavArrayBuffers) {
    if (wavArrayBuffers.length === 0) return new ArrayBuffer(0);
    if (wavArrayBuffers.length === 1) return wavArrayBuffers[0];

    // Parse header from first chunk to get format info
    const firstView = new DataView(wavArrayBuffers[0]);
    const audioFormat = firstView.getUint16(20, true);
    const numChannels = firstView.getUint16(22, true);
    const sampleRate = firstView.getUint32(24, true);
    const bitsPerSample = firstView.getUint16(34, true);
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;

    // Calculate total PCM data size (strip 44-byte header from each chunk)
    let totalDataSize = 0;
    for (const buf of wavArrayBuffers) {
        totalDataSize += buf.byteLength - 44;
    }

    // Create output buffer: 44-byte header + all PCM data
    const output = new ArrayBuffer(44 + totalDataSize);
    const view = new DataView(output);
    const bytes = new Uint8Array(output);

    // Write WAV header
    // "RIFF"
    view.setUint8(0, 0x52); view.setUint8(1, 0x49); view.setUint8(2, 0x46); view.setUint8(3, 0x46);
    // File size - 8
    view.setUint32(4, 36 + totalDataSize, true);
    // "WAVE"
    view.setUint8(8, 0x57); view.setUint8(9, 0x41); view.setUint8(10, 0x56); view.setUint8(11, 0x45);
    // "fmt "
    view.setUint8(12, 0x66); view.setUint8(13, 0x6D); view.setUint8(14, 0x74); view.setUint8(15, 0x20);
    // fmt chunk size
    view.setUint32(16, 16, true);
    // Audio format (preserve from source — 3 for IEEE float from Transformers.js)
    view.setUint16(20, audioFormat, true);
    // Number of channels
    view.setUint16(22, numChannels, true);
    // Sample rate
    view.setUint32(24, sampleRate, true);
    // Byte rate
    view.setUint32(28, byteRate, true);
    // Block align
    view.setUint16(32, blockAlign, true);
    // Bits per sample
    view.setUint16(34, bitsPerSample, true);
    // "data"
    view.setUint8(36, 0x64); view.setUint8(37, 0x61); view.setUint8(38, 0x74); view.setUint8(39, 0x61);
    // Data size
    view.setUint32(40, totalDataSize, true);

    // Copy PCM data from each chunk (skip 44-byte headers)
    let offset = 44;
    for (const buf of wavArrayBuffers) {
        const src = new Uint8Array(buf, 44);
        bytes.set(src, offset);
        offset += src.length;
    }

    return output;
}

const ttsUI = new TTSUI();
export default ttsUI;

export function addTTSButton(messageElement, messageContent, messageId) {
    const existingControls = messageElement.querySelector('.tts-controls');
    if (existingControls) {
        existingControls.remove();
        // Clean up stale Map refs
        ttsUI.downloadAudioBtns.delete(messageId);
    }

    const ttsContainer = ttsUI.createTTSButton(messageContent, messageId);
    const messageContentDiv = messageElement.querySelector('.message-content');
    if (messageContentDiv) {
        messageContentDiv.appendChild(ttsContainer);
    }
}

export function stopAllTTS() {
    ttsUI.stopAll();
}

export function destroyTTS() {
    ttsUI.destroy();
}
