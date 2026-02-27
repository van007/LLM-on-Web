import ttsEngine from './tts-engine.js';
import AudioPlayer from './audio-player.js';
import logger from '../utils/logger.js';

class TTSUI {
    constructor() {
        this.activePlayers = new Map();
        this.activeButtons = new Map();
        this.isModelLoading = false;
    }

    createTTSButton(messageContent, messageId) {
        const container = document.createElement('div');
        container.className = 'tts-controls';

        const button = document.createElement('button');
        button.className = 'tts-button';
        button.setAttribute('aria-label', 'Play text-to-speech');
        button.dataset.messageId = messageId;

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

        button.innerHTML = playIcon;

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

        container.appendChild(button);
        container.appendChild(progressContainer);

        button.addEventListener('click', async () => {
            await this.handleButtonClick(button, messageContent, messageId, progressFill, progressText);
        });

        return container;
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

                // Initialize streaming playback
                await player.initializeStreamingPlayback(result);
            } else if (Array.isArray(result)) {
                // Handle array of audio chunks (non-streaming mode)
                if (result.length === 0) {
                    throw new Error('No audio chunks generated');
                }

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
    }

    destroy() {
        this.stopAll();
        ttsEngine.abort();
    }
}

const ttsUI = new TTSUI();
export default ttsUI;

export function addTTSButton(messageElement, messageContent, messageId) {
    const existingControls = messageElement.querySelector('.tts-controls');
    if (existingControls) {
        existingControls.remove();
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