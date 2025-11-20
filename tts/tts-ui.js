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
            const textContent = this.extractTextFromHTML(messageContent);

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

    sanitizeMarkdown(text) {
        let cleaned = text;

        // Remove code blocks (```...```)
        cleaned = cleaned.replace(/```[\s\S]*?```/g, '');

        // Remove inline code (`...`) - use non-greedy
        cleaned = cleaned.replace(/`(.+?)`/g, '$1');

        // Remove bold (**...** or __...__) - use non-greedy to handle multiple sections
        cleaned = cleaned.replace(/\*\*(.+?)\*\*/g, '$1');
        cleaned = cleaned.replace(/__(.+?)__/g, '$1');

        // Remove italic (*...* or _..._) - use lookaheads to avoid matching ** as two *
        cleaned = cleaned.replace(/(?<!\*)\*(?!\*)(.+?)\*(?!\*)/g, '$1');
        cleaned = cleaned.replace(/(?<!_)_(?!_)(.+?)_(?!_)/g, '$1');

        // Remove strikethrough (~~...~~) - use non-greedy
        cleaned = cleaned.replace(/~~(.+?)~~/g, '$1');

        // Remove links [text](url) -> text - use non-greedy
        cleaned = cleaned.replace(/\[(.+?)\]\([^)]+\)/g, '$1');

        // Remove images ![alt](url)
        cleaned = cleaned.replace(/!\[([^\]]*)\]\([^)]+\)/g, '');

        // Remove headers (# ## ### etc)
        cleaned = cleaned.replace(/^#{1,6}\s+/gm, '');

        // Remove blockquotes (> )
        cleaned = cleaned.replace(/^>\s+/gm, '');

        // Remove list markers (-, *, +, 1., 2., etc)
        cleaned = cleaned.replace(/^[\s]*[-*+]\s+/gm, '');
        cleaned = cleaned.replace(/^[\s]*\d+\.\s+/gm, '');

        // Remove horizontal rules (---, ***, ___)
        cleaned = cleaned.replace(/^[\s]*[-*_]{3,}[\s]*$/gm, '');

        // Remove HTML comments
        cleaned = cleaned.replace(/<!--[\s\S]*?-->/g, '');

        // Fallback: Remove any remaining stray markdown characters
        cleaned = cleaned.replace(/[*_~`]/g, '');

        // Remove multiple consecutive spaces (fixes TextSplitterStream truncation)
        cleaned = cleaned.replace(/ {2,}/g, ' ');

        // Remove trailing spaces from each line (fixes TextSplitterStream bug)
        cleaned = cleaned.split('\n').map(line => line.trimEnd()).join('\n');

        // Clean up multiple newlines and trim
        cleaned = cleaned.replace(/\n{3,}/g, '\n\n').trim();

        return cleaned;
    }

    extractTextFromHTML(html) {
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = html;

        const codeBlocks = tempDiv.querySelectorAll('pre code');
        codeBlocks.forEach(block => {
            block.textContent = '';
        });

        const sources = tempDiv.querySelectorAll('.message-sources');
        sources.forEach(source => source.remove());

        let text = tempDiv.textContent || tempDiv.innerText || '';
        text = text.replace(/\s+/g, ' ').trim();

        // Sanitize markdown syntax before returning
        text = this.sanitizeMarkdown(text);

        return text;
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