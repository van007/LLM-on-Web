import { SimpleStore } from '../utils/idb.js';
import { chatEngine } from '../llm/chat-engine.js';
import { ragPipeline } from '../rag/rag.js';
import { vectorStore } from '../embeddings/store.js';
import { addTTSButton, stopAllTTS } from '../tts/tts-ui.js';
import logger from '../utils/logger.js';

class ChatUI {
    constructor() {
        this.elements = null;
        this.chatStore = null;
        this.isGenerating = false;
        this.abortController = null;
        this.currentSessionId = null;
        this.ragEnabled = false;
        this.ragThreshold = 0.2;
        this.currentSources = [];
    }

    async initialize(elements) {
        this.elements = elements;

        // Initialize chat store
        this.chatStore = new SimpleStore('llm-web-chats', 'messages', 1);
        await this.chatStore.init('id', [
            { name: 'sessionId', keyPath: 'sessionId' },
            { name: 'timestamp', keyPath: 'timestamp' }
        ]);

        // Start new session or restore last
        await this.initSession();

        // Bind input events
        this.bindInputEvents();

        // Restore RAG state
        this.ragEnabled = this.elements.ragToggle?.checked || false;

        return true;
    }

    bindInputEvents() {
        // Auto-resize textarea
        this.elements.chatInput.addEventListener('input', () => {
            this.autoResizeInput();
        });

        // Note: clearChatBtn and exportChatBtn event listeners are handled in app.js
        // to avoid duplicate event listeners
    }

    async initSession() {
        // Get or create session ID
        const storedSession = localStorage.getItem('llm-web-current-session');
        if (storedSession) {
            this.currentSessionId = storedSession;
            await this.restoreSession(storedSession);
        } else {
            this.currentSessionId = this.generateSessionId();
            localStorage.setItem('llm-web-current-session', this.currentSessionId);
        }
    }

    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
    }

    async restoreSession(sessionId) {
        const messages = await this.chatStore.getAll();
        const sessionMessages = messages
            .filter(msg => msg.sessionId === sessionId)
            .sort((a, b) => a.timestamp - b.timestamp);

        if (sessionMessages.length > 0) {
            // Clear welcome message
            const welcomeMessage = this.elements.chatTranscript.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }

            // Restore messages
            sessionMessages.forEach((msg, index) => {
                const messageElement = this.createMessageElement(msg.role, msg.content);
                if (msg.sources && msg.sources.length > 0) {
                    this.addSourcesToMessage(messageElement, msg.sources);
                }
                this.elements.chatTranscript.appendChild(messageElement);

                // Add TTS button for assistant messages in history
                if (msg.role === 'assistant' && msg.content) {
                    const messageId = messageElement.dataset.messageId;
                    addTTSButton(messageElement, msg.content, messageId);
                }
            });

            this.scrollToBottom();
        }
    }

    async sendMessage(content, useRAG = null) {
        if (this.isGenerating) {
            logger.warn('Generation already in progress');
            return;
        }

        const shouldUseRAG = useRAG !== null ? useRAG : this.ragEnabled;

        // Clear welcome message
        const welcomeMessage = this.elements.chatTranscript.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }

        // Add user message
        this.renderMessage('user', content);
        await this.saveMessage('user', content);

        // Prepare for generation
        this.isGenerating = true;
        this.abortController = new AbortController();
        this.updateGeneratingState(true);

        const startTime = performance.now();
        let responseText = '';
        let sources = [];

        try {
            // Prepare messages
            let messages = await this.getConversationHistory();

            // Apply RAG if enabled
            if (shouldUseRAG) {
                // Show RAG processing status
                const ragStatus = document.createElement('div');
                ragStatus.className = 'rag-status';
                ragStatus.style.cssText = 'padding: 8px; margin: 8px 0; background: #f0f4f8; border-radius: 4px; font-size: 0.9em; color: #666;';
                ragStatus.textContent = '🔍 Searching knowledge base...';
                this.elements.chatTranscript.appendChild(ragStatus);

                const ragResult = await ragPipeline.processQuery(content, {
                    maxContextTokens: 1500,
                    topK: 5,
                    threshold: this.ragThreshold  // Use configurable threshold
                });

                // Update RAG status
                if (ragResult.context && ragResult.sources.length > 0) {
                    sources = ragResult.sources;
                    messages = ragResult.messages;
                    ragStatus.textContent = `✅ Found ${ragResult.sources.length} relevant document${ragResult.sources.length > 1 ? 's' : ''}`;
                    ragStatus.style.background = '#e8f4e8';
                    ragStatus.style.color = '#2d7a2d';

                    // Add conversation history after system message with context
                    const history = await this.getConversationHistory();
                    if (history.length > 1) {
                        messages = [
                            messages[0], // System message with context
                            ...history.slice(0, -1), // Previous messages (excluding current)
                            messages[1] // Current user message
                        ];
                    }
                } else {
                    // No context found - show warning and proceed without RAG
                    ragStatus.textContent = '⚠️ No relevant documents found. Answering without additional context.';
                    ragStatus.style.background = '#fff3cd';
                    ragStatus.style.color = '#856404';

                    // Fall back to regular conversation
                    messages = await this.getConversationHistory();
                    messages.push({ role: 'user', content });

                    logger.log('[ChatUI] RAG search returned no results, proceeding without context');
                }

                // Remove status after a delay
                setTimeout(() => ragStatus.remove(), 5000);
            } else {
                messages.push({ role: 'user', content });
            }

            // Create message element for streaming
            const messageElement = this.createMessageElement('assistant', '');
            const contentElement = messageElement.querySelector('.message-content');
            this.elements.chatTranscript.appendChild(messageElement);

            // Generate response with streaming
            await chatEngine.generateStream({
                messages,
                params: {
                    maxNewTokens: this.elements.maxTokensSlider?.value || 256,
                    temperature: parseFloat(this.elements.temperatureSlider?.value || 0.7),
                    topP: parseFloat(this.elements.topPSlider?.value || 0.9)
                },
                onToken: (token, stats) => {
                    responseText += token;
                    contentElement.innerHTML = this.renderMarkdown(responseText);
                    this.scrollToBottom();

                    // Update stats
                    if (this.elements.tokenCounter) {
                        this.elements.tokenCounter.textContent = `Tokens: ${stats.totalTokens}`;
                    }
                    if (this.elements.timeCounter && this.elements.timeCounter.style.display !== 'none') {
                        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
                        this.elements.timeCounter.textContent = `Time: ${elapsed}s`;
                    }
                },
                onDone: async (stats) => {
                    // Add TTS button for the complete message
                    const messageId = messageElement.dataset.messageId || `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
                    messageElement.dataset.messageId = messageId;
                    addTTSButton(messageElement, stats.text || responseText, messageId);

                    // Add sources if available
                    if (sources && sources.length > 0) {
                        this.addSourcesToMessage(messageElement, sources);
                        logger.log(`[ChatUI] Response generated with ${sources.length} sources`);
                    } else if (shouldUseRAG) {
                        // Add note that no sources were used despite RAG being enabled
                        const noSourcesNote = document.createElement('div');
                        noSourcesNote.className = 'no-sources-note';
                        noSourcesNote.style.cssText = 'font-size: 0.85em; color: #888; margin-top: 8px; font-style: italic;';
                        noSourcesNote.textContent = 'ℹ️ Answered without document context';
                        messageElement.appendChild(noSourcesNote);
                    }

                    // Save assistant message
                    await this.saveMessage('assistant', stats.text || responseText, sources);

                    // Update final stats
                    if (stats.tokensPerSecond) {
                        logger.log(`Generation stats: ${stats.tokens} tokens in ${(stats.time / 1000).toFixed(1)}s (${stats.tokensPerSecond.toFixed(1)} tok/s`);
                    }
                },
                onError: (error) => {
                    logger.error('Generation error:', error);
                    contentElement.textContent = 'Sorry, an error occurred while generating the response.';
                },
                signal: this.abortController.signal
            });

        } catch (error) {
            logger.error('Chat error:', error);
            this.renderMessage('assistant', 'Sorry, an error occurred. Please try again.');
        } finally {
            this.isGenerating = false;
            this.abortController = null;
            this.updateGeneratingState(false);
        }
    }

    async stopGeneration() {
        if (this.abortController) {
            this.abortController.abort();
            chatEngine.stop();
        }
    }

    renderMessage(role, content, sources = null) {
        const messageElement = this.createMessageElement(role, content);

        if (sources && sources.length > 0) {
            this.addSourcesToMessage(messageElement, sources);
        }

        this.elements.chatTranscript.appendChild(messageElement);
        this.scrollToBottom();
    }

    createMessageElement(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role}`;

        const messageId = `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        messageDiv.dataset.messageId = messageId;

        const roleLabel = document.createElement('div');
        roleLabel.className = 'message-role';
        roleLabel.textContent = role === 'user' ? 'You' : 'Assistant';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (content) {
            contentDiv.innerHTML = this.renderMarkdown(content);
        }

        messageDiv.appendChild(roleLabel);
        messageDiv.appendChild(contentDiv);

        if (role === 'assistant' && content) {
            addTTSButton(messageDiv, content, messageId);
        }

        return messageDiv;
    }

    addSourcesToMessage(messageElement, sources) {
        const sourcesPanel = document.createElement('div');
        sourcesPanel.className = 'sources-panel';

        const header = document.createElement('div');
        header.className = 'sources-header';
        header.textContent = `📚 Sources (${sources.length})`;
        header.style.cursor = 'pointer';

        const list = document.createElement('div');
        list.className = 'sources-list';
        list.style.display = 'none';

        sources.forEach((source, index) => {
            const item = document.createElement('div');
            item.className = 'source-item';
            const relevance = source.relevance ? ` (${(source.relevance * 100).toFixed(0)}% relevant)` : '';
            item.textContent = `${index + 1}. ${source.name}${relevance}`;
            list.appendChild(item);
        });

        header.addEventListener('click', () => {
            list.style.display = list.style.display === 'none' ? 'block' : 'none';
            header.textContent = list.style.display === 'none'
                ? `📚 Sources (${sources.length})`
                : `📖 Sources (${sources.length})`;
        });

        sourcesPanel.appendChild(header);
        sourcesPanel.appendChild(list);
        messageElement.appendChild(sourcesPanel);
    }

    renderMarkdown(text) {
        // Basic markdown rendering
        let html = text;

        // Escape HTML
        html = html.replace(/</g, '&lt;').replace(/>/g, '&gt;');

        // Code blocks
        html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
            const languageLabel = lang ? `<span class="code-lang">${lang}</span>` : '';
            return `<pre><code>${languageLabel}${code.trim()}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

        // Links
        html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

        // Line breaks
        html = html.replace(/\n/g, '<br>');

        return html;
    }

    updateGeneratingState(isGenerating) {
        if (this.elements.sendBtn) {
            this.elements.sendBtn.style.display = isGenerating ? 'none' : 'inline-flex';
        }
        if (this.elements.stopBtn) {
            this.elements.stopBtn.style.display = isGenerating ? 'inline-flex' : 'none';
        }
        if (this.elements.timeCounter) {
            this.elements.timeCounter.style.display = isGenerating ? 'inline' : 'none';
        }
    }

    async getConversationHistory() {
        const messages = await this.chatStore.getAll();
        const sessionMessages = messages
            .filter(msg => msg.sessionId === this.currentSessionId)
            .sort((a, b) => a.timestamp - b.timestamp)
            .map(msg => ({
                role: msg.role,
                content: msg.content
            }));

        // Add system message if not present
        if (sessionMessages.length === 0 || sessionMessages[0].role !== 'system') {
            sessionMessages.unshift({
                role: 'system',
                content: 'You are a helpful assistant running locally in the browser.'
            });
        }

        return sessionMessages;
    }

    async saveMessage(role, content, sources = null) {
        const message = {
            id: `${this.currentSessionId}_${Date.now()}`,
            sessionId: this.currentSessionId,
            role,
            content,
            sources,
            timestamp: Date.now()
        };

        await this.chatStore.put(message);
        return message;
    }

    showConfirm(message, title = 'Confirm', options = {}) {
        const {
            confirmText = 'OK',
            cancelText = 'Cancel',
            type = 'default'
        } = options;

        return new Promise((resolve) => {
            // Create backdrop
            const backdrop = document.createElement('div');
            backdrop.style.cssText = `
                position: fixed;
                inset: 0;
                background: rgba(0,0,0,0.5);
                z-index: 9999;
                opacity: 0;
                transition: opacity 0.3s ease;
            `;

            // Create modal
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%) scale(0.9);
                background: var(--surface);
                padding: 0;
                border-radius: var(--radius);
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                z-index: 10000;
                min-width: 320px;
                max-width: 500px;
                opacity: 0;
                transition: all 0.3s ease;
            `;

            // Modal header
            const header = document.createElement('div');
            header.style.cssText = `
                padding: 20px;
                border-bottom: 1px solid var(--border);
            `;
            const titleEl = document.createElement('h3');
            titleEl.textContent = title;
            titleEl.style.cssText = `
                margin: 0;
                color: ${type === 'danger' ? 'var(--danger)' : 'var(--text-primary)'};
                font-size: 1.1rem;
            `;
            header.appendChild(titleEl);

            // Modal body
            const body = document.createElement('div');
            body.style.cssText = `
                padding: 20px;
                max-height: 400px;
                overflow-y: auto;
            `;
            const messageEl = document.createElement('p');
            messageEl.style.cssText = `
                margin: 0;
                color: var(--text-secondary);
                white-space: pre-wrap;
                line-height: 1.5;
            `;
            messageEl.textContent = message;
            body.appendChild(messageEl);

            // Modal footer
            const footer = document.createElement('div');
            footer.style.cssText = `
                padding: 15px 20px;
                border-top: 1px solid var(--border);
                display: flex;
                justify-content: flex-end;
                gap: 10px;
            `;

            const cancelBtn = document.createElement('button');
            cancelBtn.textContent = cancelText;
            cancelBtn.className = 'btn btn-secondary';
            cancelBtn.style.cssText = `
                min-width: 80px;
                padding: 8px 16px;
            `;

            const confirmBtn = document.createElement('button');
            confirmBtn.textContent = confirmText;
            confirmBtn.className = type === 'danger' ? 'btn btn-danger' : 'btn btn-primary';
            confirmBtn.style.cssText = `
                min-width: 80px;
                padding: 8px 16px;
            `;

            footer.appendChild(cancelBtn);
            footer.appendChild(confirmBtn);

            // Assemble modal
            modal.appendChild(header);
            modal.appendChild(body);
            modal.appendChild(footer);

            // Add to DOM
            document.body.appendChild(backdrop);
            document.body.appendChild(modal);

            // Trigger animation
            requestAnimationFrame(() => {
                backdrop.style.opacity = '1';
                modal.style.opacity = '1';
                modal.style.transform = 'translate(-50%, -50%) scale(1)';
            });

            // Event handlers
            const close = (result) => {
                backdrop.style.opacity = '0';
                modal.style.opacity = '0';
                modal.style.transform = 'translate(-50%, -50%) scale(0.9)';
                setTimeout(() => {
                    backdrop.remove();
                    modal.remove();
                    resolve(result);
                }, 300);
            };

            confirmBtn.addEventListener('click', () => close(true));
            cancelBtn.addEventListener('click', () => close(false));
            backdrop.addEventListener('click', () => close(false));

            // Keyboard support
            const handleKeydown = (e) => {
                if (e.key === 'Escape') {
                    close(false);
                    document.removeEventListener('keydown', handleKeydown);
                } else if (e.key === 'Enter' && e.target !== cancelBtn) {
                    close(true);
                    document.removeEventListener('keydown', handleKeydown);
                }
            };
            document.addEventListener('keydown', handleKeydown);

            // Focus cancel button by default for safety
            cancelBtn.focus();
        });
    }

    async clearChat() {
        const confirmed = await this.showConfirm('Clear chat history? This cannot be undone.', 'Clear Chat', {
            type: 'danger',
            confirmText: 'Clear',
            cancelText: 'Cancel'
        });

        if (!confirmed) {
            return;
        }

        // Stop all TTS playback
        stopAllTTS();

        // Clear current session messages
        const messages = await this.chatStore.getAll();
        const sessionMessages = messages.filter(msg => msg.sessionId === this.currentSessionId);

        for (const msg of sessionMessages) {
            await this.chatStore.delete(msg.id);
        }

        // Clear UI
        this.elements.chatTranscript.innerHTML = `
            <div class="welcome-message">
                <h2>Welcome to LLM on Web</h2>
                <p>Your private AI assistant running entirely in your browser.</p>
                <p class="loading-status">Ready to chat!</p>
            </div>
        `;

        // Start new session
        this.currentSessionId = this.generateSessionId();
        localStorage.setItem('llm-web-current-session', this.currentSessionId);
    }

    async exportChat() {
        const messages = await this.chatStore.getAll();
        const sessionMessages = messages
            .filter(msg => msg.sessionId === this.currentSessionId)
            .sort((a, b) => a.timestamp - b.timestamp);

        const exportData = {
            sessionId: this.currentSessionId,
            exportedAt: new Date().toISOString(),
            messages: sessionMessages.map(msg => ({
                role: msg.role,
                content: msg.content,
                timestamp: new Date(msg.timestamp).toISOString(),
                sources: msg.sources
            }))
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat-${this.currentSessionId}-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    autoResizeInput() {
        const input = this.elements.chatInput;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    }

    scrollToBottom() {
        if (this.elements.chatTranscript) {
            this.elements.chatTranscript.scrollTop = this.elements.chatTranscript.scrollHeight;
        }
    }

    setRAGEnabled(enabled) {
        this.ragEnabled = enabled;
    }

    setRAGThreshold(threshold) {
        this.ragThreshold = threshold;
        logger.log(`[ChatUI] RAG threshold set to ${threshold}`);
    }

    async getStats() {
        const messageCount = await this.chatStore.count();
        const vectorStats = await vectorStore.getStats();

        return {
            messages: messageCount,
            sessions: 1, // Current session
            ...vectorStats
        };
    }
}

// Create singleton instance
const chatUI = new ChatUI();

export { chatUI, ChatUI };

export async function initializeChatUI(elements) {
    await chatUI.initialize(elements);
    return chatUI;  // Return the chatUI instance, not the initialization result
}

export async function sendChatMessage(content, useRAG) {
    return chatUI.sendMessage(content, useRAG);
}

export async function stopChatGeneration() {
    return chatUI.stopGeneration();
}

export async function clearChatHistory() {
    return chatUI.clearChat();
}

export async function exportChatHistory() {
    return chatUI.exportChat();
}