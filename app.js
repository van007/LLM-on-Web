import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.3';
import { loadLLM, loadEmbedder, onProgress } from './llm/loader.js';
import { initializeChatUI, sendChatMessage, stopChatGeneration } from './ui/chat-ui.js';
import {
    initializeVectorStore,
    addDocument,
    getVectorStoreStats,
    getAllDocuments,
    deleteDocument,
    clearAllDocuments
} from './embeddings/store.js';
import { initializeEmbedder } from './embeddings/embedder.js';
import { processFiles } from './utils/text.js';
import logger from './utils/logger.js';

class LLMWebApp {
    constructor() {
        this.elements = {};
        this.state = {
            isReady: false,
            isGenerating: false,
            backend: 'auto',
            temperature: 1.0,
            topP: 0.9,
            maxTokens: 256,
            ragEnabled: true,
            ragThreshold: 0.2,
            messages: [],
            llmPipeline: null,
            embeddingPipeline: null,
            chatUI: null,
            vectorStoreReady: false
        };

        this.initElements();
        this.bindEvents();
        this.loadSettings();
        this.restoreChat();
        this.initialize();
    }

    initElements() {
        this.elements = {
            statusPill: document.getElementById('statusPill'),
            statusText: document.querySelector('.status-text'),
            chatTranscript: document.getElementById('chatTranscript'),
            chatInput: document.getElementById('chatInput'),
            sendBtn: document.getElementById('sendBtn'),
            stopBtn: document.getElementById('stopBtn'),
            tokenCounter: document.getElementById('tokenCounter'),
            timeCounter: document.getElementById('timeCounter'),
            sidebar: document.getElementById('sidebar'),
            sidebarToggle: document.getElementById('sidebarToggle'),
            settingsBtn: document.getElementById('settingsBtn'),
            backendSelect: document.getElementById('backendSelect'),
            llmModelSelect: document.getElementById('llmModelSelect'),
            ttsModelSelect: document.getElementById('ttsModelSelect'),
            embeddingModelSelect: document.getElementById('embeddingModelSelect'),
            temperatureSlider: document.getElementById('temperatureSlider'),
            temperatureValue: document.getElementById('temperatureValue'),
            topPSlider: document.getElementById('topPSlider'),
            topPValue: document.getElementById('topPValue'),
            maxTokensSlider: document.getElementById('maxTokensSlider'),
            maxTokensValue: document.getElementById('maxTokensValue'),
            ragToggle: document.getElementById('ragToggle'),
            ragThresholdContainer: document.getElementById('ragThresholdContainer'),
            ragThresholdSlider: document.getElementById('ragThresholdSlider'),
            ragThresholdValue: document.getElementById('ragThresholdValue'),
            fileDropZone: document.getElementById('fileDropZone'),
            fileInput: document.getElementById('fileInput'),
            docCount: document.getElementById('docCount'),
            chunkCount: document.getElementById('chunkCount'),
            clearChatBtn: document.getElementById('clearChatBtn'),
            exportChatBtn: document.getElementById('exportChatBtn'),
            warmupBtn: document.getElementById('warmupBtn'),
            clearCacheBtn: document.getElementById('clearCacheBtn'),
            documentList: document.getElementById('documentList'),
            refreshDocsBtn: document.getElementById('refreshDocsBtn'),
            clearAllDocsBtn: document.getElementById('clearAllDocsBtn'),
            // Status details elements
            statusDetails: document.getElementById('statusDetails'),
            statusLLM: document.getElementById('statusLLM'),
            statusEmbedding: document.getElementById('statusEmbedding'),
            statusTTS: document.getElementById('statusTTS'),
            statusTemp: document.getElementById('statusTemp'),
            statusTopP: document.getElementById('statusTopP'),
            statusMaxTokens: document.getElementById('statusMaxTokens'),
            statusRAG: document.getElementById('statusRAG'),
            statusDocs: document.getElementById('statusDocs')
        };
    }

    bindEvents() {
        this.elements.chatInput.addEventListener('keydown', (e) => this.handleInputKeydown(e));
        this.elements.sendBtn.addEventListener('click', () => this.sendMessage());
        this.elements.stopBtn.addEventListener('click', () => this.stopGeneration());

        this.elements.sidebarToggle.addEventListener('click', () => this.toggleSidebar());
        this.elements.settingsBtn.addEventListener('click', () => this.toggleSidebar());

        // Status pill click handler
        this.elements.statusPill.addEventListener('click', () => this.toggleStatusDetails());
        this.elements.statusPill.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.toggleStatusDetails();
            }
        });

        // Click outside to close status details
        document.addEventListener('click', (e) => {
            if (!this.elements.statusPill.contains(e.target) &&
                !this.elements.statusDetails.contains(e.target) &&
                !this.elements.statusDetails.hidden) {
                this.closeStatusDetails();
            }
        });

        // Info icon tooltip interactions
        this.initializeTooltips();

        this.elements.backendSelect.addEventListener('change', (e) => {
            this.state.backend = e.target.value;
            this.saveSettings();
        });

        this.elements.temperatureSlider.addEventListener('input', (e) => {
            this.state.temperature = parseFloat(e.target.value);
            this.elements.temperatureValue.textContent = this.state.temperature;
            this.saveSettings();
        });

        this.elements.topPSlider.addEventListener('input', (e) => {
            this.state.topP = parseFloat(e.target.value);
            this.elements.topPValue.textContent = this.state.topP;
            this.saveSettings();
        });

        this.elements.maxTokensSlider.addEventListener('input', (e) => {
            this.state.maxTokens = parseInt(e.target.value);
            this.elements.maxTokensValue.textContent = this.state.maxTokens;
            this.saveSettings();
        });

        this.elements.ragToggle.addEventListener('change', (e) => {
            this.state.ragEnabled = e.target.checked;
            this.saveSettings();
            // Show/hide threshold slider
            if (this.elements.ragThresholdContainer) {
                this.elements.ragThresholdContainer.style.display = e.target.checked ? 'block' : 'none';
            }
            // Update chat UI with RAG state if it's initialized
            if (this.state.chatUI && typeof this.state.chatUI.setRAGEnabled === 'function') {
                this.state.chatUI.setRAGEnabled(this.state.ragEnabled);
                this.state.chatUI.setRAGThreshold(this.state.ragThreshold);
            }
        });

        this.elements.ragThresholdSlider?.addEventListener('input', (e) => {
            this.state.ragThreshold = parseFloat(e.target.value);
            if (this.elements.ragThresholdValue) {
                this.elements.ragThresholdValue.textContent = this.state.ragThreshold;
            }
            this.saveSettings();
            // Update chat UI with new threshold
            if (this.state.chatUI && typeof this.state.chatUI.setRAGThreshold === 'function') {
                this.state.chatUI.setRAGThreshold(this.state.ragThreshold);
            }
        });

        this.elements.clearChatBtn.addEventListener('click', () => this.clearChat());
        this.elements.exportChatBtn.addEventListener('click', () => this.exportChat());
        this.elements.warmupBtn.addEventListener('click', () => this.warmupModel());
        this.elements.clearCacheBtn.addEventListener('click', () => this.clearCache());

        // Document management buttons
        this.elements.refreshDocsBtn?.addEventListener('click', () => this.loadDocumentList());
        this.elements.clearAllDocsBtn?.addEventListener('click', () => this.clearAllDocuments());

        this.setupFileUpload();
        this.setupResponsiveLayout();
    }

    setupFileUpload() {
        const dropZone = this.elements.fileDropZone;
        const fileInput = this.elements.fileInput;

        // Removed click handler - file input already covers the drop zone

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            this.handleFiles(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });
    }

    setupResponsiveLayout() {
        const checkWidth = () => {
            // Remove the open class on resize to larger screens
            if (window.innerWidth > 768) {
                this.elements.sidebar.classList.remove('open');
            }
        };

        window.addEventListener('resize', checkWidth);
        checkWidth();
    }

    toggleSidebar() {
        this.elements.sidebar.classList.toggle('open');
        // Update aria-expanded for accessibility
        const isOpen = this.elements.sidebar.classList.contains('open');
        this.elements.settingsBtn.setAttribute('aria-expanded', isOpen);
    }

    handleInputKeydown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (this.state.isReady && !this.state.isGenerating) {
                this.sendMessage();
            }
        }
    }

    async initialize() {
        this.updateStatus('loading', 'Initializing...');

        try {
            // Initialize vector store first
            this.updateStatus('loading', 'Setting up vector store...');
            await initializeVectorStore();
            this.state.vectorStoreReady = true;
            await this.updateDocumentStats();

            const backend = this.detectBackend();

            // Set up progress monitoring
            onProgress((event) => {
                if (event.status === 'progress') {
                    const progress = Math.round(event.progress || 0);
                    this.updateStatus('loading', `Loading models... ${progress}%`);
                } else if (event.status === 'initiate') {
                    this.updateStatus('loading', `Downloading ${event.type}...`);
                }
            });

            const llmModel = this.elements.llmModelSelect.value;
            const embeddingModel = this.elements.embeddingModelSelect.value;

            await this.loadModels(llmModel, embeddingModel, backend);

            // Initialize embedder with the loaded pipeline
            await initializeEmbedder(embeddingModel, { device: backend });

            // Initialize chat UI
            this.state.chatUI = await initializeChatUI(this.elements);

            // Set initial RAG state in chat UI
            if (this.state.chatUI) {
                if (this.state.ragEnabled) {
                    this.state.chatUI.setRAGEnabled(this.state.ragEnabled);
                }
                this.state.chatUI.setRAGThreshold(this.state.ragThreshold);
            }

            this.state.isReady = true;
            this.elements.sendBtn.disabled = false;
            this.updateStatus('ready', 'Ready');

            const welcomeMessage = this.elements.chatTranscript.querySelector('.welcome-message');
            if (welcomeMessage && this.state.messages.length === 0) {
                welcomeMessage.querySelector('.loading-status').textContent = 'Models loaded! Start chatting below.';
            }
        } catch (error) {
            logger.error('Initialization error:', error);
            this.updateStatus('error', 'Failed to load models');
            this.showError('Failed to initialize the app. Please refresh and try again.');
        }
    }

    detectBackend() {
        if (this.state.backend === 'auto') {
            return navigator.gpu ? 'webgpu' : 'wasm';
        }
        return this.state.backend;
    }

    async loadModels(llmModel, embeddingModel, backend) {
        const device = backend;

        // Load LLM with proper quantization for Qwen2.5
        this.state.llmPipeline = await loadLLM(llmModel, {
            dtype: 'q4',
            device
        });

        // Load embedder with fp32 for all-MiniLM
        this.state.embeddingPipeline = await loadEmbedder(embeddingModel, {
            dtype: 'fp32',
            device
        });
    }

    updateStatus(status, text) {
        this.elements.statusPill.setAttribute('data-status', status);
        this.elements.statusText.textContent = text;
    }

    async toggleStatusDetails() {
        if (this.elements.statusDetails.hidden) {
            await this.updateStatusDetails();
            this.elements.statusDetails.hidden = false;
            this.elements.statusPill.setAttribute('aria-expanded', 'true');
        } else {
            this.closeStatusDetails();
        }
    }

    closeStatusDetails() {
        this.elements.statusDetails.hidden = true;
        this.elements.statusPill.setAttribute('aria-expanded', 'false');
    }

    initializeTooltips() {
        // Handle info icon clicks for mobile-friendly tooltips
        const infoIcons = document.querySelectorAll('.info-icon');
        let currentOpenTooltip = null;

        infoIcons.forEach(icon => {
            // Click handler for mobile and desktop
            icon.addEventListener('click', (e) => {
                e.stopPropagation();
                e.preventDefault();
                const tooltip = icon.querySelector('.tooltip');

                // Close any other open tooltip
                if (currentOpenTooltip && currentOpenTooltip !== tooltip) {
                    currentOpenTooltip.classList.remove('active');
                }

                // Toggle current tooltip
                if (tooltip.classList.contains('active')) {
                    tooltip.classList.remove('active');
                    currentOpenTooltip = null;
                } else {
                    tooltip.classList.add('active');
                    currentOpenTooltip = tooltip;

                    // Check if tooltip goes off-screen and adjust
                    setTimeout(() => {
                        const rect = tooltip.getBoundingClientRect();
                        if (rect.right > window.innerWidth - 10) {
                            tooltip.classList.add('tooltip-right');
                        }
                        if (rect.left < 10) {
                            tooltip.classList.add('tooltip-left');
                        }
                    }, 10);
                }
            });

            // Keyboard accessibility
            icon.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    icon.click();
                }
            });
        });

        // Close tooltips when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.info-icon') && currentOpenTooltip) {
                currentOpenTooltip.classList.remove('active');
                currentOpenTooltip = null;
            }
        });
    }

    async updateStatusDetails() {
        // Update model names
        this.elements.statusLLM.textContent = this.elements.llmModelSelect.value || 'Not loaded';
        this.elements.statusEmbedding.textContent = this.elements.embeddingModelSelect.value || 'Not loaded';
        this.elements.statusTTS.textContent = this.elements.ttsModelSelect.value || 'Not loaded';

        // Update generation parameters
        this.elements.statusTemp.textContent = this.state.temperature.toFixed(1);
        this.elements.statusTopP.textContent = this.state.topP.toFixed(2);
        this.elements.statusMaxTokens.textContent = this.state.maxTokens;

        // Update RAG status
        this.elements.statusRAG.textContent = this.state.ragEnabled ? 'Enabled' : 'Disabled';

        // Update document count
        if (this.state.vectorStoreReady) {
            try {
                const docs = await getAllDocuments();
                const docCount = docs.length;
                const totalChunks = docs.reduce((sum, doc) => sum + doc.chunkCount, 0);
                this.elements.statusDocs.textContent = docCount === 0
                    ? 'No documents'
                    : `${docCount} document${docCount !== 1 ? 's' : ''} (${totalChunks} chunks)`;
            } catch (error) {
                this.elements.statusDocs.textContent = 'Error loading documents';
            }
        } else {
            this.elements.statusDocs.textContent = 'Vector store not ready';
        }
    }

    async sendMessage() {
        const input = this.elements.chatInput.value.trim();
        if (!input) return;

        this.elements.chatInput.value = '';
        this.adjustInputHeight();

        this.updateStatus('generating', 'Generating...');

        try {
            // Use the enhanced chat UI with RAG support
            await sendChatMessage(input, this.state.ragEnabled);
        } catch (error) {
            logger.error('Generation error:', error);
        } finally {
            this.updateStatus('ready', 'Ready');
        }
    }

    async generateResponse(userInput) {
        const messages = [
            { role: 'system', content: 'You are a helpful assistant running locally in the browser.' },
            ...this.state.messages.map(m => ({ role: m.role, content: m.content })),
            { role: 'user', content: userInput }
        ];

        let responseText = '';
        const messageElement = this.createMessageElement('assistant', '');
        const contentElement = messageElement.querySelector('.message-content');
        this.elements.chatTranscript.appendChild(messageElement);

        // Skip special tokens for cleaner output
        const streamer = new TextStreamer(this.state.llmPipeline.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (text) => {
                responseText += text;
                contentElement.textContent = responseText;
                this.scrollToBottom();

                // Update token count
                const tokenCount = this.state.llmPipeline.tokenizer.encode(responseText).length;
                this.elements.tokenCounter.textContent = `Tokens: ${tokenCount}`;
            }
        });

        const result = await this.state.llmPipeline(messages, {
            max_new_tokens: this.state.maxTokens,
            temperature: this.state.temperature,
            top_p: this.state.topP,
            do_sample: this.state.temperature > 0,
            streamer
        });

        return result[0].generated_text.at(-1).content;
    }

    async stopGeneration() {
        logger.log('Stop generation requested');
        await stopChatGeneration();
    }

    addMessage(role, content) {
        const message = { role, content, timestamp: Date.now() };
        this.state.messages.push(message);

        // Only add user messages here, assistant messages are added in generateResponse
        if (role === 'user') {
            const messageElement = this.createMessageElement(role, content);

            const welcomeMessage = this.elements.chatTranscript.querySelector('.welcome-message');
            if (welcomeMessage) {
                welcomeMessage.remove();
            }

            this.elements.chatTranscript.appendChild(messageElement);
            this.scrollToBottom();
        }
    }

    createMessageElement(role, content) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message message-${role}`;

        const roleLabel = document.createElement('div');
        roleLabel.className = 'message-role';
        roleLabel.textContent = role === 'user' ? 'You' : 'Assistant';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        messageDiv.appendChild(roleLabel);
        messageDiv.appendChild(contentDiv);

        return messageDiv;
    }

    scrollToBottom() {
        this.elements.chatTranscript.scrollTop = this.elements.chatTranscript.scrollHeight;
    }

    adjustInputHeight() {
        const input = this.elements.chatInput;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
    }

    async clearChat() {
        if (this.state.chatUI) {
            await this.state.chatUI.clearChat();
        } else {
            // Fallback to old implementation
            const confirmed = await this.showConfirm('Are you sure you want to clear the chat history?', 'Clear Chat', { type: 'danger', confirmText: 'Clear', cancelText: 'Cancel' });
            if (confirmed) {
                this.state.messages = [];
                this.elements.chatTranscript.innerHTML = `
                    <div class="welcome-message">
                        <h2>Welcome to LLM on Web</h2>
                        <p>Your private AI assistant running entirely in your browser. No data leaves your device.</p>
                        <p class="loading-status">Ready to chat!</p>
                    </div>
                `;
                this.saveChat();
            }
        }
    }

    async exportChat() {
        if (this.state.chatUI) {
            await this.state.chatUI.exportChat();
        } else {
            // Fallback to old implementation
            const exportData = {
                messages: this.state.messages,
                exportedAt: new Date().toISOString(),
                settings: {
                    temperature: this.state.temperature,
                    topP: this.state.topP,
                    maxTokens: this.state.maxTokens
                }
            };

            const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `llm-web-chat-${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }
    }

    async warmupModel() {
        if (!this.state.llmPipeline) {
            this.showAlert('Models not loaded yet', 'Warning');
            return;
        }

        this.updateStatus('generating', 'Warming up...');
        try {
            await this.state.llmPipeline([
                { role: 'system', content: 'You are a helpful assistant.' },
                { role: 'user', content: 'Say hello' }
            ], {
                max_new_tokens: 10,
                temperature: 1.0,
                do_sample: true
            });
            this.updateStatus('ready', 'Ready');
            this.showAlert('Model warmed up successfully!', 'Success');
        } catch (error) {
            logger.error('Warmup error:', error);
            this.updateStatus('error', 'Warmup failed');
        }
    }

    async clearCache() {
        const confirmed = await this.showConfirm('This will clear all cached models. You will need to download them again. Continue?', 'Clear Cache', { type: 'danger', confirmText: 'Clear Cache', cancelText: 'Cancel' });
        if (!confirmed) {
            return;
        }

        try {
            if ('caches' in window) {
                const cacheNames = await caches.keys();
                await Promise.all(cacheNames.map(name => caches.delete(name)));
            }

            const databases = await indexedDB.databases();
            for (const db of databases) {
                if (db.name && db.name.includes('transformers')) {
                    indexedDB.deleteDatabase(db.name);
                }
            }

            await this.showAlert('Cache cleared! Please refresh the page.', 'Success');
            location.reload();
        } catch (error) {
            logger.error('Clear cache error:', error);
            this.showAlert('Failed to clear cache', 'Error');
        }
    }

    async handleFiles(files) {
        if (!this.state.vectorStoreReady) {
            this.showAlert('Vector store not ready. Please wait for initialization.', 'Warning');
            return;
        }

        if (!this.state.embeddingPipeline) {
            this.showAlert('Embedding model not loaded. Please wait.', 'Warning');
            return;
        }

        this.updateStatus('processing', 'Processing files...');
        const progressModal = this.showProgressModal('Processing Files');

        try {
            const results = await processFiles(Array.from(files), (progress) => {
                progressModal.update(`Processing: ${progress.file} (${progress.current}/${progress.total})`);
            });

            let successCount = 0;
            let errorCount = 0;
            const processedDocs = [];

            for (const result of results) {
                if (result.success) {
                    // Check if we actually got content
                    if (result.content && result.content.length > 10) {
                        progressModal.update(`Adding to vector store: ${result.metadata.name}`);

                        // Add document and verify
                        const docResult = await addDocument(result.content, result.metadata);

                        if (docResult && docResult.docId && docResult.chunkCount > 0) {
                            successCount++;
                            processedDocs.push({
                                name: result.metadata.name,
                                chunks: docResult.chunkCount,
                                chars: result.content.length
                            });
                            logger.log(`[App] Successfully added ${result.metadata.name}: ${docResult.chunkCount} chunks, ${result.content.length} chars`);
                        } else {
                            logger.error(`[App] Failed to create embeddings for ${result.metadata.name}`);
                            errorCount++;
                        }
                    } else {
                        logger.warn(`[App] Skipping ${result.metadata.name}: No extractable content (${result.content?.length || 0} chars`);
                        errorCount++;
                    }
                } else {
                    logger.error(`[App] Failed to process ${result.metadata.name}:`, result.error);
                    errorCount++;
                }
            }

            progressModal.close();
            await this.updateDocumentStats();

            // Show detailed results
            let message = `Processed ${successCount} file(s) successfully` +
                         (errorCount > 0 ? ` (${errorCount} failed)` : '');

            if (processedDocs.length > 0) {
                message += '\n\nDocuments added:';
                processedDocs.forEach(doc => {
                    message += `\n• ${doc.name}: ${doc.chunks} chunks (${doc.chars} chars)`;
                });
            }

            if (errorCount > 0) {
                message += '\n\nCheck the console for error details.';
            }

            this.showAlert(message, 'File Upload Results');
            this.updateStatus('ready', 'Ready');
        } catch (error) {
            logger.error('File processing error:', error);
            progressModal.close();
            this.showAlert('Error processing files: ' + error.message, 'Error');
            this.updateStatus('error', 'File processing failed');
        }
    }

    saveSettings() {
        const settings = {
            backend: this.state.backend,
            temperature: this.state.temperature,
            topP: this.state.topP,
            maxTokens: this.state.maxTokens,
            ragEnabled: this.state.ragEnabled,
            ragThreshold: this.state.ragThreshold
        };
        localStorage.setItem('llm-web-settings', JSON.stringify(settings));
    }

    loadSettings() {
        const stored = localStorage.getItem('llm-web-settings');
        if (stored) {
            const settings = JSON.parse(stored);
            Object.assign(this.state, settings);

            this.elements.backendSelect.value = this.state.backend;
            this.elements.temperatureSlider.value = this.state.temperature;
            this.elements.temperatureValue.textContent = this.state.temperature;
            this.elements.topPSlider.value = this.state.topP;
            this.elements.topPValue.textContent = this.state.topP;
            this.elements.maxTokensSlider.value = this.state.maxTokens;
            this.elements.maxTokensValue.textContent = this.state.maxTokens;
            this.elements.ragToggle.checked = this.state.ragEnabled;
            if (this.elements.ragThresholdSlider) {
                this.elements.ragThresholdSlider.value = this.state.ragThreshold || 0.2;
                this.elements.ragThresholdValue.textContent = this.state.ragThreshold || 0.2;
                this.elements.ragThresholdContainer.style.display = this.state.ragEnabled ? 'block' : 'none';
            }
        }
    }

    saveChat() {
        localStorage.setItem('llm-web-chat', JSON.stringify(this.state.messages));
    }

    restoreChat() {
        const stored = localStorage.getItem('llm-web-chat');
        if (stored) {
            this.state.messages = JSON.parse(stored);
            if (this.state.messages.length > 0) {
                this.elements.chatTranscript.innerHTML = '';
                this.state.messages.forEach(msg => {
                    const messageElement = this.createMessageElement(msg.role, msg.content);
                    this.elements.chatTranscript.appendChild(messageElement);
                });
                this.scrollToBottom();
            }
        }
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--danger);
            color: white;
            padding: 12px 20px;
            border-radius: var(--radius);
            z-index: 9999;
        `;
        document.body.appendChild(errorDiv);
        setTimeout(() => errorDiv.remove(), 5000);
    }

    showProgressModal(title) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: var(--surface);
            padding: 20px;
            border-radius: var(--radius);
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            z-index: 10000;
            min-width: 300px;
        `;

        const titleEl = document.createElement('h3');
        titleEl.textContent = title;
        titleEl.style.marginBottom = '10px';
        modal.appendChild(titleEl);

        const messageEl = document.createElement('p');
        messageEl.textContent = 'Processing...';
        modal.appendChild(messageEl);

        const backdrop = document.createElement('div');
        backdrop.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0,0,0,0.5);
            z-index: 9999;
        `;

        document.body.appendChild(backdrop);
        document.body.appendChild(modal);

        return {
            update: (message) => {
                messageEl.textContent = message;
            },
            close: () => {
                modal.remove();
                backdrop.remove();
            }
        };
    }

    showAlert(message, title = 'Notice') {
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
                color: var(--text-primary);
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
            `;

            const okBtn = document.createElement('button');
            okBtn.textContent = 'OK';
            okBtn.className = 'btn btn-primary';
            okBtn.style.cssText = `
                min-width: 80px;
                padding: 8px 16px;
            `;

            footer.appendChild(okBtn);

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
            const close = () => {
                backdrop.style.opacity = '0';
                modal.style.opacity = '0';
                modal.style.transform = 'translate(-50%, -50%) scale(0.9)';
                setTimeout(() => {
                    backdrop.remove();
                    modal.remove();
                    resolve();
                }, 300);
            };

            okBtn.addEventListener('click', close);
            backdrop.addEventListener('click', close);

            // Keyboard support
            const handleKeydown = (e) => {
                if (e.key === 'Escape' || e.key === 'Enter') {
                    close();
                    document.removeEventListener('keydown', handleKeydown);
                }
            };
            document.addEventListener('keydown', handleKeydown);

            // Focus OK button
            okBtn.focus();
        });
    }

    showConfirm(message, title = 'Confirm', options = {}) {
        const {
            confirmText = 'OK',
            cancelText = 'Cancel',
            type = 'default' // 'default' or 'danger'
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

    async updateDocumentStats() {
        if (!this.state.vectorStoreReady) return;

        try {
            const stats = await getVectorStoreStats();
            if (this.elements.docCount) {
                this.elements.docCount.textContent = stats.documents || 0;
            }
            if (this.elements.chunkCount) {
                this.elements.chunkCount.textContent = stats.chunks || 0;
            }
            // Also load document list if the container exists
            if (this.elements.documentList) {
                await this.loadDocumentList();
            }
        } catch (error) {
            logger.error('Error updating document stats:', error);
        }
    }

    async loadDocumentList() {
        if (!this.state.vectorStoreReady || !this.elements.documentList) return;

        try {
            const documents = await getAllDocuments();

            if (documents.length === 0) {
                this.elements.documentList.innerHTML = '<div class="no-documents">No documents uploaded yet</div>';
                return;
            }

            // Build document list HTML
            let html = '';
            documents.forEach(doc => {
                const date = new Date(doc.createdAt);
                const dateStr = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();

                html += `
                    <div class="document-item" data-doc-id="${doc.id}">
                        <div class="document-info">
                            <div class="document-name">${this.escapeHtml(doc.name)}</div>
                            <div class="document-meta">
                                <span class="doc-size">${doc.sizeKB} KB</span>
                                <span class="doc-chunks">${doc.chunkCount} chunks</span>
                                <span class="doc-date">${dateStr}</span>
                            </div>
                        </div>
                        <button class="btn-delete-doc" data-doc-id="${doc.id}" title="Delete document">
                            🗑️
                        </button>
                    </div>
                `;
            });

            this.elements.documentList.innerHTML = html;

            // Bind delete buttons
            this.elements.documentList.querySelectorAll('.btn-delete-doc').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const docId = parseInt(btn.dataset.docId);
                    await this.deleteDocument(docId);
                });
            });

        } catch (error) {
            logger.error('Error loading document list:', error);
            this.elements.documentList.innerHTML = '<div class="error-message">Failed to load documents</div>';
        }
    }

    async deleteDocument(docId) {
        const confirmed = await this.showConfirm('Delete this document? This action cannot be undone.', 'Delete Document', { type: 'danger', confirmText: 'Delete', cancelText: 'Cancel' });
        if (!confirmed) return;

        const progressModal = this.showProgressModal('Deleting Document');

        try {
            progressModal.update('Removing document and embeddings...');
            const result = await deleteDocument(docId);

            progressModal.close();

            // Show success message
            this.showSuccess(`Document deleted (${result.deletedChunks} chunks removed)`);

            // Update UI
            await this.updateDocumentStats();

        } catch (error) {
            progressModal.close();
            logger.error('Error deleting document:', error);
            this.showError('Failed to delete document: ' + error.message);
        }
    }

    async clearAllDocuments() {
        const confirmed = await this.showConfirm('Delete ALL documents? This will remove all uploaded documents and their embeddings. This action cannot be undone.', 'Clear All Documents', { type: 'danger', confirmText: 'Delete All', cancelText: 'Cancel' });
        if (!confirmed) return;

        const progressModal = this.showProgressModal('Clearing All Documents');

        try {
            progressModal.update('Removing all documents and embeddings...');
            await clearAllDocuments();

            progressModal.close();

            // Show success message
            this.showSuccess('All documents cleared successfully');

            // Update UI
            await this.updateDocumentStats();

        } catch (error) {
            progressModal.close();
            logger.error('Error clearing documents:', error);
            this.showError('Failed to clear documents: ' + error.message);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showSuccess(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.textContent = message;
        successDiv.style.cssText = `
            position: fixed;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--success, #4CAF50);
            color: white;
            padding: 12px 20px;
            border-radius: var(--radius);
            z-index: 9999;
        `;
        document.body.appendChild(successDiv);
        setTimeout(() => successDiv.remove(), 3000);
    }
}

new LLMWebApp();

if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js').then(
            () => logger.log('ServiceWorker registered'),
            error => logger.log('ServiceWorker registration failed:', error)
        );
    });
}