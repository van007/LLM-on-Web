<div align="center">
  <img src="assets/icons/icon.png" alt="LLM on Web Logo" width="128" height="128">
  <h1>LLM on Web</h1>
  <p>
    <strong>Run AI models directly in your browser - no server required!</strong>
  </p>
  <p>
    <strong>Version 0.1.0</strong>
  </p>
  <p>
    A Progressive Web App featuring a chat interface with RAG (Retrieval Augmented Generation) capabilities for document-based Q&A.
  </p>
  <p>
    <img src="https://img.shields.io/badge/Privacy-100%25%20Local-green?style=for-the-badge" alt="Privacy">
    <img src="https://img.shields.io/badge/WebGPU-Accelerated-blue?style=for-the-badge" alt="WebGPU">
    <img src="https://img.shields.io/badge/PWA-Installable-orange?style=for-the-badge" alt="PWA">
    <img src="https://img.shields.io/badge/License-MPL%202.0-brightgreen?style=for-the-badge" alt="License">
  </p>
</div>

---

## 🚀 Features

- **100% Browser-Based**: All processing happens locally in your browser
- **Privacy-First**: Your data never leaves your device
- **RAG Support**: Upload documents for context-aware responses
- **WebGPU Acceleration**: Fast inference with WebGPU, falls back to WASM
- **Text-to-Speech**: High-quality speech synthesis with streaming support (Kokoro-82M)
- **Offline Capable**: Works offline once models are cached
- **PWA**: Install as a native app on any platform
- **Multiple File Formats**: Supports TXT, MD, PDF, HTML, JSON, and code files
- **Document Management**: View, delete individual documents or clear all at once
- **System Status**: Click status indicator to view models, parameters, and document stats
- **JetBrains Mono Font**: Clean monospace typography throughout the interface

## 🎯 Quick Start

### Option 1: Python HTTP Server (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/LLM-on-Web.git
cd LLM-on-Web

# Start local server with Python 3
python3 -m http.server 8080

# Open in browser
# Navigate to http://localhost:8080
```

### Option 2: Deploy to Vercel
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Or for development
vercel dev
```

### Option 3: Deploy to Netlify
```bash
# Drag and drop the folder to Netlify
# Or use Netlify CLI
netlify deploy
```

## 📱 Installation as PWA

1. Open the app in Chrome, Edge, or Safari
2. Click the install button in the address bar
3. Or use browser menu: "Install LLM on Web"

## 🧠 Models

### Default Models
- **Text Generation**: Qwen2.5-0.5B-Instruct (500M parameters)
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
- **Text-to-Speech**: Kokoro-82M (82M parameters)

Models are automatically downloaded and cached on first use (~200-500MB for LLM/embeddings, ~330MB for TTS).

## 📚 Using RAG (Retrieval Augmented Generation)

1. **Enable RAG**: Toggle "Enable RAG" in the settings panel
2. **Upload Documents**: Drag & drop files onto the upload zone
3. **Ask Questions**: The AI will use your documents as context

### Supported File Types
- Text files (.txt)
- Markdown (.md)
- PDF documents (.pdf)
- HTML files (.html)
- JSON data (.json)
- Code files (.js, .py, etc.)

### Example Questions After Uploading Documents
- "What does this document say about X?"
- "Summarize the main points"
- "Explain the code in this file"

## 🔊 Text-to-Speech (TTS)

The app includes high-quality text-to-speech functionality powered by Kokoro-82M model with advanced streaming support:

### How to Use TTS
1. **Automatic Integration**: Play buttons appear on all assistant messages
2. **Click to Play**: Click the play button to generate and play speech
3. **Playback Controls**: Click again to pause/resume playback
4. **Progress Indicator**: Visual progress bar shows playback position
5. **Streaming Indicator**: Shows "(streaming...)" for longer messages being generated

### TTS Features
- **WebGPU Accelerated**: Uses WebGPU for fast audio generation (falls back to WASM)
- **Streaming Generation**: Messages >100 tokens (~400 characters) use real-time streaming
  - Audio starts playing immediately while rest generates
  - Lower latency to first audio
  - Memory-efficient processing
  - Visual streaming indicator during generation
- **Smart Text Processing**: Automatically sanitizes markdown formatting before speech
  - Removes asterisks, underscores, and other formatting characters
  - Normalizes whitespace to prevent truncation issues
  - Preserves natural speech flow with proper pauses
- **Offline Capable**: Works offline once the TTS model is cached
- **High Quality**: Natural-sounding speech synthesis with Kokoro-82M
- **Smart Mode Selection**: Automatically chooses between direct and streaming generation

## ⚙️ Configuration

### Settings Panel
- **Temperature**: Controls response creativity (0.0 = focused, 1.0 = creative)
- **Max Tokens**: Maximum response length (32-512 tokens)
- **Top-P**: Nucleus sampling parameter
- **Enable RAG**: Toggle document-based responses

### Advanced Settings
Edit `app.js` to change:
- Default models
- Chunk size for document processing
- Number of retrieved contexts
- System prompts

## 🏗️ Architecture

```
LLM-on-Web/
├── app.js                 # Main application controller
├── index.html            # UI structure
├── styles.css           # Styling
├── manifest.webmanifest # PWA configuration
├── sw.js               # Service worker for offline
├── llm/               # Language model components
│   ├── loader.js      # Model loading & caching
│   └── chat-engine.js # Text generation & streaming
├── embeddings/        # Vector embedding system
│   ├── embedder.js    # Embedding generation
│   └── store.js       # Vector storage (IndexedDB)
├── rag/              # RAG pipeline
│   └── rag.js        # Document retrieval & context
├── tts/              # Text-to-Speech system
│   ├── tts-engine.js # TTS model & streaming generation
│   ├── audio-player.js # Audio playback & streaming queue
│   └── tts-ui.js    # TTS UI components & streaming UI
├── ui/              # User interface
│   └── chat-ui.js   # Chat interface & sessions
└── utils/          # Utilities
    ├── text.js     # File processing & chunking
    └── idb.js      # IndexedDB wrapper
```

## 🔧 Technical Requirements

### Browser Support
- Chrome 90+ (recommended)
- Edge 90+
- Safari 16.4+
- Firefox 110+

### Required Features
- ES Modules
- WebGPU or WASM
- IndexedDB
- Service Workers
- Cross-Origin Isolation (COOP/COEP)

## 🛡️ Security & Privacy

- **No Data Collection**: No analytics, no tracking, no telemetry
- **Local Processing**: Models run entirely in your browser
- **No Server Calls**: Except for initial model downloads from HuggingFace
- **Secure Context**: Requires HTTPS or localhost
- **Cross-Origin Isolation**: Enabled for WebGPU/WASM threading

## 🐛 Troubleshooting

### Models Won't Load
- Check internet connection for initial download
- Ensure browser supports WebGPU/WASM
- Clear browser cache and reload
- Check console for specific errors

### Slow Performance
- WebGPU provides 2-5x speedup over WASM
- Close other tabs to free memory
- Reduce max tokens for faster responses
- Use smaller models if available

### PDF Text Extraction Issues
- PDF.js library loads from CDN
- Complex PDFs may have limited extraction
- Try copying text manually from PDF viewer

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Copyright © 2025 Varun Nidhi

This project is open source and available under the [Mozilla Public License 2.0](LICENSE).

## 🙏 Acknowledgments

- [Transformers.js](https://github.com/xenova/transformers.js) - Browser-based transformer models
- [HuggingFace](https://huggingface.co) - Model hosting
- [PDF.js](https://mozilla.github.io/pdf.js/) - PDF text extraction
- [Kokoro-js](https://github.com/hexgrad/kokoro) - Text-to-speech library
- Models: Qwen2.5 by Alibaba, all-MiniLM by Microsoft, Kokoro-82M by hexgrad

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the console for error messages

---

**Note**: This is an experimental project showcasing browser-based AI capabilities. Model performance depends on your device's capabilities and browser support for WebGPU/WASM.