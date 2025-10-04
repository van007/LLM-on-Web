// Text processing utilities for document ingestion
import logger from './logger.js';

class TextProcessor {
    constructor() {
        this.supportedFormats = ['.txt', '.md', '.json', '.js', '.html', '.css', '.pdf'];
    }

    async processFile(file) {
        const fileName = file.name;
        const fileType = this.getFileType(fileName);

        if (!this.isSupported(fileType)) {
            throw new Error(`Unsupported file type: ${fileType}`);
        }

        let content = '';
        let metadata = {
            name: fileName,
            type: fileType,
            size: file.size,
            lastModified: file.lastModified
        };

        switch (fileType) {
            case '.pdf':
                content = await this.processPDF(file);
                break;
            case '.json':
                content = await this.processJSON(file);
                break;
            case '.html':
                content = await this.processHTML(file);
                break;
            case '.md':
                content = await this.processMarkdown(file);
                break;
            default:
                content = await this.processText(file);
        }

        return {
            content: this.cleanText(content),
            metadata
        };
    }

    async processText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = reject;
            reader.readAsText(file);
        });
    }

    async processMarkdown(file) {
        const text = await this.processText(file);
        // Remove markdown formatting for embedding
        return this.stripMarkdown(text);
    }

    async processHTML(file) {
        const text = await this.processText(file);
        // Strip HTML tags
        return this.stripHTML(text);
    }

    async processJSON(file) {
        const text = await this.processText(file);
        try {
            const json = JSON.parse(text);
            // Convert JSON to readable text
            return this.jsonToText(json);
        } catch (error) {
            // If not valid JSON, return as is
            return text;
        }
    }

    async processPDF(file) {
        logger.log(`[TextProcessor] Starting PDF processing with pdf.js: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`);

        try {
            // Check if pdf.js is available (access from window in ES module context)
            if (typeof window === 'undefined' || typeof window.pdfjsLib === 'undefined') {
                logger.error('[TextProcessor] PDF.js library not loaded, falling back to basic extraction');
                return this.processPDFBasic(file);
            }

            const arrayBuffer = await file.arrayBuffer();

            // Load the PDF document using window.pdfjsLib
            const loadingTask = window.pdfjsLib.getDocument({ data: arrayBuffer });
            const pdf = await loadingTask.promise;

            logger.log(`[TextProcessor] PDF loaded: ${pdf.numPages} pages`);

            let fullText = '';
            let totalChars = 0;

            // Extract text from each page
            for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
                const page = await pdf.getPage(pageNum);
                const textContent = await page.getTextContent();

                // Combine all text items from the page
                const pageText = textContent.items
                    .map(item => item.str)
                    .join(' ');

                if (pageText.trim()) {
                    fullText += `\n--- Page ${pageNum} ---\n${pageText}\n`;
                    totalChars += pageText.length;
                    logger.log(`[TextProcessor] Page ${pageNum}: extracted ${pageText.length} chars`);
                }
            }

            // Clean up the text
            fullText = fullText
                .replace(/\s+/g, ' ')  // Normalize whitespace
                .replace(/- \n/g, '')  // Remove hyphenation
                .replace(/\n{3,}/g, '\n\n')  // Limit consecutive newlines
                .trim();

            if (fullText.length < 50) {
                logger.warn(`[TextProcessor] PDF.js extracted minimal text (${fullText.length} chars), trying basic extraction`);
                return this.processPDFBasic(file);
            }

            logger.log(`[TextProcessor] Successfully extracted ${fullText.length} characters from PDF using pdf.js`);
            logger.log(`[TextProcessor] First 200 chars: "${fullText.substring(0, 200)}..."`);
            return fullText;

        } catch (error) {
            logger.error('[TextProcessor] PDF.js extraction failed:', error);
            logger.log('[TextProcessor] Falling back to basic extraction');
            return this.processPDFBasic(file);
        }
    }

    async processPDFBasic(file) {
        // Original basic PDF extraction as fallback
        logger.log(`[TextProcessor] Using basic PDF extraction for: ${file.name}`);

        const arrayBuffer = await file.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);

        // Check if this is actually a PDF
        const pdfHeader = bytes.slice(0, 4);
        const isPDF = pdfHeader[0] === 0x25 && pdfHeader[1] === 0x50 &&
                      pdfHeader[2] === 0x44 && pdfHeader[3] === 0x46; // %PDF

        if (!isPDF) {
            logger.error(`[TextProcessor] Invalid PDF header for ${file.name}`);
            throw new Error('Invalid PDF file format');
        }

        // Convert to string for pattern matching
        const decoder = new TextDecoder('utf-8', { fatal: false });
        const fullText = decoder.decode(bytes);

        // Try to extract text between BT/ET markers
        let extractedText = '';
        const textPattern = /BT[\s\S]*?\(([\s\S]*?)\)[\s\S]*?ET/g;
        let match;

        while ((match = textPattern.exec(fullText)) !== null) {
            if (match[1]) {
                let chunk = match[1]
                    .replace(/\\r/g, '\n')
                    .replace(/\\n/g, '\n')
                    .replace(/\\t/g, '\t')
                    .replace(/\\\(/g, '(')
                    .replace(/\\\)/g, ')')
                    .replace(/\\\\/g, '\\');
                extractedText += chunk + ' ';
            }
        }

        // If insufficient text, try parentheses pattern
        if (extractedText.length < 100) {
            const simplePattern = /\(([^)]+)\)/g;
            extractedText = '';

            while ((match = simplePattern.exec(fullText)) !== null) {
                if (match[1] && match[1].length > 1) {
                    const text = match[1];
                    const printableRatio = text.split('').filter(c =>
                        c.charCodeAt(0) >= 32 && c.charCodeAt(0) <= 126
                    ).length / text.length;

                    if (printableRatio > 0.8) {
                        extractedText += text + ' ';
                    }
                }
            }
        }

        // Clean up
        extractedText = extractedText
            .replace(/\s+/g, ' ')
            .replace(/[^\x20-\x7E\n\r\t]/g, '')
            .trim();

        if (extractedText.length < 50) {
            return `⚠️ Unable to extract text from PDF "${file.name}".

The PDF may contain:
• Scanned images requiring OCR
• Complex encoding or compression
• Password protection

Please try:
1. Copy and paste text from your PDF viewer
2. Save as plain text from your PDF application
3. Use an OCR tool for scanned documents`;
        }

        logger.log(`[TextProcessor] Basic extraction: ${extractedText.length} chars`);
        return extractedText;
    }

    stripMarkdown(text) {
        // Remove code blocks
        text = text.replace(/```[\s\S]*?```/g, '');
        text = text.replace(/`[^`]*`/g, '');

        // Remove headers
        text = text.replace(/^#{1,6}\s+/gm, '');

        // Remove bold and italic
        text = text.replace(/\*\*([^*]+)\*\*/g, '$1');
        text = text.replace(/\*([^*]+)\*/g, '$1');
        text = text.replace(/__([^_]+)__/g, '$1');
        text = text.replace(/_([^_]+)_/g, '$1');

        // Remove links
        text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, '$1');

        // Remove images
        text = text.replace(/!\[([^\]]*)\]\([^)]+\)/g, '');

        // Remove blockquotes
        text = text.replace(/^>\s+/gm, '');

        // Remove lists markers
        text = text.replace(/^[\*\-\+]\s+/gm, '');
        text = text.replace(/^\d+\.\s+/gm, '');

        return text;
    }

    stripHTML(html) {
        // Remove script and style elements
        html = html.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
        html = html.replace(/<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>/gi, '');

        // Remove HTML tags
        html = html.replace(/<[^>]+>/g, ' ');

        // Decode HTML entities
        const textarea = document.createElement('textarea');
        textarea.innerHTML = html;
        html = textarea.value;

        return html;
    }

    jsonToText(obj, indent = 0) {
        const lines = [];
        const spacing = '  '.repeat(indent);

        if (typeof obj === 'object' && obj !== null) {
            if (Array.isArray(obj)) {
                obj.forEach((item, index) => {
                    lines.push(`${spacing}Item ${index + 1}:`);
                    lines.push(this.jsonToText(item, indent + 1));
                });
            } else {
                Object.entries(obj).forEach(([key, value]) => {
                    if (typeof value === 'object' && value !== null) {
                        lines.push(`${spacing}${key}:`);
                        lines.push(this.jsonToText(value, indent + 1));
                    } else {
                        lines.push(`${spacing}${key}: ${value}`);
                    }
                });
            }
        } else {
            lines.push(`${spacing}${obj}`);
        }

        return lines.join('\n');
    }

    cleanText(text) {
        // Remove excessive whitespace
        text = text.replace(/\s+/g, ' ');

        // Remove non-printable characters (keep newlines)
        text = text.replace(/[^\x20-\x7E\n\r\t]/g, '');

        // Normalize line endings
        text = text.replace(/\r\n/g, '\n');
        text = text.replace(/\r/g, '\n');

        // Remove excessive newlines
        text = text.replace(/\n{3,}/g, '\n\n');

        // Trim
        text = text.trim();

        return text;
    }

    getFileType(fileName) {
        const lastDot = fileName.lastIndexOf('.');
        if (lastDot === -1) return '.txt';
        return fileName.substring(lastDot).toLowerCase();
    }

    isSupported(fileType) {
        return this.supportedFormats.includes(fileType);
    }

    async processMultipleFiles(files, onProgress) {
        const results = [];
        const total = files.length;

        for (let i = 0; i < total; i++) {
            const file = files[i];

            try {
                const result = await this.processFile(file);
                results.push({
                    success: true,
                    ...result
                });
            } catch (error) {
                results.push({
                    success: false,
                    error: error.message,
                    metadata: {
                        name: file.name,
                        type: this.getFileType(file.name)
                    }
                });
            }

            if (onProgress) {
                onProgress({
                    current: i + 1,
                    total,
                    percent: ((i + 1) / total) * 100,
                    file: file.name
                });
            }
        }

        return results;
    }

    // Chunking utilities
    chunkBySentences(text, maxChunkSize = 800, overlap = 200) {
        // Split by sentence endings
        const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
        const chunks = [];
        let currentChunk = [];
        let currentSize = 0;

        for (const sentence of sentences) {
            const sentenceSize = sentence.split(/\s+/).length;

            if (currentSize + sentenceSize > maxChunkSize && currentChunk.length > 0) {
                chunks.push(currentChunk.join(' '));

                // Add overlap
                const overlapSentences = [];
                let overlapSize = 0;
                for (let i = currentChunk.length - 1; i >= 0; i--) {
                    const size = currentChunk[i].split(/\s+/).length;
                    if (overlapSize + size <= overlap) {
                        overlapSentences.unshift(currentChunk[i]);
                        overlapSize += size;
                    } else {
                        break;
                    }
                }
                currentChunk = overlapSentences;
                currentSize = overlapSize;
            }

            currentChunk.push(sentence);
            currentSize += sentenceSize;
        }

        if (currentChunk.length > 0) {
            chunks.push(currentChunk.join(' '));
        }

        return chunks;
    }

    chunkByParagraphs(text, maxChunkSize = 800, overlap = 200) {
        // Split by double newlines (paragraphs)
        const paragraphs = text.split(/\n\n+/).filter(p => p.trim());
        const chunks = [];
        let currentChunk = [];
        let currentSize = 0;

        for (const paragraph of paragraphs) {
            const paragraphSize = paragraph.split(/\s+/).length;

            if (currentSize + paragraphSize > maxChunkSize && currentChunk.length > 0) {
                chunks.push(currentChunk.join('\n\n'));

                // Add overlap
                const overlapParagraphs = [];
                let overlapSize = 0;
                for (let i = currentChunk.length - 1; i >= 0; i--) {
                    const size = currentChunk[i].split(/\s+/).length;
                    if (overlapSize + size <= overlap) {
                        overlapParagraphs.unshift(currentChunk[i]);
                        overlapSize += size;
                    } else {
                        break;
                    }
                }
                currentChunk = overlapParagraphs;
                currentSize = overlapSize;
            }

            currentChunk.push(paragraph);
            currentSize += paragraphSize;
        }

        if (currentChunk.length > 0) {
            chunks.push(currentChunk.join('\n\n'));
        }

        return chunks;
    }
}

// Create singleton instance
const textProcessor = new TextProcessor();

export { textProcessor, TextProcessor };

export async function processFile(file) {
    return textProcessor.processFile(file);
}

export async function processFiles(files, onProgress) {
    return textProcessor.processMultipleFiles(files, onProgress);
}

export function cleanText(text) {
    return textProcessor.cleanText(text);
}

export function chunkText(text, method = 'sentences', maxSize = 800, overlap = 200) {
    if (method === 'sentences') {
        return textProcessor.chunkBySentences(text, maxSize, overlap);
    } else {
        return textProcessor.chunkByParagraphs(text, maxSize, overlap);
    }
}