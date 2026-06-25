// onboarding/steps.js
//
// Declarative step table for the first-launch guided tour. The engine
// (onboarding.js) reads this and never hardcodes DOM — content, anchors and
// orchestration hints all live here, so the flow can be re-authored without
// touching engine logic.
//
// `target` selectors are the EXISTING shell hooks, confirmed against
// index.html — no new ids are added to existing elements:
//   .status-pill        → header status pill (#statusPill)
//   .chat-input         → message composer (#chatInput)
//   #fileDropZone       → RAG upload zone, inside the settings sidebar
//   .tts-button         → per-message play button (created when a reply lands)
//   .download-chat-btn  → per-message "download chat (markdown)" button
//   #settingsBtn        → header settings/replay control cluster
//
// Orchestration / gating fields are declarative markers consumed across phases:
//   placement   'center' | 'top' | 'bottom' | 'left' | 'right'   (Phase 1–2)
//   opensPanel  'settings'  → engine opens the sidebar before anchoring (P2)
//   awaitState  'models-ready'  → step tracks an app state signal       (P3)
//   gateOn      'models-ready'  → step is soft-gated until ready         (P3)
//   jit         'assistant-message-complete' → deferred one-shot step    (P3)
//   setsFlag    true → finishing here writes the versioned onboarded flag (P6)

const STEPS = [
    {
        id: 'welcome',
        target: null,
        placement: 'center',
        eyebrow: '100% ON-DEVICE',
        title: 'Welcome to LLM on Web',
        body: 'A complete AI assistant that runs entirely in your browser — chat, document search, and speech, all on your device. Nothing you type or upload ever leaves it. This quick tour walks the whole pipeline. Take it now, or skip and explore.',
    },
    {
        id: 'models',
        target: '.status-pill',
        placement: 'bottom',
        awaitState: 'models-ready',
        blocking: false,
        eyebrow: 'STATUS',
        title: 'Models load here',
        body: 'On first run the language and embedding models download once, then cache for offline reuse. Watch this pill — its dot shifts from amber to green when everything is ready. The download runs in the background, so keep touring.',
    },
    {
        id: 'ask',
        target: '.chat-input',
        placement: 'top',
        gateOn: 'models-ready',
        eyebrow: 'CHAT',
        title: 'Ask anything',
        body: 'Type a message and press Enter to send. Replies stream in token by token, and you can stop generation at any time. If a model is still loading, your message sends the moment it is ready.',
    },
    {
        id: 'docs',
        target: '#fileDropZone',
        placement: 'left',
        opensPanel: 'settings',
        eyebrow: 'RETRIEVAL',
        title: 'Add your documents',
        body: 'Drop in TXT, Markdown, or PDF files and they are embedded locally for semantic search. With RAG enabled, answers are grounded in your own files — drawn from what you uploaded, never the open web.',
    },
    {
        id: 'rag-ask',
        target: '.chat-input',
        placement: 'top',
        eyebrow: 'GROUNDED ANSWERS',
        title: 'Ask about your files',
        body: 'Now ask a question and the assistant retrieves the most relevant passages from your documents before answering. Same composer — smarter, sourced replies.',
    },
    {
        id: 'listen',
        target: '.tts-button',
        placement: 'top',
        jit: 'assistant-message-complete',
        eyebrow: 'SPEECH',
        title: 'Hear it aloud',
        body: 'Every assistant reply gets a play button powered by on-device Kokoro TTS. Press it to listen; longer messages stream audio as they synthesise. Pause and resume whenever you like.',
    },
    {
        id: 'save',
        target: '.download-chat-btn',
        placement: 'top',
        jit: 'assistant-message-complete',
        eyebrow: 'EXPORT',
        title: 'Save your work',
        body: 'Download any conversation as Markdown, or save a spoken reply as a WAV file. Your work stays yours — exported straight from the browser, with no account required.',
    },
    {
        id: 'finish',
        target: '#settingsBtn',
        placement: 'bottom',
        setsFlag: true,
        eyebrow: 'MAKE IT YOURS',
        title: 'Tune and replay',
        body: 'Open Settings to choose models and adjust sampling, switch light and dark with the theme toggle, and replay this tour anytime from the help button up here. That is the whole pipeline — enjoy.',
    },
];

export default STEPS;
export { STEPS };
