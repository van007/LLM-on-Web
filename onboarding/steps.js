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
//   branch      'settings'  → primary action enters the SETTINGS_STEPS track (P9)
//   track       'settings'  → step belongs to the opt-in deep-dive track    (P9)
//   scrollIntoView true → engine scrolls the anchor into the sidebar's view (P9)

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
        branch: 'settings',
        eyebrow: 'MAKE IT YOURS',
        title: 'Tune and replay',
        body: 'That is the whole pipeline. Settings is where you choose models, shape replies, tune retrieval, and manage your files — want a quick tour of it? You can always replay either tour from the help button up here.',
    },
];

// Optional "Settings deep-dive" track (DESIGN_OVERHAUL_PLAN.md §10.2a). Reuses
// the same engine, card, and spotlight — it only adds steps + the finish-step
// branch above. Two rules shape it:
//   1. It never duplicates the in-place ⓘ info-icon tooltips already in the
//      sidebar; it teaches *where things live and why they matter*, then hands
//      off to those tooltips for parameter-level detail.
//   2. One representative anchor per `.settings-section`, using EXISTING ids /
//      classes only (no new ids on existing elements) — the engine scrolls each
//      into the sidebar's view before spotlighting.
// Every step carries `opensPanel: 'settings'` so the drawer is opened at S1 and
// stays pinned through S6; prior panel state is restored by finish().
const SETTINGS_STEPS = [
    {
        id: 's-open',
        target: '#settingsBtn',
        placement: 'bottom',
        opensPanel: 'settings',
        track: 'settings',
        eyebrow: 'SETTINGS',
        title: 'Everything lives here',
        body: 'This panel holds every control — model choice, sampling, retrieval, your documents, and maintenance. It opened from this button; close it the same way. Let us walk it top to bottom.',
    },
    {
        id: 's-models',
        target: '#backendSelect',
        placement: 'left',
        opensPanel: 'settings',
        track: 'settings',
        scrollIntoView: true,
        eyebrow: 'ENGINE',
        title: 'Backend & models',
        body: 'Pick the compute backend — WebGPU is fastest, WASM works everywhere, Auto-detect usually gets it right. Below it you choose the language, speech, and embedding models. Tap any ⓘ for the trade-offs.',
    },
    {
        id: 's-sampling',
        target: '#temperatureSlider',
        placement: 'left',
        opensPanel: 'settings',
        track: 'settings',
        scrollIntoView: true,
        eyebrow: 'SAMPLING',
        title: 'Shape the replies',
        body: 'Temperature, Top-p, and Max Tokens control how random, focused, and long responses get — the header readouts mirror them live. The ⓘ icons give exact ranges; nudge and watch the difference.',
    },
    {
        // `#ragToggle` itself is a visually-hidden checkbox (0×0); anchor the
        // visible "Enable RAG" row instead (unique existing class, no new id).
        id: 's-rag',
        target: '.toggle-label',
        placement: 'left',
        opensPanel: 'settings',
        track: 'settings',
        scrollIntoView: true,
        eyebrow: 'RETRIEVAL',
        title: 'Tune retrieval',
        body: 'Toggle RAG on or off, and set the similarity threshold — lower pulls in more passages, higher keeps only the closest matches. The document and chunk counts here track what is searchable.',
    },
    {
        id: 's-docs',
        target: '#documentList',
        placement: 'left',
        opensPanel: 'settings',
        track: 'settings',
        scrollIntoView: true,
        eyebrow: 'YOUR LIBRARY',
        title: 'Manage documents',
        body: 'Everything you upload is listed here, stored locally in the browser. Refresh re-reads the list; Clear All wipes every document from the vector store — it cannot be undone, so it asks first.',
    },
    {
        id: 's-actions',
        target: '.action-buttons',
        placement: 'left',
        opensPanel: 'settings',
        track: 'settings',
        scrollIntoView: true,
        setsFlag: true,
        eyebrow: 'MAINTENANCE',
        title: 'Chat & model actions',
        body: 'Clear Chat empties the conversation; Export Chat saves the whole thread (the per-message download is separate). Warm Up Model pre-loads it for an instant first reply, and Clear Model Cache frees storage — models re-download next run.',
    },
];

export default STEPS;
export { STEPS, SETTINGS_STEPS };
