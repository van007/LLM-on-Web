// onboarding/onboarding.js
//
// First-launch guided tour — engine & overlay (Phase 1) + step content,
// anchoring and panel orchestration (Phase 2) + state-awareness and JIT
// coachmarks (Phase 3) + accessibility (Phase 5) + persistence & restart
// (Phase 6).
//
// Accessibility (Phase 5): the card is a real modal dialog. Focus is moved
// into it on open and trapped there (Tab/Shift+Tab wrap inside the card) until
// the tour closes, then returned to the element that launched it. A polite,
// visually-hidden live region announces each step change to screen readers, so
// navigating the tour is conveyed without sight. Keyboard path is complete —
// Esc dismisses, Enter / → advance, ← goes back — and Enter is left to the
// focused action button when one is focused, so Back/Skip still work by keyboard.
//
// Self-contained spotlight engine: a dimmed backdrop with a cutout over the
// *real* control being taught + an anchored hairline card. Pure CSS/JS, no
// dependency. The engine is declarative — it reads ./steps.js and never
// hardcodes DOM.
//
// Persistence & restart (Phase 6): the tour remembers itself through a
// *versioned* localStorage flag (`llm-web-onboarded`) — set on finish AND skip
// (both route through finish()), so a dismissed tour won't reappear. Storing a
// version string rather than a bare boolean lets a future major release
// re-trigger a "what's new" pass without nagging existing users. A new header
// "replay tour" control (a NEW element, new id only) re-launches from step 1
// at any time, in both themes. `init()` boots the layer — install the button,
// then auto-start on a first/post-upgrade launch — and is the single thing the
// additive app.js call in Phase 7 will invoke. Nothing here edits app.js or
// index.html; the help button is created in JS, mirroring how the overlay root
// is built in mount().
//
// State-awareness (Phase 3): the engine *reads* the app's existing signals
// rather than re-running any pipeline. A MutationObserver on the status pill's
// `data-status`/text surfaces model-load progress to the `awaitState`/`gateOn`
// steps; a MutationObserver on the chat transcript detects the first completed
// assistant message (its `.tts-controls` row) to arm the `jit` steps, so the
// listen/save coachmarks never point at controls that don't exist yet.
//
// Scope guard: this is a NEW overlay layer only. It adds no ids to existing
// elements; it reads existing elements to anchor and calls existing handlers
// (e.g. clicks the real settings button to open the sidebar). Nothing here
// mutates app.js / chat-ui.js / tts-ui.js logic.

import logger from '../utils/logger.js';
import STEPS, { SETTINGS_STEPS } from './steps.js';

// Named tracks the engine can branch into (Phase 9). Keyed by the `branch` /
// `track` string used in steps.js — the core's finish step branches to
// 'settings', and the header help menu can start it standalone.
const TRACKS = { settings: SETTINGS_STEPS };

const ROOT_ID = 'onboarding-root';
const HELP_BTN_ID = 'ob-help-btn';            // new header "replay tour" control
const HELP_MENU_ID = 'ob-help-menu';          // its two-destination popover (P9)
const ONBOARDED_KEY = 'llm-web-onboarded';    // localStorage flag (versioned, not boolean)
const ONBOARDED_VERSION = '1';                // bump to re-trigger the tour on a major release
const GAP = 12;          // px between target and card
const PAD = 6;           // px the spotlight cutout extends past the target
const EDGE = 16;         // min px between card and viewport edge
const PANEL_ANIM = 220;  // ms fallback for the sidebar open transition (160ms token + margin)

class OnboardingEngine {
    constructor() {
        this.steps = [];
        this.index = -1;
        this.mounted = false;
        this.active = false;
        this.lastFocus = null;
        this._raf = 0;
        this._openedPanel = false;   // did *we* open the settings sidebar?

        // State-awareness (Phase 3): read, never re-run.
        this._modelState = 'loading';  // 'loading' | 'ready' | 'error'
        this._modelText = '';          // live status-pill text for progress copy
        this._messageReady = false;    // has an assistant message completed yet?
        this._pillObserver = null;     // watches #statusPill
        this._msgObserver = null;      // watches #chatTranscript for .tts-controls

        // Stable bound handlers so add/removeEventListener pair correctly.
        this._onReflow = this._scheduleReposition.bind(this);
        this._onKeydown = this._handleKeydown.bind(this);

        // DOM refs (filled in mount()).
        this.root = null;
        this.backdrop = null;
        this.spotlight = null;
        this.card = null;
        this.eyebrowEl = null;
        this.titleEl = null;
        this.bodyEl = null;
        this.statusEl = null;
        this.liveEl = null;
        this.counterEl = null;
        this.beakEl = null;
        this.backBtn = null;
        this.skipBtn = null;
        this.nextBtn = null;

        // Header help control + its two-destination menu (Phase 9).
        this.helpBtn = null;
        this.helpMenu = null;
        this._onHelpOutside = null;
        this._onHelpKeydown = null;
    }

    // Build the overlay DOM once. New ids only; no existing element touched.
    mount() {
        if (this.mounted) return;

        let root = document.getElementById(ROOT_ID);
        if (!root) {
            root = document.createElement('div');
            root.id = ROOT_ID;
        }
        root.className = 'ob-root';
        root.hidden = true;
        root.innerHTML = `
            <div class="ob-backdrop" data-ob-dismiss></div>
            <div class="ob-spotlight" aria-hidden="true"></div>
            <div class="ob-card" role="dialog" aria-modal="true"
                 aria-labelledby="ob-title" aria-describedby="ob-body" tabindex="-1">
                <span class="ob-beak" aria-hidden="true"></span>
                <p class="ob-counter" id="ob-counter"></p>
                <p class="ob-eyebrow"></p>
                <h2 class="ob-title" id="ob-title"></h2>
                <p class="ob-body" id="ob-body"></p>
                <p class="ob-status" id="ob-status" role="status" aria-live="polite" hidden></p>
                <div class="ob-actions">
                    <button type="button" class="ob-btn ob-skip" data-ob-skip>Skip</button>
                    <span class="ob-spacer"></span>
                    <button type="button" class="ob-btn ob-back" data-ob-back>Back</button>
                    <button type="button" class="ob-btn ob-next" data-ob-next>Next</button>
                </div>
                <div class="ob-live visually-hidden" aria-live="polite" aria-atomic="true"></div>
            </div>
        `;

        if (!root.parentNode) document.body.appendChild(root);

        this.root = root;
        this.backdrop = root.querySelector('.ob-backdrop');
        this.spotlight = root.querySelector('.ob-spotlight');
        this.card = root.querySelector('.ob-card');
        this.beakEl = root.querySelector('.ob-beak');
        this.counterEl = root.querySelector('.ob-counter');
        this.eyebrowEl = root.querySelector('.ob-eyebrow');
        this.titleEl = root.querySelector('.ob-title');
        this.bodyEl = root.querySelector('.ob-body');
        this.statusEl = root.querySelector('.ob-status');
        this.liveEl = root.querySelector('.ob-live');
        this.backBtn = root.querySelector('[data-ob-back]');
        this.skipBtn = root.querySelector('[data-ob-skip]');
        this.nextBtn = root.querySelector('[data-ob-next]');

        this.nextBtn.addEventListener('click', () => this.next());
        this.backBtn.addEventListener('click', () => this.back());
        this.skipBtn.addEventListener('click', () => this.skip());
        this.backdrop.addEventListener('click', () => this.skip());

        this.mounted = true;
        logger.log('[Onboarding] mounted');
    }

    // Launch the tour from a given index.
    start(steps, index = 0) {
        this.mount();
        this.steps = Array.isArray(steps) && steps.length ? steps : STEPS;
        this.active = true;
        this.lastFocus = document.activeElement;
        this.root.hidden = false;

        window.addEventListener('resize', this._onReflow, { passive: true });
        window.addEventListener('scroll', this._onReflow, { passive: true, capture: true });
        document.addEventListener('keydown', this._onKeydown, true);

        // Begin reading app state so gated/JIT steps reflect reality.
        this._watchModelState();
        this._watchMessages();

        this.show(index);
    }

    show(index) {
        if (index < 0 || index >= this.steps.length) return this.finish();
        this.index = index;
        const step = this.steps[index];

        this.counterEl.textContent =
            `STEP ${String(index + 1).padStart(2, '0')} / ${String(this.steps.length).padStart(2, '0')}`;
        this.eyebrowEl.textContent = step.eyebrow || '';
        this.eyebrowEl.hidden = !step.eyebrow;
        this.titleEl.textContent = step.title || '';
        this.bodyEl.textContent = step.body || '';

        this.backBtn.disabled = index === 0;
        // A branch step offers two destinations: the primary button enters the
        // named track, and "Skip" reads as "Finish" (declining still finishes).
        // Otherwise the last step's primary button reads "Done".
        const isBranch = !!(step.branch && TRACKS[step.branch]);
        this.nextBtn.textContent = isBranch
            ? 'Tour settings'
            : (index === this.steps.length - 1 ? 'Done' : 'Next');
        this.skipBtn.textContent = isBranch ? 'Finish' : 'Skip';

        // State-awareness: reflect live model-load / JIT readiness in the card.
        this._updateStatusNote(step);

        // Panel orchestration: open the real sidebar for steps that anchor
        // inside it, and restore it when we move on.
        this._syncPanel(step);
        // Bring an anchored target into view (it may live in a scroll container).
        this._revealTarget(step);

        this._position(step);
        // Announce the step to screen readers (the dialog stays mounted across
        // steps, so re-focusing it won't re-read title/body — the live region is
        // the reliable channel for step changes).
        this._announce(step);
        // Move focus into the dialog; Tab is trapped inside it (see _handleKeydown).
        this.card.focus({ preventScroll: true });
    }

    // Push the current step into the polite live region so assistive tech hears
    // each change. Cleared first so an identical re-render (e.g. a JIT upgrade)
    // still re-announces.
    _announce(step) {
        if (!this.liveEl) return;
        const heading = `Step ${this.index + 1} of ${this.steps.length}.`;
        const parts = [heading, step.title, step.body].filter(Boolean);
        this.liveEl.textContent = '';
        this.liveEl.textContent = parts.join(' ');
    }

    // Open/close the settings sidebar to match the current step, using the
    // EXISTING settings-button handler — never reimplementing it. We only ever
    // close a panel that *we* opened, so a user-opened panel is left intact.
    _syncPanel(step) {
        const sidebar = document.getElementById('sidebar');
        if (!sidebar) return;
        const settingsBtn = document.getElementById('settingsBtn');
        const needsPanel = step && step.opensPanel === 'settings';
        const isOpen = sidebar.classList.contains('open');

        if (needsPanel && !isOpen) {
            if (settingsBtn) settingsBtn.click();
            else sidebar.classList.add('open');
            this._openedPanel = true;
            // The drawer slides in (transform); reposition once it has settled
            // so the spotlight and card track the final geometry.
            const settle = () => {
                if (!this.active) return;
                this._revealTarget(step);
                this._position(step);
            };
            sidebar.addEventListener('transitionend', settle, { once: true });
            setTimeout(settle, PANEL_ANIM);
        } else if (!needsPanel && this._openedPanel && isOpen) {
            if (settingsBtn) settingsBtn.click();
            else sidebar.classList.remove('open');
            this._openedPanel = false;
        }
    }

    // Scroll an anchored target into view within its nearest scroll container
    // (e.g. the file drop zone deep inside the settings panel, or the lower
    // controls in the deep-dive track's scrolling `.sidebar-content`). The
    // scroll is smooth, or instant under prefers-reduced-motion. Smooth scroll
    // fires `scroll` events the capture listener already repositions on; the
    // trailing timeout is a safety recompute once the scroll has settled.
    _revealTarget(step) {
        if (!step || !step.target) return;
        const el = document.querySelector(step.target);
        if (!el || typeof el.scrollIntoView !== 'function') return;
        const reduce = window.matchMedia &&
            window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        el.scrollIntoView({
            block: 'center', inline: 'nearest',
            behavior: reduce ? 'auto' : 'smooth',
        });
        this._scheduleReposition();
        setTimeout(() => {
            if (this.active && this.index >= 0) this._position(this.steps[this.index]);
        }, PANEL_ANIM);
    }

    // --- state-awareness & JIT (Phase 3) ------------------------------------

    // Read the status pill's machine state + live text. We never re-run any
    // pipeline — we observe the very signal the pill itself reflects:
    //   data-status: 'loading' → 'ready' (also 'generating'/'processing' once
    //   models are loaded) | 'error'. Absent (initial markup) counts as loading.
    _watchModelState() {
        const pill = document.getElementById('statusPill');
        if (!pill) { this._modelState = 'loading'; return; }

        const read = () => {
            const s = pill.getAttribute('data-status');
            if (s === 'error') this._modelState = 'error';
            else if (s && s !== 'loading') this._modelState = 'ready';
            else this._modelState = 'loading';
            const textEl = pill.querySelector('.status-text');
            this._modelText = (textEl ? textEl.textContent : '').trim();
            this._onStateChange();
        };

        read();
        this._pillObserver = new MutationObserver(read);
        // Watch the state attribute and the live progress text.
        this._pillObserver.observe(pill, {
            attributes: true, attributeFilter: ['data-status'],
            childList: true, characterData: true, subtree: true,
        });
    }

    // Detect the first completed assistant message. The TTS control row
    // (.tts-controls, holding .tts-button / .download-chat-btn) is appended only
    // when a reply finishes, so its appearance is our "assistant-message-complete"
    // signal — the one-shot trigger that arms the listen/save coachmarks.
    _watchMessages() {
        const transcript = document.getElementById('chatTranscript');
        if (!transcript) return;

        const hasControls = () => !!transcript.querySelector('.tts-controls, .tts-button');
        this._messageReady = hasControls();

        this._msgObserver = new MutationObserver(() => {
            if (!this._messageReady && hasControls()) {
                this._messageReady = true;
                this._onMessageComplete();
            }
        });
        this._msgObserver.observe(transcript, { childList: true, subtree: true });
    }

    // Does the current step's anchor exist and have layout right now?
    _targetExists(step) {
        return !!this._resolveTarget(step);
    }

    // A jit step whose control hasn't appeared yet: shown as a "deferred
    // preview" (centered, no broken spotlight) until the trigger fires.
    _isJitDeferred(step) {
        return !!(step && step.jit && !this._targetExists(step));
    }

    // Surface live readiness in the card's status line for gated/JIT steps.
    _updateStatusNote(step) {
        let msg = '';
        if (step && (step.awaitState === 'models-ready' || step.gateOn === 'models-ready')) {
            if (this._modelState === 'error') {
                msg = 'Model load failed — check your connection, then reload.';
            } else if (this._modelState === 'ready') {
                msg = step.gateOn
                    ? '✓ Model ready — type a message and press Enter.'
                    : `✓ ${this._modelText || 'Models ready'}`;
            } else {
                msg = step.gateOn
                    ? 'Waiting for the model to finish loading… you can keep touring.'
                    : (this._modelText || 'Downloading models…');
            }
        } else if (this._isJitDeferred(step)) {
            msg = 'Appears on your first reply — the play and download buttons live on each assistant message.';
        }

        this.statusEl.textContent = msg;
        this.statusEl.hidden = !msg;
    }

    // Model state changed → refresh the live note (and keep the gentle gate in
    // sync). Never blocks; the step is always skippable.
    _onStateChange() {
        if (!this.active || this.index < 0) return;
        this._updateStatusNote(this.steps[this.index]);
    }

    // First assistant reply completed → if we're parked on a deferred JIT step,
    // upgrade it in place from centered preview to a real anchored coachmark.
    _onMessageComplete() {
        if (!this.active || this.index < 0) return;
        const step = this.steps[this.index];
        if (step && step.jit) this.show(this.index);
    }

    next() {
        // On a branch step the primary action enters the named track rather
        // than finishing — e.g. the core's finish step → the settings deep-dive.
        const step = this.steps[this.index];
        if (step && step.branch && TRACKS[step.branch]) {
            this._enterTrack(step.branch);
            return;
        }
        this.show(this.index + 1);
    }
    back() { if (this.index > 0) this.show(this.index - 1); }
    skip() { this.finish(); }

    // Swap the running tour onto another declarative track (Phase 9), starting
    // at its first step. The active session — observers, listeners, lastFocus —
    // is reused; only the step array changes, so no re-`start()`. The track's
    // own final step carries `setsFlag`, and finish() restores any opened panel.
    _enterTrack(name) {
        const steps = TRACKS[name];
        if (!Array.isArray(steps) || !steps.length) return this.finish();
        this.steps = steps;
        this.show(0);
    }

    finish() {
        if (!this.active) return;
        this.active = false;
        this.root.hidden = true;

        // Remember the tour was seen (finish OR skip → both land here) so it
        // won't auto-start again until the onboarding version bumps.
        this._markOnboarded();

        window.removeEventListener('resize', this._onReflow);
        window.removeEventListener('scroll', this._onReflow, true);
        document.removeEventListener('keydown', this._onKeydown, true);

        // Stop reading app state.
        if (this._pillObserver) { this._pillObserver.disconnect(); this._pillObserver = null; }
        if (this._msgObserver) { this._msgObserver.disconnect(); this._msgObserver = null; }

        // Restore the sidebar if the tour opened it.
        if (this._openedPanel) {
            const sidebar = document.getElementById('sidebar');
            const settingsBtn = document.getElementById('settingsBtn');
            if (sidebar && sidebar.classList.contains('open')) {
                if (settingsBtn) settingsBtn.click();
                else sidebar.classList.remove('open');
            }
            this._openedPanel = false;
        }

        if (this.lastFocus && typeof this.lastFocus.focus === 'function') {
            this.lastFocus.focus({ preventScroll: true });
        }
        logger.log('[Onboarding] finished');
    }

    // --- persistence & restart (Phase 6) ------------------------------------

    // The stored onboarded version string, or null if never completed / storage
    // is unavailable (private mode, blocked) — both treated as "not onboarded".
    _storedVersion() {
        try { return window.localStorage.getItem(ONBOARDED_KEY); }
        catch (_) { return null; }
    }

    // Persist that the tour has been seen at the current version.
    _markOnboarded() {
        try { window.localStorage.setItem(ONBOARDED_KEY, ONBOARDED_VERSION); }
        catch (_) { /* storage unavailable — tour may re-show next launch; acceptable */ }
    }

    // True when the tour should auto-start: never completed, or the stored
    // version predates the current one (post-upgrade "what's new" re-trigger).
    shouldAutoStart() {
        return this._storedVersion() !== ONBOARDED_VERSION;
    }

    // Clear the flag — for dev/testing and any future "reset" affordance.
    resetOnboarded() {
        try { window.localStorage.removeItem(ONBOARDED_KEY); }
        catch (_) { /* no-op */ }
    }

    // Replay the whole tour from step 1, regardless of the stored flag. Tears
    // down a running tour first so the header control doubles as a restart.
    replay() {
        if (this.active) this.finish();
        this.start(STEPS, 0);
    }

    // Launch the Settings deep-dive track standalone (Phase 9), skipping the
    // core pipeline tour — the header help menu's second destination. Like
    // replay(), it runs regardless of the stored flag and tears down any
    // in-progress tour first.
    replaySettings() {
        if (this.active) this.finish();
        this.start(SETTINGS_STEPS, 0);
    }

    // Boot the onboarding layer: install the header replay control, then
    // auto-start on a first (or post-upgrade) launch. Additive + idempotent —
    // the single additive app.js call in Phase 7 will invoke this once after
    // the shell mounts.
    init() {
        this.installHelpButton();
        if (this.shouldAutoStart()) this.start(STEPS, 0);
    }

    // Create the header help control as a NEW element (new ids only; no existing
    // element is touched). Reuses the shell's .icon-btn treatment so it matches
    // the theme/settings icons in both themes. The button opens a small popover
    // menu with two destinations — replay the core pipeline tour, or jump
    // straight into the Settings deep-dive track (Phase 9). Idempotent.
    installHelpButton() {
        if (document.getElementById(HELP_BTN_ID)) return document.getElementById(HELP_BTN_ID);
        const header = document.querySelector('.app-header');
        if (!header) return null;

        // Wrapper anchors the absolutely-positioned menu beneath the button
        // without disturbing the header's flex layout.
        const wrap = document.createElement('div');
        wrap.className = 'ob-help';

        const btn = document.createElement('button');
        btn.id = HELP_BTN_ID;
        btn.type = 'button';
        btn.className = 'icon-btn ob-help-btn';
        btn.setAttribute('aria-label', 'Tour & help');
        btn.setAttribute('aria-haspopup', 'true');
        btn.setAttribute('aria-expanded', 'false');
        btn.title = 'Tour & help';
        btn.innerHTML = `<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M9.09 9a3 3 0 0 1 5.82 1c0 2-3 3-3 3"></path>
                <line x1="12" y1="17" x2="12.01" y2="17"></line>
            </svg>`;

        const menu = document.createElement('div');
        menu.id = HELP_MENU_ID;
        menu.className = 'ob-help-menu';
        menu.setAttribute('role', 'menu');
        menu.setAttribute('aria-label', 'Tour & help');
        menu.hidden = true;
        menu.innerHTML = `
            <button type="button" class="ob-help-item" role="menuitem" data-ob-tour="core">Replay tour</button>
            <button type="button" class="ob-help-item" role="menuitem" data-ob-tour="settings">Tour the settings</button>
        `;

        this.helpBtn = btn;
        this.helpMenu = menu;
        // Bound handlers so the document-level listeners pair correctly.
        this._onHelpOutside = this._handleHelpOutside.bind(this);
        this._onHelpKeydown = this._handleHelpKeydown.bind(this);

        btn.addEventListener('click', (e) => { e.stopPropagation(); this._toggleHelpMenu(); });
        menu.addEventListener('click', (e) => {
            const item = e.target.closest('[data-ob-tour]');
            if (!item) return;
            const dest = item.getAttribute('data-ob-tour');
            // Restore focus to the button first so the tour captures it as the
            // return target and finish() lands focus back in the header.
            this._closeHelpMenu(true);
            if (dest === 'settings') this.replaySettings();
            else this.replay();
        });

        wrap.appendChild(btn);
        wrap.appendChild(menu);

        // First in the control cluster, before the theme toggle.
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) header.insertBefore(wrap, themeToggle);
        else header.appendChild(wrap);
        return btn;
    }

    _toggleHelpMenu() {
        if (this.helpMenu && this.helpMenu.hidden) this._openHelpMenu();
        else this._closeHelpMenu();
    }

    _openHelpMenu() {
        if (!this.helpMenu) return;
        this.helpMenu.hidden = false;
        this.helpBtn.setAttribute('aria-expanded', 'true');
        document.addEventListener('click', this._onHelpOutside, true);
        document.addEventListener('keydown', this._onHelpKeydown, true);
        const first = this.helpMenu.querySelector('[data-ob-tour]');
        if (first) first.focus();
    }

    _closeHelpMenu(restoreFocus = false) {
        if (!this.helpMenu || this.helpMenu.hidden) return;
        this.helpMenu.hidden = true;
        this.helpBtn.setAttribute('aria-expanded', 'false');
        document.removeEventListener('click', this._onHelpOutside, true);
        document.removeEventListener('keydown', this._onHelpKeydown, true);
        if (restoreFocus && this.helpBtn) this.helpBtn.focus();
    }

    // Click anywhere outside the open menu closes it.
    _handleHelpOutside(e) {
        if (this.helpMenu && !this.helpMenu.contains(e.target) && e.target !== this.helpBtn) {
            this._closeHelpMenu();
        }
    }

    // Keyboard for the open menu: Esc closes (returning focus to the button),
    // arrows move between the two items, Tab is left to the browser.
    _handleHelpKeydown(e) {
        if (!this.helpMenu || this.helpMenu.hidden) return;
        const items = Array.from(this.helpMenu.querySelectorAll('[data-ob-tour]'));
        if (!items.length) return;
        const i = items.indexOf(document.activeElement);
        switch (e.key) {
            case 'Escape':
                e.preventDefault(); this._closeHelpMenu(true); break;
            case 'ArrowDown':
                e.preventDefault(); items[(i + 1 + items.length) % items.length].focus(); break;
            case 'ArrowUp':
                e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); break;
        }
    }

    // --- geometry -----------------------------------------------------------

    _scheduleReposition() {
        if (this._raf) return;
        this._raf = requestAnimationFrame(() => {
            this._raf = 0;
            if (this.active) this._position(this.steps[this.index]);
        });
    }

    _resolveTarget(step) {
        if (!step || !step.target) return null;
        const el = document.querySelector(step.target);
        if (!el) return null;
        const r = el.getBoundingClientRect();
        // Skip invisible/zero-size anchors — caller falls back to centered.
        if (r.width === 0 && r.height === 0) return null;
        return r;
    }

    _position(step) {
        const rect = this._resolveTarget(step);
        // At mobile width the card is a CSS bottom sheet — never position it
        // inline (inline left/top would override the media-query rule).
        const isMobile = window.innerWidth <= 768;

        if (!rect || (step && step.placement === 'center')) {
            // Centered step: full dim, no cutout, card centered.
            this.backdrop.hidden = false;
            this.spotlight.hidden = true;
            this.beakEl.hidden = true;
            this.card.classList.add('ob-card--center');
            this.card.style.left = '';
            this.card.style.top = '';
            return;
        }

        // Anchored step: cutout backdrop via spotlight box-shadow.
        this.backdrop.hidden = true;
        this.spotlight.hidden = false;
        this.card.classList.remove('ob-card--center');

        const sx = rect.left - PAD;
        const sy = rect.top - PAD;
        const sw = rect.width + PAD * 2;
        const sh = rect.height + PAD * 2;
        Object.assign(this.spotlight.style, {
            left: `${sx}px`, top: `${sy}px`, width: `${sw}px`, height: `${sh}px`,
        });

        if (isMobile) {
            // Card sits as a bottom sheet; clear inline geometry so CSS wins.
            this.card.style.left = '';
            this.card.style.top = '';
            this.beakEl.hidden = true;
            return;
        }

        // Card geometry. Measure after content is set.
        const cw = this.card.offsetWidth;
        const ch = this.card.offsetHeight;
        const vw = window.innerWidth;
        const vh = window.innerHeight;

        // Side placement: card beside the target (used by the docs step, which
        // anchors inside the right-docked settings panel). Flips side if the
        // requested one lacks room.
        if (step.placement === 'left' || step.placement === 'right') {
            const spaceLeft = rect.left;
            const spaceRight = vw - rect.right;
            let placeLeft = step.placement === 'left';
            if (placeLeft && spaceLeft < cw + GAP + EDGE && spaceRight > spaceLeft) placeLeft = false;
            if (!placeLeft && spaceRight < cw + GAP + EDGE && spaceLeft > spaceRight) placeLeft = true;

            let left = placeLeft ? rect.left - GAP - cw : rect.right + GAP;
            let top = rect.top + rect.height / 2 - ch / 2;
            left = Math.max(EDGE, Math.min(left, vw - cw - EDGE));
            top = Math.max(EDGE, Math.min(top, vh - ch - EDGE));
            Object.assign(this.card.style, { left: `${left}px`, top: `${top}px` });

            // Card left of target → beak on the card's right edge, and vice versa.
            const beakY = Math.max(12, Math.min(rect.top + rect.height / 2 - top, ch - 12));
            this.beakEl.hidden = false;
            this._setBeak(placeLeft ? 'right' : 'left');
            this.beakEl.style.left = '';
            this.beakEl.style.top = `${beakY}px`;
            return;
        }

        const spaceBelow = vh - rect.bottom;
        const spaceAbove = rect.top;
        let placeBelow = step.placement !== 'top';
        if (placeBelow && spaceBelow < ch + GAP + EDGE && spaceAbove > spaceBelow) placeBelow = false;
        if (!placeBelow && spaceAbove < ch + GAP + EDGE && spaceBelow > spaceAbove) placeBelow = true;

        let top = placeBelow ? rect.bottom + GAP : rect.top - GAP - ch;
        let left = rect.left + rect.width / 2 - cw / 2;
        left = Math.max(EDGE, Math.min(left, vw - cw - EDGE));
        top = Math.max(EDGE, Math.min(top, vh - ch - EDGE));

        Object.assign(this.card.style, { left: `${left}px`, top: `${top}px` });

        // Beak points at the target from the card edge.
        const beakX = Math.max(12, Math.min(rect.left + rect.width / 2 - left, cw - 12));
        this.beakEl.hidden = false;
        this._setBeak(placeBelow ? 'up' : 'down');
        this.beakEl.style.top = '';
        this.beakEl.style.left = `${beakX}px`;
    }

    // Set the beak orientation, clearing the other three variants.
    _setBeak(dir) {
        const b = this.beakEl;
        b.classList.toggle('ob-beak--up', dir === 'up');
        b.classList.toggle('ob-beak--down', dir === 'down');
        b.classList.toggle('ob-beak--left', dir === 'left');
        b.classList.toggle('ob-beak--right', dir === 'right');
    }

    _handleKeydown(e) {
        if (!this.active) return;
        switch (e.key) {
            case 'Escape': e.preventDefault(); this.skip(); break;
            case 'Tab': this._trapFocus(e); break;
            case 'Enter':
                // Let the focused action button handle its own activation, so
                // Enter on Back/Skip does the expected thing rather than Next.
                if (this._isOwnButton(e.target)) return;
                e.preventDefault(); this.next(); break;
            case 'ArrowRight': e.preventDefault(); this.next(); break;
            case 'ArrowLeft': e.preventDefault(); this.back(); break;
        }
    }

    // Is the event target one of the card's own action buttons?
    _isOwnButton(el) {
        return el === this.nextBtn || el === this.backBtn || el === this.skipBtn;
    }

    // Visible, enabled, focusable controls inside the card, in DOM order.
    _focusable() {
        const sel = 'button:not([disabled]), [href], input:not([disabled]),' +
                    ' select:not([disabled]), textarea:not([disabled]),' +
                    ' [tabindex]:not([tabindex="-1"])';
        return Array.from(this.card.querySelectorAll(sel))
            .filter((el) => el.offsetParent !== null || el.getClientRects().length);
    }

    // Keep Tab focus cycling within the card — the modal focus trap.
    _trapFocus(e) {
        const items = this._focusable();
        if (!items.length) { e.preventDefault(); this.card.focus({ preventScroll: true }); return; }
        const first = items[0];
        const last = items[items.length - 1];
        const active = document.activeElement;
        const inside = this.card.contains(active) && active !== this.card;
        if (e.shiftKey) {
            if (!inside || active === first) { e.preventDefault(); last.focus(); }
        } else if (!inside || active === last) {
            e.preventDefault(); first.focus();
        }
    }
}

const onboarding = new OnboardingEngine();

// Expose for manual launch during Phase-1 validation and for later wiring.
if (typeof window !== 'undefined') window.__onboarding = onboarding;

export default onboarding;
export { OnboardingEngine };
