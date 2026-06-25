// onboarding/onboarding.js
//
// First-launch guided tour — Phase 1: engine & overlay.
//
// Self-contained spotlight engine: a dimmed backdrop with a cutout over the
// *real* control being taught + an anchored hairline card. Pure CSS/JS, no
// dependency. The engine is declarative — it reads a step table and never
// hardcodes DOM. Later phases author the real step table (steps.js), wire
// state-awareness, JIT coachmarks, persistence and the header replay button.
//
// Scope guard: this is a NEW overlay layer only. It adds no ids to existing
// elements; it reads existing elements to anchor and (later) calls existing
// handlers. Nothing here mutates app.js / chat-ui.js / tts-ui.js logic.

import logger from '../utils/logger.js';

const ROOT_ID = 'onboarding-root';
const GAP = 12;          // px between target and card
const PAD = 6;           // px the spotlight cutout extends past the target
const EDGE = 16;         // min px between card and viewport edge

// Phase 1 hardcoded dummy steps — exist only to validate positioning in both
// themes and both placements (centered + anchored). Phase 2 replaces these by
// importing the real declarative table from ./steps.js.
const DUMMY_STEPS = [
    {
        id: 'welcome',
        target: null,
        placement: 'center',
        eyebrow: '100% ON-DEVICE',
        title: 'Welcome to LLM on Web',
        body: 'Everything runs in your browser — nothing leaves your device. This short tour shows you the full pipeline. (Phase 1 placeholder content.)',
    },
    {
        id: 'models',
        target: '.status-pill',
        placement: 'bottom',
        eyebrow: 'STATUS',
        title: 'Models load here',
        body: 'This pill shows the one-time model download and turns from warning to success when ready. Anchored-step positioning test.',
    },
    {
        id: 'compose',
        target: '.chat-input',
        placement: 'top',
        eyebrow: 'CHAT',
        title: 'Ask a question',
        body: 'The composer streams tokens as the model replies. Above-placement test (card sits over the input).',
    },
];

class OnboardingEngine {
    constructor() {
        this.steps = [];
        this.index = -1;
        this.mounted = false;
        this.active = false;
        this.lastFocus = null;
        this._raf = 0;

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
        this.counterEl = null;
        this.beakEl = null;
        this.backBtn = null;
        this.skipBtn = null;
        this.nextBtn = null;
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
                <div class="ob-actions">
                    <button type="button" class="ob-btn ob-skip" data-ob-skip>Skip</button>
                    <span class="ob-spacer"></span>
                    <button type="button" class="ob-btn ob-back" data-ob-back>Back</button>
                    <button type="button" class="ob-btn ob-next" data-ob-next>Next</button>
                </div>
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
        this.steps = Array.isArray(steps) && steps.length ? steps : DUMMY_STEPS;
        this.active = true;
        this.lastFocus = document.activeElement;
        this.root.hidden = false;

        window.addEventListener('resize', this._onReflow, { passive: true });
        window.addEventListener('scroll', this._onReflow, { passive: true, capture: true });
        document.addEventListener('keydown', this._onKeydown, true);

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
        this.nextBtn.textContent = index === this.steps.length - 1 ? 'Done' : 'Next';

        this._position(step);
        // Move focus into the dialog so keyboard nav works (full trap = Phase 5).
        this.card.focus({ preventScroll: true });
    }

    next() { this.show(this.index + 1); }
    back() { if (this.index > 0) this.show(this.index - 1); }
    skip() { this.finish(); }

    finish() {
        if (!this.active) return;
        this.active = false;
        this.root.hidden = true;

        window.removeEventListener('resize', this._onReflow);
        window.removeEventListener('scroll', this._onReflow, true);
        document.removeEventListener('keydown', this._onKeydown, true);

        if (this.lastFocus && typeof this.lastFocus.focus === 'function') {
            this.lastFocus.focus({ preventScroll: true });
        }
        logger.log('[Onboarding] finished');
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
        this.beakEl.classList.toggle('ob-beak--up', placeBelow);
        this.beakEl.classList.toggle('ob-beak--down', !placeBelow);
        this.beakEl.style.left = `${beakX}px`;
    }

    _handleKeydown(e) {
        if (!this.active) return;
        switch (e.key) {
            case 'Escape': e.preventDefault(); this.skip(); break;
            case 'Enter':
            case 'ArrowRight': e.preventDefault(); this.next(); break;
            case 'ArrowLeft': e.preventDefault(); this.back(); break;
        }
    }
}

const onboarding = new OnboardingEngine();

// Expose for manual launch during Phase-1 validation and for later wiring.
if (typeof window !== 'undefined') window.__onboarding = onboarding;

export default onboarding;
export { OnboardingEngine };
