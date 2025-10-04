// Note: Service workers cannot use ES6 modules, so we inline the logger functionality
const LOGGING_ENABLED = true;  // Control logging from here for service worker
const logger = {
    log: LOGGING_ENABLED ? console.log.bind(console) : () => {},
    error: LOGGING_ENABLED ? console.error.bind(console) : () => {},
    warn: LOGGING_ENABLED ? console.warn.bind(console) : () => {},
    info: LOGGING_ENABLED ? console.info.bind(console) : () => {},
    debug: LOGGING_ENABLED ? console.debug.bind(console) : () => {}
};

const VERSION = '0.1.0';
const CACHE_NAME = `llm-web-v${VERSION}`;
const APP_SHELL_FILES = [
    '/',
    '/index.html',
    '/styles.css',
    '/app.js',
    '/manifest.webmanifest',
    '/assets/icons/icon-192.png',
    '/assets/icons/icon-512.png'
];

const RUNTIME_CACHE_NAME = `llm-web-runtime-v${VERSION}`;

self.addEventListener('install', (event) => {
    logger.log('[ServiceWorker] Install');

    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            logger.log('[ServiceWorker] Caching app shell');
            return cache.addAll(APP_SHELL_FILES.map(url => {
                return new Request(url, { cache: 'reload' });
            }));
        }).then(() => {
            logger.log('[ServiceWorker] Skip waiting');
            return self.skipWaiting();
        })
    );
});

self.addEventListener('activate', (event) => {
    logger.log('[ServiceWorker] Activate');

    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME && cacheName !== RUNTIME_CACHE_NAME &&
                        !cacheName.includes('transformers-cache') &&
                        !cacheName.includes('tts-models-cache')) {
                        logger.log('[ServiceWorker] Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => {
            logger.log('[ServiceWorker] Claiming clients');
            return self.clients.claim();
        })
    );
});

self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);

    // Cache Google Fonts
    if (url.origin === 'https://fonts.googleapis.com' || url.origin === 'https://fonts.gstatic.com') {
        event.respondWith(
            caches.open(RUNTIME_CACHE_NAME).then((cache) => {
                return cache.match(request).then((cachedResponse) => {
                    if (cachedResponse) {
                        return cachedResponse;
                    }
                    return fetch(request).then((response) => {
                        if (response && response.status === 200) {
                            cache.put(request, response.clone());
                        }
                        return response;
                    });
                });
            })
        );
        return;
    }

    // Cache kokoro-js library and dependencies
    if (url.origin === 'https://cdn.jsdelivr.net' && url.pathname.includes('kokoro-js')) {
        event.respondWith(
            caches.open(RUNTIME_CACHE_NAME).then((cache) => {
                return cache.match(request).then((cachedResponse) => {
                    if (cachedResponse) {
                        return cachedResponse;
                    }
                    return fetch(request).then((response) => {
                        if (response && response.status === 200) {
                            cache.put(request, response.clone());
                        }
                        return response;
                    });
                });
            })
        );
        return;
    }

    if (url.origin === 'https://cdn.jsdelivr.net' && url.pathname.includes('@huggingface/transformers')) {
        return;
    }

    // Cache ONNX models for both LLM and TTS
    if (url.origin === 'https://huggingface.co' || url.origin === 'https://cdn-lfs.huggingface.co') {
        // Skip caching for non-ONNX model files, but cache Kokoro TTS models
        if (url.pathname.includes('Kokoro-82M')) {
            event.respondWith(
                caches.open('tts-models-cache').then((cache) => {
                    return cache.match(request).then((cachedResponse) => {
                        if (cachedResponse) {
                            return cachedResponse;
                        }
                        return fetch(request).then((response) => {
                            if (response && response.status === 200 && response.headers.get('content-length') < 500000000) {
                                cache.put(request, response.clone());
                            }
                            return response;
                        });
                    });
                })
            );
        }
        return;
    }

    if (request.method !== 'GET') {
        return;
    }

    event.respondWith(
        caches.match(request).then((cachedResponse) => {
            if (cachedResponse) {
                return cachedResponse;
            }

            if (url.origin === location.origin) {
                return caches.open(RUNTIME_CACHE_NAME).then((cache) => {
                    return fetch(request).then((response) => {
                        if (response && response.status === 200) {
                            cache.put(request, response.clone());
                        }
                        return response;
                    }).catch(() => {
                        return caches.match('/index.html');
                    });
                });
            }

            return fetch(request);
        })
    );
});

self.addEventListener('message', (event) => {
    if (event.data && event.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }

    if (event.data && event.data.type === 'CHECK_UPDATE') {
        event.waitUntil(
            caches.open(CACHE_NAME).then(async (cache) => {
                return cache.match('/index.html').then((response) => {
                    if (response) {
                        return fetch('/index.html', { cache: 'no-cache' }).then((fetchResponse) => {
                            if (fetchResponse && fetchResponse.status === 200) {
                                const etag = fetchResponse.headers.get('etag');
                                const lastModified = fetchResponse.headers.get('last-modified');
                                const cachedEtag = response.headers.get('etag');
                                const cachedLastModified = response.headers.get('last-modified');

                                if ((etag && etag !== cachedEtag) ||
                                    (lastModified && lastModified !== cachedLastModified)) {
                                    event.ports[0].postMessage({ updateAvailable: true });
                                } else {
                                    event.ports[0].postMessage({ updateAvailable: false });
                                }
                            }
                        });
                    }
                });
            })
        );
    }
});

const broadcast = (message) => {
    self.clients.matchAll({ type: 'window' }).then((clients) => {
        clients.forEach((client) => {
            client.postMessage(message);
        });
    });
};

const notifyUpdate = () => {
    broadcast({
        type: 'UPDATE_AVAILABLE',
        message: 'A new version of the app is available. Refresh to update.'
    });
};