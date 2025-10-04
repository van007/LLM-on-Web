/**
 * Centralized logging utility for the PWA
 * Change LOGGING_ENABLED to false to disable all console output
 */

// ============================================
// CHANGE THIS TO CONTROL ALL LOGGING
// ============================================
const LOGGING_ENABLED = true;

// Create no-op function for disabled logging
const noop = () => {};

// Logger object with all console methods
const logger = {
    log: LOGGING_ENABLED ? console.log.bind(console) : noop,
    error: LOGGING_ENABLED ? console.error.bind(console) : noop,
    warn: LOGGING_ENABLED ? console.warn.bind(console) : noop,
    info: LOGGING_ENABLED ? console.info.bind(console) : noop,
    debug: LOGGING_ENABLED ? console.debug.bind(console) : noop,

    // Additional utility to check if logging is enabled
    isEnabled: () => LOGGING_ENABLED
};

export default logger;