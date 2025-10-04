// IndexedDB helper utilities for managing local storage

class IDBHelper {
    constructor(dbName, version = 1) {
        this.dbName = dbName;
        this.version = version;
        this.db = null;
    }

    async open(upgradeCallback) {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.version);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event) => {
                this.db = event.target.result;
                if (upgradeCallback) {
                    upgradeCallback(this.db, event.oldVersion, event.newVersion);
                }
            };
        });
    }

    async transaction(storeNames, mode = 'readonly') {
        if (!this.db) {
            throw new Error('Database not opened');
        }
        return this.db.transaction(storeNames, mode);
    }

    async getStore(storeName, mode = 'readonly') {
        const tx = await this.transaction([storeName], mode);
        return tx.objectStore(storeName);
    }

    async get(storeName, key) {
        return new Promise(async (resolve, reject) => {
            const store = await this.getStore(storeName, 'readonly');
            const request = store.get(key);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getAll(storeName, query, count) {
        return new Promise(async (resolve, reject) => {
            const store = await this.getStore(storeName, 'readonly');
            const request = store.getAll(query, count);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getAllKeys(storeName, query, count) {
        return new Promise(async (resolve, reject) => {
            const store = await this.getStore(storeName, 'readonly');
            const request = store.getAllKeys(query, count);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async put(storeName, value, key) {
        return new Promise(async (resolve, reject) => {
            const store = await this.getStore(storeName, 'readwrite');
            const request = key !== undefined ? store.put(value, key) : store.put(value);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async add(storeName, value, key) {
        return new Promise(async (resolve, reject) => {
            const store = await this.getStore(storeName, 'readwrite');
            const request = key !== undefined ? store.add(value, key) : store.add(value);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async delete(storeName, key) {
        return new Promise(async (resolve, reject) => {
            const store = await this.getStore(storeName, 'readwrite');
            const request = store.delete(key);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    async clear(storeName) {
        return new Promise(async (resolve, reject) => {
            const store = await this.getStore(storeName, 'readwrite');
            const request = store.clear();
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }

    async count(storeName) {
        return new Promise(async (resolve, reject) => {
            const store = await this.getStore(storeName, 'readonly');
            const request = store.count();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async iterate(storeName, callback) {
        return new Promise(async (resolve, reject) => {
            const store = await this.getStore(storeName, 'readonly');
            const request = store.openCursor();
            const results = [];

            request.onsuccess = (event) => {
                const cursor = event.target.result;
                if (cursor) {
                    const result = callback(cursor.value, cursor.key);
                    if (result === false) {
                        resolve(results);
                    } else {
                        results.push(result);
                        cursor.continue();
                    }
                } else {
                    resolve(results);
                }
            };

            request.onerror = () => reject(request.error);
        });
    }

    async deleteDatabase() {
        if (this.db) {
            this.db.close();
            this.db = null;
        }
        return new Promise((resolve, reject) => {
            const request = indexedDB.deleteDatabase(this.dbName);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
            request.onblocked = () => reject(new Error('Database deletion blocked'));
        });
    }

    close() {
        if (this.db) {
            this.db.close();
            this.db = null;
        }
    }
}

// Utility functions for common operations
export class SimpleStore {
    constructor(dbName, storeName, version = 1) {
        this.dbName = dbName;
        this.storeName = storeName;
        this.version = version;
        this.helper = new IDBHelper(dbName, version);
    }

    async init(keyPath = 'id', indexes = []) {
        await this.helper.open((db) => {
            if (!db.objectStoreNames.contains(this.storeName)) {
                const store = db.createObjectStore(this.storeName, {
                    keyPath,
                    autoIncrement: !keyPath
                });

                indexes.forEach(index => {
                    store.createIndex(
                        index.name,
                        index.keyPath || index.name,
                        { unique: index.unique || false }
                    );
                });
            }
        });
    }

    async get(key) {
        return this.helper.get(this.storeName, key);
    }

    async getAll(query, count) {
        return this.helper.getAll(this.storeName, query, count);
    }

    async put(value, key) {
        return this.helper.put(this.storeName, value, key);
    }

    async add(value, key) {
        return this.helper.add(this.storeName, value, key);
    }

    async delete(key) {
        return this.helper.delete(this.storeName, key);
    }

    async clear() {
        return this.helper.clear(this.storeName);
    }

    async count() {
        return this.helper.count(this.storeName);
    }

    async iterate(callback) {
        return this.helper.iterate(this.storeName, callback);
    }

    close() {
        this.helper.close();
    }

    async destroy() {
        return this.helper.deleteDatabase();
    }
}

export { IDBHelper };

export default SimpleStore;