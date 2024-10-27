const { contextBridge, ipcRenderer } = require('electron');
const GeoTIFF = require('geotiff');

contextBridge.exposeInMainWorld("GeoTIFF", GeoTIFF);
// Expose the GeoTIFF module along with other IPC methods
contextBridge.exposeInMainWorld("api", {
    send: (channel, data) => {
        let validChannels = ["toMain", "open-file-dialog"];
        if (validChannels.includes(channel)) {
            ipcRenderer.send(channel, data);
        }
    },
    receive: (channel, func) => {
        let validChannels = ["fromMain", "file-selected"];
        if (validChannels.includes(channel)) {
            ipcRenderer.on(channel, (event, ...args) => func(...args));
        }
    }
});