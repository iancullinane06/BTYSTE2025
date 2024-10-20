const { app, BrowserWindow, Menu, ipcMain, dialog } = require('electron');
const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 1200,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            nodeIntegration: true,
            contextIsolation: false,
            enableRemoteModule: true
        }
    });
    mainWindow.loadFile(path.join(__dirname, 'templates', 'index.html'));
    
    // Build menu with file options
    const menu = Menu.buildFromTemplate([
        {
            label: 'File',
            submenu: [
                {
                    label: 'Open Image',
                    click: () => {
                        openFile();
                    }
                },
                {
                    label: 'Save Selected Layer',
                    click: () => {
                        saveSelectedLayer();
                    }
                },
                {
                    label: 'Save All Layers',
                    click: () => {
                        saveAllLayers();
                    }
                }
            ]
        }
    ]);
    
    // Set the menu to the application
    Menu.setApplicationMenu(menu);
}

// Function to handle file opening
function openFile() {
    dialog.showOpenDialog(mainWindow, {
        properties: ['openFile'],
        filters: [{ name: 'Images', extensions: ['tif', 'tiff', 'png', 'jpg', 'jpeg'] }]
    }).then(result => {
        if (!result.canceled) {
            const filePath = result.filePaths[0];
            const fileData = fs.readFileSync(filePath);  // Load image data here
            mainWindow.webContents.send('file-opened', fileData);  // Send image data to renderer
        }
    }).catch(err => {
        console.error("Failed to open file: ", err);
    });
}

// Function to handle saving selected layer
function saveSelectedLayer() {
    mainWindow.webContents.send('get-selected-layer');  // Request selected layer from renderer
}

// Function to handle saving all layers
function saveAllLayers() {
    mainWindow.webContents.send('get-all-layers');  // Request all layers from renderer
}

// Respond to renderer with selected layer to save
ipcMain.on('save-selected-layer', (event, layerData) => {
    dialog.showSaveDialog(mainWindow, {
        filters: [{ name: 'TIFF', extensions: ['tiff'] }]
    }).then(result => {
        if (!result.canceled) {
            fs.writeFileSync(result.filePath, layerData);  // Save the layer data to the file
        }
    }).catch(err => {
        console.error("Failed to save layer: ", err);
    });
});

// Respond to renderer with all layers to save
ipcMain.on('save-all-layers', (event, layersData) => {
    dialog.showSaveDialog(mainWindow, {
        filters: [{ name: 'TIFF', extensions: ['tiff'] }]
    }).then(result => {
        if (!result.canceled) {
            fs.writeFileSync(result.filePath, layersData);  // Save all layers to the file
        }
    }).catch(err => {
        console.error("Failed to save all layers: ", err);
    });
});

app.whenReady().then(() => {
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});
