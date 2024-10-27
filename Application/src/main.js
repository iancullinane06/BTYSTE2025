const { app, BrowserWindow, Menu, ipcMain, dialog } = require('electron');
const fs = require('fs').promises; // Use promises for asynchronous file operations
const path = require('path');

let mainWindow

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 964,
        height: 800,
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            enableRemoteModule: false,
            nodeIntegration: false
        }
    });
    mainWindow.loadURL('http://localhost:5000');
    mainWindow.webContents.openDevTools();

    // Build menu with file options
    const menu = Menu.buildFromTemplate([
        {
            label: 'File',
            submenu: [
                { label: 'Open Image', click: openFile },
                { label: 'Save Selected Layer', click: saveSelectedLayer },
                { label: 'Save All Layers', click: saveAllLayers }
            ]
        }
    ]);
    Menu.setApplicationMenu(menu);
}

// Function to handle file opening
async function openFile() {
    const result = await dialog.showOpenDialog(mainWindow, {
        properties: ['openFile'],
        filters: [{ name: 'Images', extensions: ['tif', 'tiff', 'png', 'jpg', 'jpeg'] }]
    });
    if (!result.canceled) {
        const filePath = result.filePaths[0];
        // Send the selected file path back to the renderer process
        mainWindow.webContents.send('file-selected', filePath);
    }
}

// Function to handle saving selected layer
function saveSelectedLayer() {
    mainWindow.webContents.send('get-selected-layer');  // Request selected layer from renderer
}

// Function to handle saving all layers
function saveAllLayers() {
    mainWindow.webContents.send('get-all-layers');  // Request all layers from renderer
}

app.whenReady().then(createWindow);

app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') app.quit();
});

// Listen for IPC events in the main process
ipcMain.on("toMain", (event, args) => {
    fs.readFile("path/to/file", (error, data) => {
      if (error) {
        console.error("Error reading file:", error);
        return;
      }
      // Send the response back to the renderer
      win.webContents.send("fromMain", data.toString());
    });
  });

// Add other IPC handlers as needed
