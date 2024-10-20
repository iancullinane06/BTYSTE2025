// script.js
const { remote } = require('electron');
const { dialog } = remote;
const fs = require('fs');

const openFileButton = document.getElementById('open-file');
const rasterCanvas = document.getElementById('raster-canvas');
const layerName = document.getElementById('layer-name');
const uploadButton = document.getElementById("uploadButton");
const fileInput = document.getElementById("fileInput");

// Open file dialog to select an image
openFileButton.addEventListener('click', async () => {
    const result = await dialog.showOpenDialog({
        properties: ['openFile'],
        filters: [
            { name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'tiff'] },
        ],
    });

    if (!result.canceled) {
        const filePath = result.filePaths[0];
        loadImage(filePath);
    }
});

// Load and display the selected image
function loadImage(filePath) {
    const img = new Image();
    img.src = filePath;
    img.onload = () => {
        const ctx = rasterCanvas.getContext('2d');
        rasterCanvas.width = img.width;
        rasterCanvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        layerName.textContent = filePath.split('/').pop(); // Set layer name to the file name
    };
}

// Implement tab switching logic
document.querySelectorAll('.tablinks').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tablinks').forEach(t => {
            t.classList.remove('active');
            t.classList.remove('selected'); // Reset selected class
        });
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
        
        tab.classList.add('active');
        tab.classList.add('selected'); // Add selected class
        const tabName = tab.getAttribute('data-tab');
        document.getElementById(tabName).classList.add('active');
    });
});

// Upload image function
uploadButton.addEventListener("click", function () {
    const file = fileInput.files[0];

    if (file) {
        const formData = new FormData();
        formData.append("file", file);

        // Load image and display it
        fetch("/load_image", {
            method: "POST",
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                if (data.status) {
                    const imgElement = document.getElementById("uploadedImage");
                    imgElement.src = URL.createObjectURL(file);
                    document.getElementById("imageContainer").style.display = "block";
                } else {
                    alert(data.error);
                }
            })
            .catch(error => console.error("Error loading image:", error));
    }
});

// Tab open function
function openTab(evt, tabName) {
    // Hide all tab contents
    const tabcontents = document.getElementsByClassName("tab-pane");
    for (let i = 0; i < tabcontents.length; i++) {
        tabcontents[i].style.display = "none"; // Hide each tab content
    }

    // Remove "active" class from all buttons
    const tablinks = document.getElementsByClassName("tablinks");
    for (let i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }

    // Show the current tab and add "active" class to the button that opened it
    document.getElementById(tabName).style.display = "block"; // Show the selected tab
    evt.currentTarget.className += " active"; // Add "active" class to the clicked tab
}


// Additional functions for NDVI/NDRE, inference, and saving layers...
