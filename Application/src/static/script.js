// Accessing GeoTIFF from the global scope
const rasterCanvas = document.getElementById('raster-canvas');
const layerName = document.getElementById('layer-name');
const uploadButton = document.getElementById("uploadButton");
const fileInput = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");

// Handle file input change event
fileInput.addEventListener('change', async (event) => {
    const file = event.target.files[0];
    if (file) {
        const filePath = URL.createObjectURL(file);
        uploadStatus.innerText = `File selected: ${file.name}`;
        uploadButton.style.display = "none"; // Hide upload button
        layerName.textContent = file.name;

        // Use GeoTIFF to load the raster file and extract channels
        const tiff = await window.GeoTIFF.fromUrl(filePath);
        const image = await tiff.getImage();
        const width = image.getWidth();
        const height = image.getHeight();

        // Assuming you have 6 color channels in the file
        const channels = [];
        for (let i = 0; i < 6; i++) {
            channels.push(await image.readRasters({ samples: [i] }));
        }

        drawChannelsAsLayers(channels, width, height);
    } else {
        uploadStatus.innerText = 'No file selected.';
    }
});

// Function to draw each channel as a separate colored layer
function drawChannelsAsLayers(channels, width, height) {
    const ctx = rasterCanvas.getContext('2d');
    rasterCanvas.width = width;
    rasterCanvas.height = height;

    // Clear the canvas
    ctx.clearRect(0, 0, rasterCanvas.width, rasterCanvas.height);

    // Define color for each channel
    const colors = [
        'rgba(255, 0, 0, 0.5)',   // Red for channel 1
        'rgba(0, 255, 0, 0.5)',   // Green for channel 2
        'rgba(0, 0, 255, 0.5)',   // Blue for channel 3
        'rgba(255, 255, 0, 0.5)', // Yellow for channel 4
        'rgba(0, 255, 255, 0.5)', // Cyan for channel 5
        'rgba(255, 0, 255, 0.5)'  // Magenta for channel 6
    ];

    // Loop over each channel and draw it
    for (let i = 0; i < channels.length; i++) {
        drawSingleChannel(ctx, channels[i], width, height, colors[i]);
    }
}

// Function to draw a single channel with a specific color
function drawSingleChannel(ctx, channelData, width, height, color) {
    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < channelData.length; i++) {
        const value = channelData[i]; // Get channel pixel value
        // Set each channel to grayscale or color based on the value
        imageData.data[4 * i] = value;   // Red channel
        imageData.data[4 * i + 1] = value; // Green channel
        imageData.data[4 * i + 2] = value; // Blue channel
        imageData.data[4 * i + 3] = 255; // Full opacity
    }

    // Apply color overlay
    ctx.globalCompositeOperation = 'source-over';
    ctx.putImageData(imageData, 0, 0);

    // Apply tint (color) over the channel
    ctx.fillStyle = color;
    ctx.fillRect(0, 0, width, height);
}

// Simulate file upload
uploadButton.addEventListener('click', () => {
    fileInput.click(); // Trigger file input click
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
