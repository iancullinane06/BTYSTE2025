const canvas = document.getElementById('raster-canvas');
const ctx = canvas.getContext('2d');

// Handle the image upload process
window.api.receive('file-selected', async (filePath) => {
    uploadStatus.innerText = `File selected: ${filePath}`;
    openFileButton.style.display = 'none'; // Hide the button after selecting a file

    try {
        const img = new Image();
        img.src = filePath;

        img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear the canvas before drawing
            ctx.drawImage(img, 0, 0); // Draw the uploaded image
        };
        
        img.onerror = () => {
            uploadStatus.innerText = 'Error loading image.';
        };
    } catch (error) {
        uploadStatus.innerText = 'Error loading file.';
        console.error('Error loading file:', error);
    }
});

// Function to load and draw shapefile data (placeholder for future use)
async function loadShapefile(filePath) {
    // Implement logic to load and parse the shapefile
    // For demonstration, we return a hardcoded array of shapes
    return [
        { x: 100, y: 150 }, 
        { x: 200, y: 250 },
        { x: 300, y: 350 }
    ];
}

// Function to draw a shape on the canvas in isometric view (placeholder for future use)
function drawShape(shape) {
    const isoX = shape.x - shape.y; 
    const isoY = (shape.x + shape.y) / 2; 

    ctx.beginPath();
    ctx.arc(isoX, isoY, 5, 0, Math.PI * 2); 
    ctx.fillStyle = 'blue'; 
    ctx.fill();
    ctx.closePath();
}

// Function to draw an isometric red square for debugging
function drawIsometricRedSquare() {
    const size = 50; // The size of one side of the square in regular view
    const isoX = 150; // The isometric X position (center)
    const isoY = 150; // The isometric Y position (center)

    // Clear the canvas before drawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Begin the path for the isometric square (trapezoid in isometric projection)
    ctx.beginPath();

    // Move to the top point of the isometric square
    ctx.moveTo(isoX, isoY);

    // Calculate the isometric points by applying an isometric projection formula
    // X-axis moves right and up (30 degrees)
    ctx.lineTo(isoX + size * Math.cos(Math.PI / 6), isoY + size * Math.sin(Math.PI / 6));
    
    // Y-axis moves left and up (150 degrees)
    ctx.lineTo(isoX, isoY + size * Math.sin(Math.PI / 3));
    
    // Z-axis moves down (120 degrees from each of the other axes)
    ctx.lineTo(isoX - size * Math.cos(Math.PI / 6), isoY + size * Math.sin(Math.PI / 6));

    // Close the path back to the starting point
    ctx.closePath();

    // Set fill color to red
    ctx.fillStyle = 'red';

    // Fill the trapezoid shape
    ctx.fill();
}

// Function to process raster data (optional, keep for future use)
async function processFileData(fileData) {
    // Use geotiff.js or another library to decode the TIFF
    const tiff = await GeoTIFF.fromArrayBuffer(fileData); 
    const image = await tiff.getImage(); 
    const rasterData = await image.readRasters(); 
    return rasterData; 
}

// Function to display raster data (optional, keep for future use)
function displayRaster(rasterData) {
    const width = rasterData[0].length; 
    const height = rasterData.length; 
    const imageData = ctx.createImageData(width, height); 

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const index = y * width + x;
            const value = rasterData[y][x]; 
            const normalizedValue = Math.min(Math.max(value, 0), 255);
            const pixelIndex = index * 4; 
            imageData.data[pixelIndex] = normalizedValue; 
            imageData.data[pixelIndex + 1] = normalizedValue; 
            imageData.data[pixelIndex + 2] = normalizedValue; 
            imageData.data[pixelIndex + 3] = 255; 
        }
    }

    ctx.putImageData(imageData, 0, 0); 
}
