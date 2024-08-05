import cv2
import numpy as np

def preprocess_image(image):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Optionally, enhance contrast or perform other preprocessing steps
    # ...

    return hsv_image

def threshold_segmentation(image, lower_threshold, upper_threshold):
    # Create a binary mask based on the threshold values
    mask = cv2.inRange(image, lower_threshold, upper_threshold)

    # Apply morphological operations to clean up the mask (optional)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def detect_anomalies(image, threshold_value):
    # Identify anomalous pixels based on a threshold value (e.g., intensity)
    anomaly_mask = (image > threshold_value).astype(np.uint8)

    # Optionally, apply additional post-processing steps
    # ...

    return anomaly_mask

# Load the aerial image
image_path = r"C:\Users\rough\Downloads\archive (1)\1290091.025782968_6132926.158789683_1290874.505322890_6133709.638329607.tiff"
original_image = cv2.imread(image_path)

# Preprocess the image
preprocessed_image = preprocess_image(original_image)

# Thresholding for different components (adjust threshold values)
green_threshold = (40, 80, 40), (110, 240, 150)  # Example thresholds for green wooded areas
water_threshold = (10, 30, 10), (80, 200, 200)  # Example thresholds for water bodies
urban_threshold = (0, 0, 150), (80, 80, 255)      # Example thresholds for urban areas

green_mask = threshold_segmentation(preprocessed_image, *green_threshold)
water_mask = threshold_segmentation(preprocessed_image, *water_threshold)
urban_mask = threshold_segmentation(preprocessed_image, *urban_threshold)

# Combine the segmented components
result_image = cv2.bitwise_or(cv2.bitwise_or(green_mask, water_mask), urban_mask)

# Detect anomalies (forest fires)
anomaly_threshold = 200  # Adjust the threshold based on your needs
anomaly_mask = detect_anomalies(preprocessed_image[:, :, 2], anomaly_threshold)

# Visualize the results
cv2.imshow('Original Image', original_image)
cv2.imshow('Green Mask', green_mask)
cv2.imshow('Water Mask', water_mask)
"""
cv2.imshow('Urban Mask', urban_mask)
cv2.imshow('Anomaly Mask', anomaly_mask * 255)  # Scaling for visualization
"""
cv2.waitKey(0)
cv2.destroyAllWindows()
