import os
import cv2
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from joblib import dump


num_epochs = 50
prot = 2

def preprocess_images(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Check if the image is loaded successfully
    if image is None:
        print(f"Error loading image: {image_path}")
        return None

    # Define a threshold for off-white color
    lower_white = np.array([200, 200, 200], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    # Create a mask to identify off-white pixels
    mask_white = cv2.inRange(image, lower_white, upper_white)

    # Invert the mask to keep non-off-white pixels
    mask_colored = cv2.bitwise_not(mask_white)

    # Bitwise AND operation to keep only the colored pixels
    result = cv2.bitwise_and(image, image, mask=mask_colored)

    # Convert the result to grayscale
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Find contours in the grayscale result
    contours, _ = cv2.findContours(gray_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the region of interest
    mask = np.zeros_like(gray_result)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Bitwise AND operation to get the final result
    final_result = cv2.bitwise_and(result, result, mask=mask)

    return final_result

# Step 2: Feature Extraction


def extract_features(image):
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate the color histograms for each channel (Hue, Saturation, Value)
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv], [2], None, [256], [0, 256])

    # Normalize the histograms
    hist_hue /= hist_hue.sum() + 1e-7
    hist_saturation /= hist_saturation.sum() + 1e-7
    hist_value /= hist_value.sum() + 1e-7

    
    # Calculate additional image statistics
    mean_color = np.mean(image, axis=(0, 1))
    std_color = np.std(image, axis=(0, 1))


    # Concatenate the histograms and additional statistics to create the feature vector
    features = np.concatenate((hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten(),
                               mean_color, std_color))

    return features


# Step 3: Load Data
data = []
labels = []
Leaves_Path = r"C:\Users\rough\OneDrive\Desktop\Coding\BTYSTE-2024\Leaves\Leaves"
for leaf_type in os.listdir(Leaves_Path):
    for image_file in os.listdir(os.path.join(Leaves_Path, leaf_type)):
        image_path = os.path.join(Leaves_Path, leaf_type, image_file)
        processed_image = preprocess_images(image_path)
        features = extract_features(processed_image)

        data.append(features)
        labels.append(leaf_type)

# Step 4: Split Data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Initialize the SVM model
svm_model = SVC(kernel='linear')

# Training loop
for epoch in range(num_epochs):
    # Your training code here
    svm_model.fit(X_train, y_train)
    
    # Calculate and print the training loss
    train_loss = svm_model.score(X_train, y_train)
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss}")

# Evaluate the model
y_pred = svm_model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)

# Save the trained model to a file
model_filename = f"svm-model-prot-{prot}.joblib"
dump(svm_model, os.path.join("Models/SVM/", model_filename))
print(f"Trained model saved to {model_filename}")