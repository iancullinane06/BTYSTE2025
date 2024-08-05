import os
import csv
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import f1_score
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input

# Hyperparameters and settings
Learning_Rate = 1e-5
width = height = 254
batchSize = 32
num_classes = 2
prototype = 1
num_batches = 100  # Number of batches to run

TrainFolder = r"C:\Users\rough\OneDrive\Desktop\Coding\BTYSTE-2024\Datasets\Fire\Training"
ListImageFolders = os.listdir(TrainFolder)

# Transformations
def transformImg(img):
    img = cv2.resize(img, (height, width))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

# Read random image
def ReadRandomImage():
    while True:
        folder = np.random.choice(ListImageFolders)
        idx = np.random.randint(0, len(os.listdir(os.path.join(TrainFolder, folder))))
        img_path = os.path.join(TrainFolder, folder, os.listdir(os.path.join(TrainFolder, folder))[idx])
        img = cv2.imread(img_path)
        if img is not None:
            img = img[:, :, :3]
            label = 1 if "fire" in folder.lower() else 0  # Assuming folder names indicate fire or no fire
            img = transformImg(img)
            return img, label

# Load batch of images
def LoadBatch():
    images = np.zeros((batchSize, height, width, 3), dtype=np.float32)
    labels = np.zeros((batchSize), dtype=np.int32)
    for i in range(batchSize):
        images[i], labels[i] = ReadRandomImage()
    return images, labels

# Create and compile model
model = ResNet50(weights=None, input_shape=(height, width, 3), classes=num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_Rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create CSV files for loss and F1 score
loss_csv_file = "data/Fire-Prot-" + str(prototype) + "-loss.csv"
f1_csv_file = "data/Fire-Prot-" + str(prototype) + "-f1.csv"

# Initialize CSV files with headers
with open(loss_csv_file, "w", newline="") as file:
    db = csv.writer(file)
    db.writerow(["Loss"])

with open(f1_csv_file, "w", newline="") as file:
    db = csv.writer(file)
    db.writerow(["F1_Score"])

# Train
for epoch in range(num_batches):
    images, labels = LoadBatch()
    loss = model.train_on_batch(images, labels)
    print(epoch, ") Loss=", loss[0])

    # Save loss to CSV
    with open(loss_csv_file, "a", newline="") as file:
        db = csv.writer(file)
        db.writerow([loss[0]])

    # Calculate F1 score
    predictions = model.predict(images)
    predicted_labels = np.argmax(predictions, axis=1)
    f1 = f1_score(labels, predicted_labels, average='weighted')

    # Save F1 score to CSV
    with open(f1_csv_file, "a", newline="") as file:
        db = csv.writer(file)
        db.writerow([f1])

    if epoch % 10 == 0:
        # Save model weights every 10 epochs
        model.save_weights("Models/Resnet/Fire-Prot-" + str(prototype) + ".h5")
        print("Saving Model: Models/Resnet/Fire-Prot-" + str(prototype) + ".h5")
