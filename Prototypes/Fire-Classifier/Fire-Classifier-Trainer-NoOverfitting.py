import os
import csv
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from collections import Counter
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
from dotenv import load_dotenv

load_dotenv()

# Hyperparameters and settings
Learning_Rate = 1e-5
width = height = 254
batchSize = 16
num_classes = 2
prototype = 5
num_epochs = 50  # Number of epochs to run
images_per_epoch = 16  # Total images per epoch (8per class)

data_dir = os.getenv('FIRE-CLASSIFICATION-DATASET')

TrainFolder =  os.path.join(data_dir, "Training")
TestFolder = os.path.join(data_dir, "Test")
ListImageFolders = os.listdir(TrainFolder)

# Data augmentation settings
datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    zoom_range=0.1,
    shear_range=0.1,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Transformations
def transformImg(img):
    img = cv2.resize(img, (height, width))
    img = img_to_array(img)
    img = preprocess_input(img)
    return img

# Read images and labels
def ReadImagePathsAndLabels(folder):
    image_paths = []
    labels = []
    for folder_name in os.listdir(folder):
        folder_path = os.path.join(folder, folder_name)
        for filename in os.listdir(folder_path):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(folder_path, filename)
                if os.path.isfile(full_path):
                    image_paths.append(full_path)
                    labels.append(1 if "fire" in folder_name.lower() else 0)
                else:
                    print(f"Warning: {full_path} is not a valid file.")
    return image_paths, labels

# Create and compile model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(height, width, 3))
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Learning_Rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Create CSV files for loss and accuracy
loss_csv_file = "data/Fire-Prot-" + str(prototype) + "-loss.csv"
accuracy_csv_file = "data/Fire-Prot-" + str(prototype) + "-accuracy.csv"

# Initialize CSV files with headers
os.makedirs("data", exist_ok=True)
with open(loss_csv_file, "w", newline="") as file:
    db = csv.writer(file)
    db.writerow(["Loss"])

with open(accuracy_csv_file, "w", newline="") as file:
    db = csv.writer(file)
    db.writerow(["Accuracy"])

# Read image paths and labels
train_image_paths, train_labels = ReadImagePathsAndLabels(TrainFolder)
val_image_paths, val_labels = ReadImagePathsAndLabels(TestFolder)

# Prepare DataFrame for generators
train_df = pd.DataFrame({'filename': train_image_paths, 'class': train_labels})
val_df = pd.DataFrame({'filename': val_image_paths, 'class': val_labels})

# Convert class column to string as required by flow_from_dataframe with binary class_mode
train_df['class'] = train_df['class'].astype(str)
val_df['class'] = val_df['class'].astype(str)

# Check class distribution
print("Training set class distribution:")
print(train_df['class'].value_counts())
print("Validation set class distribution:")
print(val_df['class'].value_counts())

# Prepare validation data generator
val_gen = datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='class',
    target_size=(height, width),
    batch_size=batchSize,
    class_mode='binary',
    shuffle=False
)

# Custom balanced generator for training
def balanced_batch_generator(image_paths, labels, batch_size):
    class_0_indices = np.where(np.array(labels) == 0)[0]
    class_1_indices = np.where(np.array(labels) == 1)[0]

    while True:
        selected_indices = []
        selected_indices.extend(np.random.choice(class_0_indices, batch_size // 2, replace=False))
        selected_indices.extend(np.random.choice(class_1_indices, batch_size // 2, replace=False))
        np.random.shuffle(selected_indices)

        batch_images = []
        batch_labels = []
        for idx in selected_indices:
            img_path = image_paths[idx]
            label = labels[idx]
            img = cv2.imread(img_path)
            img = transformImg(img)
            batch_images.append(img)
            batch_labels.append(label)
        
        yield np.array(batch_images), np.array(batch_labels)

def combined_generator(train_df, datagen, batch_size):
    while True:
        fire_df = train_df[train_df['class'] == '1'].sample(n=batch_size // 2)
        no_fire_df = train_df[train_df['class'] == '0'].sample(n=batch_size // 2)
        
        combined_df = pd.concat([fire_df, no_fire_df])
        combined_gen = datagen.flow_from_dataframe(
            dataframe=combined_df,
            x_col='filename',
            y_col='class',
            target_size=(height, width),
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True
        )
        
        for batch in combined_gen:
            yield batch

balanced_train_gen = combined_generator(train_df, datagen, batchSize)

# TensorBoard callback setup
log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
train_writer = tf.summary.create_file_writer(log_dir + '/train')
val_writer = tf.summary.create_file_writer(log_dir + '/val')

# Early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Training loop
steps_per_epoch = images_per_epoch // batchSize  # Number of steps per epoch to process 100 images
validation_steps = len(val_image_paths) // batchSize

history = model.fit(
    balanced_train_gen,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    validation_data=val_gen,
    validation_steps=validation_steps,
    callbacks=[early_stopping, checkpoint, tensorboard_callback]
)

# Save the final model
model.save_weights(f"Models/Resnet/Fire-Prot-{prototype}.h5")

# Validate after training
val_predictions = []
val_true_labels = []

for val_images, val_labels in val_gen:
    preds = model.predict(val_images)
    val_predictions.extend((preds > 0.5).astype(int).flatten())
    val_true_labels.extend(val_labels)
    for i in range(len(val_labels)):
        print(f"Sample {len(val_true_labels)-len(val_labels)+i}: True Label: {val_labels[i]}, Prediction: {val_predictions[-len(val_labels)+i]}, Probabilities: [{preds[i][0]}, {1 - preds[i][0]}]")
    if len(val_true_labels) >= len(val_image_paths):
        break

# Compute confusion matrix
cm = confusion_matrix(val_true_labels, val_predictions)
print("Confusion Matrix:")
print(cm)

# Display confusion matrix
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
cm_display.plot()
plt.show()

# Calculate F1 Score
f1 = f1_score(val_true_labels, val_predictions)
print(f"F1 Score: {f1}")

# Log validation results to TensorBoard
with val_writer.as_default():
    tf.summary.scalar('f1_score', f1, step=0)
    
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(num_epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

print(f"TensorBoard logs saved to {log_dir}. To view, run: tensorboard --logdir={log_dir}")
