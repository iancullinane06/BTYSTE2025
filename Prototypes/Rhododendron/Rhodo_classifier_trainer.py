import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
from keras import layers, models, optimizers
import tifffile as tiff
import tensorflow as tf
from keras.callbacks import Callback, TensorBoard
from dotenv import load_dotenv  # Import load_dotenv from python-dotenv

load_dotenv()

# Set parameters
IMG_HEIGHT = 244
IMG_WIDTH = 244
IMG_CHANNELS = 6
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4  # Learning rate variable
LOG_DIR = os.getenv('LOG_DIR', "./.logs")  # Directory to save TensorBoard logs

data_dir = os.getenv('RHODODENDRON-DATASET')
data_dir = os.path.join(data_dir, "images")

# Custom augmentation function
def augment_image(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(image, k=np.random.randint(4))  # Random 90 degree rotation
    return image

class TiffDataGenerator(Sequence):
    def __init__(self, filepaths, labels, batch_size, img_height, img_width, img_channels, augment=False):
        self.filepaths = filepaths
        self.labels = labels
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.augment = augment
        self.indices = np.arange(len(self.filepaths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.filepaths) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_filepaths = [self.filepaths[i] for i in indices]
        batch_labels = [self.labels[i] for i in indices]

        X, y = self.__data_generation(batch_filepaths, batch_labels)

        if self.augment:
            X = np.array([augment_image(img) for img in X])
        
        return X, y

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_filepaths, batch_labels):
        X = np.empty((self.batch_size, self.img_height, self.img_width, self.img_channels))
        y = np.empty((self.batch_size), dtype=int)

        for i, (filepath, label) in enumerate(zip(batch_filepaths, batch_labels)):
            img = tiff.imread(filepath)
            if img.shape != (self.img_height, self.img_width, self.img_channels):
                img = np.resize(img, (self.img_height, self.img_width, self.img_channels))
            X[i,] = img
            y[i,] = label

        return X, y

def load_data(data_dir):
    filepaths = []
    labels = []
    classes = os.listdir(data_dir)
    class_indices = {cls: idx for idx, cls in enumerate(classes)}

    for cls in classes:
        class_dir = os.path.join(data_dir, cls)
        for file in os.listdir(class_dir):
            if file.endswith('.tif'):
                filepaths.append(os.path.join(class_dir, file))
                labels.append(class_indices[cls])

    return filepaths, labels

# Load data
filepaths, labels = load_data(data_dir)

# Split into training, validation, and testing sets
train_filepaths, test_filepaths, train_labels, test_labels = train_test_split(
    filepaths, labels, test_size=0.1, stratify=labels, random_state=42
)

train_filepaths, val_filepaths, train_labels, val_labels = train_test_split(
    train_filepaths, train_labels, test_size=0.2 / 0.9, stratify=train_labels, random_state=42
)

# Create data generators
train_generator = TiffDataGenerator(train_filepaths, train_labels, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=True)
validation_generator = TiffDataGenerator(val_filepaths, val_labels, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=False)
test_generator = TiffDataGenerator(test_filepaths, test_labels, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=False)

# Build the model
model = models.Sequential([
    layers.InputLayer(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
    layers.Conv2D(32, (3, 3), activation='relu', name='conv1'),
    layers.MaxPooling2D((2, 2), name='pool1'),
    layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    layers.MaxPooling2D((2, 2), name='pool2'),
    # layers.Conv2D(128, (3, 3), activation='relu', name='conv3'),
    # layers.MaxPooling2D((2, 2), name='pool3'),
    layers.Flatten(name='flatten'),
    layers.Dense(512, activation='relu', name='dense1'),
    layers.Dropout(0.5, name='dropout'),
    layers.Dense(1, activation='sigmoid', name='output')  # Sigmoid activation for binary classification
])

model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',  # Binary crossentropy for binary classification
              metrics=['accuracy'])

# Create a custom callback to print weights, biases, and visualize conv layers
class DebugCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Print weights and biases for each layer
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        for layer in self.model.layers:
            weights = layer.get_weights()
            if weights:
                print(f"Layer {layer.name} weights shape: {np.shape(weights[0])}")
                print(f"Layer {layer.name} biases shape: {np.shape(weights[1])}")

                # Print weights and biases (truncated for readability)
                print(f"Layer {layer.name} weights: {weights[0].flatten()[:5]} ...")  # Print first 5 weights
                print(f"Layer {layer.name} biases: {weights[1].flatten()[:5]} ...")  # Print first 5 biases

# Create TensorBoard callback
tensorboard_callback = TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

# Train the model with the DebugCallback
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    callbacks=[tensorboard_callback, DebugCallback()]  # Add DebugCallback here
)

# Plot training & validation accuracy/loss values
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

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

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Predict probabilities on the validation set
val_probs = model.predict(validation_generator, steps=len(validation_generator))
val_preds = (val_probs > 0.5).astype(int).flatten()  # Convert probabilities to binary predictions

# Print probabilities and actual labels for validation set
print("Validation Set Predictions (Probabilities):")
for i, prob in enumerate(val_probs):
    print(f"Sample {i+1}: Probability = {prob[0]:.4f}, Predicted Label = {val_preds[i]}, Actual Label = {val_labels[i]}")

# Generate classification report
val_labels_array = np.array(val_labels)
print("\nClassification Report:")
print(classification_report(val_labels_array, val_preds))

# Generate confusion matrix
cm = confusion_matrix(val_labels_array, val_preds)
print("\nConfusion Matrix:")
print(cm)
