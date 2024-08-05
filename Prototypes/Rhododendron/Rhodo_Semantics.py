import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
import tifffile as tiff
import tensorflow as tf
from keras import layers, models, optimizers
from keras.callbacks import Callback, TensorBoard
from keras.applications import DenseNet121
from keras.layers import Input
from keras.models import Model

# Load environment variables from a .env file
from dotenv import load_dotenv
load_dotenv()

# Set parameters
IMG_HEIGHT = 244
IMG_WIDTH = 244
IMG_CHANNELS = 6
BATCH_SIZE = 4  # Smaller batch size might be needed for larger models
EPOCHS = 100
LEARNING_RATE = 1e-4  # Learning rate variable
LOG_DIR = os.getenv('LOG_DIR', "./.logs")  # Directory to save TensorBoard logs

data_dir = os.getenv('RHODODENDRON-DATASET')

# Custom augmentation function
def augment_image(image, mask):
    if np.random.rand() > 0.5:
        image = tf.image.random_flip_left_right(image)
        mask = tf.image.random_flip_left_right(mask)
    if np.random.rand() > 0.5:
        image = tf.image.random_flip_up_down(image)
        mask = tf.image.random_flip_up_down(mask)
    if np.random.rand() > 0.5:
        image = tf.image.rot90(image, k=np.random.randint(4))
        mask = tf.image.rot90(mask, k=np.random.randint(4))
    return image, mask

class TiffDataGenerator(Sequence):
    def __init__(self, filepaths, masks, batch_size, img_height, img_width, img_channels, augment=False):
        self.filepaths = filepaths
        self.masks = masks
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
        batch_masks = [self.masks[i] for i in indices]

        X, y = self.__data_generation(batch_filepaths, batch_masks)

        if self.augment:
            X, y = zip(*[augment_image(x, mask) for x, mask in zip(X, y)])
            X = np.array(X)
            y = np.array(y)

        return X, y

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)

    def __data_generation(self, batch_filepaths, batch_masks):
        X = np.empty((self.batch_size, self.img_height, self.img_width, self.img_channels))
        y = np.empty((self.batch_size, self.img_height, self.img_width, 1))  # Mask with 1 channel

        for i, (filepath, maskpath) in enumerate(zip(batch_filepaths, batch_masks)):
            img = tiff.imread(filepath)
            mask = tiff.imread(maskpath)
            
            # Resize images if needed
            if img.shape != (self.img_height, self.img_width, self.img_channels):
                img = np.resize(img, (self.img_height, self.img_width, self.img_channels))
            
            if mask.shape != (self.img_height, self.img_width):
                mask = np.resize(mask, (self.img_height, self.img_width))
            
            # Normalize the images and masks
            X[i,] = img / 255.0
            y[i,] = mask[..., np.newaxis]  # Add channel dimension to mask

        return X, y

def load_data(data_dir):
    filepaths = []
    masks = []
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "maps")

    for file in os.listdir(image_dir):
        if file.endswith('.tif'):
            filepaths.append(os.path.join(image_dir, file))
            mask_name = file.replace('.tif', '.tif')  # Assuming masks have similar names
            masks.append(os.path.join(mask_dir, mask_name))

    return filepaths, masks

# Load data
filepaths, masks = load_data(data_dir)

# Split into training, validation, and testing sets
train_filepaths, test_filepaths, train_masks, test_masks = train_test_split(
    filepaths, masks, test_size=0.1, random_state=42
)

train_filepaths, val_filepaths, train_masks, val_masks = train_test_split(
    train_filepaths, train_masks, test_size=0.2 / 0.9, random_state=42
)

# Create data generators
train_generator = TiffDataGenerator(train_filepaths, train_masks, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=True)
validation_generator = TiffDataGenerator(val_filepaths, val_masks, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=False)
test_generator = TiffDataGenerator(test_filepaths, test_masks, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=False)

# Function to create DeepLabV3 model
def create_deeplabv3_model(input_shape, num_classes=1):
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Add ASPP and decoder on top of the base model
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)

    # ASPP
    b0 = layers.Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    b0 = layers.BatchNormalization()(b0)
    b0 = layers.Activation("relu")(b0)

    b1 = layers.DepthwiseConv2D((3, 3), dilation_rate=1, padding="same", use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation("relu")(b1)
    b1 = layers.Conv2D(256, (1, 1), padding="same", use_bias=False)(b1)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation("relu")(b1)

    b2 = layers.DepthwiseConv2D((3, 3), dilation_rate=2, padding="same", use_bias=False)(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation("relu")(b2)
    b2 = layers.Conv2D(256, (1, 1), padding="same", use_bias=False)(b2)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation("relu")(b2)

    b3 = layers.DepthwiseConv2D((3, 3), dilation_rate=3, padding="same", use_bias=False)(x)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation("relu")(b3)
    b3 = layers.Conv2D(256, (1, 1), padding="same", use_bias=False)(b3)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation("relu")(b3)

    b4 = layers.GlobalAveragePooling2D()(x)
    b4 = layers.Reshape((1, 1, -1))(b4)
    b4 = layers.Conv2D(256, (1, 1), padding="same", use_bias=False)(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation("relu")(b4)
    b4 = layers.UpSampling2D(size=(IMG_HEIGHT // 4, IMG_WIDTH // 4), interpolation="bilinear")(b4)

    x = layers.Concatenate()([b4, b0, b1, b2, b3])
    x = layers.Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid', padding="same")(x)

    model = Model(inputs, outputs)
    return model

# Build the DeepLabV3 model
model = create_deeplabv3_model((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='mean_squared_error',  # MSE for binary segmentation
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

# Plotting some sample outputs for visualization
def plot_sample_prediction(images, masks, predictions, n=5):
    plt.figure(figsize=(15, 15))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(images[i])
        plt.title("Input Image")
        plt.axis("off")

        ax = plt.subplot(3, n, i + n + 1)
        plt.imshow(masks[i].squeeze(), cmap='gray')
        plt.title("True Mask")
        plt.axis("off")

        ax = plt.subplot(3, n, i + 2 * n + 1)
        plt.imshow(predictions[i].squeeze(), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis("off")

    plt.show()

# Load a small batch to visualize
sample_images, sample_masks = next(iter(validation_generator))
sample_predictions = model.predict(sample_images)

plot_sample_prediction(sample_images, sample_masks, sample_predictions, n=3)
