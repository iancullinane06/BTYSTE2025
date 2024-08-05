import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
import tifffile as tiff
import tensorflow as tf
from keras import layers, models, optimizers
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, BatchNormalization, Activation, UpSampling2D, Reshape, Concatenate, Cropping2D
from keras.models import Model
from keras.applications import DenseNet121
from dotenv import load_dotenv

load_dotenv()

# Set parameters
IMG_HEIGHT = 244
IMG_WIDTH = 244
IMG_CHANNELS = 6
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4
LOG_DIR = os.getenv('LOG_DIR', "./.logs")

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
        y = np.empty((self.batch_size, self.img_height, self.img_width, 1))  # Ensure masks have 1 channel

        for i, (filepath, maskpath) in enumerate(zip(batch_filepaths, batch_masks)):
            img = tiff.imread(filepath)
            mask = tiff.imread(maskpath)
            
            # Ensure the mask has 3 dimensions by adding a channel dimension
            if mask.ndim == 2:
                mask = mask[..., np.newaxis]  # Add channel dimension if not present

            # Resize images and masks
            img = tf.image.resize(img, (self.img_height, self.img_width))
            mask = tf.image.resize(mask, (self.img_height, self.img_width))
            
            # Convert mask values for binary classification
            mask = np.where(mask == 2, 1, 0)  # Adjust this according to your mask values
            
            # Normalize the images and masks
            X[i] = img / 255.0
            y[i] = mask  # Already added the channel dimension

        return X, y

def load_data(data_dir):
    filepaths = []
    masks = []

    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "maps")

    for subdir in ["rhododendron", "non_rhododendron"]:
        sub_image_dir = os.path.join(image_dir, subdir)
        sub_mask_dir = os.path.join(mask_dir, subdir)

        if not os.path.exists(sub_image_dir) or not os.path.exists(sub_mask_dir):
            print(f"Directory {subdir} does not exist in images or maps folder.")
            continue

        for file in os.listdir(sub_image_dir):
            if file.endswith('.tif'):
                img_path = os.path.join(sub_image_dir, file)
                mask_path = os.path.join(sub_mask_dir, file)

                if os.path.exists(mask_path):
                    filepaths.append(img_path)
                    masks.append(mask_path)
                else:
                    print(f"Warning: Mask not found for {img_path}")

    print(f"Total images: {len(filepaths)}")
    print(f"Total masks: {len(masks)}")
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

def create_deeplabv3_model(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)

    # Initial Conv layer to handle 6 channels
    x = Conv2D(64, (7, 7), padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Load DenseNet121 without weights and use it as a feature extractor
    base_model = DenseNet121(weights=None, include_top=False, input_tensor=x)

    # ASPP (Atrous Spatial Pyramid Pooling)
    x = base_model.output

    # ASPP branch 0
    b0 = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    # ASPP branch 1
    b1 = DepthwiseConv2D((3, 3), dilation_rate=1, padding="same", use_bias=False)(
        x
    )
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    # ASPP branch 2
    b2 = DepthwiseConv2D((3, 3), dilation_rate=2, padding="same", use_bias=False)(
        x
    )
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    # ASPP branch 3
    b3 = DepthwiseConv2D((3, 3), dilation_rate=3, padding="same", use_bias=False)(
        x
    )
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    # ASPP branch 4
    b4 = GlobalAveragePooling2D()(x)
    b4 = Reshape((1, 1, -1))(b4)
    b4 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    # Resize b4 to match others
    b4 = UpSampling2D(size=(x.shape[1], x.shape[2]))(b4)

    # Concatenate ASPP branches
    x = Concatenate(axis=-1)([b0, b1, b2, b3, b4])

    # Final Conv layer
    x = Conv2D(256, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Calculate necessary scaling factor to upscale to input shape
    # Compute the required upsampling factor
    upscale_factor_height = input_shape[0] // x.shape[1]
    upscale_factor_width = input_shape[1] // x.shape[2]

    # Upsample to the original image size (244x244)
    x = tf.image.resize(x, (input_shape[0], input_shape[1]), method='bilinear')

    # Output layer with sigmoid activation for binary segmentation
    outputs = Conv2D(num_classes, (1, 1), activation="sigmoid", padding="same")(x)

    model = Model(inputs, outputs)
    return model

# Create the model
model = create_deeplabv3_model((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1)

# Compile the model
model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='binary_crossentropy',  # Use binary_crossentropy for binary segmentation
              metrics=['accuracy'])

# Setup TensorBoard logging
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[tensorboard_callback]
)

# Plot training & validation loss/accuracy curves
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

plot_history(history)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# Save the model
model.save(os.path.join(LOG_DIR, 'model.h5'))
