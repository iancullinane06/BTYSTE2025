import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
import tifffile as tiff
import tensorflow as tf
from keras import optimizers
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, BatchNormalization, Activation, UpSampling2D, Reshape, Concatenate, Dropout
from keras.models import Model
from keras.applications import DenseNet121
from keras.regularizers import l2
from dotenv import load_dotenv
import datetime

load_dotenv()

# Set parameters
IMG_HEIGHT = 244
IMG_WIDTH = 244
IMG_CHANNELS = 6
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-1
LOG_DIR = os.getenv('LOG_DIR', "./.logs/Rhodo-semantics-plus/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

data_dir = os.getenv('RHODODENDRON-DATASET')

# Custom augmentation function
def augment_image(image, mask):
    if np.random.rand() > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if np.random.rand() > 0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)
    if np.random.rand() > 0.5:
        image = tf.image.rot90(image, k=np.random.randint(4))
        mask = tf.image.rot90(mask, k=np.random.randint(4))
    if np.random.rand() > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.1)
    if np.random.rand() > 0.5:
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    return image, mask

def normalize_image(image):
    for c in range(image.shape[-1]):
        image[..., c] = (image[..., c] - np.mean(image[..., c])) / np.std(image[..., c])
    return image

class TiffDataGenerator(Sequence):
    def __init__(self, filepaths, masks, batch_size, img_height, img_width, img_channels, augment=False, max_batches_per_epoch=None):
        self.filepaths = filepaths
        self.masks = masks
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.img_channels = img_channels
        self.augment = augment
        self.max_batches_per_epoch = max_batches_per_epoch
        self.indices = np.arange(len(self.filepaths))
        self.on_epoch_end()

    def __len__(self):
        num_batches = int(np.floor(len(self.filepaths) / self.batch_size))
        if self.max_batches_per_epoch:
            return min(num_batches, self.max_batches_per_epoch)
        return num_batches

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
        X = np.empty((self.batch_size, self.img_height, self.img_width, self.img_channels), dtype=np.float32)
        y = np.empty((self.batch_size, self.img_height, self.img_width, 1), dtype=np.float32)  # Ensure masks have 1 channel

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
            
            # Convert image and mask to NumPy arrays
            img = img.numpy() if isinstance(img, tf.Tensor) else img
            mask = mask.numpy() if isinstance(mask, tf.Tensor) else mask

            # Normalize the images
            img = img / 255.0

            X[i] = img
            y[i] = mask  # Already added the channel dimension

        return X, y


def load_data(data_dir):
    filepaths = []
    masks = []

    # Directories for images and masks
    image_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "maps")

    # Check if the directories exist
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        print(f"Error: The 'images' or 'maps' directory does not exist.")
        return [], []

    # Iterate through all files in the images directory
    for file in os.listdir(image_dir):
        if file.endswith('.tif'):
            img_path = os.path.join(image_dir, file)
            mask_path = os.path.join(mask_dir, file)

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
train_generator = TiffDataGenerator(train_filepaths, train_masks, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=True, max_batches_per_epoch=100)
validation_generator = TiffDataGenerator(val_filepaths, val_masks, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=False)
test_generator = TiffDataGenerator(test_filepaths, test_masks, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, augment=False)

def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Dice Coefficient for binary segmentation.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice loss function, which is 1 - Dice Coefficient.
    """
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """
    Combined loss: Dice loss + Binary Cross-Entropy
    """
    return dice_loss(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)

def ASPP_module(x):
    """
    Atrous Spatial Pyramid Pooling (ASPP) module.
    """
    # Define atrous convolution layers with different rates
    aspp1 = Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    aspp1 = BatchNormalization()(aspp1)
    aspp1 = Activation('relu')(aspp1)

    aspp2 = Conv2D(256, (3, 3), padding='same', dilation_rate=6, use_bias=False)(x)
    aspp2 = BatchNormalization()(aspp2)
    aspp2 = Activation('relu')(aspp2)

    aspp3 = Conv2D(256, (3, 3), padding='same', dilation_rate=12, use_bias=False)(x)
    aspp3 = BatchNormalization()(aspp3)
    aspp3 = Activation('relu')(aspp3)

    aspp4 = Conv2D(256, (3, 3), padding='same', dilation_rate=18, use_bias=False)(x)
    aspp4 = BatchNormalization()(aspp4)
    aspp4 = Activation('relu')(aspp4)

    # Global Average Pooling
    aspp5 = GlobalAveragePooling2D()(x)
    aspp5 = Reshape((1, 1, aspp5.shape[-1]))(aspp5)
    aspp5 = Conv2D(256, (1, 1), padding='same', use_bias=False)(aspp5)
    aspp5 = BatchNormalization()(aspp5)
    aspp5 = Activation('relu')(aspp5)
    aspp5 = UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation='bilinear')(aspp5)

    # Concatenate ASPP outputs
    x = Concatenate(axis=-1)([aspp1, aspp2, aspp3, aspp4, aspp5])
    x = Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def DeepLabV3Plus(input_shape, num_classes=1):
    inputs = Input(shape=input_shape)

    # Initial Conv layer to handle multiple channels
    x = Conv2D(64, (7, 7), padding="same", use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Load base model (DenseNet121)
    base_model = DenseNet121(weights=None, include_top=False, input_tensor=x)
    
    # Extract low-level features
    low_level_feat = base_model.get_layer('conv1/relu').output

    # Apply ASPP module
    x = base_model.output
    x = ASPP_module(x)
    
    # Resize the ASPP output to match the low-level feature map size
    target_size = tf.shape(low_level_feat)[1:3]
    x = tf.image.resize(x, size=target_size, method='bilinear')

    # Ensure low_level_feat has the correct number of channels
    low_level_feat = Conv2D(48, (1, 1), padding="same", use_bias=False)(low_level_feat)
    low_level_feat = BatchNormalization()(low_level_feat)
    low_level_feat = Activation("relu")(low_level_feat)
    
    # Resize low_level_feat to match x dimensions
    low_level_feat = tf.image.resize(low_level_feat, size=tf.shape(x)[1:3], method='bilinear')

    # Concatenate
    x = Concatenate(axis=-1)([x, low_level_feat])

    # Further convolutional layers
    x = Conv2D(256, (3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Dropout(0.2)(x)

    # Final upsampling to the original image size
    final_upsample_size = (input_shape[0], input_shape[1])
    x = tf.image.resize(x, size=final_upsample_size, method='bilinear')

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation="sigmoid", padding="same")(x)

    model = Model(inputs, outputs)
    return model

# Create the model
model = DeepLabV3Plus(input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), num_classes=1)

# Compile the model with the combined loss function and metrics
model.compile(optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
              loss=combined_loss,
              metrics=[dice_coefficient])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

# Train the model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, lr_scheduler, tensorboard_callback]
)

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
    plt.plot(history.history['dice_coefficient'], label='Train Dice Coefficient')
    plt.plot(history.history['val_dice_coefficient'], label='Validation Dice Coefficient')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Coefficient')
    plt.legend()

    plt.show()

plot_history(history)

# Evaluate the model
test_loss, test_dice_coefficient = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Dice Coefficient: {test_dice_coefficient}")

# Save the model
model.save('DLV3+-model.h5')
