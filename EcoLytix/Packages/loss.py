import tensorflow as tf

from accuracy import dice_coefficient

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