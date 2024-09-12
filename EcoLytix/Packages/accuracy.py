import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1):
    """
    Dice Coefficient for binary segmentation.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# Accuracy metric for final testing
def f2_score(y_true, y_pred, beta=2):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_positive = tf.keras.backend.sum(y_true_f * y_pred_f)
    precision = true_positive / (tf.keras.backend.sum(y_pred_f) + tf.keras.backend.epsilon())
    recall = true_positive / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.epsilon())
    f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + tf.keras.backend.epsilon())
    return f2

def pixel_accuracy(y_true, y_pred):
    """
    Pixel Accuracy metric.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)

    # Compare predictions and true values
    correct_predictions = tf.keras.backend.sum(tf.keras.backend.cast(tf.keras.backend.equal(y_true_f, tf.keras.backend.round(y_pred_f)), dtype='float32'))
    
    # Divide by the total number of pixels
    total_pixels = tf.keras.backend.sum(tf.keras.backend.ones_like(y_true_f))
    return correct_predictions / total_pixels


def mean_iou(y_true, y_pred, num_classes=2):
    y_pred = tf.keras.backend.round(y_pred)
    iou_scores = []
    for i in range(num_classes):
        intersection = tf.keras.backend.sum(tf.keras.backend.cast(y_true == i, tf.float32) * tf.keras.backend.cast(y_pred == i, tf.float32))
        union = tf.keras.backend.sum(tf.keras.backend.cast(y_true == i, tf.float32) + tf.keras.backend.cast(y_pred == i, tf.float32)) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_scores.append(iou)
    return tf.keras.backend.mean(tf.keras.backend.stack(iou_scores))
