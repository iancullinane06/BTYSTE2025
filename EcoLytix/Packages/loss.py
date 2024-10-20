import tensorflow as tf
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
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

# Define the discriminative loss function components
def discriminative_loss(y_true, y_pred, delta_v=0.5, delta_d=1.5, alpha=1.0, beta=1.0, gamma=0.001):
    """
    Discriminative loss function for multi-class embedding.
    
    Args:
    y_true -- ground truth labels (binary masks)
    y_pred -- predicted embeddings (from the network)
    delta_v -- margin for intra-cluster variance loss
    delta_d -- margin for inter-cluster distance loss
    alpha -- weight for variance loss
    beta -- weight for distance loss
    gamma -- weight for regularization loss
    
    Returns:
    loss -- total discriminative loss
    """

    def Lvar():
        var_loss = 0.0
        for c in range(C):
            cluster_elements = tf.boolean_mask(y_pred, tf.equal(y_true, c))
            if tf.size(cluster_elements) == 0:
                continue  # Skip empty clusters
            
            cluster_center = tf.reduce_mean(cluster_elements, axis=0)
            
            print("Cluster elements shape:", cluster_elements.shape)
            print("Cluster center shape:", cluster_center.shape)
            
            var_term = tf.reduce_mean(tf.maximum(0.0, tf.norm(cluster_elements - cluster_center, ord=2, axis=1) - delta_v) ** 2)
            print("Variance term:", var_term.numpy())
            
            var_loss += var_term
        return var_loss / C if C > 0 else 0.0

    def Ldist():
        dist_loss = 0.0
        for cA in range(C):
            for cB in range(C):
                if cA != cB:
                    centerA = tf.reduce_mean(tf.boolean_mask(y_pred, tf.equal(y_true, cA)), axis=0)
                    centerB = tf.reduce_mean(tf.boolean_mask(y_pred, tf.equal(y_true, cB)), axis=0)
                    dist_loss += tf.maximum(0.0, delta_d - tf.norm(centerA - centerB)) ** 2
        return dist_loss / (C * (C - 1))

    def Lreg():
        reg_loss = 0.0
        for c in range(C):
            cluster_center = tf.reduce_mean(tf.boolean_mask(y_pred, tf.equal(y_true, c)), axis=0)
            reg_loss += tf.norm(cluster_center)
        return reg_loss / C

    # Number of clusters (classes)
    C = tf.reduce_max(y_true) + 1

    # Compute the three components of the discriminative loss
    var_loss = Lvar()
    dist_loss = Ldist()
    reg_loss = Lreg()

    # Total discriminative loss
    total_loss = alpha * var_loss + beta * dist_loss + gamma * reg_loss
    return total_loss
