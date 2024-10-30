import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class LrFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr=1e-6, max_lr=1, num_steps=100):
        """
        Callback for learning rate range test.
        
        Args:
            min_lr (float): The starting learning rate.
            max_lr (float): The maximum learning rate.
            num_steps (int): The number of steps (batches) over which the learning rate is increased.
        """
        super(LrFinder, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.lr_multiplier = (max_lr / min_lr) ** (1 / num_steps)
        self.lrs = []
        self.losses = []
        self.best_loss = np.inf
        self.peak_lr = None  # To store the learning rate at the peak of the second derivative

    def on_train_batch_begin(self, batch, logs=None):
        lr = self.min_lr * (self.lr_multiplier ** len(self.lrs))
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrs.append(lr)

    def on_train_batch_end(self, batch, logs=None):
        loss = logs['loss']
        self.losses.append(loss)
        
        # Track the best loss to detect divergence
        if loss < self.best_loss:
            self.best_loss = loss
        elif loss > 4 * self.best_loss:
            self.model.stop_training = True
            print("Stopping early due to rapid loss increase.")
    
    def plot(self):
        # Calculate the second derivative of the loss
        first_derivative = np.gradient(self.losses, self.lrs)
        second_derivative = np.gradient(first_derivative, self.lrs)
        
        # Find the index of the maximum second derivative
        peak_index = np.argmax(second_derivative)
        self.peak_lr = self.lrs[peak_index]  # Learning rate at the peak of the second derivative
        
        # Plot original loss and second derivative in parallel
        fig, ax1 = plt.subplots()

        # Plot the loss
        ax1.plot(self.lrs, self.losses, color="blue", label="Loss")
        ax1.set_xscale('log')
        ax1.set_xlabel("Learning Rate")
        ax1.set_ylabel("Loss", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")

        # Plot the second derivative
        ax2 = ax1.twinx()
        ax2.plot(self.lrs, second_derivative, color="red", label="Second Derivative")
        ax2.set_ylabel("Second Derivative", color="red")
        ax2.tick_params(axis='y', labelcolor="red")

        plt.title("Learning Rate Finder with Second Derivative")
        fig.tight_layout()
        plt.show()
        
        # Return the learning rate at the peak of the second derivative
        return self.peak_lr

def lr_finder(model, train_data, min_lr=1e-6, max_lr=1, num_steps=100):
    """
    Function to use the LrFinder callback and visualize results.

    Args:
        model (tf.keras.Model): The Keras model to train.
        train_data (tf.data.Dataset): The training dataset.
        min_lr (float): The minimum learning rate.
        max_lr (float): The maximum learning rate.
        num_steps (int): The number of steps to run the test.
    """
    lr_finder_callback = LrFinder(min_lr=min_lr, max_lr=max_lr, num_steps=num_steps)
    print(f"Entering Learning Rate Finder, min value: {min_lr}, max value: {max_lr}")
    model.fit(train_data, epochs=1, steps_per_epoch=num_steps, callbacks=[lr_finder_callback])
    peak_lr = lr_finder_callback.plot()
    print(f"Recommended Learning Rate (at peak of second derivative): {peak_lr}")
    return peak_lr
