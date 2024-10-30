import tensorflow as tf

class SetLearningRateCallback(tf.keras.callbacks.Callback):
    def __init__(self, peak_lr):
        """
        Callback to set the learning rate to the peak learning rate on the first epoch.
        
        Args:
            peak_lr (float): The learning rate to set at the start of training.
        """
        super(SetLearningRateCallback, self).__init__()
        self.peak_lr = peak_lr

    def on_train_begin(self, logs=None):
        # Set the model's learning rate to the peak learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, self.peak_lr)
        print(f"Learning rate set to peak learning rate: {self.peak_lr}")