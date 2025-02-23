import os
import base64
import tensorflow as tf

current_dir = os.getcwd()

# Append data/mnist.npz to the previous path to get the full path
#data_path = os.path.join(current_dir, "data/mnist.npz")

# Load data (discard test set)
(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path=r"C:\Users\Doru\Desktop\TensorFlow\MINST\Files\Files\mnist.npz")

print(f"training_images is of type {type(training_images)}.\ntraining_labels is of type {type(training_labels)}\n")

# Inspect shape of the data
data_shape = training_images.shape

print(f"There are {data_shape[0]} examples with shape ({data_shape[1]}, {data_shape[2]})")
training_images = training_images / 255.0


def create_and_compile_model():
    """Returns the compiled (but untrained) model.

    Returns:
        tf.keras.Model: The model that will be trained to predict predict handwriting digits.
    """

    ### START CODE HERE ###

    # Define the model
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    ])

    ### END CODE HERE ###

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

untrained_model = create_and_compile_model()


class EarlyStoppingCallback(tf.keras.callbacks.Callback):


    def on_epoch_end(self, epoch, logs=None):
        # Check if the accuracy is greater or equal to 0.98
        if logs['accuracy'] >= 0.98:
            # Stop training once the above condition is met
            self.model.stop_training = True

            print("\nReached 98% accuracy so cancelling training!")


def train_mnist(training_images, training_labels):
    """Trains a classifier of handwritten digits.

    Args:
        training_images (numpy.ndarray): The images of handwritten digits
        training_labels (numpy.ndarray): The labels of each image

    Returns:
        tf.keras.callbacks.History : The training history.
    """

    model = create_and_compile_model()

    history = model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])


    return history

training_history = train_mnist(training_images, training_labels)