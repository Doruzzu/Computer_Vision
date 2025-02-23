import numpy as np
import tensorflow as tf


def create_training_data():

    n_bedrooms = np.array([1, 2, 3, 4, 5, 6], dtype='float')
    price_in_hundreds_of_thousands = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype='float')

    return n_bedrooms, price_in_hundreds_of_thousands


def define_and_compile_model():
    """Returns the compiled (but untrained) model.

    Returns:
        tf.keras.Model: The model that will be trained to predict house prices.
    """


    model = tf.keras.Sequential([
        # Define the Input with the appropriate shape
        tf.keras.Input(shape=(1,)),
        # Define the Dense layer
        tf.keras.layers.Dense(units=1)])

    model.compile(optimizer='sgd', loss='mean_squared_error')


    return model


def train_model():
    """Returns the trained model.

    Returns:
        tf.keras.Model: The trained model that will predict house prices.
    """

    n_bedrooms, price_in_hundreds_of_thousands = create_training_data()

    model = define_and_compile_model()

    # Train the  model for 500 epochs by feeding the training data
    model.fit(n_bedrooms, price_in_hundreds_of_thousands, epochs=500)

    return model

trained_model = train_model()

new_n_bedrooms = np.array([7.0])
predicted_price = trained_model.predict(new_n_bedrooms, verbose=False).item()
print(f"the price predicted for a bedroom with {int(new_n_bedrooms)} is  {predicted_price*100000}")