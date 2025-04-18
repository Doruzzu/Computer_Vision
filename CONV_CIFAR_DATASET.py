import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train=x_train/255.0
x_test=x_test/255.0
print(x_train.shape)
print(x_test.shape)

model=tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(512,kernel_size=5, activation='relu', padding='valid', input_shape=(32,32,3)))
model.add(tf.keras.layers.MaxPool2D((2,2), strides=1))

model.add(tf.keras.layers.Conv2D(512,kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2), strides=2))
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2), strides=2))
model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D((2,2), strides=2))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(10))# the output layer
model.summary()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)
test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test[:10])# test the first 10 images
#print(predictions.shape)
predictions[0]

print(np.argmax(predictions, axis = 1))
print(y_test[:10])