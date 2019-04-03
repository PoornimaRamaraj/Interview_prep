#Build a neural network that predicts the price of a house according to a simple formula.
#Imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.
#Create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

#Import all packages

import tensorflow as tf
import numpy as np
from tensorflow import keras # you can import keras directly if installed or you can import from tensorflow

# Define a model with one dense layer and 1D input shape
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#Compile the model with optimizer and loss function
model.compile(optimizer='sgd', loss='mean_squared_error')

#Give the inputs and the labels
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

#Fir the model to the inputs 
model.fit(xs, ys, epochs=1000)

#Predict on unseen value
print(model.predict([7.0]))
