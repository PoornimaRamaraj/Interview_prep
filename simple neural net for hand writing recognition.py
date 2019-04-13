

# MNIST classifier



# import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# define a custom callback function
class mycallbacks(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs={}):
    if (logs.get('acc')>0.99):
      print("Reached 99% accuracy so cancelling training!")
      self.model.stop_training=True

mnist = tf.keras.datasets.mnist

# Load mnist data
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# rescale images from 0-255 to 0-1
x_train,x_test=x_train/255.0,x_test/255.0

model = tf.keras.models.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                                    keras.layers.Dense(512,activation=tf.nn.relu),
                                                      keras.layers.Dense(10,activation=tf.nn.softmax)])

callbacks=mycallbacks()    

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,y_train,epochs=10,callbacks=[callbacks])


