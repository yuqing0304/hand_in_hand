import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

# emmm 今天不想写了 :(
# 写下来为了以后可以有回忆 :) Every day with you is filled with happy moments and sweet memories.
# Let me wake up mext to you, have breakfast in the morning and wander through the city
# with your hand in mine, and I'll be happy for the rest of my life.

'''
setting up some methods to load MNIST from keras.datasets and 
preprocess them into rows of normalized 784-dimensional vectors.
'''
def mnist_dataset():
    # samples and classes(labels)
    # (x, y) = (train_images, train_labels)
    # _ = (test_images, test_labels)
    # x.shape = (60000, 28, 28)
    # x.dtype = uint8
    # len(y) = 60000
    # y = array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
    (x, y), _ = datasets.mnist.load_data()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(prepare_mnist_features_and_labels)
    ds = ds.take(20000).shuffle(20000).batch(100)
    return ds

def prepare_mnist_features_and_labels(x, y):
    # scale the data so that all values are in the [0,1] interval
    # transform it into a float32 array with values between 0 and 1
    x = tf.cast(x, tf.float32) / 255.0
    # labels: int64
    y = tf.cast(y, tf.int64)
    return x, y

'''
The network architecture
Layer: a data-processing module that extracts representations out of the data fed into them.

A loss function: How the network will be able to measure its performance on the training data, 
and thus how it will be able to steer itself in the right direc- tion.

An optimizer: The mechanism through which the network will update itself based on the data 
it sees and its loss function.

Metrics to monitor during training and testing: 
Here, accuracy (the fraction of the images that were correctly classified).
'''

# build the network as a keras.Sequential model
# and instantiate an ADAM optimizer from keras.optimizers.
model = keras.Sequential([
    layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    layers.Dense(100, activation='relu'),
    layers.Dense(100, activation='relu'),
    layers.Dense(10)])

optimizer = optimizers.Adam()