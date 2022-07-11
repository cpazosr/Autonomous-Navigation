# Original File          :   pokenet.py (The Pokenet CNN model description)
# Original Author        :   Ricardo Acevedo-Avila (racevedoaa@gmail.com)
# Version                :   1.0.2
# Description            :   Script that builds the Pokenet
# Based on: https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
# License                :   MIT
##############################################################################################################
# Modiffied File         :   sign_detector_final.py (Sign detector CNN model description)
# Date:                  :   June 17, 2022
# Adaptation Authors     :   Manuel Agustin Diaz & Carlos Antonio Pazos
# Description            :   Script that builds the Sign detector


# Import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
import tensorflow as tf


class sign_detector:
    @staticmethod
    # The build method accepts four parameters: the image dimensions, depth,
    # and number of classes in the dataset.
    def build(width, height, depth, classes):
        # Build and initialize the model/network:
        model = Sequential()

        # Set the input axis order (channel order):
        inputShape = (height, width, depth)
        chanDim = -1

        # Let's add the first set of layers to the
        # Network: CONV => RELU => BN => CONV => RELU => CONV => RELU => POOL

        # First layer, convolution (filtering) with 128
        # kernels, size of (5, 5)
        model.add(Conv2D(128, (5, 5), input_shape=inputShape))

        # Next layer, the activation layer with a ReLU function:
        model.add(Activation("relu"))

        # Batch normalization applies a transformation that
        # maintains the mean output close to 0 and the output
        # standard deviation close to 1:
        # Normalize the pixel color or "depth":
        model.add(BatchNormalization(axis=chanDim))

        # Let's add the second set of layers to the
        # Convolution (filtering) with 64 kernels, size of (5, 5):
        model.add(Conv2D(64, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(32, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # Max pooling:
        # Max pooled with a kernel of size (2,2)
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Second set of layers
        # Network: CONV => RELU => BN => POOL

        # Convolution (filtering) with 32 kernels, size of (3, 3):
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        # Max pooling:
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))

        # Flatten the mat into a vector:
        model.add(Flatten())

        # Implement the fully connected layer with N neurons
        # N is a tunable hyper parameter:
        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.25))

        model.add(Flatten())

        # Finally, the softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        model.compile(optimizer=tf.keras.optimizers.RMSprop(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.FalseNegatives()])

        # return the constructed network architecture
        return model
