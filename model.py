import numpy as np
import tensorflow as tf
import h5py
from tensorflow import keras
from tensorflow.python.keras import layers as tfl
from tensorflow.python.keras.layers import BatchNormalization
from PIL import Image
from my_utils import *
from tensorflow.python.keras.models import *

def load_h5_dataset(file_path):
    with h5py.File(file_path, 'r') as hf:
        x_train = np.array(hf['train/landmarks'])
        y_train = np.array(hf['train/labels'])
        x_test = np.array(hf['test/landmarks'])
        y_test = np.array(hf['test/labels'])
        
        num_classes = len(hf['class_names'])
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)
        
        return x_train, y_train, x_test, y_test

train_x_orig, train_y, test_x_orig, test_y = load_h5_dataset("hand_gesture_dataset.h5")

m_train = train_x_orig.shape[0]
m_test = test_x_orig.shape[0]
num_px = train_x_orig.shape[1]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

input_shape = (train_x.shape[0],)  # (63,) for 21 landmarks with 3 coordinates each
num_classes = train_y.shape[1]  # Number of classes from one-hot encoding

def convolutional_model(input_shape, num_classes):
    input_img = tf.keras.Input(shape=input_shape)
    model = Sequential([
        # Reshape input: (63,) -> (21, 3) [landmarks × coordinates]
        tfl.Reshape((21, 3), input_shape=input_shape),
        
        # First Conv Block
        tfl.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        tfl.BatchNormalization(),
        tfl.MaxPooling1D(pool_size=2, strides=2),  # Output: 11×64
        
        # Second Conv Block
        tfl.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        tfl.BatchNormalization(),
        tfl.MaxPooling1D(pool_size=2, strides=1),  # Output: 9×128
        
        # Third Conv Block
        tfl.Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        tfl.BatchNormalization(),
        tfl.MaxPooling1D(pool_size=2, strides=1),  # Output: 7×256
        
        # Global Pooling instead of Flatten
        tfl.Conv1D(512, kernel_size=3, activation='relu', padding='valid'),  # Output: 5×512
        tfl.Flatten(),
        
        # Fully Connected Layers
        tfl.Dense(256, activation='relu'),
        tfl.Dropout(0.5),
        tfl.Dense(128, activation='relu'),
        tfl.Dropout(0.3),
        
        # Output Layer
        tfl.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = convolutional_model(input_shape, num_classes)
model.summary()