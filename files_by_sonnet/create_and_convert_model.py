#!/usr/bin/env python3
"""
Creates a simple deep neural network model using TensorFlow,
trains it on synthetic data, and converts it to TFLite format.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Create a simple deep neural network model
model = keras.Sequential([
    layers.Input(shape=(16,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation=None)  # Output layer
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 2. Generate synthetic data for training
data_x = np.random.randn(1000, 16).astype(np.float32)
data_y = np.sum(data_x, axis=1, keepdims=True).astype(np.float32)

# 3. Train the model
model.fit(data_x, data_y, epochs=5, batch_size=32, verbose=1)

# 4. Convert the trained model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 5. Save the TFLite model
with open('simple_deep_model.tflite', 'wb') as f:
    f.write(tflite_model)

print('TFLite model saved as simple_deep_model.tflite')
