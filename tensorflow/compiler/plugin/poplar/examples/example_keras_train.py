import tensorflow as tf
import numpy as np

# Generate dummy data
labels = np.random.randint(2, size=(128, 1))
data = np.random.normal(0.0, 5.0, [128,100])
data += (labels * 6.0) - 3.0

# Set variables to resource variables
vscope = tf.get_variable_scope()
vscope.set_use_resource(True)

# Create Keras model
with tf.device("/device:IPU:0"):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(32, activation='relu', input_dim=100))
  model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
  model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=1, batch_size=32, verbose=False)
#print model.evaluate(data, labels, batch_size=32)
