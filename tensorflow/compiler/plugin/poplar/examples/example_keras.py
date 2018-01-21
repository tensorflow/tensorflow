import tensorflow as tf
import numpy as np

# Set variables to resource variables
vscope = tf.get_variable_scope()
vscope.set_use_resource(True)

# Create Keras model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu', input_dim=100))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

# Generate dummy data
labels = np.random.randint(2, size=(1000, 1))
data = np.random.normal(0.0, 5.0, [1000,100])
data += (labels * 6.0) - 3.0

# Train the model, iterating on the data in batches of 32 samples
#model.fit(data, labels, epochs=10, batch_size=32)

#print model.evaluate(data, labels, batch_size=32)
model.predict(data, batch_size=32)
