import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Sample data
training_sentences = [
    "I love programming!",
    "Python is great.",
    "I hate bugs.",
    "Debugging is frustrating.",
    "I enjoy solving problems."
]
training_labels = [1, 1, 0, 0, 1]  # 1 for positive sentiment, 0 for negative sentiment

# Create a simple model
model = keras.Sequential([
    layers.Embedding(input_dim=1000, output_dim=64),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Convert sentences to sequences
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

# Train the model
model.fit(padded, np.array(training_labels), epochs=10)
