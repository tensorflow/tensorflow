import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import numpy as np  # Import NumPy to handle arrays

# Sample Data
sentences = [
    'I love machine learning',
    'TensorFlow is awesome!',
    'Deep learning is amazing',
    'I dislike bugs in the code',
    'This is so difficult'
]
labels = [1, 1, 1, 0, 0]  # 1: Positive, 0: Negative

# Convert labels to a NumPy array (ensure they are of type float32)
labels = np.array(labels, dtype=np.float32)

# Tokenizing and Padding Sequences
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

# Convert padded sequences to a NumPy array (ensure they are of type float32)
padded = np.array(padded, dtype=np.float32)

# Create a TensorFlow model
model = Sequential([
    Embedding(input_dim=1000, output_dim=64),  # input_dim: size of the vocabulary
    LSTM(64),
    Dense(1, activation='sigmoid')  # Use sigmoid for binary classification
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model (for demonstration purposes, small sample size)
model.fit(padded, labels, epochs=10)

# Test with a new sentence
test_sentence = ['I hate debugging!']
test_sequence = tokenizer.texts_to_sequences(test_sentence)
test_padded = pad_sequences(test_sequence, padding='post', maxlen=padded.shape[1])
test_padded = np.array(test_padded, dtype=np.float32)  # Convert to NumPy array

# Make prediction
prediction = model.predict(test_padded)

# Output the prediction
if prediction > 0.5:
    print("Positive sentiment")
else:
    print("Negative sentiment")

