import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
import numpy as np

# Load the dataset with 4 selected categories
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

# Preprocess data using TF-IDF vectorizer (convert text to numerical)
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(newsgroups_train.data).toarray()
X_test_tfidf = vectorizer.transform(newsgroups_test.data).toarray()

# Convert target labels to binary format using LabelBinarizer
lb = LabelBinarizer()
y_train_bin = lb.fit_transform(newsgroups_train.target)
y_test_bin = lb.transform(newsgroups_test.target)

# Define Logistic Regression model in TensorFlow using Keras
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(X_train_tfidf.shape[1],)),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 classes for sentiment
])

# Compile the model (multi-class classification)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',    # use categorical crossentropy for multi-class
              metrics=['accuracy'])

# Train the model with 20% of data used for validation
model.fit(X_train_tfidf, y_train_bin, epochs=20, batch_size=64, validation_split=0.2)

# Evaluate model on test data and Print accuracy
loss, accuracy = model.evaluate(X_test_tfidf, y_test_bin)
print(f"Test Accuracy: {accuracy}")