#  Copyright 2015 Google Inc. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import csv
import numpy as np
from sklearn import metrics

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import skflow

### Training data

# Download dbpedia_csv.tar.gz from
# https://drive.google.com/folderview?id=0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M
# Unpack: tar -xvf dbpedia_csv.tar.gz

def load_dataset(filename):
    target = []
    data = []
    reader = csv.reader(open(filename), delimiter=',')
    for line in reader:
        target.append(int(line[0]))
        data.append(line[2])
    return data, np.array(target, np.float32)

X_train, y_train = load_dataset('dbpedia_csv/train.csv')
X_test, y_test = load_dataset('dbpedia_csv/test.csv')

### Process vocabulary

MAX_DOCUMENT_LENGTH = 10

vocab_processor = skflow.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

### Models

EMBEDDING_SIZE = 50

def average_model(X, y):
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    features = tf.reduce_max(word_vectors, reduction_indices=1)
    return skflow.models.logistic_regression(features, y)

def rnn_model(X, y):
    word_vectors = skflow.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    word_list = [tf.squeeze(w, [1]) for w in tf.split(1, MAX_DOCUMENT_LENGTH, word_vectors)]
    cell = rnn_cell.GRUCell(EMBEDDING_SIZE)
    _, encoding = rnn.rnn(cell, word_list, dtype=tf.float32)
    return skflow.models.logistic_regression(encoding[-1], y)

classifier = skflow.TensorFlowEstimator(model_fn=rnn_model, n_classes=15,
    steps=1000, optimizer='Adam', learning_rate=0.01, continue_training=True)

# Continuesly train for 1000 steps & predict on test set.
while True:
    classifier.fit(X_train, y_train)
    score = metrics.accuracy_score(classifier.predict(X_test), y_test)
    print('Accuracy: {0:f}'.format(score))
