#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import metrics
import pandas

import tensorflow as tf
from tensorflow.contrib import learn

### Training data

# Downloads, unpacks and reads DBpedia dataset.
dbpedia = learn.datasets.load_dataset('dbpedia')
X_train, y_train = pandas.DataFrame(dbpedia.train.data)[1], pandas.Series(dbpedia.train.target)
X_test, y_test = pandas.DataFrame(dbpedia.test.data)[1], pandas.Series(dbpedia.test.target)

### Process vocabulary

MAX_DOCUMENT_LENGTH = 10

vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
X_train = np.array(list(vocab_processor.fit_transform(X_train)))
X_test = np.array(list(vocab_processor.transform(X_test)))

n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)

### Models

EMBEDDING_SIZE = 50

def average_model(X, y):
    word_vectors = learn.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    features = tf.reduce_max(word_vectors, reduction_indices=1)
    return learn.models.logistic_regression(features, y)

def rnn_model(X, y):
    """Recurrent neural network model to predict from sequence of words
    to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = learn.ops.categorical_variable(X, n_classes=n_words,
        embedding_size=EMBEDDING_SIZE, name='words')
    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = learn.ops.split_squeeze(1, MAX_DOCUMENT_LENGTH, word_vectors)
    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.nn.rnn_cell.GRUCell(EMBEDDING_SIZE)
    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.nn.rnn(cell, word_list, dtype=tf.float32)
    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    return learn.models.logistic_regression(encoding, y)

classifier = learn.TensorFlowEstimator(model_fn=rnn_model, n_classes=15,
    steps=1000, optimizer='Adam', learning_rate=0.01, continue_training=True)

# Continuously train for 1000 steps & predict on test set.
while True:
    classifier.fit(X_train, y_train, logdir='/tmp/tf_examples/word_rnn')
    score = metrics.accuracy_score(y_test, classifier.predict(X_test))
    print('Accuracy: {0:f}'.format(score))
