#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool('test_with_fake_data', False,
                         'Test the example code with fake data.')

MAX_DOCUMENT_LENGTH = 10
EMBEDDING_SIZE = 50
n_words = 0


def input_op_fn(x):
  """Customized function to transform batched x into embeddings."""
  # Convert indexes of words into embeddings.
  # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
  # maps word indexes of the sequence into [batch_size, sequence_length,
  # EMBEDDING_SIZE].
  word_vectors = learn.ops.categorical_variable(x, n_classes=n_words,
      embedding_size=EMBEDDING_SIZE, name='words')
  # Split into list of embedding per word, while removing doc length dim.
  # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
  word_list = tf.unpack(word_vectors, axis=1)
  return word_list


def main(unused_argv):
  global n_words
  # Prepare training and testing data
  dbpedia = learn.datasets.load_dataset(
      'dbpedia', test_with_fake_data=FLAGS.test_with_fake_data)
  x_train = pandas.DataFrame(dbpedia.train.data)[1]
  y_train = pandas.Series(dbpedia.train.target)
  x_test = pandas.DataFrame(dbpedia.test.data)[1]
  y_test = pandas.Series(dbpedia.test.target)

  # Process vocabulary
  vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(vocab_processor.fit_transform(x_train)))
  x_test = np.array(list(vocab_processor.transform(x_test)))
  n_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % n_words)

  # Build model: a single direction GRU with a single layer
  classifier = learn.TensorFlowRNNClassifier(
      rnn_size=EMBEDDING_SIZE, n_classes=15, cell_type='gru',
      input_op_fn=input_op_fn, num_layers=1, bidirectional=False,
      sequence_length=None, steps=1000, optimizer='Adam',
      learning_rate=0.01, continue_training=True)

  # Train and predict
  classifier.fit(x_train, y_train, steps=100)
  y_predicted = classifier.predict(x_test)
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
  tf.app.run()
