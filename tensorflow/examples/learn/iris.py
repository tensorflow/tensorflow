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
"""Example of DNNClassifier for Iris plant dataset.

This example uses APIs in Tensorflow 1.4 or above.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves.urllib.request import urlretrieve

import tensorflow as tf

# Data sets
IRIS_TRAINING = 'iris_training.csv'
IRIS_TRAINING_URL = 'http://download.tensorflow.org/data/iris_training.csv'

IRIS_TEST = 'iris_test.csv'
IRIS_TEST_URL = 'http://download.tensorflow.org/data/iris_test.csv'

FEATURE_KEYS = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


def maybe_download_iris_data(file_name, download_url):
  """Downloads the file and returns the number of data."""
  if not os.path.exists(file_name):
    urlretrieve(download_url, file_name)

  # The first line is a comma-separated string. The first one is the number of
  # total data in the file.
  with open(file_name, 'r') as f:
    first_line = f.readline()
  num_elements = first_line.split(',')[0]
  return int(num_elements)


def input_fn(file_name, num_data, batch_size, is_training):
  """Creates an input_fn required by Estimator train/evaluate."""
  # If the data sets aren't stored locally, download them.

  def _parse_csv(rows_string_tensor):
    """Takes the string input tensor and returns tuple of (features, labels)."""
    # Last dim is the label.
    num_features = len(FEATURE_KEYS)
    num_columns = num_features + 1
    columns = tf.decode_csv(rows_string_tensor,
                            record_defaults=[[]] * num_columns)
    features = dict(zip(FEATURE_KEYS, columns[:num_features]))
    labels = tf.cast(columns[num_features], tf.int32)
    return features, labels

  def _input_fn():
    """The input_fn."""
    dataset = tf.data.TextLineDataset([file_name])
    # Skip the first line (which does not have data).
    dataset = dataset.skip(1)
    dataset = dataset.map(_parse_csv)

    if is_training:
      # For this small dataset, which can fit into memory, to achieve true
      # randomness, the shuffle buffer size is set as the total number of
      # elements in the dataset.
      dataset = dataset.shuffle(num_data)
      dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

  return _input_fn


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  num_training_data = maybe_download_iris_data(
      IRIS_TRAINING, IRIS_TRAINING_URL)
  num_test_data = maybe_download_iris_data(IRIS_TEST, IRIS_TEST_URL)

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  feature_columns = [
      tf.feature_column.numeric_column(key, shape=1) for key in FEATURE_KEYS]
  classifier = tf.estimator.DNNClassifier(
      feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3)

  # Train.
  train_input_fn = input_fn(IRIS_TRAINING, num_training_data, batch_size=32,
                            is_training=True)
  classifier.train(input_fn=train_input_fn, steps=400)

  # Eval.
  test_input_fn = input_fn(IRIS_TEST, num_test_data, batch_size=32,
                           is_training=False)
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  tf.app.run()
