# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Debug the tf-learn iris example, based on the tf-learn tutorial."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python import debug as tf_debug

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "/tmp/iris_data",
                    "Directory to save the training and test data in.")
flags.DEFINE_string("model_dir", "", "Directory to save the trained model in.")
flags.DEFINE_integer("train_steps", 10, "Number of steps to run trainer.")
flags.DEFINE_boolean("debug", False,
                     "Use debugger to track down bad values during training")

# URLs to download data sets from, if necessary.
IRIS_TRAINING_DATA_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/monitors/iris_training.csv"
IRIS_TEST_DATA_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/monitors/iris_test.csv"


def maybe_download_data():
  """Download data sets if necessary.

  Returns:
    Paths to the training and test data files.
  """

  if not os.path.isdir(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  training_data_path = os.path.join(FLAGS.data_dir,
                                    os.path.basename(IRIS_TRAINING_DATA_URL))
  if not os.path.isfile(training_data_path):
    train_file = open(training_data_path, "wt")
    urllib.request.urlretrieve(IRIS_TRAINING_DATA_URL, train_file.name)
    train_file.close()

    print("Training data are downloaded to %s" % train_file.name)

  test_data_path = os.path.join(FLAGS.data_dir,
                                os.path.basename(IRIS_TEST_DATA_URL))
  if not os.path.isfile(test_data_path):
    test_file = open(test_data_path, "wt")
    urllib.request.urlretrieve(IRIS_TEST_DATA_URL, test_file.name)
    test_file.close()

    print("Test data are downloaded to %s" % test_file.name)

  return training_data_path, test_data_path


def main(_):
  training_data_path, test_data_path = maybe_download_data()

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=training_data_path,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=test_data_path, target_dtype=np.int, features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  model_dir = FLAGS.model_dir or tempfile.mkdtemp(prefix="debug_tflearn_iris_")

  classifier = tf.contrib.learn.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=[10, 20, 10],
      n_classes=3,
      model_dir=model_dir)

  monitors = [tf_debug.LocalCLIDebugHook()] if FLAGS.debug else None

  # Fit model.
  classifier.fit(x=training_set.data,
                 y=training_set.target,
                 steps=FLAGS.train_steps,
                 monitors=monitors)

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(
      x=test_set.data, y=test_set.target)["accuracy"]
  # TODO(cais): Add debug monitor for evaluate()

  print("After training %d steps, Accuracy = %f" %
        (FLAGS.train_steps, accuracy_score))


if __name__ == "__main__":
  tf.app.run()
