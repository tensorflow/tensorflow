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

import argparse
import os
import sys
import tempfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import experiment
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python import debug as tf_debug


# URLs to download data sets from, if necessary.
IRIS_TRAINING_DATA_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/monitors/iris_training.csv"
IRIS_TEST_DATA_URL = "https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/examples/tutorials/monitors/iris_test.csv"


def maybe_download_data(data_dir):
  """Download data sets if necessary.

  Args:
    data_dir: Path to where data should be downloaded.

  Returns:
    Paths to the training and test data files.
  """

  if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

  training_data_path = os.path.join(data_dir,
                                    os.path.basename(IRIS_TRAINING_DATA_URL))
  if not os.path.isfile(training_data_path):
    train_file = open(training_data_path, "wt")
    urllib.request.urlretrieve(IRIS_TRAINING_DATA_URL, train_file.name)
    train_file.close()

    print("Training data are downloaded to %s" % train_file.name)

  test_data_path = os.path.join(data_dir, os.path.basename(IRIS_TEST_DATA_URL))
  if not os.path.isfile(test_data_path):
    test_file = open(test_data_path, "wt")
    urllib.request.urlretrieve(IRIS_TEST_DATA_URL, test_file.name)
    test_file.close()

    print("Test data are downloaded to %s" % test_file.name)

  return training_data_path, test_data_path


_IRIS_INPUT_DIM = 4


def iris_input_fn():
  iris = base.load_iris()
  features = tf.reshape(tf.constant(iris.data), [-1, _IRIS_INPUT_DIM])
  labels = tf.reshape(tf.constant(iris.target), [-1])
  return features, labels


def main(_):
  # Load datasets.
  if FLAGS.fake_data:
    training_set = tf.contrib.learn.datasets.base.Dataset(
        np.random.random([120, 4]),
        np.random.random_integers(3, size=[120]) - 1)
    test_set = tf.contrib.learn.datasets.base.Dataset(
        np.random.random([30, 4]),
        np.random.random_integers(3, size=[30]) - 1)
  else:
    training_data_path, test_data_path = maybe_download_data(FLAGS.data_dir)
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

  hooks = None
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook(ui_type=FLAGS.ui_type)
    debug_hook.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    hooks = [debug_hook]

  if not FLAGS.use_experiment:
    # Fit model.
    classifier.fit(x=training_set.data,
                   y=training_set.target,
                   steps=FLAGS.train_steps,
                   monitors=hooks)

    # Evaluate accuracy.
    accuracy_score = classifier.evaluate(x=test_set.data,
                                         y=test_set.target,
                                         hooks=hooks)["accuracy"]
  else:
    ex = experiment.Experiment(classifier,
                               train_input_fn=iris_input_fn,
                               eval_input_fn=iris_input_fn,
                               train_steps=FLAGS.train_steps,
                               eval_delay_secs=0,
                               eval_steps=1,
                               train_monitors=hooks,
                               eval_hooks=hooks)
    ex.train()
    accuracy_score = ex.evaluate()["accuracy"]

  print("After training %d steps, Accuracy = %f" %
        (FLAGS.train_steps, accuracy_score))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--data_dir",
      type=str,
      default="/tmp/iris_data",
      help="Directory to save the training and test data in.")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="",
      help="Directory to save the trained model in.")
  parser.add_argument(
      "--train_steps",
      type=int,
      default=10,
      help="Number of steps to run trainer.")
  parser.add_argument(
      "--use_experiment",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Use tf.contrib.learn Experiment to run training and evaluation")
  parser.add_argument(
      "--ui_type",
      type=str,
      default="curses",
      help="Command-line user interface type (curses | readline)")
  parser.add_argument(
      "--fake_data",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Use fake MNIST data for unit testing")
  parser.add_argument(
      "--debug",
      type="bool",
      nargs="?",
      const=True,
      default=False,
      help="Use debugger to track down bad values during training")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
