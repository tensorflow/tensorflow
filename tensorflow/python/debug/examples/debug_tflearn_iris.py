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

from six.moves import urllib
import tensorflow as tf

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
    def training_input_fn():
      return ({"features": tf.random_normal([128, 4])},
              tf.random_uniform([128], minval=0, maxval=3, dtype=tf.int32))
    def test_input_fn():
      return ({"features": tf.random_normal([32, 4])},
              tf.random_uniform([32], minval=0, maxval=3, dtype=tf.int32))
    feature_columns = [
        tf.feature_column.numeric_column("features", shape=(4,))]
  else:
    training_data_path, test_data_path = maybe_download_data(FLAGS.data_dir)
    column_names = [
        "sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
    batch_size = 32
    def training_input_fn():
      return tf.contrib.data.make_csv_dataset(
          [training_data_path], batch_size,
          column_names=column_names, label_name="label")
    def test_input_fn():
      return tf.contrib.data.make_csv_dataset(
          [test_data_path], batch_size,
          column_names=column_names, label_name="label")
    feature_columns = [tf.feature_column.numeric_column(feature)
                       for feature in column_names[:-1]]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  model_dir = FLAGS.model_dir or tempfile.mkdtemp(prefix="debug_tflearn_iris_")

  classifier = tf.estimator.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=[10, 20, 10],
      n_classes=3,
      model_dir=model_dir)

  hooks = None
  if FLAGS.debug and FLAGS.tensorboard_debug_address:
    raise ValueError(
        "The --debug and --tensorboard_debug_address flags are mutually "
        "exclusive.")
  if FLAGS.debug:
    debug_hook = tf_debug.LocalCLIDebugHook(ui_type=FLAGS.ui_type,
                                            dump_root=FLAGS.dump_root)
  elif FLAGS.tensorboard_debug_address:
    debug_hook = tf_debug.TensorBoardDebugHook(FLAGS.tensorboard_debug_address)
  hooks = [debug_hook]

  # Train model, using tfdbg hook.
  classifier.train(training_input_fn,
                   steps=FLAGS.train_steps,
                   hooks=hooks)

  # Evaluate accuracy, using tfdbg hook.
  accuracy_score = classifier.evaluate(test_input_fn,
                                       steps=FLAGS.eval_steps,
                                       hooks=hooks)["accuracy"]

  print("After training %d steps, Accuracy = %f" %
        (FLAGS.train_steps, accuracy_score))

  # Make predictions, using tfdbg hook.
  predict_results = classifier.predict(test_input_fn, hooks=hooks)
  print("A prediction result: %s" % next(predict_results))


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
      help="Number of steps to run training for.")
  parser.add_argument(
      "--eval_steps",
      type=int,
      default=1,
      help="Number of steps to run evaluation foir.")
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
      help="Use debugger to track down bad values during training. "
      "Mutually exclusive with the --tensorboard_debug_address flag.")
  parser.add_argument(
      "--dump_root",
      type=str,
      default="",
      help="Optional custom root directory for temporary debug dump data")
  parser.add_argument(
      "--tensorboard_debug_address",
      type=str,
      default=None,
      help="Connect to the TensorBoard Debugger Plugin backend specified by "
      "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
      "--debug flag.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
