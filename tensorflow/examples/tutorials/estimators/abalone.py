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
"""DNNRegressor with custom estimator for abalone dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import numpy as np
import tensorflow as tf

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

# Learning rate for the model
LEARNING_RATE = 0.001


def maybe_download(train_data, test_data, predict_data):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_train.csv",
        train_file.name)
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_test.csv", test_file.name)
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)

  if predict_data:
    predict_file_name = predict_data
  else:
    predict_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "http://download.tensorflow.org/data/abalone_predict.csv",
        predict_file.name)
    predict_file_name = predict_file.name
    predict_file.close()
    print("Prediction data is downloaded to %s" % predict_file_name)

  return train_file_name, test_file_name, predict_file_name


def model_fn(features, labels, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features["x"]) with relu activation
  first_hidden_layer = tf.layers.dense(features["x"], 10, activation=tf.nn.relu)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.layers.dense(
      first_hidden_layer, 10, activation=tf.nn.relu)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.layers.dense(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])

  # Provide an estimator spec for `ModeKeys.PREDICT`.
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"ages": predictions})

  # Calculate loss using mean squared error
  loss = tf.losses.mean_squared_error(labels, predictions)

  optimizer = tf.train.GradientDescentOptimizer(
      learning_rate=params["learning_rate"])
  train_op = optimizer.minimize(
      loss=loss, global_step=tf.train.get_global_step())

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(labels, tf.float64), predictions)
  }

  # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load datasets
  abalone_train, abalone_test, abalone_predict = maybe_download(
      FLAGS.train_data, FLAGS.test_data, FLAGS.predict_data)

  # Training examples
  training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_train, target_dtype=np.int, features_dtype=np.float64)

  # Test examples
  test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_test, target_dtype=np.int, features_dtype=np.float64)

  # Set of 7 examples for which to predict abalone ages
  prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_predict, target_dtype=np.int, features_dtype=np.float64)

  # Set model params
  model_params = {"learning_rate": LEARNING_RATE}

  # Instantiate Estimator
  nn = tf.estimator.Estimator(model_fn=model_fn, params=model_params)

  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(training_set.data)},
      y=np.array(training_set.target),
      num_epochs=None,
      shuffle=True)

  # Train
  nn.train(input_fn=train_input_fn, steps=5000)

  # Score accuracy
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": np.array(test_set.data)},
      y=np.array(test_set.target),
      num_epochs=1,
      shuffle=False)

  ev = nn.evaluate(input_fn=test_input_fn)
  print("Loss: %s" % ev["loss"])
  print("Root Mean Squared Error: %s" % ev["rmse"])

  # Print out predictions
  predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": prediction_set.data},
      num_epochs=1,
      shuffle=False)
  predictions = nn.predict(input_fn=predict_input_fn)
  for i, p in enumerate(predictions):
    print("Prediction %s: %s" % (i + 1, p["ages"]))


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--train_data", type=str, default="", help="Path to the training data.")
  parser.add_argument(
      "--test_data", type=str, default="", help="Path to the test data.")
  parser.add_argument(
      "--predict_data",
      type=str,
      default="",
      help="Path to the prediction data.")
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
