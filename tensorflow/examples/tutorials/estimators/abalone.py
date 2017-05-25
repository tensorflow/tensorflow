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
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

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


def model_fn(features, targets, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features) with relu activation
  first_hidden_layer = tf.contrib.layers.relu(features, 10)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 10)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}

  # Calculate loss using mean squared error
  loss = tf.losses.mean_squared_error(targets, predictions)

  # Calculate root mean squared error as additional eval metric
  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(targets, tf.float64), predictions)
  }

  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer="SGD")

  return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
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
  nn = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params)
  
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)
    return x, y
  
  # Fit
  nn.fit(input_fn=get_train_inputs, steps=5000)

  # Score accuracy
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)
    return x, y
  
  ev = nn.evaluate(input_fn=get_test_inputs, steps=1)
  print("Loss: %s" % ev["loss"])
  print("Root Mean Squared Error: %s" % ev["rmse"])

  # Print out predictions
  predictions = nn.predict(x=prediction_set.data, as_iterable=True)
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
