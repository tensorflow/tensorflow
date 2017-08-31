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
"""Example of DNNRegressor for Housing dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

import tensorflow as tf


def main(unused_argv):
  # Load dataset
  boston = datasets.load_boston()
  x, y = boston.data, boston.target

  # Split dataset into train / test
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      x, y, test_size=0.2, random_state=42)

  # Scale data (training set) to 0 mean and unit standard deviation.
  scaler = preprocessing.StandardScaler()
  x_train = scaler.fit_transform(x_train)

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  feature_columns = [
      tf.feature_column.numeric_column('x', shape=np.array(x_train).shape[1:])]
  regressor = tf.estimator.DNNRegressor(
      feature_columns=feature_columns, hidden_units=[10, 10])

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_train}, y=y_train, batch_size=1, num_epochs=None, shuffle=True)
  regressor.train(input_fn=train_input_fn, steps=2000)

  # Predict.
  x_transformed = scaler.transform(x_test)
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={'x': x_transformed}, y=y_test, num_epochs=1, shuffle=False)
  predictions = regressor.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['predictions'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
  score_sklearn = metrics.mean_squared_error(y_predicted, y_test)
  print('MSE (sklearn): {0:f}'.format(score_sklearn))

  # Score with tensorflow.
  scores = regressor.evaluate(input_fn=test_input_fn)
  print('MSE (tensorflow): {0:f}'.format(scores['average_loss']))


if __name__ == '__main__':
  tf.app.run()
