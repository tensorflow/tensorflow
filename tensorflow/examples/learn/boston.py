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
from sklearn import cross_validation
from sklearn import metrics
from sklearn import preprocessing
import tensorflow as tf


def main(unused_argv):
  # Load dataset
  boston = tf.contrib.learn.datasets.load_dataset('boston')
  x, y = boston.data, boston.target

  # Split dataset into train / test
  x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      x, y, test_size=0.2, random_state=42)

  # Scale data (training set) to 0 mean and unit standard deviation.
  scaler = preprocessing.StandardScaler()
  x_train = scaler.fit_transform(x_train)

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
      x_train)
  regressor = tf.contrib.learn.DNNRegressor(
      feature_columns=feature_columns, hidden_units=[10, 10])

  # Fit
  regressor.fit(x_train, y_train, steps=5000, batch_size=1)

  # Predict and score
  y_predicted = list(
      regressor.predict(
          scaler.transform(x_test), as_iterable=True))
  score = metrics.mean_squared_error(y_predicted, y_test)

  print('MSE: {0:f}'.format(score))


if __name__ == '__main__':
  tf.app.run()
