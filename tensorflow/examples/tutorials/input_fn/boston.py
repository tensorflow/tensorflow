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
"""DNNRegressor with custom input_fn for Housing dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

COLUMNS = ["crim", "zn", "indus", "nox", "rm", "age",
           "dis", "tax", "ptratio", "medv"]
FEATURES = ["crim", "zn", "indus", "nox", "rm",
            "age", "dis", "tax", "ptratio"]
LABEL = "medv"


def input_fn(data_set):
  feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels


def main(unused_argv):
  # Load datasets
  training_set = pd.read_csv("boston_train.csv", skipinitialspace=True,
                             skiprows=1, names=COLUMNS)
  test_set = pd.read_csv("boston_test.csv", skipinitialspace=True,
                         skiprows=1, names=COLUMNS)

  # Set of 6 examples for which to predict median house values
  prediction_set = pd.read_csv("boston_predict.csv", skipinitialspace=True,
                               skiprows=1, names=COLUMNS)

  # Feature cols
  feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

  # Build 2 layer fully connected DNN with 10, 10 units respectively.
  regressor = tf.contrib.learn.DNNRegressor(
      feature_columns=feature_cols, hidden_units=[10, 10])

  # Fit
  regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

  # Score accuracy
  ev = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
  loss_score = ev["loss"]
  print("Loss: {0:f}".format(loss_score))

  # Print out predictions
  y = regressor.predict(input_fn=lambda: input_fn(prediction_set))
  print("Predictions: {}".format(str(y)))

if __name__ == "__main__":
  tf.app.run()
