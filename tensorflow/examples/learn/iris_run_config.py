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

"""Example of DNNClassifier for Iris plant dataset, with run config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
import tensorflow as tf


X_FEATURE = 'x'  # Name of the input feature.


def main(unused_argv):
  # Load dataset.
  iris = datasets.load_iris()
  x_train, x_test, y_train, y_test = model_selection.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

  # You can define you configurations by providing a RunConfig object to
  # estimator to control session configurations, e.g. tf_random_seed.
  run_config = tf.estimator.RunConfig().replace(tf_random_seed=1)

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  feature_columns = [
      tf.feature_column.numeric_column(
          X_FEATURE, shape=np.array(x_train).shape[1:])]
  classifier = tf.estimator.DNNClassifier(
      feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3,
      config=run_config)

  # Train.
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: x_train}, y=y_train, num_epochs=None, shuffle=True)
  classifier.train(input_fn=train_input_fn, steps=200)

  # Predict.
  test_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={X_FEATURE: x_test}, y=y_test, num_epochs=1, shuffle=False)
  predictions = classifier.predict(input_fn=test_input_fn)
  y_predicted = np.array(list(p['class_ids'] for p in predictions))
  y_predicted = y_predicted.reshape(np.array(y_test).shape)

  # Score with sklearn.
  score = metrics.accuracy_score(y_test, y_predicted)
  print('Accuracy (sklearn): {0:f}'.format(score))

  # Score with tensorflow.
  scores = classifier.evaluate(input_fn=test_input_fn)
  print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))


if __name__ == '__main__':
  tf.app.run()
