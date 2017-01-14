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

from sklearn import cross_validation
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf


def main(unused_argv):
  # Load dataset.
  iris = datasets.load_iris()
  x_train, x_test, y_train, y_test = cross_validation.train_test_split(
      iris.data, iris.target, test_size=0.2, random_state=42)

  # You can define you configurations by providing a RunConfig object to
  # estimator to control session configurations, e.g. num_cores
  # and gpu_memory_fraction
  run_config = tf.contrib.learn.estimators.RunConfig(
      num_cores=3, gpu_memory_fraction=0.6)

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(
      x_train)
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              config=run_config)

  # Fit and predict.
  classifier.fit(x_train, y_train, steps=200)
  score = metrics.accuracy_score(y_test, classifier.predict(x_test))
  print('Accuracy: {0:f}'.format(score))


if __name__ == '__main__':
  tf.app.run()
