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
"""Non-linear estimator tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from tensorflow.contrib.layers.python.layers import feature_column
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.contrib.learn.python.learn.estimators import dnn
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import test


class NonLinearTest(test.TestCase):
  """Non-linear estimator tests."""

  def setUp(self):
    random.seed(42)
    random_seed.set_random_seed(42)

  def testIrisDNN(self):
    iris = base.load_iris()
    feature_columns = [feature_column.real_valued_column("", dimension=4)]
    classifier = dnn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        config=run_config.RunConfig(tf_random_seed=1))
    classifier.fit(iris.data, iris.target, max_steps=200)
    variable_names = classifier.get_variable_names()
    self.assertEqual(
        classifier.get_variable_value("dnn/hiddenlayer_0/weights").shape,
        (4, 10))
    self.assertEqual(
        classifier.get_variable_value("dnn/hiddenlayer_1/weights").shape,
        (10, 20))
    self.assertEqual(
        classifier.get_variable_value("dnn/hiddenlayer_2/weights").shape,
        (20, 10))
    self.assertEqual(
        classifier.get_variable_value("dnn/logits/weights").shape, (10, 3))
    self.assertIn("dnn/hiddenlayer_0/biases", variable_names)
    self.assertIn("dnn/hiddenlayer_1/biases", variable_names)
    self.assertIn("dnn/hiddenlayer_2/biases", variable_names)
    self.assertIn("dnn/logits/biases", variable_names)

  def testBostonDNN(self):
    boston = base.load_boston()
    feature_columns = [feature_column.real_valued_column("", dimension=13)]
    regressor = dnn.DNNRegressor(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        config=run_config.RunConfig(tf_random_seed=1))
    regressor.fit(boston.data,
                  boston.target,
                  steps=300,
                  batch_size=boston.data.shape[0])
    weights = ([regressor.get_variable_value("dnn/hiddenlayer_0/weights")] +
               [regressor.get_variable_value("dnn/hiddenlayer_1/weights")] +
               [regressor.get_variable_value("dnn/hiddenlayer_2/weights")] +
               [regressor.get_variable_value("dnn/logits/weights")])
    self.assertEqual(weights[0].shape, (13, 10))
    self.assertEqual(weights[1].shape, (10, 20))
    self.assertEqual(weights[2].shape, (20, 10))
    self.assertEqual(weights[3].shape, (10, 1))

    biases = ([regressor.get_variable_value("dnn/hiddenlayer_0/biases")] +
              [regressor.get_variable_value("dnn/hiddenlayer_1/biases")] +
              [regressor.get_variable_value("dnn/hiddenlayer_2/biases")] +
              [regressor.get_variable_value("dnn/logits/biases")])
    self.assertEqual(biases[0].shape, (10,))
    self.assertEqual(biases[1].shape, (20,))
    self.assertEqual(biases[2].shape, (10,))
    self.assertEqual(biases[3].shape, (1,))

  def testDNNDropout0(self):
    # Dropout prob == 0.
    iris = base.load_iris()
    feature_columns = [feature_column.real_valued_column("", dimension=4)]
    classifier = dnn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        dropout=0.0,
        config=run_config.RunConfig(tf_random_seed=1))
    classifier.fit(iris.data, iris.target, max_steps=200)

  def testDNNDropout0_1(self):
    # Dropping only a little.
    iris = base.load_iris()
    feature_columns = [feature_column.real_valued_column("", dimension=4)]
    classifier = dnn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        dropout=0.1,
        config=run_config.RunConfig(tf_random_seed=1))
    classifier.fit(iris.data, iris.target, max_steps=200)

  def testDNNDropout0_9(self):
    # Dropping out most of it.
    iris = base.load_iris()
    feature_columns = [feature_column.real_valued_column("", dimension=4)]
    classifier = dnn.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[10, 20, 10],
        n_classes=3,
        dropout=0.9,
        config=run_config.RunConfig(tf_random_seed=1))
    classifier.fit(iris.data, iris.target, max_steps=200)


if __name__ == "__main__":
  test.main()
