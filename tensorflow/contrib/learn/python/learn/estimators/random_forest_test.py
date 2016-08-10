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

"""Tests for TensorForestTrainer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class TensorForestTrainerTests(tf.test.TestCase):

  def testFloat64(self):
    hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_trees=3, max_nodes=1000, num_classes=3, num_features=4)
    classifier = tf.contrib.learn.TensorForestEstimator(hparams)
    iris = tf.contrib.learn.datasets.load_iris()
    with self.assertRaisesRegexp(TypeError, 'float32'):
      classifier.fit(x=iris.data, y=iris.target, steps=100)

  def testClassification(self):
    """Tests multi-class classification using matrix data as input."""
    hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_trees=3, max_nodes=1000, num_classes=3, num_features=4)
    classifier = tf.contrib.learn.TensorForestEstimator(hparams)

    iris = tf.contrib.learn.datasets.load_iris()
    data = iris.data.astype(np.float32)
    target = iris.target.astype(np.float32)

    classifier.fit(x=data, y=target, steps=100)
    classifier.evaluate(x=data, y=target, steps=10)

  def testRegression(self):
    """Tests multi-class classification using matrix data as input."""

    hparams = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_trees=3, max_nodes=1000, num_classes=1, num_features=13,
        regression=True)

    regressor = tf.contrib.learn.TensorForestEstimator(hparams)

    boston = tf.contrib.learn.datasets.load_boston()
    data = boston.data.astype(np.float32)
    target = boston.target.astype(np.float32)

    regressor.fit(x=data, y=target, steps=100)
    regressor.evaluate(x=data, y=target, steps=10)


if __name__ == '__main__':
  tf.test.main()
