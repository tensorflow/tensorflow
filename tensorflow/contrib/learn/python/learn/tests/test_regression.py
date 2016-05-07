#  Copyright 2015-present The Scikit Flow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn import datasets


class RegressionTest(tf.test.TestCase):

  def testLinearRegression(self):
    rng = np.random.RandomState(67)
    N = 1000
    n_weights = 10
    self.bias = 2
    self.X = rng.uniform(-1, 1, (N, n_weights))
    self.weights = 10 * rng.randn(n_weights)
    self.y = np.dot(self.X, self.weights)
    self.y += rng.randn(len(self.X)) * 0.05 + rng.normal(self.bias, 0.01)
    regressor = learn.TensorFlowLinearRegressor(optimizer="SGD")
    regressor.fit(self.X, self.y)
    # Have to flatten weights since they come in (X, 1) shape
    self.assertAllClose(self.weights, regressor.weights_.flatten(), rtol=0.01)
    assert abs(self.bias - regressor.bias_) < 0.1


if __name__ == "__main__":
  tf.test.main()
