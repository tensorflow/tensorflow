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
"""Multi-output tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np

import tensorflow as tf
from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn.estimators._sklearn import mean_squared_error


class MultiOutputTest(tf.test.TestCase):
  """Multi-output tests."""

  def testMultiRegression(self):
    random.seed(42)
    rng = np.random.RandomState(1)
    x = np.sort(200 * rng.rand(100, 1) - 100, axis=0)
    y = np.array([np.pi * np.sin(x).ravel(), np.pi * np.cos(x).ravel()]).T
    regressor = learn.LinearRegressor(
        feature_columns=learn.infer_real_valued_columns_from_input(x),
        target_dimension=2)
    regressor.fit(x, y, steps=100)
    score = mean_squared_error(regressor.predict(x), y)
    self.assertLess(score, 10, "Failed with score = {0}".format(score))


if __name__ == "__main__":
  tf.test.main()
