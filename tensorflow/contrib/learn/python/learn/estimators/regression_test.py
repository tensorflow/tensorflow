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
"""Linear regression tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.learn.python import learn
from tensorflow.python.platform import test


class RegressionTest(test.TestCase):
  """Linear regression tests."""

  def testLinearRegression(self):
    rng = np.random.RandomState(67)
    n = 1000
    n_weights = 10
    bias = 2
    x = rng.uniform(-1, 1, (n, n_weights))
    weights = 10 * rng.randn(n_weights)
    y = np.dot(x, weights)
    y += rng.randn(len(x)) * 0.05 + rng.normal(bias, 0.01)
    regressor = learn.LinearRegressor(
        feature_columns=learn.infer_real_valued_columns_from_input(x),
        optimizer="SGD")
    regressor.fit(x, y, steps=200)
    # Have to flatten weights since they come in (x, 1) shape.
    self.assertAllClose(weights, regressor.weights_.flatten(), rtol=0.01)
    # TODO(ispir): Disable centered_bias.
    # assert abs(bias - regressor.bias_) < 0.1


if __name__ == "__main__":
  test.main()
