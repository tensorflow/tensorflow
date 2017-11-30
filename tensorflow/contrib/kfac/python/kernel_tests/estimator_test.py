# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.contrib.kfac.estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kfac.python.ops import estimator
from tensorflow.contrib.kfac.python.ops import layer_collection as lc
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test


class EstimatorTest(test.TestCase):

  def testEstimatorInitManualRegistration(self):
    with ops.Graph().as_default():
      layer_collection = lc.LayerCollection()

      inputs = random_ops.random_normal((2, 2), dtype=dtypes.float32)
      weights = variable_scope.get_variable(
          'w', shape=(2, 2), dtype=dtypes.float32)
      bias = variable_scope.get_variable(
          'b', initializer=init_ops.zeros_initializer(), shape=(2, 1))
      output = math_ops.matmul(inputs, weights) + bias

      # Only register the weights.
      layer_collection.register_fully_connected((weights,), inputs, output)

      outputs = math_ops.tanh(output)
      layer_collection.register_categorical_predictive_distribution(outputs)

      # We should be able to build an estimator for only the registered vars.
      estimator.FisherEstimator([weights], 0.1, 0.2, layer_collection)

      # Check that we throw an error if we try to build an estimator for vars
      # that were not manually registered.
      with self.assertRaises(ValueError):
        estimator.FisherEstimator([weights, bias], 0.1, 0.2, layer_collection)


if __name__ == '__main__':
  test.main()
