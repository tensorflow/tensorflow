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
"""Tests for utility functions in pruning_utils.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.model_pruning.python import pruning_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class PruningUtilsTest(test.TestCase):

  def _compare_cdf(self, values):
    abs_values = math_ops.abs(values)
    max_value = math_ops.reduce_max(abs_values)
    with self.cached_session():
      variables.global_variables_initializer().run()
      cdf_from_histogram = pruning_utils.compute_cdf_from_histogram(
          abs_values, [0.0, max_value], nbins=pruning_utils._NBINS)
      cdf = pruning_utils.compute_cdf(abs_values, [0.0, max_value])
      self.assertAllEqual(cdf.eval(), cdf_from_histogram.eval())

  def testHistogram(self):
    width = 10
    height = 10
    nbins = 100
    expected_histogram = np.full(nbins, 1.0)
    init = init_ops.constant_initializer(np.linspace(0.0, 1.0, width * height))
    weights = variable_scope.get_variable(
        "weights", [width, height], initializer=init)
    histogram = pruning_utils._histogram(
        weights, [0, 1.0], nbins, dtype=np.float32)
    with self.cached_session():
      variables.global_variables_initializer().run()
      computed_histogram = histogram.eval()
    self.assertAllEqual(expected_histogram, computed_histogram)

  def testCDF(self):
    nbins = 5
    weights = constant_op.constant([-1, 0, 1, 1.5, 2, 3, 4, 5, 10, 100])
    abs_weights = math_ops.abs(weights)
    norm_cdf = pruning_utils.compute_cdf_from_histogram(
        abs_weights, [0.0, 5.0], nbins=nbins)
    expected_cdf = np.array([0.1, 0.4, 0.5, 0.6, 1.0], dtype=np.float32)
    with self.cached_session() as sess:
      variables.global_variables_initializer().run()
      norm_cdf_val = sess.run(norm_cdf)
      self.assertAllEqual(len(norm_cdf_val), nbins)
      self.assertAllEqual(expected_cdf, norm_cdf_val)

  def testCDFEquivalence2D(self):
    width = 100
    height = 100
    weights = variable_scope.get_variable("weights", shape=[width, height])
    self._compare_cdf(weights)

  def testCDFEquivalence4D(self):
    weights = variable_scope.get_variable("weights", shape=[5, 5, 128, 128])
    self._compare_cdf(weights)


@parameterized.named_parameters(
    ("Input_32x32_block_1x1", [32, 32], [1, 1]),
    # block size 6x6
    ("Input_3x3_block_6x6", [3, 3], [6, 6]),
    ("Input_32x32_block_6x6", [32, 32], [6, 6]),
    ("Input_2x32_block_6x6", [2, 32], [6, 6]),
    ("Input_32x2_block_6x6", [32, 2], [6, 6]),
    ("Input_30x30_block_6x6", [30, 30], [6, 6]),
    # block size 4x4
    ("Input_32x32_block_4x4", [32, 32], [4, 4]),
    ("Input_2x32_block_4x4", [2, 32], [4, 4]),
    ("Input_32x2_block_4x4", [32, 2], [4, 4]),
    ("Input_30x30_block_4x4", [30, 30], [4, 4]),
    # block size 1x4
    ("Input_32x32_block_1x4", [32, 32], [1, 4]),
    ("Input_2x32_block_1x4", [2, 32], [1, 4]),
    ("Input_32x2_block_1x4", [32, 2], [1, 4]),
    ("Input_30x30_block_1x4", [30, 30], [1, 4]),
    # block size 4x1
    ("Input_32x32_block_4x1", [32, 32], [4, 1]),
    ("Input_2x32_block_4x1", [2, 32], [4, 1]),
    ("Input_32x2_block_4x1", [32, 2], [4, 1]),
    ("Input_30x30_block_4x1", [30, 30], [4, 1]))
class PruningUtilsParameterizedTest(test.TestCase, parameterized.TestCase):

  def _compare_pooling_methods(self, weights, pooling_kwargs):
    with self.cached_session():
      variables.global_variables_initializer().run()
      pooled_weights_tf = array_ops.squeeze(
          nn_ops.pool(
              array_ops.reshape(
                  weights,
                  [1, weights.get_shape()[0],
                   weights.get_shape()[1], 1]), **pooling_kwargs),
          axis=[0, 3])
      pooled_weights_factorized_pool = pruning_utils.factorized_pool(
          weights, **pooling_kwargs)

      self.assertAllClose(pooled_weights_tf.eval(),
                          pooled_weights_factorized_pool.eval())

  def _compare_expand_tensor_with_kronecker_product(self, tensor, block_dim):
    with self.cached_session() as session:
      variables.global_variables_initializer().run()
      expanded_tensor = pruning_utils.expand_tensor(tensor, block_dim)
      kronecker_product = pruning_utils.kronecker_product(
          tensor, array_ops.ones(block_dim))
      expanded_tensor_val, kronecker_product_val = session.run(
          [expanded_tensor, kronecker_product])
      self.assertAllEqual(expanded_tensor_val, kronecker_product_val)

  def testFactorizedAvgPool(self, input_shape, window_shape):
    weights = variable_scope.get_variable("weights", shape=input_shape)
    pooling_kwargs = {
        "window_shape": window_shape,
        "pooling_type": "AVG",
        "strides": window_shape,
        "padding": "SAME"
    }
    self._compare_pooling_methods(weights, pooling_kwargs)

  def testFactorizedMaxPool(self, input_shape, window_shape):
    weights = variable_scope.get_variable("weights", shape=input_shape)
    pooling_kwargs = {
        "window_shape": window_shape,
        "pooling_type": "MAX",
        "strides": window_shape,
        "padding": "SAME"
    }
    self._compare_pooling_methods(weights, pooling_kwargs)

  def testExpandTensor(self, input_shape, block_dim):
    weights = random_ops.random_normal(shape=input_shape)
    self._compare_expand_tensor_with_kronecker_product(weights, block_dim)


if __name__ == "__main__":
  test.main()
