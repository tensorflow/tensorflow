# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests Sobol sequence generator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class SobolSampleOpTest(test_util.TensorFlowTestCase):

  def test_basic(self):
    for dtype in [np.float64, np.float32]:
      expected = np.array([[.5, .5], [.75, .25], [.25, .75], [.375, .375]])
      sample = self.evaluate(math_ops.sobol_sample(2, 4, dtype=dtype))
      self.assertAllClose(expected, sample, 0.001)

  def test_more_known_values(self):
    for dtype in [np.float64, np.float32]:
      sample = math_ops.sobol_sample(5, 31, dtype=dtype)
      expected = [[0.50, 0.50, 0.50, 0.50, 0.50],
                  [0.75, 0.25, 0.25, 0.25, 0.75],
                  [0.25, 0.75, 0.75, 0.75, 0.25],
                  [0.375, 0.375, 0.625, 0.875, 0.375],
                  [0.875, 0.875, 0.125, 0.375, 0.875],
                  [0.625, 0.125, 0.875, 0.625, 0.625],
                  [0.125, 0.625, 0.375, 0.125, 0.125],
                  [0.1875, 0.3125, 0.9375, 0.4375, 0.5625],
                  [0.6875, 0.8125, 0.4375, 0.9375, 0.0625],
                  [0.9375, 0.0625, 0.6875, 0.1875, 0.3125],
                  [0.4375, 0.5625, 0.1875, 0.6875, 0.8125],
                  [0.3125, 0.1875, 0.3125, 0.5625, 0.9375],
                  [0.8125, 0.6875, 0.8125, 0.0625, 0.4375],
                  [0.5625, 0.4375, 0.0625, 0.8125, 0.1875],
                  [0.0625, 0.9375, 0.5625, 0.3125, 0.6875],
                  [0.09375, 0.46875, 0.46875, 0.65625, 0.28125],
                  [0.59375, 0.96875, 0.96875, 0.15625, 0.78125],
                  [0.84375, 0.21875, 0.21875, 0.90625, 0.53125],
                  [0.34375, 0.71875, 0.71875, 0.40625, 0.03125],
                  [0.46875, 0.09375, 0.84375, 0.28125, 0.15625],
                  [0.96875, 0.59375, 0.34375, 0.78125, 0.65625],
                  [0.71875, 0.34375, 0.59375, 0.03125, 0.90625],
                  [0.21875, 0.84375, 0.09375, 0.53125, 0.40625],
                  [0.15625, 0.15625, 0.53125, 0.84375, 0.84375],
                  [0.65625, 0.65625, 0.03125, 0.34375, 0.34375],
                  [0.90625, 0.40625, 0.78125, 0.59375, 0.09375],
                  [0.40625, 0.90625, 0.28125, 0.09375, 0.59375],
                  [0.28125, 0.28125, 0.15625, 0.21875, 0.71875],
                  [0.78125, 0.78125, 0.65625, 0.71875, 0.21875],
                  [0.53125, 0.03125, 0.40625, 0.46875, 0.46875],
                  [0.03125, 0.53125, 0.90625, 0.96875, 0.96875]]
      self.assertAllClose(expected, self.evaluate(sample), .001)

  def test_skip(self):
    dim = 10
    n = 50
    skip = 17
    for dtype in [np.float64, np.float32]:
      sample_noskip = math_ops.sobol_sample(dim, n + skip, dtype=dtype)
      sample_skip = math_ops.sobol_sample(dim, n, skip, dtype=dtype)

      self.assertAllClose(
          self.evaluate(sample_noskip)[skip:, :], self.evaluate(sample_skip))

  def test_static_shape(self):
    s = math_ops.sobol_sample(10, 100, dtype=np.float32)
    self.assertAllEqual([100, 10], s.shape.as_list())

  def test_static_shape_using_placeholder_for_dim(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32)])
    def f(dim):
      s = math_ops.sobol_sample(dim, 100, dtype=dtypes.float32)
      assert s.shape.as_list() == [100, None]
      return s

    self.assertAllEqual([100, 10], self.evaluate(f(10)).shape)

  def test_static_shape_using_placeholder_for_num_results(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32)])
    def f(num_results):
      s = math_ops.sobol_sample(10, num_results, dtype=dtypes.float32)
      assert s.shape.as_list() == [None, 10]
      return s

    self.assertAllEqual([100, 10], self.evaluate(f(100)).shape)

  def test_static_shape_using_only_placeholders(self):

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32)] *
        2)
    def f(dim, num_results):
      s = math_ops.sobol_sample(dim, num_results, dtype=dtypes.float32)
      assert s.shape.as_list() == [None, None]
      return s

    self.assertAllEqual([100, 10], self.evaluate(f(10, 100)).shape)

  def test_dynamic_shape(self):
    s = math_ops.sobol_sample(10, 100, dtype=dtypes.float32)
    self.assertAllEqual([100, 10], self.evaluate(s).shape)

  def test_default_dtype(self):
    # Create an op without specifying the dtype. Dtype should be float32 in
    # this case.
    s = math_ops.sobol_sample(10, 100)
    self.assertEqual(dtypes.float32, s.dtype)

if __name__ == '__main__':
  googletest.main()
