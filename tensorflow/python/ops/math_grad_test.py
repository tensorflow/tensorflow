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
"""Tests for Python ops defined in math_grad.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class SquaredDifferenceOpTest(test.TestCase):

  def _testGrad(self, left_shape, right_shape):

    if len(left_shape) > len(right_shape):
      output_shape = left_shape
    else:
      output_shape = right_shape
    l = np.random.randn(*left_shape)
    r = np.random.randn(*right_shape)

    with self.test_session(use_gpu=True):
      left_tensor = constant_op.constant(l, shape=left_shape)
      right_tensor = constant_op.constant(r, shape=right_shape)
      output = math_ops.squared_difference(left_tensor, right_tensor)
      left_err = gradient_checker.compute_gradient_error(
          left_tensor, left_shape, output, output_shape, x_init_value=l)
      right_err = gradient_checker.compute_gradient_error(
          right_tensor, right_shape, output, output_shape, x_init_value=r)
    self.assertLess(left_err, 1e-10)
    self.assertLess(right_err, 1e-10)

  def testGrad(self):
    self._testGrad([1, 2, 3, 2], [3, 2])
    self._testGrad([2, 4], [3, 2, 4])


class AbsOpTest(test.TestCase):

  def _biasedRandN(self, shape, bias=0.1, sigma=1.0):
    """Returns samples from a normal distribution shifted `bias` away from 0."""
    value = np.random.randn(*shape) * sigma
    return value + np.sign(value) * bias

  def _testGrad(self, shape, dtype=None, max_error=None, bias=None, sigma=None):
    np.random.seed(7)
    if dtype in (dtypes.complex64, dtypes.complex128):
      value = math_ops.complex(
          self._biasedRandN(
              shape, bias=bias, sigma=sigma),
          self._biasedRandN(
              shape, bias=bias, sigma=sigma))
    else:
      value = ops.convert_to_tensor(
          self._biasedRandN(
              shape, bias=bias), dtype=dtype)

    with self.test_session(use_gpu=True):
      output = math_ops.abs(value)
      error = gradient_checker.compute_gradient_error(
          value, shape, output, output.get_shape().as_list())
    self.assertLess(error, max_error)

  def testComplexAbs(self):
    # Bias random test values away from zero to avoid numeric instabilities.
    self._testGrad(
        [3, 3], dtype=dtypes.float32, max_error=2e-5, bias=0.1, sigma=1.0)
    self._testGrad(
        [3, 3], dtype=dtypes.complex64, max_error=2e-5, bias=0.1, sigma=1.0)

    # Ensure stability near the pole at zero.
    self._testGrad(
        [3, 3], dtype=dtypes.float32, max_error=100.0, bias=0.0, sigma=0.1)
    self._testGrad(
        [3, 3], dtype=dtypes.complex64, max_error=100.0, bias=0.0, sigma=0.1)


class MinOrMaxGradientTest(test.TestCase):

  def testMinGradient(self):
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    outputs = math_ops.reduce_min(array_ops.concat([inputs, inputs], 0))
    with self.test_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], outputs, [])
      self.assertLess(error, 1e-4)

  def testMaxGradient(self):
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    outputs = math_ops.reduce_max(array_ops.concat([inputs, inputs], 0))
    with self.test_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], outputs, [])
      self.assertLess(error, 1e-4)


class ProdGradientTest(test.TestCase):

  def testProdGradient(self):
    inputs = constant_op.constant([[1., 2.], [3., 4.]],
                                  dtype=dtypes.float32)
    outputs = math_ops.reduce_prod(inputs)
    with self.test_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)

  def testProdGradientForNegativeAxis(self):
    inputs = constant_op.constant([[1., 2.], [3., 4.]],
                                  dtype=dtypes.float32)
    outputs = math_ops.reduce_prod(inputs, -1)
    with self.test_session():
      error = gradient_checker.compute_gradient_error(
          inputs, inputs.get_shape().as_list(),
          outputs, outputs.get_shape().as_list())
      self.assertLess(error, 1e-4)


class SegmentMinOrMaxGradientTest(test.TestCase):

  def testSegmentMinGradient(self):
    data = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
    segment_ids = constant_op.constant([0, 0, 1], dtype=dtypes.int64)
    segment_min = math_ops.segment_min(data, segment_ids)
    with self.test_session():
      error = gradient_checker.compute_gradient_error(data, [3], segment_min,
                                                      [2])
      self.assertLess(error, 1e-4)

  def testSegmentMaxGradient(self):
    data = constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32)
    segment_ids = constant_op.constant([0, 0, 1], dtype=dtypes.int64)
    segment_max = math_ops.segment_max(data, segment_ids)
    with self.test_session():
      error = gradient_checker.compute_gradient_error(data, [3], segment_max,
                                                      [2])
      self.assertLess(error, 1e-4)

  def testSegmentMinGradientWithTies(self):
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    data = array_ops.concat([inputs, inputs], 0)
    segment_ids = constant_op.constant([0, 0], dtype=dtypes.int64)
    segment_min = math_ops.segment_min(data, segment_ids)
    with self.test_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], segment_min,
                                                      [1])
      self.assertLess(error, 1e-4)

  def testSegmentMaxGradientWithTies(self):
    inputs = constant_op.constant([1.0], dtype=dtypes.float32)
    data = array_ops.concat([inputs, inputs], 0)
    segment_ids = constant_op.constant([0, 0], dtype=dtypes.int64)
    segment_max = math_ops.segment_max(data, segment_ids)
    with self.test_session():
      error = gradient_checker.compute_gradient_error(inputs, [1], segment_max,
                                                      [1])
      self.assertLess(error, 1e-4)


if __name__ == "__main__":
  test.main()
