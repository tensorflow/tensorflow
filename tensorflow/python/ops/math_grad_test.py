# Copyright 2016 Google Inc. All Rights Reserved.
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
import tensorflow as tf


class SquaredDifferenceOpTest(tf.test.TestCase):

  def _testGrad(self, left_shape, right_shape):

    if len(left_shape) > len(right_shape):
      output_shape = left_shape
    else:
      output_shape = right_shape
    l = np.random.randn(*left_shape)
    r = np.random.randn(*right_shape)

    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        left_tensor = tf.constant(l, shape=left_shape)
        right_tensor = tf.constant(r, shape=right_shape)
        output = tf.squared_difference(left_tensor, right_tensor)
        left_err = tf.test.compute_gradient_error(left_tensor,
                                                  left_shape,
                                                  output,
                                                  output_shape,
                                                  x_init_value=l)
        right_err = tf.test.compute_gradient_error(right_tensor,
                                                   right_shape,
                                                   output,
                                                   output_shape,
                                                   x_init_value=r)
      self.assertLess(left_err, 1e-10)
      self.assertLess(right_err, 1e-10)

  def testGrad(self):
    self._testGrad([1, 2, 3, 2], [3, 2])
    self._testGrad([2, 4], [3, 2, 4])


class AbsOpTest(tf.test.TestCase):

  def _biasedRandN(self, shape, bias=0.1, sigma=1.0):
    """Returns samples from a normal distribution shifted `bias` away from 0."""
    value = np.random.randn(*shape) * sigma
    return value + np.sign(value) * bias

  def _testGrad(self, shape, dtype=None, max_error=None, bias=None, sigma=None):
    np.random.seed(7)
    if dtype in (tf.complex64, tf.complex128):
      value = tf.complex(self._biasedRandN(shape, bias=bias, sigma=sigma),
                         self._biasedRandN(shape, bias=bias, sigma=sigma))
    else:
      value = tf.convert_to_tensor(self._biasedRandN(shape, bias=bias),
                                   dtype=dtype)

    for use_gpu in [True, False]:
      with self.test_session(use_gpu=use_gpu):
        if dtype in (tf.complex64, tf.complex128):
          output = tf.complex_abs(value)
        else:
          output = tf.abs(value)
        error = tf.test.compute_gradient_error(
            value, shape, output, output.get_shape().as_list())
    self.assertLess(error, max_error)

  def testComplexAbs(self):
    # Bias random test values away from zero to avoid numeric instabilities.
    self._testGrad([3, 3], dtype=tf.float32, max_error=2e-5, bias=0.1,
                   sigma=1.0)
    self._testGrad([3, 3], dtype=tf.complex64, max_error=2e-5, bias=0.1,
                   sigma=1.0)

    # Ensure stability near the pole at zero.
    self._testGrad([3, 3], dtype=tf.float32, max_error=100.0, bias=0.0,
                   sigma=0.1)
    self._testGrad([3, 3], dtype=tf.complex64, max_error=100.0, bias=0.0,
                   sigma=0.1)


class MinOrMaxGradientTest(tf.test.TestCase):

  def testMinGradient(self):
    inputs = tf.constant([1.0], dtype=tf.float32)
    outputs = tf.reduce_min(tf.concat(0, [inputs, inputs]))
    with self.test_session():
      error = tf.test.compute_gradient_error(inputs, [1], outputs, [])
      self.assertLess(error, 1e-4)

  def testMaxGradient(self):
    inputs = tf.constant([1.0], dtype=tf.float32)
    outputs = tf.reduce_max(tf.concat(0, [inputs, inputs]))
    with self.test_session():
      error = tf.test.compute_gradient_error(inputs, [1], outputs, [])
      self.assertLess(error, 1e-4)


class SegmentMinOrMaxGradientTest(tf.test.TestCase):

  def testSegmentMinGradient(self):
    data = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    segment_ids = tf.constant([0, 0, 1], dtype=tf.int64)
    segment_min = tf.segment_min(data, segment_ids)
    with self.test_session():
      error = tf.test.compute_gradient_error(data, [3], segment_min, [2])
      self.assertLess(error, 1e-4)

  def testSegmentMaxGradient(self):
    data = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    segment_ids = tf.constant([0, 0, 1], dtype=tf.int64)
    segment_max = tf.segment_max(data, segment_ids)
    with self.test_session():
      error = tf.test.compute_gradient_error(data, [3], segment_max, [2])
      self.assertLess(error, 1e-4)

  def testSegmentMinGradientWithTies(self):
    inputs = tf.constant([1.0], dtype=tf.float32)
    data = tf.concat(0, [inputs, inputs])
    segment_ids = tf.constant([0, 0], dtype=tf.int64)
    segment_min = tf.segment_min(data, segment_ids)
    with self.test_session():
      error = tf.test.compute_gradient_error(inputs, [1], segment_min, [1])
      self.assertLess(error, 1e-4)

  def testSegmentMaxGradientWithTies(self):
    inputs = tf.constant([1.0], dtype=tf.float32)
    data = tf.concat(0, [inputs, inputs])
    segment_ids = tf.constant([0, 0], dtype=tf.int64)
    segment_max = tf.segment_max(data, segment_ids)
    with self.test_session():
      error = tf.test.compute_gradient_error(inputs, [1], segment_max, [1])
      self.assertLess(error, 1e-4)


if __name__ == "__main__":
  tf.test.main()
