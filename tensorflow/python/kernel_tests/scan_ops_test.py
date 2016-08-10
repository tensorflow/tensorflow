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

"""Functional tests for scan ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import combinations

import numpy as np
import tensorflow as tf


def numpy_reverse(x, axis):
  ix = [slice(None, None, -1)
        if i == axis else slice(None) for i in range(len(x.shape))]
  return x[ix]

def handle_options(func, x, axis, exclusive, reverse):
  """Adds tf options to numpy scan ops"""
  if reverse:
    x = numpy_reverse(x, axis)

  if exclusive:
    ix_head = [slice(0, 1) if i == axis else slice(None)
                 for i in range(len(x.shape))]
    ix_init = [slice(0, -1) if i == axis else slice(None)
                 for i in range(len(x.shape))]
    if func == np.cumsum:
      init = np.zeros_like(x[ix_head])
    elif func == np.cumprod:
      init = np.ones_like(x[ix_head])
    else:
      raise ValueError("Unknown scan function")
    x = np.concatenate([init, func(x[ix_init], axis)], axis=axis)
  else:
    x = func(x, axis=axis)

  if reverse:
    x = numpy_reverse(x, axis)
  return x

class CumsumTest(tf.test.TestCase):

  valid_dtypes = [np.int32, np.int64, np.float16, np.float32,
                  np.float64, np.complex64, np.complex128]

  def _compare(self, x, axis, exclusive, reverse):
    np_out = handle_options(np.cumsum, x, axis, exclusive, reverse)
    with self.test_session():
      tf_out = tf.cumsum(x, axis, exclusive, reverse).eval()

    self.assertAllClose(np_out, tf_out)

  def _compareAll(self, x, axis):
    for exclusive in [True, False]:
      for reverse in [True, False]:
        self._compare(x, axis, exclusive, reverse)

  def test1D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      self._compareAll(x, 0)

  def test2D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(0, 10).reshape([2, 5]).astype(dtype)
      self._compareAll(x, 0)
      self._compareAll(x, 1)

  def test3D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(0, 20).reshape([2, 2, 5]).astype(dtype)
      self._compareAll(x, 0)
      self._compareAll(x, 1)
      self._compareAll(x, 2)

  def testInvalidAxis(self):
    x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    input_tensor = tf.convert_to_tensor(x)
    with self.test_session():
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          lambda e: "Expected scan axis in the range" in str(e)):
        tf.cumsum(input_tensor, -1).eval()
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          lambda e: "Expected scan axis in the range" in str(e)):
        tf.cumsum(input_tensor, 2).eval()
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          lambda e: "axis must be a scalar" in str(e)):
        tf.cumsum(input_tensor, [0]).eval()

  def _compareGradient(self, shape, axis, exclusive, reverse):
    x = np.arange(0, 50).reshape(shape).astype(np.float64)
    with self.test_session():
      t = tf.convert_to_tensor(x)
      result = tf.cumsum(t, axis, exclusive, reverse)
      jacob_t, jacob_n = tf.test.compute_gradient(t,
                                                  shape,
                                                  result,
                                                  shape,
                                                  x_init_value=x,
                                                  delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  def testGradient(self):
    self._compareGradient([50], 0, False, False)

  def testGradientReverse(self):
    self._compareGradient([50], 0, False, True)

  def testGradientExclusive(self):
    self._compareGradient([50], 0, True, False)

  def testGradientExclusiveReverse(self):
    self._compareGradient([50], 0, True, True)

  def testGradient2D(self):
    for axis in [0, 1]:
      for exclusive in [True, False]:
        for reverse in [True, False]:
          self._compareGradient([5, 10], axis, exclusive, reverse)


class CumprodTest(tf.test.TestCase):

  valid_dtypes = [np.int32, np.int64, np.float16, np.float32,
                  np.float64, np.complex64, np.complex128]

  def _compare(self, x, axis, exclusive, reverse):
    np_out = handle_options(np.cumprod, x, axis, exclusive, reverse)
    with self.test_session():
      tf_out = tf.cumprod(x, axis, exclusive, reverse).eval()

    self.assertAllClose(np_out, tf_out)

  def _compareAll(self, x, axis):
    for exclusive in [True, False]:
      for reverse in [True, False]:
        self._compare(x, axis, exclusive, reverse)


  def test1D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      self._compareAll(x, 0)

  def test2D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 11).reshape([2, 5]).astype(dtype)
      self._compareAll(x, 0)
      self._compareAll(x, 1)

  def test3D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 21).reshape([2, 2, 5]).astype(dtype)
      self._compareAll(x, 0)
      self._compareAll(x, 1)
      self._compareAll(x, 2)

  def testInvalidAxis(self):
    x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    input_tensor = tf.convert_to_tensor(x)
    with self.test_session():
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          lambda e: "Expected scan axis in the range" in str(e)):
        tf.cumprod(input_tensor, -1).eval()
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          lambda e: "Expected scan axis in the range" in str(e)):
        tf.cumprod(input_tensor, 2).eval()
      with self.assertRaisesWithPredicateMatch(
          tf.errors.InvalidArgumentError,
          lambda e: "axis must be a scalar" in str(e)):
        tf.cumprod(input_tensor, [0]).eval()

  def _compareGradient(self, shape, axis, exclusive, reverse):
    x = np.arange(1, 9).reshape(shape).astype(np.float64)
    with self.test_session():
      t = tf.convert_to_tensor(x)
      result = tf.cumprod(t, axis, exclusive, reverse)
      jacob_t, jacob_n = tf.test.compute_gradient(t,
                                                  shape,
                                                  result,
                                                  shape,
                                                  x_init_value=x,
                                                  delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  def testGradient(self):
    self._compareGradient([8], 0, False, False)

  def testGradientReverse(self):
    self._compareGradient([8], 0, False, True)

  def testGradientExclusive(self):
    self._compareGradient([8], 0, True, False)

  def testGradientExclusiveReverse(self):
    self._compareGradient([8], 0, True, True)

  def testGradient2D(self):
    for axis in [0, 1]:
      for exclusive in [True, False]:
        for reverse in [True, False]:
          self._compareGradient([2, 4], axis, exclusive, reverse)


if __name__ == "__main__":
  tf.test.main()
