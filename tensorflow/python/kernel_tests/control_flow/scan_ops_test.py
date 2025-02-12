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

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def numpy_reverse(x, axis):
  length = len(x.shape)
  if axis < 0:
    axis = length + axis

  ix = [
      slice(None, None, -1) if i == axis else slice(None) for i in range(length)
  ]
  return x[tuple(ix)]


def handle_options(func, x, axis, exclusive, reverse):
  """Adds tf options to numpy scan ops."""
  length = len(x.shape)
  if axis < 0:
    axis = length + axis

  if reverse:
    x = numpy_reverse(x, axis)

  if exclusive:
    ix_head = [slice(0, 1) if i == axis else slice(None) for i in range(length)]
    ix_init = [
        slice(0, -1) if i == axis else slice(None) for i in range(length)
    ]
    if func == np.cumsum:
      init = np.zeros_like(x[tuple(ix_head)])
    elif func == np.cumprod:
      init = np.ones_like(x[tuple(ix_head)])
    else:
      raise ValueError("Unknown scan function.")
    x = np.concatenate([init, func(x[tuple(ix_init)], axis)], axis=axis)
  else:
    x = func(x, axis=axis)

  if reverse:
    x = numpy_reverse(x, axis)
  return x


class CumsumTest(test.TestCase):

  valid_dtypes = [
      np.int32,
      np.int64,
      np.float16,
      np.float32,
      np.float64,
      np.complex64,
      np.complex128,
      dtypes.bfloat16.as_numpy_dtype,
  ]

  def _compare(self, x, axis, exclusive, reverse):
    np_out = handle_options(np.cumsum, x, axis, exclusive, reverse)
    with self.cached_session():
      tf_out = math_ops.cumsum(x, axis, exclusive, reverse).eval()

    self.assertAllClose(np_out, tf_out)

  def _compareAll(self, x, axis):
    for exclusive in [True, False]:
      for reverse in [True, False]:
        self._compare(x, axis, exclusive, reverse)

  @test_util.run_deprecated_v1
  def testEmpty(self):
    for dtype in self.valid_dtypes:
      x = np.zeros([0]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def testAxisType(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis_dtype in [dtypes.int64, dtypes.int32]:
        with self.cached_session():
          axis = constant_op.constant(0, axis_dtype)
          tf_out = math_ops.cumsum(x, axis).eval()

  @test_util.run_deprecated_v1
  def testNaN(self):
    for dtype in (
        np.float16,
        np.float32,
        np.float64,
        dtypes.bfloat16.as_numpy_dtype,
    ):
      for nan_idx in range(0, 5):
        x = np.arange(1, 6).reshape([5]).astype(dtype)
        x[nan_idx] = np.nan
        for axis in (-1, 0):
          self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test1D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test2D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(0, 10).reshape([2, 5]).astype(dtype)
      for axis in (-2, -1, 0, 1):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test3D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(0, 20).reshape([2, 2, 5]).astype(dtype)
      for axis in (-3, -2, -1, 0, 1, 2):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test6D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 145).reshape([2, 2, 3, 3, 2, 2]).astype(dtype)
      for axis in range(-6, 6, 3):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/123860949")  # The computation is constant folded
  def testLarge(self):
    for dtype in self.valid_dtypes:
      if np.__version__ >= np.lib.NumpyVersion("2.0.0") and dtype == np.float16:
        continue
      if dtype == dtypes.bfloat16.as_numpy_dtype:
        # https://github.com/numpy/numpy/issues/27709, which might be fixed
        # in some numpy version after 2.1.3.
        continue

      x = np.ones([1000000], dtype=dtype) / 1024
      self._compareAll(x, 0)

  def testInvalidAxis(self):
    x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    input_tensor = ops.convert_to_tensor(x)
    with self.session():
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Expected scan axis in the range [-2, 2)" in str(e)):
        math_ops.cumsum(input_tensor, -3).eval()
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Expected scan axis in the range [-2, 2)" in str(e)):
        math_ops.cumsum(input_tensor, 2).eval()
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "axis must be a scalar" in str(e)):
        math_ops.cumsum(input_tensor, [0]).eval()

  def _compareGradient(self, shape, axis, exclusive, reverse):
    x = np.arange(0, 50).reshape(shape).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      result = math_ops.cumsum(t, axis, exclusive, reverse)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, shape, result, shape, x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient(self):
    for axis in (-1, 0):
      self._compareGradient([50], axis, False, False)

  @test_util.run_deprecated_v1
  def testGradientReverse(self):
    for axis in (-1, 0):
      self._compareGradient([50], axis, False, True)

  @test_util.run_deprecated_v1
  def testGradientExclusive(self):
    for axis in (-1, 0):
      self._compareGradient([50], axis, True, False)

  @test_util.run_deprecated_v1
  def testGradientExclusiveReverse(self):
    for axis in (-1, 0):
      self._compareGradient([50], axis, True, True)

  @test_util.run_deprecated_v1
  def testGradient2D(self):
    for axis in (-1, 0, 1):
      for exclusive in [True, False]:
        for reverse in [True, False]:
          self._compareGradient([5, 10], axis, exclusive, reverse)


class CumprodTest(test.TestCase):

  valid_dtypes = [
      np.int32,
      np.int64,
      np.float16,
      np.float32,
      np.float64,
      np.complex64,
      np.complex128,
      dtypes.bfloat16.as_numpy_dtype,
  ]

  def _compare(self, x, axis, exclusive, reverse):
    np_out = handle_options(np.cumprod, x, axis, exclusive, reverse)
    with self.cached_session():
      tf_out = math_ops.cumprod(x, axis, exclusive, reverse).eval()

    atol = rtol = 1e-6
    if x.dtype == dtypes.bfloat16.as_numpy_dtype:
      atol = rtol = 1e-2
    self.assertAllClose(np_out, tf_out, atol=atol, rtol=rtol)

  def _compareAll(self, x, axis):
    for exclusive in [True, False]:
      for reverse in [True, False]:
        self._compare(x, axis, exclusive, reverse)

  @test_util.run_deprecated_v1
  def testEmpty(self):
    for dtype in self.valid_dtypes:
      x = np.zeros([0]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def testAxisType(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis_dtype in [dtypes.int64, dtypes.int32]:
        with self.cached_session():
          axis = constant_op.constant(0, axis_dtype)
          tf_out = math_ops.cumprod(x, axis).eval()

  @test_util.run_deprecated_v1
  def testNaN(self):
    for dtype in (np.float16, np.float32, np.float64):
      for nan_idx in range(0, 5):
        x = np.arange(1, 6).reshape([5]).astype(dtype)
        x[nan_idx] = np.nan
        for axis in (-1, 0):
          self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test1D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 6).reshape([5]).astype(dtype)
      for axis in (-1, 0):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test2D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 11).reshape([2, 5]).astype(dtype)
      for axis in (-2, -1, 0, 1):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test3D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 21).reshape([2, 2, 5]).astype(dtype)
      for axis in (-3, -2, -1, 0, 1, 2):
        self._compareAll(x, axis)

  @test_util.run_deprecated_v1
  def test6D(self):
    for dtype in self.valid_dtypes:
      x = np.arange(1, 145).reshape([2, 2, 3, 3, 2, 2]).astype(dtype)
      for axis in range(-6, 6, 3):
        self._compareAll(x, axis)

  def testInvalidAxis(self):
    x = np.arange(0, 10).reshape([2, 5]).astype(np.float32)
    input_tensor = ops.convert_to_tensor(x)
    with self.session():
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Expected scan axis in the range [-2, 2)" in str(e)):
        math_ops.cumprod(input_tensor, -3).eval()
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "Expected scan axis in the range [-2, 2)" in str(e)):
        math_ops.cumprod(input_tensor, 2).eval()
      with self.assertRaisesWithPredicateMatch(
          errors_impl.InvalidArgumentError,
          lambda e: "axis must be a scalar" in str(e)):
        math_ops.cumprod(input_tensor, [0]).eval()

  def _compareGradient(self, shape, axis, exclusive, reverse):
    x = np.arange(1, 9).reshape(shape).astype(np.float64)
    with self.cached_session():
      t = ops.convert_to_tensor(x)
      result = math_ops.cumprod(t, axis, exclusive, reverse)
      jacob_t, jacob_n = gradient_checker.compute_gradient(
          t, shape, result, shape, x_init_value=x, delta=1)
    self.assertAllClose(jacob_t, jacob_n, rtol=1e-8, atol=1e-8)

  @test_util.run_deprecated_v1
  def testGradient(self):
    for axis in (-1, 0):
      self._compareGradient([8], axis, False, False)

  @test_util.run_deprecated_v1
  def testGradientReverse(self):
    for axis in (-1, 0):
      self._compareGradient([8], axis, False, True)

  @test_util.run_deprecated_v1
  def testGradientExclusive(self):
    for axis in (-1, 0):
      self._compareGradient([8], axis, True, False)

  @test_util.run_deprecated_v1
  def testGradientExclusiveReverse(self):
    for axis in (-1, 0):
      self._compareGradient([8], axis, True, True)

  @test_util.run_deprecated_v1
  def testGradient2D(self):
    for axis in (-2, -1, 0, 1):
      for exclusive in [True, False]:
        for reverse in [True, False]:
          self._compareGradient([2, 4], axis, exclusive, reverse)


if __name__ == "__main__":
  test.main()
