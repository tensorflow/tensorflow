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
"""Functional tests for cumulative_logsumexp op."""

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class CumulativeLogsumexpTest(test.TestCase):
  valid_dtypes = [
      dtypes.float32,
      dtypes.float64,
      dtypes.float16,
      dtypes.bfloat16,
  ]

  def _computeLogSumExp(self, x, **kwargs):
    result_naive = math_ops.cumsum(math_ops.exp(x), **kwargs)
    result_fused = math_ops.exp(math_ops.cumulative_logsumexp(x, **kwargs))
    return result_naive, result_fused

  def _testLogSumExp(self, x, dtype=dtypes.float32, use_gpu=False, **kwargs):
    with self.cached_session(use_gpu=use_gpu):
      x = ops.convert_to_tensor(x, dtype=dtype)

      result_naive, result_fused = self.evaluate(
          self._computeLogSumExp(x, **kwargs))

    tol = 2e-2 if dtype in [dtypes.float16, dtypes.bfloat16] else 1e-6
    self.assertAllClose(result_naive, result_fused, rtol=tol, atol=tol)

  def _testLogSumExpAllArgs(self, x, axis=0, use_gpu=False):
    for dtype in self.valid_dtypes:
      for reverse in (True, False):
        for exclusive in (True, False):
          self._testLogSumExp(
              x, dtype=dtype, use_gpu=use_gpu,
              reverse=reverse, exclusive=exclusive,
              axis=axis)

  def testMinusInfinity(self):
    x = np.log([0., 0., 1., 1., 1., 1., 0., 0.])
    self._testLogSumExpAllArgs(x, use_gpu=False)
    self._testLogSumExpAllArgs(x, use_gpu=True)

  def test1D(self):
    x = np.arange(10) / 10.0 - 0.5
    self._testLogSumExpAllArgs(x, use_gpu=False)
    self._testLogSumExpAllArgs(x, use_gpu=True)

  def test2D(self):
    x = np.reshape(np.arange(20) / 20.0 - 0.5, (2, 10))

    for axis in (-2, -1, 0, 1):
      self._testLogSumExpAllArgs(x, axis=axis, use_gpu=False)
      self._testLogSumExpAllArgs(x, axis=axis, use_gpu=True)

  def _testGradient(self, x, use_gpu=False, **kwargs):
    with self.cached_session(use_gpu=use_gpu):
      x = ops.convert_to_tensor(x, dtype=dtypes.float64)

      grad_naive_theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda y: math_ops.cumsum(math_ops.exp(y), **kwargs), [x])
      grad_fused_theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda y: math_ops.exp(math_ops.cumulative_logsumexp(y, **kwargs)),
          [x])

      self.assertAllClose(grad_fused_theoretical, grad_naive_theoretical)

  def testGradient(self):
    for reverse in (True, False):
      for exclusive in (True, False):
        x = np.arange(10) / 10.0 - 0.5

        self._testGradient(x, use_gpu=False,
                           reverse=reverse, exclusive=exclusive)
        self._testGradient(x, use_gpu=True,
                           reverse=reverse, exclusive=exclusive)

  def _logSumExpMap(self, x):
    return map_fn.map_fn(
        lambda i: math_ops.reduce_logsumexp(x[:i + 1]),
        math_ops.range(array_ops.shape(x)[0]),
        dtype=x.dtype)

  def test1DLarge(self):
    # This test ensures that the operation is correct even when the naive
    # implementation would overflow.
    x_np = np.arange(20) * 20.0

    for use_gpu in (True, False):
      with self.cached_session(use_gpu=use_gpu):
        x_tf = ops.convert_to_tensor(x_np, dtype=dtypes.float32)

        result_fused = self.evaluate(math_ops.cumulative_logsumexp(x_tf))
        result_map = self.evaluate(self._logSumExpMap(x_tf))

      self.assertAllClose(result_fused, result_map)

  def testPlusInfinity(self):
    x = [np.inf, np.inf, 1.0, np.inf]
    for dtype in self.valid_dtypes:
      for use_gpu in (True, False):
        with self.cached_session(use_gpu=use_gpu):
          x_tf = ops.convert_to_tensor(x, dtype=dtype)
          result = self.evaluate(math_ops.cumulative_logsumexp(x_tf))
          expected = np.array(
              [np.inf, np.inf, np.inf, np.inf], dtype=x_tf.dtype.as_numpy_dtype
          )
          self.assertAllClose(result, expected)

  def testPlusInfinityAllInf(self):
    # All-+inf input accumulated along a 2-D axis: every pairwise step is
    # inf + inf, which yielded inf - inf = NaN before the +inf guard in the
    # LogSumExp reducer.
    x = np.array([[np.inf], [np.inf], [np.inf]])
    for dtype in self.valid_dtypes:
      for use_gpu in (True, False):
        with self.cached_session(use_gpu=use_gpu):
          x_tf = ops.convert_to_tensor(x, dtype=dtype)
          result = self.evaluate(math_ops.cumulative_logsumexp(x_tf, axis=0))
          self.assertAllEqual(
              np.full((3, 1), np.inf),
              result,
              msg=f'Expected +inf outputs for all-inf input, got {result}',
          )

  def testNaNPropagation(self):
    # Regression test for GitHub issue 111383. The LogSumExp reducer took the
    # min and max of the accumulator and the next element using Eigen's
    # default NaN mode, which reduces to `(b < a) ? b : a`. Every comparison
    # against NaN is false, so a NaN operand was discarded in favor of the
    # other one. An all-NaN input therefore returned -inf, and worse, a NaN
    # following a finite element was replaced by the running maximum, so
    # [1, NaN] returned [1, 1 + log(2)] instead of propagating NaN.
    for dtype in self.valid_dtypes:
      for use_gpu in (True, False):
        with self.cached_session(use_gpu=use_gpu):
          x_tf = ops.convert_to_tensor([np.nan, np.nan], dtype=dtype)
          result = self.evaluate(math_ops.cumulative_logsumexp(x_tf))
          self.assertAllEqual(
              [True, True],
              np.isnan(result),
              msg=f'Expected all NaN for all-NaN input, got {result}',
          )

          # A NaN must also poison every later element of the scan, not only
          # the position it appears at.
          x_tf = ops.convert_to_tensor([1.0, np.nan, 2.0], dtype=dtype)
          result = self.evaluate(math_ops.cumulative_logsumexp(x_tf))
          self.assertAllEqual(
              [False, True, True],
              np.isnan(result),
              msg=f'Expected NaN from index 1 onward, got {result}',
          )
          # The prefix before the NaN is unaffected.
          self.assertAllClose(1.0, result[0])

  def testNaNPropagationReverse(self):
    # The reverse scan walks the axis in the other direction, so the NaN must
    # poison the elements at lower indices instead.
    for dtype in self.valid_dtypes:
      for use_gpu in (True, False):
        with self.cached_session(use_gpu=use_gpu):
          x_tf = ops.convert_to_tensor([1.0, np.nan, 2.0], dtype=dtype)
          result = self.evaluate(
              math_ops.cumulative_logsumexp(x_tf, reverse=True)
          )
          self.assertAllEqual(
              [True, True, False],
              np.isnan(result),
              msg=f'Expected NaN up to index 1, got {result}',
          )
          self.assertAllClose(2.0, result[2])


if __name__ == '__main__':
  test.main()
