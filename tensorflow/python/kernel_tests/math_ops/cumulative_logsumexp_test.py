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

from tensorflow.python.eager import backprop
from tensorflow.python.eager import forwardprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
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

  def testSecondDerivativeNestedForwardAccumulator(self):
    # Regression test for GitHub issue #127243:
    # cumulative_logsumexp produces NaN second derivative under nested
    # forward-mode autodiff.
    def target(t):
      x = array_ops.reshape(
          array_ops_stack.stack([t, -t + 1, t / 2 - 2, -2 * t, t + 3, t / 4]),
          [2, 3],
      )
      y = math_ops.cumulative_logsumexp(
          x, axis=-1, exclusive=False, reverse=True
      )
      return (
          y[0, 0] + 2 * y[0, 1] - y[0, 2] + 3 * y[1, 0] + y[1, 1] - 2 * y[1, 2]
      )

    t = constant_op.constant(-40.0, dtype=dtypes.float64)
    dt = constant_op.constant(1.0, dtype=dtypes.float64)
    expected = 1.0572349592992632e-12

    # Verify forward and 1st derivative
    self.assertAllClose(self.evaluate(target(t)), 395.0000000000019)

    # 1. Reverse-over-Reverse
    with backprop.GradientTape() as t2:
      t2.watch(t)
      with backprop.GradientTape() as t1:
        t1.watch(t)
        y = target(t)
      g1 = t1.gradient(y, t)
    rr = t2.gradient(g1, t)
    self.assertAllClose(self.evaluate(g1), -9.75, atol=1e-10)
    self.assertAllClose(self.evaluate(rr), expected, atol=1e-12)

    # 2. Forward-over-Reverse
    with forwardprop.ForwardAccumulator(t, dt) as acc:
      with backprop.GradientTape() as t1:
        t1.watch(t)
        y = target(t)
      g1 = t1.gradient(y, t)
    fr = acc.jvp(g1)
    self.assertAllClose(self.evaluate(fr), expected, atol=1e-12)

    # 3. Reverse-over-Forward
    with backprop.GradientTape() as t1:
      t1.watch(t)
      with forwardprop.ForwardAccumulator(t, dt) as acc:
        y = target(t)
      j1 = acc.jvp(y)
    rf = t1.gradient(j1, t)
    self.assertAllClose(self.evaluate(rf), expected, atol=1e-12)

    # 4. Forward-over-Forward (nested ForwardAccumulator)
    with forwardprop.ForwardAccumulator(t, dt) as acc2:
      with forwardprop.ForwardAccumulator(t, dt) as acc1:
        y = target(t)
      j1 = acc1.jvp(y)
    ff = acc2.jvp(j1)
    self.assertAllClose(self.evaluate(j1), -9.75, atol=1e-10)
    self.assertAllClose(self.evaluate(ff), expected, atol=1e-12)

    # Verify positive, negative, and zero cotangents across supported dtypes
    tol_map = {
        dtypes.float64: 1e-12,
        dtypes.float32: 1e-5,
        dtypes.float16: 1e-2,
        dtypes.bfloat16: 2e-1,
    }
    for reverse in (True, False):
      for exclusive in (True, False):
        x64 = constant_op.constant([1.0, -2.0, 0.5, 3.0], dtype=dtypes.float64)
        dx64 = constant_op.constant([1.0, -1.0, 0.0, 0.5], dtype=dtypes.float64)
        w64 = constant_op.constant([2.0, -3.0, 0.0, 1.0], dtype=dtypes.float64)
        with backprop.GradientTape() as t2:
          t2.watch(x64)
          with backprop.GradientTape() as t1:
            t1.watch(x64)
            lse64 = math_ops.cumulative_logsumexp(
                x64, reverse=reverse, exclusive=exclusive
            )
            out64 = math_ops.reduce_sum(
                (math_ops.exp(lse64) if exclusive else lse64) * w64
            )
          g1 = t1.gradient(out64, x64)
          dir_g1 = math_ops.reduce_sum(g1 * dx64)
        g2 = t2.gradient(dir_g1, x64)
        ref_jvp1 = self.evaluate(dir_g1)
        ref_jvp2 = self.evaluate(math_ops.reduce_sum(g2 * dx64))

        for dtype in self.valid_dtypes:
          x_val = constant_op.constant([1.0, -2.0, 0.5, 3.0], dtype=dtype)
          dx_val = constant_op.constant([1.0, -1.0, 0.0, 0.5], dtype=dtype)
          weights = constant_op.constant([2.0, -3.0, 0.0, 1.0], dtype=dtype)
          with forwardprop.ForwardAccumulator(x_val, dx_val) as acc2:
            with forwardprop.ForwardAccumulator(x_val, dx_val) as acc1:
              lse = math_ops.cumulative_logsumexp(
                  x_val, reverse=reverse, exclusive=exclusive
              )
              out = math_ops.reduce_sum(
                  (math_ops.exp(lse) if exclusive else lse) * weights
              )
            jvp1 = acc1.jvp(out)
          jvp2 = acc2.jvp(jvp1)
          tol = tol_map[dtype]
          self.assertAllClose(self.evaluate(jvp1), ref_jvp1, rtol=tol, atol=tol)
          self.assertAllClose(self.evaluate(jvp2), ref_jvp2, rtol=tol, atol=tol)


if __name__ == '__main__':
  test.main()
