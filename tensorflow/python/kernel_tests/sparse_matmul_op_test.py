# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.tf.matmul."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def RandMatrix(rows, cols, tr, round_bfloat=False):
  if tr:
    rows, cols = cols, rows
  rand_func = np.random.randint if round_bfloat else np.random.uniform
  return (np.clip(
      rand_func(
          low=-256.0, high=256.0, size=rows * cols), -64,
      64) / 128.0).reshape([rows, cols]).astype(np.float32)


class SparseMatMulTest(test.TestCase):

  def _testCpuMatmul(self,
                     x,
                     y,
                     tr_a=False,
                     tr_b=False,
                     sp_a=True,
                     sp_b=False,
                     x_dtype=dtypes.float32,
                     y_dtype=dtypes.float32):
    with self.test_session(use_gpu=False):
      tf_x = math_ops.cast(x, x_dtype)
      tf_y = math_ops.cast(y, y_dtype)
      tf_ans = math_ops.matmul(
          tf_x,
          tf_y,
          transpose_a=tr_a,
          transpose_b=tr_b,
          a_is_sparse=sp_a,
          b_is_sparse=sp_b)
      out = tf_ans.eval()
      np_x = math_ops.cast(tf_x, dtypes.float32).eval()
      np_y = math_ops.cast(tf_y, dtypes.float32).eval()

    if tr_a:
      np_x = np.transpose(np_x)
    if tr_b:
      np_y = np.transpose(np_y)

    np_ans = np.matrix(np_x) * np.matrix(np_y)
    self.assertShapeEqual(np_ans, tf_ans)
    self.assertAllCloseAccordingToType(np_ans, out, rtol=1e-4, atol=1e-4)

  def testBasic(self):
    x = np.arange(0., 4.).reshape([4, 1]).astype(np.float32)
    y = np.arange(-1., 1.).reshape([1, 2]).astype(np.float32)
    for x_dtype in (dtypes.float32, dtypes.bfloat16):
      for y_dtype in (dtypes.float32, dtypes.bfloat16):
        self._testCpuMatmul(x, y, x_dtype=x_dtype, y_dtype=y_dtype)

  def testZeroDim(self):
    x = np.ones((4, 0)).astype(np.float32)
    y = np.ones((0, 3)).astype(np.float32)
    for x_dtype in (dtypes.float32, dtypes.bfloat16):
      for y_dtype in (dtypes.float32, dtypes.bfloat16):
        self._testCpuMatmul(x, y, x_dtype=x_dtype, y_dtype=y_dtype)

  def testEmpty(self):
    x = np.ones((0, 0)).astype(np.float32)
    y = np.ones((0, 0)).astype(np.float32)
    for x_dtype in (dtypes.float32, dtypes.bfloat16):
      for y_dtype in (dtypes.float32, dtypes.bfloat16):
        self._testCpuMatmul(x, y, x_dtype=x_dtype, y_dtype=y_dtype)

  # Tests setting one dimension to be a high value.
  def testLarge(self):
    r1 = np.random.randint(6000, 20000)
    r2 = np.random.randint(1, 10)
    r3 = np.random.randint(1, 10)
    for m, k, n in [(r1, r2, r3), (r2, r1, r3), (r2, r3, r1)]:
      for x_dtype in (dtypes.float32, dtypes.bfloat16):
        for y_dtype in (dtypes.float32, dtypes.bfloat16):
          x = RandMatrix(m, k, False)
          y = RandMatrix(k, n, False)
          self._testCpuMatmul(x, y, x_dtype=x_dtype, y_dtype=y_dtype)

  # Tests random sized matrices.
  def testRandom(self):
    for tr_a in [True, False]:
      for tr_b in [True, False]:
        for sp_a in [True, False]:
          for sp_b in [True, False]:
            for x_dtype in (dtypes.float32, dtypes.bfloat16):
              for y_dtype in (dtypes.float32, dtypes.bfloat16):
                n, k, m = np.random.randint(1, 100, size=3)
                x = RandMatrix(n, k, tr_a)
                y = RandMatrix(k, m, tr_b)
                self._testCpuMatmul(
                    x,
                    y,
                    tr_a,
                    tr_b,
                    sp_a,
                    sp_b,
                    x_dtype=x_dtype,
                    y_dtype=y_dtype)


class MatMulGradientTest(test.TestCase):

  def _testGradients(self, tr_a, tr_b, sp_a, sp_b, a_dtype, b_dtype, delta,
                     name):
    with self.test_session():
      a = constant_op.constant(
          RandMatrix(
              3, 2, tr_a, round_bfloat=True), dtype=dtypes.float32)
      b = constant_op.constant(
          RandMatrix(
              2, 4, tr_b, round_bfloat=True), dtype=dtypes.float32)
      tf_a = math_ops.cast(a, a_dtype) if a_dtype != dtypes.float32 else a
      tf_b = math_ops.cast(b, b_dtype) if b_dtype != dtypes.float32 else b

      m = math_ops.matmul(
          tf_a,
          tf_b,
          name=name,
          transpose_a=tr_a,
          transpose_b=tr_b,
          a_is_sparse=sp_a,
          b_is_sparse=sp_b)
      err = (gradient_checker.compute_gradient_error(
          a, [2, 3] if tr_a else [3, 2],
          m, [3, 4],
          x_init_value=a.eval(),
          delta=delta) + gradient_checker.compute_gradient_error(
              b, [4, 2] if tr_b else [2, 4],
              m, [3, 4],
              x_init_value=b.eval(),
              delta=delta))
    self.assertLessEqual(err, delta / 2.)

  def testGradientInput(self):
    for tr_a in [True, False]:
      for tr_b in [True, False]:
        for sp_a in [True, False]:
          for sp_b in [True, False]:
            for a_dtype in (dtypes.float32, dtypes.bfloat16):
              for b_dtype in (dtypes.float32, dtypes.bfloat16):
                # Note: bfloat16 only has 7 mantissa bits, versus float32 with
                # 10. Hence, we shift by 2 bits to pass the test.
                if a_dtype == dtypes.bfloat16 and b_dtype == dtypes.bfloat16:
                  delta = 1 / 16.
                else:
                  delta = 1 / 64.
                name = "sparse_matmul_%s_%s_%s_%s" % (tr_a, tr_b, sp_a, sp_b)
                self._testGradients(tr_a, tr_b, sp_a, sp_b, a_dtype, b_dtype,
                                    delta, name)


if __name__ == "__main__":
  test.main()
