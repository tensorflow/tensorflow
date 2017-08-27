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
"""Tests for tensorflow.python.ops.linalg_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


def _AddTest(test_class, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test_class, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test_class, test_name, fn)


def _RandomPDMatrix(n, rng):
  """Random positive definite matrix."""
  temp = rng.randn(n, n)
  return temp.dot(temp.T)


class CholeskySolveTest(test.TestCase):

  def setUp(self):
    self.rng = np.random.RandomState(0)

  def test_works_with_five_different_random_pos_def_matrices(self):
    with self.test_session(use_gpu=True):
      for n in range(1, 6):
        for np_type, atol in [(np.float32, 0.05), (np.float64, 1e-5)]:
          # Create 2 x n x n matrix
          array = np.array(
              [_RandomPDMatrix(n, self.rng),
               _RandomPDMatrix(n, self.rng)]).astype(np_type)
          chol = linalg_ops.cholesky(array)
          for k in range(1, 3):
            rhs = self.rng.randn(2, n, k).astype(np_type)
            x = linalg_ops.cholesky_solve(chol, rhs)
            self.assertAllClose(
                rhs, math_ops.matmul(array, x).eval(), atol=atol)


class EyeTest(test.TestCase):
  pass  # Will be filled in below


def _GetEyeTest(num_rows, num_columns, batch_shape, dtype):

  def Test(self):
    eye_np = np.eye(num_rows, M=num_columns, dtype=dtype.as_numpy_dtype)
    if batch_shape is not None:
      eye_np = np.tile(eye_np, batch_shape + [1, 1])
    for use_placeholder in False, True:
      if use_placeholder and (num_columns is None or batch_shape is None):
        return
      with self.test_session(use_gpu=True) as sess:
        if use_placeholder:
          num_rows_placeholder = array_ops.placeholder(
              dtypes.int32, name="num_rows")
          num_columns_placeholder = array_ops.placeholder(
              dtypes.int32, name="num_columns")
          batch_shape_placeholder = array_ops.placeholder(
              dtypes.int32, name="batch_shape")
          eye = linalg_ops.eye(
              num_rows_placeholder,
              num_columns=num_columns_placeholder,
              batch_shape=batch_shape_placeholder,
              dtype=dtype)
          eye_tf = sess.run(
              eye,
              feed_dict={
                  num_rows_placeholder: num_rows,
                  num_columns_placeholder: num_columns,
                  batch_shape_placeholder: batch_shape
              })
        else:
          eye_tf = linalg_ops.eye(
              num_rows,
              num_columns=num_columns,
              batch_shape=batch_shape,
              dtype=dtype).eval()
        self.assertAllEqual(eye_np, eye_tf)

  return Test


if __name__ == "__main__":
  for _num_rows in 0, 1, 2, 5:
    for _num_columns in None, 0, 1, 2, 5:
      for _batch_shape in None, [], [2], [2, 3]:
        for _dtype in (dtypes.int32, dtypes.int64, dtypes.float32,
                       dtypes.float64, dtypes.complex64, dtypes.complex128):
          name = "dtype_%s_num_rows_%s_num_column_%s_batch_shape_%s_" % (
              _dtype.name, _num_rows, _num_columns, _batch_shape)
          _AddTest(EyeTest, "EyeTest", name,
                   _GetEyeTest(_num_rows, _num_columns, _batch_shape, _dtype))

  test.main()
