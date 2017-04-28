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


def _random_pd_matrix(n, rng):
  """Random postive definite matrix."""
  temp = rng.randn(n, n)
  return temp.dot(temp.T)


class CholeskySolveTest(test.TestCase):
  _use_gpu = False

  def setUp(self):
    self.rng = np.random.RandomState(0)

  def test_works_with_five_different_random_pos_def_matrices(self):
    with self.test_session():
      for n in range(1, 6):
        for np_type, atol in [(np.float32, 0.05), (np.float64, 1e-5)]:
          # Create 2 x n x n matrix
          array = np.array(
              [_random_pd_matrix(n, self.rng), _random_pd_matrix(n, self.rng)
              ]).astype(np_type)
          chol = linalg_ops.cholesky(array)
          for k in range(1, 3):
            rhs = self.rng.randn(2, n, k).astype(np_type)
            x = linalg_ops.cholesky_solve(chol, rhs)
            self.assertAllClose(
                rhs, math_ops.matmul(array, x).eval(), atol=atol)


class CholeskySolveGpuTest(CholeskySolveTest):
  _use_gpu = True


class EyeTest(test.TestCase):

  def test_non_batch_2x2(self):
    num_rows = 2
    dtype = np.float32
    np_eye = np.eye(num_rows).astype(dtype)
    with self.test_session():
      eye = linalg_ops.eye(num_rows, dtype=dtype)
      self.assertAllEqual((num_rows, num_rows), eye.get_shape())
      self.assertAllEqual(np_eye, eye.eval())

  def test_non_batch_2x3(self):
    num_rows = 2
    num_columns = 3
    dtype = np.float32
    np_eye = np.eye(num_rows, num_columns).astype(dtype)
    with self.test_session():
      eye = linalg_ops.eye(num_rows, num_columns=num_columns, dtype=dtype)
      self.assertAllEqual((num_rows, num_columns), eye.get_shape())
      self.assertAllEqual(np_eye, eye.eval())

  def test_1x3_batch_4x4(self):
    num_rows = 4
    batch_shape = [1, 3]
    dtype = np.float32
    np_eye = np.eye(num_rows).astype(dtype)
    with self.test_session():
      eye = linalg_ops.eye(num_rows, batch_shape=batch_shape, dtype=dtype)
      self.assertAllEqual(batch_shape + [num_rows, num_rows], eye.get_shape())
      eye_v = eye.eval()
      for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
          self.assertAllEqual(np_eye, eye_v[i, j, :, :])

  def test_1x3_batch_4x4_dynamic(self):
    num_rows = 4
    batch_shape = [1, 3]
    dtype = np.float32
    np_eye = np.eye(num_rows).astype(dtype)
    with self.test_session():
      num_rows_ph = array_ops.placeholder(dtypes.int32)
      batch_shape_ph = array_ops.placeholder(dtypes.int32)
      eye = linalg_ops.eye(num_rows_ph, batch_shape=batch_shape_ph, dtype=dtype)
      eye_v = eye.eval(
          feed_dict={num_rows_ph: num_rows,
                     batch_shape_ph: batch_shape})
      for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
          self.assertAllEqual(np_eye, eye_v[i, j, :, :])

  def test_1x3_batch_5x4(self):
    num_rows = 5
    num_columns = 4
    batch_shape = [1, 3]
    dtype = np.float32
    np_eye = np.eye(num_rows, num_columns).astype(dtype)
    with self.test_session():
      eye = linalg_ops.eye(num_rows,
                           num_columns=num_columns,
                           batch_shape=batch_shape,
                           dtype=dtype)
      self.assertAllEqual(batch_shape + [num_rows, num_columns],
                          eye.get_shape())
      eye_v = eye.eval()
      for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
          self.assertAllEqual(np_eye, eye_v[i, j, :, :])

  def test_1x3_batch_5x4_dynamic(self):
    num_rows = 5
    num_columns = 4
    batch_shape = [1, 3]
    dtype = np.float32
    np_eye = np.eye(num_rows, num_columns).astype(dtype)
    with self.test_session():
      num_rows_ph = array_ops.placeholder(dtypes.int32)
      num_columns_ph = array_ops.placeholder(dtypes.int32)
      batch_shape_ph = array_ops.placeholder(dtypes.int32)
      eye = linalg_ops.eye(num_rows_ph,
                           num_columns=num_columns_ph,
                           batch_shape=batch_shape_ph,
                           dtype=dtype)
      eye_v = eye.eval(feed_dict={
          num_rows_ph: num_rows,
          num_columns_ph: num_columns,
          batch_shape_ph: batch_shape
      })
      for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
          self.assertAllEqual(np_eye, eye_v[i, j, :, :])

  def test_non_batch_0x0(self):
    num_rows = 0
    dtype = np.int64
    np_eye = np.eye(num_rows).astype(dtype)
    with self.test_session():
      eye = linalg_ops.eye(num_rows, dtype=dtype)
      self.assertAllEqual((num_rows, num_rows), eye.get_shape())
      self.assertAllEqual(np_eye, eye.eval())

  def test_non_batch_2x0(self):
    num_rows = 2
    num_columns = 0
    dtype = np.int64
    np_eye = np.eye(num_rows, num_columns).astype(dtype)
    with self.test_session():
      eye = linalg_ops.eye(num_rows, num_columns=num_columns, dtype=dtype)
      self.assertAllEqual((num_rows, num_columns), eye.get_shape())
      self.assertAllEqual(np_eye, eye.eval())

  def test_non_batch_0x2(self):
    num_rows = 0
    num_columns = 2
    dtype = np.int64
    np_eye = np.eye(num_rows, num_columns).astype(dtype)
    with self.test_session():
      eye = linalg_ops.eye(num_rows, num_columns=num_columns, dtype=dtype)
      self.assertAllEqual((num_rows, num_columns), eye.get_shape())
      self.assertAllEqual(np_eye, eye.eval())

  def test_1x3_batch_0x0(self):
    num_rows = 0
    batch_shape = [1, 3]
    dtype = np.float32
    np_eye = np.eye(num_rows).astype(dtype)
    with self.test_session():
      eye = linalg_ops.eye(num_rows, batch_shape=batch_shape, dtype=dtype)
      self.assertAllEqual((1, 3, 0, 0), eye.get_shape())
      eye_v = eye.eval()
      for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
          self.assertAllEqual(np_eye, eye_v[i, j, :, :])

  def test_1x3_batch_2x0(self):
    num_rows = 2
    num_columns = 0
    batch_shape = [1, 3]
    dtype = np.float32
    np_eye = np.eye(num_rows, num_columns).astype(dtype)
    with self.test_session():
      eye = linalg_ops.eye(num_rows,
                           num_columns=num_columns,
                           batch_shape=batch_shape,
                           dtype=dtype)
      self.assertAllEqual(batch_shape + [num_rows, num_columns],
                          eye.get_shape())
      eye_v = eye.eval()
      for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
          self.assertAllEqual(np_eye, eye_v[i, j, :, :])

  def test_1x3_batch_0x2(self):
    num_rows = 0
    num_columns = 2
    batch_shape = [1, 3]
    dtype = np.float32
    np_eye = np.eye(num_rows, num_columns).astype(dtype)
    with self.test_session():
      eye = linalg_ops.eye(num_rows,
                           num_columns=num_columns,
                           batch_shape=batch_shape,
                           dtype=dtype)
      self.assertAllEqual(batch_shape + [num_rows, num_columns],
                          eye.get_shape())
      eye_v = eye.eval()
      for i in range(batch_shape[0]):
        for j in range(batch_shape[1]):
          self.assertAllEqual(np_eye, eye_v[i, j, :, :])


if __name__ == '__main__':
  test.main()
