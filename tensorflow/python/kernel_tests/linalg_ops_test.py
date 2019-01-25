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

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg
from tensorflow.python.platform import test


def _AddTest(test_class, op_name, testcase_name, fn):
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test_class, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test_class, test_name, fn)


def _RandomPDMatrix(n, rng, dtype=np.float64):
  """Random positive definite matrix."""
  temp = rng.randn(n, n).astype(dtype)
  if dtype in [np.complex64, np.complex128]:
    temp.imag = rng.randn(n, n)
  return np.conj(temp).dot(temp.T)


class CholeskySolveTest(test.TestCase):

  def setUp(self):
    self.rng = np.random.RandomState(0)

  @test_util.run_deprecated_v1
  def test_works_with_five_different_random_pos_def_matrices(self):
    for n in range(1, 6):
      for np_type, atol in [(np.float32, 0.05), (np.float64, 1e-5)]:
        with self.session(use_gpu=True):
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


class LogdetTest(test.TestCase):

  def setUp(self):
    self.rng = np.random.RandomState(42)

  @test_util.run_deprecated_v1
  def test_works_with_five_different_random_pos_def_matrices(self):
    for n in range(1, 6):
      for np_dtype, atol in [(np.float32, 0.05), (np.float64, 1e-5),
                             (np.complex64, 0.05), (np.complex128, 1e-5)]:
        matrix = _RandomPDMatrix(n, self.rng, np_dtype)
        _, logdet_np = np.linalg.slogdet(matrix)
        with self.session(use_gpu=True):
          # Create 2 x n x n matrix
          # matrix = np.array(
          #     [_RandomPDMatrix(n, self.rng, np_dtype),
          #      _RandomPDMatrix(n, self.rng, np_dtype)]).astype(np_dtype)
          logdet_tf = linalg.logdet(matrix)
          self.assertAllClose(logdet_np, self.evaluate(logdet_tf), atol=atol)

  def test_works_with_underflow_case(self):
    for np_dtype, atol in [(np.float32, 0.05), (np.float64, 1e-5),
                           (np.complex64, 0.05), (np.complex128, 1e-5)]:
      matrix = (np.eye(20) * 1e-6).astype(np_dtype)
      _, logdet_np = np.linalg.slogdet(matrix)
      with self.session(use_gpu=True):
        logdet_tf = linalg.logdet(matrix)
        self.assertAllClose(logdet_np, self.evaluate(logdet_tf), atol=atol)


class SlogdetTest(test.TestCase):

  def setUp(self):
    self.rng = np.random.RandomState(42)

  @test_util.run_deprecated_v1
  def test_works_with_five_different_random_pos_def_matrices(self):
    for n in range(1, 6):
      for np_dtype, atol in [(np.float32, 0.05), (np.float64, 1e-5),
                             (np.complex64, 0.05), (np.complex128, 1e-5)]:
        matrix = _RandomPDMatrix(n, self.rng, np_dtype)
        sign_np, log_abs_det_np = np.linalg.slogdet(matrix)
        with self.session(use_gpu=True):
          sign_tf, log_abs_det_tf = linalg.slogdet(matrix)
          self.assertAllClose(
              log_abs_det_np, self.evaluate(log_abs_det_tf), atol=atol)
          self.assertAllClose(sign_np, self.evaluate(sign_tf), atol=atol)

  def test_works_with_underflow_case(self):
    for np_dtype, atol in [(np.float32, 0.05), (np.float64, 1e-5),
                           (np.complex64, 0.05), (np.complex128, 1e-5)]:
      matrix = (np.eye(20) * 1e-6).astype(np_dtype)
      sign_np, log_abs_det_np = np.linalg.slogdet(matrix)
      with self.session(use_gpu=True):
        sign_tf, log_abs_det_tf = linalg.slogdet(matrix)
        self.assertAllClose(
            log_abs_det_np, self.evaluate(log_abs_det_tf), atol=atol)
        self.assertAllClose(sign_np, self.evaluate(sign_tf), atol=atol)


class AdjointTest(test.TestCase):

  def test_compare_to_numpy(self):
    for dtype in np.float64, np.float64, np.complex64, np.complex128:
      matrix_np = np.array([[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j,
                                                       6 + 6j]]).astype(dtype)
      expected_transposed = np.conj(matrix_np.T)
      with self.session():
        matrix = ops.convert_to_tensor(matrix_np)
        transposed = linalg.adjoint(matrix)
        self.assertEqual((3, 2), transposed.get_shape())
        self.assertAllEqual(expected_transposed, self.evaluate(transposed))


class EyeTest(parameterized.TestCase, test.TestCase):

  def testShapeInferenceNoBatch(self):
    self.assertEqual((2, 2), linalg_ops.eye(num_rows=2).shape)
    self.assertEqual((2, 3), linalg_ops.eye(num_rows=2, num_columns=3).shape)

  def testShapeInferenceStaticBatch(self):
    batch_shape = (2, 3)
    self.assertEqual(
        (2, 3, 2, 2),
        linalg_ops.eye(num_rows=2, batch_shape=batch_shape).shape)
    self.assertEqual(
        (2, 3, 2, 3),
        linalg_ops.eye(
            num_rows=2, num_columns=3, batch_shape=batch_shape).shape)

  @parameterized.named_parameters(
      ("DynamicRow",
       lambda: array_ops.placeholder_with_default(2, shape=None),
       lambda: None),
      ("DynamicRowStaticColumn",
       lambda: array_ops.placeholder_with_default(2, shape=None),
       lambda: 3),
      ("StaticRowDynamicColumn",
       lambda: 2,
       lambda: array_ops.placeholder_with_default(3, shape=None)),
      ("DynamicRowDynamicColumn",
       lambda: array_ops.placeholder_with_default(2, shape=None),
       lambda: array_ops.placeholder_with_default(3, shape=None)))
  def testShapeInferenceStaticBatchWith(self, num_rows_fn, num_columns_fn):
    num_rows = num_rows_fn()
    num_columns = num_columns_fn()
    batch_shape = (2, 3)
    identity_matrix = linalg_ops.eye(
        num_rows=num_rows,
        num_columns=num_columns,
        batch_shape=batch_shape)
    self.assertEqual(4, identity_matrix.shape.ndims)
    self.assertEqual((2, 3), identity_matrix.shape[:2])
    if num_rows is not None and not isinstance(num_rows, ops.Tensor):
      self.assertEqual(2, identity_matrix.shape[-2])

    if num_columns is not None and not isinstance(num_columns, ops.Tensor):
      self.assertEqual(3, identity_matrix.shape[-1])

  @parameterized.parameters(
      itertools.product(
          # num_rows
          [0, 1, 2, 5],
          # num_columns
          [None, 0, 1, 2, 5],
          # batch_shape
          [None, [], [2], [2, 3]],
          # dtype
          [
              dtypes.int32,
              dtypes.int64,
              dtypes.float32,
              dtypes.float64,
              dtypes.complex64,
              dtypes.complex128
          ])
      )
  def test_eye_no_placeholder(self, num_rows, num_columns, batch_shape, dtype):
    eye_np = np.eye(num_rows, M=num_columns, dtype=dtype.as_numpy_dtype)
    if batch_shape is not None:
      eye_np = np.tile(eye_np, batch_shape + [1, 1])
    eye_tf = self.evaluate(linalg_ops.eye(
        num_rows,
        num_columns=num_columns,
        batch_shape=batch_shape,
        dtype=dtype))
    self.assertAllEqual(eye_np, eye_tf)

  @parameterized.parameters(
      itertools.product(
          # num_rows
          [0, 1, 2, 5],
          # num_columns
          [0, 1, 2, 5],
          # batch_shape
          [[], [2], [2, 3]],
          # dtype
          [
              dtypes.int32,
              dtypes.int64,
              dtypes.float32,
              dtypes.float64,
              dtypes.complex64,
              dtypes.complex128
          ])
      )
  @test_util.run_deprecated_v1
  def test_eye_with_placeholder(
      self, num_rows, num_columns, batch_shape, dtype):
    eye_np = np.eye(num_rows, M=num_columns, dtype=dtype.as_numpy_dtype)
    eye_np = np.tile(eye_np, batch_shape + [1, 1])
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
    with self.session(use_gpu=True) as sess:
      eye_tf = sess.run(
          eye,
          feed_dict={
              num_rows_placeholder: num_rows,
              num_columns_placeholder: num_columns,
              batch_shape_placeholder: batch_shape
          })
    self.assertAllEqual(eye_np, eye_tf)


if __name__ == "__main__":
  test.main()
