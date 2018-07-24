# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test


random_seed.set_random_seed(23)
rng = np.random.RandomState(2016)


class LinearOperatorZerosTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @property
  def _tests_to_skip(self):
    return ["log_abs_det", "solve", "solve_with_broadcast"]

  @property
  def _operator_build_infos(self):
    build_info = linear_operator_test_util.OperatorBuildInfo
    return [
        build_info((1, 1)),
        build_info((1, 3, 3)),
        build_info((3, 4, 4)),
        build_info((2, 1, 4, 4))]

  def _operator_and_matrix(self, build_info, dtype, use_placeholder):
    del use_placeholder
    shape = list(build_info.shape)
    assert shape[-1] == shape[-2]

    batch_shape = shape[:-2]
    num_rows = shape[-1]

    operator = linalg_lib.LinearOperatorZeros(
        num_rows, batch_shape=batch_shape, dtype=dtype)
    matrix = array_ops.zeros(shape=shape, dtype=dtype)

    return operator, matrix

  def test_assert_positive_definite(self):
    operator = linalg_lib.LinearOperatorZeros(num_rows=2)
    with self.assertRaisesOpError("non-positive definite"):
      operator.assert_positive_definite()

  def test_assert_non_singular(self):
    with self.assertRaisesOpError("non-invertible"):
      operator = linalg_lib.LinearOperatorZeros(num_rows=2)
      operator.assert_non_singular()

  def test_assert_self_adjoint(self):
    with self.test_session():
      operator = linalg_lib.LinearOperatorZeros(num_rows=2)
      operator.assert_self_adjoint().run()  # Should not fail

  def test_non_scalar_num_rows_raises_static(self):
    with self.assertRaisesRegexp(ValueError, "must be a 0-D Tensor"):
      linalg_lib.LinearOperatorZeros(num_rows=[2])
    with self.assertRaisesRegexp(ValueError, "must be a 0-D Tensor"):
      linalg_lib.LinearOperatorZeros(num_rows=2, num_columns=[2])

  def test_non_integer_num_rows_raises_static(self):
    with self.assertRaisesRegexp(TypeError, "must be integer"):
      linalg_lib.LinearOperatorZeros(num_rows=2.)
    with self.assertRaisesRegexp(TypeError, "must be integer"):
      linalg_lib.LinearOperatorZeros(num_rows=2, num_columns=2.)

  def test_negative_num_rows_raises_static(self):
    with self.assertRaisesRegexp(ValueError, "must be non-negative"):
      linalg_lib.LinearOperatorZeros(num_rows=-2)
    with self.assertRaisesRegexp(ValueError, "must be non-negative"):
      linalg_lib.LinearOperatorZeros(num_rows=2, num_columns=-2)

  def test_non_1d_batch_shape_raises_static(self):
    with self.assertRaisesRegexp(ValueError, "must be a 1-D"):
      linalg_lib.LinearOperatorZeros(num_rows=2, batch_shape=2)

  def test_non_integer_batch_shape_raises_static(self):
    with self.assertRaisesRegexp(TypeError, "must be integer"):
      linalg_lib.LinearOperatorZeros(num_rows=2, batch_shape=[2.])

  def test_negative_batch_shape_raises_static(self):
    with self.assertRaisesRegexp(ValueError, "must be non-negative"):
      linalg_lib.LinearOperatorZeros(num_rows=2, batch_shape=[-2])

  def test_non_scalar_num_rows_raises_dynamic(self):
    with self.test_session():
      num_rows = array_ops.placeholder(dtypes.int32)
      operator = linalg_lib.LinearOperatorZeros(
          num_rows, assert_proper_shapes=True)
      with self.assertRaisesOpError("must be a 0-D Tensor"):
        operator.to_dense().eval(feed_dict={num_rows: [2]})

  def test_negative_num_rows_raises_dynamic(self):
    with self.test_session():
      n = array_ops.placeholder(dtypes.int32)
      operator = linalg_lib.LinearOperatorZeros(
          num_rows=n, assert_proper_shapes=True)
      with self.assertRaisesOpError("must be non-negative"):
        operator.to_dense().eval(feed_dict={n: -2})

      operator = linalg_lib.LinearOperatorZeros(
          num_rows=2, num_columns=n, assert_proper_shapes=True)
      with self.assertRaisesOpError("must be non-negative"):
        operator.to_dense().eval(feed_dict={n: -2})

  def test_non_1d_batch_shape_raises_dynamic(self):
    with self.test_session():
      batch_shape = array_ops.placeholder(dtypes.int32)
      operator = linalg_lib.LinearOperatorZeros(
          num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
      with self.assertRaisesOpError("must be a 1-D"):
        operator.to_dense().eval(feed_dict={batch_shape: 2})

  def test_negative_batch_shape_raises_dynamic(self):
    with self.test_session():
      batch_shape = array_ops.placeholder(dtypes.int32)
      operator = linalg_lib.LinearOperatorZeros(
          num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
      with self.assertRaisesOpError("must be non-negative"):
        operator.to_dense().eval(feed_dict={batch_shape: [-2]})

  def test_wrong_matrix_dimensions_raises_static(self):
    operator = linalg_lib.LinearOperatorZeros(num_rows=2)
    x = rng.randn(3, 3).astype(np.float32)
    with self.assertRaisesRegexp(ValueError, "Dimensions.*not compatible"):
      operator.matmul(x)

  def test_wrong_matrix_dimensions_raises_dynamic(self):
    num_rows = array_ops.placeholder(dtypes.int32)
    x = array_ops.placeholder(dtypes.float32)

    with self.test_session():
      operator = linalg_lib.LinearOperatorZeros(
          num_rows, assert_proper_shapes=True)
      y = operator.matmul(x)
      with self.assertRaisesOpError("Incompatible.*dimensions"):
        y.eval(feed_dict={num_rows: 2, x: rng.rand(3, 3)})

  def test_is_x_flags(self):
    # The is_x flags are by default all True.
    operator = linalg_lib.LinearOperatorZeros(num_rows=2)
    self.assertFalse(operator.is_positive_definite)
    self.assertFalse(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)


class LinearOperatorZerosNotSquareTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):

  def _operator_and_matrix(self, build_info, dtype, use_placeholder):
    del use_placeholder
    shape = list(build_info.shape)

    batch_shape = shape[:-2]
    num_rows = shape[-2]
    num_columns = shape[-1]

    operator = linalg_lib.LinearOperatorZeros(
        num_rows, num_columns, is_square=False, is_self_adjoint=False,
        batch_shape=batch_shape, dtype=dtype)
    matrix = array_ops.zeros(shape=shape, dtype=dtype)

    return operator, matrix


if __name__ == "__main__":
  test.main()
