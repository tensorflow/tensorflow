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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test


rng = np.random.RandomState(2016)


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorZerosTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @staticmethod
  def skip_these_tests():
    return [
        "cholesky",
        "cond",
        "inverse",
        "log_abs_det",
        "solve",
        "solve_with_broadcast"
    ]

  @staticmethod
  def operator_shapes_infos():
    shapes_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shapes_info((1, 1)),
        shapes_info((1, 3, 3)),
        shapes_info((3, 4, 4)),
        shapes_info((2, 1, 4, 4))]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    del ensure_self_adjoint_and_pd
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
    with self.cached_session():
      operator = linalg_lib.LinearOperatorZeros(num_rows=2)
      self.evaluate(operator.assert_self_adjoint())  # Should not fail

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
    with self.cached_session():
      num_rows = array_ops.placeholder_with_default([2], shape=None)
      with self.assertRaisesError("must be a 0-D Tensor"):
        operator = linalg_lib.LinearOperatorZeros(
            num_rows, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_negative_num_rows_raises_dynamic(self):
    with self.cached_session():
      n = array_ops.placeholder_with_default(-2, shape=None)
      with self.assertRaisesError("must be non-negative"):
        operator = linalg_lib.LinearOperatorZeros(
            num_rows=n, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_non_1d_batch_shape_raises_dynamic(self):
    with self.cached_session():
      batch_shape = array_ops.placeholder_with_default(2, shape=None)
      with self.assertRaisesError("must be a 1-D"):
        operator = linalg_lib.LinearOperatorZeros(
            num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_negative_batch_shape_raises_dynamic(self):
    with self.cached_session():
      batch_shape = array_ops.placeholder_with_default([-2], shape=None)
      with self.assertRaisesError("must be non-negative"):
        operator = linalg_lib.LinearOperatorZeros(
            num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
        self.evaluate(operator.to_dense())

  def test_wrong_matrix_dimensions_raises_static(self):
    operator = linalg_lib.LinearOperatorZeros(num_rows=2)
    x = rng.randn(3, 3).astype(np.float32)
    with self.assertRaisesRegexp(ValueError, "Dimensions.*not compatible"):
      operator.matmul(x)

  def test_wrong_matrix_dimensions_raises_dynamic(self):
    num_rows = array_ops.placeholder_with_default(2, shape=None)
    x = array_ops.placeholder_with_default(rng.rand(3, 3), shape=None)

    with self.cached_session():
      with self.assertRaisesError("Dimensions.*not.compatible"):
        operator = linalg_lib.LinearOperatorZeros(
            num_rows, assert_proper_shapes=True, dtype=dtypes.float64)
        self.evaluate(operator.matmul(x))

  def test_is_x_flags(self):
    # The is_x flags are by default all True.
    operator = linalg_lib.LinearOperatorZeros(num_rows=2)
    self.assertFalse(operator.is_positive_definite)
    self.assertFalse(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)

  def test_zeros_matmul(self):
    operator1 = linalg_lib.LinearOperatorIdentity(num_rows=2)
    operator2 = linalg_lib.LinearOperatorZeros(num_rows=2)
    self.assertTrue(isinstance(
        operator1.matmul(operator2),
        linalg_lib.LinearOperatorZeros))

    self.assertTrue(isinstance(
        operator2.matmul(operator1),
        linalg_lib.LinearOperatorZeros))

  def test_ref_type_shape_args_raises(self):
    with self.assertRaisesRegexp(TypeError, "num_rows.cannot.be.reference"):
      linalg_lib.LinearOperatorZeros(num_rows=variables_module.Variable(2))

    with self.assertRaisesRegexp(TypeError, "num_columns.cannot.be.reference"):
      linalg_lib.LinearOperatorZeros(
          num_rows=2, num_columns=variables_module.Variable(3))

    with self.assertRaisesRegexp(TypeError, "batch_shape.cannot.be.reference"):
      linalg_lib.LinearOperatorZeros(
          num_rows=2, batch_shape=variables_module.Variable([2]))


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorZerosNotSquareTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    del use_placeholder
    del ensure_self_adjoint_and_pd
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
  linear_operator_test_util.add_tests(LinearOperatorZerosTest)
  linear_operator_test_util.add_tests(LinearOperatorZerosNotSquareTest)
  test.main()
