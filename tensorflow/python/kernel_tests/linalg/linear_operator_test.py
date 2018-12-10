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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(123)


class LinearOperatorShape(linalg.LinearOperator):
  """LinearOperator that implements the methods ._shape and _shape_tensor."""

  def __init__(self,
               shape,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None):
    self._stored_shape = shape
    super(LinearOperatorShape, self).__init__(
        dtype=dtypes.float32,
        graph_parents=None,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square)

  def _shape(self):
    return tensor_shape.TensorShape(self._stored_shape)

  def _shape_tensor(self):
    return constant_op.constant(self._stored_shape, dtype=dtypes.int32)

  def _matmul(self):
    raise NotImplementedError("Not needed for this test.")


class LinearOperatorMatmulSolve(linalg.LinearOperator):
  """LinearOperator that wraps a [batch] matrix and implements matmul/solve."""

  def __init__(self,
               matrix,
               is_non_singular=None,
               is_self_adjoint=None,
               is_positive_definite=None,
               is_square=None):
    self._matrix = ops.convert_to_tensor(matrix, name="matrix")
    super(LinearOperatorMatmulSolve, self).__init__(
        dtype=self._matrix.dtype,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square)

  def _shape(self):
    return self._matrix.get_shape()

  def _shape_tensor(self):
    return array_ops.shape(self._matrix)

  def _matmul(self, x, adjoint=False, adjoint_arg=False):
    x = ops.convert_to_tensor(x, name="x")
    return math_ops.matmul(
        self._matrix, x, adjoint_a=adjoint, adjoint_b=adjoint_arg)

  def _solve(self, rhs, adjoint=False, adjoint_arg=False):
    rhs = ops.convert_to_tensor(rhs, name="rhs")
    assert not adjoint_arg, "Not implemented for this test class."
    return linalg_ops.matrix_solve(self._matrix, rhs, adjoint=adjoint)


class LinearOperatorTest(test.TestCase):

  def test_all_shape_properties_defined_by_the_one_property_shape(self):

    shape = (1, 2, 3, 4)
    operator = LinearOperatorShape(shape)

    self.assertAllEqual(shape, operator.shape)
    self.assertAllEqual(4, operator.tensor_rank)
    self.assertAllEqual((1, 2), operator.batch_shape)
    self.assertAllEqual(4, operator.domain_dimension)
    self.assertAllEqual(3, operator.range_dimension)

  @test_util.run_deprecated_v1
  def test_all_shape_methods_defined_by_the_one_method_shape(self):
    with self.cached_session():
      shape = (1, 2, 3, 4)
      operator = LinearOperatorShape(shape)

      self.assertAllEqual(shape, operator.shape_tensor().eval())
      self.assertAllEqual(4, operator.tensor_rank_tensor().eval())
      self.assertAllEqual((1, 2), operator.batch_shape_tensor().eval())
      self.assertAllEqual(4, operator.domain_dimension_tensor().eval())
      self.assertAllEqual(3, operator.range_dimension_tensor().eval())

  def test_is_x_properties(self):
    operator = LinearOperatorShape(
        shape=(2, 2),
        is_non_singular=False,
        is_self_adjoint=True,
        is_positive_definite=False)
    self.assertFalse(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)
    self.assertFalse(operator.is_positive_definite)

  def test_generic_to_dense_method_non_square_matrix_static(self):
    matrix = rng.randn(2, 3, 4)
    operator = LinearOperatorMatmulSolve(matrix)
    with self.cached_session():
      operator_dense = operator.to_dense()
      self.assertAllEqual((2, 3, 4), operator_dense.get_shape())
      self.assertAllClose(matrix, self.evaluate(operator_dense))

  @test_util.run_deprecated_v1
  def test_generic_to_dense_method_non_square_matrix_tensor(self):
    matrix = rng.randn(2, 3, 4)
    matrix_ph = array_ops.placeholder(dtypes.float64)
    operator = LinearOperatorMatmulSolve(matrix_ph)
    with self.cached_session():
      operator_dense = operator.to_dense()
      self.assertAllClose(
          matrix, operator_dense.eval(feed_dict={matrix_ph: matrix}))

  def test_matvec(self):
    matrix = [[1., 0], [0., 2.]]
    operator = LinearOperatorMatmulSolve(matrix)
    x = [1., 1.]
    with self.cached_session():
      y = operator.matvec(x)
      self.assertAllEqual((2,), y.get_shape())
      self.assertAllClose([1., 2.], self.evaluate(y))

  def test_solvevec(self):
    matrix = [[1., 0], [0., 2.]]
    operator = LinearOperatorMatmulSolve(matrix)
    y = [1., 1.]
    with self.cached_session():
      x = operator.solvevec(y)
      self.assertAllEqual((2,), x.get_shape())
      self.assertAllClose([1., 1 / 2.], self.evaluate(x))

  def test_is_square_set_to_true_for_square_static_shapes(self):
    operator = LinearOperatorShape(shape=(2, 4, 4))
    self.assertTrue(operator.is_square)

  def test_is_square_set_to_false_for_square_static_shapes(self):
    operator = LinearOperatorShape(shape=(2, 3, 4))
    self.assertFalse(operator.is_square)

  def test_is_square_set_incorrectly_to_false_raises(self):
    with self.assertRaisesRegexp(ValueError, "but.*was square"):
      _ = LinearOperatorShape(shape=(2, 4, 4), is_square=False).is_square

  @test_util.run_deprecated_v1
  def test_is_square_set_inconsistent_with_other_hints_raises(self):
    with self.assertRaisesRegexp(ValueError, "is always square"):
      matrix = array_ops.placeholder(dtypes.float32)
      LinearOperatorMatmulSolve(matrix, is_non_singular=True, is_square=False)

    with self.assertRaisesRegexp(ValueError, "is always square"):
      matrix = array_ops.placeholder(dtypes.float32)
      LinearOperatorMatmulSolve(
          matrix, is_positive_definite=True, is_square=False)

  @test_util.run_deprecated_v1
  def test_non_square_operators_raise_on_determinant_and_solve(self):
    operator = LinearOperatorShape((2, 3))
    with self.assertRaisesRegexp(NotImplementedError, "not be square"):
      operator.determinant()
    with self.assertRaisesRegexp(NotImplementedError, "not be square"):
      operator.log_abs_determinant()
    with self.assertRaisesRegexp(NotImplementedError, "not be square"):
      operator.solve(rng.rand(2, 2))

    with self.assertRaisesRegexp(ValueError, "is always square"):
      matrix = array_ops.placeholder(dtypes.float32)
      LinearOperatorMatmulSolve(
          matrix, is_positive_definite=True, is_square=False)

  @test_util.run_deprecated_v1
  def test_is_square_manual_set_works(self):
    matrix = array_ops.placeholder(dtypes.float32)
    # Default is None.
    operator = LinearOperatorMatmulSolve(matrix)
    self.assertEqual(None, operator.is_square)
    # Set to True
    operator = LinearOperatorMatmulSolve(matrix, is_square=True)
    self.assertTrue(operator.is_square)

  @test_util.run_v1_only("b/120545219")
  def test_linear_operator_matmul_hints_closed(self):
    matrix = array_ops.placeholder(dtypes.float32)
    operator1 = LinearOperatorMatmulSolve(matrix)

    operator_matmul = operator1.matmul(operator1)

    self.assertEqual(None, operator_matmul.is_square)
    self.assertEqual(None, operator_matmul.is_non_singular)
    self.assertEqual(None, operator_matmul.is_self_adjoint)
    self.assertEqual(None, operator_matmul.is_positive_definite)

    operator2 = LinearOperatorMatmulSolve(
        matrix,
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=True,
        is_square=True,
    )

    operator_matmul = operator2.matmul(operator2)

    self.assertTrue(operator_matmul.is_square)
    self.assertTrue(operator_matmul.is_non_singular)
    self.assertTrue(operator_matmul.is_self_adjoint)
    self.assertEqual(None, operator_matmul.is_positive_definite)

  @test_util.run_v1_only("b/120545219")
  def test_linear_operator_matmul_hints_false(self):
    matrix = array_ops.placeholder(dtypes.float32)
    operator1 = LinearOperatorMatmulSolve(
        matrix,
        is_non_singular=False,
        is_self_adjoint=False,
        is_positive_definite=False,
        is_square=True,
    )

    operator_matmul = operator1.matmul(operator1)

    self.assertTrue(operator_matmul.is_square)
    self.assertFalse(operator_matmul.is_non_singular)
    self.assertEqual(None, operator_matmul.is_self_adjoint)
    self.assertEqual(None, operator_matmul.is_positive_definite)

    operator2 = LinearOperatorMatmulSolve(
        matrix,
        is_non_singular=False,
        is_self_adjoint=False,
        is_positive_definite=False,
        is_square=False,
    )

    operator_matmul = operator2.matmul(operator2)

    self.assertEqual(None, operator_matmul.is_square)
    self.assertEqual(None, operator_matmul.is_non_singular)
    self.assertEqual(None, operator_matmul.is_self_adjoint)
    self.assertEqual(None, operator_matmul.is_positive_definite)

  @test_util.run_v1_only("b/120545219")
  def test_linear_operator_matmul_hint_infer_square(self):
    matrix1 = array_ops.placeholder(shape=[2, 3], dtype=dtypes.float32)
    matrix2 = array_ops.placeholder(shape=[3, 2], dtype=dtypes.float32)
    matrix3 = array_ops.placeholder(shape=[3, 4], dtype=dtypes.float32)

    operator1 = LinearOperatorMatmulSolve(matrix1, is_square=False)
    operator2 = LinearOperatorMatmulSolve(matrix2, is_square=False)
    operator3 = LinearOperatorMatmulSolve(matrix3, is_square=False)

    self.assertTrue(operator1.matmul(operator2).is_square)
    self.assertTrue(operator2.matmul(operator1).is_square)
    self.assertFalse(operator1.matmul(operator3).is_square)


if __name__ == "__main__":
  test.main()
