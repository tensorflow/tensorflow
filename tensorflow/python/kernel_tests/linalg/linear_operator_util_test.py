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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test

rng = np.random.RandomState(0)


class AssertZeroImagPartTest(test.TestCase):

  def test_real_tensor_doesnt_raise(self):
    x = ops.convert_to_tensor([0., 2, 3])
    # Should not raise.
    self.evaluate(
        linear_operator_util.assert_zero_imag_part(x, message="ABC123"))

  def test_complex_tensor_with_imag_zero_doesnt_raise(self):
    x = ops.convert_to_tensor([1., 0, 3])
    y = ops.convert_to_tensor([0., 0, 0])
    z = math_ops.complex(x, y)
    # Should not raise.
    self.evaluate(
        linear_operator_util.assert_zero_imag_part(z, message="ABC123"))

  def test_complex_tensor_with_nonzero_imag_raises(self):
    x = ops.convert_to_tensor([1., 2, 0])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    with self.assertRaisesOpError("ABC123"):
      self.evaluate(
          linear_operator_util.assert_zero_imag_part(z, message="ABC123"))


class AssertNoEntriesWithModulusZeroTest(test.TestCase):

  def test_nonzero_real_tensor_doesnt_raise(self):
    x = ops.convert_to_tensor([1., 2, 3])
    # Should not raise.
    self.evaluate(
        linear_operator_util.assert_no_entries_with_modulus_zero(
            x, message="ABC123"))

  def test_nonzero_complex_tensor_doesnt_raise(self):
    x = ops.convert_to_tensor([1., 0, 3])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    # Should not raise.
    self.evaluate(
        linear_operator_util.assert_no_entries_with_modulus_zero(
            z, message="ABC123"))

  def test_zero_real_tensor_raises(self):
    x = ops.convert_to_tensor([1., 0, 3])
    with self.assertRaisesOpError("ABC123"):
      self.evaluate(
          linear_operator_util.assert_no_entries_with_modulus_zero(
              x, message="ABC123"))

  def test_zero_complex_tensor_raises(self):
    x = ops.convert_to_tensor([1., 2, 0])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    with self.assertRaisesOpError("ABC123"):
      self.evaluate(
          linear_operator_util.assert_no_entries_with_modulus_zero(
              z, message="ABC123"))


class BroadcastMatrixBatchDimsTest(test.TestCase):

  def test_zero_batch_matrices_returned_as_empty_list(self):
    self.assertAllEqual([],
                        linear_operator_util.broadcast_matrix_batch_dims([]))

  def test_one_batch_matrix_returned_after_tensor_conversion(self):
    arr = rng.rand(2, 3, 4)
    tensor, = linear_operator_util.broadcast_matrix_batch_dims([arr])
    self.assertTrue(isinstance(tensor, ops.Tensor))

    self.assertAllClose(arr, self.evaluate(tensor))

  def test_static_dims_broadcast(self):
    # x.batch_shape = [3, 1, 2]
    # y.batch_shape = [4, 1]
    # broadcast batch shape = [3, 4, 2]
    x = rng.rand(3, 1, 2, 1, 5)
    y = rng.rand(4, 1, 3, 7)
    batch_of_zeros = np.zeros((3, 4, 2, 1, 1))
    x_bc_expected = x + batch_of_zeros
    y_bc_expected = y + batch_of_zeros

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x, y])

    self.assertAllEqual(x_bc_expected.shape, x_bc.shape)
    self.assertAllEqual(y_bc_expected.shape, y_bc.shape)
    x_bc_, y_bc_ = self.evaluate([x_bc, y_bc])
    self.assertAllClose(x_bc_expected, x_bc_)
    self.assertAllClose(y_bc_expected, y_bc_)

  def test_static_dims_broadcast_second_arg_higher_rank(self):
    # x.batch_shape =    [1, 2]
    # y.batch_shape = [1, 3, 1]
    # broadcast batch shape = [1, 3, 2]
    x = rng.rand(1, 2, 1, 5)
    y = rng.rand(1, 3, 2, 3, 7)
    batch_of_zeros = np.zeros((1, 3, 2, 1, 1))
    x_bc_expected = x + batch_of_zeros
    y_bc_expected = y + batch_of_zeros

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x, y])

    self.assertAllEqual(x_bc_expected.shape, x_bc.shape)
    self.assertAllEqual(y_bc_expected.shape, y_bc.shape)
    x_bc_, y_bc_ = self.evaluate([x_bc, y_bc])
    self.assertAllClose(x_bc_expected, x_bc_)
    self.assertAllClose(y_bc_expected, y_bc_)

  def test_dynamic_dims_broadcast_32bit(self):
    # x.batch_shape = [3, 1, 2]
    # y.batch_shape = [4, 1]
    # broadcast batch shape = [3, 4, 2]
    x = rng.rand(3, 1, 2, 1, 5).astype(np.float32)
    y = rng.rand(4, 1, 3, 7).astype(np.float32)
    batch_of_zeros = np.zeros((3, 4, 2, 1, 1)).astype(np.float32)
    x_bc_expected = x + batch_of_zeros
    y_bc_expected = y + batch_of_zeros

    x_ph = array_ops.placeholder_with_default(x, shape=None)
    y_ph = array_ops.placeholder_with_default(y, shape=None)

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x_ph, y_ph])

    x_bc_, y_bc_ = self.evaluate([x_bc, y_bc])
    self.assertAllClose(x_bc_expected, x_bc_)
    self.assertAllClose(y_bc_expected, y_bc_)

  def test_dynamic_dims_broadcast_32bit_second_arg_higher_rank(self):
    # x.batch_shape =    [1, 2]
    # y.batch_shape = [3, 4, 1]
    # broadcast batch shape = [3, 4, 2]
    x = rng.rand(1, 2, 1, 5).astype(np.float32)
    y = rng.rand(3, 4, 1, 3, 7).astype(np.float32)
    batch_of_zeros = np.zeros((3, 4, 2, 1, 1)).astype(np.float32)
    x_bc_expected = x + batch_of_zeros
    y_bc_expected = y + batch_of_zeros

    x_ph = array_ops.placeholder_with_default(x, shape=None)
    y_ph = array_ops.placeholder_with_default(y, shape=None)

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x_ph, y_ph])

    x_bc_, y_bc_ = self.evaluate([x_bc, y_bc])
    self.assertAllClose(x_bc_expected, x_bc_)
    self.assertAllClose(y_bc_expected, y_bc_)

  def test_less_than_two_dims_raises_static(self):
    x = rng.rand(3)
    y = rng.rand(1, 1)

    with self.assertRaisesRegex(ValueError, "at least two dimensions"):
      linear_operator_util.broadcast_matrix_batch_dims([x, y])

    with self.assertRaisesRegex(ValueError, "at least two dimensions"):
      linear_operator_util.broadcast_matrix_batch_dims([y, x])


class MatrixSolveWithBroadcastTest(test.TestCase):

  def test_static_dims_broadcast_matrix_has_extra_dims(self):
    # batch_shape = [2]
    matrix = rng.rand(2, 3, 3)
    rhs = rng.rand(3, 7)
    rhs_broadcast = rhs + np.zeros((2, 1, 1))

    result = linear_operator_util.matrix_solve_with_broadcast(matrix, rhs)
    self.assertAllEqual((2, 3, 7), result.shape)
    expected = linalg_ops.matrix_solve(matrix, rhs_broadcast)
    self.assertAllClose(*self.evaluate([expected, result]))

  def test_static_dims_broadcast_rhs_has_extra_dims(self):
    # Since the second arg has extra dims, and the domain dim of the first arg
    # is larger than the number of linear equations, code will "flip" the extra
    # dims of the first arg to the far right, making extra linear equations
    # (then call the matrix function, then flip back).
    # We have verified that this optimization indeed happens.  How? We stepped
    # through with a debugger.
    # batch_shape = [2]
    matrix = rng.rand(3, 3)
    rhs = rng.rand(2, 3, 2)
    matrix_broadcast = matrix + np.zeros((2, 1, 1))

    result = linear_operator_util.matrix_solve_with_broadcast(matrix, rhs)
    self.assertAllEqual((2, 3, 2), result.shape)
    expected = linalg_ops.matrix_solve(matrix_broadcast, rhs)
    self.assertAllClose(*self.evaluate([expected, result]))

  def test_static_dims_broadcast_rhs_has_extra_dims_dynamic(self):
    # Since the second arg has extra dims, and the domain dim of the first arg
    # is larger than the number of linear equations, code will "flip" the extra
    # dims of the first arg to the far right, making extra linear equations
    # (then call the matrix function, then flip back).
    # We have verified that this optimization indeed happens.  How? We stepped
    # through with a debugger.
    # batch_shape = [2]
    matrix = rng.rand(3, 3)
    rhs = rng.rand(2, 3, 2)
    matrix_broadcast = matrix + np.zeros((2, 1, 1))

    matrix_ph = array_ops.placeholder_with_default(matrix, shape=[None, None])
    rhs_ph = array_ops.placeholder_with_default(rhs, shape=[None, None, None])

    result = linear_operator_util.matrix_solve_with_broadcast(matrix_ph, rhs_ph)
    self.assertAllEqual(3, result.shape.ndims)
    expected = linalg_ops.matrix_solve(matrix_broadcast, rhs)
    self.assertAllClose(*self.evaluate([expected, result]))

  def test_static_dims_broadcast_rhs_has_extra_dims_and_adjoint(self):
    # Since the second arg has extra dims, and the domain dim of the first arg
    # is larger than the number of linear equations, code will "flip" the extra
    # dims of the first arg to the far right, making extra linear equations
    # (then call the matrix function, then flip back).
    # We have verified that this optimization indeed happens.  How? We stepped
    # through with a debugger.
    # batch_shape = [2]
    matrix = rng.rand(3, 3)
    rhs = rng.rand(2, 3, 2)
    matrix_broadcast = matrix + np.zeros((2, 1, 1))

    result = linear_operator_util.matrix_solve_with_broadcast(
        matrix, rhs, adjoint=True)
    self.assertAllEqual((2, 3, 2), result.shape)
    expected = linalg_ops.matrix_solve(matrix_broadcast, rhs, adjoint=True)
    self.assertAllClose(*self.evaluate([expected, result]))

  def test_dynamic_dims_broadcast_64bit(self):
    # batch_shape = [2, 2]
    matrix = rng.rand(2, 3, 3)
    rhs = rng.rand(2, 1, 3, 7)
    matrix_broadcast = matrix + np.zeros((2, 2, 1, 1))
    rhs_broadcast = rhs + np.zeros((2, 2, 1, 1))

    matrix_ph = array_ops.placeholder_with_default(matrix, shape=None)
    rhs_ph = array_ops.placeholder_with_default(rhs, shape=None)

    result, expected = self.evaluate([
        linear_operator_util.matrix_solve_with_broadcast(matrix_ph, rhs_ph),
        linalg_ops.matrix_solve(matrix_broadcast, rhs_broadcast)
    ])
    self.assertAllClose(expected, result)


class DomainDimensionStubOperator(object):

  def __init__(self, domain_dimension):
    self._domain_dimension = ops.convert_to_tensor(domain_dimension)

  def domain_dimension_tensor(self):
    return self._domain_dimension


class AssertCompatibleMatrixDimensionsTest(test.TestCase):

  def test_compatible_dimensions_do_not_raise(self):
    x = ops.convert_to_tensor(rng.rand(2, 3, 4))
    operator = DomainDimensionStubOperator(3)
    # Should not raise
    self.evaluate(
        linear_operator_util.assert_compatible_matrix_dimensions(operator, x))

  def test_incompatible_dimensions_raise(self):
    x = ops.convert_to_tensor(rng.rand(2, 4, 4))
    operator = DomainDimensionStubOperator(3)
    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesOpError("Dimensions are not compatible"):
      self.evaluate(
          linear_operator_util.assert_compatible_matrix_dimensions(operator, x))
    # pylint: enable=g-error-prone-assert-raises


class DummyOperatorWithHint(object):

  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


class UseOperatorOrProvidedHintUnlessContradictingTest(test.TestCase,
                                                       parameterized.TestCase):

  @parameterized.named_parameters(
      ("none_none", None, None, None),
      ("none_true", None, True, True),
      ("true_none", True, None, True),
      ("true_true", True, True, True),
      ("none_false", None, False, False),
      ("false_none", False, None, False),
      ("false_false", False, False, False),
  )
  def test_computes_an_or_if_non_contradicting(self, operator_hint_value,
                                               provided_hint_value,
                                               expected_result):
    self.assertEqual(
        expected_result,
        linear_operator_util.use_operator_or_provided_hint_unless_contradicting(
            operator=DummyOperatorWithHint(my_hint=operator_hint_value),
            hint_attr_name="my_hint",
            provided_hint_value=provided_hint_value,
            message="should not be needed here"))

  @parameterized.named_parameters(
      ("true_false", True, False),
      ("false_true", False, True),
  )
  def test_raises_if_contradicting(self, operator_hint_value,
                                   provided_hint_value):
    with self.assertRaisesRegex(ValueError, "my error message"):
      linear_operator_util.use_operator_or_provided_hint_unless_contradicting(
          operator=DummyOperatorWithHint(my_hint=operator_hint_value),
          hint_attr_name="my_hint",
          provided_hint_value=provided_hint_value,
          message="my error message")


class BlockwiseTest(test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("split_dim_1", [3, 3, 4], -1),
      ("split_dim_2", [2, 5], -2),
      )
  def test_blockwise_input(self, op_dimension_values, split_dim):

    op_dimensions = [
        tensor_shape.Dimension(v) for v in op_dimension_values]
    unknown_op_dimensions = [
        tensor_shape.Dimension(None) for _ in op_dimension_values]

    batch_shape = [2, 1]
    arg_dim = 5
    if split_dim == -1:
      blockwise_arrays = [np.zeros(batch_shape + [arg_dim, d])
                          for d in op_dimension_values]
    else:
      blockwise_arrays = [np.zeros(batch_shape + [d, arg_dim])
                          for d in op_dimension_values]

    blockwise_list = [block.tolist() for block in blockwise_arrays]
    blockwise_tensors = [ops.convert_to_tensor(block)
                         for block in blockwise_arrays]
    blockwise_placeholders = [
        array_ops.placeholder_with_default(block, shape=None)
        for block in blockwise_arrays]

    # Iterables of non-nested structures are always interpreted as blockwise.
    # The list of lists is interpreted as blockwise as well, regardless of
    # whether the operator dimensions are known, since the sizes of its elements
    # along `split_dim` are non-identical.
    for op_dims in [op_dimensions, unknown_op_dimensions]:
      for blockwise_inputs in [
          blockwise_arrays, blockwise_list,
          blockwise_tensors, blockwise_placeholders]:
        self.assertTrue(linear_operator_util.arg_is_blockwise(
            op_dims, blockwise_inputs, split_dim))

  def test_non_blockwise_input(self):
    x = np.zeros((2, 3, 4, 6))
    x_tensor = ops.convert_to_tensor(x)
    x_placeholder = array_ops.placeholder_with_default(x, shape=None)
    x_list = x.tolist()

    # For known and matching operator dimensions, interpret all as non-blockwise
    op_dimension_values = [2, 1, 3]
    op_dimensions = [tensor_shape.Dimension(d) for d in op_dimension_values]
    for inputs in [x, x_tensor, x_placeholder, x_list]:
      self.assertFalse(linear_operator_util.arg_is_blockwise(
          op_dimensions, inputs, -1))

    # The input is still interpreted as non-blockwise for unknown operator
    # dimensions (`x_list` has an outermost dimension that does not matcn the
    # number of blocks, and the other inputs are not iterables).
    unknown_op_dimensions = [
        tensor_shape.Dimension(None) for _ in op_dimension_values]
    for inputs in [x, x_tensor, x_placeholder, x_list]:
      self.assertFalse(linear_operator_util.arg_is_blockwise(
          unknown_op_dimensions, inputs, -1))

  def test_ambiguous_input_raises(self):
    x = np.zeros((3, 4, 2)).tolist()
    op_dimensions = [tensor_shape.Dimension(None) for _ in range(3)]

    # Since the leftmost dimension of `x` is equal to the number of blocks, and
    # the operators have unknown dimension, the input is ambiguous.
    with self.assertRaisesRegex(ValueError, "structure is ambiguous"):
      linear_operator_util.arg_is_blockwise(op_dimensions, x, -2)

  def test_mismatched_input_raises(self):
    x = np.zeros((2, 3, 4, 6)).tolist()
    op_dimension_values = [4, 3]
    op_dimensions = [tensor_shape.Dimension(v) for v in op_dimension_values]

    # The dimensions of the two operator-blocks sum to 7. `x` is a
    # two-element list; if interpreted blockwise, its corresponding dimensions
    # sum to 12 (=6*2). If not interpreted blockwise, its corresponding
    # dimension is 6. This is a mismatch.
    with self.assertRaisesRegex(ValueError, "dimension does not match"):
      linear_operator_util.arg_is_blockwise(op_dimensions, x, -1)

if __name__ == "__main__":
  test.main()
