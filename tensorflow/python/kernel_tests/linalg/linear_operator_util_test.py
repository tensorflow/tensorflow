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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test

rng = np.random.RandomState(0)


class AssertZeroImagPartTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_real_tensor_doesnt_raise(self):
    x = ops.convert_to_tensor([0., 2, 3])
    with self.cached_session():
      # Should not raise.
      linear_operator_util.assert_zero_imag_part(x, message="ABC123").run()

  @test_util.run_deprecated_v1
  def test_complex_tensor_with_imag_zero_doesnt_raise(self):
    x = ops.convert_to_tensor([1., 0, 3])
    y = ops.convert_to_tensor([0., 0, 0])
    z = math_ops.complex(x, y)
    with self.cached_session():
      # Should not raise.
      linear_operator_util.assert_zero_imag_part(z, message="ABC123").run()

  def test_complex_tensor_with_nonzero_imag_raises(self):
    x = ops.convert_to_tensor([1., 2, 0])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    with self.cached_session():
      with self.assertRaisesOpError("ABC123"):
        linear_operator_util.assert_zero_imag_part(z, message="ABC123").run()


class AssertNoEntriesWithModulusZeroTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_nonzero_real_tensor_doesnt_raise(self):
    x = ops.convert_to_tensor([1., 2, 3])
    with self.cached_session():
      # Should not raise.
      linear_operator_util.assert_no_entries_with_modulus_zero(
          x, message="ABC123").run()

  @test_util.run_deprecated_v1
  def test_nonzero_complex_tensor_doesnt_raise(self):
    x = ops.convert_to_tensor([1., 0, 3])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    with self.cached_session():
      # Should not raise.
      linear_operator_util.assert_no_entries_with_modulus_zero(
          z, message="ABC123").run()

  def test_zero_real_tensor_raises(self):
    x = ops.convert_to_tensor([1., 0, 3])
    with self.cached_session():
      with self.assertRaisesOpError("ABC123"):
        linear_operator_util.assert_no_entries_with_modulus_zero(
            x, message="ABC123").run()

  def test_zero_complex_tensor_raises(self):
    x = ops.convert_to_tensor([1., 2, 0])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    with self.cached_session():
      with self.assertRaisesOpError("ABC123"):
        linear_operator_util.assert_no_entries_with_modulus_zero(
            z, message="ABC123").run()


class BroadcastMatrixBatchDimsTest(test.TestCase):

  def test_zero_batch_matrices_returned_as_empty_list(self):
    self.assertAllEqual([],
                        linear_operator_util.broadcast_matrix_batch_dims([]))

  def test_one_batch_matrix_returned_after_tensor_conversion(self):
    arr = rng.rand(2, 3, 4)
    tensor, = linear_operator_util.broadcast_matrix_batch_dims([arr])
    self.assertTrue(isinstance(tensor, ops.Tensor))

    with self.cached_session():
      self.assertAllClose(arr, self.evaluate(tensor))

  @test_util.run_deprecated_v1
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

    with self.cached_session() as sess:
      self.assertAllEqual(x_bc_expected.shape, x_bc.get_shape())
      self.assertAllEqual(y_bc_expected.shape, y_bc.get_shape())
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

    with self.cached_session() as sess:
      self.assertAllEqual(x_bc_expected.shape, x_bc.get_shape())
      self.assertAllEqual(y_bc_expected.shape, y_bc.get_shape())
      x_bc_, y_bc_ = self.evaluate([x_bc, y_bc])
      self.assertAllClose(x_bc_expected, x_bc_)
      self.assertAllClose(y_bc_expected, y_bc_)

  @test_util.run_deprecated_v1
  def test_dynamic_dims_broadcast_32bit(self):
    # x.batch_shape = [3, 1, 2]
    # y.batch_shape = [4, 1]
    # broadcast batch shape = [3, 4, 2]
    x = rng.rand(3, 1, 2, 1, 5).astype(np.float32)
    y = rng.rand(4, 1, 3, 7).astype(np.float32)
    batch_of_zeros = np.zeros((3, 4, 2, 1, 1)).astype(np.float32)
    x_bc_expected = x + batch_of_zeros
    y_bc_expected = y + batch_of_zeros

    x_ph = array_ops.placeholder(dtypes.float32)
    y_ph = array_ops.placeholder(dtypes.float32)

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x_ph, y_ph])

    with self.cached_session() as sess:
      x_bc_, y_bc_ = sess.run([x_bc, y_bc], feed_dict={x_ph: x, y_ph: y})
      self.assertAllClose(x_bc_expected, x_bc_)
      self.assertAllClose(y_bc_expected, y_bc_)

  @test_util.run_deprecated_v1
  def test_dynamic_dims_broadcast_32bit_second_arg_higher_rank(self):
    # x.batch_shape =    [1, 2]
    # y.batch_shape = [3, 4, 1]
    # broadcast batch shape = [3, 4, 2]
    x = rng.rand(1, 2, 1, 5).astype(np.float32)
    y = rng.rand(3, 4, 1, 3, 7).astype(np.float32)
    batch_of_zeros = np.zeros((3, 4, 2, 1, 1)).astype(np.float32)
    x_bc_expected = x + batch_of_zeros
    y_bc_expected = y + batch_of_zeros

    x_ph = array_ops.placeholder(dtypes.float32)
    y_ph = array_ops.placeholder(dtypes.float32)

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x_ph, y_ph])

    with self.cached_session() as sess:
      x_bc_, y_bc_ = sess.run([x_bc, y_bc], feed_dict={x_ph: x, y_ph: y})
      self.assertAllClose(x_bc_expected, x_bc_)
      self.assertAllClose(y_bc_expected, y_bc_)

  def test_less_than_two_dims_raises_static(self):
    x = rng.rand(3)
    y = rng.rand(1, 1)

    with self.assertRaisesRegexp(ValueError, "at least two dimensions"):
      linear_operator_util.broadcast_matrix_batch_dims([x, y])

    with self.assertRaisesRegexp(ValueError, "at least two dimensions"):
      linear_operator_util.broadcast_matrix_batch_dims([y, x])


class CholeskySolveWithBroadcastTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_static_dims_broadcast(self):
    # batch_shape = [2]
    chol = rng.rand(3, 3)
    rhs = rng.rand(2, 3, 7)
    chol_broadcast = chol + np.zeros((2, 1, 1))

    with self.cached_session():
      result = linear_operator_util.cholesky_solve_with_broadcast(chol, rhs)
      self.assertAllEqual((2, 3, 7), result.get_shape())
      expected = linalg_ops.cholesky_solve(chol_broadcast, rhs)
      self.assertAllClose(expected.eval(), self.evaluate(result))

  @test_util.run_deprecated_v1
  def test_dynamic_dims_broadcast_64bit(self):
    # batch_shape = [2, 2]
    chol = rng.rand(2, 3, 3)
    rhs = rng.rand(2, 1, 3, 7)
    chol_broadcast = chol + np.zeros((2, 2, 1, 1))
    rhs_broadcast = rhs + np.zeros((2, 2, 1, 1))

    chol_ph = array_ops.placeholder(dtypes.float64)
    rhs_ph = array_ops.placeholder(dtypes.float64)

    with self.cached_session() as sess:
      result, expected = sess.run(
          [
              linear_operator_util.cholesky_solve_with_broadcast(
                  chol_ph, rhs_ph),
              linalg_ops.cholesky_solve(chol_broadcast, rhs_broadcast)
          ],
          feed_dict={
              chol_ph: chol,
              rhs_ph: rhs,
          })
      self.assertAllClose(expected, result)


class MatmulWithBroadcastTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_static_dims_broadcast_x_has_extra_dims(self):
    # batch_shape = [2]
    # for each batch member, we have a 1x3 matrix times a 3x7 matrix ==> 1x7
    x = rng.rand(2, 1, 3)
    y = rng.rand(3, 7)
    y_broadcast = y + np.zeros((2, 1, 1))

    with self.cached_session():
      result = linear_operator_util.matmul_with_broadcast(x, y)
      self.assertAllEqual((2, 1, 7), result.get_shape())
      expected = math_ops.matmul(x, y_broadcast)
      self.assertAllClose(expected.eval(), self.evaluate(result))

  @test_util.run_deprecated_v1
  def test_static_dims_broadcast_y_has_extra_dims(self):
    # Since the second arg has extra dims, and the domain dim of the first arg
    # is larger than the number of linear equations, code will "flip" the extra
    # dims of the first arg to the far right, making extra linear equations
    # (then call the matrix function, then flip back).
    # We have verified that this optimization indeed happens.  How? We stepped
    # through with a debugger.
    x = rng.rand(5, 7)
    y = rng.rand(2, 3, 7, 5)
    x_broadcast = x + np.zeros((2, 3, 5, 7))

    with self.cached_session():
      result = linear_operator_util.matmul_with_broadcast(x, y)
      self.assertAllEqual((2, 3, 5, 5), result.get_shape())
      expected = math_ops.matmul(x_broadcast, y)
      self.assertAllClose(expected.eval(), self.evaluate(result))

  @test_util.run_deprecated_v1
  def test_static_dims_broadcast_y_has_extra_dims_transpose_a_and_b(self):
    # Since the second arg has extra dims, and the domain dim of the first arg
    # is larger than the number of linear equations, code will "flip" the extra
    # dims of the first arg to the far right, making extra linear equations
    # (then call the matrix function, then flip back).
    # We have verified that this optimization indeed happens.  How? We stepped
    # through with a debugger.
    x = rng.rand(1, 7, 5)
    y = rng.rand(2, 3, 1, 7)
    x_broadcast = x + np.zeros((2, 3, 1, 1))

    with self.cached_session():
      result = linear_operator_util.matmul_with_broadcast(
          x, y, transpose_a=True, transpose_b=True)
      self.assertAllEqual((2, 3, 5, 1), result.get_shape())
      expected = math_ops.matmul(
          x_broadcast, y, transpose_a=True, transpose_b=True)
      self.assertAllClose(expected.eval(), self.evaluate(result))

  @test_util.run_deprecated_v1
  def test_static_dims_broadcast_y_has_extra_dims_transpose_dynamic(self):
    # Since the second arg has extra dims, and the domain dim of the first arg
    # is larger than the number of linear equations, code will "flip" the extra
    # dims of the first arg to the far right, making extra linear equations
    # (then call the matrix function, then flip back).
    # We have verified that this optimization indeed happens.  How? We stepped
    # through with a debugger.
    x = rng.rand(1, 7, 5)
    y = rng.rand(2, 3, 1, 7)
    x_broadcast = x + np.zeros((2, 3, 1, 1))

    x_ph = array_ops.placeholder(dtypes.float64, [None, None, None])
    y_ph = array_ops.placeholder(dtypes.float64, [None, None, None, None])

    with self.cached_session():
      result = linear_operator_util.matmul_with_broadcast(
          x_ph, y_ph, transpose_a=True, transpose_b=True)
      self.assertAllEqual(4, result.shape.ndims)
      expected = math_ops.matmul(
          x_broadcast, y, transpose_a=True, transpose_b=True)
      self.assertAllClose(expected.eval(),
                          result.eval(feed_dict={
                              x_ph: x,
                              y_ph: y
                          }))

  @test_util.run_deprecated_v1
  def test_dynamic_dims_broadcast_64bit(self):
    # batch_shape = [2]
    # for each batch member, we have a 1x3 matrix times a 3x7 matrix ==> 1x7
    x = rng.rand(2, 1, 3)
    y = rng.rand(3, 7)
    y_broadcast = y + np.zeros((2, 1, 1))

    x_ph = array_ops.placeholder(dtypes.float64)
    y_ph = array_ops.placeholder(dtypes.float64)

    with self.cached_session() as sess:
      result, expected = sess.run(
          [
              linear_operator_util.matmul_with_broadcast(x_ph, y_ph),
              math_ops.matmul(x, y_broadcast)
          ],
          feed_dict={
              x_ph: x,
              y_ph: y
          })
      self.assertAllClose(expected, result)


class MatrixSolveWithBroadcastTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_static_dims_broadcast_matrix_has_extra_dims(self):
    # batch_shape = [2]
    matrix = rng.rand(2, 3, 3)
    rhs = rng.rand(3, 7)
    rhs_broadcast = rhs + np.zeros((2, 1, 1))

    with self.cached_session():
      result = linear_operator_util.matrix_solve_with_broadcast(
          matrix, rhs)
      self.assertAllEqual((2, 3, 7), result.get_shape())
      expected = linalg_ops.matrix_solve(matrix, rhs_broadcast)
      self.assertAllClose(expected.eval(), self.evaluate(result))

  @test_util.run_deprecated_v1
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

    with self.cached_session():
      result = linear_operator_util.matrix_solve_with_broadcast(matrix, rhs)
      self.assertAllEqual((2, 3, 2), result.get_shape())
      expected = linalg_ops.matrix_solve(matrix_broadcast, rhs)
      self.assertAllClose(expected.eval(), self.evaluate(result))

  @test_util.run_deprecated_v1
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

    matrix_ph = array_ops.placeholder(dtypes.float64, shape=[None, None])
    rhs_ph = array_ops.placeholder(dtypes.float64, shape=[None, None, None])

    with self.cached_session():
      result = linear_operator_util.matrix_solve_with_broadcast(matrix_ph,
                                                                rhs_ph)
      self.assertAllEqual(3, result.shape.ndims)
      expected = linalg_ops.matrix_solve(matrix_broadcast, rhs)
      self.assertAllClose(
          self.evaluate(expected),
          result.eval(feed_dict={
              matrix_ph: matrix,
              rhs_ph: rhs
          }))

  @test_util.run_deprecated_v1
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

    with self.cached_session():
      result = linear_operator_util.matrix_solve_with_broadcast(
          matrix, rhs, adjoint=True)
      self.assertAllEqual((2, 3, 2), result.get_shape())
      expected = linalg_ops.matrix_solve(matrix_broadcast, rhs, adjoint=True)
      self.assertAllClose(expected.eval(), self.evaluate(result))

  @test_util.run_deprecated_v1
  def test_dynamic_dims_broadcast_64bit(self):
    # batch_shape = [2, 2]
    matrix = rng.rand(2, 3, 3)
    rhs = rng.rand(2, 1, 3, 7)
    matrix_broadcast = matrix + np.zeros((2, 2, 1, 1))
    rhs_broadcast = rhs + np.zeros((2, 2, 1, 1))

    matrix_ph = array_ops.placeholder(dtypes.float64)
    rhs_ph = array_ops.placeholder(dtypes.float64)

    with self.cached_session() as sess:
      result, expected = sess.run(
          [
              linear_operator_util.matrix_solve_with_broadcast(
                  matrix_ph, rhs_ph),
              linalg_ops.matrix_solve(matrix_broadcast, rhs_broadcast)
          ],
          feed_dict={
              matrix_ph: matrix,
              rhs_ph: rhs,
          })
      self.assertAllClose(expected, result)


class MatrixTriangularSolveWithBroadcastTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_static_dims_broadcast_matrix_has_extra_dims(self):
    # batch_shape = [2]
    matrix = rng.rand(2, 3, 3)
    rhs = rng.rand(3, 7)
    rhs_broadcast = rhs + np.zeros((2, 1, 1))

    with self.cached_session():
      result = linear_operator_util.matrix_triangular_solve_with_broadcast(
          matrix, rhs)
      self.assertAllEqual((2, 3, 7), result.get_shape())
      expected = linalg_ops.matrix_triangular_solve(matrix, rhs_broadcast)
      self.assertAllClose(expected.eval(), self.evaluate(result))

  @test_util.run_deprecated_v1
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

    with self.cached_session():
      result = linear_operator_util.matrix_triangular_solve_with_broadcast(
          matrix, rhs)
      self.assertAllEqual((2, 3, 2), result.get_shape())
      expected = linalg_ops.matrix_triangular_solve(matrix_broadcast, rhs)
      self.assertAllClose(expected.eval(), self.evaluate(result))

  @test_util.run_deprecated_v1
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

    with self.cached_session():
      result = linear_operator_util.matrix_triangular_solve_with_broadcast(
          matrix, rhs, adjoint=True)
      self.assertAllEqual((2, 3, 2), result.get_shape())
      expected = linalg_ops.matrix_triangular_solve(
          matrix_broadcast, rhs, adjoint=True)
      self.assertAllClose(expected.eval(), self.evaluate(result))

  @test_util.run_deprecated_v1
  def test_dynamic_dims_broadcast_64bit(self):
    # batch_shape = [2]
    matrix = rng.rand(2, 3, 3)
    rhs = rng.rand(3, 7)
    rhs_broadcast = rhs + np.zeros((2, 1, 1))

    matrix_ph = array_ops.placeholder(dtypes.float64)
    rhs_ph = array_ops.placeholder(dtypes.float64)

    with self.cached_session() as sess:
      result, expected = sess.run(
          [
              linear_operator_util.matrix_triangular_solve_with_broadcast(
                  matrix_ph, rhs_ph),
              linalg_ops.matrix_triangular_solve(matrix, rhs_broadcast)
          ],
          feed_dict={
              matrix_ph: matrix,
              rhs_ph: rhs,
          })
      self.assertAllClose(expected, result)


class DomainDimensionStubOperator(object):

  def __init__(self, domain_dimension):
    self._domain_dimension = ops.convert_to_tensor(domain_dimension)

  def domain_dimension_tensor(self):
    return self._domain_dimension


class AssertCompatibleMatrixDimensionsTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_compatible_dimensions_do_not_raise(self):
    with self.cached_session():
      x = ops.convert_to_tensor(rng.rand(2, 3, 4))
      operator = DomainDimensionStubOperator(3)
      # Should not raise
      linear_operator_util.assert_compatible_matrix_dimensions(
          operator, x).run()  # pyformat: disable

  def test_incompatible_dimensions_raise(self):
    with self.cached_session():
      x = ops.convert_to_tensor(rng.rand(2, 4, 4))
      operator = DomainDimensionStubOperator(3)
      with self.assertRaisesOpError("Incompatible matrix dimensions"):
        linear_operator_util.assert_compatible_matrix_dimensions(
            operator, x).run()  # pyformat: disable


if __name__ == "__main__":
  test.main()
