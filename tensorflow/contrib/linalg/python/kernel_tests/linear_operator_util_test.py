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

from tensorflow.contrib import linalg as linalg_lib
from tensorflow.contrib.linalg.python.ops import linear_operator_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

linalg = linalg_lib
random_seed.set_random_seed(23)
rng = np.random.RandomState(0)


class AssertZeroImagPartTest(test.TestCase):

  def test_real_tensor_doesnt_raise(self):
    x = ops.convert_to_tensor([0., 2, 3])
    with self.test_session():
      # Should not raise.
      linear_operator_util.assert_zero_imag_part(x, message="ABC123").run()

  def test_complex_tensor_with_imag_zero_doesnt_raise(self):
    x = ops.convert_to_tensor([1., 0, 3])
    y = ops.convert_to_tensor([0., 0, 0])
    z = math_ops.complex(x, y)
    with self.test_session():
      # Should not raise.
      linear_operator_util.assert_zero_imag_part(z, message="ABC123").run()

  def test_complex_tensor_with_nonzero_imag_raises(self):
    x = ops.convert_to_tensor([1., 2, 0])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    with self.test_session():
      with self.assertRaisesOpError("ABC123"):
        linear_operator_util.assert_zero_imag_part(z, message="ABC123").run()


class AssertNoEntriesWithModulusZeroTest(test.TestCase):

  def test_nonzero_real_tensor_doesnt_raise(self):
    x = ops.convert_to_tensor([1., 2, 3])
    with self.test_session():
      # Should not raise.
      linear_operator_util.assert_no_entries_with_modulus_zero(
          x, message="ABC123").run()

  def test_nonzero_complex_tensor_doesnt_raise(self):
    x = ops.convert_to_tensor([1., 0, 3])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    with self.test_session():
      # Should not raise.
      linear_operator_util.assert_no_entries_with_modulus_zero(
          z, message="ABC123").run()

  def test_zero_real_tensor_raises(self):
    x = ops.convert_to_tensor([1., 0, 3])
    with self.test_session():
      with self.assertRaisesOpError("ABC123"):
        linear_operator_util.assert_no_entries_with_modulus_zero(
            x, message="ABC123").run()

  def test_zero_complex_tensor_raises(self):
    x = ops.convert_to_tensor([1., 2, 0])
    y = ops.convert_to_tensor([1., 2, 0])
    z = math_ops.complex(x, y)
    with self.test_session():
      with self.assertRaisesOpError("ABC123"):
        linear_operator_util.assert_no_entries_with_modulus_zero(
            z, message="ABC123").run()


class BroadcastMatrixBatchDimsTest(test.TestCase):

  def test_zero_batch_matrices_returned_as_empty_list(self):
    self.assertAllEqual(
        [], linear_operator_util.broadcast_matrix_batch_dims([]))

  def test_one_batch_matrix_returned_after_tensor_conversion(self):
    arr = rng.rand(2, 3, 4)
    tensor, = linear_operator_util.broadcast_matrix_batch_dims([arr])
    self.assertTrue(isinstance(tensor, ops.Tensor))

    with self.test_session():
      self.assertAllClose(arr, tensor.eval())

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

    with self.test_session() as sess:
      self.assertAllEqual(x_bc_expected.shape, x_bc.get_shape())
      self.assertAllEqual(y_bc_expected.shape, y_bc.get_shape())
      x_bc_, y_bc_ = sess.run([x_bc, y_bc])
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

    with self.test_session() as sess:
      self.assertAllEqual(x_bc_expected.shape, x_bc.get_shape())
      self.assertAllEqual(y_bc_expected.shape, y_bc.get_shape())
      x_bc_, y_bc_ = sess.run([x_bc, y_bc])
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

    x_ph = array_ops.placeholder(dtypes.float32)
    y_ph = array_ops.placeholder(dtypes.float32)

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x_ph, y_ph])

    with self.test_session() as sess:
      x_bc_, y_bc_ = sess.run([x_bc, y_bc], feed_dict={x_ph: x, y_ph: y})
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

    x_ph = array_ops.placeholder(dtypes.float32)
    y_ph = array_ops.placeholder(dtypes.float32)

    x_bc, y_bc = linear_operator_util.broadcast_matrix_batch_dims([x_ph, y_ph])

    with self.test_session() as sess:
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


class MatmulWithBroadcastTest(test.TestCase):

  def test_static_dims_broadcast(self):
    # batch_shape = [2]
    # for each batch member, we have a 1x3 matrix times a 3x7 matrix ==> 1x7
    x = rng.rand(2, 1, 3)
    y = rng.rand(3, 7)
    y_broadcast = y + np.zeros((2, 1, 1))

    with self.test_session():
      result = linear_operator_util.matmul_with_broadcast(x, y)
      self.assertAllEqual((2, 1, 7), result.get_shape())
      expected = math_ops.matmul(x, y_broadcast)
      self.assertAllEqual(expected.eval(), result.eval())

  def test_dynamic_dims_broadcast_32bit(self):
    # batch_shape = [2]
    # for each batch member, we have a 1x3 matrix times a 3x7 matrix ==> 1x7
    x = rng.rand(2, 1, 3)
    y = rng.rand(3, 7)
    y_broadcast = y + np.zeros((2, 1, 1))

    x_ph = array_ops.placeholder(dtypes.float64)
    y_ph = array_ops.placeholder(dtypes.float64)

    with self.test_session() as sess:
      result, expected = sess.run(
          [linear_operator_util.matmul_with_broadcast(x_ph, y_ph),
           math_ops.matmul(x, y_broadcast)],
          feed_dict={x_ph: x, y_ph: y})
      self.assertAllEqual(expected, result)


class DomainDimensionStubOperator(object):

  def __init__(self, domain_dimension):
    self._domain_dimension = ops.convert_to_tensor(domain_dimension)

  def domain_dimension_tensor(self):
    return self._domain_dimension


class AssertCompatibleMatrixDimensionsTest(test.TestCase):

  def test_compatible_dimensions_do_not_raise(self):
    with self.test_session():
      x = ops.convert_to_tensor(rng.rand(2, 3, 4))
      operator = DomainDimensionStubOperator(3)
      # Should not raise
      linear_operator_util.assert_compatible_matrix_dimensions(
          operator, x).run()

  def test_incompatible_dimensions_raise(self):
    with self.test_session():
      x = ops.convert_to_tensor(rng.rand(2, 4, 4))
      operator = DomainDimensionStubOperator(3)
      with self.assertRaisesOpError("Incompatible matrix dimensions"):
        linear_operator_util.assert_compatible_matrix_dimensions(
            operator, x).run()


if __name__ == "__main__":
  test.main()
