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
"""Tests for sparse ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
# Need array_grad to register gradient for Identity.
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2 as gradient_checker
from tensorflow.python.ops import math_ops
# Need sparse_grad to register gradient for SparseToDense.
from tensorflow.python.ops import sparse_grad  # pylint: disable=unused-import
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class SparseOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testSparseEye(self):
    def test_one(n, m, as_tensors):
      expected = np.eye(n, m)
      if as_tensors:
        m = constant_op.constant(m)
        n = constant_op.constant(n)
      s = sparse_ops.sparse_eye(n, m)
      d = sparse_ops.sparse_to_dense(s.indices, s.dense_shape, s.values)
      self.assertAllEqual(self.evaluate(d), expected)

    for n in range(2, 10, 2):
      for m in range(2, 10, 2):
        # Test with n and m as both constants and tensors.
        test_one(n, m, True)
        test_one(n, m, False)

  def testDenseFromConstantToSparse(self):
    expected_constant = np.reshape(np.arange(24, dtype=np.int64), (3, 4, 2))
    tensor = constant_op.constant(expected_constant)
    sparse = sparse_ops.from_dense(tensor)
    dense = sparse_ops.sparse_to_dense(sparse.indices, sparse.dense_shape,
                                       sparse.values)
    constant = self.evaluate(dense)
    self.assertAllEqual(expected_constant, constant)

  def testTransposePreservesShape(self):
    with ops.Graph().as_default():
      t = sparse_tensor.SparseTensor(indices=[[0, 0]],
                                     values=[0.],
                                     dense_shape=[3, 4])
      self.assertTrue(t.shape.is_fully_defined)
      transposed = sparse_ops.sparse_transpose(t)
      self.assertAllEqual(transposed.shape, [4, 3])

  def testSparseExpandDims(self):
    for rank in range(1, 4):
      # Create a dummy input. When rank=3, shape=[2, 4, 6].
      shape = np.arange(1, rank + 1) * 2
      before = np.arange(np.prod(shape)).reshape(shape)

      # Make entries sparse.
      before *= np.random.binomial(1, .2, before.shape)
      dense_shape = before.shape
      indices = np.array(np.where(before)).T
      values = before[before != 0]

      # Try every possible valid value of axis.
      for axis in range(-rank - 1, rank):
        expected_after = np.expand_dims(before, axis)

        for axis_as_tensor in [False, True]:
          dense_shape_t = constant_op.constant(dense_shape, dtype=dtypes.int64)
          indices_t = constant_op.constant(indices)
          values_t = constant_op.constant(values)
          before_t = sparse_tensor.SparseTensor(
              indices=indices_t, values=values_t, dense_shape=dense_shape_t)

          if axis_as_tensor:
            axis = constant_op.constant(axis)

          s = sparse_ops.sparse_expand_dims(before_t, axis)
          d = sparse_ops.sparse_to_dense(s.indices, s.dense_shape, s.values)
          self.assertAllEqual(self.evaluate(d), expected_after)

  @parameterized.parameters([
      (math_ops.abs, [1.0, -1.0, 3.0, -4.0], [1.0, 1.0, 3.0, 4.0]),
      (math_ops.negative, [1.0, -1.0, 3.0, -4.0], [-1.0, 1.0, -3.0, 4.0]),
      (math_ops.sign, [3.0, -2.0, 0.0, -4.0], [1.0, -1.0, 0.0, -1.0]),
      (math_ops.square, [1.0, -1.0, 3.0, -4.0], [1.0, 1.0, 9.0, 16.0]),
  ])
  def testUnarySparseDispatch(self, op, values, expected):
    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [2, 0], [2, 4]],
        values=values,
        dense_shape=[3, 6])
    result = op(st)
    result_value = self.evaluate(result)
    self.assertAllEqual(result_value.indices, st.indices)
    self.assertAllEqual(result_value.values, expected)
    self.assertAllEqual(result_value.dense_shape, st.dense_shape)

  def testSparseToDenseGradient(self):

    def f(sparse_values, default_value):
      st = sparse_tensor.SparseTensor(
          indices=[[0, 3, 6], [1, 4, 7], [2, 5, 8]],
          values=sparse_values,
          dense_shape=[3, 6, 9])
      return sparse_ops.sparse_tensor_to_dense(st, default_value)

    grads = gradient_checker.compute_gradient(
        f, [constant_op.constant([1.0, 2.0, 3.0]),
            constant_op.constant(0.0)])
    epsilon = 1e-4
    self.assertLess(gradient_checker.max_error(*grads), epsilon)

  def testSparseTensorToDenseString(self):
    sp = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=['a', 'b'], dense_shape=[2, 3])
    dense = sparse_ops.sparse_tensor_to_dense(sp)
    expected_dense = [[b'a', b'', b''], [b'', b'', b'b']]
    result_dense = self.evaluate(dense)
    self.assertAllEqual(expected_dense, result_dense)

  def testDenseSparseTensorMatMul(self):

    np.random.seed(42)
    dense_numpy_array = np.random.rand(3, 3)
    independent_dense_tf = constant_op.constant(
        dense_numpy_array, dtype='float32')

    sp = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[4., 8.], dense_shape=[3, 3])
    dense_of_sparse = sparse_ops.sparse_to_dense(sp.indices, sp.shape,
                                                 sp.values)

    result = sparse_ops.sparse_tensor_dense_matmul(
        independent_dense_tf, sp, adjoint_a=False, adjoint_b=False)
    expected = math_ops.matmul(independent_dense_tf, dense_of_sparse)
    self.assertAllEqual(expected, result)

    result = sparse_ops.sparse_tensor_dense_matmul(
        independent_dense_tf, sp, adjoint_a=False, adjoint_b=True)
    expected = math_ops.matmul(independent_dense_tf,
                               array_ops.transpose(dense_of_sparse))
    self.assertAllEqual(expected, result)

    result = sparse_ops.sparse_tensor_dense_matmul(
        independent_dense_tf, sp, adjoint_a=True, adjoint_b=False)
    expected = math_ops.matmul(
        array_ops.transpose(independent_dense_tf), dense_of_sparse)
    self.assertAllEqual(expected, result)

    result = sparse_ops.sparse_tensor_dense_matmul(
        independent_dense_tf, sp, adjoint_a=True, adjoint_b=True)
    expected = math_ops.matmul(
        array_ops.transpose(independent_dense_tf),
        array_ops.transpose(dense_of_sparse))
    self.assertAllEqual(expected, result)


if __name__ == '__main__':
  googletest.main()
