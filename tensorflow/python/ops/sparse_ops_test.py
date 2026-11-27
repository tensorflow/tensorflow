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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
# Need array_grad to register gradient for Identity.
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_sparse_ops
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
      (
          math_ops.asinh,
          [1.0, -1.0, 3.0, -4.0],
          [0.8813736, -0.8813736, 1.8184465, -2.0947125],
      ),
      (
          math_ops.sin,
          [1.0, -1.0, 3.0, -4.0],
          [0.84147096, -0.84147096, 0.14112, 0.7568025],
      ),
      (
          math_ops.asin,
          [1.0, -1.0, 0.4, -0.5],
          [1.5707964, -1.5707964, 0.41151685, -0.5235988],
      ),
      (
          math_ops.tan,
          [1.0, -1.0, 0.4, -0.5],
          [1.5574077, -1.5574077, 0.42279324, -0.5463025],
      ),
      (
          math_ops.atan,
          [0.4, -0.4, 1.0, 0.5],
          [0.3805064, -0.3805064, 0.7853982, 0.4636476],
      ),
      (
          math_ops.atanh,
          [0.4, -0.4, -0.5, 0.5],
          [0.42364895, -0.42364895, -0.54930615, 0.54930615],
      ),
  ])
  def testUnarySparseDispatch(self, op, values, expected):
    st = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [2, 0], [2, 4]],
        values=values,
        dense_shape=[3, 6])
    result = op(st)
    result_value = self.evaluate(result)
    self.assertAllEqual(result_value.indices, st.indices)
    self.assertAllClose(result_value.values, expected)
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

  @parameterized.parameters(
      (False, False, dtypes.complex64),
      (False, True, dtypes.complex64),
      (True, False, dtypes.complex64),
      (True, True, dtypes.complex64),
      (False, False, dtypes.complex128),
      (False, True, dtypes.complex128),
      (True, False, dtypes.complex128),
      (True, True, dtypes.complex128),
  )
  def testDenseSparseTensorMatMulComplex(self, adj_a, adj_b, dtype):
    np.random.seed(7)
    a_np = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(
        dtype.as_numpy_dtype)
    b_np = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(
        dtype.as_numpy_dtype)
    a_tf = constant_op.constant(a_np)
    b_sp = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2], [2, 1]],
        values=constant_op.constant(
            [b_np[0, 0], b_np[1, 2], b_np[2, 1]], dtype=dtype),
        dense_shape=[3, 3])
    b_dense = sparse_ops.sparse_to_dense(b_sp.indices, b_sp.shape, b_sp.values)
    result = sparse_ops.sparse_tensor_dense_matmul(
        a_tf, b_sp, adjoint_a=adj_a, adjoint_b=adj_b)
    expected = math_ops.matmul(a_tf, b_dense, adjoint_a=adj_a, adjoint_b=adj_b)
    self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  @parameterized.parameters(
      (False, False, dtypes.complex64),
      (False, True, dtypes.complex64),
      (True, False, dtypes.complex64),
      (True, True, dtypes.complex64),
  )
  def testDenseSparseTensorMatMulComplexSparseTensorValue(
      self, adj_a, adj_b, dtype):
    np.random.seed(13)
    a_np = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(
        dtype.as_numpy_dtype)
    b_vals = np.array([2 - 1j, 1 + 3j], dtype=dtype.as_numpy_dtype)
    a_tf = constant_op.constant(a_np)
    b_stv = sparse_tensor.SparseTensorValue(
        indices=np.array([[0, 0], [1, 2]]),
        values=b_vals,
        dense_shape=(3, 3))
    b_dense = sparse_ops.sparse_to_dense(b_stv.indices, b_stv.dense_shape,
                                         b_stv.values)
    result = sparse_ops.sparse_tensor_dense_matmul(
        a_tf, b_stv, adjoint_a=adj_a, adjoint_b=adj_b)
    expected = math_ops.matmul(a_tf, b_dense, adjoint_a=adj_a, adjoint_b=adj_b)
    self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testDenseSparseTensorMatMulEmptySparse(self):
    np.random.seed(99)
    a_np = (np.random.rand(3, 3) + 1j * np.random.rand(3, 3)).astype(
        np.complex64)
    a_tf = constant_op.constant(a_np)
    b_sp = sparse_tensor.SparseTensor(
        indices=np.zeros([0, 2], dtype=np.int64),
        values=constant_op.constant([], dtype=dtypes.complex64),
        dense_shape=[3, 3])
    result = sparse_ops.sparse_tensor_dense_matmul(a_tf, b_sp)
    expected = math_ops.matmul(
        a_tf,
        array_ops.zeros([3, 3], dtype=dtypes.complex64))
    self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testBothSparseRaisesTypeError(self):
    sp_a = sparse_tensor.SparseTensor(
        indices=[[0, 0]], values=[1.0 + 0j], dense_shape=[2, 2])
    sp_b = sparse_tensor.SparseTensor(
        indices=[[0, 1]], values=[2.0 + 0j], dense_shape=[2, 2])
    with self.assertRaisesRegex(TypeError, "Cannot multiply two SparseTensors"):
      sparse_ops.sparse_tensor_dense_matmul(sp_a, sp_b)
    sp_b_val = sparse_tensor.SparseTensorValue(
        indices=np.array([[0, 1]]), values=np.array([2.0 + 0j]),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(TypeError, "Cannot multiply two SparseTensors"):
      sparse_ops.sparse_tensor_dense_matmul(sp_a, sp_b_val)

  @parameterized.parameters(
      (False, False, dtypes.complex64),
      (False, True, dtypes.complex64),
      (True, False, dtypes.complex64),
      (True, True, dtypes.complex64),
      (False, False, dtypes.complex128),
      (True, True, dtypes.complex128),
  )
  def testDenseSparseTensorMatMulComplexGradient(self, adj_a, adj_b, dtype):
    np.random.seed(17)
    a_np = (0.1 * np.random.rand(2, 2) + 1j * 0.1 * np.random.rand(2, 2)
            ).astype(dtype.as_numpy_dtype)
    b_vals_np = np.array([0.3 + 0.1j, 0.2 - 0.4j], dtype=dtype.as_numpy_dtype)
    b_indices = np.array([[0, 0], [1, 1]], dtype=np.int64)

    def f(a_in, b_vals_in):
      b_sp = sparse_tensor.SparseTensor(
          indices=b_indices, values=b_vals_in, dense_shape=[2, 2])
      return sparse_ops.sparse_tensor_dense_matmul(
          a_in, b_sp, adjoint_a=adj_a, adjoint_b=adj_b)

    a_tf = constant_op.constant(a_np)
    b_vals_tf = constant_op.constant(b_vals_np)
    theoretical, numerical = gradient_checker.compute_gradient(
        f, [a_tf, b_vals_tf])
    for t, n in zip(theoretical, numerical):
      self.assertAllClose(t, n, rtol=1e-2, atol=1e-2)

  def testSparseTensorDenseMatMulNegativeShapeValueError(self):
    sp = sparse_tensor.SparseTensor(
        indices=[[0, 0]], values=[1.0], dense_shape=[-1, 4]
    )
    b = constant_op.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    with self.assertRaisesRegex(
        ValueError, 'dense_shape must not contain negative values'
    ):
      self.evaluate(sparse_ops.sparse_tensor_dense_matmul(sp, b))

    a = constant_op.constant([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    with self.assertRaisesRegex(
        ValueError, 'dense_shape must not contain negative values'
    ):
      self.evaluate(sparse_ops.sparse_tensor_dense_matmul(a, sp))

  def testMapValues(self):
    # supplying no sparse tensor should result in ValueError
    with self.assertRaises(ValueError):
      sparse_ops.map_values(math_ops.abs, 0.0)

    sp = sparse_ops.from_dense([[0.0, 1.0, 0.0], [-2.0, 1.0, 0.0]])

    # helper function to check equality of sparse tensor
    def assert_sparse_equal(expected, result):
      self.assertAllEqual(expected.values, result.values, msg='Values differ')
      self.assertAllEqual(
          expected.indices, result.indices, msg='Indices differ')
      self.assertAllEqual(
          expected.dense_shape, result.dense_shape, msg='Shapes differ')

    # check for a single sparse argument
    expected = sparse_ops.from_dense([[0.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
    result = sparse_ops.map_values(math_ops.abs, sp)
    assert_sparse_equal(expected, result)

    # check correct passing of keyword argument, and handling of two sparse
    # arguments at the same time
    def mapping(arg1, arg2, kwarg):
      self.assertEqual(kwarg, 'kwarg')
      return arg1 + arg2

    result = sparse_ops.map_values(mapping, sp, sp, kwarg='kwarg')
    expected = sparse_ops.from_dense([[0.0, 2.0, 0.0], [-4.0, 2.0, 0.0]])
    assert_sparse_equal(expected, result)

    # check that index mismatches are correctly detected even if the `value`s
    # have compatible shape
    sp_incomp = sparse_ops.from_dense([[0.0, 1.0, 0.0], [-2.0, 0.0, 1.0]])
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      result = sparse_ops.map_values(mapping, sp, sp_incomp, kwarg='kwarg')
      self.evaluate(result)

    # check that shape mismatches are correctly detected
    sp_incomp = sparse_tensor.SparseTensor(sp.indices, sp.values, (25, 25))
    with self.assertRaises((errors.InvalidArgumentError, ValueError)):
      result = sparse_ops.map_values(mapping, sp, sp_incomp, kwarg='kwarg')
      self.evaluate(result)

  def testConstantStringToSparse(self):
    # Test case for GitHub issue 40633.
    tensor = constant_op.constant(list('ababa'))
    sparse = sparse_ops.from_dense(tensor)
    result = self.evaluate(sparse)
    self.assertAllEqual([[0], [1], [2], [3], [4]], result.indices)
    self.assertAllEqual([b'a', b'b', b'a', b'b', b'a'], result.values)
    self.assertAllEqual([5], result.dense_shape)

  def testSparseTensorToDenseQint(self):
    x = np.asarray([1, 2])
    y = np.asarray([[1, 0, 0], [0, 0, 2]])
    for dtype in [dtypes.qint8, dtypes.qint16, dtypes.quint8, dtypes.quint16]:
      sp = sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 2]],
          values=x.astype(dtype.as_numpy_dtype),
          dense_shape=[2, 3])
      v = self.evaluate(sparse_ops.sparse_tensor_to_dense(sp))
      self.assertAllEqual(
          y.astype(dtype.as_numpy_dtype), v.astype(dtype.as_numpy_dtype))


@test_util.run_all_in_graph_and_eager_modes
class RawOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testSparseFillEmptyRowsGrad(self):
    reverse_index_map = [2, 1]
    grad_values = [0, 1, 2, 3]
    d_values, d_default_value = self.evaluate(
        gen_sparse_ops.SparseFillEmptyRowsGrad(
            reverse_index_map=reverse_index_map, grad_values=grad_values))
    self.assertAllEqual([2, 1], d_values)
    self.assertEqual(3, d_default_value)

  def testSparseFillEmptyRowsGradNegativeIndexMapValue(self):
    reverse_index_map = [2, -1]
    grad_values = [0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r'Elements in reverse index must be in \[0, 4\)'):
      self.evaluate(
          gen_sparse_ops.SparseFillEmptyRowsGrad(
              reverse_index_map=reverse_index_map, grad_values=grad_values))

  def testSparseFillEmptyRowsGradLargeIndexMapValue(self):
    reverse_index_map = [2, 10]
    grad_values = [0, 1, 2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r'Elements in reverse index must be in \[0, 4\)'):
      self.evaluate(
          gen_sparse_ops.SparseFillEmptyRowsGrad(
              reverse_index_map=reverse_index_map, grad_values=grad_values))

  def testSparseFillEmptyRowsGradMatrix(self):
    reverse_index_map = [0, 1]
    grad_values = [[0, 1], [2, 3]]
    # Note: Eager mode and graph mode throw different errors here. Graph mode
    # will fail with a ValueError from the shape checking logic, while Eager
    # will fail with an InvalidArgumentError from the kernel itself.
    if context.executing_eagerly():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  r'grad_values must be a vector'):
        self.evaluate(
            gen_sparse_ops.SparseFillEmptyRowsGrad(
                reverse_index_map=reverse_index_map, grad_values=grad_values))
    else:
      with self.assertRaisesRegex(ValueError,
                                  r'Shape must be rank 1 but is rank 2'):
        self.evaluate(
            gen_sparse_ops.SparseFillEmptyRowsGrad(
                reverse_index_map=reverse_index_map, grad_values=grad_values))

  def testSparseConcatStaticShape(self):
    if context.executing_eagerly():
      self.skipTest('sparse_spaceholder is only available in graph context.')
    input_a = array_ops.sparse_placeholder(dtypes.float32, shape=(2, 1))
    input_b = array_ops.sparse_placeholder(dtypes.float32, shape=(2, 2))

    result = sparse_ops.sparse_concat_v2(axis=1, sp_inputs=[input_a, input_b])
    self.assertEqual(result.shape, [2, 3])


if __name__ == '__main__':
  googletest.main()
