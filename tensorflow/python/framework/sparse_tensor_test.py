# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tensorflow.python.framework.sparse_tensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import googletest


class SparseTensorTest(test_util.TensorFlowTestCase):

  def testPythonConstruction(self):
    indices = [[1, 2], [2, 0], [3, 4]]
    values = [b"a", b"b", b"c"]
    shape = [4, 5]
    sp_value = sparse_tensor.SparseTensorValue(indices, values, shape)
    for sp in [
        sparse_tensor.SparseTensor(indices, values, shape),
        sparse_tensor.SparseTensor.from_value(sp_value),
        sparse_tensor.SparseTensor.from_value(
            sparse_tensor.SparseTensor(indices, values, shape))]:
      self.assertEqual(sp.indices.dtype, dtypes.int64)
      self.assertEqual(sp.values.dtype, dtypes.string)
      self.assertEqual(sp.dense_shape.dtype, dtypes.int64)
      self.assertEqual(sp.get_shape(), (4, 5))

      with self.cached_session() as sess:
        value = self.evaluate(sp)
        self.assertAllEqual(indices, value.indices)
        self.assertAllEqual(values, value.values)
        self.assertAllEqual(shape, value.dense_shape)
        sess_run_value = self.evaluate(sp)
        self.assertAllEqual(sess_run_value.indices, value.indices)
        self.assertAllEqual(sess_run_value.values, value.values)
        self.assertAllEqual(sess_run_value.dense_shape, value.dense_shape)

  def testShape(self):

    @def_function.function
    def test_fn(tensor):
      tensor = sparse_ops.sparse_transpose(tensor)
      self.assertEqual(tensor.shape.rank, 2)
      return tensor

    tensor = sparse_tensor.SparseTensor(
        indices=[[0, 0], [1, 2]], values=[1., 2], dense_shape=[3, 4])
    test_fn(tensor)

  def testIsSparse(self):
    self.assertFalse(sparse_tensor.is_sparse(3))
    self.assertFalse(sparse_tensor.is_sparse("foo"))
    self.assertFalse(sparse_tensor.is_sparse(np.array(3)))
    self.assertTrue(
        sparse_tensor.is_sparse(sparse_tensor.SparseTensor([[0]], [0], [1])))
    self.assertTrue(
        sparse_tensor.is_sparse(
            sparse_tensor.SparseTensorValue([[0]], [0], [1])))

  def testConsumers(self):
    with context.graph_mode():
      sp = sparse_tensor.SparseTensor([[0, 0], [1, 2]], [1.0, 3.0], [3, 4])
      w = ops.convert_to_tensor(np.ones([4, 1], np.float32))
      out = sparse_ops.sparse_tensor_dense_matmul(sp, w)
      self.assertEqual(len(sp.consumers()), 1)
      self.assertEqual(sp.consumers()[0], out.op)

      dense = sparse_ops.sparse_tensor_to_dense(sp)
      self.assertEqual(len(sp.consumers()), 2)
      self.assertIn(dense.op, sp.consumers())
      self.assertIn(out.op, sp.consumers())


class ConvertToTensorOrSparseTensorTest(test_util.TensorFlowTestCase):

  def test_convert_dense(self):
    with self.cached_session():
      value = [42, 43]
      from_value = sparse_tensor.convert_to_tensor_or_sparse_tensor(
          value)
      self.assertAllEqual(value, self.evaluate(from_value))

  @test_util.run_deprecated_v1
  def test_convert_sparse(self):
    with self.cached_session():
      indices = [[0, 1], [1, 0]]
      values = [42, 43]
      shape = [2, 2]
      sparse_tensor_value = sparse_tensor.SparseTensorValue(
          indices, values, shape)
      st = sparse_tensor.SparseTensor.from_value(sparse_tensor_value)
      from_value = sparse_tensor.convert_to_tensor_or_sparse_tensor(
          sparse_tensor_value).eval()
      from_tensor = sparse_tensor.convert_to_tensor_or_sparse_tensor(st).eval()
      for convertee in [from_value, from_tensor]:
        self.assertAllEqual(sparse_tensor_value.indices, convertee.indices)
        self.assertAllEqual(sparse_tensor_value.values, convertee.values)
        self.assertAllEqual(
            sparse_tensor_value.dense_shape, convertee.dense_shape)


class SparseTensorShapeTest(test_util.TensorFlowTestCase):

  def test_simple(self):
    indices = [[0, 2]]
    values = [1]
    dense_shape = [5, 5]
    sp = sparse_tensor.SparseTensor(indices, values, dense_shape)

    self.assertIsInstance(sp.shape, tensor_shape.TensorShape)
    self.assertIsInstance(sp.dense_shape, ops.Tensor)
    self.assertEqual(sp.shape.as_list(), [5, 5])

  def test_unknown_shape(self):

    @def_function.function
    def my_func(dense_shape):
      indices = [[0, 2]]
      values = [1]
      sp = sparse_tensor.SparseTensor(indices, values, dense_shape)
      self.assertEqual(sp.shape.as_list(), [None, None])
      return sp

    my_func.get_concrete_function(
        dense_shape=tensor_spec.TensorSpec(
            dtype=dtypes.int64, shape=[2,]))

  def test_partial_shape(self):

    @def_function.function
    def my_func(x):
      indices = [[0, 2]]
      values = [1]
      y = ops.convert_to_tensor(3, dtype=dtypes.int64)
      dense_shape = [x, y]
      sp = sparse_tensor.SparseTensor(indices, values, dense_shape)
      self.assertEqual(sp.shape.as_list(), [None, 3])
      return sp

    my_func.get_concrete_function(
        x=tensor_spec.TensorSpec(dtype=dtypes.int64, shape=[]))

  def test_neg_shape(self):
    indices = [[0, 2]]
    values = [1]
    dense_shape = [-1, 5]
    sp = sparse_tensor.SparseTensor(indices, values, dense_shape)
    self.assertEqual(sp.shape.as_list(), [None, 5])

  def test_unknown_tensor_shape(self):

    @def_function.function
    def my_func(x):
      indices = [[0, 0]]
      values = [1]
      dense_shape = array_ops.shape(x)
      dense_shape = math_ops.cast(dense_shape, dtypes.int64)

      sp = sparse_tensor.SparseTensor(indices, values, dense_shape)
      self.assertEqual(sp.shape.as_list(), [None, None])
      return sp

    my_func.get_concrete_function(
        x=tensor_spec.TensorSpec(dtype=dtypes.int64, shape=[None, None]))

  def test_unknown_rank(self):

    @def_function.function
    def my_func(dense_shape):
      indices = [[0, 0]]
      values = [1]
      sp = sparse_tensor.SparseTensor(indices, values, dense_shape)
      self.assertEqual(sp.shape.rank, None)
      return sp

    my_func.get_concrete_function(
        dense_shape=tensor_spec.TensorSpec(dtype=dtypes.int64, shape=[None]))


@test_util.run_all_in_graph_and_eager_modes
class SparseTensorSpecTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  def assertAllTensorsEqual(self, list1, list2):
    self.assertLen(list1, len(list2))
    for (t1, t2) in zip(list1, list2):
      self.assertAllEqual(t1, t2)

  def testConstruction(self):
    spec1 = sparse_tensor.SparseTensorSpec()
    self.assertEqual(spec1.shape.rank, None)
    self.assertEqual(spec1.dtype, dtypes.float32)

    spec2 = sparse_tensor.SparseTensorSpec([None, None], dtypes.string)
    self.assertEqual(spec2.shape.as_list(), [None, None])
    self.assertEqual(spec2.dtype, dtypes.string)

  def testValueType(self):
    spec1 = sparse_tensor.SparseTensorSpec()
    self.assertEqual(spec1.value_type, sparse_tensor.SparseTensor)

  @parameterized.parameters([
      (sparse_tensor.SparseTensorSpec(),
       (tensor_shape.TensorShape(None), dtypes.float32)),
      (sparse_tensor.SparseTensorSpec(shape=[5, None, None]),
       (tensor_shape.TensorShape([5, None, None]), dtypes.float32)),
      (sparse_tensor.SparseTensorSpec(dtype=dtypes.int32),
       (tensor_shape.TensorShape(None), dtypes.int32)),
  ])  # pyformat: disable
  def testSerialize(self, st_spec, expected):
    serialization = st_spec._serialize()
    # TensorShape has an unconventional definition of equality, so we can't use
    # assertEqual directly here.  But repr() is deterministic and lossless for
    # the expected values, so we can use that instead.
    self.assertEqual(repr(serialization), repr(expected))

  @parameterized.parameters([
      (sparse_tensor.SparseTensorSpec(dtype=dtypes.string), [
          tensor_spec.TensorSpec([None, None], dtypes.int64),
          tensor_spec.TensorSpec([None], dtypes.string),
          tensor_spec.TensorSpec([None], dtypes.int64)
      ]),
      (sparse_tensor.SparseTensorSpec(shape=[5, None, None]), [
          tensor_spec.TensorSpec([None, 3], dtypes.int64),
          tensor_spec.TensorSpec([None], dtypes.float32),
          tensor_spec.TensorSpec([3], dtypes.int64)
      ]),
  ])
  def testComponentSpecs(self, st_spec, expected):
    self.assertEqual(st_spec._component_specs, expected)

  @parameterized.parameters([
      {
          "st_spec": sparse_tensor.SparseTensorSpec(),
          "indices": [[0, 1], [10, 8]],
          "values": [3.0, 5.0],
          "dense_shape": [100, 100]
      },
      {
          "st_spec": sparse_tensor.SparseTensorSpec([100, None, None]),
          "indices": [[0, 1, 3], [10, 8, 2]],
          "values": [3.0, 5.0],
          "dense_shape": [100, 20, 20]
      },
  ])
  def testToFromComponents(self, st_spec, indices, values, dense_shape):
    st = sparse_tensor.SparseTensor(indices, values, dense_shape)
    actual_components = st_spec._to_components(st)
    self.assertAllTensorsEqual(actual_components,
                               [indices, values, dense_shape])
    st_reconstructed = st_spec._from_components(actual_components)
    self.assertAllEqual(st.indices, st_reconstructed.indices)
    self.assertAllEqual(st.values, st_reconstructed.values)
    self.assertAllEqual(st.dense_shape, st_reconstructed.dense_shape)

  @test_util.run_v1_only("SparseTensorValue is deprecated in v2")
  def testFromNumpyComponents(self):
    indices = np.array([[0], [8]])
    values = np.array([1.0, 9.0])
    dense_shape = np.array([100])
    spec = sparse_tensor.SparseTensorSpec()
    st = spec._from_components([indices, values, dense_shape])
    self.assertIsInstance(st, sparse_tensor.SparseTensorValue)
    self.assertAllEqual(st.indices, indices)
    self.assertAllEqual(st.values, values)
    self.assertAllEqual(st.dense_shape, dense_shape)

  @parameterized.parameters([
      sparse_tensor.SparseTensorSpec(dtype=dtypes.string),
      sparse_tensor.SparseTensorSpec(shape=[5, None, None]),
  ])
  def testFlatTensorSpecs(self, st_spec):
    self.assertEqual(st_spec._flat_tensor_specs,
                     [tensor_spec.TensorSpec(None, dtypes.variant)])

  @parameterized.parameters([
      {
          "st_spec": sparse_tensor.SparseTensorSpec(),
          "indices": [[0, 1], [10, 8]],
          "values": [3.0, 5.0],
          "dense_shape": [100, 100]
      },
      {
          "st_spec": sparse_tensor.SparseTensorSpec([100, None, None]),
          "indices": [[0, 1, 3], [10, 8, 2]],
          "values": [3.0, 5.0],
          "dense_shape": [100, 20, 20]
      },
  ])
  def testToFromTensorList(self, st_spec, indices, values, dense_shape):
    st = sparse_tensor.SparseTensor(indices, values, dense_shape)
    tensor_list = st_spec._to_tensor_list(st)
    st_reconstructed = st_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(st.indices, st_reconstructed.indices)
    self.assertAllEqual(st.values, st_reconstructed.values)
    self.assertAllEqual(st.dense_shape, st_reconstructed.dense_shape)

  @parameterized.parameters([
      (sparse_tensor.SparseTensorSpec([2, None], dtypes.float32), 32,
       sparse_tensor.SparseTensorSpec([32, 2, None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([4, None], dtypes.float32), None,
       sparse_tensor.SparseTensorSpec([None, 4, None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([2], dtypes.float32), 32,
       sparse_tensor.SparseTensorSpec([32, 2], dtypes.float32)),
  ])
  def testBatch(self, spec, batch_size, expected):
    self.assertEqual(spec._batch(batch_size), expected)

  @parameterized.parameters([
      (sparse_tensor.SparseTensorSpec([32, None, None], dtypes.float32),
       sparse_tensor.SparseTensorSpec([None, None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([None, None, None], dtypes.float32),
       sparse_tensor.SparseTensorSpec([None, None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([32, 2], dtypes.float32),
       sparse_tensor.SparseTensorSpec([2], dtypes.float32)),
  ])
  def testUnbatch(self, spec, expected):
    self.assertEqual(spec._unbatch(), expected)


if __name__ == "__main__":
  googletest.main()
