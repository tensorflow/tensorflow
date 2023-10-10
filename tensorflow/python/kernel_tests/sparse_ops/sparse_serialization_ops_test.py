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
"""Tests for SerializeSparse."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


class SerializeSparseTest(test.TestCase):

  def _SparseTensorPlaceholder(self, dtype=None):
    if dtype is None:
      dtype = dtypes.int32
    return sparse_tensor_lib.SparseTensor(
        array_ops.placeholder(dtypes.int64),
        array_ops.placeholder(dtype), array_ops.placeholder(dtypes.int64))

  def _SparseTensorValue_5x6(self, permutation):
    ind = np.array([[0, 0], [1, 0], [1, 3], [1, 4], [3, 2],
                    [3, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([5, 6]).astype(np.int64)
    return sparse_tensor_lib.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_3x4(self, permutation):
    ind = np.array([[0, 0], [1, 0], [1, 2], [1, 3], [2, 2],
                    [2, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([3, 4]).astype(np.int64)
    return sparse_tensor_lib.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_1x1x1(self):
    ind = np.array([[0, 0, 0]]).astype(np.int64)
    val = np.array([0]).astype(np.int32)
    shape = np.array([3, 4, 5]).astype(np.int64)
    return sparse_tensor_lib.SparseTensorValue(ind, val, shape)

  def _testSerializeDeserializeHelper(self,
                                      serialize_fn,
                                      deserialize_fn,
                                      out_type=dtypes.string):
    with self.cached_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorValue_5x6(np.arange(6))
      serialized = serialize_fn(sp_input, out_type=out_type)
      sp_deserialized = deserialize_fn(serialized, dtype=dtypes.int32)

      indices, values, shape = self.evaluate(sp_deserialized)

      self.assertAllEqual(indices, sp_input[0])
      self.assertAllEqual(values, sp_input[1])
      self.assertAllEqual(shape, sp_input[2])

  def testSerializeDeserialize(self):
    self._testSerializeDeserializeHelper(sparse_ops.serialize_sparse,
                                         sparse_ops.deserialize_sparse)

  def testVariantSerializeDeserialize(self):
    self._testSerializeDeserializeHelper(sparse_ops.serialize_sparse,
                                         sparse_ops.deserialize_sparse,
                                         dtypes.variant)

  def _testSerializeDeserializeBatchHelper(self,
                                           serialize_fn,
                                           deserialize_fn,
                                           out_type=dtypes.string):
    with self.cached_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorValue_5x6(np.arange(6))
      serialized = serialize_fn(sp_input, out_type=out_type)
      serialized = array_ops_stack.stack([serialized, serialized])

      sp_deserialized = deserialize_fn(serialized, dtype=dtypes.int32)

      combined_indices, combined_values, combined_shape = sess.run(
          sp_deserialized)

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], sp_input[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], sp_input[0])
      self.assertAllEqual(combined_values[:6], sp_input[1])
      self.assertAllEqual(combined_values[6:], sp_input[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  @test_util.run_deprecated_v1
  def testSerializeDeserializeBatch(self):
    self._testSerializeDeserializeBatchHelper(sparse_ops.serialize_sparse,
                                              sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testSerializeDeserializeManyBatch(self):
    self._testSerializeDeserializeBatchHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_many_sparse)

  @test_util.run_deprecated_v1
  def testVariantSerializeDeserializeBatch(self):
    self._testSerializeDeserializeBatchHelper(sparse_ops.serialize_sparse,
                                              sparse_ops.deserialize_sparse,
                                              dtypes.variant)

  def _testSerializeDeserializeBatchInconsistentShapeHelper(
      self, serialize_fn, deserialize_fn, out_type=dtypes.string):
    with self.cached_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorValue_5x6(np.arange(6))
      sp_input1 = self._SparseTensorValue_3x4(np.arange(6))
      serialized0 = serialize_fn(sp_input0, out_type=out_type)
      serialized1 = serialize_fn(sp_input1, out_type=out_type)
      serialized = array_ops_stack.stack([serialized0, serialized1])

      sp_deserialized = deserialize_fn(serialized, dtype=dtypes.int32)

      combined_indices, combined_values, combined_shape = sess.run(
          sp_deserialized)

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], sp_input0[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], sp_input1[0])
      self.assertAllEqual(combined_values[:6], sp_input0[1])
      self.assertAllEqual(combined_values[6:], sp_input1[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  @test_util.run_deprecated_v1
  def testSerializeDeserializeBatchInconsistentShape(self):
    self._testSerializeDeserializeBatchInconsistentShapeHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testVariantSerializeDeserializeBatchInconsistentShape(self):
    self._testSerializeDeserializeBatchInconsistentShapeHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse,
        dtypes.variant)

  def _testSerializeDeserializeNestedBatchHelper(self,
                                                 serialize_fn,
                                                 deserialize_fn,
                                                 out_type=dtypes.string):
    with self.cached_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorValue_5x6(np.arange(6))
      serialized = serialize_fn(sp_input, out_type=out_type)
      serialized = array_ops_stack.stack([serialized, serialized])
      serialized = array_ops_stack.stack([serialized, serialized])

      sp_deserialized = deserialize_fn(serialized, dtype=dtypes.int32)

      combined_indices, combined_values, combined_shape = sess.run(
          sp_deserialized)

      # minibatch 0
      self.assertAllEqual(combined_indices[:6, :2], [[0, 0]] * 6)
      self.assertAllEqual(combined_indices[:6, 2:], sp_input[0])
      self.assertAllEqual(combined_values[:6], sp_input[1])
      # minibatch 1
      self.assertAllEqual(combined_indices[6:12, :2], [[0, 1]] * 6)
      self.assertAllEqual(combined_indices[6:12, 2:], sp_input[0])
      self.assertAllEqual(combined_values[6:12], sp_input[1])
      # minibatch 2
      self.assertAllEqual(combined_indices[12:18, :2], [[1, 0]] * 6)
      self.assertAllEqual(combined_indices[12:18, 2:], sp_input[0])
      self.assertAllEqual(combined_values[12:18], sp_input[1])
      # minibatch 3
      self.assertAllEqual(combined_indices[18:, :2], [[1, 1]] * 6)
      self.assertAllEqual(combined_indices[18:, 2:], sp_input[0])
      self.assertAllEqual(combined_values[18:], sp_input[1])

      self.assertAllEqual(combined_shape, [2, 2, 5, 6])

  @test_util.run_deprecated_v1
  def testSerializeDeserializeNestedBatch(self):
    self._testSerializeDeserializeNestedBatchHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testVariantSerializeDeserializeNestedBatch(self):
    self._testSerializeDeserializeNestedBatchHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse,
        dtypes.variant)

  def _testFeedSerializeDeserializeBatchHelper(self,
                                               serialize_fn,
                                               deserialize_fn,
                                               out_type=dtypes.string):
    with self.cached_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      sp_input1 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_3x4(np.arange(6))
      serialized0 = serialize_fn(sp_input0, out_type=out_type)
      serialized1 = serialize_fn(sp_input1, out_type=out_type)
      serialized_concat = array_ops_stack.stack([serialized0, serialized1])

      sp_deserialized = deserialize_fn(serialized_concat, dtype=dtypes.int32)

      combined_indices, combined_values, combined_shape = sess.run(
          sp_deserialized, {sp_input0: input0_val,
                            sp_input1: input1_val})

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], input0_val[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], input1_val[0])
      self.assertAllEqual(combined_values[:6], input0_val[1])
      self.assertAllEqual(combined_values[6:], input1_val[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  @test_util.run_deprecated_v1
  def testFeedSerializeDeserializeBatch(self):
    self._testFeedSerializeDeserializeBatchHelper(sparse_ops.serialize_sparse,
                                                  sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testFeedSerializeDeserializeManyBatch(self):
    self._testFeedSerializeDeserializeBatchHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_many_sparse)

  @test_util.run_deprecated_v1
  def testFeedVariantSerializeDeserializeBatch(self):
    self._testFeedSerializeDeserializeBatchHelper(sparse_ops.serialize_sparse,
                                                  sparse_ops.deserialize_sparse,
                                                  dtypes.variant)

  def _testSerializeManyShapeHelper(self,
                                    serialize_many_fn,
                                    out_type=dtypes.string):
    with self.cached_session(use_gpu=False) as sess:
      # N == 4 because shape_value == [4, 5]
      indices_value = np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64)
      values_value = np.array([b"a", b"b", b"c"])
      shape_value = np.array([4, 5], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder(dtype=dtypes.string)
      serialized = serialize_many_fn(sparse_tensor, out_type=out_type)
      serialized_value = sess.run(
          serialized,
          feed_dict={
              sparse_tensor.indices: indices_value,
              sparse_tensor.values: values_value,
              sparse_tensor.dense_shape: shape_value
          })
      self.assertEqual(serialized_value.shape, (4, 3))

  @test_util.run_deprecated_v1
  def testSerializeManyShape(self):
    self._testSerializeManyShapeHelper(sparse_ops.serialize_many_sparse)

  def testVariantSerializeManyShape(self):
    # NOTE: The following test is a no-op as it is currently not possible to
    # convert the serialized variant value to a numpy value.
    pass

  def _testSerializeManyDeserializeBatchHelper(self,
                                               serialize_many_fn,
                                               deserialize_fn,
                                               out_type=dtypes.string):
    with self.cached_session(use_gpu=False) as sess:
      # N == 4 because shape_value == [4, 5]
      indices_value = np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64)
      values_value = np.array([b"a", b"b", b"c"])
      shape_value = np.array([4, 5], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder(dtype=dtypes.string)
      serialized = serialize_many_fn(sparse_tensor, out_type=out_type)
      deserialized = deserialize_fn(serialized, dtype=dtypes.string)
      deserialized_value = sess.run(
          deserialized,
          feed_dict={
              sparse_tensor.indices: indices_value,
              sparse_tensor.values: values_value,
              sparse_tensor.dense_shape: shape_value
          })
      self.assertAllEqual(deserialized_value.indices, indices_value)
      self.assertAllEqual(deserialized_value.values, values_value)
      self.assertAllEqual(deserialized_value.dense_shape, shape_value)

  @test_util.run_deprecated_v1
  def testSerializeManyDeserializeBatch(self):
    self._testSerializeManyDeserializeBatchHelper(
        sparse_ops.serialize_many_sparse, sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testSerializeManyDeserializeManyBatch(self):
    self._testSerializeManyDeserializeBatchHelper(
        sparse_ops.serialize_many_sparse, sparse_ops.deserialize_many_sparse)

  @test_util.run_deprecated_v1
  def testVariantSerializeManyDeserializeBatch(self):
    self._testSerializeManyDeserializeBatchHelper(
        sparse_ops.serialize_many_sparse, sparse_ops.deserialize_sparse,
        dtypes.variant)

  @test_util.run_deprecated_v1
  def testVariantSerializeDeserializeScalar(self):
    with self.session(use_gpu=False) as sess:
      indices_value = np.array([[]], dtype=np.int64)
      values_value = np.array([37], dtype=np.int32)
      shape_value = np.array([], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder()
      serialized = sparse_ops.serialize_sparse(
          sparse_tensor, out_type=dtypes.variant)
      deserialized = sparse_ops.deserialize_sparse(
          serialized, dtype=dtypes.int32)
      deserialized_value = sess.run(
          deserialized,
          feed_dict={
              sparse_tensor.indices: indices_value,
              sparse_tensor.values: values_value,
              sparse_tensor.dense_shape: shape_value
          })
      self.assertAllEqual(deserialized_value.indices, indices_value)
      self.assertAllEqual(deserialized_value.values, values_value)
      self.assertAllEqual(deserialized_value.dense_shape, shape_value)

  @test_util.run_deprecated_v1
  def testVariantSerializeDeserializeScalarBatch(self):
    with self.session(use_gpu=False) as sess:
      indices_value = np.array([[]], dtype=np.int64)
      values_value = np.array([37], dtype=np.int32)
      shape_value = np.array([], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder()
      serialized = sparse_ops.serialize_sparse(
          sparse_tensor, out_type=dtypes.variant)
      stacked = array_ops_stack.stack([serialized, serialized])
      deserialized = sparse_ops.deserialize_sparse(stacked, dtype=dtypes.int32)
      deserialized_value = sess.run(
          deserialized,
          feed_dict={
              sparse_tensor.indices: indices_value,
              sparse_tensor.values: values_value,
              sparse_tensor.dense_shape: shape_value
          })
      self.assertAllEqual(deserialized_value.indices,
                          np.array([[0], [1]], dtype=np.int64))
      self.assertAllEqual(deserialized_value.values,
                          np.array([37, 37], dtype=np.int32))
      self.assertAllEqual(deserialized_value.dense_shape,
                          np.array([2], dtype=np.int64))

  def _testDeserializeFailsWrongTypeHelper(self,
                                           serialize_fn,
                                           deserialize_fn,
                                           out_type=dtypes.string):
    with self.cached_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      sp_input1 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_3x4(np.arange(6))
      serialized0 = serialize_fn(sp_input0, out_type=out_type)
      serialized1 = serialize_fn(sp_input1, out_type=out_type)
      serialized_concat = array_ops_stack.stack([serialized0, serialized1])

      sp_deserialized = deserialize_fn(serialized_concat, dtype=dtypes.int64)

      with self.assertRaisesOpError(
          r"Requested SparseTensor of type int64 but "
          r"SparseTensor\[0\].values.dtype\(\) == int32"):
        sess.run(sp_deserialized,
                 {sp_input0: input0_val,
                  sp_input1: input1_val})

  @test_util.run_deprecated_v1
  def testDeserializeFailsWrongType(self):
    self._testDeserializeFailsWrongTypeHelper(sparse_ops.serialize_sparse,
                                              sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testDeserializeManyFailsWrongType(self):
    self._testDeserializeFailsWrongTypeHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_many_sparse)

  @test_util.run_deprecated_v1
  def testVariantDeserializeFailsWrongType(self):
    self._testDeserializeFailsWrongTypeHelper(sparse_ops.serialize_sparse,
                                              sparse_ops.deserialize_sparse,
                                              dtypes.variant)

  def _testDeserializeFailsInconsistentRankHelper(self,
                                                  serialize_fn,
                                                  deserialize_fn,
                                                  out_type=dtypes.string):
    with self.cached_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      sp_input1 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_1x1x1()
      serialized0 = serialize_fn(sp_input0, out_type=out_type)
      serialized1 = serialize_fn(sp_input1, out_type=out_type)
      serialized_concat = array_ops_stack.stack([serialized0, serialized1])

      sp_deserialized = deserialize_fn(serialized_concat, dtype=dtypes.int32)

      with self.assertRaisesOpError(
          r"Inconsistent shape across SparseTensors: rank prior to "
          r"SparseTensor\[1\] was: 2 but rank of SparseTensor\[1\] is: 3"):
        sess.run(sp_deserialized,
                 {sp_input0: input0_val,
                  sp_input1: input1_val})

  @test_util.run_deprecated_v1
  def testDeserializeFailsInconsistentRank(self):
    self._testDeserializeFailsInconsistentRankHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testDeserializeManyFailsInconsistentRank(self):
    self._testDeserializeFailsInconsistentRankHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_many_sparse)

  @test_util.run_deprecated_v1
  def testVariantDeserializeFailsInconsistentRank(self):
    self._testDeserializeFailsInconsistentRankHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_sparse,
        dtypes.variant)

  def _testDeserializeFailsInvalidProtoHelper(self,
                                              serialize_fn,
                                              deserialize_fn,
                                              out_type=dtypes.string):
    with self.cached_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      serialized0 = serialize_fn(sp_input0, out_type=out_type)
      serialized1 = ["a", "b", "c"]
      serialized_concat = array_ops_stack.stack([serialized0, serialized1])

      sp_deserialized = deserialize_fn(serialized_concat, dtype=dtypes.int32)

      with self.assertRaisesOpError(r"Could not parse serialized proto"):
        sess.run(sp_deserialized, {sp_input0: input0_val})

  @test_util.run_deprecated_v1
  def testDeserializeFailsInvalidProto(self):
    self._testDeserializeFailsInvalidProtoHelper(sparse_ops.serialize_sparse,
                                                 sparse_ops.deserialize_sparse)

  @test_util.run_deprecated_v1
  def testDeserializeManyFailsInvalidProto(self):
    self._testDeserializeFailsInvalidProtoHelper(
        sparse_ops.serialize_sparse, sparse_ops.deserialize_many_sparse)

  def testDeserializeInvalidVariant(self):
    mu = gen_resource_variable_ops.mutex_v2()
    mu_lock = gen_resource_variable_ops.mutex_lock(mutex=mu)

    @def_function.function
    def f():
      return sparse_ops.deserialize_sparse(
          serialized_sparse=mu_lock, dtype=dtypes.int32)

    with self.assertRaisesRegex(ValueError, r"Shape must be at least rank 1"):
      f()


if __name__ == "__main__":
  test.main()
