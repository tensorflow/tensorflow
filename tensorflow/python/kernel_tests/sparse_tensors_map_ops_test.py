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
"""Tests for SparseTensorsMap."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test

# pylint: disable=protected-access
add_sparse_to_tensors_map = sparse_ops._add_sparse_to_tensors_map
add_many_sparse_to_tensors_map = sparse_ops._add_many_sparse_to_tensors_map
take_many_sparse_from_tensors_map = (
    sparse_ops._take_many_sparse_from_tensors_map)

# pylint: enable=protected-access


class SparseTensorsMapTest(test.TestCase):

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

  @test_util.run_deprecated_v1
  def testAddTakeMany(self):
    with self.session(graph=ops.Graph(), use_gpu=False) as sess:
      sp_input0 = self._SparseTensorValue_5x6(np.arange(6))
      sp_input1 = self._SparseTensorValue_3x4(np.arange(6))
      handle0 = add_sparse_to_tensors_map(sp_input0, shared_name="a")
      handle1 = add_sparse_to_tensors_map(sp_input1, shared_name="a")
      self.assertEqual(handle0.get_shape(), ())
      handles_concat = array_ops.stack([handle0, handle1])

      sp_out = take_many_sparse_from_tensors_map(
          sparse_map_op=handle0.op, sparse_handles=handles_concat)

      combined_indices, combined_values, combined_shape = self.evaluate(sp_out)

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], sp_input0[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], sp_input1[0])
      self.assertAllEqual(combined_values[:6], sp_input0[1])
      self.assertAllEqual(combined_values[6:], sp_input1[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  @test_util.run_deprecated_v1
  def testFeedAddTakeMany(self):
    with self.session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_3x4(np.arange(6))
      handle = add_sparse_to_tensors_map(sp_input)

      handle0_value = sess.run(handle, feed_dict={sp_input: input0_val})
      handle1_value = sess.run(handle, feed_dict={sp_input: input1_val})

      sparse_handles = ops.convert_to_tensor(
          [handle0_value, handle1_value], dtype=dtypes.int64)

      sp_roundtrip = take_many_sparse_from_tensors_map(
          sparse_map_op=handle.op, sparse_handles=sparse_handles)

      combined_indices, combined_values, combined_shape = self.evaluate(
          sp_roundtrip)

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], input0_val[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], input1_val[0])
      self.assertAllEqual(combined_values[:6], input0_val[1])
      self.assertAllEqual(combined_values[6:], input1_val[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  @test_util.run_deprecated_v1
  def testAddManyTakeManyRoundTrip(self):
    with self.session(use_gpu=False) as sess:
      # N == 4 because shape_value == [4, 5]
      indices_value = np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64)
      values_value = np.array([b"a", b"b", b"c"])
      shape_value = np.array([4, 5], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder(dtype=dtypes.string)
      handles = add_many_sparse_to_tensors_map(sparse_tensor)
      roundtrip = take_many_sparse_from_tensors_map(
          sparse_map_op=handles.op, sparse_handles=handles)
      handles_value, roundtrip_value = sess.run(
          [handles, roundtrip],
          feed_dict={
              sparse_tensor.indices: indices_value,
              sparse_tensor.values: values_value,
              sparse_tensor.dense_shape: shape_value
          })
      self.assertEqual(handles_value.shape, (4,))
      self.assertAllEqual(roundtrip_value.indices, indices_value)
      self.assertAllEqual(roundtrip_value.values, values_value)
      self.assertAllEqual(roundtrip_value.dense_shape, shape_value)

  @test_util.run_deprecated_v1
  def testDeserializeFailsInconsistentRank(self):
    with self.session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_1x1x1()
      handle = add_sparse_to_tensors_map(sp_input)

      handle0_value = sess.run(handle, feed_dict={sp_input: input0_val})
      handle1_value = sess.run(handle, feed_dict={sp_input: input1_val})

      handle_concat = ops.convert_to_tensor(
          [handle0_value, handle1_value], dtype=dtypes.int64)

      sp_roundtrip = take_many_sparse_from_tensors_map(
          sparse_map_op=handle.op, sparse_handles=handle_concat)

      with self.assertRaisesOpError(
          r"Inconsistent rank across SparseTensors: rank prior to "
          r"SparseTensor\[1\] was: 3 but rank of SparseTensor\[1\] is: 4"):
        self.evaluate(sp_roundtrip)

  @test_util.run_deprecated_v1
  def testTakeManyFailsWrongInputOp(self):
    with self.session(use_gpu=False) as sess:
      input_val = self._SparseTensorValue_5x6(np.arange(6))
      handle = add_sparse_to_tensors_map(input_val)
      handle_value = self.evaluate(handle)
      bad_handle = handle_value + 10
      sp_roundtrip = take_many_sparse_from_tensors_map(
          sparse_map_op=handle.op, sparse_handles=[handle_value, bad_handle])

      with self.assertRaisesOpError(r"Unable to find SparseTensor: 10"):
        self.evaluate(sp_roundtrip)


class BenchmarkSparseTensorsMapVsSerialization(test.Benchmark):

  def benchmarkVeryLarge2DFloatSparseTensor(self):
    np.random.seed(127)
    num_elements = 10000
    batch_size = 64
    indices_batch = np.random.randint(
        batch_size, size=num_elements, dtype=np.int64)
    indices_value = np.arange(num_elements, dtype=np.int64)
    indices = np.asarray(
        sorted(zip(indices_batch, indices_value)), dtype=np.int64)
    values = ["feature_value_for_embedding_lookup"] * num_elements
    shape = np.asarray([batch_size, num_elements], dtype=np.int64)
    with session.Session(config=benchmark.benchmark_config()) as sess:
      with ops.device("/cpu:0"):
        indices = variables.Variable(indices)
        values = variables.Variable(values)
        shape = variables.Variable(shape)
        st = sparse_tensor_lib.SparseTensor(indices, values, shape)

        st_handles = add_many_sparse_to_tensors_map(st)
        st_roundtrip = take_many_sparse_from_tensors_map(
            sparse_map_op=st_handles.op, sparse_handles=st_handles)
        st_roundtrip_op = st_roundtrip.values.op

        st_serialized = sparse_ops.serialize_many_sparse(st)
        st_deserialized = sparse_ops.deserialize_many_sparse(
            st_serialized, dtype=values.dtype)
        st_deserialized_op = st_deserialized.values.op

        self.evaluate(variables.global_variables_initializer())

        st_roundtrip_values = self.evaluate(st_roundtrip)
        st_deserialized_values = self.evaluate(st_deserialized)
        np.testing.assert_equal(st_roundtrip_values.values,
                                st_deserialized_values.values)
        np.testing.assert_equal(st_roundtrip_values.indices,
                                st_deserialized_values.indices)
        np.testing.assert_equal(st_roundtrip_values.dense_shape,
                                st_deserialized_values.dense_shape)

        self.run_op_benchmark(
            sess,
            st_roundtrip_op,
            min_iters=2000,
            name="benchmark_very_large_2d_float_st_tensor_maps")
        self.run_op_benchmark(
            sess,
            st_deserialized_op,
            min_iters=2000,
            name="benchmark_very_large_2d_float_st_serialization")


if __name__ == "__main__":
  test.main()
