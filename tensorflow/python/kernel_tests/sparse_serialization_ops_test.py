# Copyright 2015 Google Inc. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SerializeSparseTest(tf.test.TestCase):

  def _SparseTensorPlaceholder(self, dtype=None):
    if dtype is None: dtype = tf.int32
    return tf.SparseTensor(
        tf.placeholder(tf.int64),
        tf.placeholder(dtype),
        tf.placeholder(tf.int64))

  def _SparseTensorValue_5x6(self, permutation):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([5, 6]).astype(np.int64)
    return tf.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_3x4(self, permutation):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 2], [1, 3],
        [2, 2], [2, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([3, 4]).astype(np.int64)
    return tf.SparseTensorValue(ind, val, shape)

  def _SparseTensorValue_1x1x1(self):
    ind = np.array([[0, 0, 0]]).astype(np.int64)
    val = np.array([0]).astype(np.int32)
    shape = np.array([3, 4, 5]).astype(np.int64)
    return tf.SparseTensorValue(ind, val, shape)

  def testSerializeDeserializeMany(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      sp_input1 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_3x4(np.arange(6))
      serialized0 = tf.serialize_sparse(sp_input0)
      serialized1 = tf.serialize_sparse(sp_input1)
      serialized_concat = tf.pack([serialized0, serialized1])

      sp_deserialized = tf.deserialize_many_sparse(
          serialized_concat, dtype=tf.int32)

      combined_indices, combined_values, combined_shape = sess.run(
          sp_deserialized, {sp_input0: input0_val, sp_input1: input1_val})

      self.assertAllEqual(combined_indices[:6, 0], [0] * 6)  # minibatch 0
      self.assertAllEqual(combined_indices[:6, 1:], input0_val[0])
      self.assertAllEqual(combined_indices[6:, 0], [1] * 6)  # minibatch 1
      self.assertAllEqual(combined_indices[6:, 1:], input1_val[0])
      self.assertAllEqual(combined_values[:6], input0_val[1])
      self.assertAllEqual(combined_values[6:], input1_val[1])
      self.assertAllEqual(combined_shape, [2, 5, 6])

  def testSerializeManyDeserializeManyRoundTrip(self):
    with self.test_session(use_gpu=False) as sess:
      # N == 4 because shape_value == [4, 5]
      indices_value = np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64)
      values_value = np.array([b"a", b"b", b"c"])
      shape_value = np.array([4, 5], dtype=np.int64)
      sparse_tensor = self._SparseTensorPlaceholder(dtype=tf.string)
      serialized = tf.serialize_many_sparse(sparse_tensor)
      deserialized = tf.deserialize_many_sparse(serialized, dtype=tf.string)
      serialized_value, deserialized_value = sess.run(
          [serialized, deserialized],
          feed_dict={sparse_tensor.indices: indices_value,
                     sparse_tensor.values: values_value,
                     sparse_tensor.shape: shape_value})
      self.assertEqual(serialized_value.shape, (4, 3))
      self.assertAllEqual(deserialized_value.indices, indices_value)
      self.assertAllEqual(deserialized_value.values, values_value)
      self.assertAllEqual(deserialized_value.shape, shape_value)

  def testDeserializeFailsWrongType(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      sp_input1 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_3x4(np.arange(6))
      serialized0 = tf.serialize_sparse(sp_input0)
      serialized1 = tf.serialize_sparse(sp_input1)
      serialized_concat = tf.pack([serialized0, serialized1])

      sp_deserialized = tf.deserialize_many_sparse(
          serialized_concat, dtype=tf.int64)

      with self.assertRaisesOpError(
          r"Requested SparseTensor of type int64 but "
          r"SparseTensor\[0\].values.dtype\(\) == int32"):
        sess.run(
            sp_deserialized, {sp_input0: input0_val, sp_input1: input1_val})

  def testDeserializeFailsInconsistentRank(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      sp_input1 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      input1_val = self._SparseTensorValue_1x1x1()
      serialized0 = tf.serialize_sparse(sp_input0)
      serialized1 = tf.serialize_sparse(sp_input1)
      serialized_concat = tf.pack([serialized0, serialized1])

      sp_deserialized = tf.deserialize_many_sparse(
          serialized_concat, dtype=tf.int32)

      with self.assertRaisesOpError(
          r"Inconsistent rank across SparseTensors: rank prior to "
          r"SparseTensor\[1\] was: 3 but rank of SparseTensor\[1\] is: 4"):
        sess.run(
            sp_deserialized, {sp_input0: input0_val, sp_input1: input1_val})

  def testDeserializeFailsInvalidProto(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input0 = self._SparseTensorPlaceholder()
      input0_val = self._SparseTensorValue_5x6(np.arange(6))
      serialized0 = tf.serialize_sparse(sp_input0)
      serialized1 = ["a", "b", "c"]
      serialized_concat = tf.pack([serialized0, serialized1])

      sp_deserialized = tf.deserialize_many_sparse(
          serialized_concat, dtype=tf.int32)

      with self.assertRaisesOpError(
          r"Could not parse serialized_sparse\[1, 0\]"):
        sess.run(sp_deserialized, {sp_input0: input0_val})


if __name__ == "__main__":
  tf.test.main()
