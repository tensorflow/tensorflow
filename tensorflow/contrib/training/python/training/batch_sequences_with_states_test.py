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
"""Tests for tf.batch_sequences_with_states."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.contrib.training.python.training import sequence_queueing_state_saver as sqss
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl
from tensorflow.python.training import saver


class BatchSequencesWithStatesTest(test.TestCase):

  def setUp(self):
    super(BatchSequencesWithStatesTest, self).setUp()
    self.value_length = 4
    ind1 = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]])
    val1 = np.array([0, 10, 13, 14, 32, 33])
    shape1 = np.array([self.value_length, 6])
    sp_tensor1 = sparse_tensor.SparseTensor(
        array_ops.constant(ind1, dtypes.int64),
        array_ops.constant(val1, dtypes.int64),
        array_ops.constant(shape1, dtypes.int64))
    ind2 = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 2],
        [1, 0, 3],
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 2],
        [1, 2, 2]])
    val2 = np.array([1, 10, 12, 103, 150, 149, 150, 122])
    shape2 = np.array([self.value_length, 3, 4])
    sp_tensor2 = sparse_tensor.SparseTensor(
        array_ops.constant(ind2, dtypes.int64),
        array_ops.constant(val2, dtypes.int64),
        array_ops.constant(shape2, dtypes.int64))
    sp_tensor3 = sparse_tensor.SparseTensor(
        array_ops.constant([[1, 9], [2, 2], [2, 10]], dtypes.int64),
        array_ops.constant([7, 15, 2], dtypes.int64),
        array_ops.constant([5, 12], dtypes.int64)
    )
    self.sp_tensor3_expected = sparse_tensor.SparseTensorValue(
        [[0, 1, 9], [0, 2, 2], [0, 2, 10], [1, 1, 9], [1, 2, 2], [1, 2, 10]],
        [7, 15, 2, 7, 15, 2],
        [2, 5, 12]
    )
    self.batch_size = 2
    self.key = string_ops.string_join([
        "key_", string_ops.as_string(
            math_ops.cast(10000 * random_ops.random_uniform(()), dtypes.int32))
    ])
    self.sequences = {
        "seq1": np.random.rand(self.value_length, 5),
        "seq2": np.random.rand(self.value_length, 4, 2),
        "seq3": sp_tensor1,
        "seq4": sp_tensor2}
    self.context = {
        "context1": [3, 4],
        "sp_context": sp_tensor3}
    self.initial_states = {
        "state1": np.random.rand(6, 7),
        "state2": np.random.rand(8)
    }

  def _prefix(self, key_value):
    return set(
        [s.decode("ascii").split(":")[0].encode("ascii") for s in key_value])

  def _testBasics(self, num_unroll, length, pad,
                  expected_seq1_batch1, expected_seq2_batch1,
                  expected_seq1_batch2, expected_seq2_batch2,
                  expected_seq3_batch1, expected_seq3_batch2,
                  expected_seq4_batch1, expected_seq4_batch2,
                  key=None, make_keys_unique=False):

    with self.test_session() as sess:
      next_batch = sqss.batch_sequences_with_states(
          input_key=key if key is not None else self.key,
          input_sequences=self.sequences,
          input_context=self.context,
          input_length=length,
          initial_states=self.initial_states,
          num_unroll=num_unroll,
          batch_size=self.batch_size,
          num_threads=3,
          # to enforce that we only move on to the next examples after finishing
          # all segments of the first ones.
          capacity=2,
          pad=pad,
          make_keys_unique=make_keys_unique,
          make_keys_unique_seed=9)

      state1 = next_batch.state("state1")
      state2 = next_batch.state("state2")
      state1_update = next_batch.save_state("state1", state1 + 1)
      state2_update = next_batch.save_state("state2", state2 - 1)

      # Make sure queue runner with SQSS is added properly to meta graph def.
      # Saver requires at least one variable.
      v0 = variables.Variable(10.0, name="v0")
      ops.add_to_collection("variable_collection", v0)
      variables.global_variables_initializer()
      save = saver.Saver([v0])
      test_dir = os.path.join(test.get_temp_dir(), "sqss_test")
      filename = os.path.join(test_dir, "metafile")
      meta_graph_def = save.export_meta_graph(filename)
      qr_saved = meta_graph_def.collection_def[ops.GraphKeys.QUEUE_RUNNERS]
      self.assertTrue(qr_saved.bytes_list.value is not None)

      coord = coordinator.Coordinator()
      threads = queue_runner_impl.start_queue_runners(coord=coord)

      # Step 1
      (key_value, next_key_value, seq1_value, seq2_value, seq3_value,
       seq4_value, context1_value, context2_value, state1_value, state2_value,
       length_value, _, _) = sess.run(
           (next_batch.key, next_batch.next_key, next_batch.sequences["seq1"],
            next_batch.sequences["seq2"], next_batch.sequences["seq3"],
            next_batch.sequences["seq4"], next_batch.context["context1"],
            next_batch.context["sp_context"], state1, state2, next_batch.length,
            state1_update, state2_update))
      expected_first_keys = set([b"00000_of_00002"])
      expected_second_keys = set([b"00001_of_00002"])
      expected_final_keys = set([b"STOP"])

      self.assertEqual(expected_first_keys, self._prefix(key_value))
      self.assertEqual(expected_second_keys, self._prefix(next_key_value))
      self.assertAllEqual(
          np.tile(self.context["context1"], (self.batch_size, 1)),
          context1_value)
      self.assertAllEqual(self.sp_tensor3_expected.indices,
                          context2_value.indices)
      self.assertAllEqual(self.sp_tensor3_expected.values,
                          context2_value.values)
      self.assertAllEqual(self.sp_tensor3_expected.dense_shape,
                          context2_value.dense_shape)
      self.assertAllEqual(expected_seq1_batch1, seq1_value)
      self.assertAllEqual(expected_seq2_batch1, seq2_value)
      self.assertAllEqual(expected_seq3_batch1.indices, seq3_value.indices)
      self.assertAllEqual(expected_seq3_batch1.values, seq3_value.values)
      self.assertAllEqual(expected_seq3_batch1.dense_shape,
                          seq3_value.dense_shape)
      self.assertAllEqual(expected_seq4_batch1.indices, seq4_value.indices)
      self.assertAllEqual(expected_seq4_batch1.values, seq4_value.values)
      self.assertAllEqual(expected_seq4_batch1.dense_shape,
                          seq4_value.dense_shape)
      self.assertAllEqual(
          np.tile(self.initial_states["state1"], (self.batch_size, 1, 1)),
          state1_value)
      self.assertAllEqual(
          np.tile(self.initial_states["state2"], (self.batch_size, 1)),
          state2_value)
      self.assertAllEqual(length_value, [num_unroll, num_unroll])

      # Step 2
      (key_value, next_key_value, seq1_value, seq2_value, seq3_value,
       seq4_value, context1_value, context2_value, state1_value, state2_value,
       length_value, _, _) = sess.run(
           (next_batch.key, next_batch.next_key, next_batch.sequences["seq1"],
            next_batch.sequences["seq2"], next_batch.sequences["seq3"],
            next_batch.sequences["seq4"], next_batch.context["context1"],
            next_batch.context["sp_context"], state1, state2, next_batch.length,
            state1_update, state2_update))

      self.assertEqual(expected_second_keys, self._prefix(key_value))
      self.assertEqual(expected_final_keys, self._prefix(next_key_value))
      self.assertAllEqual(
          np.tile(self.context["context1"], (self.batch_size, 1)),
          context1_value)
      self.assertAllEqual(self.sp_tensor3_expected.indices,
                          context2_value.indices)
      self.assertAllEqual(self.sp_tensor3_expected.values,
                          context2_value.values)
      self.assertAllEqual(self.sp_tensor3_expected.dense_shape,
                          context2_value.dense_shape)
      self.assertAllEqual(expected_seq1_batch2, seq1_value)
      self.assertAllEqual(expected_seq2_batch2, seq2_value)
      self.assertAllEqual(expected_seq3_batch2.indices, seq3_value.indices)
      self.assertAllEqual(expected_seq3_batch2.values, seq3_value.values)
      self.assertAllEqual(expected_seq3_batch2.dense_shape,
                          seq3_value.dense_shape)
      self.assertAllEqual(expected_seq4_batch2.indices, seq4_value.indices)
      self.assertAllEqual(expected_seq4_batch2.values, seq4_value.values)
      self.assertAllEqual(expected_seq4_batch2.dense_shape,
                          seq4_value.dense_shape)
      self.assertAllEqual(1 + np.tile(self.initial_states["state1"],
                                      (self.batch_size, 1, 1)), state1_value)
      self.assertAllEqual(-1 + np.tile(self.initial_states["state2"],
                                       (self.batch_size, 1)), state2_value)
      self.assertAllEqual([1, 1], length_value)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=2)

  def _testBasicPadding(self, pad, key=None, make_keys_unique=False):
    num_unroll = 2  # Divisor of value_length - so no padding necessary.
    expected_seq1_batch1 = np.tile(
        self.sequences["seq1"][np.newaxis, 0:num_unroll, :],
        (self.batch_size, 1, 1))
    expected_seq2_batch1 = np.tile(
        self.sequences["seq2"][np.newaxis, 0:num_unroll, :, :],
        (self.batch_size, 1, 1, 1))
    expected_seq1_batch2 = np.tile(
        self.sequences["seq1"][np.newaxis, num_unroll:self.value_length, :],
        (self.batch_size, 1, 1))
    expected_seq2_batch2 = np.tile(
        self.sequences["seq2"][np.newaxis, num_unroll:self.value_length, :, :],
        (self.batch_size, 1, 1, 1))
    ind1_1 = np.array([
        # batch entry 1
        [0, 0, 0],
        [0, 1, 0], [0, 1, 3], [0, 1, 4],
        # batch entry 2
        [1, 0, 0],
        [1, 1, 0], [1, 1, 3], [1, 1, 4]])
    ind1_2 = np.array([
        # batch entry 1
        [0, 1, 2], [0, 1, 3],
        # batch entry 2
        [1, 1, 2], [1, 1, 3]])
    val1_1 = np.array([0, 10, 13, 14,
                       0, 10, 13, 14])
    val1_2 = np.array([32, 33,
                       32, 33])
    shape1 = np.array([self.batch_size, num_unroll, 6])

    # For sp_tensor2 all values fall into the first segment.
    ind2_1 = np.array([
        # batch entry 1
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 2],
        [0, 1, 0, 3],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 2],
        [0, 1, 2, 2],
        # batch entry 2
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 2],
        [1, 1, 0, 3],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 2],
        [1, 1, 2, 2],
    ])
    val2_1 = np.array([1, 10, 12, 103, 150, 149, 150, 122,
                       1, 10, 12, 103, 150, 149, 150, 122])
    shape2 = np.array([self.batch_size, num_unroll, 3, 4])
    expected_seq3_batch1 = sparse_tensor.SparseTensorValue(
        ind1_1, val1_1, shape1)
    expected_seq3_batch2 = sparse_tensor.SparseTensorValue(
        ind1_2, val1_2, shape1)
    expected_seq4_batch1 = sparse_tensor.SparseTensorValue(
        ind2_1, val2_1, shape2)
    expected_seq4_batch2 = sparse_tensor.SparseTensorValue(
        np.empty(shape=[0, 4], dtype=np.int64), np.array([]), shape2)
    self._testBasics(
        num_unroll=num_unroll,
        length=3,
        pad=pad,
        expected_seq1_batch1=expected_seq1_batch1,
        expected_seq1_batch2=expected_seq1_batch2,
        expected_seq2_batch1=expected_seq2_batch1,
        expected_seq2_batch2=expected_seq2_batch2,
        expected_seq3_batch1=expected_seq3_batch1,
        expected_seq3_batch2=expected_seq3_batch2,
        expected_seq4_batch1=expected_seq4_batch1,
        expected_seq4_batch2=expected_seq4_batch2,
        key=key,
        make_keys_unique=make_keys_unique)

  def testBasicPadding(self):
    self._testBasicPadding(pad=True)

  def testBasicNoPadding(self):
    self._testBasicPadding(pad=False)

  def testRandomKeyGen(self):
    self._testBasicPadding(pad=False,
                           key=constant_op.constant("fixed_key"),
                           make_keys_unique=True)

  def testNotAMultiple(self):
    num_unroll = 3  # Not a divisor of value_length -
    # so padding would have been necessary.
    with self.test_session() as sess:
      with self.assertRaisesRegexp(errors_impl.InvalidArgumentError,
                                   ".*should be a multiple of: 3, but saw "
                                   "value: 4. Consider setting pad=True."):
        coord = coordinator.Coordinator()
        threads = None
        try:
          with coord.stop_on_exception():
            next_batch = sqss.batch_sequences_with_states(
                input_key=self.key,
                input_sequences=self.sequences,
                input_context=self.context,
                input_length=3,
                initial_states=self.initial_states,
                num_unroll=num_unroll,
                batch_size=self.batch_size,
                num_threads=3,
                # to enforce that we only move on to the next examples after
                # finishing all segments of the first ones.
                capacity=2,
                pad=False)
            threads = queue_runner_impl.start_queue_runners(coord=coord)
            sess.run([next_batch.key])
        except errors_impl.OutOfRangeError:
          pass
        finally:
          coord.request_stop()
          if threads is not None:
            coord.join(threads, stop_grace_period_secs=2)

  def testAdvancedPadding(self):
    num_unroll = 3  # Not a divisor of value_length - so padding to 6 necessary.
    expected_seq1_batch1 = np.tile(
        self.sequences["seq1"][np.newaxis, 0:num_unroll, :],
        (self.batch_size, 1, 1))
    expected_seq2_batch1 = np.tile(
        self.sequences["seq2"][np.newaxis, 0:num_unroll, :, :],
        (self.batch_size, 1, 1, 1))

    padded_seq1 = np.concatenate(
        [
            self.sequences["seq1"][np.newaxis, num_unroll:self.value_length, :],
            np.zeros((1, 1, 5)), np.zeros((1, 1, 5))
        ],
        axis=1)
    expected_seq1_batch2 = np.concatenate(
        [padded_seq1] * self.batch_size, axis=0)

    padded_seq2 = np.concatenate(
        [
            self.sequences["seq2"][np.newaxis, num_unroll:self.value_length, :],
            np.zeros((1, 1, 4, 2)), np.zeros((1, 1, 4, 2))
        ],
        axis=1)
    expected_seq2_batch2 = np.concatenate(
        [padded_seq2] * self.batch_size, axis=0)

    ind1_1 = np.array([
        # batch entry 1
        [0, 0, 0],
        [0, 1, 0], [0, 1, 3], [0, 1, 4],
        # batch entry 2
        [1, 0, 0],
        [1, 1, 0], [1, 1, 3], [1, 1, 4]])
    ind1_2 = np.array([
        # batch entry 1
        [0, 0, 2], [0, 0, 3],
        # batch entry 2
        [1, 0, 2], [1, 0, 3]])
    val1_1 = np.array([0, 10, 13, 14,
                       0, 10, 13, 14])
    val1_2 = np.array([32, 33,
                       32, 33])
    shape1 = np.array([self.batch_size, num_unroll, 6])

    # For sp_tensor2 all values fall into the first segment.
    ind2_1 = np.array([
        # batch entry 1
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 2],
        [0, 1, 0, 3],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 2],
        [0, 1, 2, 2],
        # batch entry 2
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 2],
        [1, 1, 0, 3],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 2],
        [1, 1, 2, 2],
    ])
    val2_1 = np.array([1, 10, 12, 103, 150, 149, 150, 122,
                       1, 10, 12, 103, 150, 149, 150, 122])
    shape2 = np.array([self.batch_size, num_unroll, 3, 4])
    expected_seq3_batch1 = sparse_tensor.SparseTensorValue(
        ind1_1, val1_1, shape1)
    expected_seq3_batch2 = sparse_tensor.SparseTensorValue(
        ind1_2, val1_2, shape1)
    expected_seq4_batch1 = sparse_tensor.SparseTensorValue(
        ind2_1, val2_1, shape2)
    expected_seq4_batch2 = sparse_tensor.SparseTensorValue(
        np.empty(shape=[0, 4], dtype=np.int64), np.array([]), shape2)

    ind1_1 = np.array([
        # batch entry 1
        [0, 0, 0],
        [0, 1, 0], [0, 1, 3], [0, 1, 4],
        # batch entry 2
        [1, 0, 0],
        [1, 1, 0], [1, 1, 3], [1, 1, 4]])
    ind1_2 = np.array([
        # batch entry 1
        [0, 0, 2], [0, 0, 3],
        # batch entry 2
        [1, 0, 2], [1, 0, 3]])
    val1_1 = np.array([0, 10, 13, 14,
                       0, 10, 13, 14])
    val1_2 = np.array([32, 33,
                       32, 33])
    shape1 = np.array([self.batch_size, num_unroll, 6])

    # For sp_tensor2 all values fall into the first segment.
    ind2_1 = np.array([
        # batch entry 1
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 2],
        [0, 1, 0, 3],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [0, 1, 1, 2],
        [0, 1, 2, 2],
        # batch entry 2
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 2],
        [1, 1, 0, 3],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 2],
        [1, 1, 2, 2],
    ])
    val2_1 = np.array([1, 10, 12, 103, 150, 149, 150, 122,
                       1, 10, 12, 103, 150, 149, 150, 122])
    shape2 = np.array([self.batch_size, num_unroll, 3, 4])
    expected_seq3_batch1 = sparse_tensor.SparseTensorValue(
        ind1_1, val1_1, shape1)
    expected_seq3_batch2 = sparse_tensor.SparseTensorValue(
        ind1_2, val1_2, shape1)
    expected_seq4_batch1 = sparse_tensor.SparseTensorValue(
        ind2_1, val2_1, shape2)
    expected_seq4_batch2 = sparse_tensor.SparseTensorValue(
        np.empty(shape=[0, 4], dtype=np.int64), np.array([]), shape2)

    self._testBasics(
        num_unroll=num_unroll,
        length=None,
        pad=True,
        expected_seq1_batch1=expected_seq1_batch1,
        expected_seq1_batch2=expected_seq1_batch2,
        expected_seq2_batch1=expected_seq2_batch1,
        expected_seq2_batch2=expected_seq2_batch2,
        expected_seq3_batch1=expected_seq3_batch1,
        expected_seq3_batch2=expected_seq3_batch2,
        expected_seq4_batch1=expected_seq4_batch1,
        expected_seq4_batch2=expected_seq4_batch2)


class PaddingTest(test.TestCase):

  def testPaddingInvalidLengths(self):
    with ops.Graph().as_default() as g, self.test_session(graph=g):
      sequences = {
          "key_1": constant_op.constant([1, 2, 3]),  # length 3
          "key_2": constant_op.constant([1.5, 2.5])  # length 2
      }

      _, padded_seq = sqss._padding(sequences, 2)
      with self.assertRaisesOpError(
          ".*All sequence lengths must match, but received lengths.*"):
        padded_seq["key_1"].eval()

  def testPadding(self):
    with ops.Graph().as_default() as g, self.test_session(graph=g):
      sequences = {
          "key_1": constant_op.constant([1, 2]),
          "key_2": constant_op.constant([0.5, -1.0]),
          "key_3": constant_op.constant(["a", "b"]),  # padding strings
          "key_4": constant_op.constant([[1, 2, 3], [4, 5, 6]])
      }
      _, padded_seq = sqss._padding(sequences, 5)

      expected_padded_seq = {
          "key_1": [1, 2, 0, 0, 0],
          "key_2": [0.5, -1.0, 0.0, 0.0, 0.0],
          "key_3": ["a", "b", "", "", ""],
          "key_4": [[1, 2, 3], [4, 5, 6], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
      }

      for key, val in expected_padded_seq.items():
        self.assertTrue(
            math_ops.reduce_all(math_ops.equal(val, padded_seq[key])).eval())


class SparseTensorReConstructionTest(test.TestCase):

  def testAddManyTakeManyRoundTripBatched(self):
    with self.test_session(use_gpu=False) as sess:
      # N == 4 because shape_value == [4, 5]
      indices_value_1 = np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64)
      values_value_1 = np.array([b"a", b"b", b"c"])
      shape_value_1 = np.array([4, 5], dtype=np.int64)
      sparse_tensor_1 = sparse_tensor.SparseTensor(
          array_ops.placeholder(dtypes.int64),
          array_ops.placeholder(dtypes.string),
          array_ops.placeholder(dtypes.int64))
      dict1 = {"key": sparse_tensor_1}
      indices_value_2 = np.array([[1, 4], [2, 3]], dtype=np.int64)
      values_value_2 = np.array([b"d", b"e"])
      shape_value_2 = np.array([4, 5], dtype=np.int64)
      sparse_tensor_2 = sparse_tensor.SparseTensor(
          array_ops.placeholder(dtypes.int64),
          array_ops.placeholder(dtypes.string),
          array_ops.placeholder(dtypes.int64))
      dict2 = {"key": sparse_tensor_2}

      input_seq1, keys1, tensor_list1 = sqss._deconstruct_sparse_tensor_seq(
          dict1, shared_name="a")
      handles_1 = input_seq1["key"]
      input_seq2, _, _ = sqss._deconstruct_sparse_tensor_seq(
          dict2, shared_name="a")
      handles_2 = input_seq2["key"]

      combined_handles = array_ops.stack(
          [handles_1[1], handles_1[2], handles_1[3],
           handles_2[1], handles_2[2], handles_2[3]])
      batched_dict = {"key": combined_handles}
      sqss._reconstruct_sparse_tensor_seq(
          batched_dict,
          keys1,
          tensor_list1,
          batch_size=2,
          num_unroll=3)

      roundtrip_value, = sess.run(
          [batched_dict["key"]],
          feed_dict={sparse_tensor_1.indices: indices_value_1,
                     sparse_tensor_1.values: values_value_1,
                     sparse_tensor_1.dense_shape: shape_value_1,
                     sparse_tensor_2.indices: indices_value_2,
                     sparse_tensor_2.values: values_value_2,
                     sparse_tensor_2.dense_shape: shape_value_2})

      self.assertAllEqual(roundtrip_value.indices,
                          np.array([[0, 1, 0], [1, 0, 4], [1, 1, 3]],
                                   dtype=np.int64))
      self.assertAllEqual(roundtrip_value.values,
                          np.array([b"c", b"d", b"e"]))
      self.assertAllEqual(roundtrip_value.dense_shape,
                          np.array([2, 3, 5], dtype=np.int64))


if __name__ == "__main__":
  test.main()
