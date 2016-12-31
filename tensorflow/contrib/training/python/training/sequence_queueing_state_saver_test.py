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
"""Tests for tf.SequenceQueueingStateSaver."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np

from tensorflow.contrib.training.python.training import sequence_queueing_state_saver as sqss
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class SequenceQueueingStateSaverTest(test.TestCase):

  def testSequenceInputWrapper(self):
    with self.test_session():
      length = 3
      key = "key"
      padded_length = 4
      sequences = {
          "seq1": np.random.rand(padded_length, 5),
          "seq2": np.random.rand(padded_length, 4, 2)
      }
      context = {"context1": [3, 4]}
      input_wrapper = sqss._SequenceInputWrapper(length, key, sequences,
                                                 context)
      self.assertTrue(isinstance(input_wrapper.length, ops.Tensor))
      self.assertTrue(isinstance(input_wrapper.key, ops.Tensor))
      self.assertTrue(isinstance(input_wrapper.sequences["seq1"], ops.Tensor))
      self.assertTrue(isinstance(input_wrapper.sequences["seq2"], ops.Tensor))
      self.assertTrue(isinstance(input_wrapper.context["context1"], ops.Tensor))

  def testStateSaverWithTwoSimpleSteps(self):
    with self.test_session() as sess:
      batch_size_value = 2
      batch_size = constant_op.constant(batch_size_value)
      num_unroll = 2
      length = 3
      key = string_ops.string_join([
          "key_", string_ops.as_string(
              math_ops.cast(10000 * random_ops.random_uniform(()),
                            dtypes.int32))
      ])
      padded_length = 4
      sequences = {
          "seq1": np.random.rand(padded_length, 5),
          "seq2": np.random.rand(padded_length, 4, 2)
      }
      context = {"context1": [3, 4]}
      initial_states = {
          "state1": np.random.rand(6, 7),
          "state2": np.random.rand(8)
      }
      state_saver = sqss.SequenceQueueingStateSaver(
          batch_size=batch_size,
          num_unroll=num_unroll,
          input_length=length,
          input_key=key,
          input_sequences=sequences,
          input_context=context,
          initial_states=initial_states)

      initial_key_value_0, _ = sess.run((key, state_saver.prefetch_op))
      initial_key_value_1, _ = sess.run((key, state_saver.prefetch_op))

      initial_key_value_0 = initial_key_value_0.decode("ascii")
      initial_key_value_1 = initial_key_value_1.decode("ascii")

      # Step 1
      next_batch = state_saver.next_batch
      (key_value, next_key_value, seq1_value, seq2_value, context1_value,
       state1_value, state2_value, length_value, _, _) = sess.run(
           (next_batch.key, next_batch.next_key, next_batch.sequences["seq1"],
            next_batch.sequences["seq2"], next_batch.context["context1"],
            next_batch.state("state1"), next_batch.state("state2"),
            next_batch.length,
            next_batch.save_state("state1", next_batch.state("state1") + 1),
            next_batch.save_state("state2", next_batch.state("state2") - 1)))

      expected_first_keys = set(
          ("00000_of_00002:%s" % x).encode("ascii")
          for x in (initial_key_value_0, initial_key_value_1))
      expected_second_keys = set(
          ("00001_of_00002:%s" % x).encode("ascii")
          for x in (initial_key_value_0, initial_key_value_1))
      expected_final_keys = set(
          ("STOP:%s" % x).encode("ascii")
          for x in (initial_key_value_0, initial_key_value_1))

      self.assertEqual(set(key_value), expected_first_keys)
      self.assertEqual(set(next_key_value), expected_second_keys)
      self.assertAllEqual(context1_value,
                          np.tile(context["context1"], (batch_size_value, 1)))
      self.assertAllEqual(seq1_value,
                          np.tile(sequences["seq1"][np.newaxis, 0:2, :],
                                  (batch_size_value, 1, 1)))
      self.assertAllEqual(seq2_value,
                          np.tile(sequences["seq2"][np.newaxis, 0:2, :, :],
                                  (batch_size_value, 1, 1, 1)))
      self.assertAllEqual(state1_value,
                          np.tile(initial_states["state1"],
                                  (batch_size_value, 1, 1)))
      self.assertAllEqual(state2_value,
                          np.tile(initial_states["state2"],
                                  (batch_size_value, 1)))
      self.assertAllEqual(length_value, [2, 2])

      # Step 2
      (key_value, next_key_value, seq1_value, seq2_value, context1_value,
       state1_value, state2_value, length_value, _, _) = sess.run(
           (next_batch.key, next_batch.next_key, next_batch.sequences["seq1"],
            next_batch.sequences["seq2"], next_batch.context["context1"],
            next_batch.state("state1"), next_batch.state("state2"),
            next_batch.length,
            next_batch.save_state("state1", next_batch.state("state1") + 1),
            next_batch.save_state("state2", next_batch.state("state2") - 1)))

      self.assertEqual(set(key_value), expected_second_keys)
      self.assertEqual(set(next_key_value), expected_final_keys)
      self.assertAllEqual(context1_value,
                          np.tile(context["context1"], (batch_size_value, 1)))
      self.assertAllEqual(seq1_value,
                          np.tile(sequences["seq1"][np.newaxis, 2:4, :],
                                  (batch_size_value, 1, 1)))
      self.assertAllEqual(seq2_value,
                          np.tile(sequences["seq2"][np.newaxis, 2:4, :, :],
                                  (batch_size_value, 1, 1, 1)))
      self.assertAllEqual(state1_value, 1 + np.tile(initial_states["state1"],
                                                    (batch_size_value, 1, 1)))
      self.assertAllEqual(state2_value, -1 + np.tile(initial_states["state2"],
                                                     (batch_size_value, 1)))
      self.assertAllEqual(length_value, [1, 1])

      # Finished.  Let's make sure there's nothing left in the barrier.
      self.assertEqual(0, state_saver.barrier.ready_size().eval())

  def testStateSaverFailsIfPaddedLengthIsNotMultipleOfNumUnroll(self):
    with self.test_session() as sess:
      batch_size = constant_op.constant(32)
      num_unroll = 17
      bad_padded_length = 3
      length = array_ops.placeholder(dtypes.int32)
      key = array_ops.placeholder(dtypes.string)
      sequences = {
          "seq1": array_ops.placeholder(
              dtypes.float32, shape=(None, 5))
      }
      context = {}
      initial_states = {
          "state1": array_ops.placeholder(
              dtypes.float32, shape=())
      }
      state_saver = sqss.SequenceQueueingStateSaver(
          batch_size=batch_size,
          num_unroll=num_unroll,
          input_length=length,
          input_key=key,
          input_sequences=sequences,
          input_context=context,
          initial_states=initial_states)

      with self.assertRaisesOpError(
          "should be a multiple of: 17, but saw value: %d" % bad_padded_length):
        sess.run([state_saver.prefetch_op],
                 feed_dict={
                     length: 1,
                     key: "key",
                     sequences["seq1"]: np.random.rand(bad_padded_length, 5),
                     initial_states["state1"]: 1.0
                 })

  def testStateSaverFailsIfInconsistentPaddedLength(self):
    with self.test_session() as sess:
      batch_size = constant_op.constant(32)
      num_unroll = 17
      length = array_ops.placeholder(dtypes.int32)
      key = array_ops.placeholder(dtypes.string)
      sequences = {
          "seq1": array_ops.placeholder(
              dtypes.float32, shape=(None, 5)),
          "seq2": array_ops.placeholder(
              dtypes.float32, shape=(None,))
      }
      context = {}
      initial_states = {
          "state1": array_ops.placeholder(
              dtypes.float32, shape=())
      }
      state_saver = sqss.SequenceQueueingStateSaver(
          batch_size=batch_size,
          num_unroll=num_unroll,
          input_length=length,
          input_key=key,
          input_sequences=sequences,
          input_context=context,
          initial_states=initial_states)

      with self.assertRaisesOpError(
          "Dimension 0 of tensor labeled sorted_sequences_seq2 "
          "should be: %d, shape received: %d" % (num_unroll, 2 * num_unroll)):
        sess.run([state_saver.prefetch_op],
                 feed_dict={
                     length: 1,
                     key: "key",
                     sequences["seq1"]: np.random.rand(num_unroll, 5),
                     sequences["seq2"]: np.random.rand(2 * num_unroll),
                     initial_states["state1"]: 1.0
                 })

  def testStateSaverFailsIfInconsistentWriteState(self):
    # TODO(b/26910386): Identify why this infrequently causes timeouts.
    with self.test_session() as sess:
      batch_size = constant_op.constant(1)
      num_unroll = 17
      length = array_ops.placeholder(dtypes.int32)
      key = array_ops.placeholder(dtypes.string)
      sequences = {
          "seq1": array_ops.placeholder(
              dtypes.float32, shape=(None, 5))
      }
      context = {}
      initial_states = {
          "state1": array_ops.placeholder(
              dtypes.float32, shape=())
      }
      state_saver = sqss.SequenceQueueingStateSaver(
          batch_size=batch_size,
          num_unroll=num_unroll,
          input_length=length,
          input_key=key,
          input_sequences=sequences,
          input_context=context,
          initial_states=initial_states)
      next_batch = state_saver.next_batch
      with self.assertRaisesRegexp(KeyError, "state was not declared: state2"):
        save_op = next_batch.save_state("state2", None)
      with self.assertRaisesRegexp(ValueError, "Rank check failed for.*state1"):
        save_op = next_batch.save_state("state1", np.random.rand(1, 1))
      with self.assertRaisesOpError(
          r"convert_state1:0 should be: 1, shape received:\] \[1 1\]"):
        state_input = array_ops.placeholder(dtypes.float32)
        with ops.control_dependencies([state_saver.prefetch_op]):
          save_op = next_batch.save_state("state1", state_input)
        sess.run([save_op],
                 feed_dict={
                     length: 1,
                     key: "key",
                     sequences["seq1"]: np.random.rand(num_unroll, 5),
                     initial_states["state1"]: 1.0,
                     state_input: np.random.rand(1, 1)
                 })

  def testStateSaverWithManyInputsReadWriteThread(self):
    batch_size_value = 32
    num_proc_threads = 100
    with self.test_session() as sess:
      batch_size = constant_op.constant(batch_size_value)
      num_unroll = 17
      length = array_ops.placeholder(dtypes.int32)
      key = array_ops.placeholder(dtypes.string)
      sequences = {
          "seq1": array_ops.placeholder(
              dtypes.float32, shape=(None, 5)),
          "seq2": array_ops.placeholder(
              dtypes.float32, shape=(None, 4, 2)),
          "seq3": array_ops.placeholder(
              dtypes.float64, shape=(None,))
      }
      context = {
          "context1": array_ops.placeholder(
              dtypes.string, shape=(3, 4)),
          "context2": array_ops.placeholder(
              dtypes.int64, shape=())
      }
      initial_states = {
          "state1": array_ops.placeholder(
              dtypes.float32, shape=(6, 7)),
          "state2": array_ops.placeholder(
              dtypes.int32, shape=())
      }
      state_saver = sqss.SequenceQueueingStateSaver(
          batch_size=batch_size,
          num_unroll=num_unroll,
          input_length=length,
          input_key=key,
          input_sequences=sequences,
          input_context=context,
          initial_states=initial_states)
      next_batch = state_saver.next_batch
      cancel_op = state_saver.close(cancel_pending_enqueues=True)

      update_1 = next_batch.save_state("state1", 1 + next_batch.state("state1"))
      update_2 = next_batch.save_state("state2",
                                       -1 + next_batch.state("state2"))

      original_values = dict()

      def insert(which):
        for i in range(20):
          # Insert varying length inputs
          pad_i = num_unroll * (1 + (i % 10))
          length_i = int(np.random.rand() * pad_i)
          key_value = "key_%02d_%04d" % (which, i)
          stored_state = {
              "length": length_i,
              "seq1": np.random.rand(pad_i, 5),
              "seq2": np.random.rand(pad_i, 4, 2),
              "seq3": np.random.rand(pad_i),
              "context1": np.random.rand(3, 4).astype(np.str),
              "context2": np.asarray(
                  100 * np.random.rand(), dtype=np.int32),
              "state1": np.random.rand(6, 7),
              "state2": np.asarray(
                  100 * np.random.rand(), dtype=np.int32)
          }
          original_values[key_value] = stored_state
          sess.run([state_saver.prefetch_op],
                   feed_dict={
                       length: stored_state["length"],
                       key: key_value,
                       sequences["seq1"]: stored_state["seq1"],
                       sequences["seq2"]: stored_state["seq2"],
                       sequences["seq3"]: stored_state["seq3"],
                       context["context1"]: stored_state["context1"],
                       context["context2"]: stored_state["context2"],
                       initial_states["state1"]: stored_state["state1"],
                       initial_states["state2"]: stored_state["state2"]
                   })

      processed_count = [0]

      def process_and_check_state():
        next_batch = state_saver.next_batch
        while True:
          try:
            (got_key, next_key, length, total_length, sequence, sequence_count,
             context1, context2, seq1, seq2, seq3, state1, state2, _,
             _) = (sess.run([
                 next_batch.key, next_batch.next_key, next_batch.length,
                 next_batch.total_length, next_batch.sequence,
                 next_batch.sequence_count, next_batch.context["context1"],
                 next_batch.context["context2"], next_batch.sequences["seq1"],
                 next_batch.sequences["seq2"], next_batch.sequences["seq3"],
                 next_batch.state("state1"), next_batch.state("state2"),
                 update_1, update_2
             ]))

          except errors_impl.OutOfRangeError:
            # SQSS has been closed
            break

          self.assertEqual(len(got_key), batch_size_value)

          processed_count[0] += len(got_key)

          for i in range(batch_size_value):
            key_name = got_key[i].decode("ascii").split(":")[1]
            # We really saved this unique key
            self.assertTrue(key_name in original_values)
            # The unique key matches next_key
            self.assertEqual(key_name,
                             next_key[i].decode("ascii").split(":")[1])
            # Pull out the random values we used to create this example
            stored_state = original_values[key_name]
            self.assertEqual(total_length[i], stored_state["length"])
            self.assertEqual("%05d_of_%05d:%s" %
                             (sequence[i], sequence_count[i], key_name),
                             got_key[i].decode("ascii"))
            expected_length = max(
                0,
                min(num_unroll,
                    stored_state["length"] - sequence[i] * num_unroll))
            self.assertEqual(length[i], expected_length)
            expected_state1 = stored_state["state1"] + sequence[i]
            expected_state2 = stored_state["state2"] - sequence[i]
            expected_sequence1 = stored_state["seq1"][sequence[i] * num_unroll:(
                sequence[i] + 1) * num_unroll]
            expected_sequence2 = stored_state["seq2"][sequence[i] * num_unroll:(
                sequence[i] + 1) * num_unroll]
            expected_sequence3 = stored_state["seq3"][sequence[i] * num_unroll:(
                sequence[i] + 1) * num_unroll]

            self.assertAllClose(state1[i], expected_state1)
            self.assertAllEqual(state2[i], expected_state2)
            # context1 is strings, which come back as bytes
            self.assertAllEqual(context1[i].astype(np.str),
                                stored_state["context1"])
            self.assertAllEqual(context2[i], stored_state["context2"])
            self.assertAllClose(seq1[i], expected_sequence1)
            self.assertAllClose(seq2[i], expected_sequence2)
            self.assertAllClose(seq3[i], expected_sequence3)

      # Total number of inserts will be a multiple of batch_size
      insert_threads = [
          self.checkedThread(
              insert, args=(which,)) for which in range(batch_size_value)
      ]
      process_threads = [
          self.checkedThread(process_and_check_state)
          for _ in range(num_proc_threads)
      ]

      for t in insert_threads:
        t.start()
      for t in process_threads:
        t.start()
      for t in insert_threads:
        t.join()

      time.sleep(3)  # Allow the threads to run and process for a while
      cancel_op.run()

      for t in process_threads:
        t.join()

      # Each thread processed at least 2 sequence segments
      self.assertGreater(processed_count[0], 2 * 20 * batch_size_value)

  def testStateSaverProcessesExamplesInOrder(self):
    with self.test_session() as sess:
      batch_size_value = 32
      batch_size = constant_op.constant(batch_size_value)
      num_unroll = 17
      length = array_ops.placeholder(dtypes.int32)
      key = array_ops.placeholder(dtypes.string)
      sequences = {
          "seq1": array_ops.placeholder(
              dtypes.float32, shape=(None, 5))
      }
      context = {"context1": array_ops.placeholder(dtypes.string, shape=(3, 4))}
      initial_states = {
          "state1": array_ops.placeholder(
              dtypes.float32, shape=())
      }
      state_saver = sqss.SequenceQueueingStateSaver(
          batch_size=batch_size,
          num_unroll=num_unroll,
          input_length=length,
          input_key=key,
          input_sequences=sequences,
          input_context=context,
          initial_states=initial_states)
      next_batch = state_saver.next_batch

      update = next_batch.save_state("state1", 1 + next_batch.state("state1"))
      get_ready_size = state_saver.barrier.ready_size()
      get_incomplete_size = state_saver.barrier.incomplete_size()

      global_insert_key = [0]

      def insert(insert_key):
        # Insert varying length inputs
        sess.run([state_saver.prefetch_op],
                 feed_dict={
                     length: np.random.randint(2 * num_unroll),
                     key: "%05d" % insert_key[0],
                     sequences["seq1"]: np.random.rand(2 * num_unroll, 5),
                     context["context1"]: np.random.rand(3, 4).astype(np.str),
                     initial_states["state1"]: 0.0
                 })
        insert_key[0] += 1

      for _ in range(batch_size_value * 100):
        insert(global_insert_key)

      def process_and_validate(check_key):
        true_step = int(check_key[0] / 2)  # Each entry has two slices
        check_key[0] += 1
        got_keys, input_index, _ = sess.run(
            [next_batch.key, next_batch.insertion_index, update])
        decoded_keys = [int(x.decode("ascii").split(":")[-1]) for x in got_keys]
        min_key = min(decoded_keys)
        min_index = int(min(input_index))  # numpy scalar
        max_key = max(decoded_keys)
        max_index = int(max(input_index))  # numpy scalar
        # The current min key should be above the previous min
        self.assertEqual(min_key, true_step * batch_size_value)
        self.assertEqual(max_key, (true_step + 1) * batch_size_value - 1)
        self.assertEqual(2**63 + min_index, true_step * batch_size_value)
        self.assertEqual(2**63 + max_index,
                         (true_step + 1) * batch_size_value - 1)

      # There are now (batch_size * 100 * 2) / batch_size = 200 full steps
      global_step_key = [0]
      for _ in range(200):
        process_and_validate(global_step_key)

      # Processed everything in the queue
      self.assertEqual(get_incomplete_size.eval(), 0)
      self.assertEqual(get_ready_size.eval(), 0)

  def testStateSaverCanHandleVariableBatchsize(self):
    with self.test_session() as sess:
      batch_size = array_ops.placeholder(dtypes.int32)
      num_unroll = 17
      length = array_ops.placeholder(dtypes.int32)
      key = array_ops.placeholder(dtypes.string)
      sequences = {
          "seq1": array_ops.placeholder(
              dtypes.float32, shape=(None, 5))
      }
      context = {"context1": array_ops.placeholder(dtypes.string, shape=(3, 4))}
      initial_states = {
          "state1": array_ops.placeholder(
              dtypes.float32, shape=())
      }
      state_saver = sqss.SequenceQueueingStateSaver(
          batch_size=batch_size,
          num_unroll=num_unroll,
          input_length=length,
          input_key=key,
          input_sequences=sequences,
          input_context=context,
          initial_states=initial_states)
      next_batch = state_saver.next_batch

      update = next_batch.save_state("state1", 1 + next_batch.state("state1"))

      for insert_key in range(128):
        # Insert varying length inputs
        sess.run([state_saver.prefetch_op],
                 feed_dict={
                     length: np.random.randint(2 * num_unroll),
                     key: "%05d" % insert_key,
                     sequences["seq1"]: np.random.rand(2 * num_unroll, 5),
                     context["context1"]: np.random.rand(3, 4).astype(np.str),
                     initial_states["state1"]: 0.0
                 })

      all_received_indices = []
      # Pull out and validate batch sizes 0, 1, ..., 7
      for batch_size_value in range(8):
        got_keys, input_index, context1, seq1, state1, _ = sess.run(
            [
                next_batch.key, next_batch.insertion_index,
                next_batch.context["context1"], next_batch.sequences["seq1"],
                next_batch.state("state1"), update
            ],
            feed_dict={batch_size: batch_size_value})
        # Indices may have come in out of order within the batch
        all_received_indices.append(input_index.tolist())
        self.assertEqual(got_keys.size, batch_size_value)
        self.assertEqual(input_index.size, batch_size_value)
        self.assertEqual(context1.shape, (batch_size_value, 3, 4))
        self.assertEqual(seq1.shape, (batch_size_value, num_unroll, 5))
        self.assertEqual(state1.shape, (batch_size_value,))

      # Each input was split into 2 iterations (sequences size == 2*num_unroll)
      expected_indices = [[], [0], [0, 1], [1, 2, 3], [2, 3, 4, 5],
                          [4, 5, 6, 7, 8], [6, 7, 8, 9, 10, 11],
                          [9, 10, 11, 12, 13, 14, 15]]
      self.assertEqual(len(all_received_indices), len(expected_indices))
      for received, expected in zip(all_received_indices, expected_indices):
        self.assertAllEqual([x + 2**63 for x in received], expected)

  def testStateSaverScopeNames(self):
    batch_size = constant_op.constant(2)
    sqss_scope_name = "unique_scope_name_for_sqss"
    num_unroll = 2
    length = 3
    key = string_ops.string_join([
        "key_", string_ops.as_string(
            math_ops.cast(10000 * random_ops.random_uniform(()), dtypes.int32))
    ])
    padded_length = 4
    sequences = {
        "seq1": np.random.rand(padded_length, 5),
        "seq2": np.random.rand(padded_length, 4, 2)
    }
    context = {"context1": [3, 4]}
    initial_states = {
        "state1": np.random.rand(6, 7),
        "state2": np.random.rand(8)
    }
    state_saver = sqss.SequenceQueueingStateSaver(
        batch_size=batch_size,
        num_unroll=num_unroll,
        input_length=length,
        input_key=key,
        input_sequences=sequences,
        input_context=context,
        initial_states=initial_states,
        name=sqss_scope_name)
    prefetch_op = state_saver.prefetch_op
    next_batch = state_saver.next_batch
    self.assertTrue(
        state_saver.barrier.barrier_ref.name.startswith("%s/" %
                                                        sqss_scope_name))
    self.assertTrue(prefetch_op.name.startswith("%s/" % sqss_scope_name))
    self.assertTrue(next_batch.key.name.startswith("%s/" % sqss_scope_name))


if __name__ == "__main__":
  test.main()
