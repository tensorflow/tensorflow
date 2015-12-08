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

"""Tests for rnn module."""

# pylint: disable=g-bad-import-order,unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class Plus1RNNCell(tf.nn.rnn_cell.RNNCell):
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    return 5

  @property
  def state_size(self):
    return 5

  def __call__(self, input_, state):
    return (input_ + 1, state + 1)


class TestStateSaver(object):

  def __init__(self, batch_size, state_size):
    self._batch_size = batch_size
    self._state_size = state_size

  def state(self, _):
    return tf.zeros(tf.pack([self._batch_size, self._state_size]))

  def save_state(self, _, state):
    self.saved_state = state
    return tf.identity(state)


class RNNTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def testRNN(self):
    cell = Plus1RNNCell()
    batch_size = 2
    input_size = 5
    max_length = 8  # unrolled up to this length
    inputs = max_length * [
        tf.placeholder(tf.float32, shape=(batch_size, input_size))]
    outputs, states = tf.nn.rnn(cell, inputs, dtype=tf.float32)
    self.assertEqual(len(outputs), len(inputs))
    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape(), inp.get_shape())
      self.assertEqual(out.dtype, inp.dtype)

    with self.test_session(use_gpu=False) as sess:
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs + [states[-1]],
                        feed_dict={inputs[0]: input_value})

      # Outputs
      for v in values[:-1]:
        self.assertAllClose(v, input_value + 1.0)

      # Final state
      self.assertAllClose(
          values[-1],
          max_length * np.ones((batch_size, input_size), dtype=np.float32))

  def testDropout(self):
    cell = Plus1RNNCell()
    full_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, input_keep_prob=1e-12, seed=0)
    batch_size = 2
    input_size = 5
    max_length = 8
    inputs = max_length * [
        tf.placeholder(tf.float32, shape=(batch_size, input_size))]
    with tf.variable_scope("share_scope"):
      outputs, states = tf.nn.rnn(cell, inputs, dtype=tf.float32)
    with tf.variable_scope("drop_scope"):
      dropped_outputs, _ = tf.nn.rnn(
          full_dropout_cell, inputs, dtype=tf.float32)
    self.assertEqual(len(outputs), len(inputs))
    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape().as_list(), inp.get_shape().as_list())
      self.assertEqual(out.dtype, inp.dtype)

    with self.test_session(use_gpu=False) as sess:
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs + [states[-1]],
                        feed_dict={inputs[0]: input_value})
      full_dropout_values = sess.run(dropped_outputs,
                                     feed_dict={inputs[0]: input_value})

      for v in values[:-1]:
        self.assertAllClose(v, input_value + 1.0)
      for d_v in full_dropout_values[:-1]:  # Add 1.0 to dropped_out (all zeros)
        self.assertAllClose(d_v, np.ones_like(input_value))

  def testDynamicCalculation(self):
    cell = Plus1RNNCell()
    sequence_length = tf.placeholder(tf.int64)
    batch_size = 2
    input_size = 5
    max_length = 8
    inputs = max_length * [
        tf.placeholder(tf.float32, shape=(batch_size, input_size))]
    with tf.variable_scope("drop_scope"):
      dynamic_outputs, dynamic_states = tf.nn.rnn(
          cell, inputs, sequence_length=sequence_length, dtype=tf.float32)
    self.assertEqual(len(dynamic_outputs), len(inputs))
    self.assertEqual(len(dynamic_states), len(inputs))

    with self.test_session(use_gpu=False) as sess:
      input_value = np.random.randn(batch_size, input_size)
      dynamic_values = sess.run(dynamic_outputs,
                                feed_dict={inputs[0]: input_value,
                                           sequence_length: [2, 3]})
      dynamic_state_values = sess.run(dynamic_states,
                                      feed_dict={inputs[0]: input_value,
                                                 sequence_length: [2, 3]})

      # fully calculated for t = 0, 1, 2
      for v in dynamic_values[:3]:
        self.assertAllClose(v, input_value + 1.0)
      for vi, v in enumerate(dynamic_state_values[:3]):
        self.assertAllEqual(v, 1.0 * (vi + 1) *
                            np.ones((batch_size, input_size)))
      # zeros for t = 3+
      for v in dynamic_values[3:]:
        self.assertAllEqual(v, np.zeros_like(input_value))
      for v in dynamic_state_values[3:]:
        self.assertAllEqual(v, np.zeros_like(input_value))


class LSTMTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _testNoProjNoSharding(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size, initializer=initializer)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      outputs, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testCellClipping(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          cell_clip=0.0, initializer=initializer)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      outputs, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value})

    for value in values:
      # if cell c is clipped to 0, tanh(c) = 0 => m==0
      self.assertAllEqual(value, np.zeros((batch_size, num_units)))

  def _testNoProjNoShardingSimpleStateSaver(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(batch_size, 2 * num_units)
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=False, initializer=initializer)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      with tf.variable_scope("share_scope"):
        outputs, states = tf.nn.state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name="save_lstm")
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      (last_state_value, saved_state_value) = sess.run(
          [states[-1], state_saver.saved_state],
          feed_dict={inputs[0]: input_value})
      self.assertAllEqual(last_state_value, saved_state_value)

  def _testProjNoSharding(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(None, input_size))]
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          num_proj=num_proj, initializer=initializer)
      outputs, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), len(inputs))

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testProjSharding(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 4
    num_unit_shards = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)

      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(None, input_size))]

      cell = tf.nn.rnn_cell.LSTMCell(
          num_units,
          input_size=input_size,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer)

      outputs, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)

      self.assertEqual(len(outputs), len(inputs))

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testDoubleInput(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 4
    num_unit_shards = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-1, 1, seed=self._seed)
      inputs = max_length * [tf.placeholder(tf.float64)]

      cell = tf.nn.rnn_cell.LSTMCell(
          num_units,
          input_size=input_size,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer)

      outputs, _ = tf.nn.rnn(
          cell, inputs, initial_state=cell.zero_state(batch_size, tf.float64))

      self.assertEqual(len(outputs), len(inputs))

      tf.initialize_all_variables().run()
      input_value = np.asarray(np.random.randn(batch_size, input_size),
                               dtype=np.float64)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value})
      self.assertEqual(values[0].dtype, input_value.dtype)

  def _testShardNoShardEquivalentOutput(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 4
    num_unit_shards = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      inputs = max_length * [tf.placeholder(tf.float32)]
      initializer = tf.constant_initializer(0.001)

      cell_noshard = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size,
          num_proj=num_proj,
          use_peepholes=True,
          initializer=initializer,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards)

      cell_shard = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          initializer=initializer, num_proj=num_proj)

      with tf.variable_scope("noshard_scope"):
        outputs_noshard, states_noshard = tf.nn.rnn(
            cell_noshard, inputs, dtype=tf.float32)
      with tf.variable_scope("shard_scope"):
        outputs_shard, states_shard = tf.nn.rnn(
            cell_shard, inputs, dtype=tf.float32)

      self.assertEqual(len(outputs_noshard), len(inputs))
      self.assertEqual(len(outputs_noshard), len(outputs_shard))

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      feeds = dict((x, input_value) for x in inputs)
      values_noshard = sess.run(outputs_noshard, feed_dict=feeds)
      values_shard = sess.run(outputs_shard, feed_dict=feeds)
      state_values_noshard = sess.run(states_noshard, feed_dict=feeds)
      state_values_shard = sess.run(states_shard, feed_dict=feeds)
      self.assertEqual(len(values_noshard), len(values_shard))
      self.assertEqual(len(state_values_noshard), len(state_values_shard))
      for (v_noshard, v_shard) in zip(values_noshard, values_shard):
        self.assertAllClose(v_noshard, v_shard, atol=1e-3)
      for (s_noshard, s_shard) in zip(state_values_noshard, state_values_shard):
        self.assertAllClose(s_noshard, s_shard, atol=1e-3)

  def _testDoubleInputWithDropoutAndDynamicCalculation(
      self, use_gpu):
    """Smoke test for using LSTM with doubles, dropout, dynamic calculation."""

    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 4
    num_unit_shards = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      sequence_length = tf.placeholder(tf.int64)
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      inputs = max_length * [tf.placeholder(tf.float64)]

      cell = tf.nn.rnn_cell.LSTMCell(
          num_units,
          input_size=input_size,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer)
      dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, 0.5, seed=0)

      outputs, states = tf.nn.rnn(
          dropout_cell, inputs, sequence_length=sequence_length,
          initial_state=cell.zero_state(batch_size, tf.float64))

      self.assertEqual(len(outputs), len(inputs))
      self.assertEqual(len(outputs), len(states))

      tf.initialize_all_variables().run(feed_dict={sequence_length: [2, 3]})
      input_value = np.asarray(np.random.randn(batch_size, input_size),
                               dtype=np.float64)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value,
                                            sequence_length: [2, 3]})
      state_values = sess.run(states, feed_dict={inputs[0]: input_value,
                                                 sequence_length: [2, 3]})
      self.assertEqual(values[0].dtype, input_value.dtype)
      self.assertEqual(state_values[0].dtype, input_value.dtype)

  def testSharingWeightsWithReuse(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    with self.test_session(graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-1, 1, seed=self._seed)
      initializer_d = tf.random_uniform_initializer(-1, 1, seed=self._seed+1)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(None, input_size))]
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          num_proj=num_proj, initializer=initializer)
      cell_d = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          num_proj=num_proj, initializer=initializer_d)

      with tf.variable_scope("share_scope"):
        outputs0, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)
      with tf.variable_scope("share_scope", reuse=True):
        outputs1, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)
      with tf.variable_scope("diff_scope"):
        outputs2, _ = tf.nn.rnn(cell_d, inputs, dtype=tf.float32)

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      output_values = sess.run(
          outputs0 + outputs1 + outputs2, feed_dict={inputs[0]: input_value})
      outputs0_values = output_values[:max_length]
      outputs1_values = output_values[max_length:2*max_length]
      outputs2_values = output_values[2*max_length:]
      self.assertEqual(len(outputs0_values), len(outputs1_values))
      self.assertEqual(len(outputs0_values), len(outputs2_values))
      for o1, o2, o3 in zip(outputs0_values, outputs1_values, outputs2_values):
        # Same weights used by both RNNs so outputs should be the same.
        self.assertAllEqual(o1, o2)
        # Different weights used so outputs should be different.
        self.assertTrue(np.linalg.norm(o1-o3) > 1e-6)

  def testSharingWeightsWithDifferentNamescope(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    with self.test_session(graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-1, 1, seed=self._seed)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(None, input_size))]
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size, use_peepholes=True,
          num_proj=num_proj, initializer=initializer)

      with tf.name_scope("scope0"):
        with tf.variable_scope("share_scope"):
          outputs0, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)
      with tf.name_scope("scope1"):
        with tf.variable_scope("share_scope", reuse=True):
          outputs1, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      output_values = sess.run(
          outputs0 + outputs1, feed_dict={inputs[0]: input_value})
      outputs0_values = output_values[:max_length]
      outputs1_values = output_values[max_length:]
      self.assertEqual(len(outputs0_values), len(outputs1_values))
      for out0, out1 in zip(outputs0_values, outputs1_values):
        self.assertAllEqual(out0, out1)

  def testNoProjNoShardingSimpleStateSaver(self):
    self._testNoProjNoShardingSimpleStateSaver(use_gpu=False)
    self._testNoProjNoShardingSimpleStateSaver(use_gpu=True)

  def testNoProjNoSharding(self):
    self._testNoProjNoSharding(use_gpu=False)
    self._testNoProjNoSharding(use_gpu=True)

  def testCellClipping(self):
    self._testCellClipping(use_gpu=False)
    self._testCellClipping(use_gpu=True)

  def testProjNoSharding(self):
    self._testProjNoSharding(use_gpu=False)
    self._testProjNoSharding(use_gpu=True)

  def testProjSharding(self):
    self._testProjSharding(use_gpu=False)
    self._testProjSharding(use_gpu=True)

  def testShardNoShardEquivalentOutput(self):
    self._testShardNoShardEquivalentOutput(use_gpu=False)
    self._testShardNoShardEquivalentOutput(use_gpu=True)

  def testDoubleInput(self):
    self._testDoubleInput(use_gpu=False)
    self._testDoubleInput(use_gpu=True)

  def testDoubleInputWithDropoutAndDynamicCalculation(self):
    self._testDoubleInputWithDropoutAndDynamicCalculation(use_gpu=False)
    self._testDoubleInputWithDropoutAndDynamicCalculation(use_gpu=True)


class BidirectionalRNNTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _testBidirectionalRNN(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      sequence_length = tf.placeholder(tf.int64)
      cell_fw = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size, initializer=initializer)
      cell_bw = tf.nn.rnn_cell.LSTMCell(
          num_units, input_size, initializer=initializer)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      outputs = tf.nn.bidirectional_rnn(
          cell_fw, cell_bw, inputs, dtype=tf.float32,
          sequence_length=sequence_length)

      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, 2 * num_units])

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      # Run with pre-specified sequence length of 2, 3
      out = sess.run(outputs, feed_dict={inputs[0]: input_value,
                                         sequence_length: [2, 3]})

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward output has to be the same,
      # but reversed in time. The format is output[time][batch][depth], and
      # due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      #
      # First sequence in batch is length=2
      # Check that the time=0 forward output is equal to time=1 backward output
      self.assertEqual(out[0][0][0], out[1][0][3])
      self.assertEqual(out[0][0][1], out[1][0][4])
      self.assertEqual(out[0][0][2], out[1][0][5])
      # Check that the time=1 forward output is equal to time=0 backward output
      self.assertEqual(out[1][0][0], out[0][0][3])
      self.assertEqual(out[1][0][1], out[0][0][4])
      self.assertEqual(out[1][0][2], out[0][0][5])

      # Second sequence in batch is length=3
      # Check that the time=0 forward output is equal to time=2 backward output
      self.assertEqual(out[0][1][0], out[2][1][3])
      self.assertEqual(out[0][1][1], out[2][1][4])
      self.assertEqual(out[0][1][2], out[2][1][5])
      # Check that the time=1 forward output is equal to time=1 backward output
      self.assertEqual(out[1][1][0], out[1][1][3])
      self.assertEqual(out[1][1][1], out[1][1][4])
      self.assertEqual(out[1][1][2], out[1][1][5])
      # Check that the time=2 forward output is equal to time=0 backward output
      self.assertEqual(out[2][1][0], out[0][1][3])
      self.assertEqual(out[2][1][1], out[0][1][4])
      self.assertEqual(out[2][1][2], out[0][1][5])

  def testBidirectionalRNN(self):
    self._testBidirectionalRNN(use_gpu=False)
    self._testBidirectionalRNN(use_gpu=True)


if __name__ == "__main__":
  tf.test.main()
