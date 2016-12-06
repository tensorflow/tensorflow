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

"""Tests for rnn module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.util import nest


class Plus1RNNCell(tf.contrib.rnn.RNNCell):
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    return 5

  @property
  def state_size(self):
    return 5

  def __call__(self, input_, state, scope=None):
    return (input_ + 1, state + 1)


class DummyMultiDimensionalLSTM(tf.contrib.rnn.RNNCell):
  """LSTM Cell generating (output, new_state) = (input + 1, state + 1).

  The input to this cell may have an arbitrary number of dimensions that follow
  the preceding 'Time' and 'Batch' dimensions.
  """

  def __init__(self, dims):
    """Initialize the Multi-dimensional LSTM cell.

    Args:
      dims: tuple that contains the dimensions of the output of the cell,
      without including 'Time' or 'Batch' dimensions.
    """
    if not isinstance(dims, tuple):
      raise TypeError("The dimensions passed to DummyMultiDimensionalLSTM"
                      "should be a tuple of ints.")
    self._dims = dims
    self._output_size = tf.TensorShape(self._dims)
    self._state_size = (tf.TensorShape(self._dims), tf.TensorShape(self._dims))

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def __call__(self, input_, state, scope=None):
    h, c = state
    return (input_ + 1, (h + 1, c + 1))


class NestedRNNCell(tf.contrib.rnn.RNNCell):
  """RNN Cell generating (output, new_state) = (input + 1, state + 1).

  The input, output and state of this cell is a tuple of two tensors.
  """

  @property
  def output_size(self):
    return (5, 5)

  @property
  def state_size(self):
    return (6, 6)

  def __call__(self, input_, state, scope=None):
    h, c = state
    x, y = input_
    return ((x + 1, y + 1), (h + 1, c + 1))


class TestStateSaver(object):

  def __init__(self, batch_size, state_size):
    self._batch_size = batch_size
    self._state_size = state_size
    self.saved_state = {}

  def state(self, name):

    if isinstance(self._state_size, dict):
      state_size = self._state_size[name]
    else:
      state_size = self._state_size
    if isinstance(state_size, int):
      state_size = (state_size,)
    elif isinstance(state_size, tuple):
      pass
    else:
      raise TypeError("state_size should either be an int or a tuple")

    return tf.zeros((self._batch_size,) + state_size)

  def save_state(self, name, state):
    self.saved_state[name] = state
    return tf.identity(state)


class RNNTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def testInvalidSequenceLengthShape(self):
    cell = Plus1RNNCell()
    inputs = [tf.placeholder(tf.float32, shape=(3, 4))]
    with self.assertRaisesRegexp(ValueError, "must be a vector"):
      tf.contrib.rnn.static_rnn(
          cell, inputs, dtype=tf.float32, sequence_length=4)

  def testRNN(self):
    cell = Plus1RNNCell()
    batch_size = 2
    input_size = 5
    max_length = 8  # unrolled up to this length
    inputs = max_length * [
        tf.placeholder(tf.float32, shape=(batch_size, input_size))]
    outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)
    self.assertEqual(len(outputs), len(inputs))
    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape(), inp.get_shape())
      self.assertEqual(out.dtype, inp.dtype)

    with self.test_session(use_gpu=False) as sess:
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs + [state],
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
    full_dropout_cell = tf.contrib.rnn.DropoutWrapper(
        cell, input_keep_prob=1e-12, seed=0)
    batch_size = 2
    input_size = 5
    max_length = 8
    inputs = max_length * [
        tf.placeholder(tf.float32, shape=(batch_size, input_size))]
    with tf.variable_scope("share_scope"):
      outputs, state = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)
    with tf.variable_scope("drop_scope"):
      dropped_outputs, _ = tf.contrib.rnn.static_rnn(
          full_dropout_cell, inputs, dtype=tf.float32)
    self.assertEqual(len(outputs), len(inputs))
    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape().as_list(), inp.get_shape().as_list())
      self.assertEqual(out.dtype, inp.dtype)

    with self.test_session(use_gpu=False) as sess:
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs + [state],
                        feed_dict={inputs[0]: input_value})
      full_dropout_values = sess.run(dropped_outputs,
                                     feed_dict={inputs[0]: input_value})

      for v in values[:-1]:
        self.assertAllClose(v, input_value + 1.0)
      for d_v in full_dropout_values[:-1]:  # Add 1.0 to dropped_out (all zeros)
        self.assertAllClose(d_v, np.ones_like(input_value))

  def _testDynamicCalculation(self, use_gpu):
    cell = Plus1RNNCell()
    sequence_length = tf.placeholder(tf.int64)
    batch_size = 2
    input_size = 5
    max_length = 8
    inputs = max_length * [
        tf.placeholder(tf.float32, shape=(batch_size, input_size))]
    with tf.variable_scope("drop_scope"):
      dynamic_outputs, dynamic_state = tf.contrib.rnn.static_rnn(
          cell, inputs, sequence_length=sequence_length, dtype=tf.float32)
    self.assertEqual(len(dynamic_outputs), len(inputs))

    with self.test_session(use_gpu=use_gpu) as sess:
      input_value = np.random.randn(batch_size, input_size)
      dynamic_values = sess.run(dynamic_outputs,
                                feed_dict={inputs[0]: input_value,
                                           sequence_length: [2, 3]})
      dynamic_state_value = sess.run([dynamic_state],
                                     feed_dict={inputs[0]: input_value,
                                                sequence_length: [2, 3]})

      # outputs are fully calculated for t = 0, 1
      for v in dynamic_values[:2]:
        self.assertAllClose(v, input_value + 1.0)

      # outputs at t = 2 are zero for entry 0, calculated for entry 1
      self.assertAllClose(
          dynamic_values[2],
          np.vstack((
              np.zeros((input_size)),
              1.0 + input_value[1, :])))

      # outputs at t = 3+ are zero
      for v in dynamic_values[3:]:
        self.assertAllEqual(v, np.zeros_like(input_value))

      # the final states are:
      #  entry 0: the values from the calculation at t=1
      #  entry 1: the values from the calculation at t=2
      self.assertAllEqual(
          dynamic_state_value[0],
          np.vstack((
              1.0 * (1 + 1) * np.ones((input_size)),
              1.0 * (2 + 1) * np.ones((input_size)))))

  def testDynamicCalculation(self):
    self._testDynamicCalculation(True)
    self._testDynamicCalculation(False)

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    with self.test_session(use_gpu=True, graph=tf.Graph()):
      if use_outer_scope:
        with tf.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)

      # check that all the variables names starts
      # with the proper scope.
      tf.global_variables_initializer()
      all_vars = tf.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf.logging.info("RNN with scope: %s (%s)"
                      % (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf.logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  def testScope(self):
    def factory(scope):
      cell = Plus1RNNCell()
      batch_size = 2
      input_size = 5
      max_length = 8  # unrolled up to this length
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      return tf.contrib.rnn.static_rnn(
          cell, inputs, dtype=tf.float32, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


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
      cell = tf.contrib.rnn.LSTMCell(num_units, initializer=initializer,
                                     state_is_tuple=False)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      outputs, _ = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testCellClipping(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      cell = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=True, cell_clip=0.0, initializer=initializer,
          state_is_tuple=False)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      outputs, _ = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.global_variables_initializer().run()
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
      cell = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=False, initializer=initializer,
          state_is_tuple=False)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      with tf.variable_scope("share_scope"):
        outputs, state = tf.contrib.rnn.static_state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name="save_lstm")
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      (last_state_value, saved_state_value) = sess.run(
          [state, state_saver.saved_state["save_lstm"]],
          feed_dict={inputs[0]: input_value})
      self.assertAllEqual(last_state_value, saved_state_value)

  def testNoProjNoShardingTupleStateSaver(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.test_session(graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(batch_size, num_units)
      cell = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=False, initializer=initializer,
          state_is_tuple=True)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      with tf.variable_scope("share_scope"):
        outputs, state = tf.contrib.rnn.static_state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name=("c", "m"))
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      last_and_saved_states = sess.run(
          state + (state_saver.saved_state["c"], state_saver.saved_state["m"]),
          feed_dict={inputs[0]: input_value})
      self.assertEqual(4, len(last_and_saved_states))
      self.assertAllEqual(last_and_saved_states[:2], last_and_saved_states[2:])

  def testNoProjNoShardingNestedTupleStateSaver(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.test_session(graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(batch_size, {"c0": num_units,
                                                "m0": num_units,
                                                "c1": num_units + 1,
                                                "m1": num_units + 1,
                                                "c2": num_units + 2,
                                                "m2": num_units + 2,
                                                "c3": num_units + 3,
                                                "m3": num_units + 3})
      def _cell(i):
        return tf.contrib.rnn.LSTMCell(
            num_units + i, use_peepholes=False, initializer=initializer,
            state_is_tuple=True)

      # This creates a state tuple which has 4 sub-tuples of length 2 each.
      cell = tf.contrib.rnn.MultiRNNCell(
          [_cell(i) for i in range(4)], state_is_tuple=True)

      self.assertEqual(len(cell.state_size), 4)
      for i in range(4):
        self.assertEqual(len(cell.state_size[i]), 2)

      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]

      state_names = (("c0", "m0"), ("c1", "m1"),
                     ("c2", "m2"), ("c3", "m3"))
      with tf.variable_scope("share_scope"):
        outputs, state = tf.contrib.rnn.static_state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name=state_names)
      self.assertEqual(len(outputs), len(inputs))

      # Final output comes from _cell(3) which has state size num_units + 3
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units + 3])

      tf.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      last_states = sess.run(
          list(nest.flatten(state)), feed_dict={inputs[0]: input_value})
      saved_states = sess.run(
          list(state_saver.saved_state.values()),
          feed_dict={inputs[0]: input_value})
      self.assertEqual(8, len(last_states))
      self.assertEqual(8, len(saved_states))
      flat_state_names = nest.flatten(state_names)
      named_saved_states = dict(
          zip(state_saver.saved_state.keys(), saved_states))

      for i in range(8):
        self.assertAllEqual(
            last_states[i],
            named_saved_states[flat_state_names[i]])

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
      cell = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer,
          state_is_tuple=False)
      outputs, _ = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), len(inputs))

      tf.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def testStateTupleWithProjAndSequenceLength(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    sequence_length = [4, 6]
    with self.test_session(graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(None, input_size))]
      cell_notuple = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer, state_is_tuple=False)
      cell_tuple = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer, state_is_tuple=True)
      with tf.variable_scope("root") as scope:
        outputs_notuple, state_notuple = tf.contrib.rnn.static_rnn(
            cell_notuple, inputs, dtype=tf.float32,
            sequence_length=sequence_length, scope=scope)
        scope.reuse_variables()
        outputs_tuple, state_tuple = tf.contrib.rnn.static_rnn(
            cell_tuple, inputs, dtype=tf.float32,
            sequence_length=sequence_length, scope=scope)
      self.assertEqual(len(outputs_notuple), len(inputs))
      self.assertEqual(len(outputs_tuple), len(inputs))
      self.assertTrue(isinstance(state_tuple, tuple))
      self.assertTrue(isinstance(state_notuple, tf.Tensor))

      tf.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      outputs_notuple_v = sess.run(
          outputs_notuple, feed_dict={inputs[0]: input_value})
      outputs_tuple_v = sess.run(
          outputs_tuple, feed_dict={inputs[0]: input_value})
      self.assertAllEqual(outputs_notuple_v, outputs_tuple_v)

      (state_notuple_v,) = sess.run(
          (state_notuple,), feed_dict={inputs[0]: input_value})
      state_tuple_v = sess.run(
          state_tuple, feed_dict={inputs[0]: input_value})
      self.assertAllEqual(state_notuple_v, np.hstack(state_tuple_v))

  def _testProjSharding(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)

      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(None, input_size))]

      cell = tf.contrib.rnn.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)

      outputs, _ = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)

      self.assertEqual(len(outputs), len(inputs))

      tf.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testDoubleInput(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      initializer = tf.random_uniform_initializer(-1, 1, seed=self._seed)
      inputs = max_length * [
          tf.placeholder(tf.float64, shape=(None, input_size))]

      cell = tf.contrib.rnn.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)

      outputs, _ = tf.contrib.rnn.static_rnn(
          cell, inputs, initial_state=cell.zero_state(batch_size, tf.float64))

      self.assertEqual(len(outputs), len(inputs))

      tf.global_variables_initializer().run()
      input_value = np.asarray(np.random.randn(batch_size, input_size),
                               dtype=np.float64)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value})
      self.assertEqual(values[0].dtype, input_value.dtype)

  def _testShardNoShardEquivalentOutput(self, use_gpu):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(None, input_size))]
      initializer = tf.constant_initializer(0.001)

      cell_noshard = tf.contrib.rnn.LSTMCell(
          num_units,
          num_proj=num_proj,
          use_peepholes=True,
          initializer=initializer,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          state_is_tuple=False)

      cell_shard = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=True,
          initializer=initializer, num_proj=num_proj,
          state_is_tuple=False)

      with tf.variable_scope("noshard_scope"):
        outputs_noshard, state_noshard = tf.contrib.rnn.static_rnn(
            cell_noshard, inputs, dtype=tf.float32)
      with tf.variable_scope("shard_scope"):
        outputs_shard, state_shard = tf.contrib.rnn.static_rnn(
            cell_shard, inputs, dtype=tf.float32)

      self.assertEqual(len(outputs_noshard), len(inputs))
      self.assertEqual(len(outputs_noshard), len(outputs_shard))

      tf.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      feeds = dict((x, input_value) for x in inputs)
      values_noshard = sess.run(outputs_noshard, feed_dict=feeds)
      values_shard = sess.run(outputs_shard, feed_dict=feeds)
      state_values_noshard = sess.run([state_noshard], feed_dict=feeds)
      state_values_shard = sess.run([state_shard], feed_dict=feeds)
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
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      sequence_length = tf.placeholder(tf.int64)
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      inputs = max_length * [
          tf.placeholder(tf.float64, shape=(None, input_size))]

      cell = tf.contrib.rnn.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)
      dropout_cell = tf.contrib.rnn.DropoutWrapper(cell, 0.5, seed=0)

      outputs, state = tf.contrib.rnn.static_rnn(
          dropout_cell, inputs, sequence_length=sequence_length,
          initial_state=cell.zero_state(batch_size, tf.float64))

      self.assertEqual(len(outputs), len(inputs))

      tf.global_variables_initializer().run(feed_dict={sequence_length: [2, 3]})
      input_value = np.asarray(np.random.randn(batch_size, input_size),
                               dtype=np.float64)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value,
                                            sequence_length: [2, 3]})
      state_value = sess.run([state], feed_dict={inputs[0]: input_value,
                                                 sequence_length: [2, 3]})
      self.assertEqual(values[0].dtype, input_value.dtype)
      self.assertEqual(state_value[0].dtype, input_value.dtype)

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
      cell = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer,
          state_is_tuple=False)
      cell_d = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer_d,
          state_is_tuple=False)

      with tf.variable_scope("share_scope"):
        outputs0, _ = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)
      with tf.variable_scope("share_scope", reuse=True):
        outputs1, _ = tf.contrib.rnn.static_rnn(cell, inputs, dtype=tf.float32)
      with tf.variable_scope("diff_scope"):
        outputs2, _ = tf.contrib.rnn.static_rnn(
            cell_d, inputs, dtype=tf.float32)

      tf.global_variables_initializer().run()
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
      cell = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer,
          state_is_tuple=False)

      with tf.name_scope("scope0"):
        with tf.variable_scope("share_scope"):
          outputs0, _ = tf.contrib.rnn.static_rnn(
              cell, inputs, dtype=tf.float32)
      with tf.name_scope("scope1"):
        with tf.variable_scope("share_scope", reuse=True):
          outputs1, _ = tf.contrib.rnn.static_rnn(
              cell, inputs, dtype=tf.float32)

      tf.global_variables_initializer().run()
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

  def _createBidirectionalRNN(self,
                              use_gpu,
                              use_shape,
                              use_sequence_length,
                              scope=None):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8

    initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
    sequence_length = tf.placeholder(tf.int64) if use_sequence_length else None
    cell_fw = tf.contrib.rnn.LSTMCell(num_units,
                                      input_size,
                                      initializer=initializer,
                                      state_is_tuple=False)
    cell_bw = tf.contrib.rnn.LSTMCell(num_units,
                                      input_size,
                                      initializer=initializer,
                                      state_is_tuple=False)
    inputs = max_length * [
        tf.placeholder(
            tf.float32,
            shape=(batch_size, input_size) if use_shape else (None, input_size))
    ]
    outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(
        cell_fw,
        cell_bw,
        inputs,
        dtype=tf.float32,
        sequence_length=sequence_length,
        scope=scope)
    self.assertEqual(len(outputs), len(inputs))
    for out in outputs:
      self.assertEqual(
          out.get_shape().as_list(),
          [batch_size if use_shape else None, 2 * num_units])

    input_value = np.random.randn(batch_size, input_size)
    outputs = tf.stack(outputs)

    return input_value, inputs, outputs, state_fw, state_bw, sequence_length

  def _testBidirectionalRNN(self, use_gpu, use_shape):
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createBidirectionalRNN(use_gpu, use_shape, True))
      tf.global_variables_initializer().run()
      # Run with pre-specified sequence length of 2, 3
      out, s_fw, s_bw = sess.run([outputs, state_fw, state_bw],
                                 feed_dict={inputs[0]: input_value,
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
      # Via the reasoning above, the forward and backward final state should be
      # exactly the same
      self.assertAllClose(s_fw, s_bw)

  def _testBidirectionalRNNWithoutSequenceLength(self, use_gpu, use_shape):
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, _ = (
          self._createBidirectionalRNN(use_gpu, use_shape, False))
      tf.global_variables_initializer().run()
      out, s_fw, s_bw = sess.run([outputs, state_fw, state_bw],
                                 feed_dict={inputs[0]: input_value})

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward output has to be the same,
      # but reversed in time. The format is output[time][batch][depth], and
      # due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      #
      # Both sequences in batch are length=8.  Check that the time=i
      # forward output is equal to time=8-1-i backward output
      for i in xrange(8):
        self.assertEqual(out[i][0][0], out[8 - 1 - i][0][3])
        self.assertEqual(out[i][0][1], out[8 - 1 - i][0][4])
        self.assertEqual(out[i][0][2], out[8 - 1 - i][0][5])
      for i in xrange(8):
        self.assertEqual(out[i][1][0], out[8 - 1 - i][1][3])
        self.assertEqual(out[i][1][1], out[8 - 1 - i][1][4])
        self.assertEqual(out[i][1][2], out[8 - 1 - i][1][5])
      # Via the reasoning above, the forward and backward final state should be
      # exactly the same
      self.assertAllClose(s_fw, s_bw)

  def testBidirectionalRNN(self):
    self._testBidirectionalRNN(use_gpu=False, use_shape=False)
    self._testBidirectionalRNN(use_gpu=True, use_shape=False)
    self._testBidirectionalRNN(use_gpu=False, use_shape=True)
    self._testBidirectionalRNN(use_gpu=True, use_shape=True)

  def testBidirectionalRNNWithoutSequenceLength(self):
    self._testBidirectionalRNNWithoutSequenceLength(use_gpu=False,
                                                    use_shape=False)
    self._testBidirectionalRNNWithoutSequenceLength(use_gpu=True,
                                                    use_shape=False)
    self._testBidirectionalRNNWithoutSequenceLength(use_gpu=False,
                                                    use_shape=True)
    self._testBidirectionalRNNWithoutSequenceLength(use_gpu=True,
                                                    use_shape=True)

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    # REMARKS: factory(scope) is a function accepting a scope
    #          as an argument, such scope can be None, a string
    #          or a VariableScope instance.
    with self.test_session(use_gpu=True, graph=tf.Graph()):
      if use_outer_scope:
        with tf.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)

      # check that all the variables names starts
      # with the proper scope.
      tf.global_variables_initializer()
      all_vars = tf.global_variables()
      prefix = prefix or "bidirectional_rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf.logging.info("BiRNN with scope: %s (%s)"
                      % (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf.logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  def testBidirectionalRNNScope(self):
    def factory(scope):
      return self._createBidirectionalRNN(
          use_gpu=True, use_shape=True,
          use_sequence_length=True, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


class MultiDimensionalLSTMTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def testMultiDimensionalLSTMAllRNNContainers(self):
    feature_dims = (3, 4, 5)
    input_size = feature_dims
    batch_size = 2
    max_length = 8
    sequence_length = [4, 6]
    with self.test_session(graph=tf.Graph()) as sess:
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(None,) + input_size)]
      inputs_using_dim = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size,) + input_size)]
      inputs_c = tf.stack(inputs)
      # Create a cell for the whole test. This is fine because the cell has no
      # variables.
      cell = DummyMultiDimensionalLSTM(feature_dims)
      state_saver = TestStateSaver(batch_size, input_size)
      outputs_static, state_static = tf.contrib.rnn.static_rnn(
          cell, inputs, dtype=tf.float32,
          sequence_length=sequence_length)
      outputs_bid, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(
          cell, cell, inputs_using_dim, dtype=tf.float32,
          sequence_length=sequence_length)
      outputs_sav, state_sav = tf.contrib.rnn.static_state_saving_rnn(
          cell, inputs_using_dim, sequence_length=sequence_length,
          state_saver=state_saver, state_name=("h", "c"))
      for out, inp in zip(outputs_static, inputs):
        self.assertEqual(out.get_shape().as_list(), inp.get_shape().as_list())
      for out, inp in zip(outputs_bid, inputs_using_dim):
        input_shape_list = inp.get_shape().as_list()
        # fwd and bwd activations are concatenated along the second dim.
        input_shape_list[1] *= 2
        self.assertEqual(out.get_shape().as_list(), input_shape_list)

      tf.global_variables_initializer().run()

      input_total_size = (batch_size,) + input_size
      input_value = np.random.randn(*input_total_size)
      outputs_static_v = sess.run(
          outputs_static, feed_dict={inputs[0]: input_value})
      outputs_bid_v = sess.run(
          outputs_bid, feed_dict={inputs_using_dim[0]: input_value})
      outputs_sav_v = sess.run(
          outputs_sav, feed_dict={inputs_using_dim[0]: input_value})

      self.assertAllEqual(outputs_static_v, outputs_sav_v)
      outputs_static_array = np.array(outputs_static_v)
      outputs_static_array_double = np.concatenate(
          (outputs_static_array, outputs_static_array), axis=2)
      outputs_bid_array = np.array(outputs_bid_v)
      self.assertAllEqual(outputs_static_array_double, outputs_bid_array)

      state_static_v = sess.run(
          state_static, feed_dict={inputs[0]: input_value})
      state_bid_fw_v = sess.run(
          state_fw, feed_dict={inputs_using_dim[0]: input_value})
      state_bid_bw_v = sess.run(
          state_bw, feed_dict={inputs_using_dim[0]: input_value})
      state_sav_v = sess.run(
          state_sav, feed_dict={inputs_using_dim[0]: input_value})
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_sav_v))
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_bid_fw_v))
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_bid_bw_v))


class NestedLSTMTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def testNestedIOLSTMAllRNNContainers(self):
    input_size = 5
    batch_size = 2
    state_size = 6
    max_length = 8
    sequence_length = [4, 6]
    with self.test_session(graph=tf.Graph()) as sess:
      state_saver = TestStateSaver(batch_size, state_size)
      single_input = (tf.placeholder(tf.float32, shape=(None, input_size)),
                      tf.placeholder(tf.float32, shape=(None, input_size)))
      inputs = max_length * [single_input]
      inputs_c = (tf.stack([input_[0] for input_ in inputs]),
                  tf.stack([input_[1] for input_ in inputs]))
      single_input_using_dim = (
          tf.placeholder(tf.float32, shape=(batch_size, input_size)),
          tf.placeholder(tf.float32, shape=(batch_size, input_size)))
      inputs_using_dim = max_length * [single_input_using_dim]

      # Create a cell for the whole test. This is fine because the cell has no
      # variables.
      cell = NestedRNNCell()
      outputs_static, state_static = tf.contrib.rnn.static_rnn(
          cell, inputs, dtype=tf.float32,
          sequence_length=sequence_length)
      outputs_bid, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(
          cell, cell, inputs_using_dim, dtype=tf.float32,
          sequence_length=sequence_length)
      outputs_sav, state_sav = tf.contrib.rnn.static_state_saving_rnn(
          cell, inputs_using_dim, sequence_length=sequence_length,
          state_saver=state_saver, state_name=("h", "c"))

      def _assert_same_shape(input1, input2, double=False):
        flat_input1 = nest.flatten(input1)
        flat_input2 = nest.flatten(input2)
        for inp1, inp2 in zip(flat_input1, flat_input2):
          input_shape = inp1.get_shape().as_list()
          if double:
            input_shape[1] *= 2
          self.assertEqual(input_shape, inp2.get_shape().as_list())

      _assert_same_shape(inputs, outputs_static)
      _assert_same_shape(inputs_using_dim, outputs_sav)
      _assert_same_shape(inputs_using_dim, outputs_bid, double=True)

      tf.global_variables_initializer().run()

      input_total_size = (batch_size, input_size)
      input_value = (np.random.randn(*input_total_size),
                     np.random.randn(*input_total_size))
      outputs_static_v = sess.run(
          outputs_static, feed_dict={single_input: input_value})
      outputs_sav_v = sess.run(
          outputs_sav, feed_dict={single_input_using_dim: input_value})
      outputs_bid_v = sess.run(
          outputs_bid, feed_dict={single_input_using_dim: input_value})

      self.assertAllEqual(outputs_static_v, outputs_sav_v)
      outputs_static_array = np.array(outputs_static_v)
      outputs_static_array_double = np.concatenate(
          (outputs_static_array, outputs_static_array), axis=3)
      outputs_bid_array = np.array(outputs_bid_v)
      self.assertAllEqual(outputs_static_array_double, outputs_bid_array)

      state_static_v = sess.run(
          state_static, feed_dict={single_input: input_value})
      state_bid_fw_v = sess.run(
          state_fw, feed_dict={single_input_using_dim: input_value})
      state_bid_bw_v = sess.run(
          state_bw, feed_dict={single_input_using_dim: input_value})
      state_sav_v = sess.run(
          state_sav, feed_dict={single_input_using_dim: input_value})
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_sav_v))
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_bid_fw_v))
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_bid_bw_v))


class StateSaverRNNTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    with self.test_session(use_gpu=True, graph=tf.Graph()):
      if use_outer_scope:
        with tf.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)
        tf.global_variables_initializer()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = tf.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf.logging.info("RNN with scope: %s (%s)"
                      % (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf.logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  def testStateSaverRNNScope(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    def factory(scope):
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(batch_size, 2 * num_units)
      cell = tf.contrib.rnn.LSTMCell(
          num_units, use_peepholes=False, initializer=initializer,
          state_is_tuple=False)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      return tf.contrib.rnn.static_state_saving_rnn(
          cell, inputs, state_saver=state_saver,
          state_name="save_lstm", scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)

if __name__ == "__main__":
  tf.test.main()
