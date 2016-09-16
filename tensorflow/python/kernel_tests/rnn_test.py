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

import itertools
import time
import timeit

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.util import nest


class Plus1RNNCell(tf.nn.rnn_cell.RNNCell):
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    return 5

  @property
  def state_size(self):
    return 5

  def __call__(self, input_, state, scope=None):
    return (input_ + 1, state + 1)


class DummyMultiDimensionalLSTM(tf.nn.rnn_cell.RNNCell):
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


class NestedRNNCell(tf.nn.rnn_cell.RNNCell):
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

  def testRNN(self):
    cell = Plus1RNNCell()
    batch_size = 2
    input_size = 5
    max_length = 8  # unrolled up to this length
    inputs = max_length * [
        tf.placeholder(tf.float32, shape=(batch_size, input_size))]
    outputs, state = tf.nn.rnn(cell, inputs, dtype=tf.float32)
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
    full_dropout_cell = tf.nn.rnn_cell.DropoutWrapper(
        cell, input_keep_prob=1e-12, seed=0)
    batch_size = 2
    input_size = 5
    max_length = 8
    inputs = max_length * [
        tf.placeholder(tf.float32, shape=(batch_size, input_size))]
    with tf.variable_scope("share_scope"):
      outputs, state = tf.nn.rnn(cell, inputs, dtype=tf.float32)
    with tf.variable_scope("drop_scope"):
      dropped_outputs, _ = tf.nn.rnn(
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
      dynamic_outputs, dynamic_state = tf.nn.rnn(
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
      tf.initialize_all_variables()
      all_vars = tf.all_variables()
      prefix = prefix or "RNN"
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
      return tf.nn.rnn(cell, inputs, dtype=tf.float32, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


class GRUTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _testDynamic(self, use_gpu):
    time_steps = 8
    num_units = 3
    input_size = 5
    batch_size = 2

    input_values = np.random.randn(time_steps, batch_size, input_size)

    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      concat_inputs = tf.placeholder(
          tf.float32, shape=(time_steps, batch_size, input_size))

      cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)

      with tf.variable_scope("dynamic_scope"):
        outputs_dynamic, state_dynamic = tf.nn.dynamic_rnn(
            cell, inputs=concat_inputs, sequence_length=sequence_length,
            time_major=True, dtype=tf.float32)

      feeds = {concat_inputs: input_values}

      # Initialize
      tf.initialize_all_variables().run(feed_dict=feeds)

      sess.run([outputs_dynamic, state_dynamic], feed_dict=feeds)

  def testDynamic(self):
    self._testDynamic(use_gpu=False)
    self._testDynamic(use_gpu=True)

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    with self.test_session(use_gpu=True, graph=tf.Graph()):
      if use_outer_scope:
        with tf.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)
        tf.initialize_all_variables()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = tf.all_variables()
      prefix = prefix or "RNN"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf.logging.info("RNN with scope: %s (%s)"
                      % (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf.logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  def testDynamicScope(self):
    time_steps = 8
    num_units = 3
    input_size = 5
    batch_size = 2
    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    def factory(scope):
      concat_inputs = tf.placeholder(
          tf.float32, shape=(time_steps, batch_size, input_size))
      cell = tf.nn.rnn_cell.GRUCell(num_units=num_units)
      return tf.nn.dynamic_rnn(cell, inputs=concat_inputs,
                               sequence_length=sequence_length,
                               time_major=True, dtype=tf.float32,
                               scope=scope)

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
      cell = tf.nn.rnn_cell.LSTMCell(num_units, initializer=initializer,
                                     state_is_tuple=False)
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
          num_units, use_peepholes=True, cell_clip=0.0, initializer=initializer,
          state_is_tuple=False)
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
          num_units, use_peepholes=False, initializer=initializer,
          state_is_tuple=False)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      with tf.variable_scope("share_scope"):
        outputs, state = tf.nn.state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name="save_lstm")
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.initialize_all_variables().run()
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
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=False, initializer=initializer,
          state_is_tuple=True)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      with tf.variable_scope("share_scope"):
        outputs, state = tf.nn.state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name=("c", "m"))
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      tf.initialize_all_variables().run()
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
        return tf.nn.rnn_cell.LSTMCell(
            num_units + i, use_peepholes=False, initializer=initializer,
            state_is_tuple=True)

      # This creates a state tuple which has 4 sub-tuples of length 2 each.
      cell = tf.nn.rnn_cell.MultiRNNCell(
          [_cell(i) for i in range(4)], state_is_tuple=True)

      self.assertEqual(len(cell.state_size), 4)
      for i in range(4):
        self.assertEqual(len(cell.state_size[i]), 2)

      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]

      state_names = (("c0", "m0"), ("c1", "m1"),
                     ("c2", "m2"), ("c3", "m3"))
      with tf.variable_scope("share_scope"):
        outputs, state = tf.nn.state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name=state_names)
      self.assertEqual(len(outputs), len(inputs))

      # Final output comes from _cell(3) which has state size num_units + 3
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units + 3])

      tf.initialize_all_variables().run()
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
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer,
          state_is_tuple=False)
      outputs, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)
      self.assertEqual(len(outputs), len(inputs))

      tf.initialize_all_variables().run()
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
      cell_notuple = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer, state_is_tuple=False)
      cell_tuple = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer, state_is_tuple=True)
      outputs_notuple, state_notuple = tf.nn.rnn(
          cell_notuple, inputs, dtype=tf.float32,
          sequence_length=sequence_length)
      tf.get_variable_scope().reuse_variables()
      outputs_tuple, state_tuple = tf.nn.rnn(
          cell_tuple, inputs, dtype=tf.float32,
          sequence_length=sequence_length)
      self.assertEqual(len(outputs_notuple), len(inputs))
      self.assertEqual(len(outputs_tuple), len(inputs))
      self.assertTrue(isinstance(state_tuple, tuple))
      self.assertTrue(isinstance(state_notuple, tf.Tensor))

      tf.initialize_all_variables().run()
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

      cell = tf.nn.rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)

      outputs, _ = tf.nn.rnn(cell, inputs, dtype=tf.float32)

      self.assertEqual(len(outputs), len(inputs))

      tf.initialize_all_variables().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testTooManyShards(self, use_gpu):
    num_units = 3
    input_size = 5
    num_proj = 4
    num_proj_shards = 4
    num_unit_shards = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()):
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)

      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(None, input_size))]

      cell = tf.nn.rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)

      with self.assertRaises(ValueError):
        tf.nn.rnn(cell, inputs, dtype=tf.float32)

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

      cell = tf.nn.rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)

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
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(None, input_size))]
      initializer = tf.constant_initializer(0.001)

      cell_noshard = tf.nn.rnn_cell.LSTMCell(
          num_units,
          num_proj=num_proj,
          use_peepholes=True,
          initializer=initializer,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          state_is_tuple=False)

      cell_shard = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=True,
          initializer=initializer, num_proj=num_proj,
          state_is_tuple=False)

      with tf.variable_scope("noshard_scope"):
        outputs_noshard, state_noshard = tf.nn.rnn(
            cell_noshard, inputs, dtype=tf.float32)
      with tf.variable_scope("shard_scope"):
        outputs_shard, state_shard = tf.nn.rnn(
            cell_shard, inputs, dtype=tf.float32)

      self.assertEqual(len(outputs_noshard), len(inputs))
      self.assertEqual(len(outputs_noshard), len(outputs_shard))

      tf.initialize_all_variables().run()
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

      cell = tf.nn.rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)
      dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell, 0.5, seed=0)

      outputs, state = tf.nn.rnn(
          dropout_cell, inputs, sequence_length=sequence_length,
          initial_state=cell.zero_state(batch_size, tf.float64))

      self.assertEqual(len(outputs), len(inputs))

      tf.initialize_all_variables().run(feed_dict={sequence_length: [2, 3]})
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
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer,
          state_is_tuple=False)
      cell_d = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer_d,
          state_is_tuple=False)

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
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer,
          state_is_tuple=False)

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

  def testDynamicRNNWithTupleStates(self):
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
      inputs_c = tf.pack(inputs)
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=True,
          num_proj=num_proj, initializer=initializer, state_is_tuple=True)
      outputs_static, state_static = tf.nn.rnn(
          cell, inputs, dtype=tf.float32,
          sequence_length=sequence_length)
      tf.get_variable_scope().reuse_variables()
      outputs_dynamic, state_dynamic = tf.nn.dynamic_rnn(
          cell, inputs_c, dtype=tf.float32, time_major=True,
          sequence_length=sequence_length)
      self.assertTrue(isinstance(state_static, tf.nn.rnn_cell.LSTMStateTuple))
      self.assertTrue(isinstance(state_dynamic, tf.nn.rnn_cell.LSTMStateTuple))
      self.assertEqual(state_static[0], state_static.c)
      self.assertEqual(state_static[1], state_static.h)
      self.assertEqual(state_dynamic[0], state_dynamic.c)
      self.assertEqual(state_dynamic[1], state_dynamic.h)

      tf.initialize_all_variables().run()

      input_value = np.random.randn(batch_size, input_size)
      outputs_static_v = sess.run(
          outputs_static, feed_dict={inputs[0]: input_value})
      outputs_dynamic_v = sess.run(
          outputs_dynamic, feed_dict={inputs[0]: input_value})
      self.assertAllEqual(outputs_static_v, outputs_dynamic_v)

      state_static_v = sess.run(
          state_static, feed_dict={inputs[0]: input_value})
      state_dynamic_v = sess.run(
          state_dynamic, feed_dict={inputs[0]: input_value})
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_dynamic_v))

  def testDynamicRNNWithNestedTupleStates(self):
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
      inputs_c = tf.pack(inputs)
      def _cell(i):
        return tf.nn.rnn_cell.LSTMCell(
            num_units + i, use_peepholes=True,
            num_proj=num_proj + i, initializer=initializer, state_is_tuple=True)

      # This creates a state tuple which has 4 sub-tuples of length 2 each.
      cell = tf.nn.rnn_cell.MultiRNNCell(
          [_cell(i) for i in range(4)], state_is_tuple=True)

      self.assertEqual(len(cell.state_size), 4)
      for i in range(4):
        self.assertEqual(len(cell.state_size[i]), 2)

      test_zero = cell.zero_state(1, tf.float32)
      self.assertEqual(len(test_zero), 4)
      for i in range(4):
        self.assertEqual(test_zero[i][0].get_shape()[1], cell.state_size[i][0])
        self.assertEqual(test_zero[i][1].get_shape()[1], cell.state_size[i][1])

      outputs_static, state_static = tf.nn.rnn(
          cell, inputs, dtype=tf.float32,
          sequence_length=sequence_length)
      tf.get_variable_scope().reuse_variables()
      outputs_dynamic, state_dynamic = tf.nn.dynamic_rnn(
          cell, inputs_c, dtype=tf.float32, time_major=True,
          sequence_length=sequence_length)

      tf.initialize_all_variables().run()

      input_value = np.random.randn(batch_size, input_size)
      outputs_static_v = sess.run(
          outputs_static, feed_dict={inputs[0]: input_value})
      outputs_dynamic_v = sess.run(
          outputs_dynamic, feed_dict={inputs[0]: input_value})
      self.assertAllEqual(outputs_static_v, outputs_dynamic_v)

      state_static_v = sess.run(
          nest.flatten(state_static), feed_dict={inputs[0]: input_value})
      state_dynamic_v = sess.run(
          nest.flatten(state_dynamic), feed_dict={inputs[0]: input_value})
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_dynamic_v))

  def _testDynamicEquivalentToStaticRNN(self, use_gpu, use_sequence_length):
    time_steps = 8
    num_units = 3
    num_proj = 4
    input_size = 5
    batch_size = 2

    input_values = np.random.randn(time_steps, batch_size, input_size)

    if use_sequence_length:
      sequence_length = np.random.randint(0, time_steps, size=batch_size)
    else:
      sequence_length = None

    ########### Step 1: Run static graph and generate readouts
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      concat_inputs = tf.placeholder(tf.float32,
                                     shape=(time_steps, batch_size, input_size))
      inputs = tf.unpack(concat_inputs)
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)

      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=True,
          initializer=initializer, num_proj=num_proj, state_is_tuple=False)

      with tf.variable_scope("dynamic_scope"):
        outputs_static, state_static = tf.nn.rnn(
            cell, inputs, sequence_length=sequence_length, dtype=tf.float32)

      feeds = {concat_inputs: input_values}

      # Initialize
      tf.initialize_all_variables().run(feed_dict=feeds)

      # Generate gradients of sum of outputs w.r.t. inputs
      static_gradients = tf.gradients(
          outputs_static + [state_static], [concat_inputs])

      # Generate gradients of individual outputs w.r.t. inputs
      static_individual_gradients = nest.flatten([
          tf.gradients(y, [concat_inputs])
          for y in [outputs_static[0],
                    outputs_static[-1],
                    state_static]])

      # Generate gradients of individual variables w.r.t. inputs
      trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      assert len(trainable_variables) > 1, (
          "Count of trainable variables: %d" % len(trainable_variables))
      # pylint: disable=bad-builtin
      static_individual_variable_gradients = nest.flatten([
          tf.gradients(y, trainable_variables)
          for y in [outputs_static[0],
                    outputs_static[-1],
                    state_static]])

      # Test forward pass
      values_static = sess.run(outputs_static, feed_dict=feeds)
      (state_value_static,) = sess.run((state_static,), feed_dict=feeds)

      # Test gradients to inputs and variables w.r.t. outputs & final state
      static_grad_values = sess.run(static_gradients, feed_dict=feeds)

      static_individual_grad_values = sess.run(
          static_individual_gradients, feed_dict=feeds)

      static_individual_var_grad_values = sess.run(
          static_individual_variable_gradients, feed_dict=feeds)

    ########## Step 2: Run dynamic graph and generate readouts
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      concat_inputs = tf.placeholder(tf.float32,
                                     shape=(time_steps, batch_size, input_size))
      inputs = tf.unpack(concat_inputs)
      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)

      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=True,
          initializer=initializer, num_proj=num_proj, state_is_tuple=False)

      with tf.variable_scope("dynamic_scope"):
        outputs_dynamic, state_dynamic = tf.nn.dynamic_rnn(
            cell, inputs=concat_inputs, sequence_length=sequence_length,
            time_major=True, dtype=tf.float32)
        split_outputs_dynamic = tf.unpack(outputs_dynamic, time_steps)

      feeds = {concat_inputs: input_values}

      # Initialize
      tf.initialize_all_variables().run(feed_dict=feeds)

      # Generate gradients of sum of outputs w.r.t. inputs
      dynamic_gradients = tf.gradients(
          split_outputs_dynamic + [state_dynamic], [concat_inputs])

      # Generate gradients of several individual outputs w.r.t. inputs
      dynamic_individual_gradients = nest.flatten([
          tf.gradients(y, [concat_inputs])
          for y in [split_outputs_dynamic[0],
                    split_outputs_dynamic[-1],
                    state_dynamic]])

      # Generate gradients of individual variables w.r.t. inputs
      trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
      assert len(trainable_variables) > 1, (
          "Count of trainable variables: %d" % len(trainable_variables))
      dynamic_individual_variable_gradients = nest.flatten([
          tf.gradients(y, trainable_variables)
          for y in [split_outputs_dynamic[0],
                    split_outputs_dynamic[-1],
                    state_dynamic]])

      # Test forward pass
      values_dynamic = sess.run(split_outputs_dynamic, feed_dict=feeds)
      (state_value_dynamic,) = sess.run(
          (state_dynamic,), feed_dict=feeds)

      # Test gradients to inputs and variables w.r.t. outputs & final state
      dynamic_grad_values = sess.run(dynamic_gradients, feed_dict=feeds)

      dynamic_individual_grad_values = sess.run(
          dynamic_individual_gradients, feed_dict=feeds)

      dynamic_individual_var_grad_values = sess.run(
          dynamic_individual_variable_gradients, feed_dict=feeds)

    ######### Step 3: Comparisons
    self.assertEqual(len(values_static), len(values_dynamic))
    for (value_static, value_dynamic) in zip(values_static, values_dynamic):
      self.assertAllEqual(value_static, value_dynamic)
    self.assertAllEqual(state_value_static, state_value_dynamic)

    self.assertAllEqual(static_grad_values, dynamic_grad_values)

    self.assertEqual(len(static_individual_grad_values),
                     len(dynamic_individual_grad_values))
    self.assertEqual(len(static_individual_var_grad_values),
                     len(dynamic_individual_var_grad_values))

    for i, (a, b) in enumerate(zip(static_individual_grad_values,
                                   dynamic_individual_grad_values)):
      tf.logging.info("Comparing individual gradients iteration %d" % i)
      self.assertAllEqual(a, b)

    for i, (a, b) in enumerate(zip(static_individual_var_grad_values,
                                   dynamic_individual_var_grad_values)):
      tf.logging.info(
          "Comparing individual variable gradients iteration %d" % i)
      self.assertAllEqual(a, b)

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

  def testTooManyShards(self):
    self._testTooManyShards(use_gpu=False)
    self._testTooManyShards(use_gpu=True)

  def testShardNoShardEquivalentOutput(self):
    self._testShardNoShardEquivalentOutput(use_gpu=False)
    self._testShardNoShardEquivalentOutput(use_gpu=True)

  def testDoubleInput(self):
    self._testDoubleInput(use_gpu=False)
    self._testDoubleInput(use_gpu=True)

  def testDoubleInputWithDropoutAndDynamicCalculation(self):
    self._testDoubleInputWithDropoutAndDynamicCalculation(use_gpu=False)
    self._testDoubleInputWithDropoutAndDynamicCalculation(use_gpu=True)

  def testDynamicEquivalentToStaticRNN(self):
    self._testDynamicEquivalentToStaticRNN(
        use_gpu=False, use_sequence_length=False)
    self._testDynamicEquivalentToStaticRNN(
        use_gpu=True, use_sequence_length=False)
    self._testDynamicEquivalentToStaticRNN(
        use_gpu=False, use_sequence_length=True)
    self._testDynamicEquivalentToStaticRNN(
        use_gpu=True, use_sequence_length=True)


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
    cell_fw = tf.nn.rnn_cell.LSTMCell(num_units,
                                      input_size,
                                      initializer=initializer,
                                      state_is_tuple=False)
    cell_bw = tf.nn.rnn_cell.LSTMCell(num_units,
                                      input_size,
                                      initializer=initializer,
                                      state_is_tuple=False)
    inputs = max_length * [
        tf.placeholder(
            tf.float32,
            shape=(batch_size, input_size) if use_shape else (None, input_size))
    ]
    outputs, state_fw, state_bw = tf.nn.bidirectional_rnn(
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
    outputs = tf.pack(outputs)

    return input_value, inputs, outputs, state_fw, state_bw, sequence_length

  def _testBidirectionalRNN(self, use_gpu, use_shape):
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createBidirectionalRNN(use_gpu, use_shape, True))
      tf.initialize_all_variables().run()
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
      tf.initialize_all_variables().run()
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

  def _createBidirectionalDynamicRNN(self, use_gpu, use_shape,
                                     use_state_tuple, use_time_major,
                                     scope=None):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8

    initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
    sequence_length = tf.placeholder(tf.int64)
    cell_fw = tf.nn.rnn_cell.LSTMCell(num_units,
                                      initializer=initializer,
                                      state_is_tuple=use_state_tuple)
    cell_bw = tf.nn.rnn_cell.LSTMCell(num_units,
                                      initializer=initializer,
                                      state_is_tuple=use_state_tuple)
    inputs = max_length * [
        tf.placeholder(tf.float32,
                       shape=(batch_size if use_shape else None, input_size))]
    inputs_c = tf.pack(inputs)
    if not use_time_major:
      inputs_c = tf.transpose(inputs_c, [1, 0, 2])
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        inputs_c,
        sequence_length,
        dtype=tf.float32,
        time_major=use_time_major,
        scope=scope)
    outputs = tf.concat(2, outputs)
    state_fw, state_bw = states
    outputs_shape = [None, max_length, 2 * num_units]
    if use_shape:
      outputs_shape[0] = batch_size
    if use_time_major:
      outputs_shape[0], outputs_shape[1] = outputs_shape[1], outputs_shape[0]
    self.assertEqual(
        outputs.get_shape().as_list(),
        outputs_shape)

    input_value = np.random.randn(batch_size, input_size)

    return input_value, inputs, outputs, state_fw, state_bw, sequence_length

  def _testBidirectionalDynamicRNN(self, use_gpu, use_shape,
                                   use_state_tuple, use_time_major):
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createBidirectionalDynamicRNN(
              use_gpu, use_shape, use_state_tuple, use_time_major))
      tf.initialize_all_variables().run()
      # Run with pre-specified sequence length of 2, 3
      if use_state_tuple:
        out, c_fw, m_fw, c_bw, m_bw = sess.run(
            [outputs, state_fw[0], state_fw[1], state_bw[0], state_bw[1]],
            feed_dict={inputs[0]: input_value,
                       sequence_length: [2, 3]})
        s_fw = (c_fw, m_fw)
        s_bw = (c_bw, m_bw)
      else:
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
      if not use_time_major:
        out = np.swapaxes(out, 0, 1)
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

  def testBidirectionalDynamicRNN(self):
    # Generate 2^4 option values
    # from [True, True, True, True] to [False, False, False, False]
    options = itertools.product([True, False], repeat=4)
    for option in options:
      self._testBidirectionalDynamicRNN(use_gpu=option[0], use_shape=option[1],
                                       use_state_tuple=option[2],
                                       use_time_major=option[3])

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
      tf.initialize_all_variables()
      all_vars = tf.all_variables()
      prefix = prefix or "BiRNN"
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

  def testBidirectionalDynamicRNNScope(self):
    def get_factory(use_time_major):
      def factory(scope):
        return self._createBidirectionalDynamicRNN(
            use_gpu=True, use_shape=True, use_state_tuple=True,
            use_time_major=use_time_major, scope=scope)
      return factory

    self._testScope(get_factory(True), use_outer_scope=True)
    self._testScope(get_factory(True), use_outer_scope=False)
    self._testScope(get_factory(True), prefix=None, use_outer_scope=False)
    self._testScope(get_factory(False), use_outer_scope=True)
    self._testScope(get_factory(False), use_outer_scope=False)
    self._testScope(get_factory(False), prefix=None, use_outer_scope=False)


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
      inputs_c = tf.pack(inputs)
      # Create a cell for the whole test. This is fine because the cell has no
      # variables.
      cell = DummyMultiDimensionalLSTM(feature_dims)
      state_saver = TestStateSaver(batch_size, input_size)
      outputs_static, state_static = tf.nn.rnn(
          cell, inputs, dtype=tf.float32,
          sequence_length=sequence_length)
      outputs_dynamic, state_dynamic = tf.nn.dynamic_rnn(
          cell, inputs_c, dtype=tf.float32, time_major=True,
          sequence_length=sequence_length)
      outputs_bid, state_bid_fw, state_bid_bw = tf.nn.bidirectional_rnn(
          cell, cell, inputs_using_dim, dtype=tf.float32,
          sequence_length=sequence_length)
      outputs_sav, state_sav = tf.nn.state_saving_rnn(
          cell, inputs_using_dim, sequence_length=sequence_length,
          state_saver=state_saver, state_name=("h", "c"))
      for out, inp in zip(outputs_static, inputs):
        self.assertEqual(out.get_shape().as_list(), inp.get_shape().as_list())
      self.assertEqual(outputs_dynamic.get_shape().as_list(),
                       inputs_c.get_shape().as_list())
      for out, inp in zip(outputs_bid, inputs_using_dim):
        input_shape_list = inp.get_shape().as_list()
        # fwd and bwd activations are concatenated along the second dim.
        input_shape_list[1] *= 2
        self.assertEqual(out.get_shape().as_list(), input_shape_list)

      tf.initialize_all_variables().run()

      input_total_size = (batch_size,) + input_size
      input_value = np.random.randn(*input_total_size)
      outputs_static_v = sess.run(
          outputs_static, feed_dict={inputs[0]: input_value})
      outputs_dynamic_v = sess.run(
          outputs_dynamic, feed_dict={inputs[0]: input_value})
      outputs_bid_v = sess.run(
          outputs_bid, feed_dict={inputs_using_dim[0]: input_value})
      outputs_sav_v = sess.run(
          outputs_sav, feed_dict={inputs_using_dim[0]: input_value})

      self.assertAllEqual(outputs_static_v, outputs_dynamic_v)
      self.assertAllEqual(outputs_static_v, outputs_sav_v)
      outputs_static_array = np.array(outputs_static_v)
      outputs_static_array_double = np.concatenate(
          (outputs_static_array, outputs_static_array), axis=2)
      outputs_bid_array = np.array(outputs_bid_v)
      self.assertAllEqual(outputs_static_array_double, outputs_bid_array)

      state_static_v = sess.run(
          state_static, feed_dict={inputs[0]: input_value})
      state_dynamic_v = sess.run(
          state_dynamic, feed_dict={inputs[0]: input_value})
      state_bid_fw_v = sess.run(
          state_bid_fw, feed_dict={inputs_using_dim[0]: input_value})
      state_bid_bw_v = sess.run(
          state_bid_bw, feed_dict={inputs_using_dim[0]: input_value})
      state_sav_v = sess.run(
          state_sav, feed_dict={inputs_using_dim[0]: input_value})
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_dynamic_v))
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
      inputs_c = (tf.pack([input_[0] for input_ in inputs]),
                  tf.pack([input_[1] for input_ in inputs]))
      single_input_using_dim = (
          tf.placeholder(tf.float32, shape=(batch_size, input_size)),
          tf.placeholder(tf.float32, shape=(batch_size, input_size)))
      inputs_using_dim = max_length * [single_input_using_dim]

      # Create a cell for the whole test. This is fine because the cell has no
      # variables.
      cell = NestedRNNCell()
      outputs_dynamic, state_dynamic = tf.nn.dynamic_rnn(
          cell, inputs_c, dtype=tf.float32, time_major=True,
          sequence_length=sequence_length)
      outputs_static, state_static = tf.nn.rnn(
          cell, inputs, dtype=tf.float32,
          sequence_length=sequence_length)
      outputs_bid, state_bid_fw, state_bid_bw = tf.nn.bidirectional_rnn(
          cell, cell, inputs_using_dim, dtype=tf.float32,
          sequence_length=sequence_length)
      outputs_sav, state_sav = tf.nn.state_saving_rnn(
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

      _assert_same_shape(inputs_c, outputs_dynamic)
      _assert_same_shape(inputs, outputs_static)
      _assert_same_shape(inputs_using_dim, outputs_sav)
      _assert_same_shape(inputs_using_dim, outputs_bid, double=True)

      tf.initialize_all_variables().run()

      input_total_size = (batch_size, input_size)
      input_value = (np.random.randn(*input_total_size),
                     np.random.randn(*input_total_size))
      outputs_dynamic_v = sess.run(
          outputs_dynamic, feed_dict={single_input: input_value})
      outputs_static_v = sess.run(
          outputs_static, feed_dict={single_input: input_value})
      outputs_sav_v = sess.run(
          outputs_sav, feed_dict={single_input_using_dim: input_value})
      outputs_bid_v = sess.run(
          outputs_bid, feed_dict={single_input_using_dim: input_value})

      self.assertAllEqual(outputs_static_v,
                          np.transpose(outputs_dynamic_v, (1, 0, 2, 3)))
      self.assertAllEqual(outputs_static_v, outputs_sav_v)
      outputs_static_array = np.array(outputs_static_v)
      outputs_static_array_double = np.concatenate(
          (outputs_static_array, outputs_static_array), axis=3)
      outputs_bid_array = np.array(outputs_bid_v)
      self.assertAllEqual(outputs_static_array_double, outputs_bid_array)

      state_dynamic_v = sess.run(
          state_dynamic, feed_dict={single_input: input_value})
      state_static_v = sess.run(
          state_static, feed_dict={single_input: input_value})
      state_bid_fw_v = sess.run(
          state_bid_fw, feed_dict={single_input_using_dim: input_value})
      state_bid_bw_v = sess.run(
          state_bid_bw, feed_dict={single_input_using_dim: input_value})
      state_sav_v = sess.run(
          state_sav, feed_dict={single_input_using_dim: input_value})
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_dynamic_v))
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_sav_v))
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_bid_fw_v))
      self.assertAllEqual(
          np.hstack(state_static_v), np.hstack(state_bid_bw_v))


class RawRNNTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _testRawRNN(self, max_time):
    with self.test_session(graph=tf.Graph()) as sess:
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
                              dtype=tf.float32)
      sequence_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
      inputs_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(inputs)[0])
      inputs_ta = inputs_ta.unpack(inputs)

      cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, unused_loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:  # time == 0
          next_state = cell.zero_state(batch_size, tf.float32)
        else:
          next_state = cell_state  # copy state through
        elements_finished = (time_ >= sequence_length)
        finished = tf.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
      outputs = outputs_ta.pack()

      tf.get_variable_scope().reuse_variables()
      outputs_dynamic_rnn, final_state_dynamic_rnn = tf.nn.dynamic_rnn(
          cell, inputs, time_major=True, dtype=tf.float32,
          sequence_length=sequence_length)

      variables = tf.trainable_variables()
      gradients = tf.gradients([outputs, final_state], [inputs] + variables)
      gradients_dynamic_rnn = tf.gradients(
          [outputs_dynamic_rnn, final_state_dynamic_rnn], [inputs] + variables)

      tf.initialize_all_variables().run()

      rand_input = np.random.randn(max_time, batch_size, input_depth)
      if max_time == 0:
        rand_seq_len = np.zeros(batch_size)
      else:
        rand_seq_len = np.random.randint(max_time, size=batch_size)

      # To ensure same output lengths for dynamic_rnn and raw_rnn
      rand_seq_len[0] = max_time

      (outputs_val, outputs_dynamic_rnn_val,
       final_state_val, final_state_dynamic_rnn_val) = sess.run(
           [outputs, outputs_dynamic_rnn, final_state, final_state_dynamic_rnn],
           feed_dict={inputs: rand_input, sequence_length: rand_seq_len})

      self.assertAllClose(outputs_dynamic_rnn_val, outputs_val)
      self.assertAllClose(final_state_dynamic_rnn_val, final_state_val)

      # NOTE: Because with 0 time steps, raw_rnn does not have shape
      # information about the input, it is impossible to perform
      # gradients comparisons as the gradients eval will fail.  So
      # this case skips the gradients test.
      if max_time > 0:
        self.assertEqual(len(gradients), len(gradients_dynamic_rnn))
        gradients_val = sess.run(
            gradients,
            feed_dict={inputs: rand_input, sequence_length: rand_seq_len})
        gradients_dynamic_rnn_val = sess.run(
            gradients_dynamic_rnn,
            feed_dict={inputs: rand_input, sequence_length: rand_seq_len})
        self.assertEqual(len(gradients_val), len(gradients_dynamic_rnn_val))
        input_gradients_val = gradients_val[0]
        input_gradients_dynamic_rnn_val = gradients_dynamic_rnn_val[0]
        self.assertAllClose(
            input_gradients_val, input_gradients_dynamic_rnn_val)
        for i in range(1, len(gradients_val)):
          self.assertAllClose(gradients_dynamic_rnn_val[i], gradients_val[i])

  def testRawRNNZeroLength(self):
    # NOTE: Because with 0 time steps, raw_rnn does not have shape
    # information about the input, it is impossible to perform
    # gradients comparisons as the gradients eval will fail.  So this
    # case skips the gradients test.
    self._testRawRNN(max_time=0)

  def testRawRNN(self):
    self._testRawRNN(max_time=10)

  def testLoopState(self):
    with self.test_session(graph=tf.Graph()):
      max_time = 10
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(inputs)[0])
      inputs_ta = inputs_ta.unpack(inputs)

      cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, loop_state):
        if cell_output is None:
          loop_state = tf.constant([0])
          next_state = cell.zero_state(batch_size, tf.float32)
        else:
          loop_state = tf.pack([tf.squeeze(loop_state) + 1])
          next_state = cell_state
        emit_output = cell_output  # == None for time == 0
        elements_finished = tf.tile([time_ >= max_time], [batch_size])
        finished = tf.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input,
                next_state, emit_output, loop_state)

      r = tf.nn.raw_rnn(cell, loop_fn)
      loop_state = r[-1]
      self.assertEqual([10], loop_state.eval())

  def testLoopStateWithTensorArray(self):
    with self.test_session(graph=tf.Graph()):
      max_time = 4
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(inputs)[0])
      inputs_ta = inputs_ta.unpack(inputs)

      cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
      def loop_fn(time_, cell_output, cell_state, loop_state):
        if cell_output is None:
          loop_state = tf.TensorArray(
              dynamic_size=True, size=0, dtype=tf.int32, clear_after_read=False)
          loop_state = loop_state.write(0, 1)
          next_state = cell.zero_state(batch_size, tf.float32)
        else:
          loop_state = loop_state.write(
              time_, loop_state.read(time_ - 1) + time_)
          next_state = cell_state
        emit_output = cell_output  # == None for time == 0
        elements_finished = tf.tile([time_ >= max_time], [batch_size])
        finished = tf.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input,
                next_state, emit_output, loop_state)

      r = tf.nn.raw_rnn(cell, loop_fn)
      loop_state = r[-1]
      loop_state = loop_state.pack()
      self.assertAllEqual([1, 2, 2 + 2, 4 + 3, 7 + 4], loop_state.eval())

  def testEmitDifferentStructureThanCellOutput(self):
    with self.test_session(graph=tf.Graph()) as sess:
      max_time = 10
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(inputs)[0])
      inputs_ta = inputs_ta.unpack(inputs)

      cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
      def loop_fn(time_, cell_output, cell_state, _):
        if cell_output is None:
          emit_output = (tf.zeros([2, 3], dtype=tf.int32),
                         tf.zeros([1], dtype=tf.int64))
          next_state = cell.zero_state(batch_size, tf.float32)
        else:
          emit_output = (tf.ones([batch_size, 2, 3], dtype=tf.int32),
                         tf.ones([batch_size, 1], dtype=tf.int64))
          next_state = cell_state
        elements_finished = tf.tile([time_ >= max_time], [batch_size])
        finished = tf.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      r = tf.nn.raw_rnn(cell, loop_fn)
      output_ta = r[0]
      self.assertEqual(2, len(output_ta))
      self.assertEqual([tf.int32, tf.int64], [ta.dtype for ta in output_ta])
      output = [ta.pack() for ta in output_ta]
      output_vals = sess.run(output)
      self.assertAllEqual(
          np.ones((max_time, batch_size, 2, 3), np.int32), output_vals[0])
      self.assertAllEqual(
          np.ones((max_time, batch_size, 1), np.int64), output_vals[1])

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    with self.test_session(use_gpu=True, graph=tf.Graph()):
      if use_outer_scope:
        with tf.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)
        tf.initialize_all_variables()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = tf.all_variables()
      prefix = prefix or "RNN"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf.logging.info("RNN with scope: %s (%s)"
                      % (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf.logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  def testRawRNNScope(self):
    max_time = 10
    batch_size = 16
    input_depth = 4
    num_units = 3

    def factory(scope):
      inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
                              dtype=tf.float32)
      sequence_length = tf.placeholder(shape=(batch_size,), dtype=tf.int32)
      inputs_ta = tf.TensorArray(dtype=tf.float32, size=tf.shape(inputs)[0])
      inputs_ta = inputs_ta.unpack(inputs)

      cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=True)
      def loop_fn(time_, cell_output, cell_state, unused_loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:  # time == 0
          next_state = cell.zero_state(batch_size, tf.float32)
        else:
          next_state = cell_state

        elements_finished = (time_ >= sequence_length)
        finished = tf.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = tf.cond(
            finished,
            lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      return tf.nn.raw_rnn(cell, loop_fn, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


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
        tf.initialize_all_variables()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = tf.all_variables()
      prefix = prefix or "RNN"
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
      cell = tf.nn.rnn_cell.LSTMCell(
          num_units, use_peepholes=False, initializer=initializer,
          state_is_tuple=False)
      inputs = max_length * [
          tf.placeholder(tf.float32, shape=(batch_size, input_size))]
      return tf.nn.state_saving_rnn(
          cell, inputs, state_saver=state_saver,
          state_name="save_lstm", scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)

######### Benchmarking RNN code

def _static_vs_dynamic_rnn_benchmark_static(inputs_list_t, sequence_length):
  (_, input_size) = inputs_list_t[0].get_shape().as_list()
  initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = tf.nn.rnn_cell.LSTMCell(
      num_units=input_size, use_peepholes=True, initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = tf.nn.rnn(
      cell, inputs_list_t, sequence_length=sequence_length, dtype=tf.float32)

  trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  gradients = tf.gradients(outputs + [final_state], trainable_variables)

  return tf.group(final_state, *(gradients + outputs))


def _static_vs_dynamic_rnn_benchmark_dynamic(inputs_t, sequence_length):
  (unused_0, unused_1, input_size) = inputs_t.get_shape().as_list()
  initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = tf.nn.rnn_cell.LSTMCell(
      num_units=input_size, use_peepholes=True, initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = tf.nn.dynamic_rnn(
      cell, inputs_t, sequence_length=sequence_length, dtype=tf.float32)

  trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  gradients = tf.gradients([outputs, final_state], trainable_variables)

  return tf.group(final_state, outputs, *gradients)


def graph_creation_static_vs_dynamic_rnn_benchmark(max_time):
  config = tf.ConfigProto()
  config.allow_soft_placement = True

  # These parameters don't matter
  batch_size = 512
  num_units = 512

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  def _create_static_rnn():
    with tf.Session(config=config, graph=tf.Graph()) as sess:
      inputs_list_t = [
          tf.Variable(x, trainable=False).value() for x in inputs_list]
      ops = _static_vs_dynamic_rnn_benchmark_static(
          inputs_list_t, sequence_length)

  def _create_dynamic_rnn():
    with tf.Session(config=config, graph=tf.Graph()) as sess:
      inputs_t = tf.Variable(inputs, trainable=False).value()
      ops = _static_vs_dynamic_rnn_benchmark_dynamic(
          inputs_t, sequence_length)

  delta_static = timeit.timeit(_create_static_rnn, number=5)
  delta_dynamic = timeit.timeit(_create_dynamic_rnn, number=5)

  print("%d \t %f \t %f \t %f" %
        (max_time, delta_static, delta_dynamic, delta_dynamic/delta_static))
  return delta_static, delta_dynamic


def _timer(sess, ops):
  # Warm in
  for _ in range(2):
    sess.run(ops)

  # Timing run
  runs = 20
  start = time.time()
  for _ in range(runs):
    sess.run(ops)
  end = time.time()
  return (end - start)/float(runs)


def static_vs_dynamic_rnn_benchmark(batch_size, max_time, num_units, use_gpu):
  config = tf.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  # Using rnn()
  with tf.Session(config=config, graph=tf.Graph()) as sess:
    with tf.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          tf.Variable(x, trainable=False).value() for x in inputs_list]
      ops = _static_vs_dynamic_rnn_benchmark_static(
          inputs_list_t, sequence_length)
    tf.initialize_all_variables().run()
    delta_static = _timer(sess, ops)

  # Using dynamic_rnn()
  with tf.Session(config=config, graph=tf.Graph()) as sess:
    with tf.device("/cpu:0" if not use_gpu else None):
      inputs_t = tf.Variable(inputs, trainable=False).value()
      ops = _static_vs_dynamic_rnn_benchmark_dynamic(
          inputs_t, sequence_length)
    tf.initialize_all_variables().run()
    delta_dynamic = _timer(sess, ops)

  print("%d \t %d \t %d \t %s \t %f \t %f \t %f" %
        (batch_size, max_time, num_units, use_gpu, delta_static,
         delta_dynamic, delta_dynamic/delta_static))

  return delta_static, delta_dynamic


def _half_seq_len_vs_unroll_half_rnn_benchmark(inputs_list_t, sequence_length):
  (_, input_size) = inputs_list_t[0].get_shape().as_list()
  initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = tf.nn.rnn_cell.LSTMCell(
      num_units=input_size, use_peepholes=True, initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = tf.nn.rnn(
      cell, inputs_list_t, sequence_length=sequence_length, dtype=tf.float32)

  trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  gradients = tf.gradients(outputs + [final_state], trainable_variables)

  return tf.group(final_state, *(gradients + outputs))


def half_seq_len_vs_unroll_half_rnn_benchmark(
    batch_size, max_time, num_units, use_gpu):
  config = tf.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = max_time * np.ones((batch_size,))
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)]

  # Halve the sequence length, full static unroll
  with tf.Session(config=config, graph=tf.Graph()) as sess:
    with tf.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          tf.Variable(x, trainable=False).value() for x in inputs_list]
      ops = _half_seq_len_vs_unroll_half_rnn_benchmark(
          inputs_list_t, sequence_length / 2)
    tf.initialize_all_variables().run()
    delta_half_seq_len = _timer(sess, ops)

  # Halve the unroll size, don't use sequence length
  with tf.Session(config=config, graph=tf.Graph()) as sess:
    with tf.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          tf.Variable(x, trainable=False).value() for x in inputs_list]
      ops = _half_seq_len_vs_unroll_half_rnn_benchmark(
          inputs_list_t[:(max_time // 2)], sequence_length / 2)
    tf.initialize_all_variables().run()
    delta_unroll_half = _timer(sess, ops)
  print("%d \t %d \t\t %d \t %s \t %f \t\t %f \t\t %f" %
        (batch_size, max_time, num_units, use_gpu, delta_half_seq_len,
         delta_unroll_half, delta_half_seq_len/delta_unroll_half))

  return delta_half_seq_len, delta_unroll_half


def _concat_state_vs_tuple_state_rnn_benchmark(
    inputs_list_t, sequence_length, state_is_tuple):
  (_, input_size) = inputs_list_t[0].get_shape().as_list()
  initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = tf.nn.rnn_cell.LSTMCell(
      num_units=input_size, use_peepholes=True,
      initializer=initializer, state_is_tuple=state_is_tuple)
  outputs, final_state = tf.nn.rnn(
      cell, inputs_list_t, sequence_length=sequence_length, dtype=tf.float32)

  final_state = list(final_state) if state_is_tuple else [final_state]

  trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  gradients = tf.gradients(outputs + final_state, trainable_variables)

  return tf.group(*(final_state + gradients + outputs))


def concat_state_vs_tuple_state_rnn_benchmark(
    batch_size, max_time, num_units, use_gpu):
  config = tf.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = max_time * np.ones((batch_size,))
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)]

  # Run with concatenated states (default)
  with tf.Session(config=config, graph=tf.Graph()) as sess:
    with tf.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          tf.Variable(x, trainable=False).value() for x in inputs_list]
      ops = _concat_state_vs_tuple_state_rnn_benchmark(
          inputs_list_t, sequence_length, state_is_tuple=False)
    tf.initialize_all_variables().run()
    delta_concat_state = _timer(sess, ops)

  # Run with tuple states (new)
  with tf.Session(config=config, graph=tf.Graph()) as sess:
    with tf.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          tf.Variable(x, trainable=False).value() for x in inputs_list]
      ops = _concat_state_vs_tuple_state_rnn_benchmark(
          inputs_list_t, sequence_length, state_is_tuple=True)
    tf.initialize_all_variables().run()
    delta_tuple_state = _timer(sess, ops)
  print("%d \t %d \t %d \t %s \t %f \t\t %f \t\t %f" %
        (batch_size, max_time, num_units, use_gpu, delta_concat_state,
         delta_tuple_state, delta_concat_state/delta_tuple_state))

  return delta_concat_state, delta_tuple_state


def _dynamic_rnn_swap_memory_benchmark(inputs_t, sequence_length,
                                       swap_memory):
  (unused_0, unused_1, input_size) = inputs_t.get_shape().as_list()
  initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = tf.nn.rnn_cell.LSTMCell(
      num_units=input_size, use_peepholes=True, initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = tf.nn.dynamic_rnn(
      cell, inputs_t, sequence_length=sequence_length,
      swap_memory=swap_memory, dtype=tf.float32)

  trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  gradients = tf.gradients([outputs, final_state], trainable_variables)

  return tf.group(final_state, outputs, *gradients)


def dynamic_rnn_swap_memory_benchmark(batch_size, max_time, num_units):
  config = tf.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  # No memory swap
  with tf.Session(config=config, graph=tf.Graph()) as sess:
    inputs_t = tf.Variable(inputs, trainable=False).value()
    ops = _dynamic_rnn_swap_memory_benchmark(
        inputs_t, sequence_length, swap_memory=False)
    tf.initialize_all_variables().run()
    no_swap = _timer(sess, ops)

  # Memory swap
  with tf.Session(config=config, graph=tf.Graph()) as sess:
    inputs_t = tf.Variable(inputs, trainable=False).value()
    ops = _dynamic_rnn_swap_memory_benchmark(
        inputs_t, sequence_length, swap_memory=True)
    tf.initialize_all_variables().run()
    swap = _timer(sess, ops)

  print("%d \t %d \t %d \t %f \t %f \t %f" %
        (batch_size, max_time, num_units, no_swap, swap, swap/no_swap))
  return no_swap, swap


def rnn_long_sequence_benchmark(batch_size, seqlen, num_units,
                                dynamic, swap_memory):
  config = tf.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = [seqlen for _ in range(batch_size)]
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(seqlen)]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  for _ in range(5):
    if dynamic:
      with tf.Session(config=config, graph=tf.Graph()) as sess:
        inputs_t = tf.Variable(inputs, trainable=False).value()
        ops = _dynamic_rnn_swap_memory_benchmark(
            inputs_t, sequence_length, swap_memory=swap_memory)
        tf.initialize_all_variables().run()
        elapsed = _timer(sess, ops)
    else:
      with tf.Session(config=config, graph=tf.Graph()) as sess:
        inputs_list_t = [
            tf.Variable(x, trainable=False).value() for x in inputs_list]
        ops = _static_vs_dynamic_rnn_benchmark_static(
            inputs_list_t, sequence_length)
        tf.initialize_all_variables().run()
        elapsed = _timer(sess, ops)

    print("%d \t %d \t %d \t %s \t %f \t %f" %
          (batch_size, seqlen, num_units, dynamic, elapsed,
           elapsed/seqlen))


class BenchmarkRNN(tf.test.Benchmark):

  def benchmarkGraphCreationStaticVsDynamicLSTM(self):
    print("Graph Creation: Static Unroll vs. Dynamic Unroll LSTM")
    print("max_t \t dt(static) \t dt(dynamic) \t dt(dynamic)/dt(static)")
    for max_time in (1, 25, 50):
      s_dt, d_dt = graph_creation_static_vs_dynamic_rnn_benchmark(max_time)
      self.report_benchmark(name="graph_creation_time_static_T%02d" % max_time,
                            iters=5, wall_time=s_dt)
      self.report_benchmark(name="graph_creation_time_dynamic_T%02d" % max_time,
                            iters=5, wall_time=d_dt)

  def benchmarkStaticUnrollVsDynamicFlowLSTM(self):
    print("Calculation: Static Unroll with Dynamic Flow LSTM "
          "vs. Dynamic Unroll LSTM")
    print("batch \t max_t \t units \t gpu \t dt(static) \t dt(dynamic) "
          "\t dt(dynamic)/dt(static)")
    for batch_size in (256,):
      for max_time in (50,):
        for num_units in (512, 256, 128):
          for use_gpu in (False, True):
            s_dt, d_dt = static_vs_dynamic_rnn_benchmark(
                batch_size, max_time, num_units, use_gpu)
            self.report_benchmark(
                name="static_unroll_time_T%02d_B%03d_N%03d_gpu_%s"
                % (max_time, batch_size, num_units, use_gpu),
                iters=20, wall_time=s_dt)
            self.report_benchmark(
                name="dynamic_unroll_time_T%02d_B%03d_N%03d_gpu_%s"
                % (max_time, batch_size, num_units, use_gpu),
                iters=20, wall_time=d_dt)

  def benchmarkDynamicLSTMNoMemorySwapVsMemorySwap(self):
    print("Calculation: Dynamic LSTM No Memory Swap vs. Memory Swap")
    print("batch \t max_t \t units \t no_swap \t swap \t swap/no_swap")
    for batch_size in (256, 512):
      for max_time in (100,):
        for num_units in (512, 256, 128):
          no_swap, swap = dynamic_rnn_swap_memory_benchmark(
              batch_size, max_time, num_units)
          self.report_benchmark(
              name="dynamic_lstm_no_memory_swap_T%02d_B%03d_N%03d"
              % (max_time, batch_size, num_units),
              iters=20, wall_time=no_swap)
          self.report_benchmark(
              name="dynamic_lstm_with_memory_swap_T%02d_B%03d_N%03d"
              % (max_time, batch_size, num_units),
              iters=20, wall_time=swap)

  def benchmarkStaticUnrollHalfSequenceLengthVsHalfUnroll(self):
    print("Calculation: Static Unroll with Halved Sequence Length "
          "vs. Half Static Unroll")
    print("batch \t full_t \t units \t gpu \t dt(half_seq_len) "
          "\t dt(unroll_half) \t dt(half_seq_len)/dt(unroll_half)")
    for batch_size in (128,):
      for max_time in (50,):
        for num_units in (256,):
          for use_gpu in (False, True):
            s_dt, d_dt = half_seq_len_vs_unroll_half_rnn_benchmark(
                batch_size, max_time, num_units, use_gpu)
            self.report_benchmark(
                name="half_seq_len_time_T%02d_B%03d_N%03d_gpu_%s"
                % (max_time, batch_size, num_units, use_gpu),
                iters=20, wall_time=s_dt)
            self.report_benchmark(
                name="unroll_half_time_T%02d_B%03d_N%03d_gpu_%s"
                % (max_time, batch_size, num_units, use_gpu),
                iters=20, wall_time=d_dt)

  def benchmarkStaticUnrollStateConcatVsStateTuple(self):
    print("Calculation: Static Unroll with Concatenated State "
          "vs. Tuple State")
    print("batch \t time \t units \t gpu \t dt(concat_state) "
          "\t dt(tuple_state) \t dt(concat_state)/dt(tuple_state)")
    for batch_size in (16, 128,):
      for max_time in (50,):
        for num_units in (16, 128,):
          for use_gpu in (False, True):
            c_dt, t_dt = concat_state_vs_tuple_state_rnn_benchmark(
                batch_size, max_time, num_units, use_gpu)
            self.report_benchmark(
                name="concat_state_time_T%02d_B%03d_N%03d_gpu_%s"
                % (max_time, batch_size, num_units, use_gpu),
                iters=20, wall_time=c_dt)
            self.report_benchmark(
                name="tuple_state_time_T%02d_B%03d_N%03d_gpu_%s"
                % (max_time, batch_size, num_units, use_gpu),
                iters=20, wall_time=t_dt)


if __name__ == "__main__":
  tf.test.main()
