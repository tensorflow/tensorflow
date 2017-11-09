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

import numpy as np

from tensorflow.contrib.rnn.python.ops import rnn as contrib_rnn
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


class StackBidirectionalRNNTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _createStackBidirectionalRNN(self,
                                   use_gpu,
                                   use_shape,
                                   use_sequence_length,
                                   initial_states_fw=None,
                                   initial_states_bw=None,
                                   scope=None):
    self.layers = [2, 3]
    input_size = 5
    batch_size = 2
    max_length = 8

    initializer = init_ops.random_uniform_initializer(
        -0.01, 0.01, seed=self._seed)
    sequence_length = array_ops.placeholder(
        dtypes.int64) if use_sequence_length else None

    self.cells_fw = [
        rnn_cell.LSTMCell(
            num_units,
            input_size,
            initializer=initializer,
            state_is_tuple=False) for num_units in self.layers
    ]
    self.cells_bw = [
        rnn_cell.LSTMCell(
            num_units,
            input_size,
            initializer=initializer,
            state_is_tuple=False) for num_units in self.layers
    ]

    inputs = max_length * [
        array_ops.placeholder(
            dtypes.float32,
            shape=(batch_size, input_size) if use_shape else (None, input_size))
    ]
    outputs, state_fw, state_bw = contrib_rnn.stack_bidirectional_rnn(
        self.cells_fw,
        self.cells_bw,
        inputs,
        initial_states_fw,
        initial_states_bw,
        dtype=dtypes.float32,
        sequence_length=sequence_length,
        scope=scope)

    self.assertEqual(len(outputs), len(inputs))
    for out in outputs:
      self.assertAlmostEqual(
          out.get_shape().as_list(),
          [batch_size if use_shape else None, 2 * self.layers[-1]])

    input_value = np.random.randn(batch_size, input_size)
    outputs = array_ops.stack(outputs)

    return input_value, inputs, outputs, state_fw, state_bw, sequence_length

  def _testStackBidirectionalRNN(self, use_gpu, use_shape):
    with self.test_session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createStackBidirectionalRNN(use_gpu, use_shape, True))
      variables.global_variables_initializer().run()
      # Run with pre-specified sequence lengths of 2, 3.
      out, s_fw, s_bw = sess.run(
          [outputs, state_fw, state_bw],
          feed_dict={inputs[0]: input_value,
                     sequence_length: [2, 3]})

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward states of the first layer
      # must be the same.
      # For the next layers, since the input is a concat of forward and backward
      # outputs of the previous layers the symmetry is broken and the following
      # states and outputs differ.
      # We cannot access the intermediate values between layers but we can
      # check that the forward and backward states of the first layer match.

      self.assertAllClose(s_fw[0], s_bw[0])

      # If outputs are not concat between layers the output of the forward
      # and backward would be the same but symmetric.
      # Check that it is not the case.
      # Due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      # First sequence in batch is length=2
      # Check that the time=0 forward output is not equal to time=1 backward.
      self.assertNotEqual(out[0][0][0], out[1][0][3])
      self.assertNotEqual(out[0][0][1], out[1][0][4])
      self.assertNotEqual(out[0][0][2], out[1][0][5])
      # Check that the time=1 forward output is not equal to time=0 backward.
      self.assertNotEqual(out[1][0][0], out[0][0][3])
      self.assertNotEqual(out[1][0][1], out[0][0][4])
      self.assertNotEqual(out[1][0][2], out[0][0][5])

      # Second sequence in batch is length=3
      # Check that the time=0 forward output is not equal to time=2 backward.
      self.assertNotEqual(out[0][1][0], out[2][1][3])
      self.assertNotEqual(out[0][1][1], out[2][1][4])
      self.assertNotEqual(out[0][1][2], out[2][1][5])
      # Check that the time=1 forward output is not equal to time=1 backward.
      self.assertNotEqual(out[1][1][0], out[1][1][3])
      self.assertNotEqual(out[1][1][1], out[1][1][4])
      self.assertNotEqual(out[1][1][2], out[1][1][5])
      # Check that the time=2 forward output is not equal to time=0 backward.
      self.assertNotEqual(out[2][1][0], out[0][1][3])
      self.assertNotEqual(out[2][1][1], out[0][1][4])
      self.assertNotEqual(out[2][1][2], out[0][1][5])

  def _testStackBidirectionalRNNStates(self, use_gpu):
    # Check that the states are correctly initialized.
    # - Create a net and iterate for 3 states. Keep the state (state_3).
    # - Reset states, and iterate for 5 steps. Last state is state_5.
    # - Reset the sets to state_3 and iterate for 2 more steps,
    #   last state will be state_5'.
    # - Check that the state_5 and state_5' (forward and backward) are the
    #   same for the first layer (it does not apply for the second layer since
    #   it has forward-backward dependencies).
    with self.test_session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
      batch_size = 2
      # Create states placeholders.
      initial_states_fw = [
          array_ops.placeholder(
              dtypes.float32, shape=(batch_size, layer * 2))
          for layer in self.layers
      ]
      initial_states_bw = [
          array_ops.placeholder(
              dtypes.float32, shape=(batch_size, layer * 2))
          for layer in self.layers
      ]
      # Create the net
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createStackBidirectionalRNN(use_gpu, True, True,
                                            initial_states_fw,
                                            initial_states_bw))
      variables.global_variables_initializer().run()

      # Run 3 steps.
      feed_dict = {inputs[0]: input_value, sequence_length: [3, 2]}
      # Initialize to empty state.
      for i, layer in enumerate(self.layers):
        feed_dict[initial_states_fw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
        feed_dict[initial_states_bw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
      _, st_3_fw, st_3_bw = sess.run([outputs, state_fw, state_bw],
                                     feed_dict=feed_dict)

      # Reset the net and run 5 steps.
      feed_dict = {inputs[0]: input_value, sequence_length: [5, 3]}
      for i, layer in enumerate(self.layers):
        feed_dict[initial_states_fw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
        feed_dict[initial_states_bw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
      _, st_5_fw, st_5_bw = sess.run([outputs, state_fw, state_bw],
                                     feed_dict=feed_dict)

      # Reset the net to state_3 and run 2 more steps.
      feed_dict = {inputs[0]: input_value, sequence_length: [2, 1]}
      for i, _ in enumerate(self.layers):
        feed_dict[initial_states_fw[i]] = st_3_fw[i]
        feed_dict[initial_states_bw[i]] = st_3_bw[i]
      out_5p, st_5p_fw, st_5p_bw = sess.run([outputs, state_fw, state_bw],
                                            feed_dict=feed_dict)

      # Check that the 3+2 and 5 first layer states.
      self.assertAllEqual(st_5_fw[0], st_5p_fw[0])
      self.assertAllEqual(st_5_bw[0], st_5p_bw[0])

  def testStackBidirectionalRNN(self):
    self._testStackBidirectionalRNN(use_gpu=False, use_shape=False)
    self._testStackBidirectionalRNN(use_gpu=True, use_shape=False)
    self._testStackBidirectionalRNN(use_gpu=False, use_shape=True)
    self._testStackBidirectionalRNN(use_gpu=True, use_shape=True)
    self._testStackBidirectionalRNNStates(use_gpu=False)
    self._testStackBidirectionalRNNStates(use_gpu=True)

  def _createStackBidirectionalDynamicRNN(self,
                                          use_gpu,
                                          use_shape,
                                          use_state_tuple,
                                          initial_states_fw=None,
                                          initial_states_bw=None,
                                          scope=None):
    self.layers = [2, 3]
    input_size = 5
    batch_size = 2
    max_length = 8

    initializer = init_ops.random_uniform_initializer(
        -0.01, 0.01, seed=self._seed)
    sequence_length = array_ops.placeholder(dtypes.int64)

    self.cells_fw = [
        rnn_cell.LSTMCell(
            num_units,
            input_size,
            initializer=initializer,
            state_is_tuple=False) for num_units in self.layers
    ]
    self.cells_bw = [
        rnn_cell.LSTMCell(
            num_units,
            input_size,
            initializer=initializer,
            state_is_tuple=False) for num_units in self.layers
    ]

    inputs = max_length * [
        array_ops.placeholder(
            dtypes.float32,
            shape=(batch_size, input_size) if use_shape else (None, input_size))
    ]
    inputs_c = array_ops.stack(inputs)
    inputs_c = array_ops.transpose(inputs_c, [1, 0, 2])
    outputs, st_fw, st_bw = contrib_rnn.stack_bidirectional_dynamic_rnn(
        self.cells_fw,
        self.cells_bw,
        inputs_c,
        initial_states_fw=initial_states_fw,
        initial_states_bw=initial_states_bw,
        dtype=dtypes.float32,
        sequence_length=sequence_length,
        scope=scope)

    # Outputs has shape (batch_size, max_length, 2* layer[-1].
    output_shape = [None, max_length, 2 * self.layers[-1]]
    if use_shape:
      output_shape[0] = batch_size

    self.assertAllEqual(outputs.get_shape().as_list(), output_shape)

    input_value = np.random.randn(batch_size, input_size)

    return input_value, inputs, outputs, st_fw, st_bw, sequence_length

  def _testStackBidirectionalDynamicRNN(self, use_gpu, use_shape,
                                        use_state_tuple):
    with self.test_session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createStackBidirectionalDynamicRNN(use_gpu, use_shape,
                                                   use_state_tuple))
      variables.global_variables_initializer().run()
      # Run with pre-specified sequence length of 2, 3
      out, s_fw, s_bw = sess.run(
          [outputs, state_fw, state_bw],
          feed_dict={inputs[0]: input_value,
                     sequence_length: [2, 3]})

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward states of the first layer has
      # to be the same.
      # For the next layers, since the input is a concat of forward and backward
      # outputs of the previous layers the symmetry is broken and the following
      # states and outputs differ.
      # We cannot access the intermediate values between layers but we can
      # check that the forward and backward states of the first layer match.

      self.assertAllClose(s_fw[0], s_bw[0])
      out = np.swapaxes(out, 0, 1)
      # If outputs are not concat between layers the output of the forward
      # and backward would be the same but symmetric.
      # Check that is not the case.
      # Due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      # First sequence in batch is length=2
      # Check that the time=0 forward output is not equal to time=1 backward.
      self.assertNotEqual(out[0][0][0], out[1][0][3])
      self.assertNotEqual(out[0][0][1], out[1][0][4])
      self.assertNotEqual(out[0][0][2], out[1][0][5])
      # Check that the time=1 forward output is not equal to time=0 backward.
      self.assertNotEqual(out[1][0][0], out[0][0][3])
      self.assertNotEqual(out[1][0][1], out[0][0][4])
      self.assertNotEqual(out[1][0][2], out[0][0][5])

      # Second sequence in batch is length=3
      # Check that the time=0 forward output is not equal to time=2 backward.
      self.assertNotEqual(out[0][1][0], out[2][1][3])
      self.assertNotEqual(out[0][1][1], out[2][1][4])
      self.assertNotEqual(out[0][1][2], out[2][1][5])
      # Check that the time=1 forward output is not equal to time=1 backward.
      self.assertNotEqual(out[1][1][0], out[1][1][3])
      self.assertNotEqual(out[1][1][1], out[1][1][4])
      self.assertNotEqual(out[1][1][2], out[1][1][5])
      # Check that the time=2 forward output is not equal to time=0 backward.
      self.assertNotEqual(out[2][1][0], out[0][1][3])
      self.assertNotEqual(out[2][1][1], out[0][1][4])
      self.assertNotEqual(out[2][1][2], out[0][1][5])

  def _testStackBidirectionalDynamicRNNStates(self, use_gpu):

    # Check that the states are correctly initialized.
    # - Create a net and iterate for 3 states. Keep the state (state_3).
    # - Reset states, and iterate for 5 steps. Last state is state_5.
    # - Reset the sets to state_3 and iterate for 2 more steps,
    #   last state will be state_5'.
    # - Check that the state_5 and state_5' (forward and backward) are the
    #   same for the first layer (it does not apply for the second layer since
    #   it has forward-backward dependencies).
    with self.test_session(use_gpu=use_gpu, graph=ops.Graph()) as sess:
      batch_size = 2
      # Create states placeholders.
      initial_states_fw = [
          array_ops.placeholder(
              dtypes.float32, shape=(batch_size, layer * 2))
          for layer in self.layers
      ]
      initial_states_bw = [
          array_ops.placeholder(
              dtypes.float32, shape=(batch_size, layer * 2))
          for layer in self.layers
      ]
      # Create the net
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createStackBidirectionalDynamicRNN(
              use_gpu,
              use_shape=True,
              use_state_tuple=False,
              initial_states_fw=initial_states_fw,
              initial_states_bw=initial_states_bw))
      variables.global_variables_initializer().run()

      # Run 3 steps.
      feed_dict = {inputs[0]: input_value, sequence_length: [3, 2]}
      # Initialize to empty state.
      for i, layer in enumerate(self.layers):
        feed_dict[initial_states_fw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
        feed_dict[initial_states_bw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
      _, st_3_fw, st_3_bw = sess.run([outputs, state_fw, state_bw],
                                     feed_dict=feed_dict)

      # Reset the net and run 5 steps.
      feed_dict = {inputs[0]: input_value, sequence_length: [5, 3]}
      for i, layer in enumerate(self.layers):
        feed_dict[initial_states_fw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
        feed_dict[initial_states_bw[i]] = np.zeros(
            (batch_size, layer * 2), dtype=np.float32)
      _, st_5_fw, st_5_bw = sess.run([outputs, state_fw, state_bw],
                                     feed_dict=feed_dict)

      # Reset the net to state_3 and run 2 more steps.
      feed_dict = {inputs[0]: input_value, sequence_length: [2, 1]}
      for i, _ in enumerate(self.layers):
        feed_dict[initial_states_fw[i]] = st_3_fw[i]
        feed_dict[initial_states_bw[i]] = st_3_bw[i]
      out_5p, st_5p_fw, st_5p_bw = sess.run([outputs, state_fw, state_bw],
                                            feed_dict=feed_dict)

      # Check that the 3+2 and 5 first layer states.
      self.assertAllEqual(st_5_fw[0], st_5p_fw[0])
      self.assertAllEqual(st_5_bw[0], st_5p_bw[0])

  def testBidirectionalRNN(self):
    # Generate 2^3 option values
    # from [True, True, True] to [False, False, False]
    options = itertools.product([True, False], repeat=3)
    for option in options:
      self._testStackBidirectionalDynamicRNN(
          use_gpu=option[0], use_shape=option[1], use_state_tuple=option[2])
    # Check States.
    self._testStackBidirectionalDynamicRNNStates(use_gpu=False)
    self._testStackBidirectionalDynamicRNNStates(use_gpu=True)

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    # REMARKS: factory(scope) is a function accepting a scope
    #          as an argument, such scope can be None, a string
    #          or a VariableScope instance.
    with self.test_session(use_gpu=True, graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)

      # check that all the variables names starts with the proper scope.
      variables.global_variables_initializer()
      all_vars = variables.global_variables()
      prefix = prefix or "stack_bidirectional_rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("StackRNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  def testStackBidirectionalRNNScope(self):

    def factory(scope):
      return self._createStackBidirectionalRNN(
          use_gpu=True, use_shape=True, use_sequence_length=True, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)

  def testBidirectionalDynamicRNNScope(self):

    def factory(scope):
      return self._createStackBidirectionalDynamicRNN(
          use_gpu=True, use_shape=True, use_state_tuple=True, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


if __name__ == "__main__":
  test.main()
