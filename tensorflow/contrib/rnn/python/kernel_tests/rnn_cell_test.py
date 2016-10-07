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

"""Tests for RNN cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class RNNCellTest(tf.test.TestCase):

  def testCoupledInputForgetGateLSTMCell(self):
    with self.test_session() as sess:
      num_units = 2
      state_size = num_units * 2
      batch_size = 3
      input_size = 4
      expected_output = np.array(
          [[0.121753, 0.121753],
           [0.103349, 0.103349],
           [0.100178, 0.100178]],
          dtype=np.float32)
      expected_state = np.array(
          [[0.137523, 0.137523, 0.121753, 0.121753],
           [0.105450, 0.105450, 0.103349, 0.103349],
           [0.100742, 0.100742, 0.100178, 0.100178]],
          dtype=np.float32)
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([batch_size, input_size])
        m = tf.zeros([batch_size, state_size])
        output, state = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
            num_units=num_units, forget_bias=1.0)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([output, state],
                       {x.name: np.array([[1., 1., 1., 1.],
                                          [2., 2., 2., 2.],
                                          [3., 3., 3., 3.]]),
                        m.name: 0.1 * np.ones((batch_size, state_size))})
        # This is a smoke test: Only making sure expected values didn't change.
        self.assertEqual(len(res), 2)
        self.assertAllClose(res[0], expected_output)
        self.assertAllClose(res[1], expected_state)

  def testTimeFreqLSTMCell(self):
    with self.test_session() as sess:
      num_units = 8
      state_size = num_units * 2
      batch_size = 3
      input_size = 4
      feature_size = 2
      frequency_skip = 1
      num_shifts = (input_size - feature_size) / frequency_skip + 1
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([batch_size, input_size])
        m = tf.zeros([batch_size, state_size*num_shifts])
        output, state = tf.contrib.rnn.TimeFreqLSTMCell(
            num_units=num_units, feature_size=feature_size,
            frequency_skip=frequency_skip, forget_bias=1.0)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([output, state],
                       {x.name: np.array([[1., 1., 1., 1.],
                                          [2., 2., 2., 2.],
                                          [3., 3., 3., 3.]]),
                        m.name: 0.1 * np.ones((batch_size, state_size*(
                            num_shifts)))})
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is mostly just a
        # smoke test.
        self.assertEqual(res[0].shape, (batch_size, num_units*num_shifts))
        self.assertEqual(res[1].shape, (batch_size, state_size*num_shifts))
        # Different inputs so different outputs and states
        for i in range(1, batch_size):
          self.assertTrue(
              float(np.linalg.norm((res[0][0, :] - res[0][i, :]))) > 1e-6)
          self.assertTrue(
              float(np.linalg.norm((res[1][0, :] - res[1][i, :]))) > 1e-6)

  def testGridLSTMCell(self):
    with self.test_session() as sess:
      num_units = 8
      state_size = num_units * 2
      batch_size = 3
      input_size = 4
      feature_size = 2
      frequency_skip = 1
      num_shifts = (input_size - feature_size) / frequency_skip + 1
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([batch_size, input_size])
        m = tf.zeros([batch_size, state_size*num_shifts])
        output, state = tf.contrib.rnn.GridLSTMCell(
            num_units=num_units, feature_size=feature_size,
            frequency_skip=frequency_skip, forget_bias=1.0,
            num_frequency_blocks=num_shifts)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([output, state],
                       {x.name: np.array([[1., 1., 1., 1.],
                                          [2., 2., 2., 2.],
                                          [3., 3., 3., 3.]]),
                        m.name: 0.1 * np.ones((batch_size, state_size*(
                            num_shifts)))})
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is mostly just a
        # smoke test.
        self.assertEqual(res[0].shape, (batch_size, num_units*num_shifts*2))
        self.assertEqual(res[1].shape, (batch_size, state_size*num_shifts))
        # Different inputs so different outputs and states
        for i in range(1, batch_size):
          self.assertTrue(
              float(np.linalg.norm((res[0][0, :] - res[0][i, :]))) > 1e-6)
          self.assertTrue(
              float(np.linalg.norm((res[1][0, :] - res[1][i, :]))) > 1e-6)

  def testGridLstmCellWithCoupledInputForgetGates(self):
    num_units = 2
    state_size = num_units * 2
    batch_size = 3
    input_size = 4
    feature_size = 2
    frequency_skip = 1
    num_shifts = int((input_size - feature_size) / frequency_skip + 1)
    expected_output = np.array(
        [[ 0.416383, 0.416383, 0.403238, 0.403238, 0.524020, 0.524020,
           0.565425, 0.565425, 0.557865, 0.557865, 0.609699, 0.609699],
         [ 0.627331, 0.627331, 0.622393, 0.622393, 0.688342, 0.688342,
           0.708078, 0.708078, 0.694245, 0.694245, 0.715171, 0.715171],
         [ 0.711050, 0.711050, 0.709197, 0.709197, 0.736533, 0.736533,
           0.744264, 0.744264, 0.737390, 0.737390, 0.745250, 0.745250]],
        dtype=np.float32)
    expected_state = np.array(
        [[ 0.625556, 0.625556, 0.416383, 0.416383, 0.759134, 0.759134,
           0.524020, 0.524020, 0.798795, 0.798795, 0.557865, 0.557865],
         [ 0.875488, 0.875488, 0.627331, 0.627331, 0.936432, 0.936432,
           0.688342, 0.688342, 0.941961, 0.941961, 0.694245, 0.694245],
         [ 0.957327, 0.957327, 0.711050, 0.711050, 0.979522, 0.979522,
           0.736533, 0.736533, 0.980245, 0.980245, 0.737390, 0.737390]],
        dtype=np.float32)
    for state_is_tuple in [False, True]:
      with self.test_session() as sess:
        with tf.variable_scope("state_is_tuple" + str(state_is_tuple),
                               initializer=tf.constant_initializer(0.5)):
          cell = tf.contrib.rnn.GridLSTMCell(
              num_units=num_units, feature_size=feature_size,
              frequency_skip=frequency_skip, forget_bias=1.0,
              num_frequency_blocks=num_shifts,
              couple_input_forget_gates=True,
              state_is_tuple=state_is_tuple)
          inputs = tf.constant(np.array([[1., 1., 1., 1.],
                                         [2., 2., 2., 2.],
                                         [3., 3., 3., 3.]],
                                        dtype=np.float32), dtype=tf.float32)
          if state_is_tuple:
            state_value = tf.constant(
                0.1 * np.ones((batch_size, num_units), dtype=np.float32),
                dtype=tf.float32)
            init_state = cell.state_tuple_type(
                *([state_value, state_value] * num_shifts))
          else:
            init_state = tf.constant(
                0.1 * np.ones((batch_size, num_units * num_shifts * 2),
                              dtype=np.float32),
                dtype=tf.float32)
          output, state = cell(inputs, init_state)
          sess.run([tf.initialize_all_variables()])
          res = sess.run([output, state])
          # This is a smoke test: Only making sure expected values not change.
          self.assertEqual(len(res), 2)
          self.assertAllClose(res[0], expected_output)
          if not state_is_tuple:
            self.assertAllClose(res[1], expected_state)
          else:
            # There should be num_shifts * 2 states in the tuple.
            self.assertEqual(len(res[1]), num_shifts * 2)
            # Checking the shape of each state to be batch_size * num_units
            for ss in res[1]:
              self.assertEqual(ss.shape[0], batch_size)
              self.assertEqual(ss.shape[1], num_units)
            self.assertAllClose(np.concatenate(res[1], axis=1), expected_state)

  def testAttentionCellWrapperFailures(self):
    with self.assertRaisesRegexp(
        TypeError, "The parameter cell is not RNNCell."):
      tf.contrib.rnn.AttentionCellWrapper(None, 0)

    num_units = 8
    for state_is_tuple in [False, True]:
      with tf.Graph().as_default():
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            num_units, state_is_tuple=state_is_tuple)
        with self.assertRaisesRegexp(
            ValueError, "attn_length should be greater than zero, got 0"):
          tf.contrib.rnn.AttentionCellWrapper(lstm_cell, 0,
                                              state_is_tuple=state_is_tuple)
        with self.assertRaisesRegexp(
            ValueError, "attn_length should be greater than zero, got -1"):
          tf.contrib.rnn.AttentionCellWrapper(lstm_cell, -1,
                                              state_is_tuple=state_is_tuple)
      with tf.Graph().as_default():
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
            num_units, state_is_tuple=True)
        with self.assertRaisesRegexp(
            ValueError, "Cell returns tuple of states, but the flag "
            "state_is_tuple is not set. State size is: *"):
          tf.contrib.rnn.AttentionCellWrapper(
              lstm_cell, 4, state_is_tuple=False)

  def testAttentionCellWrapperZeros(self):
    num_units = 8
    attn_length = 16
    batch_size = 3
    input_size = 4
    for state_is_tuple in [False, True]:
      with tf.Graph().as_default():
        with self.test_session() as sess:
          with tf.variable_scope("state_is_tuple_" + str(state_is_tuple)):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                num_units, state_is_tuple=state_is_tuple)
            cell = tf.contrib.rnn.AttentionCellWrapper(
                lstm_cell, attn_length, state_is_tuple=state_is_tuple)
            if state_is_tuple:
              zeros = tf.zeros(
                  [batch_size, num_units], dtype=np.float32)
              attn_state_zeros = tf.zeros(
                  [batch_size, attn_length * num_units], dtype=np.float32)
              zero_state = ((zeros, zeros), zeros, attn_state_zeros)
            else:
              zero_state = tf.zeros(
                  [batch_size, num_units * 2 + attn_length
                   * num_units + num_units], dtype=np.float32)
            inputs = tf.zeros([batch_size, input_size], dtype=tf.float32)
            output, state = cell(inputs, zero_state)
            self.assertEquals(output.get_shape(), [batch_size, num_units])
            if state_is_tuple:
              self.assertEquals(len(state), 3)
              self.assertEquals(len(state[0]), 2)
              self.assertEquals(state[0][0].get_shape(),
                                [batch_size, num_units])
              self.assertEquals(state[0][1].get_shape(),
                                [batch_size, num_units])
              self.assertEquals(state[1].get_shape(), [batch_size, num_units])
              self.assertEquals(state[2].get_shape(),
                                [batch_size, attn_length * num_units])
              tensors = [output] + list(state)
            else:
              self.assertEquals(
                  state.get_shape(), [batch_size, num_units * 2 + num_units
                                      + attn_length * num_units])
              tensors = [output, state]
            zero_result = sum([tf.reduce_sum(tf.abs(x)) for x in tensors])
            sess.run(tf.initialize_all_variables())
            self.assertTrue(sess.run(zero_result) < 1e-6)

  def testAttentionCellWrapperValues(self):
    num_units = 8
    attn_length = 16
    batch_size = 3
    for state_is_tuple in [False, True]:
      with tf.Graph().as_default():
        with self.test_session() as sess:
          with tf.variable_scope("state_is_tuple_" + str(state_is_tuple)):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
                num_units, state_is_tuple=state_is_tuple)
            cell = tf.contrib.rnn.AttentionCellWrapper(
                lstm_cell, attn_length, state_is_tuple=state_is_tuple)
            if state_is_tuple:
              zeros = tf.constant(
                  0.1 * np.ones([batch_size, num_units],
                                dtype=np.float32), dtype=tf.float32)
              attn_state_zeros = tf.constant(
                  0.1 * np.ones([batch_size, attn_length * num_units],
                                dtype=np.float32), dtype=tf.float32)
              zero_state = ((zeros, zeros), zeros, attn_state_zeros)
            else:
              zero_state = tf.constant(
                  0.1 * np.ones([batch_size, num_units * 2 + num_units
                                 + attn_length * num_units],
                                dtype=np.float32), dtype=tf.float32)
            inputs = tf.constant(np.array([[1., 1., 1., 1.],
                                           [2., 2., 2., 2.],
                                           [3., 3., 3., 3.]],
                                          dtype=np.float32), dtype=tf.float32)
            output, state = cell(inputs, zero_state)
            if state_is_tuple:
              concat_state = tf.concat(
                  1, [state[0][0], state[0][1], state[1], state[2]])
            else:
              concat_state = state
            sess.run(tf.initialize_all_variables())
            output, state = sess.run([output, concat_state])
            # Different inputs so different outputs and states
            for i in range(1, batch_size):
              self.assertTrue(
                  float(np.linalg.norm((output[0, :] - output[i, :]))) > 1e-6)
              self.assertTrue(
                  float(np.linalg.norm((state[0, :] - state[i, :]))) > 1e-6)

  def testAttentionCellWrapperCorrectResult(self):
    num_units = 4
    attn_length = 6
    batch_size = 2
    expected_output = np.array(
        [[-0.76951641, -0.77613342, 0.1882876, 0.4528169],
         [-0.62173879, -0.53987527, 0.06999521, 0.43236512]], dtype=np.float32)
    expected_state = np.array(
        [[0.00686008, 0.32751927, 0.65842509, 0.13517606, 0.00341745,
          0.21539585, 0.2087716, 0.04919484, 0.65901887, 0.71350443, 0.45989594,
          0.32038051, 0.58086717, 0.49446869, 0.7641536, 0.12814975, 0.92231739,
          0.89857256, 0.21889746, 0.38442063, 0.53481543, 0.8876909, 0.45823169,
          0.5905602, 0.78038228, 0.56501579, 0.03971386, 0.09870267, 0.8074435,
          0.66821432, 0.99211812, 0.12295902, -0.78066337, -0.55385113,
          0.25296241, 0.29621673],
         [-2.65422642e-01, 7.69232273e-01, 2.61641771e-01, 3.12298536e-03,
          -1.54120743e-01, 4.68760282e-01, 9.73877981e-02, 9.45428968e-04,
          7.77730405e-01, 6.53964162e-01, 4.54966187e-01, 4.93799955e-01,
          7.30002642e-01, 6.69868946e-01, 7.35766888e-01, 8.63012671e-01,
          8.78873706e-01, 3.51857543e-01, 9.34172153e-01, 6.47329569e-01,
          6.31730437e-01, 6.66278243e-01, 5.36446571e-01, 2.04774857e-01,
          9.84584212e-01, 3.82772446e-01, 3.74667645e-02, 9.25101876e-01,
          5.77141643e-01, 8.49329710e-01, 3.61274123e-01, 1.21259212e-01,
          -6.95882142e-01, -7.14576960e-01, 5.69079161e-01, 3.14788610e-01]],
        dtype=np.float32)
    seed = 12345
    tf.set_random_seed(seed)
    for state_is_tuple in [False, True]:
      with tf.Session() as sess:
        with tf.variable_scope("state_is_tuple", reuse=state_is_tuple):
          lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(
              num_units, state_is_tuple=state_is_tuple)
          cell = tf.contrib.rnn.AttentionCellWrapper(
              lstm_cell, attn_length, state_is_tuple=state_is_tuple)
          zeros1 = tf.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed+1)
          zeros2 = tf.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed+2)
          zeros3 = tf.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed+3)
          attn_state_zeros = tf.random_uniform(
              (batch_size, attn_length * num_units), 0.0, 1.0, seed=seed+4)
          zero_state = ((zeros1, zeros2), zeros3, attn_state_zeros)
          if not state_is_tuple:
            zero_state = tf.concat(1,
                                   [zero_state[0][0], zero_state[0][1],
                                    zero_state[1], zero_state[2]])
          inputs = tf.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed+5)
          output, state = cell(inputs, zero_state)
          if state_is_tuple:
            state = tf.concat(1, [state[0][0], state[0][1], state[1], state[2]])
          sess.run(tf.initialize_all_variables())
          self.assertAllClose(sess.run(output), expected_output)
          self.assertAllClose(sess.run(state), expected_state)


class LayerNormBasicLSTMCellTest(tf.test.TestCase):

  # NOTE: all the values in the current test case have been calculated.

  def testBasicLSTMCell(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        c0 = tf.zeros([1, 2])
        h0 = tf.zeros([1, 2])
        state0 = tf.nn.rnn_cell.LSTMStateTuple(c0, h0)
        c1 = tf.zeros([1, 2])
        h1 = tf.zeros([1, 2])
        state1 = tf.nn.rnn_cell.LSTMStateTuple(c1, h1)
        state = (state0, state1)
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(2)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
        g, out_m = cell(x, state)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, out_m],
                       {
                           x.name: np.array([[1., 1.]]),
                           c0.name: 0.1 * np.asarray([[0, 1]]),
                           h0.name: 0.1 * np.asarray([[2, 3]]),
                           c1.name: 0.1 * np.asarray([[4, 5]]),
                           h1.name: 0.1 * np.asarray([[6, 7]]),
                       })

        expected_h = np.array([[-0.38079708, 0.38079708]])
        expected_state0_c = np.array([[-1.0, 1.0]])
        expected_state0_h = np.array([[-0.38079708, 0.38079708]])
        expected_state1_c = np.array([[-1.0, 1.0]])
        expected_state1_h = np.array([[-0.38079708, 0.38079708]])

        actual_h = res[0]
        actual_state0_c = res[1][0].c
        actual_state0_h = res[1][0].h
        actual_state1_c = res[1][1].c
        actual_state1_h = res[1][1].h

        self.assertAllClose(actual_h, expected_h, 1e-5)
        self.assertAllClose(expected_state0_c, actual_state0_c, 1e-5)
        self.assertAllClose(expected_state0_h, actual_state0_h, 1e-5)
        self.assertAllClose(expected_state1_c, actual_state1_c, 1e-5)
        self.assertAllClose(expected_state1_h, actual_state1_h, 1e-5)

      with tf.variable_scope("other", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 3])  # Test BasicLSTMCell with input_size != num_units.
        c = tf.zeros([1, 2])
        h = tf.zeros([1, 2])
        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(2)
        g, out_m = cell(x, state)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, out_m],
                       {
                           x.name: np.array([[1., 1., 1.]]),
                           c.name: 0.1 * np.asarray([[0, 1]]),
                           h.name: 0.1 * np.asarray([[2, 3]]),
                       })

        expected_h = np.array([[-0.38079708, 0.38079708]])
        expected_c = np.array([[-1.0, 1.0]])
        self.assertEqual(len(res), 2)
        self.assertAllClose(res[0], expected_h, 1e-5)
        self.assertAllClose(res[1].c, expected_c, 1e-5)
        self.assertAllClose(res[1].h, expected_h, 1e-5)

  def testBasicLSTMCellWithStateTuple(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        c0 = tf.zeros([1, 2])
        h0 = tf.zeros([1, 2])
        state0 = tf.nn.rnn_cell.LSTMStateTuple(c0, h0)
        c1 = tf.zeros([1, 2])
        h1 = tf.zeros([1, 2])
        state1 = tf.nn.rnn_cell.LSTMStateTuple(c1, h1)
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(2)
        cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2)
        h, (s0, s1) = cell(x, (state0, state1))
        sess.run([tf.initialize_all_variables()])
        res = sess.run([h, s0, s1],
                       {
                           x.name: np.array([[1., 1.]]),
                           c0.name: 0.1 * np.asarray([[0, 1]]),
                           h0.name: 0.1 * np.asarray([[2, 3]]),
                           c1.name: 0.1 * np.asarray([[4, 5]]),
                           h1.name: 0.1 * np.asarray([[6, 7]]),
                       })

        expected_h = np.array([[-0.38079708, 0.38079708]])
        expected_h0 = np.array([[-0.38079708, 0.38079708]])
        expected_c0 = np.array([[-1.0, 1.0]])
        expected_h1 = np.array([[-0.38079708, 0.38079708]])
        expected_c1 = np.array([[-1.0, 1.0]])

        self.assertEqual(len(res), 3)
        self.assertAllClose(res[0], expected_h, 1e-5)
        self.assertAllClose(res[1].c, expected_c0, 1e-5)
        self.assertAllClose(res[1].h, expected_h0, 1e-5)
        self.assertAllClose(res[2].c, expected_c1, 1e-5)
        self.assertAllClose(res[2].h, expected_h1, 1e-5)

  def testBasicLSTMCellWithDropout(self):

    def _is_close(x, y, digits=4):
      delta = x - y
      return delta < 10 ** (-digits)

    def _is_close_in(x, items, digits=4):
      for i in items:
        if _is_close(x, i, digits):
          return True
      return False

    keep_prob = 0.5
    c_high = 2.9998924946
    c_low = 0.999983298578
    h_low = 0.761552567265
    h_high = 0.995008519604
    num_units = 5
    allowed_low = [2, 3]

    with self.test_session() as sess:
      with tf.variable_scope("other", initializer=tf.constant_initializer(1)):
        x = tf.zeros([1, 5])
        c = tf.zeros([1, 5])
        h = tf.zeros([1, 5])
        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units, layer_norm=False, dropout_keep_prob=keep_prob)

        g, s = cell(x, state)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s],
                       {
                           x.name: np.ones([1, 5]),
                           c.name: np.ones([1, 5]),
                           h.name: np.ones([1, 5]),
                       })

        # Since the returned tensors are of size [1,n]
        # get the first component right now.
        actual_h = res[0][0]
        actual_state_c = res[1].c[0]
        actual_state_h = res[1].h[0]

        # For each item in `c` (the cell inner state) check that
        # it is equal to one of the allowed values `c_high` (not
        # dropped out) or `c_low` (dropped out) and verify that the
        # corresponding item in `h` (the cell activation) is coherent.
        # Count the dropped activations and check that their number is
        # coherent with the dropout probability.
        dropped_count = 0
        self.assertTrue((actual_h == actual_state_h).all())
        for citem, hitem in zip(actual_state_c, actual_state_h):
          self.assertTrue(_is_close_in(citem, [c_low, c_high]))
          if _is_close(citem, c_low):
            self.assertTrue(_is_close(hitem, h_low))
            dropped_count += 1
          elif _is_close(citem, c_high):
            self.assertTrue(_is_close(hitem, h_high))
        self.assertIn(dropped_count, allowed_low)


if __name__ == "__main__":
  tf.test.main()
