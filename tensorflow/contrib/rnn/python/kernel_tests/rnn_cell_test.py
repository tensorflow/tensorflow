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

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.rnn.python.ops import rnn_cell
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class RNNCellTest(test.TestCase):

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
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size, state_size])
        output, state = rnn_cell.CoupledInputForgetGateLSTMCell(
            num_units=num_units, forget_bias=1.0)(x, m)
        sess.run([variables.global_variables_initializer()])
        res = sess.run([output, state], {
            x.name:
                np.array([[1., 1., 1., 1.],
                          [2., 2., 2., 2.],
                          [3., 3., 3., 3.]]),
            m.name:
                0.1 * np.ones((batch_size, state_size))
        })
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
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size, state_size * num_shifts])
        output, state = rnn_cell.TimeFreqLSTMCell(
            num_units=num_units,
            feature_size=feature_size,
            frequency_skip=frequency_skip,
            forget_bias=1.0)(x, m)
        sess.run([variables.global_variables_initializer()])
        res = sess.run([output, state], {
            x.name:
                np.array([[1., 1., 1., 1.],
                          [2., 2., 2., 2.],
                          [3., 3., 3., 3.]]),
            m.name:
                0.1 * np.ones((batch_size, int(state_size * (num_shifts))))
        })
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is mostly just a
        # smoke test.
        self.assertEqual(res[0].shape, (batch_size, num_units * num_shifts))
        self.assertEqual(res[1].shape, (batch_size, state_size * num_shifts))
        # Different inputs so different outputs and states
        for i in range(1, batch_size):
          self.assertTrue(
              float(np.linalg.norm((res[0][0, :] - res[0][i, :]))) > 1e-6)
          self.assertTrue(
              float(np.linalg.norm((res[1][0, :] - res[1][i, :]))) > 1e-6)

  def testGridLSTMCell(self):
    with self.test_session() as sess:
      num_units = 8
      batch_size = 3
      input_size = 4
      feature_size = 2
      frequency_skip = 1
      num_shifts = int((input_size - feature_size) / frequency_skip + 1)
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        cell = rnn_cell.GridLSTMCell(
            num_units=num_units,
            feature_size=feature_size,
            frequency_skip=frequency_skip,
            forget_bias=1.0,
            num_frequency_blocks=[num_shifts],
            couple_input_forget_gates=True,
            state_is_tuple=True)
        inputs = constant_op.constant(
            np.array(
                [[1., 1., 1., 1.],
                 [2., 2., 2., 2.],
                 [3., 3., 3., 3.]],
                dtype=np.float32),
            dtype=dtypes.float32)
        state_value = constant_op.constant(
            0.1 * np.ones(
                (batch_size, num_units), dtype=np.float32),
            dtype=dtypes.float32)
        init_state = cell.state_tuple_type(
            *([state_value, state_value] * num_shifts))
        output, state = cell(inputs, init_state)
        sess.run([variables.global_variables_initializer()])
        res = sess.run([output, state])
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is mostly just a
        # smoke test.
        self.assertEqual(res[0].shape, (batch_size, num_units * num_shifts * 2))
        for ss in res[1]:
          self.assertEqual(ss.shape, (batch_size, num_units))
        # Different inputs so different outputs and states
        for i in range(1, batch_size):
          self.assertTrue(
              float(np.linalg.norm((res[0][0, :] - res[0][i, :]))) > 1e-6)
          self.assertTrue(
              float(
                  np.linalg.norm((res[1].state_f00_b00_c[0, :] - res[1]
                                  .state_f00_b00_c[i, :]))) > 1e-6)

  def testGridLSTMCellWithFrequencyBlocks(self):
    with self.test_session() as sess:
      num_units = 8
      batch_size = 3
      input_size = 4
      feature_size = 2
      frequency_skip = 1
      num_frequency_blocks = [1, 1]
      total_blocks = num_frequency_blocks[0] + num_frequency_blocks[1]
      start_freqindex_list = [0, 2]
      end_freqindex_list = [2, 4]
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        cell = rnn_cell.GridLSTMCell(
            num_units=num_units,
            feature_size=feature_size,
            frequency_skip=frequency_skip,
            forget_bias=1.0,
            num_frequency_blocks=num_frequency_blocks,
            start_freqindex_list=start_freqindex_list,
            end_freqindex_list=end_freqindex_list,
            couple_input_forget_gates=True,
            state_is_tuple=True)
        inputs = constant_op.constant(
            np.array(
                [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
                dtype=np.float32),
            dtype=dtypes.float32)
        state_value = constant_op.constant(
            0.1 * np.ones(
                (batch_size, num_units), dtype=np.float32),
            dtype=dtypes.float32)
        init_state = cell.state_tuple_type(
            *([state_value, state_value] * total_blocks))
        output, state = cell(inputs, init_state)
        sess.run([variables.global_variables_initializer()])
        res = sess.run([output, state])
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is mostly just a
        # smoke test.
        self.assertEqual(res[0].shape,
                         (batch_size, num_units * total_blocks * 2))
        for ss in res[1]:
          self.assertEqual(ss.shape, (batch_size, num_units))
        # Different inputs so different outputs and states
        for i in range(1, batch_size):
          self.assertTrue(
              float(np.linalg.norm((res[0][0, :] - res[0][i, :]))) > 1e-6)
          self.assertTrue(
              float(
                  np.linalg.norm((res[1].state_f00_b00_c[0, :] - res[1]
                                  .state_f00_b00_c[i, :]))) > 1e-6)

  def testGridLstmCellWithCoupledInputForgetGates(self):
    num_units = 2
    batch_size = 3
    input_size = 4
    feature_size = 2
    frequency_skip = 1
    num_shifts = int((input_size - feature_size) / frequency_skip + 1)
    expected_output = np.array(
        [[0.416383, 0.416383, 0.403238, 0.403238, 0.524020, 0.524020,
          0.565425, 0.565425, 0.557865, 0.557865, 0.609699, 0.609699],
         [0.627331, 0.627331, 0.622393, 0.622393, 0.688342, 0.688342,
          0.708078, 0.708078, 0.694245, 0.694245, 0.715171, 0.715171],
         [0.711050, 0.711050, 0.709197, 0.709197, 0.736533, 0.736533,
          0.744264, 0.744264, 0.737390, 0.737390, 0.745250, 0.745250]],
        dtype=np.float32)
    expected_state = np.array(
        [[0.625556, 0.625556, 0.416383, 0.416383, 0.759134, 0.759134,
          0.524020, 0.524020, 0.798795, 0.798795, 0.557865, 0.557865],
         [0.875488, 0.875488, 0.627331, 0.627331, 0.936432, 0.936432,
          0.688342, 0.688342, 0.941961, 0.941961, 0.694245, 0.694245],
         [0.957327, 0.957327, 0.711050, 0.711050, 0.979522, 0.979522,
          0.736533, 0.736533, 0.980245, 0.980245, 0.737390, 0.737390]],
        dtype=np.float32)
    for state_is_tuple in [False, True]:
      with self.test_session() as sess:
        with variable_scope.variable_scope(
            "state_is_tuple" + str(state_is_tuple),
            initializer=init_ops.constant_initializer(0.5)):
          cell = rnn_cell.GridLSTMCell(
              num_units=num_units,
              feature_size=feature_size,
              frequency_skip=frequency_skip,
              forget_bias=1.0,
              num_frequency_blocks=[num_shifts],
              couple_input_forget_gates=True,
              state_is_tuple=state_is_tuple)
          inputs = constant_op.constant(
              np.array([[1., 1., 1., 1.],
                        [2., 2., 2., 2.],
                        [3., 3., 3., 3.]],
                       dtype=np.float32),
              dtype=dtypes.float32)
          if state_is_tuple:
            state_value = constant_op.constant(
                0.1 * np.ones(
                    (batch_size, num_units), dtype=np.float32),
                dtype=dtypes.float32)
            init_state = cell.state_tuple_type(
                *([state_value, state_value] * num_shifts))
          else:
            init_state = constant_op.constant(
                0.1 * np.ones(
                    (batch_size, num_units * num_shifts * 2), dtype=np.float32),
                dtype=dtypes.float32)
          output, state = cell(inputs, init_state)
          sess.run([variables.global_variables_initializer()])
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

  def testBidirectionGridLSTMCell(self):
    with self.test_session() as sess:
      num_units = 2
      batch_size = 3
      input_size = 4
      feature_size = 2
      frequency_skip = 1
      num_shifts = int((input_size - feature_size) / frequency_skip + 1)
      expected_output = np.array(
          [[0.464130, 0.464130, 0.419165, 0.419165, 0.593283, 0.593283,
            0.738350, 0.738350, 0.661638, 0.661638, 0.866774, 0.866774,
            0.520789, 0.520789, 0.476968, 0.476968, 0.604341, 0.604341,
            0.760207, 0.760207, 0.635773, 0.635773, 0.850218, 0.850218],
           [0.669636, 0.669636, 0.628966, 0.628966, 0.736057, 0.736057,
            0.895927, 0.895927, 0.755559, 0.755559, 0.954359, 0.954359,
            0.692621, 0.692621, 0.652363, 0.652363, 0.737517, 0.737517,
            0.899558, 0.899558, 0.745984, 0.745984, 0.946840, 0.946840],
           [0.751109, 0.751109, 0.711716, 0.711716, 0.778357, 0.778357,
            0.940779, 0.940779, 0.784530, 0.784530, 0.980604, 0.980604,
            0.759940, 0.759940, 0.720652, 0.720652, 0.778552, 0.778552,
            0.941606, 0.941606, 0.781035, 0.781035, 0.977731, 0.977731]],
          dtype=np.float32)
      expected_state = np.array(
          [[0.710660, 0.710660, 0.464130, 0.464130, 0.877293, 0.877293,
            0.593283, 0.593283, 0.958505, 0.958505, 0.661638, 0.661638,
            0.785405, 0.785405, 0.520789, 0.520789, 0.890836, 0.890836,
            0.604341, 0.604341, 0.928512, 0.928512, 0.635773, 0.635773],
           [0.967579, 0.967579, 0.669636, 0.669636, 1.038811, 1.038811,
            0.736057, 0.736057, 1.058201, 1.058201, 0.755559, 0.755559,
            0.993088, 0.993088, 0.692621, 0.692621, 1.040288, 1.040288,
            0.737517, 0.737517, 1.048773, 1.048773, 0.745984, 0.745984],
           [1.053842, 1.053842, 0.751109, 0.751109, 1.079919, 1.079919,
            0.778357, 0.778357, 1.085620, 1.085620, 0.784530, 0.784530,
            1.062455, 1.062455, 0.759940, 0.759940, 1.080101, 1.080101,
            0.778552, 0.778552, 1.082402, 1.082402, 0.781035, 0.781035]],
          dtype=np.float32)
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        cell = rnn_cell.BidirectionalGridLSTMCell(
            num_units=num_units,
            feature_size=feature_size,
            share_time_frequency_weights=True,
            frequency_skip=frequency_skip,
            forget_bias=1.0,
            num_frequency_blocks=[num_shifts])
        inputs = constant_op.constant(
            np.array([[1.0, 1.1, 1.2, 1.3],
                      [2.0, 2.1, 2.2, 2.3],
                      [3.0, 3.1, 3.2, 3.3]],
                     dtype=np.float32),
            dtype=dtypes.float32)
        state_value = constant_op.constant(
            0.1 * np.ones(
                (batch_size, num_units), dtype=np.float32),
            dtype=dtypes.float32)
        init_state = cell.state_tuple_type(
            *([state_value, state_value] * num_shifts * 2))
        output, state = cell(inputs, init_state)
        sess.run([variables.global_variables_initializer()])
        res = sess.run([output, state])
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is mostly just a
        # smoke test.
        self.assertEqual(res[0].shape, (batch_size, num_units * num_shifts * 4))
        self.assertAllClose(res[0], expected_output)
        # There should be num_shifts * 4 states in the tuple.
        self.assertEqual(len(res[1]), num_shifts * 4)
        # Checking the shape of each state to be batch_size * num_units
        for ss in res[1]:
          self.assertEqual(ss.shape[0], batch_size)
          self.assertEqual(ss.shape[1], num_units)
        self.assertAllClose(np.concatenate(res[1], axis=1), expected_state)

  def testBidirectionGridLSTMCellWithSliceOffset(self):
    with self.test_session() as sess:
      num_units = 2
      batch_size = 3
      input_size = 4
      feature_size = 2
      frequency_skip = 1
      num_shifts = int((input_size - feature_size) / frequency_skip + 1)
      expected_output = np.array(
          [[0.464130, 0.464130, 0.419165, 0.419165, 0.593283, 0.593283,
            0.738350, 0.738350, 0.661638, 0.661638, 0.866774, 0.866774,
            0.322645, 0.322645, 0.276068, 0.276068, 0.584654, 0.584654,
            0.690292, 0.690292, 0.640446, 0.640446, 0.840071, 0.840071],
           [0.669636, 0.669636, 0.628966, 0.628966, 0.736057, 0.736057,
            0.895927, 0.895927, 0.755559, 0.755559, 0.954359, 0.954359,
            0.493625, 0.493625, 0.449236, 0.449236, 0.730828, 0.730828,
            0.865996, 0.865996, 0.749429, 0.749429, 0.944958, 0.944958],
           [0.751109, 0.751109, 0.711716, 0.711716, 0.778357, 0.778357,
            0.940779, 0.940779, 0.784530, 0.784530, 0.980604, 0.980604,
            0.608587, 0.608587, 0.566683, 0.566683, 0.777345, 0.777345,
            0.925820, 0.925820, 0.782597, 0.782597, 0.976858, 0.976858]],
          dtype=np.float32)
      expected_state = np.array(
          [[0.710660, 0.710660, 0.464130, 0.464130, 0.877293, 0.877293,
            0.593283, 0.593283, 0.958505, 0.958505, 0.661638, 0.661638,
            0.516575, 0.516575, 0.322645, 0.322645, 0.866628, 0.866628,
            0.584654, 0.584654, 0.934002, 0.934002, 0.640446, 0.640446],
           [0.967579, 0.967579, 0.669636, 0.669636, 1.038811, 1.038811,
            0.736057, 0.736057, 1.058201, 1.058201, 0.755559, 0.755559,
            0.749836, 0.749836, 0.493625, 0.493625, 1.033488, 1.033488,
            0.730828, 0.730828, 1.052186, 1.052186, 0.749429, 0.749429],
           [1.053842, 1.053842, 0.751109, 0.751109, 1.079919, 1.079919,
            0.778357, 0.778357, 1.085620, 1.085620, 0.784530, 0.784530,
            0.895999, 0.895999, 0.608587, 0.608587, 1.078978, 1.078978,
            0.777345, 0.777345, 1.083843, 1.083843, 0.782597, 0.782597]],
          dtype=np.float32)
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        cell = rnn_cell.BidirectionalGridLSTMCell(
            num_units=num_units,
            feature_size=feature_size,
            share_time_frequency_weights=True,
            frequency_skip=frequency_skip,
            forget_bias=1.0,
            num_frequency_blocks=[num_shifts],
            backward_slice_offset=1)
        inputs = constant_op.constant(
            np.array([[1.0, 1.1, 1.2, 1.3],
                      [2.0, 2.1, 2.2, 2.3],
                      [3.0, 3.1, 3.2, 3.3]],
                     dtype=np.float32),
            dtype=dtypes.float32)
        state_value = constant_op.constant(
            0.1 * np.ones(
                (batch_size, num_units), dtype=np.float32),
            dtype=dtypes.float32)
        init_state = cell.state_tuple_type(
            *([state_value, state_value] * num_shifts * 2))
        output, state = cell(inputs, init_state)
        sess.run([variables.global_variables_initializer()])
        res = sess.run([output, state])
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is mostly just a
        # smoke test.
        self.assertEqual(res[0].shape, (batch_size, num_units * num_shifts * 4))
        self.assertAllClose(res[0], expected_output)
        # There should be num_shifts * 4 states in the tuple.
        self.assertEqual(len(res[1]), num_shifts * 4)
        # Checking the shape of each state to be batch_size * num_units
        for ss in res[1]:
          self.assertEqual(ss.shape[0], batch_size)
          self.assertEqual(ss.shape[1], num_units)
        self.assertAllClose(np.concatenate(res[1], axis=1), expected_state)

  def testAttentionCellWrapperFailures(self):
    with self.assertRaisesRegexp(TypeError,
                                 "The parameter cell is not RNNCell."):
      rnn_cell.AttentionCellWrapper(None, 0)

    num_units = 8
    for state_is_tuple in [False, True]:
      with ops.Graph().as_default():
        lstm_cell = core_rnn_cell_impl.BasicLSTMCell(
            num_units, state_is_tuple=state_is_tuple)
        with self.assertRaisesRegexp(
            ValueError, "attn_length should be greater than zero, got 0"):
          rnn_cell.AttentionCellWrapper(
              lstm_cell, 0, state_is_tuple=state_is_tuple)
        with self.assertRaisesRegexp(
            ValueError, "attn_length should be greater than zero, got -1"):
          rnn_cell.AttentionCellWrapper(
              lstm_cell, -1, state_is_tuple=state_is_tuple)
      with ops.Graph().as_default():
        lstm_cell = core_rnn_cell_impl.BasicLSTMCell(
            num_units, state_is_tuple=True)
        with self.assertRaisesRegexp(
            ValueError, "Cell returns tuple of states, but the flag "
            "state_is_tuple is not set. State size is: *"):
          rnn_cell.AttentionCellWrapper(lstm_cell, 4, state_is_tuple=False)

  def testAttentionCellWrapperZeros(self):
    num_units = 8
    attn_length = 16
    batch_size = 3
    input_size = 4
    for state_is_tuple in [False, True]:
      with ops.Graph().as_default():
        with self.test_session() as sess:
          with variable_scope.variable_scope("state_is_tuple_" + str(
              state_is_tuple)):
            lstm_cell = core_rnn_cell_impl.BasicLSTMCell(
                num_units, state_is_tuple=state_is_tuple)
            cell = rnn_cell.AttentionCellWrapper(
                lstm_cell, attn_length, state_is_tuple=state_is_tuple)
            if state_is_tuple:
              zeros = array_ops.zeros([batch_size, num_units], dtype=np.float32)
              attn_state_zeros = array_ops.zeros(
                  [batch_size, attn_length * num_units], dtype=np.float32)
              zero_state = ((zeros, zeros), zeros, attn_state_zeros)
            else:
              zero_state = array_ops.zeros(
                  [
                      batch_size,
                      num_units * 2 + attn_length * num_units + num_units
                  ],
                  dtype=np.float32)
            inputs = array_ops.zeros(
                [batch_size, input_size], dtype=dtypes.float32)
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
              self.assertEquals(state.get_shape(), [
                  batch_size,
                  num_units * 2 + num_units + attn_length * num_units
              ])
              tensors = [output, state]
            zero_result = sum(
                [math_ops.reduce_sum(math_ops.abs(x)) for x in tensors])
            sess.run(variables.global_variables_initializer())
            self.assertTrue(sess.run(zero_result) < 1e-6)

  def testAttentionCellWrapperValues(self):
    num_units = 8
    attn_length = 16
    batch_size = 3
    for state_is_tuple in [False, True]:
      with ops.Graph().as_default():
        with self.test_session() as sess:
          with variable_scope.variable_scope("state_is_tuple_" + str(
              state_is_tuple)):
            lstm_cell = core_rnn_cell_impl.BasicLSTMCell(
                num_units, state_is_tuple=state_is_tuple)
            cell = rnn_cell.AttentionCellWrapper(
                lstm_cell, attn_length, state_is_tuple=state_is_tuple)
            if state_is_tuple:
              zeros = constant_op.constant(
                  0.1 * np.ones(
                      [batch_size, num_units], dtype=np.float32),
                  dtype=dtypes.float32)
              attn_state_zeros = constant_op.constant(
                  0.1 * np.ones(
                      [batch_size, attn_length * num_units], dtype=np.float32),
                  dtype=dtypes.float32)
              zero_state = ((zeros, zeros), zeros, attn_state_zeros)
            else:
              zero_state = constant_op.constant(
                  0.1 * np.ones(
                      [
                          batch_size,
                          num_units * 2 + num_units + attn_length * num_units
                      ],
                      dtype=np.float32),
                  dtype=dtypes.float32)
            inputs = constant_op.constant(
                np.array(
                    [[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]],
                    dtype=np.float32),
                dtype=dtypes.float32)
            output, state = cell(inputs, zero_state)
            if state_is_tuple:
              concat_state = array_ops.concat(
                  [state[0][0], state[0][1], state[1], state[2]], 1)
            else:
              concat_state = state
            sess.run(variables.global_variables_initializer())
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
        [[1.068372, 0.45496, -0.678277, 0.340538],
         [1.018088, 0.378983, -0.572179, 0.268591]],
        dtype=np.float32)
    expected_state = np.array(
        [[0.74946702, 0.34681597, 0.26474735, 1.06485605, 0.38465962,
          0.11420801, 0.10272158, 0.30925757, 0.63899988, 0.7181077,
          0.47534478, 0.33715725, 0.58086717, 0.49446869, 0.7641536,
          0.12814975, 0.92231739, 0.89857256, 0.21889746, 0.38442063,
          0.53481543, 0.8876909, 0.45823169, 0.5905602, 0.78038228,
          0.56501579, 0.03971386, 0.09870267, 0.8074435, 0.66821432,
          0.99211812, 0.12295902, 1.14606023, 0.34370938, -0.79251152,
          0.51843399],
         [0.5179342, 0.48682183, -0.25426468, 0.96810579, 0.28809637,
          0.13607743, -0.11446252, 0.26792109, 0.78047138, 0.63460857,
          0.49122369, 0.52007174, 0.73000264, 0.66986895, 0.73576689,
          0.86301267, 0.87887371, 0.35185754, 0.93417215, 0.64732957,
          0.63173044, 0.66627824, 0.53644657, 0.20477486, 0.98458421,
          0.38277245, 0.03746676, 0.92510188, 0.57714164, 0.84932971,
          0.36127412, 0.12125921, 1.1362772, 0.34361625, -0.78150457,
          0.70582712]],
        dtype=np.float32)
    seed = 12345
    random_seed.set_random_seed(seed)
    for state_is_tuple in [False, True]:
      with session.Session() as sess:
        with variable_scope.variable_scope(
            "state_is_tuple", reuse=state_is_tuple,
            initializer=init_ops.glorot_uniform_initializer()):
          lstm_cell = core_rnn_cell_impl.BasicLSTMCell(
              num_units, state_is_tuple=state_is_tuple)
          cell = rnn_cell.AttentionCellWrapper(
              lstm_cell, attn_length, state_is_tuple=state_is_tuple)
          zeros1 = random_ops.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed + 1)
          zeros2 = random_ops.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed + 2)
          zeros3 = random_ops.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed + 3)
          attn_state_zeros = random_ops.random_uniform(
              (batch_size, attn_length * num_units), 0.0, 1.0, seed=seed + 4)
          zero_state = ((zeros1, zeros2), zeros3, attn_state_zeros)
          if not state_is_tuple:
            zero_state = array_ops.concat([
                zero_state[0][0], zero_state[0][1], zero_state[1], zero_state[2]
            ], 1)
          inputs = random_ops.random_uniform(
              (batch_size, num_units), 0.0, 1.0, seed=seed + 5)
          output, state = cell(inputs, zero_state)
          if state_is_tuple:
            state = array_ops.concat(
                [state[0][0], state[0][1], state[1], state[2]], 1)
          sess.run(variables.global_variables_initializer())
          self.assertAllClose(sess.run(output), expected_output)
          self.assertAllClose(sess.run(state), expected_state)


class LayerNormBasicLSTMCellTest(test.TestCase):

  # NOTE: all the values in the current test case have been calculated.

  def testBasicLSTMCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        c0 = array_ops.zeros([1, 2])
        h0 = array_ops.zeros([1, 2])
        state0 = core_rnn_cell_impl.LSTMStateTuple(c0, h0)
        c1 = array_ops.zeros([1, 2])
        h1 = array_ops.zeros([1, 2])
        state1 = core_rnn_cell_impl.LSTMStateTuple(c1, h1)
        state = (state0, state1)
        cell = rnn_cell.LayerNormBasicLSTMCell(2)
        cell = core_rnn_cell_impl.MultiRNNCell([cell] * 2)
        g, out_m = cell(x, state)
        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, out_m], {
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

      with variable_scope.variable_scope(
          "other", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros(
            [1, 3])  # Test BasicLSTMCell with input_size != num_units.
        c = array_ops.zeros([1, 2])
        h = array_ops.zeros([1, 2])
        state = core_rnn_cell_impl.LSTMStateTuple(c, h)
        cell = rnn_cell.LayerNormBasicLSTMCell(2)
        g, out_m = cell(x, state)
        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, out_m], {
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
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        c0 = array_ops.zeros([1, 2])
        h0 = array_ops.zeros([1, 2])
        state0 = core_rnn_cell_impl.LSTMStateTuple(c0, h0)
        c1 = array_ops.zeros([1, 2])
        h1 = array_ops.zeros([1, 2])
        state1 = core_rnn_cell_impl.LSTMStateTuple(c1, h1)
        cell = rnn_cell.LayerNormBasicLSTMCell(2)
        cell = core_rnn_cell_impl.MultiRNNCell([cell] * 2)
        h, (s0, s1) = cell(x, (state0, state1))
        sess.run([variables.global_variables_initializer()])
        res = sess.run([h, s0, s1], {
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
      return delta < 10**(-digits)

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
      with variable_scope.variable_scope(
          "other", initializer=init_ops.constant_initializer(1)):
        x = array_ops.zeros([1, 5])
        c = array_ops.zeros([1, 5])
        h = array_ops.zeros([1, 5])
        state = core_rnn_cell_impl.LSTMStateTuple(c, h)
        cell = rnn_cell.LayerNormBasicLSTMCell(
            num_units, layer_norm=False, dropout_keep_prob=keep_prob)

        g, s = cell(x, state)
        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s], {
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
  test.main()
