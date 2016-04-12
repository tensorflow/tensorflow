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

"""Tests for RNN cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.contrib.rnn.Load()


class RNNCellTest(tf.test.TestCase):

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
                       {x.name: np.array([[1., 1., 1., 1.,],
                                          [2., 2., 2., 2.], [3., 3., 3., 3.]]),
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
            frequency_skip=frequency_skip, forget_bias=1.0)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([output, state],
                       {x.name: np.array([[1., 1., 1., 1.,],
                                          [2., 2., 2., 2.], [3., 3., 3., 3.]]),
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

  def _testLSTMCellBlock(self, use_gpu):
    with self.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        m = tf.zeros([1, 8])
        g, out_m = tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.LSTMCellBlock(2)] * 2)(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, out_m], {x.name: np.array([[1., 1.]]),
                                    m.name: 0.1 * np.ones([1, 8])})
        self.assertEqual(len(res), 2)
        self.assertAllClose(res[0], [[0.24024698, 0.24024698]])
        # These numbers are from testBasicLSTMCell and only test c/h.
        expected_mem = np.array([[0.68967271, 0.68967271,
                                  0.44848421, 0.44848421,
                                  0.39897051, 0.39897051,
                                  0.24024698, 0.24024698]])
        self.assertAllClose(res[1].flatten(), expected_mem.flatten())

  def testLSTMCellBlock(self):
    self._testLSTMCellBlock(use_gpu=False)
    self._testLSTMCellBlock(use_gpu=True)

  def testLSTMBasicToBlockCell(self):
    with self.test_session() as sess:
      x = tf.zeros([1, 2])
      x_values = np.random.randn(1, 2)

      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=19890212)
      with tf.variable_scope("basic", initializer=initializer):
        m = tf.zeros([1, 8])
        g, out_m = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(2)] * 2)(x, m)
        sess.run([tf.initialize_all_variables()])
        basic_res = sess.run([g, out_m], {x.name: x_values,
                                          m.name: 0.1 * np.ones([1, 8])})

      with tf.variable_scope("block", initializer=initializer):
        m = tf.zeros([1, 8])
        g, out_m = tf.nn.rnn_cell.MultiRNNCell(
            [tf.contrib.rnn.LSTMCellBlock(2)] * 2)(x, m)
        sess.run([tf.initialize_all_variables()])
        block_res = sess.run([g, out_m], {x.name: x_values,
                                          m.name: 0.1 * np.ones([1, 8])})

      self.assertAllClose(basic_res[0], block_res[0])
      self.assertAllClose(basic_res[1].flatten(), block_res[1].flatten())

  def testLSTMBasicToBlock(self):
    with self.test_session() as sess:
      x = tf.zeros([1, 2])
      x_values = np.random.randn(1, 2)

      initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=19890212)
      with tf.variable_scope("basic", initializer=initializer):
        m = tf.zeros([1, 8])
        g, out_m = tf.nn.rnn_cell.BasicLSTMCell(4)(x, m)
        sess.run([tf.initialize_all_variables()])
        basic_res = sess.run([g, out_m], {x.name: x_values,
                                          m.name: 0.1 * np.ones([1, 8])})

      with tf.variable_scope("block", initializer=initializer):
        m = tf.zeros([1, 8])
        g, out_m = tf.contrib.rnn.lstm_block([x], 4, initial_state=m)
        sess.run([tf.initialize_all_variables()])
        block_res = sess.run(g + out_m, {x.name: x_values,
                                          m.name: 0.1 * np.ones([1, 8])})

      self.assertAllClose(basic_res[0], block_res[0])
      self.assertAllClose(basic_res[1].flatten(), block_res[1].flatten())


if __name__ == "__main__":
  tf.test.main()
