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

"""Tests for GridRNN cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class GridRNNCellTest(tf.test.TestCase):

  def testGrid2BasicLSTMCell(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.2)):
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 8])
        cell = tf.contrib.grid_rnn.Grid2BasicLSTMCell(2, input_size=3)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 3)
        self.assertEqual(cell.state_size, 8)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1., 1.]]),
                             m.name: np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[ 0.36617181, 0.36617181]])
        self.assertAllClose(res[1], [[ 0.71053141, 0.71053141, 0.36617181, 0.36617181,
                                       0.72320831, 0.80555487, 0.39102408, 0.42150158]])

        # make sure variables are reused properly
        res = sess.run([g, s], {x.name: np.array([[1., 1., 1.]]), m.name: res[1]})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[0.36703536, 0.36703536]])
        self.assertAllClose(res[1], [[ 0.71200621, 0.71200621, 0.36703536, 0.36703536,
                                       0.80941606, 0.87550586, 0.40108523, 0.42199609]])

  def testGrid2BasicLSTMCellTied(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.2)):
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 8])
        cell = tf.contrib.grid_rnn.Grid2BasicLSTMCell(2, input_size=3, tied=True)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 3)
        self.assertEqual(cell.state_size, 8)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1., 1.]]),
                             m.name: np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[ 0.36617181, 0.36617181]])
        self.assertAllClose(res[1], [[ 0.71053141, 0.71053141, 0.36617181, 0.36617181,
                                       0.72320831, 0.80555487, 0.39102408, 0.42150158]])

  def testGrid2BasicLSTMCellWithRelu(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.2)):
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 4])
        cell = tf.contrib.grid_rnn.Grid2BasicLSTMCell(2, input_size=3, tied=False, non_recurrent_fn=tf.nn.relu)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 3)
        self.assertEqual(cell.state_size, 4)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1., 1.]]),
                             m.name: np.array([[0.1, 0.2, 0.3, 0.4]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 4))
        self.assertAllClose(res[0], [[ 0.31667367, 0.31667367]])
        self.assertAllClose(res[1], [[ 0.29530135, 0.37520045, 0.17044567, 0.21292259]])

  """
  LSTMCell
  """

  def testGrid2LSTMCell(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 8])
        cell = tf.contrib.grid_rnn.Grid2LSTMCell(2, input_size=3, use_peepholes=True)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 3)
        self.assertEqual(cell.state_size, 8)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1., 1.]]),
                                m.name: np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[ 0.95686918, 0.95686918]])
        self.assertAllClose(res[1], [[ 2.41515064, 2.41515064, 0.95686918, 0.95686918,
                                       1.38917875, 1.49043763, 0.83884692, 0.86036491]])

  def testGrid2LSTMCellTied(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 8])
        cell = tf.contrib.grid_rnn.Grid2LSTMCell(2, input_size=3, tied=True, use_peepholes=True)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 3)
        self.assertEqual(cell.state_size, 8)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1., 1.]]),
                                m.name: np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[ 0.95686918, 0.95686918]])
        self.assertAllClose(res[1], [[ 2.41515064, 2.41515064, 0.95686918, 0.95686918,
                                       1.38917875, 1.49043763, 0.83884692, 0.86036491]])

  def testGrid2LSTMCellWithRelu(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 4])
        cell = tf.contrib.grid_rnn.Grid2LSTMCell(2, input_size=3, use_peepholes=True, non_recurrent_fn=tf.nn.relu)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 3)
        self.assertEqual(cell.state_size, 4)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1., 1.]]),
                                m.name: np.array([[0.1, 0.2, 0.3, 0.4]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 4))
        self.assertAllClose(res[0], [[ 2.1831727, 2.1831727]])
        self.assertAllClose(res[1], [[ 0.92270052, 1.02325559, 0.66159075, 0.70475441]])

  """
  RNNCell
  """

  def testGrid2BasicRNNCell(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([2, 2])
        m = tf.zeros([2, 4])
        cell = tf.contrib.grid_rnn.Grid2BasicRNNCell(2)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 2)
        self.assertEqual(cell.state_size, 4)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1.], [2., 2.]]),
                                m.name: np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]])})
        self.assertEqual(res[0].shape, (2, 2))
        self.assertEqual(res[1].shape, (2, 4))
        # print(res[0], res[1])
      #TODO: check correctness of this

  def testGrid2BasicRNNCellWithRelu(self):
    with self.test_session() as sess:
      with tf.variable_scope("root", initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 2])
        m = tf.zeros([1, 2])
        cell = tf.contrib.grid_rnn.Grid2BasicRNNCell(2, non_recurrent_fn=tf.nn.relu)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 2)
        self.assertEqual(cell.state_size, 2)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1.]]), m.name: np.array([[0.1, 0.1]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 2))
        # print(res[0], res[1])
      #TODO: check correctness of this

if __name__ == "__main__":
  tf.test.main()
