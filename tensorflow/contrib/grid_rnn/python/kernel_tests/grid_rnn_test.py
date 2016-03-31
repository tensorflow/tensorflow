# Copyright 2016 Google Inc. All Rights Reserved.
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
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.2)) as root_scope:
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

        # emulate a training loop, where we call cell() multiple times
        root_scope.reuse_variables()
        g2, s2 = cell(x, m)
        res = sess.run([g2, s2], {x.name: np.array([[2., 2., 2.]]), m.name: res[1]})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[0.58847463, 0.58847463]])
        self.assertAllClose(res[1], [[1.40469193, 1.40469193, 0.58847463, 0.58847463,
                                      0.97726452, 1.04626071, 0.4927212, 0.51137757]])

  def testGrid2BasicLSTMCellTied(self):
    with self.test_session() as sess:
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.2)):
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

        res = sess.run([g, s], {x.name: np.array([[1., 1., 1.]]), m.name: res[1]})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[0.36703536, 0.36703536]])
        self.assertAllClose(res[1], [[0.71200621, 0.71200621, 0.36703536, 0.36703536,
                                      0.80941606, 0.87550586, 0.40108523, 0.42199609]])

  def testGrid2BasicLSTMCellWithRelu(self):
    with self.test_session() as sess:
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.2)):
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
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.5)):
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
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.5)):
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
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.5)):
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
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.5)):
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
        self.assertAllClose(res[0], [[0.94685763, 0.94685763],
                                     [0.99480951, 0.99480951]])
        self.assertAllClose(res[1], [[0.94685763, 0.94685763, 0.80049908, 0.80049908],
                                     [0.99480951, 0.99480951, 0.97574311, 0.97574311]])

  def testGrid2BasicRNNCellTied(self):
    with self.test_session() as sess:
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([2, 2])
        m = tf.zeros([2, 4])
        cell = tf.contrib.grid_rnn.Grid2BasicRNNCell(2, tied=True)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 2)
        self.assertEqual(cell.state_size, 4)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1.], [2., 2.]]),
                                m.name: np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]])})
        self.assertEqual(res[0].shape, (2, 2))
        self.assertEqual(res[1].shape, (2, 4))
        self.assertAllClose(res[0], [[0.94685763, 0.94685763],
                                     [0.99480951, 0.99480951]])
        self.assertAllClose(res[1], [[0.94685763, 0.94685763, 0.80049908, 0.80049908],
                                     [0.99480951, 0.99480951, 0.97574311, 0.97574311]])

  def testGrid2BasicRNNCellWithRelu(self):
    with self.test_session() as sess:
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.5)):
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
        self.assertAllClose(res[0], [[1.80049896, 1.80049896]])
        self.assertAllClose(res[1], [[0.80049896, 0.80049896]])

  """
  1-LSTM
  """

  def testGrid1LSTMCell(self):
    with self.test_session() as sess:
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.5)) as root_scope:
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 4])
        cell = tf.contrib.grid_rnn.Grid1LSTMCell(2, input_size=3, use_peepholes=True)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 3)
        self.assertEqual(cell.state_size, 4)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1., 1.]]),
                                m.name: np.array([[0.1, 0.2, 0.3, 0.4]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 4))
        self.assertAllClose(res[0], [[0.91287315, 0.91287315]])
        self.assertAllClose(res[1], [[2.26285243, 2.26285243, 0.91287315, 0.91287315]])

        root_scope.reuse_variables()

        # for 1LSTM, next iterations we will set input is None
        g2, s2 = cell(None, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g2, s2], {m.name: res[1]})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 4))
        self.assertAllClose(res[0], [[0.9032144, 0.9032144]])
        self.assertAllClose(res[1], [[2.79966092, 2.79966092, 0.9032144, 0.9032144]])

        g3, s3 = cell(None, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g3, s3], {m.name: res[1]})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 4))
        self.assertAllClose(res[0], [[0.92727238, 0.92727238]])
        self.assertAllClose(res[1], [[3.3529923, 3.3529923, 0.92727238, 0.92727238]])

  """
  3-LSTM
  """
  def testGrid3LSTMCell(self):
    with self.test_session() as sess:
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([1, 3])
        m = tf.zeros([1, 12])
        cell = tf.contrib.grid_rnn.Grid3LSTMCell(2, input_size=3, use_peepholes=True)
        self.assertEqual(cell.output_size, 2)
        self.assertEqual(cell.input_size, 3)
        self.assertEqual(cell.state_size, 12)

        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1., 1.]]),
                                m.name: np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 12))

        self.assertAllClose(res[0], [[0.96892911, 0.96892911]])
        self.assertAllClose(res[1], [[2.45227885, 2.45227885, 0.96892911, 0.96892911,
                                      1.33592629, 1.4373529, 0.80867189, 0.83247656,
                                      0.7317788, 0.63205892, 0.56548983, 0.50446129]])

  """
  Edge cases
  """
  def testGridRNNEdgeCasesLikeRelu(self):
    with self.test_session() as sess:
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.5)):
        x = tf.zeros([3, 2])
        m = tf.zeros([0, 0])

        # this is equivalent to relu
        cell = tf.contrib.grid_rnn.GridRNNCell(num_units=2, num_dims=1, input_dims=0, output_dims=0,
                                               non_recurrent_dims=0, non_recurrent_fn=tf.nn.relu)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., -1.], [-2, 1], [2, -1]])})
        self.assertEqual(res[0].shape, (3, 2))
        self.assertEqual(res[1].shape, (0, 0))
        self.assertAllClose(res[0], [[0, 0], [0, 0], [0.5, 0.5]])

  def testGridRNNEdgeCasesNoOutput(self):
    with self.test_session() as sess:
      with tf.variable_scope('root', initializer=tf.constant_initializer(0.5)):
        x =  tf.zeros([1, 2])
        m = tf.zeros([1, 4])

        # This cell produces no output
        cell = tf.contrib.grid_rnn.GridRNNCell(num_units=2, num_dims=2, input_dims=0, output_dims=None,
                                               non_recurrent_dims=0, non_recurrent_fn=tf.nn.relu)
        g, s = cell(x, m)
        sess.run([tf.initialize_all_variables()])
        res = sess.run([g, s], {x.name: np.array([[1., 1.]]),
                                m.name: np.array([[0.1, 0.1, 0.1, 0.1]])})
        self.assertEqual(res[0].shape, (0, 0))
        self.assertEqual(res[1].shape, (1, 4))

if __name__ == "__main__":
  tf.test.main()
