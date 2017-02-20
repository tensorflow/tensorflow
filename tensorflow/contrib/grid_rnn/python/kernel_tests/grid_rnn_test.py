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
"""Tests for GridRNN cells."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

from tensorflow.contrib.grid_rnn.python.ops import grid_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class GridRNNCellTest(test.TestCase):

  def testGrid2BasicLSTMCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.2)) as root_scope:
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 8])
        cell = grid_rnn_cell.Grid2BasicLSTMCell(2)
        self.assertEqual(cell.state_size, 8)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (1, 2))
        self.assertEqual(s.get_shape(), (1, 8))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s], {
            x: np.array([[1., 1., 1.]]),
            m: np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        })
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[0.36617181, 0.36617181]])
        self.assertAllClose(res[1], [[0.71053141, 0.71053141, 0.36617181,
                                      0.36617181, 0.72320831, 0.80555487,
                                      0.39102408, 0.42150158]])

        # emulate a loop through the input sequence,
        # where we call cell() multiple times
        root_scope.reuse_variables()
        g2, s2 = cell(x, m)
        self.assertEqual(g2.get_shape(), (1, 2))
        self.assertEqual(s2.get_shape(), (1, 8))

        res = sess.run([g2, s2], {x: np.array([[2., 2., 2.]]), m: res[1]})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[0.58847463, 0.58847463]])
        self.assertAllClose(res[1], [[1.40469193, 1.40469193, 0.58847463,
                                      0.58847463, 0.97726452, 1.04626071,
                                      0.4927212, 0.51137757]])

  def testGrid2BasicLSTMCellTied(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.2)):
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 8])
        cell = grid_rnn_cell.Grid2BasicLSTMCell(2, tied=True)
        self.assertEqual(cell.state_size, 8)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (1, 2))
        self.assertEqual(s.get_shape(), (1, 8))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s], {
            x: np.array([[1., 1., 1.]]),
            m: np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        })
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[0.36617181, 0.36617181]])
        self.assertAllClose(res[1], [[0.71053141, 0.71053141, 0.36617181,
                                      0.36617181, 0.72320831, 0.80555487,
                                      0.39102408, 0.42150158]])

        res = sess.run([g, s], {x: np.array([[1., 1., 1.]]), m: res[1]})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[0.36703536, 0.36703536]])
        self.assertAllClose(res[1], [[0.71200621, 0.71200621, 0.36703536,
                                      0.36703536, 0.80941606, 0.87550586,
                                      0.40108523, 0.42199609]])

  def testGrid2BasicLSTMCellWithRelu(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.2)):
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 4])
        cell = grid_rnn_cell.Grid2BasicLSTMCell(
            2, tied=False, non_recurrent_fn=nn_ops.relu)
        self.assertEqual(cell.state_size, 4)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (1, 2))
        self.assertEqual(s.get_shape(), (1, 4))

        sess.run([variables.global_variables_initializer()])
        res = sess.run(
            [g, s],
            {x: np.array([[1., 1., 1.]]),
             m: np.array([[0.1, 0.2, 0.3, 0.4]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 4))
        self.assertAllClose(res[0], [[0.31667367, 0.31667367]])
        self.assertAllClose(res[1], [[0.29530135, 0.37520045, 0.17044567,
                                      0.21292259]])

  """LSTMCell
  """

  def testGrid2LSTMCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 8])
        cell = grid_rnn_cell.Grid2LSTMCell(2, use_peepholes=True)
        self.assertEqual(cell.state_size, 8)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (1, 2))
        self.assertEqual(s.get_shape(), (1, 8))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s], {
            x: np.array([[1., 1., 1.]]),
            m: np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        })
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[0.95686918, 0.95686918]])
        self.assertAllClose(res[1], [[2.41515064, 2.41515064, 0.95686918,
                                      0.95686918, 1.38917875, 1.49043763,
                                      0.83884692, 0.86036491]])

  def testGrid2LSTMCellTied(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 8])
        cell = grid_rnn_cell.Grid2LSTMCell(2, tied=True, use_peepholes=True)
        self.assertEqual(cell.state_size, 8)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (1, 2))
        self.assertEqual(s.get_shape(), (1, 8))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s], {
            x: np.array([[1., 1., 1.]]),
            m: np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
        })
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 8))
        self.assertAllClose(res[0], [[0.95686918, 0.95686918]])
        self.assertAllClose(res[1], [[2.41515064, 2.41515064, 0.95686918,
                                      0.95686918, 1.38917875, 1.49043763,
                                      0.83884692, 0.86036491]])

  def testGrid2LSTMCellWithRelu(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 4])
        cell = grid_rnn_cell.Grid2LSTMCell(
            2, use_peepholes=True, non_recurrent_fn=nn_ops.relu)
        self.assertEqual(cell.state_size, 4)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (1, 2))
        self.assertEqual(s.get_shape(), (1, 4))

        sess.run([variables.global_variables_initializer()])
        res = sess.run(
            [g, s],
            {x: np.array([[1., 1., 1.]]),
             m: np.array([[0.1, 0.2, 0.3, 0.4]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 4))
        self.assertAllClose(res[0], [[2.1831727, 2.1831727]])
        self.assertAllClose(res[1], [[0.92270052, 1.02325559, 0.66159075,
                                      0.70475441]])

  """RNNCell
  """

  def testGrid2BasicRNNCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([2, 2])
        m = array_ops.zeros([2, 4])
        cell = grid_rnn_cell.Grid2BasicRNNCell(2)
        self.assertEqual(cell.state_size, 4)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (2, 2))
        self.assertEqual(s.get_shape(), (2, 4))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s], {
            x: np.array([[1., 1.], [2., 2.]]),
            m: np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]])
        })
        self.assertEqual(res[0].shape, (2, 2))
        self.assertEqual(res[1].shape, (2, 4))
        self.assertAllClose(res[0], [[0.94685763, 0.94685763],
                                     [0.99480951, 0.99480951]])
        self.assertAllClose(res[1],
                            [[0.94685763, 0.94685763, 0.80049908, 0.80049908],
                             [0.99480951, 0.99480951, 0.97574311, 0.97574311]])

  def testGrid2BasicRNNCellTied(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([2, 2])
        m = array_ops.zeros([2, 4])
        cell = grid_rnn_cell.Grid2BasicRNNCell(2, tied=True)
        self.assertEqual(cell.state_size, 4)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (2, 2))
        self.assertEqual(s.get_shape(), (2, 4))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s], {
            x: np.array([[1., 1.], [2., 2.]]),
            m: np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]])
        })
        self.assertEqual(res[0].shape, (2, 2))
        self.assertEqual(res[1].shape, (2, 4))
        self.assertAllClose(res[0], [[0.94685763, 0.94685763],
                                     [0.99480951, 0.99480951]])
        self.assertAllClose(res[1],
                            [[0.94685763, 0.94685763, 0.80049908, 0.80049908],
                             [0.99480951, 0.99480951, 0.97574311, 0.97574311]])

  def testGrid2BasicRNNCellWithRelu(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 2])
        cell = grid_rnn_cell.Grid2BasicRNNCell(2, non_recurrent_fn=nn_ops.relu)
        self.assertEqual(cell.state_size, 2)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (1, 2))
        self.assertEqual(s.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s],
                       {x: np.array([[1., 1.]]),
                        m: np.array([[0.1, 0.1]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 2))
        self.assertAllClose(res[0], [[1.80049896, 1.80049896]])
        self.assertAllClose(res[1], [[0.80049896, 0.80049896]])

  """1-LSTM
  """

  def testGrid1LSTMCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)) as root_scope:
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 4])
        cell = grid_rnn_cell.Grid1LSTMCell(2, use_peepholes=True)
        self.assertEqual(cell.state_size, 4)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (1, 2))
        self.assertEqual(s.get_shape(), (1, 4))

        sess.run([variables.global_variables_initializer()])
        res = sess.run(
            [g, s],
            {x: np.array([[1., 1., 1.]]),
             m: np.array([[0.1, 0.2, 0.3, 0.4]])})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 4))
        self.assertAllClose(res[0], [[0.91287315, 0.91287315]])
        self.assertAllClose(res[1],
                            [[2.26285243, 2.26285243, 0.91287315, 0.91287315]])

        root_scope.reuse_variables()

        x2 = array_ops.zeros([0, 0])
        g2, s2 = cell(x2, m)
        self.assertEqual(g2.get_shape(), (1, 2))
        self.assertEqual(s2.get_shape(), (1, 4))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g2, s2], {m: res[1]})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 4))
        self.assertAllClose(res[0], [[0.9032144, 0.9032144]])
        self.assertAllClose(res[1],
                            [[2.79966092, 2.79966092, 0.9032144, 0.9032144]])

        g3, s3 = cell(x2, m)
        self.assertEqual(g3.get_shape(), (1, 2))
        self.assertEqual(s3.get_shape(), (1, 4))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g3, s3], {m: res[1]})
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 4))
        self.assertAllClose(res[0], [[0.92727238, 0.92727238]])
        self.assertAllClose(res[1],
                            [[3.3529923, 3.3529923, 0.92727238, 0.92727238]])

  """3-LSTM
  """

  def testGrid3LSTMCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 12])
        cell = grid_rnn_cell.Grid3LSTMCell(2, use_peepholes=True)
        self.assertEqual(cell.state_size, 12)

        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (1, 2))
        self.assertEqual(s.get_shape(), (1, 12))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s], {
            x:
                np.array([[1., 1., 1.]]),
            m:
                np.array([[
                    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3,
                    -0.4
                ]])
        })
        self.assertEqual(res[0].shape, (1, 2))
        self.assertEqual(res[1].shape, (1, 12))

        self.assertAllClose(res[0], [[0.96892911, 0.96892911]])
        self.assertAllClose(res[1], [[2.45227885, 2.45227885, 0.96892911,
                                      0.96892911, 1.33592629, 1.4373529,
                                      0.80867189, 0.83247656, 0.7317788,
                                      0.63205892, 0.56548983, 0.50446129]])

  """Edge cases
  """

  def testGridRNNEdgeCasesLikeRelu(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([3, 2])
        m = array_ops.zeros([0, 0])

        # this is equivalent to relu
        cell = grid_rnn_cell.GridRNNCell(
            num_units=2,
            num_dims=1,
            input_dims=0,
            output_dims=0,
            non_recurrent_dims=0,
            non_recurrent_fn=nn_ops.relu)
        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (3, 2))
        self.assertEqual(s.get_shape(), (0, 0))

        sess.run([variables.global_variables_initializer()])
        res = sess.run([g, s], {x: np.array([[1., -1.], [-2, 1], [2, -1]])})
        self.assertEqual(res[0].shape, (3, 2))
        self.assertEqual(res[1].shape, (0, 0))
        self.assertAllClose(res[0], [[0, 0], [0, 0], [0.5, 0.5]])

  def testGridRNNEdgeCasesNoOutput(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 4])

        # This cell produces no output
        cell = grid_rnn_cell.GridRNNCell(
            num_units=2,
            num_dims=2,
            input_dims=0,
            output_dims=None,
            non_recurrent_dims=0,
            non_recurrent_fn=nn_ops.relu)
        g, s = cell(x, m)
        self.assertEqual(g.get_shape(), (0, 0))
        self.assertEqual(s.get_shape(), (1, 4))

        sess.run([variables.global_variables_initializer()])
        res = sess.run(
            [g, s],
            {x: np.array([[1., 1.]]),
             m: np.array([[0.1, 0.1, 0.1, 0.1]])})
        self.assertEqual(res[0].shape, (0, 0))
        self.assertEqual(res[1].shape, (1, 4))

  """Test with tf.nn.rnn
  """

  def testGrid2LSTMCellWithRNN(self):
    batch_size = 3
    input_size = 5
    max_length = 6  # unrolled up to this length
    num_units = 2

    with variable_scope.variable_scope(
        'root', initializer=init_ops.constant_initializer(0.5)):
      cell = grid_rnn_cell.Grid2LSTMCell(num_units=num_units)

      inputs = max_length * [
          array_ops.placeholder(
              dtypes.float32, shape=(batch_size, input_size))
      ]

      outputs, state = core_rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

    self.assertEqual(len(outputs), len(inputs))
    self.assertEqual(state.get_shape(), (batch_size, 8))

    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape()[0], inp.get_shape()[0])
      self.assertEqual(out.get_shape()[1], num_units)
      self.assertEqual(out.dtype, inp.dtype)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      input_value = np.ones((batch_size, input_size))
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      for v in values:
        self.assertTrue(np.all(np.isfinite(v)))

  def testGrid2LSTMCellReLUWithRNN(self):
    batch_size = 3
    input_size = 5
    max_length = 6  # unrolled up to this length
    num_units = 2

    with variable_scope.variable_scope(
        'root', initializer=init_ops.constant_initializer(0.5)):
      cell = grid_rnn_cell.Grid2LSTMCell(
          num_units=num_units, non_recurrent_fn=nn_ops.relu)

      inputs = max_length * [
          array_ops.placeholder(
              dtypes.float32, shape=(batch_size, input_size))
      ]

      outputs, state = core_rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

    self.assertEqual(len(outputs), len(inputs))
    self.assertEqual(state.get_shape(), (batch_size, 4))

    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape()[0], inp.get_shape()[0])
      self.assertEqual(out.get_shape()[1], num_units)
      self.assertEqual(out.dtype, inp.dtype)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      input_value = np.ones((batch_size, input_size))
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      for v in values:
        self.assertTrue(np.all(np.isfinite(v)))

  def testGrid3LSTMCellReLUWithRNN(self):
    batch_size = 3
    input_size = 5
    max_length = 6  # unrolled up to this length
    num_units = 2

    with variable_scope.variable_scope(
        'root', initializer=init_ops.constant_initializer(0.5)):
      cell = grid_rnn_cell.Grid3LSTMCell(
          num_units=num_units, non_recurrent_fn=nn_ops.relu)

      inputs = max_length * [
          array_ops.placeholder(
              dtypes.float32, shape=(batch_size, input_size))
      ]

      outputs, state = core_rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

    self.assertEqual(len(outputs), len(inputs))
    self.assertEqual(state.get_shape(), (batch_size, 8))

    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape()[0], inp.get_shape()[0])
      self.assertEqual(out.get_shape()[1], num_units)
      self.assertEqual(out.dtype, inp.dtype)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      input_value = np.ones((batch_size, input_size))
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      for v in values:
        self.assertTrue(np.all(np.isfinite(v)))

  def testGrid1LSTMCellWithRNN(self):
    batch_size = 3
    input_size = 5
    max_length = 6  # unrolled up to this length
    num_units = 2

    with variable_scope.variable_scope(
        'root', initializer=init_ops.constant_initializer(0.5)):
      cell = grid_rnn_cell.Grid1LSTMCell(num_units=num_units)

      # for 1-LSTM, we only feed the first step
      inputs = ([
          array_ops.placeholder(
              dtypes.float32, shape=(batch_size, input_size))
      ] + (max_length - 1) * [array_ops.zeros([batch_size, input_size])])

      outputs, state = core_rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

    self.assertEqual(len(outputs), len(inputs))
    self.assertEqual(state.get_shape(), (batch_size, 4))

    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape(), (3, num_units))
      self.assertEqual(out.dtype, inp.dtype)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      input_value = np.ones((batch_size, input_size))
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      for v in values:
        self.assertTrue(np.all(np.isfinite(v)))


if __name__ == '__main__':
  test.main()
