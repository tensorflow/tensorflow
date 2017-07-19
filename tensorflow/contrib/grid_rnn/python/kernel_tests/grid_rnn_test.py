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

import numpy as np

from tensorflow.contrib.grid_rnn.python.ops import grid_rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class GridRNNCellTest(test.TestCase):

  def testGrid2BasicLSTMCell(self):
    with self.test_session(use_gpu=False) as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.2)) as root_scope:
        x = array_ops.zeros([1, 3])
        m = ((array_ops.zeros([1, 2]), array_ops.zeros([1, 2])),
             (array_ops.zeros([1, 2]), array_ops.zeros([1, 2])))
        cell = grid_rnn_cell.Grid2BasicLSTMCell(2)
        self.assertEqual(cell.state_size, ((2, 2), (2, 2)))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (1, 2))
        self.assertEqual(s[0].c.get_shape(), (1, 2))
        self.assertEqual(s[0].h.get_shape(), (1, 2))
        self.assertEqual(s[1].c.get_shape(), (1, 2))
        self.assertEqual(s[1].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x:
                np.array([[1., 1., 1.]]),
            m: ((np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])),
                (np.array([[0.5, 0.6]]), np.array([[0.7, 0.8]])))
        })
        self.assertEqual(res_g[0].shape, (1, 2))
        self.assertEqual(res_s[0].c.shape, (1, 2))
        self.assertEqual(res_s[0].h.shape, (1, 2))
        self.assertEqual(res_s[1].c.shape, (1, 2))
        self.assertEqual(res_s[1].h.shape, (1, 2))

        self.assertAllClose(res_g, ([[0.27813572, 0.27813572]],))
        self.assertAllClose(
            res_s, (([[0.58389962, 0.58389962]], [[0.27813572, 0.27813572]]),
                    ([[0.52472770, 0.60331118]], [[0.27650252, 0.30985516]])))

        # emulate a loop through the input sequence,
        # where we call cell() multiple times
        root_scope.reuse_variables()
        g2, s2 = cell(x, m)
        self.assertEqual(g2[0].get_shape(), (1, 2))
        self.assertEqual(s2[0].c.get_shape(), (1, 2))
        self.assertEqual(s2[0].h.get_shape(), (1, 2))
        self.assertEqual(s2[1].c.get_shape(), (1, 2))
        self.assertEqual(s2[1].h.get_shape(), (1, 2))

        res_g2, res_s2 = sess.run([g2, s2],
                                  {x: np.array([[2., 2., 2.]]),
                                   m: res_s})
        self.assertEqual(res_g2[0].shape, (1, 2))
        self.assertEqual(res_s2[0].c.shape, (1, 2))
        self.assertEqual(res_s2[0].h.shape, (1, 2))
        self.assertEqual(res_s2[1].c.shape, (1, 2))
        self.assertEqual(res_s2[1].h.shape, (1, 2))
        self.assertAllClose(res_g2[0], [[0.44053382, 0.44053382]])
        self.assertAllClose(
            res_s2, (([[1.1822226, 1.18222260]], [[0.44053382, 0.44053382]]),
                     ([[0.6710391, 0.73025036]], [[0.30998221, 0.32985979]])))

  def testGrid2BasicLSTMCellTied(self):
    with self.test_session(use_gpu=False) as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.2)):
        x = array_ops.zeros([1, 3])
        m = ((array_ops.zeros([1, 2]), array_ops.zeros([1, 2])),
             (array_ops.zeros([1, 2]), array_ops.zeros([1, 2])))
        cell = grid_rnn_cell.Grid2BasicLSTMCell(2, tied=True)
        self.assertEqual(cell.state_size, ((2, 2), (2, 2)))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (1, 2))
        self.assertEqual(s[0].c.get_shape(), (1, 2))
        self.assertEqual(s[0].h.get_shape(), (1, 2))
        self.assertEqual(s[1].c.get_shape(), (1, 2))
        self.assertEqual(s[1].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x:
                np.array([[1., 1., 1.]]),
            m: ((np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])),
                (np.array([[0.5, 0.6]]), np.array([[0.7, 0.8]])))
        })
        self.assertEqual(res_g[0].shape, (1, 2))
        self.assertEqual(res_s[0].c.shape, (1, 2))
        self.assertEqual(res_s[0].h.shape, (1, 2))
        self.assertEqual(res_s[1].c.shape, (1, 2))
        self.assertEqual(res_s[1].h.shape, (1, 2))

        self.assertAllClose(res_g[0], [[0.27813572, 0.27813572]])
        self.assertAllClose(
            res_s, (([[0.58389962, 0.58389962]], [[0.27813572, 0.27813572]]),
                    ([[0.52472770, 0.60331118]], [[0.27650252, 0.30985516]])))

        res_g, res_s = sess.run([g, s], {x: np.array([[1., 1., 1.]]), m: res_s})
        self.assertEqual(res_g[0].shape, (1, 2))

        self.assertAllClose(res_g[0], [[0.27634764, 0.27634764]])
        self.assertAllClose(
            res_s, (([[0.58274859, 0.58274859]], [[0.27634764, 0.27634764]]),
                    ([[0.52718318, 0.58639443]], [[0.25576338, 0.27909029]])))

  def testGrid2BasicLSTMCellWithRelu(self):
    with self.test_session(use_gpu=False) as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.2)):
        x = array_ops.zeros([1, 3])
        m = ((array_ops.zeros([1, 2]), array_ops.zeros([1, 2])),)
        cell = grid_rnn_cell.Grid2BasicLSTMCell(
            2, tied=False, non_recurrent_fn=nn_ops.relu)
        self.assertEqual(cell.state_size, ((2, 2),))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (1, 2))
        self.assertEqual(s[0].c.get_shape(), (1, 2))
        self.assertEqual(s[0].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x: np.array([[1., 1., 1.]]),
            m: ((np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])),)
        })
        self.assertEqual(res_g[0].shape, (1, 2))
        self.assertAllClose(res_g[0], [[0.29142371, 0.29142371]])
        self.assertAllClose(res_s, (([[0.20757815, 0.28334612]],
                                     [[0.10947458, 0.14764377]]),))

  """LSTMCell
  """

  def testGrid2LSTMCell(self):
    with self.test_session(use_gpu=False) as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = ((array_ops.zeros([1, 2]), array_ops.zeros([1, 2])),
             (array_ops.zeros([1, 2]), array_ops.zeros([1, 2])))
        cell = grid_rnn_cell.Grid2LSTMCell(2, use_peepholes=True)
        self.assertEqual(cell.state_size, ((2, 2), (2, 2)))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (1, 2))
        self.assertEqual(s[0].c.get_shape(), (1, 2))
        self.assertEqual(s[0].h.get_shape(), (1, 2))
        self.assertEqual(s[1].c.get_shape(), (1, 2))
        self.assertEqual(s[1].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x:
                np.array([[1., 1., 1.]]),
            m: ((np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])),
                (np.array([[0.5, 0.6]]), np.array([[0.7, 0.8]])))
        })
        self.assertEqual(res_g[0].shape, (1, 2))
        self.assertEqual(res_s[0].c.shape, (1, 2))
        self.assertEqual(res_s[0].h.shape, (1, 2))
        self.assertEqual(res_s[1].c.shape, (1, 2))
        self.assertEqual(res_s[1].h.shape, (1, 2))

        self.assertAllClose(res_g[0], [[0.83462638, 0.83462638]])
        self.assertAllClose(
            res_s, (([[2.19742370, 2.19742370]], [[0.83462638, 0.83462638]]),
                    ([[1.21154201, 1.30832052]], [[0.66558737, 0.69353604]])))

  def testGrid2LSTMCellTied(self):
    with self.test_session(use_gpu=False) as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = ((array_ops.zeros([1, 2]), array_ops.zeros([1, 2])),
             (array_ops.zeros([1, 2]), array_ops.zeros([1, 2])))
        cell = grid_rnn_cell.Grid2LSTMCell(2, tied=True, use_peepholes=True)
        self.assertEqual(cell.state_size, ((2, 2), (2, 2)))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (1, 2))
        self.assertEqual(s[0].c.get_shape(), (1, 2))
        self.assertEqual(s[0].h.get_shape(), (1, 2))
        self.assertEqual(s[1].c.get_shape(), (1, 2))
        self.assertEqual(s[1].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x:
                np.array([[1., 1., 1.]]),
            m: ((np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])),
                (np.array([[0.5, 0.6]]), np.array([[0.7, 0.8]])))
        })
        self.assertEqual(res_g[0].shape, (1, 2))
        self.assertEqual(res_s[0].c.shape, (1, 2))
        self.assertEqual(res_s[0].h.shape, (1, 2))
        self.assertEqual(res_s[1].c.shape, (1, 2))
        self.assertEqual(res_s[1].h.shape, (1, 2))

        self.assertAllClose(res_g[0], [[0.83462638, 0.83462638]])
        self.assertAllClose(
            res_s, (([[2.19742370, 2.19742370]], [[0.83462638, 0.83462638]]),
                    ([[1.21154201, 1.30832052]], [[0.66558737, 0.69353604]])))

  def testGrid2LSTMCellWithRelu(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = ((array_ops.zeros([1, 2]), array_ops.zeros([1, 2])),)
        cell = grid_rnn_cell.Grid2LSTMCell(
            2, use_peepholes=True, non_recurrent_fn=nn_ops.relu)
        self.assertEqual(cell.state_size, ((2, 2),))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (1, 2))
        self.assertEqual(s[0].c.get_shape(), (1, 2))
        self.assertEqual(s[0].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x: np.array([[1., 1., 1.]]),
            m: ((np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])),)
        })
        self.assertEqual(res_g[0].shape, (1, 2))
        self.assertAllClose(res_g[0], [[1.98171163, 1.98171163]])
        self.assertAllClose(res_s, (([[0.82688755, 0.91509962]],
                                     [[0.46301097, 0.50041234]]),))

  """RNNCell
  """

  def testGrid2BasicRNNCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([2, 2])
        m = (array_ops.zeros([2, 2]), array_ops.zeros([2, 2]))
        cell = grid_rnn_cell.Grid2BasicRNNCell(2)
        self.assertEqual(cell.state_size, (2, 2))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (2, 2))
        self.assertEqual(s[0].get_shape(), (2, 2))
        self.assertEqual(s[1].get_shape(), (2, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x:
                np.array([[1., 1.], [2., 2.]]),
            m: (np.array([[0.1, 0.1], [0.2, 0.2]]), np.array([[0.1, 0.1],
                                                              [0.2, 0.2]]))
        })
        self.assertEqual(res_g[0].shape, (2, 2))
        self.assertEqual(res_s[0].shape, (2, 2))
        self.assertEqual(res_s[1].shape, (2, 2))

        self.assertAllClose(res_g, ([[0.76159418, 0.40584856],
                                     [0.96402758, 0.52317500]],))
        self.assertAllClose(
            res_s, ([[0.76159418, 0.40584856], [0.96402758, 0.523175]],
                    [[0.76159418, 0.09966799], [0.96402758, 0.19737528]]))

  def testGrid2BasicRNNCellTied(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([2, 2])
        m = (array_ops.zeros([2, 2]), array_ops.zeros([2, 2]))
        cell = grid_rnn_cell.Grid2BasicRNNCell(2, tied=True)
        self.assertEqual(cell.state_size, (2, 2))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (2, 2))
        self.assertEqual(s[0].get_shape(), (2, 2))
        self.assertEqual(s[1].get_shape(), (2, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x:
                np.array([[1., 1.], [2., 2.]]),
            m: (np.array([[0.1, 0.1], [0.2, 0.2]]), np.array([[0.1, 0.1],
                                                              [0.2, 0.2]]))
        })
        self.assertEqual(res_g[0].shape, (2, 2))
        self.assertEqual(res_s[0].shape, (2, 2))
        self.assertEqual(res_s[1].shape, (2, 2))

        self.assertAllClose(res_g, ([[0.76159418, 0.40584856],
                                     [0.96402758, 0.52317500]],))
        self.assertAllClose(
            res_s, ([[0.76159418, 0.40584856], [0.96402758, 0.52317500]],
                    [[0.76159418, 0.09966799], [0.96402758, 0.19737528]]))

  def testGrid2BasicRNNCellWithRelu(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = (array_ops.zeros([1, 2]),)
        cell = grid_rnn_cell.Grid2BasicRNNCell(2, non_recurrent_fn=nn_ops.relu)
        self.assertEqual(cell.state_size, (2,))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (1, 2))
        self.assertEqual(s[0].get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run(
            [g, s], {x: np.array([[1., 1.]]),
                     m: np.array([[0.1, 0.1]])})
        self.assertEqual(res_g[0].shape, (1, 2))
        self.assertEqual(res_s[0].shape, (1, 2))
        self.assertAllClose(res_g, ([[1.43063116, 1.43063116]],))
        self.assertAllClose(res_s, ([[0.76159418, 0.09966799]],))

  """1-LSTM
  """

  def testGrid1LSTMCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)) as root_scope:
        x = array_ops.zeros([1, 3])
        m = ((array_ops.zeros([1, 2]), array_ops.zeros([1, 2])),)
        cell = grid_rnn_cell.Grid1LSTMCell(2, use_peepholes=True)
        self.assertEqual(cell.state_size, ((2, 2),))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (1, 2))
        self.assertEqual(s[0].c.get_shape(), (1, 2))
        self.assertEqual(s[0].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x: np.array([[1., 1., 1.]]),
            m: ((np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])),)
        })
        self.assertEqual(res_g[0].shape, (1, 2))
        self.assertEqual(res_s[0].c.shape, (1, 2))
        self.assertEqual(res_s[0].h.shape, (1, 2))

        self.assertAllClose(res_g, ([[0.91287315, 0.91287315]],))
        self.assertAllClose(res_s, (([[2.26285243, 2.26285243]],
                                     [[0.91287315, 0.91287315]]),))

        root_scope.reuse_variables()

        x2 = array_ops.zeros([0, 0])
        g2, s2 = cell(x2, m)
        self.assertEqual(g2[0].get_shape(), (1, 2))
        self.assertEqual(s2[0].c.get_shape(), (1, 2))
        self.assertEqual(s2[0].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g2, res_s2 = sess.run([g2, s2], {m: res_s})
        self.assertEqual(res_g2[0].shape, (1, 2))
        self.assertEqual(res_s2[0].c.shape, (1, 2))
        self.assertEqual(res_s2[0].h.shape, (1, 2))

        self.assertAllClose(res_g2, ([[0.9032144, 0.9032144]],))
        self.assertAllClose(res_s2, (([[2.79966092, 2.79966092]],
                                      [[0.9032144, 0.9032144]]),))

        g3, s3 = cell(x2, m)
        self.assertEqual(g3[0].get_shape(), (1, 2))
        self.assertEqual(s3[0].c.get_shape(), (1, 2))
        self.assertEqual(s3[0].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g3, res_s3 = sess.run([g3, s3], {m: res_s2})
        self.assertEqual(res_g3[0].shape, (1, 2))
        self.assertEqual(res_s3[0].c.shape, (1, 2))
        self.assertEqual(res_s3[0].h.shape, (1, 2))
        self.assertAllClose(res_g3, ([[0.92727238, 0.92727238]],))
        self.assertAllClose(res_s3, (([[3.3529923, 3.3529923]],
                                      [[0.92727238, 0.92727238]]),))

  """3-LSTM
  """

  def testGrid3LSTMCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = ((array_ops.zeros([1, 2]), array_ops.zeros([1, 2])),
             (array_ops.zeros([1, 2]), array_ops.zeros([1, 2])),
             (array_ops.zeros([1, 2]), array_ops.zeros([1, 2])))
        cell = grid_rnn_cell.Grid3LSTMCell(2, use_peepholes=True)
        self.assertEqual(cell.state_size, ((2, 2), (2, 2), (2, 2)))

        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (1, 2))
        self.assertEqual(s[0].c.get_shape(), (1, 2))
        self.assertEqual(s[0].h.get_shape(), (1, 2))
        self.assertEqual(s[1].c.get_shape(), (1, 2))
        self.assertEqual(s[1].h.get_shape(), (1, 2))
        self.assertEqual(s[2].c.get_shape(), (1, 2))
        self.assertEqual(s[2].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x:
                np.array([[1., 1., 1.]]),
            m: ((np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]])),
                (np.array([[0.5, 0.6]]), np.array([[0.7, 0.8]])), (np.array(
                    [[-0.1, -0.2]]), np.array([[-0.3, -0.4]])))
        })
        self.assertEqual(res_g[0].shape, (1, 2))
        self.assertEqual(res_s[0].c.shape, (1, 2))
        self.assertEqual(res_s[0].h.shape, (1, 2))
        self.assertEqual(res_s[1].c.shape, (1, 2))
        self.assertEqual(res_s[1].h.shape, (1, 2))
        self.assertEqual(res_s[2].c.shape, (1, 2))
        self.assertEqual(res_s[2].h.shape, (1, 2))

        self.assertAllClose(res_g, ([[0.96892911, 0.96892911]],))
        self.assertAllClose(
            res_s, (([[2.45227885, 2.45227885]], [[0.96892911, 0.96892911]]),
                    ([[1.33592629, 1.4373529]], [[0.80867189, 0.83247656]]),
                    ([[0.7317788, 0.63205892]], [[0.56548983, 0.50446129]])))

  """Edge cases
  """

  def testGridRNNEdgeCasesLikeRelu(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([3, 2])
        m = ()

        # this is equivalent to relu
        cell = grid_rnn_cell.GridRNNCell(
            num_units=2,
            num_dims=1,
            input_dims=0,
            output_dims=0,
            non_recurrent_dims=0,
            non_recurrent_fn=nn_ops.relu)
        g, s = cell(x, m)
        self.assertEqual(g[0].get_shape(), (3, 2))
        self.assertEqual(s, ())

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s],
                                {x: np.array([[1., -1.], [-2, 1], [2, -1]])})
        self.assertEqual(res_g[0].shape, (3, 2))
        self.assertEqual(res_s, ())
        self.assertAllClose(res_g, ([[0, 0], [0, 0], [0.5, 0.5]],))

  def testGridRNNEdgeCasesNoOutput(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = ((array_ops.zeros([1, 2]), array_ops.zeros([1, 2])),)

        # This cell produces no output
        cell = grid_rnn_cell.GridRNNCell(
            num_units=2,
            num_dims=2,
            input_dims=0,
            output_dims=None,
            non_recurrent_dims=0,
            non_recurrent_fn=nn_ops.relu)
        g, s = cell(x, m)
        self.assertEqual(g, ())
        self.assertEqual(s[0].c.get_shape(), (1, 2))
        self.assertEqual(s[0].h.get_shape(), (1, 2))

        sess.run([variables.global_variables_initializer()])
        res_g, res_s = sess.run([g, s], {
            x: np.array([[1., 1.]]),
            m: ((np.array([[0.1, 0.1]]), np.array([[0.1, 0.1]])),)
        })
        self.assertEqual(res_g, ())
        self.assertEqual(res_s[0].c.shape, (1, 2))
        self.assertEqual(res_s[0].h.shape, (1, 2))

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

      outputs, state = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

    self.assertEqual(len(outputs), len(inputs))
    self.assertEqual(state[0].c.get_shape(), (batch_size, 2))
    self.assertEqual(state[0].h.get_shape(), (batch_size, 2))
    self.assertEqual(state[1].c.get_shape(), (batch_size, 2))
    self.assertEqual(state[1].h.get_shape(), (batch_size, 2))

    for out, inp in zip(outputs, inputs):
      self.assertEqual(len(out), 1)
      self.assertEqual(out[0].get_shape()[0], inp.get_shape()[0])
      self.assertEqual(out[0].get_shape()[1], num_units)
      self.assertEqual(out[0].dtype, inp.dtype)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      input_value = np.ones((batch_size, input_size))
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      for tp in values[:-1]:
        for v in tp:
          self.assertTrue(np.all(np.isfinite(v)))
      for tp in values[-1]:
        for st in tp:
          for v in st:
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
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]

      outputs, state = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

    self.assertEqual(len(outputs), len(inputs))
    self.assertEqual(state[0].c.get_shape(), (batch_size, 2))
    self.assertEqual(state[0].h.get_shape(), (batch_size, 2))

    for out, inp in zip(outputs, inputs):
      self.assertEqual(len(out), 1)
      self.assertEqual(out[0].get_shape()[0], inp.get_shape()[0])
      self.assertEqual(out[0].get_shape()[1], num_units)
      self.assertEqual(out[0].dtype, inp.dtype)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      input_value = np.ones((batch_size, input_size))
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      for tp in values[:-1]:
        for v in tp:
          self.assertTrue(np.all(np.isfinite(v)))
      for tp in values[-1]:
        for st in tp:
          for v in st:
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
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]

      outputs, state = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

    self.assertEqual(len(outputs), len(inputs))
    self.assertEqual(state[0].c.get_shape(), (batch_size, 2))
    self.assertEqual(state[0].h.get_shape(), (batch_size, 2))
    self.assertEqual(state[1].c.get_shape(), (batch_size, 2))
    self.assertEqual(state[1].h.get_shape(), (batch_size, 2))

    for out, inp in zip(outputs, inputs):
      self.assertEqual(len(out), 1)
      self.assertEqual(out[0].get_shape()[0], inp.get_shape()[0])
      self.assertEqual(out[0].get_shape()[1], num_units)
      self.assertEqual(out[0].dtype, inp.dtype)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      input_value = np.ones((batch_size, input_size))
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      for tp in values[:-1]:
        for v in tp:
          self.assertTrue(np.all(np.isfinite(v)))
      for tp in values[-1]:
        for st in tp:
          for v in st:
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

      outputs, state = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

    self.assertEqual(len(outputs), len(inputs))
    self.assertEqual(state[0].c.get_shape(), (batch_size, 2))
    self.assertEqual(state[0].h.get_shape(), (batch_size, 2))

    for out, inp in zip(outputs, inputs):
      self.assertEqual(len(out), 1)
      self.assertEqual(out[0].get_shape(), (3, num_units))
      self.assertEqual(out[0].dtype, inp.dtype)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      input_value = np.ones((batch_size, input_size))
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      for tp in values[:-1]:
        for v in tp:
          self.assertTrue(np.all(np.isfinite(v)))
      for tp in values[-1]:
        for st in tp:
          for v in st:
            self.assertTrue(np.all(np.isfinite(v)))

  def testGrid2LSTMCellWithRNNAndDynamicBatchSize(self):
    """Test for #4296."""
    input_size = 5
    max_length = 6  # unrolled up to this length
    num_units = 2

    with variable_scope.variable_scope(
        'root', initializer=init_ops.constant_initializer(0.5)):
      cell = grid_rnn_cell.Grid2LSTMCell(num_units=num_units)

      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]

      outputs, state = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

    self.assertEqual(len(outputs), len(inputs))

    for out, inp in zip(outputs, inputs):
      self.assertEqual(len(out), 1)
      self.assertTrue(out[0].get_shape()[0].value is None)
      self.assertEqual(out[0].get_shape()[1], num_units)
      self.assertEqual(out[0].dtype, inp.dtype)

    with self.test_session() as sess:
      sess.run(variables.global_variables_initializer())

      input_value = np.ones((3, input_size))
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      for tp in values[:-1]:
        for v in tp:
          self.assertTrue(np.all(np.isfinite(v)))
      for tp in values[-1]:
        for st in tp:
          for v in st:
            self.assertTrue(np.all(np.isfinite(v)))

  def testGrid2LSTMCellLegacy(self):
    """Test for legacy case (when state_is_tuple=False)."""
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          'root', initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 8])
        cell = grid_rnn_cell.Grid2LSTMCell(
            2, use_peepholes=True, state_is_tuple=False, output_is_tuple=False)
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
        self.assertAllClose(res[0], [[0.83462638, 0.83462638]])
        self.assertAllClose(res[1], [[
            2.19742370, 2.19742370, 0.83462638, 0.83462638, 1.21154201,
            1.30832052, 0.66558737, 0.69353604
        ]])

if __name__ == '__main__':
  test.main()
