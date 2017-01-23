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

import functools
import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

# TODO(ebrevdo): Remove once _linear is fully deprecated.
# pylint: disable=protected-access

from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear as linear
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test

# pylint: enable=protected-access


class RNNCellTest(test.TestCase):

  def testLinear(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(1.0)):
        x = array_ops.zeros([1, 2])
        l = linear([x], 2, False)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([l], {x.name: np.array([[1., 2.]])})
        self.assertAllClose(res[0], [[3.0, 3.0]])

        # Checks prevent you from accidentally creating a shared function.
        with self.assertRaises(ValueError):
          l1 = linear([x], 2, False)

        # But you can create a new one in a new scope and share the variables.
        with variable_scope.variable_scope("l1") as new_scope:
          l1 = linear([x], 2, False)
        with variable_scope.variable_scope(new_scope, reuse=True):
          linear([l1], 2, False)
        self.assertEqual(len(variables_lib.trainable_variables()), 2)

  def testBasicRNNCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 2])
        g, _ = core_rnn_cell_impl.BasicRNNCell(2)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [g], {x.name: np.array([[1., 1.]]),
                  m.name: np.array([[0.1, 0.1]])})
        self.assertEqual(res[0].shape, (1, 2))

  def testGRUCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 2])
        g, _ = core_rnn_cell_impl.GRUCell(2)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [g], {x.name: np.array([[1., 1.]]),
                  m.name: np.array([[0.1, 0.1]])})
        # Smoke test
        self.assertAllClose(res[0], [[0.175991, 0.175991]])
      with variable_scope.variable_scope(
          "other", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros(
            [1, 3])  # Test GRUCell with input_size != num_units.
        m = array_ops.zeros([1, 2])
        g, _ = core_rnn_cell_impl.GRUCell(2)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [g],
            {x.name: np.array([[1., 1., 1.]]),
             m.name: np.array([[0.1, 0.1]])})
        # Smoke test
        self.assertAllClose(res[0], [[0.156736, 0.156736]])

  def testBasicLSTMCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 8])
        g, out_m = core_rnn_cell_impl.MultiRNNCell(
            [core_rnn_cell_impl.BasicLSTMCell(
                2, state_is_tuple=False)] * 2,
            state_is_tuple=False)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [g, out_m],
            {x.name: np.array([[1., 1.]]),
             m.name: 0.1 * np.ones([1, 8])})
        self.assertEqual(len(res), 2)
        variables = variables_lib.global_variables()
        self.assertEqual(4, len(variables))
        self.assertEquals(variables[0].op.name,
                          "root/multi_rnn_cell/cell_0/basic_lstm_cell/weights")
        self.assertEquals(variables[1].op.name,
                          "root/multi_rnn_cell/cell_0/basic_lstm_cell/biases")
        self.assertEquals(variables[2].op.name,
                          "root/multi_rnn_cell/cell_1/basic_lstm_cell/weights")
        self.assertEquals(variables[3].op.name,
                          "root/multi_rnn_cell/cell_1/basic_lstm_cell/biases")
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.24024698, 0.24024698]])
        expected_mem = np.array([[
            0.68967271, 0.68967271, 0.44848421, 0.44848421, 0.39897051,
            0.39897051, 0.24024698, 0.24024698
        ]])
        self.assertAllClose(res[1], expected_mem)
      with variable_scope.variable_scope(
          "other", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros(
            [1, 3])  # Test BasicLSTMCell with input_size != num_units.
        m = array_ops.zeros([1, 4])
        g, out_m = core_rnn_cell_impl.BasicLSTMCell(
            2, state_is_tuple=False)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [g, out_m],
            {x.name: np.array([[1., 1., 1.]]),
             m.name: 0.1 * np.ones([1, 4])})
        self.assertEqual(len(res), 2)

  def testBasicLSTMCellStateTupleType(self):
    with self.test_session():
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m0 = (array_ops.zeros([1, 2]),) * 2
        m1 = (array_ops.zeros([1, 2]),) * 2
        cell = core_rnn_cell_impl.MultiRNNCell(
            [core_rnn_cell_impl.BasicLSTMCell(2)] * 2, state_is_tuple=True)
        self.assertTrue(isinstance(cell.state_size, tuple))
        self.assertTrue(
            isinstance(cell.state_size[0], core_rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(
            isinstance(cell.state_size[1], core_rnn_cell_impl.LSTMStateTuple))

        # Pass in regular tuples
        _, (out_m0, out_m1) = cell(x, (m0, m1))
        self.assertTrue(isinstance(out_m0, core_rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(isinstance(out_m1, core_rnn_cell_impl.LSTMStateTuple))

        # Pass in LSTMStateTuples
        variable_scope.get_variable_scope().reuse_variables()
        zero_state = cell.zero_state(1, dtypes.float32)
        self.assertTrue(isinstance(zero_state, tuple))
        self.assertTrue(
            isinstance(zero_state[0], core_rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(
            isinstance(zero_state[1], core_rnn_cell_impl.LSTMStateTuple))
        _, (out_m0, out_m1) = cell(x, zero_state)
        self.assertTrue(isinstance(out_m0, core_rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(isinstance(out_m1, core_rnn_cell_impl.LSTMStateTuple))

  def testBasicLSTMCellWithStateTuple(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m0 = array_ops.zeros([1, 4])
        m1 = array_ops.zeros([1, 4])
        cell = core_rnn_cell_impl.MultiRNNCell(
            [core_rnn_cell_impl.BasicLSTMCell(
                2, state_is_tuple=False)] * 2,
            state_is_tuple=True)
        g, (out_m0, out_m1) = cell(x, (m0, m1))
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g, out_m0, out_m1], {
            x.name: np.array([[1., 1.]]),
            m0.name: 0.1 * np.ones([1, 4]),
            m1.name: 0.1 * np.ones([1, 4])
        })
        self.assertEqual(len(res), 3)
        # The numbers in results were not calculated, this is just a smoke test.
        # Note, however, these values should match the original
        # version having state_is_tuple=False.
        self.assertAllClose(res[0], [[0.24024698, 0.24024698]])
        expected_mem0 = np.array(
            [[0.68967271, 0.68967271, 0.44848421, 0.44848421]])
        expected_mem1 = np.array(
            [[0.39897051, 0.39897051, 0.24024698, 0.24024698]])
        self.assertAllClose(res[1], expected_mem0)
        self.assertAllClose(res[2], expected_mem1)

  def testLSTMCell(self):
    with self.test_session() as sess:
      num_units = 8
      num_proj = 6
      state_size = num_units + num_proj
      batch_size = 3
      input_size = 2
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size, state_size])
        cell = core_rnn_cell_impl.LSTMCell(
            num_units=num_units,
            num_proj=num_proj,
            forget_bias=1.0,
            state_is_tuple=False)
        output, state = cell(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([output, state], {
            x.name: np.array([[1., 1.], [2., 2.], [3., 3.]]),
            m.name: 0.1 * np.ones((batch_size, state_size))
        })
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is mostly just a
        # smoke test.
        self.assertEqual(res[0].shape, (batch_size, num_proj))
        self.assertEqual(res[1].shape, (batch_size, state_size))
        # Different inputs so different outputs and states
        for i in range(1, batch_size):
          self.assertTrue(
              float(np.linalg.norm((res[0][0, :] - res[0][i, :]))) > 1e-6)
          self.assertTrue(
              float(np.linalg.norm((res[1][0, :] - res[1][i, :]))) > 1e-6)

  def testLSTMCellVariables(self):
    with self.test_session():
      num_units = 8
      num_proj = 6
      state_size = num_units + num_proj
      batch_size = 3
      input_size = 2
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size, state_size])
        cell = core_rnn_cell_impl.LSTMCell(
            num_units=num_units,
            num_proj=num_proj,
            forget_bias=1.0,
            state_is_tuple=False)
        cell(x, m)  # Execute to create variables
      variables = variables_lib.global_variables()
      self.assertEquals(variables[0].op.name, "root/lstm_cell/weights")
      self.assertEquals(variables[1].op.name, "root/lstm_cell/biases")
      self.assertEquals(variables[2].op.name,
                        "root/lstm_cell/projection/weights")

  def testOutputProjectionWrapper(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 3])
        cell = core_rnn_cell_impl.OutputProjectionWrapper(
            core_rnn_cell_impl.GRUCell(3), 2)
        g, new_m = cell(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g, new_m], {
            x.name: np.array([[1., 1., 1.]]),
            m.name: np.array([[0.1, 0.1, 0.1]])
        })
        self.assertEqual(res[1].shape, (1, 3))
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.231907, 0.231907]])

  def testInputProjectionWrapper(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 3])
        cell = core_rnn_cell_impl.InputProjectionWrapper(
            core_rnn_cell_impl.GRUCell(3), num_proj=3)
        g, new_m = cell(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [g, new_m],
            {x.name: np.array([[1., 1.]]),
             m.name: np.array([[0.1, 0.1, 0.1]])})
        self.assertEqual(res[1].shape, (1, 3))
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.154605, 0.154605, 0.154605]])

  def testDropoutWrapper(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 3])
        keep = array_ops.zeros([]) + 1
        g, new_m = core_rnn_cell_impl.DropoutWrapper(
            core_rnn_cell_impl.GRUCell(3), keep, keep)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g, new_m], {
            x.name: np.array([[1., 1., 1.]]),
            m.name: np.array([[0.1, 0.1, 0.1]])
        })
        self.assertEqual(res[1].shape, (1, 3))
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.154605, 0.154605, 0.154605]])

  def testEmbeddingWrapper(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 1], dtype=dtypes.int32)
        m = array_ops.zeros([1, 2])
        embedding_cell = core_rnn_cell_impl.EmbeddingWrapper(
            core_rnn_cell_impl.GRUCell(2),
            embedding_classes=3,
            embedding_size=2)
        self.assertEqual(embedding_cell.output_size, 2)
        g, new_m = embedding_cell(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [g, new_m],
            {x.name: np.array([[1]]),
             m.name: np.array([[0.1, 0.1]])})
        self.assertEqual(res[1].shape, (1, 2))
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res[0], [[0.17139, 0.17139]])

  def testEmbeddingWrapperWithDynamicRnn(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope("root"):
        inputs = ops.convert_to_tensor([[[0], [0]]], dtype=dtypes.int64)
        input_lengths = ops.convert_to_tensor([2], dtype=dtypes.int64)
        embedding_cell = core_rnn_cell_impl.EmbeddingWrapper(
            core_rnn_cell_impl.BasicLSTMCell(
                1, state_is_tuple=True),
            embedding_classes=1,
            embedding_size=2)
        outputs, _ = rnn.dynamic_rnn(
            cell=embedding_cell,
            inputs=inputs,
            sequence_length=input_lengths,
            dtype=dtypes.float32)
        sess.run([variables_lib.global_variables_initializer()])
        # This will fail if output's dtype is inferred from input's.
        sess.run(outputs)

  def testMultiRNNCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 4])
        _, ml = core_rnn_cell_impl.MultiRNNCell(
            [core_rnn_cell_impl.GRUCell(2)] * 2, state_is_tuple=False)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(ml, {
            x.name: np.array([[1., 1.]]),
            m.name: np.array([[0.1, 0.1, 0.1, 0.1]])
        })
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res, [[0.175991, 0.175991, 0.13248, 0.13248]])

  def testMultiRNNCellWithStateTuple(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m_bad = array_ops.zeros([1, 4])
        m_good = (array_ops.zeros([1, 2]), array_ops.zeros([1, 2]))

        # Test incorrectness of state
        with self.assertRaisesRegexp(ValueError, "Expected state .* a tuple"):
          core_rnn_cell_impl.MultiRNNCell(
              [core_rnn_cell_impl.GRUCell(2)] * 2,
              state_is_tuple=True)(x, m_bad)

        _, ml = core_rnn_cell_impl.MultiRNNCell(
            [core_rnn_cell_impl.GRUCell(2)] * 2, state_is_tuple=True)(x, m_good)

        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(ml, {
            x.name: np.array([[1., 1.]]),
            m_good[0].name: np.array([[0.1, 0.1]]),
            m_good[1].name: np.array([[0.1, 0.1]])
        })

        # The numbers in results were not calculated, this is just a
        # smoke test.  However, these numbers should match those of
        # the test testMultiRNNCell.
        self.assertAllClose(res[0], [[0.175991, 0.175991]])
        self.assertAllClose(res[1], [[0.13248, 0.13248]])


class SlimRNNCellTest(test.TestCase):

  def testBasicRNNCell(self):
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 2])
        my_cell = functools.partial(basic_rnn_cell, num_units=2)
        # pylint: disable=protected-access
        g, _ = core_rnn_cell_impl._SlimRNNCell(my_cell)(x, m)
        # pylint: enable=protected-access
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [g], {x.name: np.array([[1., 1.]]),
                  m.name: np.array([[0.1, 0.1]])})
        self.assertEqual(res[0].shape, (1, 2))

  def testBasicRNNCellMatch(self):
    batch_size = 32
    input_size = 100
    num_units = 10
    with self.test_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        inputs = random_ops.random_uniform((batch_size, input_size))
        _, initial_state = basic_rnn_cell(inputs, None, num_units)
        my_cell = functools.partial(basic_rnn_cell, num_units=num_units)
        # pylint: disable=protected-access
        slim_cell = core_rnn_cell_impl._SlimRNNCell(my_cell)
        # pylint: enable=protected-access
        slim_outputs, slim_state = slim_cell(inputs, initial_state)
        rnn_cell = core_rnn_cell_impl.BasicRNNCell(num_units)
        variable_scope.get_variable_scope().reuse_variables()
        outputs, state = rnn_cell(inputs, initial_state)
        self.assertEqual(slim_outputs.get_shape(), outputs.get_shape())
        self.assertEqual(slim_state.get_shape(), state.get_shape())
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([slim_outputs, slim_state, outputs, state])
        self.assertAllClose(res[0], res[2])
        self.assertAllClose(res[1], res[3])


def basic_rnn_cell(inputs, state, num_units, scope=None):
  if state is None:
    if inputs is not None:
      batch_size = inputs.get_shape()[0]
      dtype = inputs.dtype
    else:
      batch_size = 0
      dtype = dtypes.float32
    init_output = array_ops.zeros(
        array_ops.stack([batch_size, num_units]), dtype=dtype)
    init_state = array_ops.zeros(
        array_ops.stack([batch_size, num_units]), dtype=dtype)
    init_output.set_shape([batch_size, num_units])
    init_state.set_shape([batch_size, num_units])
    return init_output, init_state
  else:
    with variable_scope.variable_scope(scope, "basic_rnn_cell",
                                       [inputs, state]):
      output = math_ops.tanh(linear([inputs, state], num_units, True))
    return output, output


if __name__ == "__main__":
  test.main()
