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

import itertools
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_rnn_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest


class Plus1RNNCell(rnn_cell.RNNCell):
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    return 5

  @property
  def state_size(self):
    return 5

  def __call__(self, input_, state, scope=None):
    return (input_ + 1, state + 1)


class DummyMultiDimensionalLSTM(rnn_cell.RNNCell):
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
      raise TypeError("The dimensions passed to DummyMultiDimensionalLSTM "
                      "should be a tuple of ints.")
    self._dims = dims
    self._output_size = tensor_shape.TensorShape(self._dims)
    self._state_size = (tensor_shape.TensorShape(self._dims),
                        tensor_shape.TensorShape(self._dims))

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def __call__(self, input_, state, scope=None):
    h, c = state
    return (input_ + 1, (h + 1, c + 1))


class NestedRNNCell(rnn_cell.RNNCell):
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

    return array_ops.zeros((self._batch_size,) + state_size)

  def save_state(self, name, state):
    self.saved_state[name] = state
    return array_ops.identity(state)

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def state_size(self):
    return self._state_size


class TestStateSaverWithCounters(TestStateSaver):
  """Class wrapper around TestStateSaver.

  A dummy class used for testing of static_state_saving_rnn. It helps test if
  save_state and state functions got called same number of time when we
  evaluate output of rnn cell and state or either of them separately. It
  inherits from the TestStateSaver and adds the counters for calls of functions.
  """

  @test_util.run_v1_only("b/124229375")
  def __init__(self, batch_size, state_size):
    super(TestStateSaverWithCounters, self).__init__(batch_size, state_size)
    self._num_state_calls = variables_lib.VariableV1(0)
    self._num_save_state_calls = variables_lib.VariableV1(0)

  def state(self, name):
    with ops.control_dependencies(
        [state_ops.assign_add(self._num_state_calls, 1)]):
      return super(TestStateSaverWithCounters, self).state(name)

  def save_state(self, name, state):
    with ops.control_dependencies([state_ops.assign_add(
        self._num_save_state_calls, 1)]):
      return super(TestStateSaverWithCounters, self).save_state(name, state)

  @property
  def num_state_calls(self):
    return self._num_state_calls

  @property
  def num_save_state_calls(self):
    return self._num_save_state_calls


class RNNTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def testInvalidSequenceLengthShape(self):
    cell = Plus1RNNCell()
    inputs = [array_ops.placeholder(dtypes.float32, shape=(3, 4))]
    with self.assertRaisesRegex(ValueError, "must be a vector"):
      rnn.static_rnn(cell, inputs, dtype=dtypes.float32, sequence_length=4)

  @test_util.run_v1_only("b/124229375")
  def testRNN(self):
    cell = Plus1RNNCell()
    batch_size = 2
    input_size = 5
    max_length = 8  # unrolled up to this length
    inputs = max_length * [
        array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
    ]
    outputs, state = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
    self.assertEqual(len(outputs), len(inputs))
    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape(), inp.get_shape())
      self.assertEqual(out.dtype, inp.dtype)

    with self.session() as sess:
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})

      # Outputs
      for v in values[:-1]:
        self.assertAllClose(v, input_value + 1.0)

      # Final state
      self.assertAllClose(values[-1],
                          max_length * np.ones(
                              (batch_size, input_size), dtype=np.float32))

  @test_util.run_v1_only("b/124229375")
  def testDropout(self):
    cell = Plus1RNNCell()
    full_dropout_cell = rnn_cell.DropoutWrapper(
        cell, input_keep_prob=1e-6, seed=0)
    self.assertIn("cell", full_dropout_cell._trackable_children())
    self.assertIs(full_dropout_cell._trackable_children()["cell"], cell)
    batch_size = 2
    input_size = 5
    max_length = 8
    inputs = max_length * [
        array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
    ]
    with variable_scope.variable_scope("share_scope"):
      outputs, state = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
    with variable_scope.variable_scope("drop_scope"):
      dropped_outputs, _ = rnn.static_rnn(
          full_dropout_cell, inputs, dtype=dtypes.float32)
    self.assertEqual(len(outputs), len(inputs))
    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape().as_list(), inp.get_shape().as_list())
      self.assertEqual(out.dtype, inp.dtype)

    with self.session() as sess:
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      full_dropout_values = sess.run(
          dropped_outputs, feed_dict={
              inputs[0]: input_value
          })

      for v in values[:-1]:
        self.assertAllClose(v, input_value + 1.0)
      for d_v in full_dropout_values[:-1]:  # Add 1.0 to dropped_out (all zeros)
        self.assertAllClose(d_v, np.ones_like(input_value))

  @test_util.run_v1_only("b/124229375")
  def testDynamicCalculation(self):
    cell = Plus1RNNCell()
    sequence_length = array_ops.placeholder(dtypes.int64)
    batch_size = 2
    input_size = 5
    max_length = 8
    inputs = max_length * [
        array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
    ]
    with variable_scope.variable_scope("drop_scope"):
      dynamic_outputs, dynamic_state = rnn.static_rnn(
          cell, inputs, sequence_length=sequence_length, dtype=dtypes.float32)
    self.assertEqual(len(dynamic_outputs), len(inputs))

    with self.session() as sess:
      input_value = np.random.randn(batch_size, input_size)
      dynamic_values = sess.run(
          dynamic_outputs,
          feed_dict={
              inputs[0]: input_value,
              sequence_length: [2, 3]
          })
      dynamic_state_value = sess.run(
          [dynamic_state],
          feed_dict={
              inputs[0]: input_value,
              sequence_length: [2, 3]
          })

      # outputs are fully calculated for t = 0, 1
      for v in dynamic_values[:2]:
        self.assertAllClose(v, input_value + 1.0)

      # outputs at t = 2 are zero for entry 0, calculated for entry 1
      self.assertAllClose(dynamic_values[2],
                          np.vstack((np.zeros((input_size)),
                                     1.0 + input_value[1, :])))

      # outputs at t = 3+ are zero
      for v in dynamic_values[3:]:
        self.assertAllEqual(v, np.zeros_like(input_value))

      # the final states are:
      #  entry 0: the values from the calculation at t=1
      #  entry 1: the values from the calculation at t=2
      self.assertAllEqual(dynamic_state_value[0],
                          np.vstack((1.0 * (1 + 1) * np.ones((input_size)),
                                     1.0 * (2 + 1) * np.ones((input_size)))))

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)

      # check that all the variables names starts
      # with the proper scope.
      variables_lib.global_variables_initializer()
      all_vars = variables_lib.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("RNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  @test_util.run_v1_only("b/124229375")
  def testScope(self):

    def factory(scope):
      cell = Plus1RNNCell()
      batch_size = 2
      input_size = 5
      max_length = 8  # unrolled up to this length
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]
      return rnn.static_rnn(cell, inputs, dtype=dtypes.float32, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


class LSTMTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def testDType(self):
    # Test case for GitHub issue 16228
    # Not passing dtype in constructor results in default float32
    lstm = rnn_cell.LSTMCell(10)
    input_tensor = array_ops.ones([10, 50])
    lstm.build(input_tensor.get_shape())
    self.assertEqual(lstm._bias.dtype.base_dtype, dtypes.float32)

    # Explicitly pass dtype in constructor
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      lstm = rnn_cell.LSTMCell(10, dtype=dtype)
      input_tensor = array_ops.ones([10, 50])
      lstm.build(input_tensor.get_shape())
      self.assertEqual(lstm._bias.dtype.base_dtype, dtype)

  @test_util.run_v1_only("b/124229375")
  def testNoProjNoSharding(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      cell = rnn_cell.LSTMCell(
          num_units, initializer=initializer, state_is_tuple=False)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]
      outputs, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  @test_util.run_v1_only("b/124229375")
  def testCellClipping(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          cell_clip=0.0,
          initializer=initializer,
          state_is_tuple=False)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]
      outputs, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value})

    for value in values:
      # if cell c is clipped to 0, tanh(c) = 0 => m==0
      self.assertAllEqual(value, np.zeros((batch_size, num_units)))

  @test_util.run_v1_only("b/124229375")
  def testNoProjNoShardingSimpleStateSaver(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(batch_size, 2 * num_units)
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=False,
          initializer=initializer,
          state_is_tuple=False)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]
      with variable_scope.variable_scope("share_scope"):
        outputs, state = rnn.static_state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name="save_lstm")
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      (last_state_value, saved_state_value) = sess.run(
          [state, state_saver.saved_state["save_lstm"]],
          feed_dict={
              inputs[0]: input_value
          })
      self.assertAllEqual(last_state_value, saved_state_value)

  @test_util.run_v1_only("b/124229375")
  def testNoProjNoShardingTupleStateSaver(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(batch_size, num_units)
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=False,
          initializer=initializer,
          state_is_tuple=True)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]
      with variable_scope.variable_scope("share_scope"):
        outputs, state = rnn.static_state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name=("c", "m"))
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      last_and_saved_states = sess.run(
          state + (state_saver.saved_state["c"], state_saver.saved_state["m"]),
          feed_dict={
              inputs[0]: input_value
          })
      self.assertEqual(4, len(last_and_saved_states))
      self.assertAllEqual(last_and_saved_states[:2], last_and_saved_states[2:])

  @test_util.run_v1_only("b/124229375")
  def testNoProjNoShardingNestedTupleStateSaver(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(
          batch_size, {
              "c0": num_units,
              "m0": num_units,
              "c1": num_units + 1,
              "m1": num_units + 1,
              "c2": num_units + 2,
              "m2": num_units + 2,
              "c3": num_units + 3,
              "m3": num_units + 3
          })

      def _cell(i):
        return rnn_cell.LSTMCell(
            num_units + i,
            use_peepholes=False,
            initializer=initializer,
            state_is_tuple=True)

      # This creates a state tuple which has 4 sub-tuples of length 2 each.
      cell = rnn_cell.MultiRNNCell(
          [_cell(i) for i in range(4)], state_is_tuple=True)

      self.assertEqual(len(cell.state_size), 4)
      for i in range(4):
        self.assertEqual(len(cell.state_size[i]), 2)

      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]

      state_names = (("c0", "m0"), ("c1", "m1"), ("c2", "m2"), ("c3", "m3"))
      with variable_scope.variable_scope("share_scope"):
        outputs, state = rnn.static_state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name=state_names)
      self.assertEqual(len(outputs), len(inputs))

      # Final output comes from _cell(3) which has state size num_units + 3
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units + 3])

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      last_states = sess.run(
          list(nest.flatten(state)), feed_dict={
              inputs[0]: input_value
          })
      saved_states = sess.run(
          list(state_saver.saved_state.values()),
          feed_dict={
              inputs[0]: input_value
          })
      self.assertEqual(8, len(last_states))
      self.assertEqual(8, len(saved_states))
      flat_state_names = nest.flatten(state_names)
      named_saved_states = dict(
          zip(state_saver.saved_state.keys(), saved_states))

      for i in range(8):
        self.assertAllEqual(last_states[i],
                            named_saved_states[flat_state_names[i]])

  @test_util.run_v1_only("b/124229375")
  def testProjNoSharding(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=False)
      outputs, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      self.assertEqual(len(outputs), len(inputs))

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testStateTupleWithProjAndSequenceLength(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    sequence_length = [4, 6]
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]
      cell_notuple = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=False)
      cell_tuple = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=True)
      with variable_scope.variable_scope("root") as scope:
        outputs_notuple, state_notuple = rnn.static_rnn(
            cell_notuple,
            inputs,
            dtype=dtypes.float32,
            sequence_length=sequence_length,
            scope=scope)
        scope.reuse_variables()
        # TODO(ebrevdo): For this test, we ensure values are identical and
        # therefore the weights here are tied.  In the future, we may consider
        # making the state_is_tuple property mutable so we can avoid
        # having to do this - especially if users ever need to reuse
        # the parameters from different RNNCell instances.  Right now,
        # this seems an unrealistic use case except for testing.
        cell_tuple._scope = cell_notuple._scope  # pylint: disable=protected-access
        outputs_tuple, state_tuple = rnn.static_rnn(
            cell_tuple,
            inputs,
            dtype=dtypes.float32,
            sequence_length=sequence_length,
            scope=scope)
      self.assertEqual(len(outputs_notuple), len(inputs))
      self.assertEqual(len(outputs_tuple), len(inputs))
      self.assertTrue(isinstance(state_tuple, tuple))
      self.assertTrue(isinstance(state_notuple, ops.Tensor))

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      outputs_notuple_v = sess.run(
          outputs_notuple, feed_dict={
              inputs[0]: input_value
          })
      outputs_tuple_v = sess.run(
          outputs_tuple, feed_dict={
              inputs[0]: input_value
          })
      self.assertAllEqual(outputs_notuple_v, outputs_tuple_v)

      (state_notuple_v,) = sess.run(
          (state_notuple,), feed_dict={
              inputs[0]: input_value
          })
      state_tuple_v = sess.run(state_tuple, feed_dict={inputs[0]: input_value})
      self.assertAllEqual(state_notuple_v, np.hstack(state_tuple_v))

  @test_util.run_v1_only("b/124229375")
  def testProjSharding(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)

      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]

      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)

      outputs, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

      self.assertEqual(len(outputs), len(inputs))

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  @test_util.run_v1_only("b/124229375")
  def testDoubleInput(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(-1, 1, seed=self._seed)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float64, shape=(None, input_size))
      ]

      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)

      outputs, _ = rnn.static_rnn(
          cell,
          inputs,
          initial_state=cell.zero_state(batch_size, dtypes.float64))

      self.assertEqual(len(outputs), len(inputs))

      variables_lib.global_variables_initializer().run()
      input_value = np.asarray(
          np.random.randn(batch_size, input_size), dtype=np.float64)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value})
      self.assertEqual(values[0].dtype, input_value.dtype)

  @test_util.run_v1_only("b/124229375")
  def testShardNoShardEquivalentOutput(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]
      initializer = init_ops.constant_initializer(0.001)

      cell_noshard = rnn_cell.LSTMCell(
          num_units,
          num_proj=num_proj,
          use_peepholes=True,
          initializer=initializer,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          state_is_tuple=False)

      cell_shard = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          initializer=initializer,
          num_proj=num_proj,
          state_is_tuple=False)

      with variable_scope.variable_scope("noshard_scope"):
        outputs_noshard, state_noshard = rnn.static_rnn(
            cell_noshard, inputs, dtype=dtypes.float32)
      with variable_scope.variable_scope("shard_scope"):
        outputs_shard, state_shard = rnn.static_rnn(
            cell_shard, inputs, dtype=dtypes.float32)

      self.assertEqual(len(outputs_noshard), len(inputs))
      self.assertEqual(len(outputs_noshard), len(outputs_shard))

      variables_lib.global_variables_initializer().run()
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

  @test_util.run_v1_only("b/124229375")
  def testDoubleInputWithDropoutAndDynamicCalculation(self):
    """Smoke test for using LSTM with doubles, dropout, dynamic calculation."""

    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      sequence_length = array_ops.placeholder(dtypes.int64)
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float64, shape=(None, input_size))
      ]

      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)
      dropout_cell = rnn_cell.DropoutWrapper(cell, 0.5, seed=0)

      outputs, state = rnn.static_rnn(
          dropout_cell,
          inputs,
          sequence_length=sequence_length,
          initial_state=cell.zero_state(batch_size, dtypes.float64))

      self.assertEqual(len(outputs), len(inputs))

      variables_lib.global_variables_initializer().run(feed_dict={
          sequence_length: [2, 3]
      })
      input_value = np.asarray(
          np.random.randn(batch_size, input_size), dtype=np.float64)
      values = sess.run(
          outputs, feed_dict={
              inputs[0]: input_value,
              sequence_length: [2, 3]
          })
      state_value = sess.run(
          [state], feed_dict={
              inputs[0]: input_value,
              sequence_length: [2, 3]
          })
      self.assertEqual(values[0].dtype, input_value.dtype)
      self.assertEqual(state_value[0].dtype, input_value.dtype)

  @test_util.run_v1_only("b/124229375")
  def testSharingWeightsWithReuse(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(-1, 1, seed=self._seed)
      initializer_d = init_ops.random_uniform_initializer(
          -1, 1, seed=self._seed + 1)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=False)
      cell_d = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer_d,
          state_is_tuple=False)

      with variable_scope.variable_scope("share_scope"):
        outputs0, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      with variable_scope.variable_scope("share_scope", reuse=True):
        outputs1, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      with variable_scope.variable_scope("diff_scope"):
        outputs2, _ = rnn.static_rnn(cell_d, inputs, dtype=dtypes.float32)

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      output_values = sess.run(
          outputs0 + outputs1 + outputs2, feed_dict={
              inputs[0]: input_value
          })
      outputs0_values = output_values[:max_length]
      outputs1_values = output_values[max_length:2 * max_length]
      outputs2_values = output_values[2 * max_length:]
      self.assertEqual(len(outputs0_values), len(outputs1_values))
      self.assertEqual(len(outputs0_values), len(outputs2_values))
      for o1, o2, o3 in zip(outputs0_values, outputs1_values, outputs2_values):
        # Same weights used by both RNNs so outputs should be the same.
        self.assertAllEqual(o1, o2)
        # Different weights used so outputs should be different.
        self.assertTrue(np.linalg.norm(o1 - o3) > 1e-6)

  @test_util.run_v1_only("b/124229375")
  def testSharingWeightsWithDifferentNamescope(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(-1, 1, seed=self._seed)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=False)

      with ops.name_scope("scope0"):
        with variable_scope.variable_scope("share_scope"):
          outputs0, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      with ops.name_scope("scope1"):
        with variable_scope.variable_scope("share_scope", reuse=True):
          outputs1, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      output_values = sess.run(
          outputs0 + outputs1, feed_dict={
              inputs[0]: input_value
          })
      outputs0_values = output_values[:max_length]
      outputs1_values = output_values[max_length:]
      self.assertEqual(len(outputs0_values), len(outputs1_values))
      for out0, out1 in zip(outputs0_values, outputs1_values):
        self.assertAllEqual(out0, out1)

  @test_util.run_v1_only("b/124229375")
  def testDynamicRNNAllowsUnknownTimeDimension(self):
    inputs = array_ops.placeholder(dtypes.float32, shape=[1, None, 20])
    cell = rnn_cell.GRUCell(30)
    # Smoke test, this should not raise an error
    rnn.dynamic_rnn(cell, inputs, dtype=dtypes.float32)

  @test_util.run_in_graph_and_eager_modes
  def testDynamicRNNWithTupleStates(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    sequence_length = [4, 6]
    in_graph_mode = not context.executing_eagerly()
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      if in_graph_mode:
        inputs = max_length * [
            array_ops.placeholder(dtypes.float32, shape=(None, input_size))
        ]
      else:
        inputs = max_length * [
            constant_op.constant(
                np.random.randn(batch_size, input_size).astype(np.float32))
        ]
      inputs_c = array_ops_stack.stack(inputs)
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=True)
      with variable_scope.variable_scope("root") as scope:
        outputs_static, state_static = rnn.static_rnn(
            cell,
            inputs,
            dtype=dtypes.float32,
            sequence_length=sequence_length,
            scope=scope)
        scope.reuse_variables()
        outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
            cell,
            inputs_c,
            dtype=dtypes.float32,
            time_major=True,
            sequence_length=sequence_length,
            scope=scope)
      self.assertTrue(isinstance(state_static, rnn_cell.LSTMStateTuple))
      self.assertTrue(isinstance(state_dynamic, rnn_cell.LSTMStateTuple))
      self.assertIs(state_static[0], state_static.c)
      self.assertIs(state_static[1], state_static.h)
      self.assertIs(state_dynamic[0], state_dynamic.c)
      self.assertIs(state_dynamic[1], state_dynamic.h)

      if in_graph_mode:
        variables_lib.global_variables_initializer().run()
        input_value = np.random.randn(batch_size, input_size)
        outputs_static = sess.run(
            outputs_static, feed_dict={
                inputs[0]: input_value
            })
        outputs_dynamic = sess.run(
            outputs_dynamic, feed_dict={
                inputs[0]: input_value
            })
        state_static = sess.run(
            state_static, feed_dict={
                inputs[0]: input_value
            })
        state_dynamic = sess.run(
            state_dynamic, feed_dict={
                inputs[0]: input_value
            })

      comparison_fn = self.assertAllEqual
      if test_util.is_xla_enabled():
        comparison_fn = self.assertAllClose
      if in_graph_mode:
        comparison_fn(outputs_static, outputs_dynamic)
      else:
        self.assertAllEqual(
            array_ops_stack.stack(outputs_static), outputs_dynamic)
      comparison_fn(np.hstack(state_static), np.hstack(state_dynamic))

  @test_util.run_in_graph_and_eager_modes
  def testDynamicRNNWithNestedTupleStates(self):
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    sequence_length = [4, 6]
    in_graph_mode = not context.executing_eagerly()
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      if in_graph_mode:
        inputs = max_length * [
            array_ops.placeholder(dtypes.float32, shape=(None, input_size))
        ]
      else:
        inputs = max_length * [
            constant_op.constant(
                np.random.randn(batch_size, input_size).astype(np.float32))
        ]
      inputs_c = array_ops_stack.stack(inputs)

      def _cell(i):
        return rnn_cell.LSTMCell(
            num_units + i,
            use_peepholes=True,
            num_proj=num_proj + i,
            initializer=initializer,
            state_is_tuple=True)

      # This creates a state tuple which has 4 sub-tuples of length 2 each.
      cell = rnn_cell.MultiRNNCell(
          [_cell(i) for i in range(4)], state_is_tuple=True)

      self.assertEqual(len(cell.state_size), 4)
      for i in range(4):
        self.assertEqual(len(cell.state_size[i]), 2)

      test_zero = cell.zero_state(1, dtypes.float32)
      self.assertEqual(len(test_zero), 4)
      for i in range(4):
        self.assertEqual(test_zero[i][0].get_shape()[1], cell.state_size[i][0])
        self.assertEqual(test_zero[i][1].get_shape()[1], cell.state_size[i][1])

      with variable_scope.variable_scope("root") as scope:
        outputs_static, state_static = rnn.static_rnn(
            cell,
            inputs,
            dtype=dtypes.float32,
            sequence_length=sequence_length,
            scope=scope)
        scope.reuse_variables()
        outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
            cell,
            inputs_c,
            dtype=dtypes.float32,
            time_major=True,
            sequence_length=sequence_length,
            scope=scope)

      if in_graph_mode:
        input_value = np.random.randn(batch_size, input_size)
        variables_lib.global_variables_initializer().run()
        outputs_static = sess.run(
            outputs_static, feed_dict={
                inputs[0]: input_value
            })
        outputs_dynamic = sess.run(
            outputs_dynamic, feed_dict={
                inputs[0]: input_value
            })
        state_static = sess.run(
            nest.flatten(state_static), feed_dict={
                inputs[0]: input_value
            })
        state_dynamic = sess.run(
            nest.flatten(state_dynamic), feed_dict={
                inputs[0]: input_value
            })

      comparison_fn = self.assertAllEqual
      if test_util.is_xla_enabled():
        comparison_fn = self.assertAllClose
      if in_graph_mode:
        comparison_fn(outputs_static, outputs_dynamic)
      else:
        self.assertAllEqual(
            array_ops_stack.stack(outputs_static), outputs_dynamic)
        state_static = nest.flatten(state_static)
        state_dynamic = nest.flatten(state_dynamic)
      comparison_fn(np.hstack(state_static), np.hstack(state_dynamic))

  def _testDynamicEquivalentToStaticRNN(self, use_sequence_length):
    time_steps = 8
    num_units = 3
    num_proj = 4
    input_size = 5
    batch_size = 2

    input_values = np.random.randn(time_steps, batch_size, input_size).astype(
        np.float32)

    if use_sequence_length:
      sequence_length = np.random.randint(0, time_steps, size=batch_size)
    else:
      sequence_length = None

    in_graph_mode = not context.executing_eagerly()

    # TODO(b/68017812): Eager ignores operation seeds, so we need to create a
    # single cell and reuse it across the static and dynamic RNNs. Remove this
    # special case once is fixed.
    if not in_graph_mode:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          initializer=initializer,
          num_proj=num_proj,
          state_is_tuple=False)

    ########### Step 1: Run static graph and generate readouts
    with self.session(graph=ops.Graph()) as sess:
      if in_graph_mode:
        concat_inputs = array_ops.placeholder(
            dtypes.float32, shape=(time_steps, batch_size, input_size))
      else:
        concat_inputs = constant_op.constant(input_values)
      inputs = array_ops_stack.unstack(concat_inputs)
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)

      # TODO(akshayka): Remove special case once b/68017812 is fixed.
      if in_graph_mode:
        cell = rnn_cell.LSTMCell(
            num_units,
            use_peepholes=True,
            initializer=initializer,
            num_proj=num_proj,
            state_is_tuple=False)

      with variable_scope.variable_scope("dynamic_scope"):
        outputs_static, state_static = rnn.static_rnn(
            cell, inputs, sequence_length=sequence_length, dtype=dtypes.float32)

      if in_graph_mode:
        # Generate gradients of sum of outputs w.r.t. inputs
        static_gradients = gradients_impl.gradients(
            outputs_static + [state_static], [concat_inputs])
        # Generate gradients of individual outputs w.r.t. inputs
        static_individual_gradients = nest.flatten([
            gradients_impl.gradients(y, [concat_inputs])
            for y in [outputs_static[0], outputs_static[-1], state_static]
        ])
        # Generate gradients of individual variables w.r.t. inputs
        trainable_variables = ops.get_collection(
            ops.GraphKeys.TRAINABLE_VARIABLES)
        assert len(trainable_variables) > 1, (
            "Count of trainable variables: %d" % len(trainable_variables))
        # pylint: disable=bad-builtin
        static_individual_variable_gradients = nest.flatten([
            gradients_impl.gradients(y, trainable_variables)
            for y in [outputs_static[0], outputs_static[-1], state_static]
        ])
        # Generate gradients and run sessions to obtain outputs
        feeds = {concat_inputs: input_values}
        # Initialize
        variables_lib.global_variables_initializer().run(feed_dict=feeds)
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
    with self.session(graph=ops.Graph()) as sess:
      if in_graph_mode:
        concat_inputs = array_ops.placeholder(
            dtypes.float32, shape=(time_steps, batch_size, input_size))
      else:
        concat_inputs = constant_op.constant(input_values)
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)

      # TODO(akshayka): Remove this special case once b/68017812 is
      # fixed.
      if in_graph_mode:
        cell = rnn_cell.LSTMCell(
            num_units,
            use_peepholes=True,
            initializer=initializer,
            num_proj=num_proj,
            state_is_tuple=False)

      with variable_scope.variable_scope("dynamic_scope"):
        outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
            cell,
            inputs=concat_inputs,
            sequence_length=sequence_length,
            time_major=True,
            dtype=dtypes.float32)
        split_outputs_dynamic = array_ops_stack.unstack(
            outputs_dynamic, time_steps)

      if in_graph_mode:

        # Generate gradients of sum of outputs w.r.t. inputs
        dynamic_gradients = gradients_impl.gradients(
            split_outputs_dynamic + [state_dynamic], [concat_inputs])

        # Generate gradients of several individual outputs w.r.t. inputs
        dynamic_individual_gradients = nest.flatten([
            gradients_impl.gradients(y, [concat_inputs])
            for y in [
                split_outputs_dynamic[0], split_outputs_dynamic[-1],
                state_dynamic
            ]
        ])

        # Generate gradients of individual variables w.r.t. inputs
        trainable_variables = ops.get_collection(
            ops.GraphKeys.TRAINABLE_VARIABLES)
        assert len(trainable_variables) > 1, (
            "Count of trainable variables: %d" % len(trainable_variables))
        dynamic_individual_variable_gradients = nest.flatten([
            gradients_impl.gradients(y, trainable_variables)
            for y in [
                split_outputs_dynamic[0], split_outputs_dynamic[-1],
                state_dynamic
            ]
        ])

        feeds = {concat_inputs: input_values}

        # Initialize
        variables_lib.global_variables_initializer().run(feed_dict=feeds)

        # Test forward pass
        values_dynamic = sess.run(split_outputs_dynamic, feed_dict=feeds)
        (state_value_dynamic,) = sess.run((state_dynamic,), feed_dict=feeds)

        # Test gradients to inputs and variables w.r.t. outputs & final state
        dynamic_grad_values = sess.run(dynamic_gradients, feed_dict=feeds)

        dynamic_individual_grad_values = sess.run(
            dynamic_individual_gradients, feed_dict=feeds)

        dynamic_individual_var_grad_values = sess.run(
            dynamic_individual_variable_gradients, feed_dict=feeds)

    ######### Step 3: Comparisons
    if not in_graph_mode:
      values_static = outputs_static
      values_dynamic = split_outputs_dynamic
      state_value_static = state_static
      state_value_dynamic = state_dynamic

    self.assertEqual(len(values_static), len(values_dynamic))
    for (value_static, value_dynamic) in zip(values_static, values_dynamic):
      self.assertAllClose(value_static, value_dynamic)
    self.assertAllClose(state_value_static, state_value_dynamic)

    if in_graph_mode:

      self.assertAllClose(static_grad_values, dynamic_grad_values)

      self.assertEqual(
          len(static_individual_grad_values),
          len(dynamic_individual_grad_values))
      self.assertEqual(
          len(static_individual_var_grad_values),
          len(dynamic_individual_var_grad_values))

      for i, (a, b) in enumerate(
          zip(static_individual_grad_values, dynamic_individual_grad_values)):
        tf_logging.info("Comparing individual gradients iteration %d" % i)
        self.assertAllClose(a, b)

      for i, (a, b) in enumerate(
          zip(static_individual_var_grad_values,
              dynamic_individual_var_grad_values)):
        tf_logging.info(
            "Comparing individual variable gradients iteration %d" % i)
        self.assertAllClose(a, b)

  @test_util.run_in_graph_and_eager_modes
  def testDynamicEquivalentToStaticRNN(self):
    self._testDynamicEquivalentToStaticRNN(use_sequence_length=False)

  @test_util.run_in_graph_and_eager_modes
  def testDynamicEquivalentToStaticRNNWithSequenceLength(self):
    self._testDynamicEquivalentToStaticRNN(use_sequence_length=True)

  @test_util.run_in_graph_and_eager_modes
  def testLSTMBlockCellErrorHandling(self):
    forget_bias = 1
    cell_clip = 0
    use_peephole = False
    x = constant_op.constant(0.837607, shape=[28, 29], dtype=dtypes.float32)
    cs_prev = constant_op.constant(0, shape=[28, 17], dtype=dtypes.float32)
    h_prev = constant_op.constant(
        0.592631638, shape=[28, 17], dtype=dtypes.float32)
    w = constant_op.constant(0.887386262, shape=[46, 68], dtype=dtypes.float32)
    wci = constant_op.constant(0, shape=[], dtype=dtypes.float32)
    wcf = constant_op.constant(0, shape=[17], dtype=dtypes.float32)
    wco = constant_op.constant(
        0.592631638, shape=[28, 17], dtype=dtypes.float32)
    b = constant_op.constant(0.75259006, shape=[68], dtype=dtypes.float32)
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(
          gen_rnn_ops.lstm_block_cell(
              x=x,
              cs_prev=cs_prev,
              h_prev=h_prev,
              w=w,
              wci=wci,
              wcf=wcf,
              wco=wco,
              b=b,
              forget_bias=forget_bias,
              cell_clip=cell_clip,
              use_peephole=use_peephole))
      
  @test_util.run_in_graph_and_eager_modes
  def testLSTMBlockCellEmptyInputRaisesError(self):
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError, "is empty"):
      self.evaluate(
          gen_rnn_ops.lstm_block_cell(
              x=constant_op.constant(0, shape=[2, 16], dtype=dtypes.half),
              cs_prev=constant_op.constant(0, shape=[2, 0], dtype=dtypes.half),
              h_prev=constant_op.constant(0, shape=[2, 0], dtype=dtypes.half),
              w=constant_op.constant(0, shape=[16, 0], dtype=dtypes.half),
              wci=constant_op.constant(0, shape=[5], dtype=dtypes.half),
              wcf=constant_op.constant(0, shape=[16], dtype=dtypes.half),
              wco=constant_op.constant(0, shape=[13], dtype=dtypes.half),
              b=constant_op.constant(0, shape=[0], dtype=dtypes.half),
              forget_bias=112.66590343649887,
              cell_clip=67.12389445926587,
              use_peephole=False))

  @test_util.run_in_graph_and_eager_modes
  def testLSTMBlockCellGradErrorHandling(self):
    use_peephole = False
    seq_len_max = constant_op.constant(1, shape=[], dtype=dtypes.int64)
    x = constant_op.constant(0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    cs_prev = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    h_prev = constant_op.constant(
        0.504355371, shape=[1, 1], dtype=dtypes.float32)
    w = constant_op.constant(0.504355371, shape=[1, 1], dtype=dtypes.float32)
    wci = constant_op.constant(0.504355371, shape=[1], dtype=dtypes.float32)
    wcf = constant_op.constant(0.504355371, shape=[1], dtype=dtypes.float32)
    wco = constant_op.constant(0.504355371, shape=[1], dtype=dtypes.float32)
    b = constant_op.constant(0.504355371, shape=[1], dtype=dtypes.float32)
    i = constant_op.constant(0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    cs = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    f = constant_op.constant(0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    o = constant_op.constant(0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    ci = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    co = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    h = constant_op.constant(0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    cs_grad = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    h_grad = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "must be rank"):
      self.evaluate(
          gen_rnn_ops.block_lstm_grad_v2(
              seq_len_max=seq_len_max,
              x=x,
              cs_prev=cs_prev,
              h_prev=h_prev,
              w=w,
              wci=wci,
              wcf=wcf,
              wco=wco,
              b=b,
              i=i,
              cs=cs,
              f=f,
              o=o,
              ci=ci,
              co=co,
              h=h,
              cs_grad=cs_grad,
              h_grad=h_grad,
              use_peephole=use_peephole))

  def testLSTMBlockInvalidArgument(self):
    # Test case for GitHub issue 58175
    forget_bias = -121.22699269620765
    cell_clip = -106.82307555235684
    use_peephole = False
    seq_len_max = math_ops.saturate_cast(
        random_ops.random_uniform(
            [13, 11, 0], minval=0, maxval=64, dtype=dtypes.int64
        ),
        dtype=dtypes.int64,
    )
    x = random_ops.random_uniform([1, 3, 15], dtype=dtypes.float32)
    cs_prev = random_ops.random_uniform([3, 0], dtype=dtypes.float32)
    h_prev = random_ops.random_uniform([3, 0], dtype=dtypes.float32)
    w = random_ops.random_uniform([15, 0], dtype=dtypes.float32)
    wci = random_ops.random_uniform([0], dtype=dtypes.float32)
    wcf = random_ops.random_uniform([0], dtype=dtypes.float32)
    wco = random_ops.random_uniform([0], dtype=dtypes.float32)
    b = random_ops.random_uniform([0], dtype=dtypes.float32)
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(
          gen_rnn_ops.BlockLSTM(
              forget_bias=forget_bias,
              cell_clip=cell_clip,
              use_peephole=use_peephole,
              seq_len_max=seq_len_max,
              x=x,
              cs_prev=cs_prev,
              h_prev=h_prev,
              w=w,
              wci=wci,
              wcf=wcf,
              wco=wco,
              b=b,
          )
      )


class BidirectionalRNNTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _createBidirectionalRNN(self, use_shape, use_sequence_length, scope=None):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8

    initializer = init_ops.random_uniform_initializer(
        -0.01, 0.01, seed=self._seed)
    sequence_length = array_ops.placeholder(
        dtypes.int64) if use_sequence_length else None
    cell_fw = rnn_cell.LSTMCell(
        num_units, input_size, initializer=initializer, state_is_tuple=False)
    cell_bw = rnn_cell.LSTMCell(
        num_units, input_size, initializer=initializer, state_is_tuple=False)
    inputs = max_length * [
        array_ops.placeholder(
            dtypes.float32,
            shape=(batch_size, input_size) if use_shape else (None, input_size))
    ]
    outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(
        cell_fw,
        cell_bw,
        inputs,
        dtype=dtypes.float32,
        sequence_length=sequence_length,
        scope=scope)
    self.assertEqual(len(outputs), len(inputs))
    for out in outputs:
      self.assertEqual(out.get_shape().as_list(),
                       [batch_size if use_shape else None, 2 * num_units])

    input_value = np.random.randn(batch_size, input_size)
    outputs = array_ops_stack.stack(outputs)

    return input_value, inputs, outputs, state_fw, state_bw, sequence_length

  def _testBidirectionalRNN(self, use_shape):
    with self.session(graph=ops.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createBidirectionalRNN(use_shape, True))
      variables_lib.global_variables_initializer().run()
      # Run with pre-specified sequence length of 2, 3
      out, s_fw, s_bw = sess.run(
          [outputs, state_fw, state_bw],
          feed_dict={
              inputs[0]: input_value,
              sequence_length: [2, 3]
          })

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward output has to be the same,
      # but reversed in time. The format is output[time][batch][depth], and
      # due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      #
      # First sequence in batch is length=2
      # Check that the time=0 forward output is equal to time=1 backward output
      self.assertAllClose(out[0][0][0], out[1][0][3])
      self.assertAllClose(out[0][0][1], out[1][0][4])
      self.assertAllClose(out[0][0][2], out[1][0][5])
      # Check that the time=1 forward output is equal to time=0 backward output
      self.assertAllClose(out[1][0][0], out[0][0][3])
      self.assertAllClose(out[1][0][1], out[0][0][4])
      self.assertAllClose(out[1][0][2], out[0][0][5])

      # Second sequence in batch is length=3
      # Check that the time=0 forward output is equal to time=2 backward output
      self.assertAllClose(out[0][1][0], out[2][1][3])
      self.assertAllClose(out[0][1][1], out[2][1][4])
      self.assertAllClose(out[0][1][2], out[2][1][5])
      # Check that the time=1 forward output is equal to time=1 backward output
      self.assertAllClose(out[1][1][0], out[1][1][3])
      self.assertAllClose(out[1][1][1], out[1][1][4])
      self.assertAllClose(out[1][1][2], out[1][1][5])
      # Check that the time=2 forward output is equal to time=0 backward output
      self.assertAllClose(out[2][1][0], out[0][1][3])
      self.assertAllClose(out[2][1][1], out[0][1][4])
      self.assertAllClose(out[2][1][2], out[0][1][5])
      # Via the reasoning above, the forward and backward final state should be
      # exactly the same
      self.assertAllClose(s_fw, s_bw)

  def _testBidirectionalRNNWithoutSequenceLength(self, use_shape):
    with self.session(graph=ops.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, _ = (
          self._createBidirectionalRNN(use_shape, False))
      variables_lib.global_variables_initializer().run()
      out, s_fw, s_bw = sess.run(
          [outputs, state_fw, state_bw], feed_dict={
              inputs[0]: input_value
          })

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward output has to be the same,
      # but reversed in time. The format is output[time][batch][depth], and
      # due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      #
      # Both sequences in batch are length=8.  Check that the time=i
      # forward output is equal to time=8-1-i backward output
      for i in range(8):
        self.assertAllClose(out[i][0][0:3], out[8 - 1 - i][0][3:6])
        self.assertAllClose(out[i][1][0:3], out[8 - 1 - i][1][3:6])
      # Via the reasoning above, the forward and backward final state should be
      # exactly the same
      self.assertAllClose(s_fw, s_bw)

  @test_util.run_v1_only("b/124229375")
  def testBidirectionalRNN(self):
    self._testBidirectionalRNN(use_shape=False)
    self._testBidirectionalRNN(use_shape=True)

  @test_util.run_v1_only("b/124229375")
  def testBidirectionalRNNWithoutSequenceLength(self):
    self._testBidirectionalRNNWithoutSequenceLength(use_shape=False)
    self._testBidirectionalRNNWithoutSequenceLength(use_shape=True)

  def _createBidirectionalDynamicRNN(self,
                                     use_shape,
                                     use_state_tuple,
                                     use_time_major,
                                     use_sequence_length,
                                     scope=None):
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8

    initializer = init_ops.random_uniform_initializer(
        -0.01, 0.01, seed=self._seed)
    sequence_length = (
        array_ops.placeholder(dtypes.int64) if use_sequence_length else None)
    cell_fw = rnn_cell.LSTMCell(
        num_units, initializer=initializer, state_is_tuple=use_state_tuple)
    cell_bw = rnn_cell.LSTMCell(
        num_units, initializer=initializer, state_is_tuple=use_state_tuple)
    inputs = max_length * [
        array_ops.placeholder(
            dtypes.float32,
            shape=(batch_size if use_shape else None, input_size))
    ]
    inputs_c = array_ops_stack.stack(inputs)
    if not use_time_major:
      inputs_c = array_ops.transpose(inputs_c, [1, 0, 2])
    outputs, states = rnn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        inputs_c,
        sequence_length,
        dtype=dtypes.float32,
        time_major=use_time_major,
        scope=scope)
    outputs = array_ops.concat(outputs, 2)
    state_fw, state_bw = states
    outputs_shape = [None, max_length, 2 * num_units]
    if use_shape:
      outputs_shape[0] = batch_size
    if use_time_major:
      outputs_shape[0], outputs_shape[1] = outputs_shape[1], outputs_shape[0]
    self.assertEqual(outputs.get_shape().as_list(), outputs_shape)

    input_value = np.random.randn(batch_size, input_size)

    return input_value, inputs, outputs, state_fw, state_bw, sequence_length

  def _testBidirectionalDynamicRNN(self, use_shape, use_state_tuple,
                                   use_time_major, use_sequence_length):
    with self.session(graph=ops.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createBidirectionalDynamicRNN(
              use_shape, use_state_tuple, use_time_major, use_sequence_length))
      variables_lib.global_variables_initializer().run()
      # Run with pre-specified sequence length of 2, 3
      feed_dict = ({sequence_length: [2, 3]} if use_sequence_length else {})
      feed_dict.update({inputs[0]: input_value})
      if use_state_tuple:
        out, c_fw, m_fw, c_bw, m_bw = sess.run(
            [outputs, state_fw[0], state_fw[1], state_bw[0], state_bw[1]],
            feed_dict=feed_dict)
        s_fw = (c_fw, m_fw)
        s_bw = (c_bw, m_bw)
      else:
        feed_dict.update({inputs[0]: input_value})
        out, s_fw, s_bw = sess.run(
            [outputs, state_fw, state_bw], feed_dict=feed_dict)

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward output has to be the same,
      # but reversed in time. The format is output[time][batch][depth], and
      # due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      #
      if not use_time_major:
        out = np.swapaxes(out, 0, 1)

      if use_sequence_length:
        # First sequence in batch is length=2
        # Check that the t=0 forward output is equal to t=1 backward output
        self.assertEqual(out[0][0][0], out[1][0][3])
        self.assertEqual(out[0][0][1], out[1][0][4])
        self.assertEqual(out[0][0][2], out[1][0][5])
        # Check that the t=1 forward output is equal to t=0 backward output
        self.assertEqual(out[1][0][0], out[0][0][3])
        self.assertEqual(out[1][0][1], out[0][0][4])
        self.assertEqual(out[1][0][2], out[0][0][5])

        # Second sequence in batch is length=3
        # Check that the t=0 forward output is equal to t=2 backward output
        self.assertEqual(out[0][1][0], out[2][1][3])
        self.assertEqual(out[0][1][1], out[2][1][4])
        self.assertEqual(out[0][1][2], out[2][1][5])
        # Check that the t=1 forward output is equal to t=1 backward output
        self.assertEqual(out[1][1][0], out[1][1][3])
        self.assertEqual(out[1][1][1], out[1][1][4])
        self.assertEqual(out[1][1][2], out[1][1][5])
        # Check that the t=2 forward output is equal to t=0 backward output
        self.assertEqual(out[2][1][0], out[0][1][3])
        self.assertEqual(out[2][1][1], out[0][1][4])
        self.assertEqual(out[2][1][2], out[0][1][5])
        # Via the reasoning above, the forward and backward final state should
        # be exactly the same
        self.assertAllClose(s_fw, s_bw)
      else:  # not use_sequence_length
        max_length = 8  # from createBidirectionalDynamicRNN
        for t in range(max_length):
          self.assertAllEqual(out[t, :, 0:3], out[max_length - t - 1, :, 3:6])
        self.assertAllClose(s_fw, s_bw)

  @test_util.run_v1_only("b/124229375")
  def testBidirectionalDynamicRNN(self):
    # Generate 2^5 option values
    # from [True, True, True, True, True] to [False, False, False, False, False]
    options = itertools.product([True, False], repeat=4)
    for option in options:
      self._testBidirectionalDynamicRNN(
          use_shape=option[0],
          use_state_tuple=option[1],
          use_time_major=option[2],
          use_sequence_length=option[3])

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    # REMARKS: factory(scope) is a function accepting a scope
    #          as an argument, such scope can be None, a string
    #          or a VariableScope instance.
    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)

      # check that all the variables names starts
      # with the proper scope.
      variables_lib.global_variables_initializer()
      all_vars = variables_lib.global_variables()
      prefix = prefix or "bidirectional_rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("BiRNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  @test_util.run_v1_only("b/124229375")
  def testBidirectionalRNNScope(self):

    def factory(scope):
      return self._createBidirectionalRNN(
          use_shape=True, use_sequence_length=True, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)

  @test_util.run_v1_only("b/124229375")
  def testBidirectionalDynamicRNNScope(self):

    def get_factory(use_time_major):

      def factory(scope):
        return self._createBidirectionalDynamicRNN(
            use_shape=True,
            use_state_tuple=True,
            use_sequence_length=True,
            use_time_major=use_time_major,
            scope=scope)

      return factory

    self._testScope(get_factory(True), use_outer_scope=True)
    self._testScope(get_factory(True), use_outer_scope=False)
    self._testScope(get_factory(True), prefix=None, use_outer_scope=False)
    self._testScope(get_factory(False), use_outer_scope=True)
    self._testScope(get_factory(False), use_outer_scope=False)
    self._testScope(get_factory(False), prefix=None, use_outer_scope=False)


class MultiDimensionalLSTMTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def testMultiDimensionalLSTMAllRNNContainers(self):
    feature_dims = (3, 4, 5)
    input_size = feature_dims
    batch_size = 2
    max_length = 8
    sequence_length = [4, 6]
    with self.session(graph=ops.Graph()) as sess:
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None,) + input_size)
      ]
      inputs_using_dim = max_length * [
          array_ops.placeholder(
              dtypes.float32, shape=(batch_size,) + input_size)
      ]
      inputs_c = array_ops_stack.stack(inputs)
      # Create a cell for the whole test. This is fine because the cell has no
      # variables.
      cell = DummyMultiDimensionalLSTM(feature_dims)
      state_saver = TestStateSaver(batch_size, input_size)
      outputs_static, state_static = rnn.static_rnn(
          cell, inputs, dtype=dtypes.float32, sequence_length=sequence_length)
      outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
          cell,
          inputs_c,
          dtype=dtypes.float32,
          time_major=True,
          sequence_length=sequence_length)
      outputs_bid, state_fw, state_bw = rnn.static_bidirectional_rnn(
          cell,
          cell,
          inputs_using_dim,
          dtype=dtypes.float32,
          sequence_length=sequence_length)
      outputs_sav, state_sav = rnn.static_state_saving_rnn(
          cell,
          inputs_using_dim,
          sequence_length=sequence_length,
          state_saver=state_saver,
          state_name=("h", "c"))

      self.assertEqual(outputs_dynamic.get_shape().as_list(),
                       inputs_c.get_shape().as_list())
      for out, inp in zip(outputs_static, inputs):
        self.assertEqual(out.get_shape().as_list(), inp.get_shape().as_list())
      for out, inp in zip(outputs_bid, inputs_using_dim):
        input_shape_list = inp.get_shape().as_list()
        # fwd and bwd activations are concatenated along the second dim.
        input_shape_list[1] *= 2
        self.assertEqual(out.get_shape().as_list(), input_shape_list)

      variables_lib.global_variables_initializer().run()

      input_total_size = (batch_size,) + input_size
      input_value = np.random.randn(*input_total_size)
      outputs_static_v = sess.run(
          outputs_static, feed_dict={
              inputs[0]: input_value
          })
      outputs_dynamic_v = sess.run(
          outputs_dynamic, feed_dict={
              inputs[0]: input_value
          })
      outputs_bid_v = sess.run(
          outputs_bid, feed_dict={
              inputs_using_dim[0]: input_value
          })
      outputs_sav_v = sess.run(
          outputs_sav, feed_dict={
              inputs_using_dim[0]: input_value
          })

      self.assertAllEqual(outputs_static_v, outputs_dynamic_v)
      self.assertAllEqual(outputs_static_v, outputs_sav_v)
      outputs_static_array = np.array(outputs_static_v)
      outputs_static_array_double = np.concatenate(
          (outputs_static_array, outputs_static_array), axis=2)
      outputs_bid_array = np.array(outputs_bid_v)
      self.assertAllEqual(outputs_static_array_double, outputs_bid_array)

      state_static_v = sess.run(
          state_static, feed_dict={
              inputs[0]: input_value
          })
      state_dynamic_v = sess.run(
          state_dynamic, feed_dict={
              inputs[0]: input_value
          })
      state_bid_fw_v = sess.run(
          state_fw, feed_dict={
              inputs_using_dim[0]: input_value
          })
      state_bid_bw_v = sess.run(
          state_bw, feed_dict={
              inputs_using_dim[0]: input_value
          })
      state_sav_v = sess.run(
          state_sav, feed_dict={
              inputs_using_dim[0]: input_value
          })
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_dynamic_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_sav_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_bid_fw_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_bid_bw_v))


class NestedLSTMTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def testNestedIOLSTMAllRNNContainers(self):
    input_size = 5
    batch_size = 2
    state_size = 6
    max_length = 8
    sequence_length = [4, 6]
    with self.session(graph=ops.Graph()) as sess:
      state_saver = TestStateSaver(batch_size, state_size)
      single_input = (array_ops.placeholder(
          dtypes.float32, shape=(None, input_size)),
                      array_ops.placeholder(
                          dtypes.float32, shape=(None, input_size)))
      inputs = max_length * [single_input]
      inputs_c = (array_ops_stack.stack([input_[0] for input_ in inputs]),
                  array_ops_stack.stack([input_[1] for input_ in inputs]))
      single_input_using_dim = (array_ops.placeholder(
          dtypes.float32, shape=(batch_size, input_size)),
                                array_ops.placeholder(
                                    dtypes.float32,
                                    shape=(batch_size, input_size)))
      inputs_using_dim = max_length * [single_input_using_dim]

      # Create a cell for the whole test. This is fine because the cell has no
      # variables.
      cell = NestedRNNCell()
      outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
          cell,
          inputs_c,
          dtype=dtypes.float32,
          time_major=True,
          sequence_length=sequence_length)
      outputs_static, state_static = rnn.static_rnn(
          cell, inputs, dtype=dtypes.float32, sequence_length=sequence_length)
      outputs_bid, state_fw, state_bw = rnn.static_bidirectional_rnn(
          cell,
          cell,
          inputs_using_dim,
          dtype=dtypes.float32,
          sequence_length=sequence_length)
      outputs_sav, state_sav = rnn.static_state_saving_rnn(
          cell,
          inputs_using_dim,
          sequence_length=sequence_length,
          state_saver=state_saver,
          state_name=("h", "c"))

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

      variables_lib.global_variables_initializer().run()

      input_total_size = (batch_size, input_size)
      input_value = (np.random.randn(*input_total_size),
                     np.random.randn(*input_total_size))
      outputs_dynamic_v = sess.run(
          outputs_dynamic, feed_dict={
              single_input: input_value
          })
      outputs_static_v = sess.run(
          outputs_static, feed_dict={
              single_input: input_value
          })
      outputs_sav_v = sess.run(
          outputs_sav, feed_dict={
              single_input_using_dim: input_value
          })
      outputs_bid_v = sess.run(
          outputs_bid, feed_dict={
              single_input_using_dim: input_value
          })

      self.assertAllEqual(outputs_static_v,
                          np.transpose(outputs_dynamic_v, (1, 0, 2, 3)))
      self.assertAllEqual(outputs_static_v, outputs_sav_v)
      outputs_static_array = np.array(outputs_static_v)
      outputs_static_array_double = np.concatenate(
          (outputs_static_array, outputs_static_array), axis=3)
      outputs_bid_array = np.array(outputs_bid_v)
      self.assertAllEqual(outputs_static_array_double, outputs_bid_array)

      state_dynamic_v = sess.run(
          state_dynamic, feed_dict={
              single_input: input_value
          })
      state_static_v = sess.run(
          state_static, feed_dict={
              single_input: input_value
          })
      state_bid_fw_v = sess.run(
          state_fw, feed_dict={
              single_input_using_dim: input_value
          })
      state_bid_bw_v = sess.run(
          state_bw, feed_dict={
              single_input_using_dim: input_value
          })
      state_sav_v = sess.run(
          state_sav, feed_dict={
              single_input_using_dim: input_value
          })
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_dynamic_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_sav_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_bid_fw_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_bid_bw_v))


class StateSaverRNNTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _factory(self, scope, state_saver):
    num_units = state_saver.state_size // 2
    batch_size = state_saver.batch_size
    input_size = 5
    max_length = 8
    initializer = init_ops.random_uniform_initializer(
        -0.01, 0.01, seed=self._seed)
    cell = rnn_cell.LSTMCell(
        num_units,
        use_peepholes=False,
        initializer=initializer,
        state_is_tuple=False)
    inputs = max_length * [
        array_ops.zeros(dtype=dtypes.float32, shape=(batch_size, input_size))
    ]
    out, state = rnn.static_state_saving_rnn(
        cell,
        inputs,
        state_saver=state_saver,
        state_name="save_lstm",
        scope=scope)
    return out, state, state_saver

  def _testScope(self, prefix="prefix", use_outer_scope=True):
    num_units = 3
    batch_size = 2
    state_saver = TestStateSaver(batch_size, 2 * num_units)

    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          self._factory(scope=scope, state_saver=state_saver)
      else:
        self._factory(scope=prefix, state_saver=state_saver)
        variables_lib.global_variables_initializer()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = variables_lib.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("RNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  def testStateSaverRNNScope(self):
    self._testScope(use_outer_scope=True)
    self._testScope(use_outer_scope=False)
    self._testScope(prefix=None, use_outer_scope=False)

  def testStateSaverCallsSaveState(self):
    """Test that number of calls to state and save_state is equal.

    Test if the order of actual evaluating or skipping evaluation of out,
    state tensors, which are the output tensors from static_state_saving_rnn,
    have influence on number of calls to save_state and state methods of
    state_saver object (the number of calls should be same.)
    """
    self.skipTest("b/124196246 Breakage for sess.run([out, ...]): 2 != 1")

    num_units = 3
    batch_size = 2
    state_saver = TestStateSaverWithCounters(batch_size, 2 * num_units)
    out, state, state_saver = self._factory(scope=None, state_saver=state_saver)

    with self.cached_session() as sess:
      sess.run(variables_lib.global_variables_initializer())
      sess.run(variables_lib.local_variables_initializer())

      _, _, num_state_calls, num_save_state_calls = sess.run([
          out,
          state,
          state_saver.num_state_calls,
          state_saver.num_save_state_calls])
      self.assertEqual(num_state_calls, num_save_state_calls)

      _, num_state_calls, num_save_state_calls = sess.run([
          out,
          state_saver.num_state_calls,
          state_saver.num_save_state_calls])
      self.assertEqual(num_state_calls, num_save_state_calls)

      _, num_state_calls, num_save_state_calls = sess.run([
          state,
          state_saver.num_state_calls,
          state_saver.num_save_state_calls])
      self.assertEqual(num_state_calls, num_save_state_calls)

class GRUTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def testDynamic(self):
    time_steps = 8
    num_units = 3
    input_size = 5
    batch_size = 2

    input_values = np.random.randn(time_steps, batch_size, input_size)

    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    with self.session(graph=ops.Graph()) as sess:
      concat_inputs = array_ops.placeholder(
          dtypes.float32, shape=(time_steps, batch_size, input_size))

      cell = rnn_cell.GRUCell(num_units=num_units)

      with variable_scope.variable_scope("dynamic_scope"):
        outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
            cell,
            inputs=concat_inputs,
            sequence_length=sequence_length,
            time_major=True,
            dtype=dtypes.float32)

      feeds = {concat_inputs: input_values}

      # Initialize
      variables_lib.global_variables_initializer().run(feed_dict=feeds)

      sess.run([outputs_dynamic, state_dynamic], feed_dict=feeds)

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)
        variables_lib.global_variables_initializer()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = variables_lib.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("RNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  @test_util.run_v1_only("b/124229375")
  def testDynamicScope(self):
    time_steps = 8
    num_units = 3
    input_size = 5
    batch_size = 2
    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    def factory(scope):
      concat_inputs = array_ops.placeholder(
          dtypes.float32, shape=(time_steps, batch_size, input_size))
      cell = rnn_cell.GRUCell(num_units=num_units)
      return rnn.dynamic_rnn(
          cell,
          inputs=concat_inputs,
          sequence_length=sequence_length,
          time_major=True,
          dtype=dtypes.float32,
          scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


class RawRNNTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def _testRawRNN(self, max_time):
    with self.session(graph=ops.Graph()) as sess:
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = array_ops.placeholder(
          shape=(max_time, batch_size, input_depth), dtype=dtypes.float32)
      sequence_length = array_ops.placeholder(
          shape=(batch_size,), dtype=dtypes.int32)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, unused_loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:  # time == 0
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          next_state = cell_state  # copy state through
        elements_finished = (time_ >= sequence_length)
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = cond.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      reuse_scope = variable_scope.get_variable_scope()

      outputs_ta, final_state, _ = rnn.raw_rnn(cell, loop_fn, scope=reuse_scope)
      outputs = outputs_ta.stack()

      reuse_scope.reuse_variables()
      outputs_dynamic_rnn, final_state_dynamic_rnn = rnn.dynamic_rnn(
          cell,
          inputs,
          time_major=True,
          dtype=dtypes.float32,
          sequence_length=sequence_length,
          scope=reuse_scope)

      variables = variables_lib.trainable_variables()
      gradients = gradients_impl.gradients([outputs, final_state],
                                           [inputs] + variables)
      gradients_dynamic_rnn = gradients_impl.gradients(
          [outputs_dynamic_rnn, final_state_dynamic_rnn], [inputs] + variables)

      variables_lib.global_variables_initializer().run()

      rand_input = np.random.randn(max_time, batch_size, input_depth)
      if max_time == 0:
        rand_seq_len = np.zeros(batch_size)
      else:
        rand_seq_len = np.random.randint(max_time, size=batch_size)

      # To ensure same output lengths for dynamic_rnn and raw_rnn
      rand_seq_len[0] = max_time

      (outputs_val, outputs_dynamic_rnn_val, final_state_val,
       final_state_dynamic_rnn_val) = sess.run(
           [outputs, outputs_dynamic_rnn, final_state, final_state_dynamic_rnn],
           feed_dict={
               inputs: rand_input,
               sequence_length: rand_seq_len
           })

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
            feed_dict={
                inputs: rand_input,
                sequence_length: rand_seq_len
            })
        gradients_dynamic_rnn_val = sess.run(
            gradients_dynamic_rnn,
            feed_dict={
                inputs: rand_input,
                sequence_length: rand_seq_len
            })
        self.assertEqual(len(gradients_val), len(gradients_dynamic_rnn_val))
        input_gradients_val = gradients_val[0]
        input_gradients_dynamic_rnn_val = gradients_dynamic_rnn_val[0]
        self.assertAllClose(input_gradients_val,
                            input_gradients_dynamic_rnn_val)
        for i in range(1, len(gradients_val)):
          self.assertAllClose(gradients_dynamic_rnn_val[i], gradients_val[i])

  @test_util.run_v1_only("b/124229375")
  def testRawRNNZeroLength(self):
    # NOTE: Because with 0 time steps, raw_rnn does not have shape
    # information about the input, it is impossible to perform
    # gradients comparisons as the gradients eval will fail.  So this
    # case skips the gradients test.
    self._testRawRNN(max_time=0)

  def testRawRNN(self):
    self._testRawRNN(max_time=10)

  @test_util.run_v1_only("b/124229375")
  def testLoopState(self):
    with self.session(graph=ops.Graph()):
      max_time = 10
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, loop_state):
        if cell_output is None:
          loop_state = constant_op.constant([0])
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          loop_state = array_ops_stack.stack(
              [array_ops.squeeze(loop_state) + 1])
          next_state = cell_state
        emit_output = cell_output  # == None for time == 0
        elements_finished = array_ops.tile([time_ >= max_time], [batch_size])
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = cond.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output,
                loop_state)

      r = rnn.raw_rnn(cell, loop_fn)
      loop_state = r[-1]
      self.assertEqual([10], self.evaluate(loop_state))

  @test_util.run_v1_only("b/124229375")
  def testLoopStateWithTensorArray(self):
    with self.session(graph=ops.Graph()):
      max_time = 4
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, loop_state):
        if cell_output is None:
          loop_state = tensor_array_ops.TensorArray(
              dynamic_size=True,
              size=0,
              dtype=dtypes.int32,
              clear_after_read=False)
          loop_state = loop_state.write(0, 1)
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          loop_state = loop_state.write(time_,
                                        loop_state.read(time_ - 1) + time_)
          next_state = cell_state
        emit_output = cell_output  # == None for time == 0
        elements_finished = array_ops.tile([time_ >= max_time], [batch_size])
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = cond.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output,
                loop_state)

      r = rnn.raw_rnn(cell, loop_fn)
      loop_state = r[-1]
      loop_state = loop_state.stack()
      self.assertAllEqual([1, 2, 2 + 2, 4 + 3, 7 + 4], loop_state)

  @test_util.run_v1_only("b/124229375")
  def testEmitDifferentStructureThanCellOutput(self):
    with self.session(graph=ops.Graph()) as sess:
      max_time = 10
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)
      # Verify emit shapes may be unknown by feeding a placeholder that
      # determines an emit shape.
      unknown_dim = array_ops.placeholder(dtype=dtypes.int32)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, _):
        if cell_output is None:
          emit_output = (array_ops.zeros([2, 3], dtype=dtypes.int32),
                         array_ops.zeros([unknown_dim], dtype=dtypes.int64))
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          emit_output = (array_ops.ones([batch_size, 2, 3], dtype=dtypes.int32),
                         array_ops.ones(
                             [batch_size, unknown_dim], dtype=dtypes.int64))
          next_state = cell_state
        elements_finished = array_ops.tile([time_ >= max_time], [batch_size])
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = cond.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      r = rnn.raw_rnn(cell, loop_fn)
      output_ta = r[0]
      self.assertEqual(2, len(output_ta))
      self.assertEqual([dtypes.int32, dtypes.int64],
                       [ta.dtype for ta in output_ta])
      output = [ta.stack() for ta in output_ta]
      output_vals = sess.run(output, feed_dict={unknown_dim: 1})
      self.assertAllEqual(
          np.ones((max_time, batch_size, 2, 3), np.int32), output_vals[0])
      self.assertAllEqual(
          np.ones((max_time, batch_size, 1), np.int64), output_vals[1])

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)
        variables_lib.global_variables_initializer()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = variables_lib.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("RNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  @test_util.run_v1_only("b/124229375")
  def testRawRNNScope(self):
    max_time = 10
    batch_size = 16
    input_depth = 4
    num_units = 3

    def factory(scope):
      inputs = array_ops.placeholder(
          shape=(max_time, batch_size, input_depth), dtype=dtypes.float32)
      sequence_length = array_ops.placeholder(
          shape=(batch_size,), dtype=dtypes.int32)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, unused_loop_state):
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:  # time == 0
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          next_state = cell_state

        elements_finished = (time_ >= sequence_length)
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = cond.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      return rnn.raw_rnn(cell, loop_fn, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


class DeviceWrapperCell(rnn_cell.RNNCell):
  """Class to ensure cell calculation happens on a specific device."""

  def __init__(self, cell, device):
    self._cell = cell
    self._device = device

  @property
  def output_size(self):
    return self._cell.output_size

  @property
  def state_size(self):
    return self._cell.state_size

  def __call__(self, input_, state, scope=None):
    if self._device is not None:
      with ops.device(self._device):
        return self._cell(input_, state, scope=scope)
    else:
      return self._cell(input_, state, scope=scope)


class TensorArrayOnCorrectDeviceTest(test.TestCase):

  def _execute_rnn_on(self,
                      rnn_device=None,
                      cell_device=None,
                      input_device=None):
    batch_size = 3
    time_steps = 7
    input_size = 5
    num_units = 10

    cell = rnn_cell.LSTMCell(num_units, use_peepholes=True)
    gpu_cell = DeviceWrapperCell(cell, cell_device)
    inputs = np.random.randn(batch_size, time_steps, input_size).astype(
        np.float32)
    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    if input_device is not None:
      with ops.device(input_device):
        inputs = constant_op.constant(inputs)

    if rnn_device is not None:
      with ops.device(rnn_device):
        outputs, _ = rnn.dynamic_rnn(
            gpu_cell,
            inputs,
            sequence_length=sequence_length,
            dtype=dtypes.float32)
    else:
      outputs, _ = rnn.dynamic_rnn(
          gpu_cell,
          inputs,
          sequence_length=sequence_length,
          dtype=dtypes.float32)

    with self.session() as sess:
      opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      variables_lib.global_variables_initializer().run()
      sess.run(outputs, options=opts, run_metadata=run_metadata)

    return run_metadata

  def _retrieve_cpu_gpu_stats(self, run_metadata):
    cpu_stats = None
    gpu_stats = None
    step_stats = run_metadata.step_stats
    for ds in step_stats.dev_stats:
      if "cpu:0" in ds.device[-5:].lower():
        cpu_stats = ds.node_stats
      if "gpu:0" == ds.device[-5:].lower():
        gpu_stats = ds.node_stats
    return cpu_stats, gpu_stats

  @test_util.run_v1_only("b/124229375")
  def testRNNOnCPUCellOnGPU(self):
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    gpu_dev = test.gpu_device_name()
    run_metadata = self._execute_rnn_on(
        rnn_device="/cpu:0", cell_device=gpu_dev)
    cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)

    def _assert_in(op_str, in_stats, out_stats):
      self.assertTrue(any(op_str in s.node_name for s in in_stats))
      self.assertFalse(any(op_str in s.node_name for s in out_stats))

    # Writes happen at output of RNN cell
    _assert_in("TensorArrayWrite", gpu_stats, cpu_stats)
    # Gather happens on final TensorArray
    _assert_in("TensorArrayGather", gpu_stats, cpu_stats)
    # Reads happen at input to RNN cell
    _assert_in("TensorArrayRead", cpu_stats, gpu_stats)
    # Scatters happen to get initial input into TensorArray
    _assert_in("TensorArrayScatter", cpu_stats, gpu_stats)

  @test_util.run_v1_only("b/124229375")
  def testRNNOnCPUCellOnCPU(self):
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    gpu_dev = test.gpu_device_name()
    run_metadata = self._execute_rnn_on(
        rnn_device="/cpu:0", cell_device="/cpu:0", input_device=gpu_dev)
    cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)

    def _assert_in(op_str, in_stats, out_stats):
      self.assertTrue(any(op_str in s.node_name for s in in_stats))
      self.assertFalse(any(op_str in s.node_name for s in out_stats))

    # All TensorArray operations happen on CPU
    _assert_in("TensorArray", cpu_stats, gpu_stats)

  @test_util.run_v1_only("b/124229375")
  def testInputOnGPUCellNotDeclared(self):
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    gpu_dev = test.gpu_device_name()
    run_metadata = self._execute_rnn_on(input_device=gpu_dev)
    cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)

    def _assert_in(op_str, in_stats, out_stats):
      self.assertTrue(any(op_str in s.node_name for s in in_stats))
      self.assertFalse(any(op_str in s.node_name for s in out_stats))

    # Everything happens on GPU
    _assert_in("TensorArray", gpu_stats, cpu_stats)


class RNNCellTest(test.TestCase, parameterized.TestCase):

  @test_util.run_v1_only("b/124229375")
  def testBasicRNNCell(self):
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 2])
        cell = rnn_cell_impl.BasicRNNCell(2)
        g, _ = cell(x, m)
        self.assertEqual([
            "root/basic_rnn_cell/%s:0" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
            "root/basic_rnn_cell/%s:0" % rnn_cell_impl._BIAS_VARIABLE_NAME
        ], [v.name for v in cell.trainable_variables])
        self.assertFalse(cell.non_trainable_variables)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g], {
            x: np.array([[1., 1.]]),
            m: np.array([[0.1, 0.1]])
        })
        self.assertEqual(res[0].shape, (1, 2))

  @test_util.run_v1_only("b/124229375")
  def testBasicRNNCellNotTrainable(self):
    with self.cached_session() as sess:

      def not_trainable_getter(getter, *args, **kwargs):
        kwargs["trainable"] = False
        return getter(*args, **kwargs)

      with variable_scope.variable_scope(
          "root",
          initializer=init_ops.constant_initializer(0.5),
          custom_getter=not_trainable_getter):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 2])
        cell = rnn_cell_impl.BasicRNNCell(2)
        g, _ = cell(x, m)
        self.assertFalse(cell.trainable_variables)
        self.assertEqual([
            "root/basic_rnn_cell/%s:0" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
            "root/basic_rnn_cell/%s:0" % rnn_cell_impl._BIAS_VARIABLE_NAME
        ], [v.name for v in cell.non_trainable_variables])
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g], {
            x: np.array([[1., 1.]]),
            m: np.array([[0.1, 0.1]])
        })
        self.assertEqual(res[0].shape, (1, 2))

  @test_util.run_v1_only("b/124229375")
  def testGRUCell(self):
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 2])
        g, _ = rnn_cell_impl.GRUCell(2)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g], {
            x: np.array([[1., 1.]]),
            m: np.array([[0.1, 0.1]])
        })
        # Smoke test
        self.assertAllClose(res[0], [[0.175991, 0.175991]])
      with variable_scope.variable_scope(
          "other", initializer=init_ops.constant_initializer(0.5)):
        # Test GRUCell with input_size != num_units.
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 2])
        g, _ = rnn_cell_impl.GRUCell(2)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g], {
            x: np.array([[1., 1., 1.]]),
            m: np.array([[0.1, 0.1]])
        })
        # Smoke test
        self.assertAllClose(res[0], [[0.156736, 0.156736]])

  @test_util.run_v1_only("b/124229375")
  def testBasicLSTMCell(self):
    for dtype in [dtypes.float16, dtypes.float32]:
      np_dtype = dtype.as_numpy_dtype
      with self.session(graph=ops.Graph()) as sess:
        with variable_scope.variable_scope(
            "root", initializer=init_ops.constant_initializer(0.5)):
          x = array_ops.zeros([1, 2], dtype=dtype)
          m = array_ops.zeros([1, 8], dtype=dtype)
          cell = rnn_cell_impl.MultiRNNCell(
              [
                  rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=False)
                  for _ in range(2)
              ],
              state_is_tuple=False)
          self.assertEqual(cell.dtype, None)
          self.assertIn("cell-0", cell._trackable_children())
          self.assertIn("cell-1", cell._trackable_children())
          cell.get_config()  # Should not throw an error
          g, out_m = cell(x, m)
          # Layer infers the input type.
          self.assertEqual(cell.dtype, dtype.name)
          expected_variable_names = [
              "root/multi_rnn_cell/cell_0/basic_lstm_cell/%s:0" %
              rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
              "root/multi_rnn_cell/cell_0/basic_lstm_cell/%s:0" %
              rnn_cell_impl._BIAS_VARIABLE_NAME,
              "root/multi_rnn_cell/cell_1/basic_lstm_cell/%s:0" %
              rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
              "root/multi_rnn_cell/cell_1/basic_lstm_cell/%s:0" %
              rnn_cell_impl._BIAS_VARIABLE_NAME
          ]
          self.assertEqual(expected_variable_names,
                           [v.name for v in cell.trainable_variables])
          self.assertFalse(cell.non_trainable_variables)
          sess.run([variables_lib.global_variables_initializer()])
          res = sess.run([g, out_m], {
              x: np.array([[1., 1.]]),
              m: 0.1 * np.ones([1, 8])
          })
          self.assertEqual(len(res), 2)
          variables = variables_lib.global_variables()
          self.assertEqual(expected_variable_names, [v.name for v in variables])
          # The numbers in results were not calculated, this is just a
          # smoke test.
          self.assertAllClose(res[0], np.array(
              [[0.240, 0.240]], dtype=np_dtype), 1e-2)
          expected_mem = np.array(
              [[0.689, 0.689, 0.448, 0.448, 0.398, 0.398, 0.240, 0.240]],
              dtype=np_dtype)
          self.assertAllClose(res[1], expected_mem, 1e-2)
        with variable_scope.variable_scope(
            "other", initializer=init_ops.constant_initializer(0.5)):
          # Test BasicLSTMCell with input_size != num_units.
          x = array_ops.zeros([1, 3], dtype=dtype)
          m = array_ops.zeros([1, 4], dtype=dtype)
          g, out_m = rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=False)(x, m)
          sess.run([variables_lib.global_variables_initializer()])
          res = sess.run(
              [g, out_m], {
                  x: np.array([[1., 1., 1.]], dtype=np_dtype),
                  m: 0.1 * np.ones([1, 4], dtype=np_dtype)
              })
          self.assertEqual(len(res), 2)

  @test_util.run_v1_only("b/124229375")
  def testBasicLSTMCellDimension0Error(self):
    """Tests that dimension 0 in both(x and m) shape must be equal."""
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        num_units = 2
        state_size = num_units * 2
        batch_size = 3
        input_size = 4
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size - 1, state_size])
        with self.assertRaises(ValueError):
          g, out_m = rnn_cell_impl.BasicLSTMCell(
              num_units, state_is_tuple=False)(x, m)
          sess.run([variables_lib.global_variables_initializer()])
          sess.run(
              [g, out_m], {
                  x: 1 * np.ones([batch_size, input_size]),
                  m: 0.1 * np.ones([batch_size - 1, state_size])
              })

  def testBasicLSTMCellStateSizeError(self):
    """Tests that state_size must be num_units * 2."""
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        num_units = 2
        state_size = num_units * 3  # state_size must be num_units * 2
        batch_size = 3
        input_size = 4
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size, state_size])
        with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
          g, out_m = rnn_cell_impl.BasicLSTMCell(
              num_units, state_is_tuple=False)(x, m)
          sess.run([variables_lib.global_variables_initializer()])
          sess.run(
              [g, out_m], {
                  x: 1 * np.ones([batch_size, input_size]),
                  m: 0.1 * np.ones([batch_size, state_size])
              })

  @test_util.run_v1_only("b/124229375")
  def testBasicLSTMCellStateTupleType(self):
    with self.cached_session():
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m0 = (array_ops.zeros([1, 2]),) * 2
        m1 = (array_ops.zeros([1, 2]),) * 2
        cell = rnn_cell_impl.MultiRNNCell(
            [rnn_cell_impl.BasicLSTMCell(2) for _ in range(2)],
            state_is_tuple=True)
        self.assertTrue(isinstance(cell.state_size, tuple))
        self.assertTrue(
            isinstance(cell.state_size[0], rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(
            isinstance(cell.state_size[1], rnn_cell_impl.LSTMStateTuple))

        # Pass in regular tuples
        _, (out_m0, out_m1) = cell(x, (m0, m1))
        self.assertTrue(isinstance(out_m0, rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(isinstance(out_m1, rnn_cell_impl.LSTMStateTuple))

        # Pass in LSTMStateTuples
        variable_scope.get_variable_scope().reuse_variables()
        zero_state = cell.zero_state(1, dtypes.float32)
        self.assertTrue(isinstance(zero_state, tuple))
        self.assertTrue(isinstance(zero_state[0], rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(isinstance(zero_state[1], rnn_cell_impl.LSTMStateTuple))
        _, (out_m0, out_m1) = cell(x, zero_state)
        self.assertTrue(isinstance(out_m0, rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(isinstance(out_m1, rnn_cell_impl.LSTMStateTuple))

  @test_util.run_v1_only("b/124229375")
  def testBasicLSTMCellWithStateTuple(self):
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m0 = array_ops.zeros([1, 4])
        m1 = array_ops.zeros([1, 4])
        cell = rnn_cell_impl.MultiRNNCell(
            [
                rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=False)
                for _ in range(2)
            ],
            state_is_tuple=True)
        g, (out_m0, out_m1) = cell(x, (m0, m1))
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [g, out_m0, out_m1], {
                x: np.array([[1., 1.]]),
                m0: 0.1 * np.ones([1, 4]),
                m1: 0.1 * np.ones([1, 4])
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

  @test_util.run_v1_only("b/124229375")
  def testLSTMCell(self):
    with self.cached_session() as sess:
      num_units = 8
      num_proj = 6
      state_size = num_units + num_proj
      batch_size = 3
      input_size = 2
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size, state_size])
        cell = rnn_cell_impl.LSTMCell(
            num_units=num_units,
            num_proj=num_proj,
            forget_bias=1.0,
            state_is_tuple=False)
        output, state = cell(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [output, state], {
                x: np.array([[1., 1.], [2., 2.], [3., 3.]]),
                m: 0.1 * np.ones((batch_size, state_size))
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

  @test_util.run_v1_only("b/124229375")
  def testLSTMCellVariables(self):
    with self.cached_session():
      num_units = 8
      num_proj = 6
      state_size = num_units + num_proj
      batch_size = 3
      input_size = 2
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size, state_size])
        cell = rnn_cell_impl.LSTMCell(
            num_units=num_units,
            num_proj=num_proj,
            forget_bias=1.0,
            state_is_tuple=False)
        cell(x, m)  # Execute to create variables
      variables = variables_lib.global_variables()
      self.assertEqual(variables[0].op.name, "root/lstm_cell/kernel")
      self.assertEqual(variables[1].op.name, "root/lstm_cell/bias")
      self.assertEqual(variables[2].op.name, "root/lstm_cell/projection/kernel")

  @test_util.run_in_graph_and_eager_modes
  def testWrapperCheckpointing(self):
    for wrapper_type in [
        rnn_cell_impl.DropoutWrapper,
        rnn_cell_impl.ResidualWrapper,
        lambda cell: rnn_cell_impl.MultiRNNCell([cell])]:
      cell = rnn_cell_impl.BasicRNNCell(1)
      wrapper = wrapper_type(cell)
      wrapper(array_ops.ones([1, 1]),
              state=wrapper.zero_state(batch_size=1, dtype=dtypes.float32))
      self.evaluate([v.initializer for v in cell.variables])
      checkpoint = trackable_utils.Checkpoint(wrapper=wrapper)
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      self.evaluate(cell._bias.assign([40.]))
      save_path = checkpoint.save(prefix)
      self.evaluate(cell._bias.assign([0.]))
      checkpoint.restore(save_path).assert_consumed().run_restore_ops()
      self.assertAllEqual([40.], self.evaluate(cell._bias))

  @test_util.run_in_graph_and_eager_modes
  def testResidualWrapper(self):
    wrapper_type = rnn_cell_impl.ResidualWrapper
    x = ops.convert_to_tensor(np.array([[1., 1., 1.]]))
    m = ops.convert_to_tensor(np.array([[0.1, 0.1, 0.1]]))
    base_cell = rnn_cell_impl.GRUCell(
        3, kernel_initializer=init_ops.constant_initializer(0.5),
        bias_initializer=init_ops.constant_initializer(0.5))
    g, m_new = base_cell(x, m)
    wrapper_object = wrapper_type(base_cell)
    wrapper_object.get_config()  # Should not throw an error

    self.assertIn("cell", wrapper_object._trackable_children())
    self.assertIs(wrapper_object._trackable_children()["cell"], base_cell)

    g_res, m_new_res = wrapper_object(x, m)
    self.evaluate([variables_lib.global_variables_initializer()])
    res = self.evaluate([g, g_res, m_new, m_new_res])
    # Residual connections
    self.assertAllClose(res[1], res[0] + [1., 1., 1.])
    # States are left untouched
    self.assertAllClose(res[2], res[3])

  @test_util.run_in_graph_and_eager_modes
  def testResidualWrapperWithSlice(self):
    wrapper_type = rnn_cell_impl.ResidualWrapper
    x = ops.convert_to_tensor(np.array([[1., 1., 1., 1., 1.]]))
    m = ops.convert_to_tensor(np.array([[0.1, 0.1, 0.1]]))
    base_cell = rnn_cell_impl.GRUCell(
        3, kernel_initializer=init_ops.constant_initializer(0.5),
        bias_initializer=init_ops.constant_initializer(0.5))
    g, m_new = base_cell(x, m)

    def residual_with_slice_fn(inp, out):
      inp_sliced = array_ops.slice(inp, [0, 0], [-1, 3])
      return inp_sliced + out

    g_res, m_new_res = wrapper_type(
        base_cell, residual_with_slice_fn)(x, m)
    self.evaluate([variables_lib.global_variables_initializer()])
    res_g, res_g_res, res_m_new, res_m_new_res = self.evaluate(
        [g, g_res, m_new, m_new_res])
    # Residual connections
    self.assertAllClose(res_g_res, res_g + [1., 1., 1.])
    # States are left untouched
    self.assertAllClose(res_m_new, res_m_new_res)

  def testDeviceWrapper(self):
    wrapper_type = rnn_cell_impl.DeviceWrapper
    x = array_ops.zeros([1, 3])
    m = array_ops.zeros([1, 3])
    cell = rnn_cell_impl.GRUCell(3)
    wrapped_cell = wrapper_type(cell, "/cpu:0")
    wrapped_cell.get_config()  # Should not throw an error
    self.assertEqual(wrapped_cell._trackable_children()["cell"], cell)

    outputs, _ = wrapped_cell(x, m)
    self.assertIn("cpu:0", outputs.device.lower())

  def _retrieve_cpu_gpu_stats(self, run_metadata):
    cpu_stats = None
    gpu_stats = None
    step_stats = run_metadata.step_stats
    for ds in step_stats.dev_stats:
      if "cpu:0" in ds.device[-5:].lower():
        cpu_stats = ds.node_stats
      if "gpu:0" == ds.device[-5:].lower():
        gpu_stats = ds.node_stats
    return cpu_stats, gpu_stats

  @test_util.run_v1_only("b/124229375")
  def testDeviceWrapperDynamicExecutionNodesAreAllProperlyLocated(self):
    if not test.is_gpu_available():
      # Can't perform this test w/o a GPU
      return

    gpu_dev = test.gpu_device_name()
    with self.session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 1, 3])
        cell = rnn_cell_impl.DeviceWrapper(rnn_cell_impl.GRUCell(3), gpu_dev)
        with ops.device("/cpu:0"):
          outputs, _ = rnn.dynamic_rnn(
              cell=cell, inputs=x, dtype=dtypes.float32)
        run_metadata = config_pb2.RunMetadata()
        opts = config_pb2.RunOptions(
            trace_level=config_pb2.RunOptions.FULL_TRACE)

        sess.run([variables_lib.global_variables_initializer()])
        _ = sess.run(outputs, options=opts, run_metadata=run_metadata)

      cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)
      self.assertFalse([s for s in cpu_stats if "gru_cell" in s.node_name])
      self.assertTrue([s for s in gpu_stats if "gru_cell" in s.node_name])

  @test_util.run_v1_only("b/124229375")
  def testMultiRNNCell(self):
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 4])
        multi_rnn_cell = rnn_cell_impl.MultiRNNCell(
            [rnn_cell_impl.GRUCell(2) for _ in range(2)],
            state_is_tuple=False)
        _, ml = multi_rnn_cell(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(ml, {
            x: np.array([[1., 1.]]),
            m: np.array([[0.1, 0.1, 0.1, 0.1]])
        })
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res, [[0.175991, 0.175991, 0.13248, 0.13248]])
        self.assertEqual(len(multi_rnn_cell.weights), 2 * 4)
        self.assertTrue(
            [x.dtype == dtypes.float32 for x in multi_rnn_cell.weights])

  @test_util.run_v1_only("b/124229375")
  def testMultiRNNCellWithStateTuple(self):
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m_bad = array_ops.zeros([1, 4])
        m_good = (array_ops.zeros([1, 2]), array_ops.zeros([1, 2]))

        # Test incorrectness of state
        with self.assertRaisesRegex(ValueError, "Expected state .* a tuple"):
          rnn_cell_impl.MultiRNNCell(
              [rnn_cell_impl.GRUCell(2) for _ in range(2)],
              state_is_tuple=True)(x, m_bad)

        _, ml = rnn_cell_impl.MultiRNNCell(
            [rnn_cell_impl.GRUCell(2) for _ in range(2)],
            state_is_tuple=True)(x, m_good)

        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            ml, {
                x: np.array([[1., 1.]]),
                m_good[0]: np.array([[0.1, 0.1]]),
                m_good[1]: np.array([[0.1, 0.1]])
            })

        # The numbers in results were not calculated, this is just a
        # smoke test.  However, these numbers should match those of
        # the test testMultiRNNCell.
        self.assertAllClose(res[0], [[0.175991, 0.175991]])
        self.assertAllClose(res[1], [[0.13248, 0.13248]])

  def testDeviceWrapperSerialization(self):
    wrapper_cls = rnn_cell_impl.DeviceWrapper
    cell = rnn_cell_impl.LSTMCell(10)
    wrapper = wrapper_cls(cell, "/cpu:0")
    config = wrapper.get_config()

    # Replace the cell in the config with real cell instance to work around the
    # reverse keras dependency issue.
    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    self.assertDictEqual(config, reconstructed_wrapper.get_config())
    self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

  def testResidualWrapperSerialization(self):
    wrapper_cls = rnn_cell_impl.ResidualWrapper
    cell = rnn_cell_impl.LSTMCell(10)
    wrapper = wrapper_cls(cell)
    config = wrapper.get_config()

    # Replace the cell in the config with real cell instance to work around the
    # reverse keras dependency issue.
    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    self.assertDictEqual(config, reconstructed_wrapper.get_config())
    self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

    wrapper = wrapper_cls(cell, residual_fn=lambda i, o: i + i + o)
    config = wrapper.get_config()

    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    # Assert the reconstructed function will perform the math correctly.
    self.assertEqual(reconstructed_wrapper._residual_fn(1, 2), 4)

    def residual_fn(inputs, outputs):
      return inputs * 3 + outputs

    wrapper = wrapper_cls(cell, residual_fn=residual_fn)
    config = wrapper.get_config()

    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    # Assert the reconstructed function will perform the math correctly.
    self.assertEqual(reconstructed_wrapper._residual_fn(1, 2), 5)

  def testDropoutWrapperSerialization(self):
    wrapper_cls = rnn_cell_impl.DropoutWrapper
    cell = rnn_cell_impl.LSTMCell(10)
    wrapper = wrapper_cls(cell)
    config = wrapper.get_config()

    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    self.assertDictEqual(config, reconstructed_wrapper.get_config())
    self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

    wrapper = wrapper_cls(cell, dropout_state_filter_visitor=lambda s: True)
    config = wrapper.get_config()

    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    self.assertTrue(reconstructed_wrapper._dropout_state_filter(None))

    def dropout_state_filter_visitor(unused_state):
      return False

    wrapper = wrapper_cls(
        cell, dropout_state_filter_visitor=dropout_state_filter_visitor)
    config = wrapper.get_config()

    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    self.assertFalse(reconstructed_wrapper._dropout_state_filter(None))

  def testSavedModel(self):
    if test_util.is_gpu_available():
      self.skipTest("b/175887901")

    with self.cached_session():
      root = autotrackable.AutoTrackable()
      root.cell = rnn_cell_impl.LSTMCell(8)
      @def_function.function(input_signature=[tensor_spec.TensorSpec([3, 8])])
      def call(x):
        state = root.cell.zero_state(3, dtype=x.dtype)
        y, _ = root.cell(x, state)
        return y
      root.call = call
      expected = root.call(array_ops.zeros((3, 8)))
      self.evaluate(variables_lib.global_variables_initializer())

      save_dir = os.path.join(self.get_temp_dir(), "saved_model")
      save.save(root, save_dir)
      loaded = load.load(save_dir)
      self.evaluate(variables_lib.global_variables_initializer())
      self.assertAllClose(
          expected, loaded.call(array_ops.zeros((3, 8))))


@test_util.run_all_in_graph_and_eager_modes
@test_util.run_all_without_tensor_float_32(
    "Uses an LSTMCell, which calls matmul")
class DropoutWrapperTest(test.TestCase, parameterized.TestCase):

  def _testDropoutWrapper(self,
                          batch_size=None,
                          time_steps=None,
                          parallel_iterations=None,
                          wrapper_type=None,
                          scope="root",
                          **kwargs):
    if batch_size is None and time_steps is None:
      # 2 time steps, batch size 1, depth 3
      batch_size = 1
      time_steps = 2
      x = constant_op.constant(
          [[[2., 2., 2.]], [[1., 1., 1.]]], dtype=dtypes.float32)
      m = rnn_cell_impl.LSTMStateTuple(
          *[constant_op.constant([[0.1, 0.1, 0.1]], dtype=dtypes.float32)] * 2)
    else:
      x = constant_op.constant(
          np.random.randn(time_steps, batch_size, 3).astype(np.float32))
      m = rnn_cell_impl.LSTMStateTuple(*[
          constant_op.
          constant([[0.1, 0.1, 0.1]] * batch_size, dtype=dtypes.float32)] * 2)
    outputs, final_state = rnn.dynamic_rnn(
        cell=wrapper_type(
            rnn_cell_impl.LSTMCell(
                3, initializer=init_ops.constant_initializer(0.5)),
            dtype=x.dtype, **kwargs),
        time_major=True,
        parallel_iterations=parallel_iterations,
        inputs=x,
        initial_state=m,
        scope=scope)
    self.evaluate([variables_lib.global_variables_initializer()])
    res = self.evaluate([outputs, final_state])
    self.assertEqual(res[0].shape, (time_steps, batch_size, 3))
    self.assertEqual(res[1].c.shape, (batch_size, 3))
    self.assertEqual(res[1].h.shape, (batch_size, 3))
    return res

  def testDropoutWrapperProperties(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper
    cell = rnn_cell_impl.BasicRNNCell(10)
    wrapper = wrapper_type(cell)
    # Github issue 15810
    self.assertEqual(wrapper.wrapped_cell, cell)
    self.assertEqual(wrapper.state_size, 10)
    self.assertEqual(wrapper.output_size, 10)

  def testDropoutWrapperZeroState(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper

    class _Cell(rnn_cell_impl.BasicRNNCell):

      def zero_state(self, batch_size=None, dtype=None):
        return "wrapped_cell_zero_state"
    wrapper = wrapper_type(_Cell(10))
    self.assertEqual(wrapper.zero_state(10, dtypes.float32),
                     "wrapped_cell_zero_state")

  def testDropoutWrapperKeepAllConstantInput(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep = array_ops.ones([])
    res = self._testDropoutWrapper(
        input_keep_prob=keep, output_keep_prob=keep, state_keep_prob=keep,
        wrapper_type=wrapper_type)
    true_full_output = np.array(
        [[[0.751109, 0.751109, 0.751109]], [[0.895509, 0.895509, 0.895509]]],
        dtype=np.float32)
    true_full_final_c = np.array(
        [[1.949385, 1.949385, 1.949385]], dtype=np.float32)
    self.assertAllClose(true_full_output, res[0])
    self.assertAllClose(true_full_output[1], res[1].h)
    self.assertAllClose(true_full_final_c, res[1].c)

  def testDropoutWrapperKeepAll(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep = variable_scope.get_variable("all", initializer=1.0)
    res = self._testDropoutWrapper(
        input_keep_prob=keep, output_keep_prob=keep, state_keep_prob=keep,
        wrapper_type=wrapper_type)
    true_full_output = np.array(
        [[[0.751109, 0.751109, 0.751109]], [[0.895509, 0.895509, 0.895509]]],
        dtype=np.float32)
    true_full_final_c = np.array(
        [[1.949385, 1.949385, 1.949385]], dtype=np.float32)
    self.assertAllClose(true_full_output, res[0])
    self.assertAllClose(true_full_output[1], res[1].h)
    self.assertAllClose(true_full_final_c, res[1].c)

  def testDropoutWrapperWithSeed(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_some = 0.5
    random_seed.set_random_seed(2)
    ## Use parallel_iterations = 1 in both calls to
    ## _testDropoutWrapper to ensure the (per-time step) dropout is
    ## consistent across both calls.  Otherwise the seed may not end
    ## up being munged consistently across both graphs.
    res_standard_1 = self._testDropoutWrapper(
        input_keep_prob=keep_some,
        output_keep_prob=keep_some,
        state_keep_prob=keep_some,
        seed=10,
        parallel_iterations=1,
        wrapper_type=wrapper_type,
        scope="root_1")
    random_seed.set_random_seed(2)
    res_standard_2 = self._testDropoutWrapper(
        input_keep_prob=keep_some,
        output_keep_prob=keep_some,
        state_keep_prob=keep_some,
        seed=10,
        parallel_iterations=1,
        wrapper_type=wrapper_type,
        scope="root_2")
    self.assertAllClose(res_standard_1[0], res_standard_2[0])
    self.assertAllClose(res_standard_1[1].c, res_standard_2[1].c)
    self.assertAllClose(res_standard_1[1].h, res_standard_2[1].h)

  def testDropoutWrapperKeepNoOutput(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_all = variable_scope.get_variable("all", initializer=1.0)
    keep_none = variable_scope.get_variable("none", initializer=1e-6)
    res = self._testDropoutWrapper(
        input_keep_prob=keep_all,
        output_keep_prob=keep_none,
        state_keep_prob=keep_all,
        wrapper_type=wrapper_type)
    true_full_output = np.array(
        [[[0.751109, 0.751109, 0.751109]], [[0.895509, 0.895509, 0.895509]]],
        dtype=np.float32)
    true_full_final_c = np.array(
        [[1.949385, 1.949385, 1.949385]], dtype=np.float32)
    self.assertAllClose(np.zeros(res[0].shape), res[0])
    self.assertAllClose(true_full_output[1], res[1].h)
    self.assertAllClose(true_full_final_c, res[1].c)

  def testDropoutWrapperKeepNoStateExceptLSTMCellMemory(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_all = variable_scope.get_variable("all", initializer=1.0)
    keep_none = variable_scope.get_variable("none", initializer=1e-6)
    # Even though we dropout state, by default DropoutWrapper never
    # drops out the memory ("c") term of an LSTMStateTuple.
    res = self._testDropoutWrapper(
        input_keep_prob=keep_all,
        output_keep_prob=keep_all,
        state_keep_prob=keep_none,
        wrapper_type=wrapper_type)
    true_c_state = np.array([[1.713925, 1.713925, 1.713925]], dtype=np.float32)
    true_full_output = np.array(
        [[[0.751109, 0.751109, 0.751109]], [[0.895509, 0.895509, 0.895509]]],
        dtype=np.float32)
    self.assertAllClose(true_full_output[0], res[0][0])
    # Second output is modified by zero input state
    self.assertGreater(np.linalg.norm(true_full_output[1] - res[0][1]), 1e-4)
    # h state has been set to zero
    self.assertAllClose(np.zeros(res[1].h.shape), res[1].h)
    # c state of an LSTMStateTuple is NEVER modified.
    self.assertAllClose(true_c_state, res[1].c)

  def testDropoutWrapperKeepNoInput(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_all = variable_scope.get_variable("all", initializer=1.0)
    keep_none = variable_scope.get_variable("none", initializer=1e-6)
    true_full_output = np.array(
        [[[0.751109, 0.751109, 0.751109]], [[0.895509, 0.895509, 0.895509]]],
        dtype=np.float32)
    true_full_final_c = np.array(
        [[1.949385, 1.949385, 1.949385]], dtype=np.float32)
    # All outputs are different because inputs are zeroed out
    res = self._testDropoutWrapper(
        input_keep_prob=keep_none,
        output_keep_prob=keep_all,
        state_keep_prob=keep_all,
        wrapper_type=wrapper_type)
    self.assertGreater(np.linalg.norm(res[0] - true_full_output), 1e-4)
    self.assertGreater(np.linalg.norm(res[1].h - true_full_output[1]), 1e-4)
    self.assertGreater(np.linalg.norm(res[1].c - true_full_final_c), 1e-4)

  def testDropoutWrapperRecurrentOutput(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_some = 0.8
    keep_all = variable_scope.get_variable("all", initializer=1.0)
    res = self._testDropoutWrapper(
        input_keep_prob=keep_all,
        output_keep_prob=keep_some,
        state_keep_prob=keep_all,
        variational_recurrent=True,
        wrapper_type=wrapper_type,
        input_size=3,
        batch_size=5,
        time_steps=7)
    # Ensure the same dropout pattern for all time steps
    output_mask = np.abs(res[0]) > 1e-6
    for m in output_mask[1:]:
      self.assertAllClose(output_mask[0], m)

  def testDropoutWrapperRecurrentStateInputAndOutput(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_some = 0.9
    res = self._testDropoutWrapper(
        input_keep_prob=keep_some,
        output_keep_prob=keep_some,
        state_keep_prob=keep_some,
        variational_recurrent=True,
        wrapper_type=wrapper_type,
        input_size=3,
        batch_size=5,
        time_steps=7)

    # Smoke test for the state/input masks.
    output_mask = np.abs(res[0]) > 1e-6
    for time_step in output_mask:
      # Ensure the same dropout output pattern for all time steps
      self.assertAllClose(output_mask[0], time_step)
      for batch_entry in time_step:
        # Assert all batch entries get the same mask
        self.assertAllClose(batch_entry, time_step[0])

    # For state, ensure all batch entries have the same mask
    state_c_mask = np.abs(res[1].c) > 1e-6
    state_h_mask = np.abs(res[1].h) > 1e-6
    for batch_entry in state_c_mask:
      self.assertAllClose(batch_entry, state_c_mask[0])
    for batch_entry in state_h_mask:
      self.assertAllClose(batch_entry, state_h_mask[0])

  def testDropoutWrapperRecurrentStateInputAndOutputWithSeed(self):
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_some = 0.9
    random_seed.set_random_seed(2347)
    np.random.seed(23487)
    res0 = self._testDropoutWrapper(
        input_keep_prob=keep_some,
        output_keep_prob=keep_some,
        state_keep_prob=keep_some,
        variational_recurrent=True,
        wrapper_type=wrapper_type,
        input_size=3,
        batch_size=5,
        time_steps=7,
        seed=-234987,
        scope="root_0")
    random_seed.set_random_seed(2347)
    np.random.seed(23487)
    res1 = self._testDropoutWrapper(
        input_keep_prob=keep_some,
        output_keep_prob=keep_some,
        state_keep_prob=keep_some,
        variational_recurrent=True,
        wrapper_type=wrapper_type,
        input_size=3,
        batch_size=5,
        time_steps=7,
        seed=-234987,
        scope="root_1")

    output_mask = np.abs(res0[0]) > 1e-6
    for time_step in output_mask:
      # Ensure the same dropout output pattern for all time steps
      self.assertAllClose(output_mask[0], time_step)
      for batch_entry in time_step:
        # Assert all batch entries get the same mask
        self.assertAllClose(batch_entry, time_step[0])

    # For state, ensure all batch entries have the same mask
    state_c_mask = np.abs(res0[1].c) > 1e-6
    state_h_mask = np.abs(res0[1].h) > 1e-6
    for batch_entry in state_c_mask:
      self.assertAllClose(batch_entry, state_c_mask[0])
    for batch_entry in state_h_mask:
      self.assertAllClose(batch_entry, state_h_mask[0])

    # Ensure seeded calculation is identical.
    self.assertAllClose(res0[0], res1[0])
    self.assertAllClose(res0[1].c, res1[1].c)
    self.assertAllClose(res0[1].h, res1[1].h)


if __name__ == "__main__":
  test.main()
