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

import os
import time
import timeit

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import network as keras_network
from tensorflow.python.layers import base as base_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables as variables_lib
import tensorflow.python.ops.data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops.losses import losses
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.training import saver
from tensorflow.python.training import training
from tensorflow.python.util import nest


class Plus1RNNCell(rnn_cell_impl.RNNCell):
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    return 5

  @property
  def state_size(self):
    return 5

  def call(self, input_, state, scope=None):
    return (input_ + 1, state + 1)


class ScalarStateRNNCell(rnn_cell_impl.RNNCell):
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    return 1

  @property
  def state_size(self):
    return tensor_shape.TensorShape([])

  def zero_state(self, batch_size, dtype):
    return array_ops.zeros([], dtype=dtypes.int32)

  def call(self, input_, state, scope=None):
    return (input_, state + 1)


class UnbalancedOutputRNNCell(rnn_cell_impl.RNNCell):
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    return  tensor_shape.TensorShape(1), tensor_shape.TensorShape((2))

  @property
  def state_size(self):
    return tensor_shape.TensorShape([])

  def zero_state(self, batch_size, dtype):
    return array_ops.zeros([], dtype=dtypes.int32)

  def call(self, input_, state, scope=None):
    concatenated = array_ops.concat((input_, input_), axis=-1)
    return (input_, concatenated), state + 1


class TensorArrayStateRNNCell(rnn_cell_impl.RNNCell):
  """RNN Cell its state as a TensorArray."""

  @property
  def output_size(self):
    return 1

  @property
  def state_size(self):
    return (tensor_shape.TensorShape([]), ())

  def zero_state(self, batch_size, dtype):
    return (array_ops.zeros([], dtype=dtypes.int32),
            tensor_array_ops.TensorArray(
                dtype=dtype, size=0, dynamic_size=True))

  def call(self, input_, state, scope=None):
    new_array = state[1].write(state[0], input_)
    return (input_, (state[0] + 1, new_array))


class KerasNetworkTFRNNs(keras_network.Network):

  def __init__(self, name=None):
    super(KerasNetworkTFRNNs, self).__init__(name=name)
    self._cell = rnn_cell_impl.MultiRNNCell(
        [rnn_cell_impl.LSTMCell(1) for _ in range(2)])

  def call(self, inputs):
    return self._cell(inputs, self._cell.get_initial_state(inputs))


class KerasNetworkKerasRNNs(keras_network.Network):

  def __init__(self, name=None):
    super(KerasNetworkKerasRNNs, self).__init__(name=name)
    self._cell = keras.layers.StackedRNNCells(
        [keras.layers.LSTMCell(1) for _ in range(2)])

  def call(self, inputs):
    return self._cell(inputs, self._cell.get_initial_state(inputs))


class RNNTest(test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_in_graph_and_eager_modes
  def testInvalidSequenceLengthShape(self):
    cell = Plus1RNNCell()
    if context.executing_eagerly():
      inputs = [constant_op.constant(np.ones((3, 4)))]
    else:
      inputs = [array_ops.placeholder(dtypes.float32, shape=(3, 4))]
    with self.assertRaisesRegexp(ValueError, "must be a vector"):
      rnn.dynamic_rnn(
          cell,
          array_ops.stack(inputs),
          dtype=dtypes.float32,
          sequence_length=[[4]])

  @test_util.run_in_graph_and_eager_modes
  def testInvalidDtype(self):
    if context.executing_eagerly():
      inputs = np.zeros((3, 4, 5), dtype=np.int32)
    else:
      inputs = array_ops.placeholder(dtypes.int32, shape=(3, 4, 5))

    cells = [
        rnn_cell_impl.BasicRNNCell,
        rnn_cell_impl.GRUCell,
        rnn_cell_impl.BasicLSTMCell,
        rnn_cell_impl.LSTMCell,
    ]
    for cell_cls in cells:
      with self.cached_session():
        with self.assertRaisesRegexp(
            ValueError, "RNN cell only supports floating"):
          cell = cell_cls(2, dtype=dtypes.int32)
          rnn.dynamic_rnn(cell, inputs, dtype=dtypes.int32)

  @test_util.run_in_graph_and_eager_modes
  def testBatchSizeFromInput(self):
    cell = Plus1RNNCell()
    in_eager_mode = context.executing_eagerly()
    # With static batch size
    if in_eager_mode:
      inputs = np.zeros((3, 4, 5), dtype=np.float32)
      initial_state = np.zeros((3, 5), dtype=np.float32)
    else:
      inputs = array_ops.placeholder(dtypes.float32, shape=(3, 4, 5))
      initial_state = array_ops.placeholder(dtypes.float32, shape=(3, 5))

    # - Without initial_state
    outputs, state = rnn.dynamic_rnn(cell, inputs, dtype=dtypes.float32)
    self.assertEqual(3, outputs.shape[0])
    self.assertEqual(3, state.shape[0])

    # - With initial_state
    outputs, state = rnn.dynamic_rnn(
        cell, inputs, initial_state=initial_state)
    self.assertEqual(3, outputs.shape[0])
    self.assertEqual(3, state.shape[0])

    # Without static batch size
    # Tensor shapes are fully determined with eager execution enabled,
    # so only run this test for graph construction.
    if not in_eager_mode:
      inputs = array_ops.placeholder(dtypes.float32, shape=(None, 4, 5))
      # - Without initial_state
      outputs, state = rnn.dynamic_rnn(cell, inputs, dtype=dtypes.float32)
      self.assertEqual(None, outputs.shape.dims[0].value)
      self.assertEqual(None, state.shape.dims[0].value)
      # - With initial_state
      outputs, state = rnn.dynamic_rnn(
          cell,
          inputs,
          initial_state=array_ops.placeholder(dtypes.float32, shape=(None, 5)))
      self.assertEqual(None, outputs.shape.dims[0].value)
      self.assertEqual(None, state.shape.dims[0].value)

  @test_util.run_in_graph_and_eager_modes
  def testScalarStateIsAccepted(self):
    cell = ScalarStateRNNCell()
    in_eager_mode = context.executing_eagerly()

    if in_eager_mode:
      inputs = np.array([[[1], [2], [3], [4]]], dtype=np.float32)
    else:
      inputs = array_ops.placeholder(dtypes.float32, shape=(1, 4, 1))

    with self.cached_session(use_gpu=True) as sess:
      outputs, state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32, sequence_length=[4])
      if not in_eager_mode:
        outputs, state = sess.run(
            [outputs, state], feed_dict={inputs: [[[1], [2], [3], [4]]]})

    self.assertAllEqual([[[1], [2], [3], [4]]], outputs)
    self.assertAllEqual(4, state)

  @test_util.run_in_graph_and_eager_modes
  def testUnbalancedOutputIsAccepted(self):
    cell = UnbalancedOutputRNNCell()
    in_eager_mode = context.executing_eagerly()

    if in_eager_mode:
      inputs = np.array([[[1], [2], [3], [4]]], dtype=np.float32)
    else:
      inputs = array_ops.placeholder(dtypes.float32, shape=(1, 4, 1))

    with self.cached_session(use_gpu=True) as sess:
      outputs, state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32, sequence_length=[4])
      if not in_eager_mode:
        outputs, state = sess.run(
            [outputs, state], feed_dict={inputs: [[[1], [2], [3], [4]]]})

    self.assertIsInstance(outputs, tuple)
    self.assertAllEqual([[[1], [2], [3], [4]]], outputs[0])
    self.assertAllEqual([[[1, 1], [2, 2], [3, 3], [4, 4]]], outputs[1])
    self.assertAllEqual(4, state)

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testEagerMemory(self):
    with context.eager_mode():
      cell = TensorArrayStateRNNCell()
      inputs = np.array([[[1], [2], [3], [4]]], dtype=np.float32)
      rnn.dynamic_rnn(cell, inputs, dtype=dtypes.float32, sequence_length=[4])

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
  def testTensorArrayStateIsAccepted(self):
    cell = TensorArrayStateRNNCell()
    in_eager_mode = context.executing_eagerly()

    if in_eager_mode:
      inputs = np.array([[[1], [2], [3], [4]]], dtype=np.float32)
    else:
      inputs = array_ops.placeholder(dtypes.float32, shape=(1, 4, 1))

    with self.cached_session(use_gpu=True) as sess:
      outputs, state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32, sequence_length=[4])
      state = (state[0], state[1].stack())
      if not in_eager_mode:
        outputs, state = sess.run(
            [outputs, state], feed_dict={
                inputs: [[[1], [2], [3], [4]]]
            })

    self.assertAllEqual([[[1], [2], [3], [4]]], outputs)
    self.assertAllEqual(4, state[0])
    self.assertAllEqual([[[1]], [[2]], [[3]], [[4]]], state[1])

  @test_util.run_deprecated_v1
  def testCellGetInitialState(self):
    cell = rnn_cell_impl.BasicRNNCell(5)
    with self.assertRaisesRegexp(
        ValueError, "batch_size and dtype cannot be None"):
      cell.get_initial_state(None, None, None)

    inputs = array_ops.placeholder(dtypes.float32, shape=(None, 4, 1))
    with self.assertRaisesRegexp(
        ValueError, "batch size from input tensor is different from"):
      cell.get_initial_state(inputs=inputs, batch_size=50, dtype=None)

    with self.assertRaisesRegexp(
        ValueError, "batch size from input tensor is different from"):
      cell.get_initial_state(
          inputs=inputs, batch_size=constant_op.constant(50), dtype=None)

    with self.assertRaisesRegexp(
        ValueError, "dtype from input tensor is different from"):
      cell.get_initial_state(inputs=inputs, batch_size=None, dtype=dtypes.int16)

    initial_state = cell.get_initial_state(
        inputs=inputs, batch_size=None, dtype=None)
    self.assertEqual(initial_state.shape.as_list(), [None, 5])
    self.assertEqual(initial_state.dtype, inputs.dtype)

    batch = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
    initial_state = cell.get_initial_state(None, batch, dtype)
    self.assertEqual(initial_state.shape.as_list(), [None, 5])
    self.assertEqual(initial_state.dtype, inputs.dtype)

  def _assert_cell_builds(self, cell_class, dtype, batch_size, in_size,
                          out_size):
    cell = cell_class(out_size, dtype=dtype)
    in_shape = tensor_shape.TensorShape((batch_size, in_size))
    cell.build(in_shape)
    state_output = cell.get_initial_state(
        inputs=None, batch_size=batch_size, dtype=dtype)
    cell_output, _ = cell(array_ops.zeros(in_shape, dtype), state_output)
    self.assertAllEqual([batch_size, out_size], cell_output.shape.as_list())

  @test_util.run_in_graph_and_eager_modes
  def testCellsBuild(self):
    f32 = dtypes.float32
    f64 = dtypes.float64
    self._assert_cell_builds(rnn_cell_impl.BasicRNNCell, f32, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.BasicRNNCell, f64, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.BasicLSTMCell, f32, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.BasicLSTMCell, f64, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.GRUCell, f32, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.GRUCell, f64, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.LSTMCell, f32, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.LSTMCell, f64, 5, 7, 3)

  @test_util.run_deprecated_v1
  def testRNNWithKerasSimpleRNNCell(self):
    with self.cached_session() as sess:
      input_shape = 10
      output_shape = 5
      timestep = 4
      batch = 100
      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = keras.utils.to_categorical(y_train)
      cell = keras.layers.SimpleRNNCell(output_shape)

      inputs = array_ops.placeholder(
          dtypes.float32, shape=(None, timestep, input_shape))
      predict = array_ops.placeholder(
          dtypes.float32, shape=(None, output_shape))

      outputs, state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32)
      self.assertEqual(outputs.shape.as_list(), [None, timestep, output_shape])
      self.assertEqual(state.shape.as_list(), [None, output_shape])
      loss = losses.softmax_cross_entropy(predict, state)
      train_op = training.GradientDescentOptimizer(0.001).minimize(loss)

      sess.run([variables_lib.global_variables_initializer()])
      _, outputs, state = sess.run(
          [train_op, outputs, state], {inputs: x_train, predict: y_train})

      self.assertEqual(len(outputs), batch)
      self.assertEqual(len(state), batch)

  @test_util.run_deprecated_v1
  def testRNNWithKerasGRUCell(self):
    with self.cached_session() as sess:
      input_shape = 10
      output_shape = 5
      timestep = 4
      batch = 100
      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = keras.utils.to_categorical(y_train)
      cell = keras.layers.GRUCell(output_shape)

      inputs = array_ops.placeholder(
          dtypes.float32, shape=(None, timestep, input_shape))
      predict = array_ops.placeholder(
          dtypes.float32, shape=(None, output_shape))

      outputs, state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32)
      self.assertEqual(outputs.shape.as_list(), [None, timestep, output_shape])
      self.assertEqual(state.shape.as_list(), [None, output_shape])
      loss = losses.softmax_cross_entropy(predict, state)
      train_op = training.GradientDescentOptimizer(0.001).minimize(loss)

      sess.run([variables_lib.global_variables_initializer()])
      _, outputs, state = sess.run(
          [train_op, outputs, state], {inputs: x_train, predict: y_train})

      self.assertEqual(len(outputs), batch)
      self.assertEqual(len(state), batch)

  @test_util.run_deprecated_v1
  def testRNNWithKerasLSTMCell(self):
    with self.cached_session() as sess:
      input_shape = 10
      output_shape = 5
      timestep = 4
      batch = 100
      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = keras.utils.to_categorical(y_train)
      cell = keras.layers.LSTMCell(output_shape)

      inputs = array_ops.placeholder(
          dtypes.float32, shape=(None, timestep, input_shape))
      predict = array_ops.placeholder(
          dtypes.float32, shape=(None, output_shape))

      outputs, state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32)
      self.assertEqual(outputs.shape.as_list(), [None, timestep, output_shape])
      self.assertEqual(len(state), 2)
      self.assertEqual(state[0].shape.as_list(), [None, output_shape])
      self.assertEqual(state[1].shape.as_list(), [None, output_shape])
      loss = losses.softmax_cross_entropy(predict, state[0])
      train_op = training.GradientDescentOptimizer(0.001).minimize(loss)

      sess.run([variables_lib.global_variables_initializer()])
      _, outputs, state = sess.run(
          [train_op, outputs, state], {inputs: x_train, predict: y_train})

      self.assertEqual(len(outputs), batch)
      self.assertEqual(len(state), 2)
      self.assertEqual(len(state[0]), batch)
      self.assertEqual(len(state[1]), batch)

  @test_util.run_deprecated_v1
  def testRNNWithStackKerasCell(self):
    with self.cached_session() as sess:
      input_shape = 10
      output_shape = 5
      timestep = 4
      batch = 100
      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      y_train = keras.utils.to_categorical(y_train)
      cell = keras.layers.StackedRNNCells(
          [keras.layers.LSTMCell(2 * output_shape),
           keras.layers.LSTMCell(output_shape)])

      inputs = array_ops.placeholder(
          dtypes.float32, shape=(None, timestep, input_shape))
      predict = array_ops.placeholder(
          dtypes.float32, shape=(None, output_shape))

      outputs, state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32)
      self.assertEqual(outputs.shape.as_list(), [None, timestep, output_shape])
      self.assertEqual(len(state), 2)
      state = nest.flatten(state)
      self.assertEqual(len(state), 4)
      self.assertEqual(state[0].shape.as_list(), [None, 2 * output_shape])
      self.assertEqual(state[1].shape.as_list(), [None, 2 * output_shape])
      self.assertEqual(state[2].shape.as_list(), [None, output_shape])
      self.assertEqual(state[3].shape.as_list(), [None, output_shape])
      loss = losses.softmax_cross_entropy(predict, state[2])
      train_op = training.GradientDescentOptimizer(0.001).minimize(loss)

      sess.run([variables_lib.global_variables_initializer()])
      _, outputs, state = sess.run(
          [train_op, outputs, state], {inputs: x_train, predict: y_train})

      self.assertEqual(len(outputs), batch)
      self.assertEqual(len(state), 4)
      for s in state:
        self.assertEqual(len(s), batch)

  @test_util.run_deprecated_v1
  def testStaticRNNWithKerasSimpleRNNCell(self):
    with self.cached_session() as sess:
      input_shape = 10
      output_shape = 5
      timestep = 4
      batch = 100
      (x_train, y_train), _ = testing_utils.get_test_data(
          train_samples=batch,
          test_samples=0,
          input_shape=(timestep, input_shape),
          num_classes=output_shape)
      x_train = np.transpose(x_train, (1, 0, 2))
      y_train = keras.utils.to_categorical(y_train)
      cell = keras.layers.SimpleRNNCell(output_shape)

      inputs = [array_ops.placeholder(
          dtypes.float32, shape=(None, input_shape))] * timestep
      predict = array_ops.placeholder(
          dtypes.float32, shape=(None, output_shape))

      outputs, state = rnn.static_rnn(
          cell, inputs, dtype=dtypes.float32)
      self.assertEqual(len(outputs), timestep)
      self.assertEqual(outputs[0].shape.as_list(), [None, output_shape])
      self.assertEqual(state.shape.as_list(), [None, output_shape])
      loss = losses.softmax_cross_entropy(predict, state)
      train_op = training.GradientDescentOptimizer(0.001).minimize(loss)

      sess.run([variables_lib.global_variables_initializer()])
      feed_dict = {i: d for i, d in zip(inputs, x_train)}
      feed_dict[predict] = y_train
      _, outputs, state = sess.run(
          [train_op, outputs, state], feed_dict)

      self.assertEqual(len(outputs), timestep)
      self.assertEqual(len(outputs[0]), batch)
      self.assertEqual(len(state), batch)

  @test_util.run_deprecated_v1
  def testKerasAndTFRNNLayerOutputComparison(self):
    input_shape = 10
    output_shape = 5
    timestep = 4
    batch = 20
    (x_train, _), _ = testing_utils.get_test_data(
        train_samples=batch,
        test_samples=0,
        input_shape=(timestep, input_shape),
        num_classes=output_shape)
    fix_weights_generator = keras.layers.SimpleRNNCell(output_shape)
    fix_weights_generator.build((None, input_shape))
    weights = fix_weights_generator.get_weights()

    with self.session(graph=ops_lib.Graph()) as sess:
      inputs = array_ops.placeholder(
          dtypes.float32, shape=(None, timestep, input_shape))
      cell = keras.layers.SimpleRNNCell(output_shape)
      tf_out, tf_state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32)
      cell.set_weights(weights)
      [tf_out, tf_state] = sess.run([tf_out, tf_state], {inputs: x_train})
    with self.session(graph=ops_lib.Graph()) as sess:
      k_input = keras.Input(shape=(timestep, input_shape),
                            dtype=dtypes.float32)
      cell = keras.layers.SimpleRNNCell(output_shape)
      layer = keras.layers.RNN(cell, return_sequences=True, return_state=True)
      keras_out = layer(k_input)
      cell.set_weights(weights)
      k_out, k_state = sess.run(keras_out, {k_input: x_train})
    self.assertAllClose(tf_out, k_out)
    self.assertAllClose(tf_state, k_state)

  @test_util.run_deprecated_v1
  def testSimpleRNNCellAndBasicRNNCellComparison(self):
    input_shape = 10
    output_shape = 5
    timestep = 4
    batch = 20
    (x_train, _), _ = testing_utils.get_test_data(
        train_samples=batch,
        test_samples=0,
        input_shape=(timestep, input_shape),
        num_classes=output_shape)
    fix_weights_generator = keras.layers.SimpleRNNCell(output_shape)
    fix_weights_generator.build((None, input_shape))
    # The SimpleRNNCell contains 3 weights: kernel, recurrent_kernel, and bias
    # The BasicRNNCell contains 2 weight: kernel and bias, where kernel is
    # zipped [kernel, recurrent_kernel] in SimpleRNNCell.
    keras_weights = fix_weights_generator.get_weights()
    kernel, recurrent_kernel, bias = keras_weights
    tf_weights = [np.concatenate((kernel, recurrent_kernel)), bias]

    with self.session(graph=ops_lib.Graph()) as sess:
      inputs = array_ops.placeholder(
          dtypes.float32, shape=(None, timestep, input_shape))
      cell = keras.layers.SimpleRNNCell(output_shape)
      k_out, k_state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32)
      cell.set_weights(keras_weights)
      [k_out, k_state] = sess.run([k_out, k_state], {inputs: x_train})
    with self.session(graph=ops_lib.Graph()) as sess:
      inputs = array_ops.placeholder(
          dtypes.float32, shape=(None, timestep, input_shape))
      cell = rnn_cell_impl.BasicRNNCell(output_shape)
      tf_out, tf_state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32)
      cell.set_weights(tf_weights)
      [tf_out, tf_state] = sess.run([tf_out, tf_state], {inputs: x_train})

    self.assertAllClose(tf_out, k_out, atol=1e-5)
    self.assertAllClose(tf_state, k_state, atol=1e-5)

  @test_util.run_deprecated_v1
  def testBasicLSTMCellInterchangeWithLSTMCell(self):
    with self.session(graph=ops_lib.Graph()) as sess:
      basic_cell = rnn_cell_impl.BasicLSTMCell(1)
      basic_cell(array_ops.ones([1, 1]),
                 state=basic_cell.get_initial_state(inputs=None,
                                                    batch_size=1,
                                                    dtype=dtypes.float32))
      self.evaluate([v.initializer for v in basic_cell.variables])
      self.evaluate(basic_cell._bias.assign([10.] * 4))
      save = saver.Saver()
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      save_path = save.save(sess, prefix)

    with self.session(graph=ops_lib.Graph()) as sess:
      lstm_cell = rnn_cell_impl.LSTMCell(1, name="basic_lstm_cell")
      lstm_cell(array_ops.ones([1, 1]),
                state=lstm_cell.get_initial_state(inputs=None,
                                                  batch_size=1,
                                                  dtype=dtypes.float32))
      self.evaluate([v.initializer for v in lstm_cell.variables])
      save = saver.Saver()
      save.restore(sess, save_path)
      self.assertAllEqual([10.] * 4, self.evaluate(lstm_cell._bias))

  # TODO(scottzhu): Look into updating for V2 Intializers.
  @test_util.run_deprecated_v1
  def testRNNCellSerialization(self):
    for cell in [
        rnn_cell_impl.LSTMCell(32, use_peepholes=True, cell_clip=True),
        rnn_cell_impl.BasicLSTMCell(32, dtype=dtypes.float32),
        rnn_cell_impl.BasicRNNCell(32, activation="relu", dtype=dtypes.float32),
        rnn_cell_impl.GRUCell(32, dtype=dtypes.float32)
    ]:
      with self.cached_session():
        x = keras.Input((None, 5))
        layer = keras.layers.RNN(cell)
        y = layer(x)
        model = keras.models.Model(x, y)
        model.compile(optimizer="rmsprop", loss="mse")

        # Test basic case serialization.
        x_np = np.random.random((6, 5, 5))
        y_np = model.predict(x_np)
        weights = model.get_weights()
        config = layer.get_config()
        # The custom_objects is important here since rnn_cell_impl is
        # not visible as a Keras layer, and also has a name conflict with
        # keras.LSTMCell and GRUCell.
        layer = keras.layers.RNN.from_config(
            config,
            custom_objects={
                "BasicRNNCell": rnn_cell_impl.BasicRNNCell,
                "GRUCell": rnn_cell_impl.GRUCell,
                "LSTMCell": rnn_cell_impl.LSTMCell,
                "BasicLSTMCell": rnn_cell_impl.BasicLSTMCell
            })
        y = layer(x)
        model = keras.models.Model(x, y)
        model.set_weights(weights)
        y_np_2 = model.predict(x_np)
        self.assertAllClose(y_np, y_np_2, atol=1e-4)

  def testRNNCellActsLikeKerasRNNCellInProperScope(self):
    with base_layers.keras_style_scope():
      kn1 = KerasNetworkTFRNNs(name="kn1")
      kn2 = KerasNetworkKerasRNNs(name="kn2")

    z = array_ops.zeros((2, 3))

    kn1(z)
    kn2(z)

    # pylint: disable=protected-access
    self.assertTrue(all("kn1" in v.name for v in kn1._cell.variables))
    self.assertTrue(all("kn2" in v.name for v in kn2._cell.variables))

    with base_layers.keras_style_scope():
      kn1_new = KerasNetworkTFRNNs(name="kn1_new")
      kn2_new = KerasNetworkKerasRNNs(name="kn2_new")

    kn2_new(z)
    # Most importantly, this doesn't fail due to variable scope reuse issues.
    kn1_new(z)

    self.assertTrue(all("kn1_new" in v.name for v in kn1_new._cell.variables))
    self.assertTrue(all("kn2_new" in v.name for v in kn2_new._cell.variables))


######### Benchmarking RNN code


def _static_vs_dynamic_rnn_benchmark_static(inputs_list_t, sequence_length):
  (_, input_size) = inputs_list_t[0].get_shape().as_list()
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = rnn_cell_impl.LSTMCell(
      num_units=input_size,
      use_peepholes=True,
      initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = rnn.static_rnn(
      cell,
      inputs_list_t,
      sequence_length=sequence_length,
      dtype=dtypes.float32)

  trainable_variables = ops_lib.get_collection(
      ops_lib.GraphKeys.TRAINABLE_VARIABLES)
  gradients = gradients_impl.gradients(outputs + [final_state],
                                       trainable_variables)

  return control_flow_ops.group(final_state, *(gradients + outputs))


def _static_vs_dynamic_rnn_benchmark_dynamic(inputs_t, sequence_length):
  (unused_0, unused_1, input_size) = inputs_t.get_shape().as_list()
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = rnn_cell_impl.LSTMCell(
      num_units=input_size,
      use_peepholes=True,
      initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = rnn.dynamic_rnn(
      cell, inputs_t, sequence_length=sequence_length, dtype=dtypes.float32)

  trainable_variables = ops_lib.get_collection(
      ops_lib.GraphKeys.TRAINABLE_VARIABLES)
  gradients = gradients_impl.gradients([outputs, final_state],
                                       trainable_variables)

  return control_flow_ops.group(final_state, outputs, *gradients)


def graph_creation_static_vs_dynamic_rnn_benchmark(max_time):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # These parameters don't matter
  batch_size = 512
  num_units = 512

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)
  ]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  def _create_static_rnn():
    with session.Session(config=config, graph=ops_lib.Graph()):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      _static_vs_dynamic_rnn_benchmark_static(inputs_list_t, sequence_length)

  def _create_dynamic_rnn():
    with session.Session(config=config, graph=ops_lib.Graph()):
      inputs_t = variables_lib.Variable(inputs, trainable=False).value()
      _static_vs_dynamic_rnn_benchmark_dynamic(inputs_t, sequence_length)

  delta_static = timeit.timeit(_create_static_rnn, number=5)
  delta_dynamic = timeit.timeit(_create_dynamic_rnn, number=5)

  print("%d \t %f \t %f \t %f" %
        (max_time, delta_static, delta_dynamic, delta_dynamic / delta_static))
  return delta_static, delta_dynamic


def _timer(sess, ops):
  # Warm in
  for _ in range(2):
    sess.run(ops)

  # Timing run
  runs = 20
  start = time.time()
  for _ in range(runs):
    sess.run(ops)
  end = time.time()
  return (end - start) / float(runs)


def static_vs_dynamic_rnn_benchmark(batch_size, max_time, num_units, use_gpu):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)
  ]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  # Using rnn()
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      ops = _static_vs_dynamic_rnn_benchmark_static(inputs_list_t,
                                                    sequence_length)
    variables_lib.global_variables_initializer().run()
    delta_static = _timer(sess, ops)

  # Using dynamic_rnn()
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_t = variables_lib.Variable(inputs, trainable=False).value()
      ops = _static_vs_dynamic_rnn_benchmark_dynamic(inputs_t, sequence_length)
    variables_lib.global_variables_initializer().run()
    delta_dynamic = _timer(sess, ops)

  print("%d \t %d \t %d \t %s \t %f \t %f \t %f" %
        (batch_size, max_time, num_units, use_gpu, delta_static, delta_dynamic,
         delta_dynamic / delta_static))

  return delta_static, delta_dynamic


def _half_seq_len_vs_unroll_half_rnn_benchmark(inputs_list_t, sequence_length):
  (_, input_size) = inputs_list_t[0].get_shape().as_list()
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = rnn_cell_impl.LSTMCell(
      num_units=input_size,
      use_peepholes=True,
      initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = rnn.static_rnn(
      cell,
      inputs_list_t,
      sequence_length=sequence_length,
      dtype=dtypes.float32)

  trainable_variables = ops_lib.get_collection(
      ops_lib.GraphKeys.TRAINABLE_VARIABLES)
  gradients = gradients_impl.gradients(outputs + [final_state],
                                       trainable_variables)

  return control_flow_ops.group(final_state, *(gradients + outputs))


def half_seq_len_vs_unroll_half_rnn_benchmark(batch_size, max_time, num_units,
                                              use_gpu):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = max_time * np.ones((batch_size,))
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)
  ]

  # Halve the sequence length, full static unroll
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      ops = _half_seq_len_vs_unroll_half_rnn_benchmark(inputs_list_t,
                                                       sequence_length / 2)
    variables_lib.global_variables_initializer().run()
    delta_half_seq_len = _timer(sess, ops)

  # Halve the unroll size, don't use sequence length
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      ops = _half_seq_len_vs_unroll_half_rnn_benchmark(
          inputs_list_t[:(max_time // 2)], sequence_length / 2)
    variables_lib.global_variables_initializer().run()
    delta_unroll_half = _timer(sess, ops)
  print("%d \t %d \t\t %d \t %s \t %f \t\t %f \t\t %f" %
        (batch_size, max_time, num_units, use_gpu, delta_half_seq_len,
         delta_unroll_half, delta_half_seq_len / delta_unroll_half))

  return delta_half_seq_len, delta_unroll_half


def _concat_state_vs_tuple_state_rnn_benchmark(inputs_list_t, sequence_length,
                                               state_is_tuple):
  (_, input_size) = inputs_list_t[0].get_shape().as_list()
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = rnn_cell_impl.LSTMCell(
      num_units=input_size,
      use_peepholes=True,
      initializer=initializer,
      state_is_tuple=state_is_tuple)
  outputs, final_state = rnn.static_rnn(
      cell,
      inputs_list_t,
      sequence_length=sequence_length,
      dtype=dtypes.float32)

  final_state = list(final_state) if state_is_tuple else [final_state]

  trainable_variables = ops_lib.get_collection(
      ops_lib.GraphKeys.TRAINABLE_VARIABLES)
  gradients = gradients_impl.gradients(outputs + final_state,
                                       trainable_variables)

  return control_flow_ops.group(*(final_state + gradients + outputs))


def concat_state_vs_tuple_state_rnn_benchmark(batch_size, max_time, num_units,
                                              use_gpu):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = max_time * np.ones((batch_size,))
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)
  ]

  # Run with concatenated states (default)
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      ops = _concat_state_vs_tuple_state_rnn_benchmark(
          inputs_list_t, sequence_length, state_is_tuple=False)
    variables_lib.global_variables_initializer().run()
    delta_concat_state = _timer(sess, ops)

  # Run with tuple states (new)
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      ops = _concat_state_vs_tuple_state_rnn_benchmark(
          inputs_list_t, sequence_length, state_is_tuple=True)
    variables_lib.global_variables_initializer().run()
    delta_tuple_state = _timer(sess, ops)
  print("%d \t %d \t %d \t %s \t %f \t\t %f \t\t %f" %
        (batch_size, max_time, num_units, use_gpu, delta_concat_state,
         delta_tuple_state, delta_concat_state / delta_tuple_state))

  return delta_concat_state, delta_tuple_state


def _dynamic_rnn_swap_memory_benchmark(inputs_t, sequence_length, swap_memory):
  (unused_0, unused_1, input_size) = inputs_t.get_shape().as_list()
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = rnn_cell_impl.LSTMCell(
      num_units=input_size,
      use_peepholes=True,
      initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = rnn.dynamic_rnn(
      cell,
      inputs_t,
      sequence_length=sequence_length,
      swap_memory=swap_memory,
      dtype=dtypes.float32)

  trainable_variables = ops_lib.get_collection(
      ops_lib.GraphKeys.TRAINABLE_VARIABLES)
  gradients = gradients_impl.gradients([outputs, final_state],
                                       trainable_variables)

  return control_flow_ops.group(final_state, outputs, *gradients)


def dynamic_rnn_swap_memory_benchmark(batch_size, max_time, num_units):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)
  ]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  # No memory swap
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    inputs_t = variables_lib.Variable(inputs, trainable=False).value()
    ops = _dynamic_rnn_swap_memory_benchmark(
        inputs_t, sequence_length, swap_memory=False)
    variables_lib.global_variables_initializer().run()
    no_swap = _timer(sess, ops)

  # Memory swap
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    inputs_t = variables_lib.Variable(inputs, trainable=False).value()
    ops = _dynamic_rnn_swap_memory_benchmark(
        inputs_t, sequence_length, swap_memory=True)
    variables_lib.global_variables_initializer().run()
    swap = _timer(sess, ops)

  print("%d \t %d \t %d \t %f \t %f \t %f" %
        (batch_size, max_time, num_units, no_swap, swap, swap / no_swap))
  return no_swap, swap


def rnn_long_sequence_benchmark(batch_size, seqlen, num_units, dynamic,
                                swap_memory, nn):
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = [seqlen for _ in range(batch_size)]
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(seqlen)
  ]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  for _ in range(nn):
    if dynamic:
      with session.Session(config=config, graph=ops_lib.Graph()) as sess:
        inputs_t = variables_lib.Variable(inputs, trainable=False).value()
        ops = _dynamic_rnn_swap_memory_benchmark(
            inputs_t, sequence_length, swap_memory=swap_memory)
        variables_lib.global_variables_initializer().run()
        elapsed = _timer(sess, ops)
    else:
      with session.Session(config=config, graph=ops_lib.Graph()) as sess:
        inputs_list_t = [
            variables_lib.Variable(
                x, trainable=False).value() for x in inputs_list
        ]
        ops = _static_vs_dynamic_rnn_benchmark_static(inputs_list_t,
                                                      sequence_length)
        variables_lib.global_variables_initializer().run()
        elapsed = _timer(sess, ops)

    print("%d \t %d \t %d \t %s \t %f \t %f" % (batch_size, seqlen, num_units,
                                                dynamic, elapsed,
                                                elapsed / seqlen))


class BenchmarkRNN(test.Benchmark):

  def benchmarkGraphCreationStaticVsDynamicLSTM(self):
    print("Graph Creation: Static Unroll vs. Dynamic Unroll LSTM")
    print("max_t \t dt(static) \t dt(dynamic) \t dt(dynamic)/dt(static)")
    for max_time in (1, 25, 50):
      s_dt, d_dt = graph_creation_static_vs_dynamic_rnn_benchmark(max_time)
      self.report_benchmark(
          name="graph_creation_time_static_T%02d" % max_time,
          iters=5,
          wall_time=s_dt)
      self.report_benchmark(
          name="graph_creation_time_dynamic_T%02d" % max_time,
          iters=5,
          wall_time=d_dt)

  def benchmarkStaticUnrollVsDynamicFlowLSTM(self):
    print("Calculation: Static Unroll with Dynamic Flow LSTM "
          "vs. Dynamic Unroll LSTM")
    print("batch \t max_t \t units \t gpu \t dt(static) \t dt(dynamic) "
          "\t dt(dynamic)/dt(static)")
    for batch_size in (256,):
      for max_time in (50,):
        for num_units in (512, 256, 128):
          for use_gpu in (False, True):
            s_dt, d_dt = static_vs_dynamic_rnn_benchmark(batch_size, max_time,
                                                         num_units, use_gpu)
            self.report_benchmark(
                name="static_unroll_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=s_dt)
            self.report_benchmark(
                name="dynamic_unroll_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=d_dt)

  def benchmarkDynamicLSTMNoMemorySwapVsMemorySwap(self):
    print("Calculation: Dynamic LSTM No Memory Swap vs. Memory Swap")
    print("batch \t max_t \t units \t no_swap \t swap \t swap/no_swap")
    for batch_size in (256, 512):
      for max_time in (100,):
        for num_units in (512, 256, 128):
          no_swap, swap = dynamic_rnn_swap_memory_benchmark(batch_size,
                                                            max_time, num_units)
          self.report_benchmark(
              name="dynamic_lstm_no_memory_swap_T%02d_B%03d_N%03d" %
              (max_time, batch_size, num_units),
              iters=20,
              wall_time=no_swap)
          self.report_benchmark(
              name="dynamic_lstm_with_memory_swap_T%02d_B%03d_N%03d" %
              (max_time, batch_size, num_units),
              iters=20,
              wall_time=swap)

  def benchmarkStaticUnrollHalfSequenceLengthVsHalfUnroll(self):
    print("Calculation: Static Unroll with Halved Sequence Length "
          "vs. Half Static Unroll")
    print("batch \t full_t \t units \t gpu \t dt(half_seq_len) "
          "\t dt(unroll_half) \t dt(half_seq_len)/dt(unroll_half)")
    for batch_size in (128,):
      for max_time in (50,):
        for num_units in (256,):
          for use_gpu in (False, True):
            s_dt, d_dt = half_seq_len_vs_unroll_half_rnn_benchmark(batch_size,
                                                                   max_time,
                                                                   num_units,
                                                                   use_gpu)
            self.report_benchmark(
                name="half_seq_len_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=s_dt)
            self.report_benchmark(
                name="unroll_half_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=d_dt)

  def benchmarkStaticUnrollStateConcatVsStateTuple(self):
    print("Calculation: Static Unroll with Concatenated State "
          "vs. Tuple State")
    print("batch \t time \t units \t gpu \t dt(concat_state) "
          "\t dt(tuple_state) \t dt(concat_state)/dt(tuple_state)")
    for batch_size in (
        16,
        128,):
      for max_time in (50,):
        for num_units in (
            16,
            128,):
          for use_gpu in (False, True):
            c_dt, t_dt = concat_state_vs_tuple_state_rnn_benchmark(batch_size,
                                                                   max_time,
                                                                   num_units,
                                                                   use_gpu)
            self.report_benchmark(
                name="concat_state_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=c_dt)
            self.report_benchmark(
                name="tuple_state_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=t_dt)

  def _benchmarkDynamicLSTMMemorySwapLongSeq(self):
    """The memory swapping test for the SOSP submission."""
    print("Calculation: Long LSTM Sequence")
    print("batch \t len \t units \t dynamic \t elapsed_t \t elapsed_t/len")
    batch_size = 512
    seqlen = 800
    num_units = 512
    dynamic = True
    swap_memory = True
    # Some warming up.
    if swap_memory:
      rnn_long_sequence_benchmark(batch_size, seqlen, num_units,
                                  dynamic, swap_memory, 2)
    # Measure the performance.
    for slen in xrange(100, 1100, 100):
      rnn_long_sequence_benchmark(batch_size, slen, num_units, dynamic,
                                  swap_memory, 3)

if __name__ == "__main__":
  test.main()
