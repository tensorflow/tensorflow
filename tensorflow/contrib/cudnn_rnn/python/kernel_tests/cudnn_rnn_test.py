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
"""Tests for Cudnn RNN models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import functools
import itertools
import os
import sys
import unittest

import numpy as np

from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.contrib.rnn.python.ops import rnn as contrib_rnn_lib
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradients_impl as gradients
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn as rnn_lib
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import adagrad
from tensorflow.python.training import adam
from tensorflow.python.training import checkpointable_utils
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.training import rmsprop
from tensorflow.python.training import saver as saver_lib


CUDNN_LSTM = cudnn_rnn_ops.CUDNN_LSTM
CUDNN_GRU = cudnn_rnn_ops.CUDNN_GRU
CUDNN_RNN_RELU = cudnn_rnn_ops.CUDNN_RNN_RELU
CUDNN_RNN_TANH = cudnn_rnn_ops.CUDNN_RNN_TANH
CUDNN_RNN_UNIDIRECTION = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
CUDNN_RNN_BIDIRECTION = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION

CUDNN_LSTM_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_LSTM_PARAMS_PER_LAYER
CUDNN_GRU_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_GRU_PARAMS_PER_LAYER
CUDNN_RNN_TANH_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_RNN_TANH_PARAMS_PER_LAYER
CUDNN_RNN_RELU_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_RNN_RELU_PARAMS_PER_LAYER


class CudnnTestModel(object):
  """Model with convenient APIs for easier building and running test graph.

  The graph built is used by all tests below to avoid repeatedly building
  similar test graphs.
  """

  def __init__(self,
               rnn_mode,
               num_layers,
               num_units,
               input_size,
               direction=CUDNN_RNN_UNIDIRECTION,
               dropout=0.,
               dtype=dtypes.float32,
               training=False,
               seed=None,
               kernel_initializer=None,
               bias_initializer=None):
    if dtype not in (dtypes.float16, dtypes.float32, dtypes.float64):
      raise ValueError("Invalid dtype: %s" % dtype)
    self._dtype = dtype

    self._inputs = array_ops.placeholder(
        dtype=dtype, shape=[None, None, input_size], name="inputs")
    h = array_ops.placeholder(
        dtype=dtype, shape=[None, None, num_units], name="h")
    c = array_ops.placeholder(
        dtype=dtype, shape=[None, None, num_units], name="c")
    if rnn_mode == CUDNN_LSTM:
      model_fn = cudnn_rnn.CudnnLSTM
      self._initial_state = (h, c)
    elif rnn_mode == CUDNN_GRU:
      model_fn = cudnn_rnn.CudnnGRU
      self._initial_state = (h,)
    elif rnn_mode == CUDNN_RNN_TANH:
      model_fn = cudnn_rnn.CudnnRNNTanh
      self._initial_state = (h,)
    elif rnn_mode == CUDNN_RNN_RELU:
      model_fn = cudnn_rnn.CudnnRNNRelu
      self._initial_state = (h,)
    else:
      raise ValueError("Invalid rnn_mode: %s" % rnn_mode)
    self._rnn = model_fn(
        num_layers,
        num_units,
        direction=direction,
        dropout=dropout,
        dtype=dtype,
        seed=seed,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer)
    self._rnn.build([None, None, input_size])

    self._outputs, self._output_state = self._rnn(
        self._inputs, initial_state=self._initial_state, training=training)

  def _AddUp(self, outputs, output_state):
    total = math_ops.reduce_sum(outputs)
    for s in output_state:
      total += math_ops.reduce_sum(s)
    return total

  @property
  def inputs(self):
    return self._inputs

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def outputs(self):
    return self._outputs

  @property
  def output_state(self):
    return self._output_state

  @property
  def rnn(self):
    return self._rnn

  @property
  def total_sum(self):
    return self._AddUp(self.outputs, self.output_state)

  def SynthesizeInput(self, seq_length, batch_size, seed=1234):
    """Synthesizes input and initial state values for testing."""
    np.random.seed(seed)
    num_layers = self._rnn.num_layers
    dir_count = self._rnn.num_dirs
    num_units = self._rnn.num_units
    input_size = self._rnn.input_size

    np_dtype = np.float32 if self._dtype == dtypes.float32 else np.float64
    inputs = np.random.randn(seq_length, batch_size,
                             input_size).astype(np_dtype)
    input_h = np.random.randn(num_layers * dir_count, batch_size,
                              num_units).astype(np_dtype)
    if self._rnn.rnn_mode == CUDNN_LSTM:
      input_c = np.random.randn(num_layers * dir_count, batch_size,
                                num_units).astype(np_dtype)
      initial_state = (input_h, input_c)
    else:
      initial_state = (input_h,)
    return inputs, initial_state

  def ZeroState(self, batch_size):
    num_layers = self._rnn.num_layers
    dir_count = self._rnn.num_dirs
    num_units = self._rnn.num_units

    np_dtype = np.float32 if self._dtype == dtypes.float32 else np.float64
    input_h = np.zeros((num_layers * dir_count, batch_size,
                        num_units)).astype(np_dtype)
    if self._rnn.rnn_mode == CUDNN_LSTM:
      input_c = np.zeros((num_layers * dir_count, batch_size,
                          num_units)).astype(np_dtype)
      initial_state = (input_h, input_c)
    else:
      initial_state = (input_h,)
    return initial_state

  def FProp(self, inputs_t, initial_state_t, training):
    """Builds additional subgraph with given inputs and state.

    Args:
      inputs_t: a tensor.
      initial_state_t: a tensor.
      training: boolean, true if training mode.
    Returns:
      A tensor of the forward pass output of the model.
    """
    outputs, output_state = self._rnn(
        inputs_t, initial_state=initial_state_t, training=training)
    return self._AddUp(outputs, output_state)

  def Feed(self, sess, inputs, initial_state=None, return_sum=True):
    """Runs graph with given inputs and initial state."""
    batch_size = inputs.shape[1]
    if initial_state is None:
      initial_state = self.ZeroState(batch_size)
    if return_sum:
      return sess.run(
          self.total_sum,
          feed_dict={self.inputs: inputs,
                     self.initial_state: initial_state})
    else:
      return sess.run(
          [self.outputs, self.output_state],
          feed_dict={self.inputs: inputs,
                     self.initial_state: initial_state})


def _CreateCudnnCompatibleCanonicalRNN(rnn, inputs, is_bidi=False, scope=None):
  mode = rnn.rnn_mode
  num_units = rnn.num_units
  num_layers = rnn.num_layers

  # To reuse cuDNN-trained models, must use cudnn compatible rnn cells.
  if mode == CUDNN_LSTM:
    single_cell = lambda: cudnn_rnn_ops.CudnnCompatibleLSTMCell(num_units)
  elif mode == CUDNN_GRU:
    single_cell = lambda: cudnn_rnn_ops.CudnnCompatibleGRUCell(num_units)
  elif mode == CUDNN_RNN_TANH:
    single_cell = (lambda: rnn_cell_impl.BasicRNNCell(num_units, math_ops.tanh))
  elif mode == CUDNN_RNN_RELU:
    single_cell = (
        lambda: rnn_cell_impl.BasicRNNCell(num_units, gen_nn_ops.relu))
  else:
    raise ValueError("%s is not supported!" % mode)

  if not is_bidi:
    cell = rnn_cell_impl.MultiRNNCell(
        [single_cell() for _ in range(num_layers)])
    return rnn_lib.dynamic_rnn(
        cell, inputs, dtype=dtypes.float32, time_major=True, scope=scope)
  else:
    cells_fw = [single_cell() for _ in range(num_layers)]
    cells_bw = [single_cell() for _ in range(num_layers)]

    (outputs, output_state_fw,
     output_state_bw) = contrib_rnn_lib.stack_bidirectional_dynamic_rnn(
         cells_fw,
         cells_bw,
         inputs,
         dtype=dtypes.float32,
         time_major=True,
         scope=scope)
    return outputs, (output_state_fw, output_state_bw)


class CudnnRNNTestBasic(test_util.TensorFlowTestCase):

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testLayerBasic(self):
    num_layers = 4
    num_units = 2
    batch_size = 8
    direction = CUDNN_RNN_UNIDIRECTION
    dir_count = 1

    with vs.variable_scope("main"):
      kernel_initializer = init_ops.constant_initializer(0.)
      bias_initializer = init_ops.constant_initializer(0.)
      inputs = random_ops.random_uniform([
          num_layers * dir_count, batch_size, num_units], dtype=dtypes.float32)

      lstm = cudnn_rnn.CudnnLSTM(num_layers, num_units,
                                 direction=direction,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name="awesome_lstm")

      # Build the layer
      outputs1, _ = lstm(inputs)
      # Reuse the layer
      outputs2, _ = lstm(inputs)

      total_sum1 = math_ops.reduce_sum(outputs1)
      total_sum2 = math_ops.reduce_sum(outputs2)

    with vs.variable_scope("main", reuse=True):
      lstm = cudnn_rnn.CudnnLSTM(num_layers, num_units,
                                 direction=direction,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name="awesome_lstm")

      # Reuse the layer
      outputs3, _ = lstm(inputs)
      total_sum3 = math_ops.reduce_sum(outputs3)

    self.assertEqual(1, len(variables.trainable_variables()))
    self.assertEqual(1, len(ops.get_collection(ops.GraphKeys.SAVEABLE_OBJECTS)))
    self.assertEqual("main/awesome_lstm/opaque_kernel",
                     variables.trainable_variables()[0].op.name)

    with self.test_session(use_gpu=True) as sess:
      sess.run(variables.global_variables_initializer())
      (total_sum1_v, total_sum2_v, total_sum3_v) = sess.run(
          [total_sum1, total_sum2, total_sum3])
      self.assertEqual(0, total_sum1_v)
      self.assertEqual(0, total_sum2_v)
      self.assertEqual(0, total_sum3_v)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testOptimizersSupport(self):
    for opt in ("adagrad", "adam", "rmsprop", "momentum", "sgd"):
      self._TestOptimizerSupportHelper(opt)

  def _GetOptimizer(self, opt):
    if opt == "adagrad":
      return adagrad.AdagradOptimizer(learning_rate=1e-2)
    elif opt == "adam":
      return adam.AdamOptimizer(learning_rate=1e-2)
    elif opt == "rmsprop":
      return rmsprop.RMSPropOptimizer(learning_rate=1e-2)
    elif opt == "momentum":
      return momentum.MomentumOptimizer(learning_rate=1e-2, momentum=0.9)
    elif opt == "sgd":
      return gradient_descent.GradientDescentOptimizer(learning_rate=1e-2)
    else:
      raise ValueError("Unsupported optimizer: %s" % opt)

  def _TestOptimizerSupportHelper(self, opt):
    num_layers = 4
    num_units = 2
    batch_size = 8
    direction = CUDNN_RNN_UNIDIRECTION
    dir_count = 1

    with ops.Graph().as_default() as g:
      kernel_initializer = init_ops.constant_initializer(0.)
      bias_initializer = init_ops.constant_initializer(0.)
      inputs = random_ops.random_uniform([
          num_layers * dir_count, batch_size, num_units], dtype=dtypes.float32)

      lstm = cudnn_rnn.CudnnLSTM(num_layers, num_units,
                                 direction=direction,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name="awesome_lstm")
      outputs, _ = lstm(inputs)
      loss = math_ops.reduce_sum(outputs)
      optimizer = self._GetOptimizer(opt)
      train_op = optimizer.minimize(loss)

    with self.test_session(use_gpu=True, graph=g) as sess:
      sess.run(variables.global_variables_initializer())
      sess.run(train_op)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSaveableGraphDeviceAssignment(self):
    num_layers = 4
    num_units = 2
    batch_size = 8
    direction = CUDNN_RNN_UNIDIRECTION
    dir_count = 1

    def DeviceFn(op):
      if op.type in ("Variable", "VariableV2"):
        return "/cpu:0"
      else:
        return "/gpu:0"

    with ops.Graph().as_default() as g:
      with ops.device(DeviceFn):
        with vs.variable_scope("main"):
          kernel_initializer = init_ops.constant_initializer(3.14)
          bias_initializer = init_ops.constant_initializer(1.59)
          inputs = random_ops.random_uniform(
              [num_layers * dir_count, batch_size, num_units],
              dtype=dtypes.float32)

          lstm = cudnn_rnn.CudnnLSTM(num_layers, num_units,
                                     direction=direction,
                                     kernel_initializer=kernel_initializer,
                                     bias_initializer=bias_initializer,
                                     name="awesome_lstm")
          outputs = lstm(inputs)

        # saver is created in the scope of DeviceFn.
        saver = saver_lib.Saver()

    with self.test_session(use_gpu=True, graph=g) as sess:
      save_path = os.path.join(self.get_temp_dir(),
                               "test-saveable-device-assignment")
      sess.run(variables.global_variables_initializer())

      saver.save(sess, save_path)
      saver.restore(sess, save_path)
      sess.run(outputs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testDifferentShapesEager(self):
    # Checks that kernel caching does not cause sharing of temporary storage
    # across different input shapes when executing eagerly.
    with context.eager_mode():
      with ops.device("gpu:0"):
        first_output, _ = cudnn_rnn.CudnnGRU(1, 100)(
            array_ops.zeros([28, 100, 28]))
        second_output, _ = cudnn_rnn.CudnnGRU(1, 100)(
            array_ops.zeros([28, 100, 100]))
        self.assertAllEqual([28, 100, 100], first_output.shape)
        self.assertAllEqual([28, 100, 100], second_output.shape)

        def _LossFunc():
          first_output, _ = cudnn_rnn.CudnnGRU(1, 100)(
              array_ops.zeros([28, 100, 28]))
          second_output, _ = cudnn_rnn.CudnnGRU(1, 100)(
              array_ops.zeros([28, 100, 100]))
          return (math_ops.reduce_sum(first_output) +
                  math_ops.reduce_sum(second_output))

        backprop.implicit_grad(_LossFunc)()

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testDifferentShapesGraph(self):
    # Tests that a single kernel instance presented with multiple input shapes
    # does not crash with graph execution.
    with ops.device("gpu:0"):
      layer = cudnn_rnn.CudnnGRU(1, 100)
      layer(array_ops.zeros([28, 100, 100]))

      def _Cond(index, accumulation):
        del accumulation  # unused
        return math_ops.less(index, 4)

      def _Body(index, accumulation):
        layer_input = accumulation[:, :, 10 * (1 + index % 2):]
        output, _ = layer(layer_input)
        return index + 1, accumulation + output

      original_input = array_ops.zeros([28, 100, 100])
      _, accumulation = control_flow_ops.while_loop(_Cond, _Body,
                                                    [0, original_input])
      grad, = gradients.gradients(
          math_ops.reduce_sum(accumulation), (original_input,))
    init_op = variables.global_variables_initializer()
    with self.test_session() as sess:
      sess.run(init_op)
      accumulation_eval, grad_eval = sess.run((accumulation, grad))
      self.assertAllEqual([28, 100, 100], accumulation_eval.shape)
      self.assertAllEqual([28, 100, 100], grad_eval.shape)


# TODO(jamesqin): Transform to parameterized test after it is included in the
# TF open source codebase.
class CudnnRNNTestSaveRestore(test_util.TensorFlowTestCase):

  def _CompareWeights(self, lhs, rhs):
    self.assertEqual(len(lhs), len(rhs))
    for lw, rw in zip(lhs, rhs):
      self.assertAllEqual(lw, rw)

  def _CompareBiases(self, lhs, rhs, rnn_mode, num_layers, direction):
    self.assertEqual(len(lhs), len(rhs))
    if rnn_mode == CUDNN_LSTM:
      num_params_per_layer = CUDNN_LSTM_PARAMS_PER_LAYER
    elif rnn_mode == CUDNN_GRU:
      num_params_per_layer = CUDNN_GRU_PARAMS_PER_LAYER
    elif rnn_mode == CUDNN_RNN_TANH:
      num_params_per_layer = CUDNN_RNN_TANH_PARAMS_PER_LAYER
    else:
      num_params_per_layer = CUDNN_RNN_RELU_PARAMS_PER_LAYER
    num_dirs = 1 if direction == CUDNN_RNN_UNIDIRECTION else 2
    num_params_per_layer *= num_dirs
    self.assertEqual(num_params_per_layer * num_layers, len(lhs))

    for i in range(num_layers):
      layer_lhs = lhs[i * num_params_per_layer: (i+1) * num_params_per_layer]
      layer_rhs = rhs[i * num_params_per_layer: (i+1) * num_params_per_layer]
      if direction == CUDNN_RNN_UNIDIRECTION:
        self._CompareSingleLayerBiases(layer_lhs, layer_rhs)
      else:
        size = len(layer_lhs)
        fw_lhs, bw_lhs = layer_lhs[:size//2], layer_lhs[size//2:]
        fw_rhs, bw_rhs = layer_rhs[:size//2], layer_rhs[size//2:]
        self._CompareSingleLayerBiases(fw_lhs, fw_rhs)
        self._CompareSingleLayerBiases(bw_lhs, bw_rhs)

  def _CompareSingleLayerBiases(self, lhs, rhs):
    self.assertEqual(len(lhs), len(rhs))

    lf_lhs, rt_lhs = lhs[:len(lhs)//2], lhs[len(lhs)//2:]
    lf_rhs, rt_rhs = rhs[:len(rhs)//2], rhs[len(rhs)//2:]
    self.assertEqual(len(lf_lhs), len(rt_lhs))
    self.assertEqual(len(lf_rhs), len(rt_rhs))

    sum_lhs, sum_rhs = [], []
    for lf, rt in zip(lf_lhs, rt_lhs):
      sum_lhs.append(lf + rt)
    for lf, rt in zip(lf_rhs, rt_rhs):
      sum_rhs.append(lf + rt)
    self.assertEqual(len(sum_lhs), len(sum_rhs))
    for lf, rt in zip(sum_lhs, sum_rhs):
      self.assertAllEqual(lf, rt)

  def _TestSaveRestoreVariable(self, rnn_mode, direction, dtype):
    input_size = 3
    num_layers = 2
    num_units = 7
    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(1234)
      model = CudnnTestModel(
          rnn_mode,
          num_layers,
          num_units,
          input_size,
          direction=direction,
          dtype=dtype)
      rnn = model.rnn
      save_path = os.path.join(self.get_temp_dir(),
                               "save-restore-variable-test")
      saver = saver_lib.Saver()
      weights, biases = model.rnn.saveable._OpaqueParamsToCanonical()
      opaque_params = rnn.trainable_variables[0]
      # CudnnTestModel() creates CudnnOpaqueParamsSaveable that helps saver save
      # Cudnn vars in canonical format.
      reset_op = state_ops.assign(
          opaque_params,
          array_ops.zeros(array_ops.shape(opaque_params), dtype=dtype))
      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(use_gpu=True, graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        val = saver.save(sess, save_path)
        self.assertEqual(save_path, val)
        weights_v, biases_v = sess.run([weights, biases])

        # Reset opaque param
        sess.run(reset_op)
        saver.restore(sess, save_path)
        weights_v_restored, biases_v_restored = sess.run([weights, biases])

        self._CompareWeights(weights_v, weights_v_restored)
        self._CompareBiases(biases_v, biases_v_restored, rnn_mode, num_layers,
                            direction)

  def _TestSaveRestoreTwoVariables(self, rnn_mode, direction, dtype):
    input_size = 3
    num_layers = 2
    num_units = 7
    with ops.Graph().as_default() as g:
      random_seed.set_random_seed(1234)
      with vs.variable_scope("m1"):
        model1 = CudnnTestModel(
            rnn_mode,
            num_layers,
            num_units,
            input_size,
            direction=direction,
            dtype=dtype)
      with vs.variable_scope("m2"):
        model2 = CudnnTestModel(
            rnn_mode,
            num_layers,
            num_units,
            input_size,
            direction=direction,
            dtype=dtype)
      opaque_params = (model1.rnn.trainable_variables[0],
                       model2.rnn.trainable_variables[0])
      weights1, biases1 = model1.rnn.saveable._OpaqueParamsToCanonical()
      weights2, biases2 = model2.rnn.saveable._OpaqueParamsToCanonical()
      reset_params = [
          state_ops.assign(params,
                           array_ops.zeros_like(params, dtype=dtype))
          for params in opaque_params
      ]
      reset_op = control_flow_ops.group(*reset_params)
      save_path = os.path.join(self.get_temp_dir(),
                               "save-restore-variable-test2")
      saver = saver_lib.Saver()
      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(use_gpu=True, graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        val = saver.save(sess, save_path)
        self.assertEqual(save_path, val)

        weights1_v, biases1_v = sess.run([weights1, biases1])
        weights2_v, biases2_v = sess.run([weights2, biases2])

        sess.run(reset_op)
        saver.restore(sess, save_path)
        weights1_v_restored, biases1_v_restored = sess.run([weights1, biases1])
        weights2_v_restored, biases2_v_restored = sess.run([weights2, biases2])

        self._CompareWeights(weights1_v, weights1_v_restored)
        self._CompareWeights(weights2_v, weights2_v_restored)
        self._CompareBiases(biases1_v, biases1_v_restored, rnn_mode, num_layers,
                            direction)
        self._CompareBiases(biases2_v, biases2_v_restored, rnn_mode, num_layers,
                            direction)

  def _TestSaveRestoreOutput(self, rnn_mode, direction, dtype):
    with ops.Graph().as_default() as g:
      num_layers = 2
      num_units = 7
      input_size = 7
      seq_length = 8
      batch_size = 4
      model = CudnnTestModel(
          rnn_mode,
          num_layers,
          num_units,
          input_size,
          direction=direction,
          dtype=dtype,
          training=False)
      rnn = model.rnn

      save_path = os.path.join(self.get_temp_dir(), "save-restore-output-test")
      saver = saver_lib.Saver()

      # Only one opaque var in a cudnn layer.
      assert len(rnn.trainable_variables) == 1
      reset_params = state_ops.assign(
          rnn.trainable_variables[0],
          array_ops.zeros(
              array_ops.shape(rnn.trainable_variables[0]), dtype=dtype))

      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(use_gpu=True, graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        inputs, initial_state = model.SynthesizeInput(seq_length, batch_size)
        total_sum_v = model.Feed(sess, inputs, initial_state)
        val = saver.save(sess, save_path)
        self.assertEqual(save_path, val)

        sess.run(reset_params)
        saver.restore(sess, save_path)
        total_sum_v_restored = model.Feed(sess, inputs, initial_state)
        self.assertAllClose(total_sum_v, total_sum_v_restored, atol=1e-5)

  def _TestSaveRestoreHelper(self, rnn_mode):
    directions = [CUDNN_RNN_UNIDIRECTION, CUDNN_RNN_BIDIRECTION]
    dtype_list = [dtypes.float16, dtypes.float32, dtypes.float64]
    for direction, dtype in itertools.product(directions, dtype_list):
      self._TestSaveRestoreVariable(rnn_mode, direction, dtype)
      self._TestSaveRestoreTwoVariables(rnn_mode, direction, dtype)
      self._TestSaveRestoreOutput(rnn_mode, direction, dtype)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSaveRestoreRepeatedlyCreateCustomSaveable(self):
    input_size = 3
    num_layers = 2
    num_units = 7
    with ops.Graph().as_default():
      random_seed.set_random_seed(1234)
      model = CudnnTestModel(
          CUDNN_LSTM,
          num_layers,
          num_units,
          input_size,
          direction=CUDNN_RNN_UNIDIRECTION,
          dtype=dtypes.float32)
      with self.assertRaisesRegexp(RuntimeError,
                                   "Cudnn saveable already created"):
        model.rnn._create_saveable()

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSaveRestoreLSTM(self):
    self._TestSaveRestoreHelper(CUDNN_LSTM)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSaveRestoreGRU(self):
    self._TestSaveRestoreHelper(CUDNN_GRU)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSaveRestoreRNNTanh(self):
    self._TestSaveRestoreHelper(CUDNN_RNN_TANH)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSaveRestoreRNNRelu(self):
    self._TestSaveRestoreHelper(CUDNN_RNN_RELU)


class CudnnRNNTestSaveRestoreCheckpointable(test_util.TensorFlowTestCase):

  def _VerifyCheckpoint(
      self, checkpoint_path, compatible_cell_fn, cudnn_cell_fn,
      num_layers, input_size, expected_variable_values, num_applications=3):
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    with ops.device("gpu:0"):
      cudnn_layer = cudnn_cell_fn()
      cudnn_checkpoint = checkpointable_utils.Checkpoint(cell=cudnn_layer)
      status = cudnn_checkpoint.restore(checkpoint_path)
      inputs = 3. * array_ops.ones([num_applications, num_layers, input_size],
                                   dtype=dtypes.float32)
      cudnn_output, _ = cudnn_layer(inputs)
      status.assert_consumed().run_restore_ops()
    second_save_path = cudnn_checkpoint.save(checkpoint_prefix)
    restore_layer = compatible_cell_fn()
    restore_layer_checkpoint = checkpointable_utils.Checkpoint(
        cell=restore_layer)
    status = restore_layer_checkpoint.restore(second_save_path)
    current_state = restore_layer.zero_state(1, dtypes.float32)
    for _ in range(num_applications):
      restore_layer_output, current_state = restore_layer(
          inputs=3. * array_ops.ones([1, input_size]),
          state=current_state)
    status.assert_consumed().run_restore_ops()
    self.assertTrue(restore_layer.variables)
    for variable, expected_value in zip(
        restore_layer.variables, expected_variable_values):
      self.assertAllClose(expected_value, self.evaluate(variable))
    self.assertAllClose(self.evaluate(restore_layer_output),
                        self.evaluate(cudnn_output)[-1, -1:, ...])

  def _CheckpointableSingleCellUnidirectionalTestTemplate(
      self, single_cell_fn, cudnn_cell_fn):
    # Single-layer cuDNN cells with object-based checkpointing should be
    # checkpoint compatible with either single CudnnCompatible cells or
    # MultiRnnCells with one cell.
    input_size = 3
    save_cell_layer = single_cell_fn()
    save_cell_layer(
        inputs=array_ops.ones([1, input_size]),
        state=save_cell_layer.zero_state(1, dtypes.float32))
    self.assertTrue(save_cell_layer.variables)
    expected_values = []
    np.random.seed(10)
    for variable in save_cell_layer.variables:
      value = np.random.normal(size=variable.shape)
      expected_values.append(value)
      self.evaluate(variable.assign(value))
    save_checkpoint = checkpointable_utils.Checkpoint(cell=save_cell_layer)
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    first_save_path = save_checkpoint.save(checkpoint_prefix)
    self._VerifyCheckpoint(
        checkpoint_path=first_save_path,
        compatible_cell_fn=
        lambda: rnn_cell_impl.MultiRNNCell([single_cell_fn()]),
        cudnn_cell_fn=cudnn_cell_fn,
        num_layers=1,
        expected_variable_values=expected_values,
        input_size=input_size)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  @test_util.run_in_graph_and_eager_modes()
  def testLSTMCheckpointableSingleLayer(self):
    num_units = 2
    direction = CUDNN_RNN_UNIDIRECTION
    self._CheckpointableSingleCellUnidirectionalTestTemplate(
        single_cell_fn=functools.partial(
            cudnn_rnn_ops.CudnnCompatibleLSTMCell, num_units=num_units),
        cudnn_cell_fn=functools.partial(
            cudnn_rnn.CudnnLSTM, num_layers=1, num_units=num_units,
            direction=direction, name="awesome_lstm"))

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  @test_util.run_in_graph_and_eager_modes()
  def testGRUCheckpointableSingleLayer(self):
    num_units = 2
    direction = CUDNN_RNN_UNIDIRECTION
    with self.assertRaises(NotImplementedError):
      # TODO(allenl): Implement object-based saving for GRUs and other cells.
      self._CheckpointableSingleCellUnidirectionalTestTemplate(
          single_cell_fn=functools.partial(
              cudnn_rnn_ops.CudnnCompatibleGRUCell, num_units=num_units),
          cudnn_cell_fn=functools.partial(
              cudnn_rnn.CudnnGRU, num_layers=1, num_units=num_units,
              direction=direction, name="awesome_gru"))

  def _CheckpointableMultiLayerTestTemplate(
      self, single_cell_fn, cudnn_cell_fn, num_layers):

    def _MultiCellFn():
      return rnn_cell_impl.MultiRNNCell(
          [single_cell_fn() for _ in range(num_layers)])
    input_size = 3
    save_graph = ops.Graph()
    with save_graph.as_default(), self.test_session(graph=save_graph):
      save_layer = _MultiCellFn()
      save_layer(inputs=array_ops.ones([1, input_size]),
                 state=save_layer.zero_state(1, dtypes.float32))
      self.assertTrue(save_layer.variables)
      expected_values = []
      np.random.seed(10)
      for variable in save_layer.variables:
        value = np.random.normal(size=variable.shape)
        expected_values.append(value)
        self.evaluate(variable.assign(value))
      save_checkpoint = checkpointable_utils.Checkpoint(cell=save_layer)
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
      first_save_path = save_checkpoint.save(checkpoint_prefix)
    self._VerifyCheckpoint(
        checkpoint_path=first_save_path,
        compatible_cell_fn=_MultiCellFn, cudnn_cell_fn=cudnn_cell_fn,
        num_layers=num_layers,
        expected_variable_values=expected_values,
        input_size=input_size)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  @test_util.run_in_graph_and_eager_modes()
  def testCudnnCompatibleLSTMCheckpointablMultiLayer(self):
    num_units = 2
    num_layers = 3
    direction = CUDNN_RNN_UNIDIRECTION
    self._CheckpointableMultiLayerTestTemplate(
        single_cell_fn=functools.partial(
            cudnn_rnn_ops.CudnnCompatibleLSTMCell, num_units=num_units),
        cudnn_cell_fn=functools.partial(
            cudnn_rnn.CudnnLSTM, num_layers=num_layers, num_units=num_units,
            direction=direction, name="awesome_lstm"),
        num_layers=num_layers)


# TODO(jamesqin): Transform to parameterized test after it is included in the
# TF open source codebase.
class CudnnRNNTestCompatibleRNNCells(test_util.TensorFlowTestCase):

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testCudnnCompatibleLSTM(self):
    self._TestCudnnCompatibleRnnCellsHelper(CUDNN_LSTM)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testCudnnCompatibleGRU(self):
    self._TestCudnnCompatibleRnnCellsHelper(CUDNN_GRU)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testCudnnCompatibleRNNTanh(self):
    self._TestCudnnCompatibleRnnCellsHelper(CUDNN_RNN_TANH)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testCudnnCompatibleRNNRelu(self):
    self._TestCudnnCompatibleRnnCellsHelper(CUDNN_RNN_RELU)

  def _TestCudnnCompatibleRnnCellsHelper(self, rnn_mode):
    configs = [
        {
            "num_layers": 1,
            "seq_length": 3,
            "num_units": 4,
            "input_size": 5,
            "batch_size": 6,
        },
        {
            "num_layers": 2,
            "seq_length": 8,
            "num_units": 4,
            "input_size": 8,
            "batch_size": 16,
        },
        {
            "num_layers": 2,
            "seq_length": 3,
            "num_units": 4,
            "input_size": 5,
            "batch_size": 6,
        },
        {
            "num_layers": 1,
            "seq_length": 2,
            "num_units": 2,
            "input_size": 4,
            "batch_size": 1,
        },
    ]
    directions = [CUDNN_RNN_UNIDIRECTION, CUDNN_RNN_BIDIRECTION]
    for cfg, direction in zip(configs, directions):
      self._TestCudnnCompatibleRnnCells(cfg["num_layers"], cfg["seq_length"],
                                        cfg["num_units"], cfg["input_size"],
                                        cfg["batch_size"], rnn_mode, direction)

  def _TestCudnnCompatibleRnnCells(self, num_layers, seq_length, num_units,
                                   input_size, batch_size, rnn_mode, direction):
    dtype = dtypes.float32
    # Train graph
    with ops.Graph().as_default() as g:
      model = CudnnTestModel(
          rnn_mode,
          num_layers,
          num_units,
          input_size,
          direction=direction,
          dtype=dtype,
          training=True)
      target_output = array_ops.placeholder(dtype=dtype)
      loss_op = losses.log_loss(
          labels=target_output, predictions=model.total_sum)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1e-2)
      train_op = optimizer.minimize(loss_op)

      saver = saver_lib.Saver()

      # Train Cudnn model
      seed = 0
      with self.test_session(use_gpu=True, graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        # Train 128 steps
        num_steps = 128
        for _ in range(num_steps):
          inputs, _ = model.SynthesizeInput(seq_length, batch_size, seed)
          targets = np.random.rand()
          sess.run(
              train_op,
              feed_dict={
                  model.inputs: inputs,
                  model.initial_state: model.ZeroState(batch_size),
                  target_output: targets
              })
          seed += 1

        save_path = os.path.join(self.get_temp_dir(),
                                 ("cudnn-rnn-%s-test" % rnn_mode))
        save_v = saver.save(sess, save_path)
        self.assertEqual(save_path, save_v)

    # Cudnn inference graph
    with ops.Graph().as_default() as g:
      model = CudnnTestModel(
          rnn_mode,
          num_layers,
          num_units,
          input_size,
          direction=direction,
          dtype=dtype,
          training=False)
      rnn = model.rnn
      saver = saver_lib.Saver()

      inference_input = np.random.rand(seq_length, batch_size,
                                       input_size).astype(np.float32)
      with self.test_session(use_gpu=True, graph=g) as sess:
        sess.run(variables.global_variables_initializer())
        saver.restore(sess, save_path)

        # Cudnn inference
        cudnn_outputs_v, cudnn_output_states_v = model.Feed(
            sess, inference_input, return_sum=False)

    # Canonical RNN inference graph
    with ops.Graph().as_default() as g:
      cell_inputs = array_ops.placeholder(
          dtype, shape=[seq_length, batch_size, input_size])
      if direction == CUDNN_RNN_UNIDIRECTION:
        # outputs is one tensor, states are num_layer tuples, each 2 tensors
        (outputs, states) = _CreateCudnnCompatibleCanonicalRNN(rnn, cell_inputs)
        if rnn_mode == CUDNN_LSTM:
          output_h = array_ops.stack([s.h for s in states])
          output_c = array_ops.stack([s.c for s in states])
        else:
          output_state = array_ops.stack([s for s in states])
      else:
        # outputs is one tensor.
        # states is a tuple of 2 tuples:
        # each sub tuple is num_layer tuples, each with 2 tensors.
        (outputs, states) = _CreateCudnnCompatibleCanonicalRNN(
            rnn, cell_inputs, is_bidi=True)
        output_state_fw, output_state_bw = states
        if rnn_mode == CUDNN_LSTM:
          output_h, output_c = [], []
          for s_fw, s_bw in zip(output_state_fw, output_state_bw):
            output_h.append(array_ops.stack([s_fw.h, s_bw.h]))
            output_c.append(array_ops.stack([s_fw.c, s_bw.c]))
          output_h = array_ops.concat(output_h, axis=0)
          output_c = array_ops.concat(output_c, axis=0)
        else:
          output_state = []
          for s_fw, s_bw in zip(output_state_fw, output_state_bw):
            output_state.append(array_ops.stack([s_fw, s_bw]))
          output_state = array_ops.concat(output_state, axis=0)
      saver = saver_lib.Saver()

      with self.test_session(use_gpu=True, graph=g) as sess:
        saver.restore(sess, save_path)

        # BlockCell inference
        if rnn_mode == CUDNN_LSTM:
          outputs_v, output_h_v, output_c_v = sess.run(
              [outputs, output_h, output_c],
              feed_dict={cell_inputs: inference_input})
          self.assertAllClose(cudnn_outputs_v, outputs_v)
          cudnn_output_h_v, cudnn_output_c_v = cudnn_output_states_v
          self.assertAllClose(cudnn_output_h_v, output_h_v)
          self.assertAllClose(cudnn_output_c_v, output_c_v)
        else:
          outputs_v, output_state_v = sess.run(
              [outputs, output_state],
              feed_dict={cell_inputs: inference_input})
          self.assertAllClose(cudnn_outputs_v, outputs_v, atol=2e-5, rtol=2e-5)
          (cudnn_output_h_v,) = cudnn_output_states_v
          self.assertAllClose(cudnn_output_h_v, output_state_v, atol=2e-5,
                              rtol=2e-5)


class CudnnRNNTestParamsSize(test_util.TensorFlowTestCase):

  def _TestOpaqueParamsSize(self, rnn_mode, num_layers, num_units, input_size,
                            dtype, direction):
    logging.info("Testing one lstm param size with config: %s", locals())
    model = CudnnTestModel(
        rnn_mode,
        num_layers,
        num_units,
        input_size,
        dtype=dtype,
        direction=direction)
    rnn = model.rnn

    # Min param size estimate = sum(weights.size) + sum(biases.size)
    min_params_size = (
        np.sum(map(np.prod, rnn.canonical_weight_shapes)) +
        np.sum([sp[0] for sp in rnn.canonical_bias_shapes]))

    opaque_params = rnn.trainable_variables[0]
    with self.test_session(use_gpu=True, graph=ops.get_default_graph()):
      variables.global_variables_initializer().run()
      opaque_params_size_v = opaque_params.eval().size
      self.assertLessEqual(min_params_size, opaque_params_size_v)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testOpaqueParamsSize(self):
    test_configs = [
        [4, 200, 200],
        [4, 200, 300],
        [4, 200, 100],
        [1, 100, 200],
        [2, 200, 100],
        [3, 200, 400],
    ]
    directions = [CUDNN_RNN_UNIDIRECTION, CUDNN_RNN_BIDIRECTION]
    dtype_list = [dtypes.float16, dtypes.float32, dtypes.float64]
    rnns = [CUDNN_LSTM, CUDNN_GRU, CUDNN_RNN_RELU, CUDNN_RNN_TANH]
    for (rnn, config, dtype, direction) in itertools.product(
        rnns, test_configs, dtype_list, directions):
      num_layers, num_units, input_size = config
      with ops.Graph().as_default():
        self._TestOpaqueParamsSize(rnn, num_layers, num_units, input_size,
                                   dtype, direction)


class CudnnRNNTestTraining(test_util.TensorFlowTestCase):

  def setUp(self):
    super(CudnnRNNTestTraining, self).setUp()
    self._reset_rnd_gen_state = os.environ.get("TF_CUDNN_RESET_RND_GEN_STATE",
                                               str(False))
    self._rnn_use_v2 = os.environ.get("TF_CUDNN_RNN_USE_V2", "0")

  def tearDown(self):
    super(CudnnRNNTestTraining, self).tearDown()
    os.environ["TF_CUDNN_RESET_RND_GEN_STATE"] = self._reset_rnd_gen_state
    os.environ["TF_CUDNN_RNN_USE_V2"] = self._rnn_use_v2

  def _ComputeNumericGrad(self, sess, y, x, delta=1e-4, step=1):
    """Compute the numeric gradient of y wrt to x.

    Args:
      sess: The TF session constructed with a graph containing x and y.
      y: A scalar TF Tensor in the graph constructed in sess.
      x: A TF Tensor in the graph constructed in sess.
      delta: Gradient checker's small perturbation of x[i].
      step: Only compute numerical gradients for a subset of x values.
        I.e. dy/dx[i] is computed if i % step == 0.
    Returns:
      A Tensor of the same shape and dtype as x. If x[i] is not chosen
      to compute the numerical gradient dy/x[i], the corresponding
      value is set to 0.
    """

    x_data = sess.run(x)
    x_size = x_data.size
    x_shape = x_data.shape

    numeric_grad = np.zeros(x_size, dtype=x_data.dtype)

    for i in range(0, x_size, step):
      x_pos = x_data.copy()
      if x_size == 1:
        x_pos += delta
      else:
        x_pos.flat[i] += delta
      y_pos_feed_dict = dict([(x.name, x_pos)])
      y_pos = sess.run(y, feed_dict=y_pos_feed_dict)

      x_neg = x_data.copy()
      if x_size == 1:
        x_neg -= delta
      else:
        x_neg.flat[i] -= delta
      y_neg_feed_dict = dict([(x.name, x_neg)])
      y_neg = sess.run(y, feed_dict=y_neg_feed_dict)
      numeric_grad[i] = (y_pos - y_neg) / (2 * delta)
    return numeric_grad.reshape(x_shape)

  def _GetShape(self, sess, inputs):
    if not isinstance(inputs, collections.Iterable):
      return sess.run(array_ops.shape(inputs))
    else:
      return sess.run([array_ops.shape(x) for x in inputs])

  def _GradientCheckFp16(self, sess, y, xs, num_samples,
                         tolerance=1e-6, delta=1e-4):
    """Gradient check for Fp16.

    Fp16 numerical gradients end up being zeros. Use a new way to check
    gradients:

    Given multi-variant function:
    y = f(x1, x2, ... xn)
    delta_y = f(x1 + delta_x1, x2+delta_x2, ..., xn+delta_xn) -
              f(x1, x2, ..., xn)
            = f'(x1) * delta_x1 + f'(x2) * delta_x2 + .. + f'(xn) * delta_xn
    where:
      delta_xi are very small disturbance.
      f'(xi) is the gradient of y w.r.t xi.

    The gradient check verifies the expected delta_y calculated by the above
    equation is close to the actual delta_y.
    Args:
      sess: tf.Session object.
      y: output tensor.
      xs: a tensor or a list of input tensors.
      num_samples: number of test samples to run.
      tolerance: error tolerance.
      delta: the order of magnititued of input disturbance to apply to calculate
        the output change w.r.t inputs.
    """
    sym_grads = self._ComputeSymGrads(sess, y, xs)
    xs_shapes = self._GetShape(sess, xs)

    x_vals = [sess.run(x) for x in xs]
    for _ in range(num_samples):
      delta_xs = [delta * np.random.rand(*shape.tolist())
                  for shape in xs_shapes]

      feed_dict = {}
      for x, x_val, delta_x in zip(xs, x_vals, delta_xs):
        feed_dict[x] = x_val + delta_x
      actual_delta_y = (float(sess.run(y, feed_dict=feed_dict)) -
                        float(sess.run(y)))

      expected_delta_y = 0.
      for sym_grad, delta_x in zip(sym_grads, delta_xs):
        expected_delta_y += np.dot(
            sym_grad.astype(np.float32).flatten(),
            delta_x.astype(np.float32).flatten())
      self.assertAllClose(expected_delta_y, actual_delta_y,
                          atol=tolerance, rtol=tolerance)

  def _GradientCheck(self, sess, y, xs, tolerance=1e-6, delta=1e-4):
    sym_grads = self._ComputeSymGrads(sess, y, xs)

    num_grads = [self._ComputeNumericGrad(sess, y, x, delta) for x in xs]
    self.assertEqual(len(sym_grads), len(num_grads))
    for sym, num in zip(sym_grads, num_grads):
      self.assertFalse(np.any(np.isnan(sym)))
      self.assertFalse(np.any(np.isnan(num)))
      self.assertAllClose(sym, num, atol=tolerance, rtol=tolerance)

  def _ComputeSymGrads(self, sess, y, xs):
    sym_grads_t = gradients.gradients(y, xs)
    return sess.run(sym_grads_t)

  def _TestOneSimpleTraining(self, rnn_mode, num_layers, num_units, input_size,
                             batch_size, seq_length, dir_count, dropout, dtype,
                             use_v2, delta, tolerance):
    # Gradient checking runs two forward ops with almost the same input. Need to
    # make sure the drop patterns across the two runs are the same.
    logging.info("Training test with config: %s", locals())
    os.environ["TF_CUDNN_RESET_RND_GEN_STATE"] = str(True)

    np.random.seed(1234)
    random_seed.set_random_seed(5678)
    has_input_c = (rnn_mode == CUDNN_LSTM)
    direction = (CUDNN_RNN_UNIDIRECTION
                 if dir_count == 1 else CUDNN_RNN_BIDIRECTION)
    if use_v2:
      os.environ["TF_CUDNN_RNN_USE_V2"] = "1"
    else:
      os.environ["TF_CUDNN_RNN_USE_V2"] = "0"
    model = CudnnTestModel(
        rnn_mode,
        num_layers,
        num_units,
        input_size,
        direction=direction,
        dropout=dropout,
        dtype=dtype,
        training=True,
        bias_initializer=init_ops.random_normal_initializer(
            mean=1., dtype=dtype))
    rnn = model.rnn
    params = rnn.trainable_variables[0]

    inputs = variables.Variable(
        random_ops.random_uniform(
            [seq_length, batch_size, input_size], dtype=dtype),
        dtype=dtype)
    input_h = variables.Variable(
        random_ops.random_uniform(
            [num_layers * dir_count, batch_size, num_units], dtype=dtype),
        dtype=dtype)
    if has_input_c:
      input_c = variables.Variable(
          random_ops.random_uniform(
              [num_layers * dir_count, batch_size, num_units], dtype=dtype),
          dtype=dtype)
      initial_state = (input_h, input_c)
    else:
      initial_state = (input_h,)
    total_sum = model.FProp(inputs, initial_state, training=True)

    with self.test_session(use_gpu=True, graph=ops.get_default_graph()) as sess:
      sess.run(variables.global_variables_initializer())
      all_inputs = [inputs, params]
      for s in initial_state:
        all_inputs.append(s)
      if dtype == dtypes.float16:
        self._GradientCheckFp16(
            sess, total_sum, all_inputs,
            num_samples=FLAGS.grad_check_num_samples,
            tolerance=tolerance, delta=delta)
      else:
        for _ in range(FLAGS.grad_check_num_samples):
          # Each time choose a different set of inputs.
          sess.run(variables.global_variables_initializer())
          self._GradientCheck(
              sess, total_sum, all_inputs,
              tolerance=tolerance, delta=delta)

  def _TestSimpleTrainingHelper(self, rnn_mode, test_configs):
    dropouts = [0, 0.5, 1.]
    v2_options = [str(False), str(True)]
    for config, dropout, use_v2 in itertools.product(test_configs, dropouts,
                                                     v2_options):
      dtype = config.get("dtype", dtypes.float32)
      delta = config.get("delta", 1e-4)
      tolerance = config.get("tolerance", 1e-6)
      dir_count = config.get("dir_count", 1)
      shape = config["shape"]
      with ops.Graph().as_default():
        self._TestOneSimpleTraining(
            rnn_mode, shape["num_layers"], shape["num_units"],
            shape["input_size"], shape["batch_size"], shape["seq_length"],
            dir_count, dropout, dtype, use_v2, delta, tolerance)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingLSTMFp64(self):
    test_configs = [
        {
            "dtype": dtypes.float64,
            "tolerance": 5e-6,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_LSTM, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingLSTMFp32(self):
    test_configs = [
        {
            "dtype": dtypes.float32,
            "delta": 1e-4,
            "tolerance": 9e-2,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_LSTM, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingLSTMFp16(self):
    test_configs = [
        {
            "dtype": dtypes.float16,
            "delta": 1e-3,
            "tolerance": 9e-2,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
        {
            "dtype": dtypes.float16,
            "delta": 1e-2,
            "tolerance": 9e-2,
            "shape": {
                "num_layers": 2,
                "num_units": 6,
                "input_size": 8,
                "batch_size": 6,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_LSTM, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingGRUFp64(self):
    test_configs = [
        {
            "dtype": dtypes.float64,
            "tolerance": 5e-6,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            }
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_GRU, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingGRUFp32(self):
    test_configs = [
        {
            "dtype": dtypes.float32,
            "delta": 1e-3,
            "tolerance": 4e-3,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_GRU, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingGRUFp16(self):
    test_configs = [
        {
            "dtype": dtypes.float16,
            "delta": 2e-3,
            "tolerance": 6e-2,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_GRU, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingRNNTanhFp64(self):
    test_configs = [
        {
            "dtype": dtypes.float64,
            "tolerance": 5e-6,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_RNN_TANH, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingRNNTanhFp32(self):
    test_configs = [
        {
            "dtype": dtypes.float32,
            "delta": 1e-3,
            "tolerance": 5e-3,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_RNN_TANH, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingRNNTanhFp16(self):
    test_configs = [
        {
            "dtype": dtypes.float16,
            "delta": 1e-3,
            "tolerance": 5e-2,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_RNN_TANH, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingRNNReluFp64(self):
    test_configs = [
        {
            "dtype": dtypes.float64,
            "tolerance": 5e-6,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_RNN_RELU, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingRNNReluFp32(self):
    test_configs = [
        {
            "dtype": dtypes.float32,
            "delta": 1e-4,
            "tolerance": 3e-1,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_RNN_RELU, test_configs)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTrainingRNNReluFp16(self):
    test_configs = [
        {
            "dtype": dtypes.float16,
            "delta": 1e-3,
            "tolerance": 7e-2,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    self._TestSimpleTrainingHelper(CUDNN_RNN_RELU, test_configs)


if __name__ == "__main__":
  argv0 = sys.argv[0]
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--grad_check_num_samples",
      type=int,
      default=5,
      help="Number of samples to run for gradient check.")
  FLAGS, unparsed = parser.parse_known_args()
  sys.argv = [argv0] + unparsed
  googletest.main()
