# -*- coding: utf-8 -*-
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

import itertools
import os
import unittest

import numpy as np

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn as rnn_lib
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver as saver_lib


def _CreateModel(rnn_mode,
                 num_layers,
                 num_units,
                 input_size,
                 input_mode="linear_input",
                 direction=cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION,
                 dtype=dtypes.float32,
                 dropout=0.):
  del input_mode
  if rnn_mode == cudnn_rnn_ops.CUDNN_LSTM:
    model_fn = cudnn_rnn_ops.CudnnLSTM
  elif rnn_mode == cudnn_rnn_ops.CUDNN_GRU:
    model_fn = cudnn_rnn_ops.CudnnGRU
  elif rnn_mode == cudnn_rnn_ops.CUDNN_RNN_TANH:
    model_fn = cudnn_rnn_ops.CudnnRNNTanh
  elif rnn_mode == cudnn_rnn_ops.CUDNN_RNN_RELU:
    model_fn = cudnn_rnn_ops.CudnnRNNRelu
  else:
    raise ValueError("Invalid rnn_mode: %s" % rnn_mode)
  return model_fn(
      num_layers,
      num_units,
      input_size,
      direction=direction,
      dtype=dtype,
      dropout=dropout)


def _CreateParamsSavable(params,
                         model,
                         base_variable_scope="rnn",
                         name="params_canonical"):
  """Create a RNNParamsSaveable for the weight and bias parameters.

  Args:
    params: a Variable for weight and bias parameters.
    model: a CudnnRNN model.
    base_variable_scope: a string, prefix of names of saved variables.
    name: a string, name of the RNNParamsSaveable object.
  """
  params_saveable = cudnn_rnn_ops.RNNParamsSaveable(
      model,
      model.params_to_canonical,
      model.canonical_to_params, [params],
      base_variable_scope=base_variable_scope,
      name=name)
  ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, params_saveable)


def _BuildCudnnForward(rnn_mode,
                       num_layers,
                       num_units,
                       input_data,
                       is_training=False):
  input_data_shape = input_data.get_shape().with_rank(3)
  batch_size = input_data_shape[1].value
  input_size = input_data_shape[2].value
  model = _CreateModel(rnn_mode, num_layers, num_units, input_size)

  # Set zero init input states
  input_h = constant_op.constant(
      np.zeros([num_layers, batch_size, num_units]), dtype=dtypes.float32)
  has_input_c = (rnn_mode == cudnn_rnn_ops.CUDNN_LSTM)
  if has_input_c:
    input_c = constant_op.constant(
        np.zeros([num_layers, batch_size, num_units]), dtype=dtypes.float32)

  # Set rnn params
  params_size_t = model.params_size()
  params = variables.Variable(
      random_ops.random_uniform([params_size_t]), validate_shape=False)
  args = {
      "input_data": input_data,
      "input_h": input_h,
      "params": params,
      "is_training": is_training
  }
  if has_input_c:
    args["input_c"] = input_c
  # Build cell
  output_tuple = model(**args)

  # Create savable objects for params
  _CreateParamsSavable(params, model)

  return output_tuple, model, params


def _MinLSTMParamSize(num_layers,
                      num_units,
                      input_size,
                      direction=cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION):
  if direction == cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION:
    first_layer_weights = 4 * num_units * (num_units + input_size)
    higher_layer_weights = 8 * (num_layers - 1) * num_units * num_units
    all_biases = 8 * num_layers * num_units
    return first_layer_weights + higher_layer_weights + all_biases
  elif direction == cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION:
    first_layer_weights = 4 * num_units * (num_units + input_size)
    higher_layer_weights = (num_layers - 1) * (
        4 * 2 * num_units * num_units + 4 * num_units**2)
    all_biases = 8 * num_layers * num_units
    return 2 * (first_layer_weights + higher_layer_weights + all_biases)
  else:
    raise ValueError("%s direction is not supported.")


def _CreateCudnnCompatibleCanonicalRNN(cudnn_model,
                                       inputs,
                                       scope="rnn"):
  model = cudnn_model.rnn_mode
  if model not in (cudnn_rnn_ops.CUDNN_LSTM, cudnn_rnn_ops.CUDNN_GRU):
    raise ValueError("%s is not supported!" % model)

  num_units = cudnn_model.num_units
  num_layers = cudnn_model.num_layers
  # To reuse cuDNN-trained models, must use cudnn compatible rnn cells.
  if model == cudnn_rnn_ops.CUDNN_LSTM:
    single_cell = lambda: cudnn_rnn_ops.CudnnCompatibleLSTMCell(num_units)
  else:
    single_cell = lambda: cudnn_rnn_ops.CudnnCompatibleGRUCell(num_units)
  cell = rnn_cell_impl.MultiRNNCell([single_cell() for _ in range(num_layers)])
  return rnn_lib.dynamic_rnn(
      cell, inputs, dtype=dtypes.float32, time_major=True, scope=scope)


# TODO(jamesqin): Merge the transform logic into RNNParamsSaveable.
def _TransformBidirectionalCudnnLSTMParams(lstm, params):
  """Transforms bidi CuDNN LSTM params to canonical weights and biases.

  Args:
    lstm: tf.contrib.cudnn_rnn.CudnnLSTM instance.
    params: A monolithic Tensor used by CudnnLSTM to store all vars across all
      layers and directions.
  Returns:
    One weights list and one biases list. Each list stores the params in
    canonical shape and in "first forward, then backward" order.
  """
  weights, biases = lstm.params_to_canonical(params)
  transformed_weights, transformed_biases = [], []

  # canonical bidirectional lstm
  def _SwitchInner(array, base_idx):
    array[base_idx + 1], array[base_idx + 2] = (array[base_idx + 2],
                                                array[base_idx + 1])

  for i in range(lstm.num_layers):
    base_idx = i * 16
    num_units = lstm.num_units
    input_size = lstm.input_size if i == 0 else num_units
    stitched_w = []
    for j in range(4):
      stitched_w.append(
          array_ops.concat(
              [
                  array_ops.reshape(weights[base_idx + j],
                                    [num_units, input_size]),
                  array_ops.reshape(weights[base_idx + j + 4],
                                    [num_units, num_units])
              ],
              axis=1))
    # cuDNN weights are in ifco order, convert to icfo order.
    _SwitchInner(stitched_w, 0)
    transformed_weights.append(
        array_ops.transpose(array_ops.concat(stitched_w, axis=0)))

    # Stitch biases together in this layer.
    # Convert to icfo order.
    _SwitchInner(biases, base_idx)
    _SwitchInner(biases, base_idx + 4)
    # The bias for layer input.
    b_in = array_ops.concat(biases[base_idx:base_idx + 4], axis=0)
    # The bias for recurrent input.
    b_rec = array_ops.concat(biases[base_idx + 4:base_idx + 8], axis=0)

    transformed_biases.append(b_in + b_rec)

    # backward
    base_idx = i * 16 + 8
    num_units = lstm.num_units
    input_size = lstm.input_size if i == 0 else num_units
    stitched_w = []
    for j in range(4):
      stitched_w.append(
          array_ops.concat(
              [
                  array_ops.reshape(weights[base_idx + j],
                                    [num_units, input_size]),
                  array_ops.reshape(weights[base_idx + j + 4],
                                    [num_units, num_units])
              ],
              axis=1))
    # cuDNN weights are in ifco order, convert to icfo order.
    _SwitchInner(stitched_w, 0)
    transformed_weights.append(
        array_ops.transpose(array_ops.concat(stitched_w, axis=0)))

    # Stitch biases together in this layer.
    # Convert to icfo order.
    _SwitchInner(biases, base_idx)
    _SwitchInner(biases, base_idx + 4)
    # The bias for layer input.
    b_in = array_ops.concat(biases[base_idx:base_idx + 4], axis=0)
    # The bias for recurrent input.
    b_rec = array_ops.concat(biases[base_idx + 4:base_idx + 8], axis=0)

    transformed_biases.append(b_in + b_rec)
  return transformed_weights, transformed_biases


class CudnnRNNTestSaveRestore(TensorFlowTestCase):

  def _testSaveRestoreVariable(self, rnn_mode, direction, dtype):
    with ops.Graph().as_default():
      model = _CreateModel(
          rnn_mode,
          num_layers=2,
          num_units=7,
          input_size=3,
          direction=direction,
          dtype=dtype)
      random_seed.set_random_seed(1234)
      params_size_t = model.params_size()
      params = variables.Variable(
          random_ops.random_uniform([params_size_t], dtype=dtype),
          dtype=dtype,
          validate_shape=False)
      _CreateParamsSavable(params, model)
      save_path = os.path.join(self.get_temp_dir(),
                               "save-restore-variable-test")
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(
          use_gpu=True, graph=ops.get_default_graph()) as sess:
        sess.run(variables.global_variables_initializer())
        params_v = sess.run(params)
        val = saver.save(sess, save_path)
        self.assertEqual(save_path, val)
      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(
          use_gpu=True, graph=ops.get_default_graph()) as sess:
        reset_params = state_ops.assign(
            params,
            array_ops.zeros([params_size_t], dtype=dtype),
            validate_shape=False)
        sess.run(reset_params)
        saver.restore(sess, save_path)
        params_v_restored = sess.run(params)
        self.assertAllEqual(params_v, params_v_restored)

  def _testSaveRestoreTwoVariables(self, rnn_mode, direction, dtype):
    with ops.Graph().as_default():
      model = _CreateModel(
          rnn_mode,
          num_layers=2,
          num_units=7,
          input_size=3,
          direction=direction,
          dtype=dtype)
      random_seed.set_random_seed(1234)
      params_size_t = model.params_size()
      names = ["rnn_1", "rnn_2"]
      param_vars = [
          variables.Variable(
              random_ops.random_uniform([params_size_t], dtype=dtype),
              dtype=dtype,
              validate_shape=False) for name in names
      ]
      for name, params in zip(names, param_vars):
        _CreateParamsSavable(params, model, name, name)
      save_path = os.path.join(self.get_temp_dir(),
                               "save-restore-variable-test")
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(
          use_gpu=True, graph=ops.get_default_graph()) as sess:
        sess.run(variables.global_variables_initializer())
        params_v = sess.run(param_vars)
        val = saver.save(sess, save_path)
        self.assertEqual(save_path, val)
      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(
          use_gpu=True, graph=ops.get_default_graph()) as sess:
        reset_params = [
            state_ops.assign(
                params,
                array_ops.zeros([params_size_t], dtype=dtype),
                validate_shape=False) for params in param_vars
        ]
        sess.run(reset_params)
        saver.restore(sess, save_path)
        params_v_restored = sess.run(param_vars)
        for v, v_restored in zip(params_v, params_v_restored):
          self.assertAllEqual(v, v_restored)

  def _testSaveRestoreOutput(self, rnn_mode, direction, dtype):
    with ops.Graph().as_default():
      num_layers = 2
      num_units = 7
      input_size = 7
      seq_length = 10
      batch_size = 5
      dir_count = 1 if direction == cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION else 2
      model = _CreateModel(
          rnn_mode,
          num_layers,
          num_units,
          input_size,
          direction=direction,
          dtype=dtype)
      params_size_t = model.params_size()
      params = variables.Variable(
          array_ops.ones([params_size_t], dtype=dtype),
          validate_shape=False,
          dtype=dtype)
      _CreateParamsSavable(params, model)
      save_path = os.path.join(self.get_temp_dir(), "save-restore-output-test")
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

      np.random.seed(1234)
      has_input_c = (rnn_mode == cudnn_rnn_ops.CUDNN_LSTM)
      input_data = constant_op.constant(
          np.random.randn(seq_length, batch_size, input_size), dtype=dtype)
      input_h = constant_op.constant(
          np.random.randn(num_layers * dir_count, batch_size, num_units),
          dtype=dtype)
      if has_input_c:
        input_c = constant_op.constant(
            np.random.randn(num_layers * dir_count, batch_size, num_units),
            dtype=dtype)
        outputs = model(
            input_data=input_data,
            input_h=input_h,
            input_c=input_c,
            params=params,
            is_training=False)
      else:
        outputs = model(
            input_data=input_data,
            input_h=input_h,
            params=params,
            is_training=False)
      total_sum = sum(map(math_ops.reduce_sum, outputs))
      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(
          use_gpu=True, graph=ops.get_default_graph()) as sess:
        sess.run(variables.global_variables_initializer())
        total_sum_v = sess.run(total_sum)
        val = saver.save(sess, save_path)
        self.assertEqual(save_path, val)
      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(
          use_gpu=True, graph=ops.get_default_graph()) as sess:
        reset_params = state_ops.assign(
            params,
            array_ops.zeros([params_size_t], dtype=dtype),
            validate_shape=False)
        sess.run(reset_params)
        saver.restore(sess, save_path)
        total_sum_v_restored = sess.run(total_sum)
        self.assertAllClose(total_sum_v, total_sum_v_restored, atol=1e-5)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSaveRestore(self):
    rnn_modes = [
        cudnn_rnn_ops.CUDNN_LSTM, cudnn_rnn_ops.CUDNN_GRU,
        cudnn_rnn_ops.CUDNN_RNN_TANH, cudnn_rnn_ops.CUDNN_RNN_RELU
    ]
    directions = [
        cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION,
        cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION
    ]
    dtype_list = [dtypes.float32, dtypes.float64]
    for rnn_mode, direction, dtype in itertools.product(rnn_modes, directions,
                                                        dtype_list):
      self._testSaveRestoreVariable(rnn_mode, direction, dtype)
      self._testSaveRestoreTwoVariables(rnn_mode, direction, dtype)
      self._testSaveRestoreOutput(rnn_mode, direction, dtype)


class CudnnRNNTestCompatibleRnnCells(TensorFlowTestCase):

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testCudnnCompatibleRnnCells(self):
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
    for rnn, cfg in itertools.product((cudnn_rnn_ops.CUDNN_LSTM,), configs):
      self._testCudnnCompatibleRnnCells(cfg["num_layers"], cfg["seq_length"],
                                        cfg["num_units"], cfg["input_size"],
                                        cfg["batch_size"], rnn)
    # TODO(jamesqin): Add CudnnCompatibleGRUBlockCell.
    for rnn, cfg in itertools.product((cudnn_rnn_ops.CUDNN_GRU,), configs):
      self._testCudnnCompatibleRnnCells(cfg["num_layers"], cfg["seq_length"],
                                        cfg["num_units"], cfg["input_size"],
                                        cfg["batch_size"], rnn)

  def _testCudnnCompatibleRnnCells(self, num_layers, seq_length, num_units,
                                   input_size, batch_size, rnn_mode):
    has_state_c = rnn_mode == cudnn_rnn_ops.CUDNN_LSTM
    np.random.seed(0)
    # Train graph
    with ops.Graph().as_default():
      random_seed.set_random_seed(299)
      input_data = array_ops.placeholder(
          dtypes.float32, shape=[seq_length, batch_size, input_size])
      output_tuple, cudnn_model, cudnn_params = _BuildCudnnForward(
          rnn_mode, num_layers, num_units, input_data, is_training=True)
      target_output = array_ops.placeholder(dtype=dtypes.float32, shape=None)
      total_sum = sum(map(math_ops.reduce_sum, output_tuple))

      loss_op = losses.log_loss(labels=target_output, predictions=total_sum)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=1e-2)
      train_op = optimizer.minimize(loss_op)

      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

      # Train Cudnn model
      with self.test_session(
          use_gpu=True, graph=ops.get_default_graph()) as sess:
        sess.run(variables.global_variables_initializer())
        # Train 128 steps
        num_steps = 128
        for _ in range(num_steps):
          inputs = np.random.rand(seq_length, batch_size,
                                  input_size).astype(np.float32)
          targets = np.random.rand()
          sess.run(
              train_op, feed_dict={input_data: inputs,
                                   target_output: targets})

        save_path = os.path.join(self.get_temp_dir(),
                                 ("cudnn-rnn-%s-test" % rnn_mode))
        save_v = saver.save(sess, save_path)
        self.assertEqual(save_path, save_v)
        cudnn_params_v = sess.run(cudnn_params)

    # cuDNN inference graph
    with ops.Graph().as_default():
      random_seed.set_random_seed(299)
      cudnn_inputs = array_ops.placeholder(
          dtypes.float32, shape=[seq_length, batch_size, input_size])
      (cudnn_output_tuple, cudnn_model, cudnn_params) = _BuildCudnnForward(
          rnn_mode, num_layers, num_units, cudnn_inputs, is_training=False)
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

      inference_input = np.random.rand(seq_length, batch_size,
                                       input_size).astype(np.float32)
      with self.test_session(
          use_gpu=True, graph=ops.get_default_graph()) as sess:
        sess.run(variables.global_variables_initializer())
        saver.restore(sess, save_path)
        restored_cudnn_params_v = sess.run(cudnn_params)
        self.assertAllEqual(cudnn_params_v, restored_cudnn_params_v)

        # Cudnn inference
        cudnn_output = sess.run(
            cudnn_output_tuple, feed_dict={cudnn_inputs: inference_input})

    # Canonical RNN inference graph
    with ops.Graph().as_default():
      random_seed.set_random_seed(299)
      cell_inputs = array_ops.placeholder(
          dtypes.float32, shape=[seq_length, batch_size, input_size])
      (output, states) = _CreateCudnnCompatibleCanonicalRNN(
          cudnn_model, cell_inputs)
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

      with self.test_session(
          use_gpu=True, graph=ops.get_default_graph()) as sess:
        saver.restore(sess, save_path)

        # BlockCell inference
        output_v, states_v = sess.run(
            [output, states], feed_dict={cell_inputs: inference_input})

        # output across timestamps are packed into one tensor.
        self.assertAllClose(cudnn_output[0], output_v, atol=1e-6, rtol=1e-6)

        for i in range(num_layers):
          if has_state_c:
            # output_h
            self.assertAllClose(
                cudnn_output[1][i, :], states_v[i].h, atol=1e-6, rtol=1e-6)
            # output_c
            self.assertAllClose(
                cudnn_output[2][i, :], states_v[i].c, atol=1e-6, rtol=1e-6)
          else:
            self.assertAllClose(
                cudnn_output[1][i, :], states_v[i], atol=1e-6, rtol=1e-6)


class CudnnRNNTestParamsSize(TensorFlowTestCase):

  def _testOneLSTMParamsSize(self, num_layers, num_units, input_size,
                             direction):
    logging.info("Testing one lstm param size with config: %s", locals())
    min_params_size = _MinLSTMParamSize(num_layers, num_units, input_size,
                                        direction)
    model = _CreateModel(
        cudnn_rnn_ops.CUDNN_LSTM,
        num_layers,
        num_units,
        input_size,
        direction=direction)
    params_size = model.params_size()
    with self.test_session(use_gpu=True, graph=ops.get_default_graph()) as sess:
      params_size_v = sess.run(params_size)
      self.assertLessEqual(min_params_size, params_size_v)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testLSTMParamsSize(self):
    test_configs = [
        [4, 200, 200],
        [4, 200, 300],
        [4, 200, 100],
        [1, 100, 200],
        [2, 200, 100],
        [3, 200, 400],
    ]
    directions = [
        cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION,
        cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION
    ]
    for (config, direction) in itertools.product(test_configs, directions):
      num_layers, num_units, input_size = config
      with ops.Graph().as_default():
        self._testOneLSTMParamsSize(num_layers, num_units, input_size,
                                    direction)


class CudnnRNNTestInference(TensorFlowTestCase):

  def _testOneSimpleInference(self, rnn_mode, num_layers, num_units, input_size,
                              batch_size, seq_length, dir_count, dropout,
                              expected, tolerance):
    random_seed.set_random_seed(5678)
    model = _CreateModel(
        rnn_mode,
        num_layers,
        num_units,
        input_size,
        input_mode="auto_select",
        direction=(cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION if dir_count == 1
                   else cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION),
        dropout=dropout)
    has_input_c = (rnn_mode == cudnn_rnn_ops.CUDNN_LSTM)
    params_size_t = model.params_size()
    input_data = array_ops.ones([seq_length, batch_size, input_size])
    input_h = array_ops.ones([num_layers * dir_count, batch_size, num_units])
    params = variables.Variable(
        array_ops.ones([params_size_t]), validate_shape=False)
    if has_input_c:
      input_c = array_ops.ones([num_layers * dir_count, batch_size, num_units])
      output, output_h, output_c = model(
          input_data=input_data,
          input_h=input_h,
          input_c=input_c,
          params=params,
          is_training=False)
    else:
      output, output_h = model(
          input_data=input_data,
          input_h=input_h,
          params=params,
          is_training=False)
    output_sum = math_ops.reduce_sum(output)
    output_h_sum = math_ops.reduce_sum(output_h)
    total_sum = output_sum + output_h_sum
    if has_input_c:
      output_c_sum = math_ops.reduce_sum(output_c)
      total_sum += output_c_sum
    with self.test_session(use_gpu=True, graph=ops.get_default_graph()) as sess:
      sess.run(variables.global_variables_initializer())
      total_sum_v = sess.run([total_sum])

      self.assertAllClose(
          total_sum_v[0], expected, atol=tolerance, rtol=tolerance)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleInference(self):
    test_configs = [
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_LSTM,
            "expected": 231833.22,
            "tolerance": 1e-2,
            "shape": {
                "num_layers": 4,
                "num_units": 200,
                "input_size": 200,
                "batch_size": 20,
                "seq_length": 10,
                "dir_count": 1,
            },
        },
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_GRU,
            "expected": 56000,
            "tolerance": 1e-2,
            "shape": {
                "num_layers": 4,
                "num_units": 200,
                "input_size": 200,
                "batch_size": 20,
                "seq_length": 10,
                "dir_count": 1,
            },
        },
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_RNN_TANH,
            "expected": 56000,
            "tolerance": 1e-2,
            "shape": {
                "num_layers": 4,
                "num_units": 200,
                "input_size": 200,
                "batch_size": 20,
                "seq_length": 10,
                "dir_count": 1,
            },
        },
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_RNN_RELU,
            "expected": 130688,
            "tolerance": 1e-2,
            "shape": {
                "num_layers": 2,
                "num_units": 8,
                "input_size": 4,
                "batch_size": 4,
                "seq_length": 2,
                "dir_count": 1,
            },
        },
    ]
    # Cudnn scales result for dropout during training, therefore dropout has no
    # impact for inference results.
    # (lstm, gru, rnn_tanh are saturated in the test. rnn_relu case is most
    # demonstrative of the dropout-invariant nature of CudnnRnn.)
    dropouts = [0., 0.5, 1.]
    for (config, dropout) in itertools.product(test_configs, dropouts):
      rnn_mode = config["rnn_mode"]
      expected = config["expected"]
      tolerance = config["tolerance"]
      shape = config["shape"]
      with ops.Graph().as_default():
        self._testOneSimpleInference(
            rnn_mode, shape["num_layers"], shape["num_units"],
            shape["input_size"], shape["batch_size"], shape["seq_length"],
            shape["dir_count"], dropout, expected, tolerance)


class CudnnRNNTestTraining(TensorFlowTestCase):

  def _testOneSimpleTraining(self, rnn_mode, num_layers, num_units, input_size,
                             batch_size, seq_length, dir_count, dropout, dtype,
                             delta, tolerance):
    # Gradient checking runs two forward ops with almost the same input. Need to
    # make sure the drop patterns across the two runs are the same.
    logging.info("Training test with config: %s", locals())
    old_env_state = os.environ.get("TF_CUDNN_RESET_RND_GEN_STATE", str(False))
    os.environ["TF_CUDNN_RESET_RND_GEN_STATE"] = str(True)
    has_input_c = (rnn_mode == cudnn_rnn_ops.CUDNN_LSTM)
    random_seed.set_random_seed(5678)
    direction = (cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION if dir_count == 1
                 else cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION)
    model = _CreateModel(
        rnn_mode,
        num_layers,
        num_units,
        input_size,
        direction=direction,
        dtype=dtype,
        dropout=dropout)
    params_size_t = model.params_size()
    input_data = variables.Variable(
        random_ops.random_uniform(
            [seq_length, batch_size, input_size], dtype=dtype),
        dtype=dtype)
    input_h = variables.Variable(
        random_ops.random_uniform(
            [num_layers * dir_count, batch_size, num_units], dtype=dtype),
        dtype=dtype)
    params = variables.Variable(
        random_ops.random_uniform([params_size_t], dtype=dtype),
        validate_shape=False,
        dtype=dtype)
    if has_input_c:
      input_c = variables.Variable(
          random_ops.random_uniform(
              [num_layers * dir_count, batch_size, num_units], dtype=dtype),
          dtype=dtype)

      output, output_h, output_c = model(
          input_data=input_data,
          input_h=input_h,
          input_c=input_c,
          params=params)
    else:
      output, output_h = model(
          input_data=input_data, input_h=input_h, params=params)
    output_sum = math_ops.reduce_sum(output)
    output_h_sum = math_ops.reduce_sum(output_h)
    total_sum = output_sum + output_h_sum
    if has_input_c:
      output_c_sum = math_ops.reduce_sum(output_c)
      total_sum += output_c_sum

    with self.test_session(use_gpu=True, graph=ops.get_default_graph()) as sess:
      params_size_v = sess.run(params_size_t)
      inputs_and_shapes = [
          (input_data, [seq_length, batch_size, input_size]),
          (input_h, [num_layers * dir_count, batch_size, num_units]),
          (params, [params_size_v]),
      ]
      if has_input_c:
        inputs_and_shapes.append(
            (input_c, [num_layers * dir_count, batch_size, num_units]),)
      sess.run(variables.global_variables_initializer())
      all_inputs = [entry[0] for entry in inputs_and_shapes]
      all_shapes = [entry[1] for entry in inputs_and_shapes]

      err = gradient_checker.compute_gradient_error(
          all_inputs, all_shapes, total_sum, [1], delta=delta)

      self.assertLess(err, tolerance)
      os.environ["TF_CUDNN_RESET_RND_GEN_STATE"] = old_env_state

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTraining(self):
    test_configs = [
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_LSTM,
            "dtype": dtypes.float64,
            "delta": 1e-4,
            "tolerance": 5e-6,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
                "dir_count": 1,
            },
        },
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_GRU,
            "dtype": dtypes.float64,
            "delta": 1e-4,
            "tolerance": 5e-6,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
                "dir_count": 1,
            },
        },
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_RNN_TANH,
            "dtype": dtypes.float64,
            "delta": 1e-4,
            "tolerance": 5e-6,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
                "dir_count": 1,
            },
        },
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_RNN_RELU,
            "dtype": dtypes.float64,
            "delta": 1e-4,
            "tolerance": 5e-6,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
                "dir_count": 1,
            },
        },
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_LSTM,
            "dtype": dtypes.float32,
            "tolerance": 1.5e-2,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_GRU,
            "dtype": dtypes.float32,
            "tolerance": 4e-3,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_RNN_TANH,
            "dtype": dtypes.float32,
            "tolerance": 5e-3,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_RNN_RELU,
            "dtype": dtypes.float32,
            "tolerance": 4e-1,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
            },
        },
    ]
    dropouts = [0., 0.5, 1.]
    dir_counts = [1]
    for config, dropout, dir_count in itertools.product(test_configs, dropouts,
                                                        dir_counts):
      rnn_mode = config["rnn_mode"]
      dtype = config.get("dtype", dtypes.float32)
      delta = config.get("delta", 1e-3)
      tolerance = config["tolerance"]
      shape = config["shape"]
      with ops.Graph().as_default():
        self._testOneSimpleTraining(rnn_mode, shape["num_layers"],
                                    shape["num_units"], shape["input_size"],
                                    shape["batch_size"], shape["seq_length"],
                                    dir_count, dropout, dtype, delta, tolerance)


if __name__ == "__main__":
  googletest.main()
