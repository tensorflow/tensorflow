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
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_lib

CUDNN_RNN_UNIDIRECTION = cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION
CUDNN_RNN_BIDIRECTION = cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION

CUDNN_LSTM = cudnn_rnn_ops.CUDNN_LSTM
CUDNN_GRU = cudnn_rnn_ops.CUDNN_GRU
CUDNN_RNN_RELU = cudnn_rnn_ops.CUDNN_RNN_RELU
CUDNN_RNN_TANH = cudnn_rnn_ops.CUDNN_RNN_TANH

CUDNN_LSTM_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_LSTM_PARAMS_PER_LAYER
CUDNN_GRU_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_GRU_PARAMS_PER_LAYER
CUDNN_RNN_TANH_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_RNN_TANH_PARAMS_PER_LAYER
CUDNN_RNN_RELU_PARAMS_PER_LAYER = cudnn_rnn_ops.CUDNN_RNN_RELU_PARAMS_PER_LAYER


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
                         base_variable_scope=None,
                         name="params_canonical"):
  """Create a RNNParamsSaveable for the weight and bias parameters.

  Args:
    params: a Variable for weight and bias parameters.
    model: a CudnnRNN model.
    base_variable_scope: a string, prefix of names of saved variables.
    name: a string, name of the RNNParamsSaveable object.
  Returns:
    a RNNParamsSaveable object.
  """
  if model._rnn_mode == CUDNN_LSTM:
    fn = cudnn_rnn_ops.CudnnLSTMSaveable
  elif model._rnn_mode == CUDNN_GRU:
    fn = cudnn_rnn_ops.CudnnGRUSaveable
  elif model._rnn_mode == CUDNN_RNN_TANH:
    fn = cudnn_rnn_ops.CudnnRNNTanhSaveable
  elif model._rnn_mode == CUDNN_RNN_RELU:
    fn = cudnn_rnn_ops.CudnnRNNReluSaveable
  params_saveable = fn(
      params,
      model.num_layers,
      model.num_units,
      model.input_size,
      model.input_mode,
      model.direction,
      scope=base_variable_scope,
      name=name)
  ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
  return params_saveable


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


class CudnnRNNTestSaveRestore(TensorFlowTestCase):

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

  def _testSaveRestoreVariable(self, rnn_mode, direction, dtype):
    num_layers = 2
    num_units = 7
    input_size = 3
    with ops.Graph().as_default():
      model = _CreateModel(
          rnn_mode,
          num_layers=num_layers,
          num_units=num_units,
          input_size=input_size,
          direction=direction,
          dtype=dtype)
      random_seed.set_random_seed(1234)
      params_size_t = model.params_size()
      params = variables.Variable(
          random_ops.random_uniform([params_size_t], dtype=dtype),
          dtype=dtype,
          validate_shape=False)
      saveable = _CreateParamsSavable(params, model)
      weights, biases = saveable._OpaqueParamsToCanonical()
      reset_params = state_ops.assign(
          params,
          array_ops.zeros([params_size_t], dtype=dtype),
          validate_shape=False)
      save_path = os.path.join(self.get_temp_dir(),
                               "save-restore-variable-test")
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(
          use_gpu=True, graph=ops.get_default_graph()) as sess:
        sess.run(variables.global_variables_initializer())
        val = saver.save(sess, save_path)
        self.assertEqual(save_path, val)

        weights_v, biases_v = sess.run([weights, biases])

        sess.run(reset_params)
        saver.restore(sess, save_path)
        weights_v_restored, biases_v_restored = sess.run([weights, biases])

        self._CompareWeights(weights_v, weights_v_restored)
        self._CompareBiases(biases_v, biases_v_restored, rnn_mode, num_layers,
                            direction)

  def _testSaveRestoreTwoVariables(self, rnn_mode, direction, dtype):
    num_layers = 2
    num_units = 7
    input_size = 3
    with ops.Graph().as_default():
      model = _CreateModel(
          rnn_mode,
          num_layers=num_layers,
          num_units=num_units,
          input_size=input_size,
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
      saveables = []
      for name, params in zip(names, param_vars):
        saveables.append(_CreateParamsSavable(params, model, name, name))
      weights1, biases1 = saveables[0]._OpaqueParamsToCanonical()
      weights2, biases2 = saveables[1]._OpaqueParamsToCanonical()
      reset_params = [
          state_ops.assign(
              params,
              array_ops.zeros([params_size_t], dtype=dtype),
              validate_shape=False) for params in param_vars
      ]
      save_path = os.path.join(self.get_temp_dir(),
                               "save-restore-variable-test")
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
      # Passing graph explicitly, otherwise an old sess would be reused.
      with self.test_session(use_gpu=True,
                             graph=ops.get_default_graph()) as sess:
        sess.run(variables.global_variables_initializer())
        val = saver.save(sess, save_path)
        self.assertEqual(save_path, val)
        weights1_v, biases1_v = sess.run([weights1, biases1])
        weights2_v, biases2_v = sess.run([weights2, biases2])

        sess.run(reset_params)
        saver.restore(sess, save_path)
        weights1_v_restored, biases1_v_restored = sess.run([weights1, biases1])
        weights2_v_restored, biases2_v_restored = sess.run([weights2, biases2])

        self._CompareWeights(weights1_v, weights1_v_restored)
        self._CompareWeights(weights2_v, weights2_v_restored)
        self._CompareBiases(biases1_v, biases1_v_restored, rnn_mode, num_layers,
                            direction)
        self._CompareBiases(biases2_v, biases2_v_restored, rnn_mode, num_layers,
                            direction)

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
            "tolerance": 5e-1,
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
