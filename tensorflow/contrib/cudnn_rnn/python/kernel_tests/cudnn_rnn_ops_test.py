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
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import saver as saver_lib


def _create_cudnn_compatible_canonical_rnn(cudnn_model,
                                           inputs,
                                           use_block_cell,
                                           scope="rnn"):
  model = cudnn_model.rnn_mode
  if model not in (cudnn_rnn_ops.CUDNN_LSTM, cudnn_rnn_ops.CUDNN_GRU):
    raise ValueError("%s is not supported!" % model)
  if model == cudnn_rnn_ops.CUDNN_GRU and use_block_cell:
    raise ValueError("gru is not supported when using block cell!")

  num_units = cudnn_model.num_units
  num_layers = cudnn_model.num_layers
  # To reuse cuDNN-trained models, must use cudnn compatible rnn cells.
  if use_block_cell:
    single_cell = lambda: cudnn_rnn_ops.CudnnCompatibleLSTMBlockCell(num_units)
  else:
    if model == cudnn_rnn_ops.CUDNN_LSTM:
      single_cell = lambda: cudnn_rnn_ops.CudnnCompatibleLSTMCell(num_units)
    else:
      single_cell = lambda: cudnn_rnn_ops.CudnnCompatibleGRUCell(num_units)
  cell = rnn_cell_impl.MultiRNNCell([single_cell() for _ in range(num_layers)])
  return rnn_lib.dynamic_rnn(
      cell, inputs, dtype=dtypes.float32, time_major=True, scope=scope)


class CudnnRNNTest(TensorFlowTestCase):

  def _CreateModel(self,
                   rnn_mode,
                   num_layers,
                   num_units,
                   input_size,
                   input_mode="linear_input",
                   dropout=0.):
    if rnn_mode == cudnn_rnn_ops.CUDNN_LSTM:
      model = cudnn_rnn_ops.CudnnLSTM(
          num_layers, num_units, input_size, dropout=dropout)
    elif rnn_mode == cudnn_rnn_ops.CUDNN_GRU:
      model = cudnn_rnn_ops.CudnnGRU(
          num_layers, num_units, input_size, dropout=dropout)
    elif rnn_mode == cudnn_rnn_ops.CUDNN_RNN_TANH:
      model = cudnn_rnn_ops.CudnnRNNTanh(
          num_layers, num_units, input_size, dropout=dropout)
    elif rnn_mode == cudnn_rnn_ops.CUDNN_RNN_RELU:
      model = cudnn_rnn_ops.CudnnRNNRelu(
          num_layers, num_units, input_size, dropout=dropout)
    else:
      raise ValueError("Invalid rnn_mode: %s" % rnn_mode)
    return model

  def _create_params_savable(self, params, model):
    """Create a RNNParamsSaveable for the weight and bias parameters.

    Args:
      params: a Variable for weight and bias parameters.
      model: a CudnnRNN model.
    """
    params_saveable = cudnn_rnn_ops.RNNParamsSaveable(
        model, model.params_to_canonical, model.canonical_to_params, [params],
        "rnn")
    ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, params_saveable)

  def _testSaveRestoreVariable(self, rnn_mode):
    model = self._CreateModel(rnn_mode, num_layers=2, num_units=7, input_size=3)
    random_seed.set_random_seed(1234)
    params_size_t = model.params_size()
    params = variables.Variable(
        random_ops.random_uniform([params_size_t]), validate_shape=False)
    self._create_params_savable(params, model)
    save_path = os.path.join(self.get_temp_dir(), "save-restore-variable-test")
    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)
    with self.test_session(use_gpu=True) as sess:
      sess.run(variables.global_variables_initializer())
      params_v = sess.run(params)
      val = saver.save(sess, save_path)
      self.assertEqual(save_path, val)
    with self.test_session(use_gpu=True) as sess:
      reset_params = state_ops.assign(params, array_ops.zeros([params_size_t]))
      sess.run(reset_params)
      saver.restore(sess, save_path)
      params_v_restored = sess.run(params)
      self.assertAllEqual(params_v, params_v_restored)

  def _build_forward_cudnn_model(self,
                                 rnn_mode,
                                 num_layers,
                                 num_units,
                                 input_data,
                                 is_training=False):
    input_data_shape = input_data.get_shape().with_rank(3)
    batch_size = input_data_shape[1].value
    input_size = input_data_shape[2].value
    model = self._CreateModel(rnn_mode, num_layers, num_units, input_size)

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
    self._create_params_savable(params, model)

    return output_tuple, model, params

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
    for rnn, cfg, use_block_cell in itertools.product(
        (cudnn_rnn_ops.CUDNN_LSTM,), configs, (True, False,)):
      self._testCudnnCompatibleRnnCells(cfg["num_layers"], cfg["seq_length"],
                                        cfg["num_units"], cfg["input_size"],
                                        cfg["batch_size"], rnn, use_block_cell)
    # TODO(jamesqin): Add CudnnCompatibleGRUBlockCell.
    for rnn, cfg, use_block_cell in itertools.product(
        (cudnn_rnn_ops.CUDNN_GRU,), configs, (False,)):
      self._testCudnnCompatibleRnnCells(cfg["num_layers"], cfg["seq_length"],
                                        cfg["num_units"], cfg["input_size"],
                                        cfg["batch_size"], rnn, use_block_cell)

  def _testCudnnCompatibleRnnCells(self, num_layers, seq_length, num_units,
                                   input_size, batch_size, rnn_mode,
                                   use_block_cell):
    has_state_c = rnn_mode == cudnn_rnn_ops.CUDNN_LSTM
    np.random.seed(0)
    # Train graph
    with ops.Graph().as_default():
      random_seed.set_random_seed(299)
      input_data = array_ops.placeholder(
          dtypes.float32, shape=[seq_length, batch_size, input_size])
      output_tuple, cudnn_model, cudnn_params = self._build_forward_cudnn_model(
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
      (cudnn_output_tuple, cudnn_model,
       cudnn_params) = self._build_forward_cudnn_model(
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
      (output, states) = _create_cudnn_compatible_canonical_rnn(
          cudnn_model, cell_inputs, use_block_cell)
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

  def _testSaveRestoreOutput(self, rnn_mode):
    num_layers = 2
    num_units = 7
    input_size = 7
    seq_length = 10
    batch_size = 5
    dir_count = 1
    model = self._CreateModel(rnn_mode, num_layers, num_units, input_size)
    params_size_t = model.params_size()
    params = variables.Variable(
        array_ops.ones([params_size_t]), validate_shape=False)
    self._create_params_savable(params, model)
    save_path = os.path.join(self.get_temp_dir(), "save-restore-output-test")
    saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

    has_input_c = (rnn_mode == cudnn_rnn_ops.CUDNN_LSTM)
    input_data = array_ops.ones([seq_length, batch_size, input_size])
    input_h = array_ops.ones([num_layers * dir_count, batch_size, num_units])
    if has_input_c:
      input_c = array_ops.ones([num_layers * dir_count, batch_size, num_units])
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
    with self.test_session(use_gpu=True) as sess:
      sess.run(variables.global_variables_initializer())
      total_sum_v = sess.run(total_sum)
      val = saver.save(sess, save_path)
      self.assertEqual(save_path, val)
    with self.test_session(use_gpu=True) as sess:
      reset_params = state_ops.assign(params, array_ops.zeros([params_size_t]))
      sess.run(reset_params)
      saver.restore(sess, save_path)
      total_sum_v_restored = sess.run(total_sum)
      self.assertAllEqual(total_sum_v, total_sum_v_restored)

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSaveRestore(self):
    rnn_modes = [
        cudnn_rnn_ops.CUDNN_LSTM, cudnn_rnn_ops.CUDNN_GRU,
        cudnn_rnn_ops.CUDNN_RNN_TANH, cudnn_rnn_ops.CUDNN_RNN_RELU
    ]
    for rnn_mode in rnn_modes:
      self._testSaveRestoreVariable(rnn_mode)
      self._testSaveRestoreOutput(rnn_mode)

  def _MinLSTMParamSize(self,
                        num_layers,
                        num_units,
                        input_size,
                        input_mode="auto_select",
                        direction=cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION):
    if direction != cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION:
      # TODO(zhengxq): support bidirection in parameter size estimate.
      raise ValueError("Only unidirection in parameter size estimate")
    first_layer_weights = 4 * num_units * (num_units + input_size)
    higher_layer_weights = 8 * (num_layers - 1) * num_units * num_units
    all_biases = 8 * num_layers * num_units
    return first_layer_weights + higher_layer_weights + all_biases

  def _testOneLSTMParamsSize(self, num_layers, num_units, input_size):
    min_params_size = self._MinLSTMParamSize(num_layers, num_units, input_size)
    model = self._CreateModel(cudnn_rnn_ops.CUDNN_LSTM, num_layers, num_units,
                              input_size)
    params_size = model.params_size()
    with self.test_session(use_gpu=True) as sess:
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
    with ops.Graph().as_default():
      for (num_layers, num_units, input_size) in test_configs:
        self._testOneLSTMParamsSize(num_layers, num_units, input_size)

  def _testOneSimpleInference(self, rnn_mode, num_layers, num_units, input_size,
                              batch_size, seq_length, dir_count, dropout,
                              expected, tolerance):
    random_seed.set_random_seed(5678)
    model = self._CreateModel(
        rnn_mode,
        num_layers,
        num_units,
        input_size,
        input_mode="auto_select",
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
    # Cudnn scales result for dropout during training, therefore dropout has no
    # impact for inference results.
    # (lstm, gru, rnn_tanh are saturated in the test. rnn_relu case is most
    # demonstrative of the dropout-invariant nature of CudnnRnn.)
    test_configs = [
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_LSTM,
            "dropout": [0., 0.5, 1.],
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
            "dropout": [0., 0.5, 1.],
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
            "dropout": [0., 0.5, 1.],
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
            "dropout": [0., 0.5, 1.],
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
    with ops.Graph().as_default():
      for config in test_configs:
        rnn_mode = config["rnn_mode"]
        dropout_list = config.get("dropout", [0.])
        expected = config["expected"]
        tolerance = config["tolerance"]
        shape = config["shape"]
        for dropout in dropout_list:
          self._testOneSimpleInference(
              rnn_mode, shape["num_layers"], shape["num_units"],
              shape["input_size"], shape["batch_size"], shape["seq_length"],
              shape["dir_count"], dropout, expected, tolerance)

  def _testOneSimpleTraining(self, rnn_mode, num_layers, num_units, input_size,
                             batch_size, seq_length, dir_count, dropout,
                             tolerance):
    # Gradient checking runs two forward ops with almost the same input. Need to
    # make sure the drop patterns across the two runs are the same.
    old_env_state = os.environ.get("TF_CUDNN_RESET_RND_GEN_STATE", str(False))
    os.environ["TF_CUDNN_RESET_RND_GEN_STATE"] = str(True)
    has_input_c = (rnn_mode == cudnn_rnn_ops.CUDNN_LSTM)
    random_seed.set_random_seed(1234)
    model = self._CreateModel(
        rnn_mode, num_layers, num_units, input_size, dropout=dropout)
    params_size_t = model.params_size()
    input_data = variables.Variable(
        random_ops.random_uniform([seq_length, batch_size, input_size]))
    input_h = variables.Variable(
        random_ops.random_uniform(
            [num_layers * dir_count, batch_size, num_units]))
    params = variables.Variable(
        random_ops.random_uniform([params_size_t]), validate_shape=False)
    if has_input_c:
      input_c = variables.Variable(
          random_ops.random_uniform(
              [num_layers * dir_count, batch_size, num_units]))

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

    with self.test_session(use_gpu=True) as sess:
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

      err = gradient_checker.compute_gradient_error(all_inputs, all_shapes,
                                                    total_sum, [1])

      self.assertLess(err, tolerance)
      os.environ["TF_CUDNN_RESET_RND_GEN_STATE"] = old_env_state

  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def testSimpleTraining(self):
    test_configs = [
        {
            "rnn_mode": cudnn_rnn_ops.CUDNN_LSTM,
            "dropout": [0., 0.5, 1.],
            "tolerance": 1e-2,
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
            "dropout": [0., 0.5, 1.],
            "tolerance": 4e-3,
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
            "dropout": [0., 0.5, 1.],
            "tolerance": 5e-3,
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
            "dropout": [0., 0.5, 1.],
            "tolerance": 4e-1,
            "shape": {
                "num_layers": 2,
                "num_units": 3,
                "input_size": 4,
                "batch_size": 3,
                "seq_length": 4,
                "dir_count": 1,
            },
        },
    ]
    ops.reset_default_graph()
    with ops.Graph().as_default():
      for config in test_configs:
        rnn_mode = config["rnn_mode"]
        dropout_list = config.get("dropout", [0.])
        tolerance = config["tolerance"]
        shape = config["shape"]
        for dropout in dropout_list:
          self._testOneSimpleTraining(rnn_mode, shape["num_layers"],
                                      shape["num_units"], shape["input_size"],
                                      shape["batch_size"], shape["seq_length"],
                                      shape["dir_count"], dropout, tolerance)


if __name__ == "__main__":
  googletest.main()
