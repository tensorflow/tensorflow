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

import collections
import itertools
import os
import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
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


def RunLSTM(sess,
            num_units,
            input_size,
            batch_size,
            time,
            num_layers=1,
            is_training=True,
            dropout=0.,
            num_dirs=True,
            dtype=dtypes.float32):
  # TODO(jamesqin): add multi-layer tests.
  # TODO(jamesqin): add multi-dir tests
  assert num_layers == 1
  assert num_dirs == 1
  if is_training and not np.isclose(dropout, 0):
    raise ValueError("dropout can not be 0. when test training.")

  # set graph level random seed and numpy random seed.
  random_seed.set_random_seed(0)
  np.random.seed(0)

  inputs = variable_scope.get_variable(
      "inputs",
      initializer=np.random.rand(time, batch_size,
                                 input_size).astype(dtype.as_numpy_dtype),
      dtype=dtype)
  initial_h_op = variable_scope.get_variable(
      "initial_h_op",
      initializer=np.random.rand(batch_size,
                                 num_units).astype(dtype.as_numpy_dtype),
      dtype=dtype)
  initial_c_op = variable_scope.get_variable(
      "initial_c_op",
      initializer=np.random.rand(batch_size,
                                 num_units).astype(dtype.as_numpy_dtype),
      dtype=dtype)

  initializer = init_ops.random_uniform_initializer(
      -0.01, 0.01, dtype=dtype, seed=19980904)

  with variable_scope.variable_scope("test", initializer=initializer):
    w = variable_scope.get_variable(
        "rnn/lstm_cell/kernel",
        shape=[input_size + num_units, num_units * 4],
        dtype=dtype)
    b = variable_scope.get_variable(
        "rnn/lstm_cell/bias", shape=[num_units * 4], dtype=dtype)

    # canonical lstm. must set forget_bias to 0. to align with cudnn lstm.
    cell = rnn_cell_impl.LSTMCell(num_units, forget_bias=0., reuse=True)
    outputs_op, state_tuple_op = rnn.dynamic_rnn(
        cell,
        inputs,
        initial_state=rnn_cell_impl.LSTMStateTuple(
            h=initial_h_op, c=initial_c_op),
        dtype=dtype,
        time_major=True,
        scope=None)

  # Convert to cudnn opaque param.
  format_converter = cudnn_rnn_ops.CudnnParamsFormatConverterLSTM(
      num_layers, num_units, input_size)
  opaque_params = format_converter.tf_canonical_to_opaque([w, b])

  cu_initial_h_op = array_ops.expand_dims(initial_h_op, axis=0)
  cu_initial_c_op = array_ops.expand_dims(initial_c_op, axis=0)
  cu_outputs_op, cu_h_op, cu_c_op = cudnn_rnn_ops._cudnn_rnn(
      inputs,
      cu_initial_h_op,
      cu_initial_c_op,
      opaque_params,
      dropout=dropout,
      is_training=is_training,
      rnn_mode=cudnn_rnn_ops.CUDNN_LSTM)
  # Remove the trivial 1st dimension.
  cu_state_tuple_op = rnn_cell_impl.LSTMStateTuple(
      c=array_ops.squeeze(cu_c_op, axis=0),
      h=array_ops.squeeze(cu_h_op, axis=0))

  if is_training:
    (inp_grad_op, hgrad_op,
     cgrad_op, wgrad_op, bgrad_op) = gradients_impl.gradients(
         outputs_op, [inputs, initial_h_op, initial_c_op, w, b])

    (cu_inp_grad_op, cu_hgrad_op,
     cu_cgrad_op, opaque_grad_op) = gradients_impl.gradients(
         cu_outputs_op,
         [inputs, cu_initial_h_op, cu_initial_c_op, opaque_params])
    # Remove the trivial 1st dimension
    cu_hgrad_op = array_ops.squeeze(cu_hgrad_op, axis=0)
    # Remove the trivial 1st dimension
    cu_cgrad_op = array_ops.squeeze(cu_cgrad_op, axis=0)

    cu_wgrad_op, cu_bgrad_op = format_converter.opaque_to_tf_canonical(
        opaque_grad_op)
    cu_wgrad_op = cu_wgrad_op[0]
    cu_bgrad_op = cu_bgrad_op[0]
    # cudnn lstm has 2 biases each gate. When converting to tf canonical format,
    # the two biases are summed into one. Thus here bias gradient should be
    # halved when comparing with tf lstm.
    cu_bgrad_op *= 0.5

  init_op = variables.global_variables_initializer()
  sess.run(init_op)

  if is_training:
    outputs, state_tuple, inp_grad, state_grad, wgrad, bgrad = sess.run([
        outputs_op, state_tuple_op, inp_grad_op,
        (hgrad_op, cgrad_op), wgrad_op, bgrad_op
    ])
    (cu_outputs, cu_state_tuple, cu_inp_grad, cu_state_grad, cu_wgrad,
     cu_bgrad) = sess.run([
         cu_outputs_op, cu_state_tuple_op, cu_inp_grad_op,
         (cu_hgrad_op, cu_cgrad_op), cu_wgrad_op, cu_bgrad_op
     ])

    logging.vlog(1, "outputs: %s" % outputs)
    logging.vlog(1, "cu_outputs: %s" % cu_outputs)
    logging.vlog(1, "state_tuple: %s" % str(state_tuple))
    logging.vlog(1, "cu_state_tuple: %s" % str(cu_state_tuple))
    logging.vlog(1, "inp_grad: %s" % inp_grad)
    logging.vlog(1, "cu_inp_grad: %s" % cu_inp_grad)
    logging.vlog(1, "state_grad: %s" % str(state_grad))
    logging.vlog(1, "cu_state_grad: %s" % str(cu_state_grad))
    logging.vlog(1, "wgrad: %s" % str(wgrad))
    logging.vlog(1, "bgrad: %s" % str(bgrad))
    logging.vlog(1, "cu_wgrad: %s" % str(cu_wgrad))
    logging.vlog(1, "cu_bgrad: %s" % str(cu_bgrad))
    return (outputs, cu_outputs, state_tuple, cu_state_tuple, inp_grad,
            cu_inp_grad, state_grad, cu_state_grad, wgrad, bgrad, cu_wgrad,
            cu_bgrad)
  else:
    outputs, state_tuple = sess.run([outputs_op, state_tuple_op])
    cu_outputs, cu_state_tuple = sess.run([cu_outputs_op, cu_state_tuple_op])

    logging.vlog(1, "outputs: %s" % outputs)
    logging.vlog(1, "cu_outputs: %s" % cu_outputs)
    logging.vlog(1, "state_tuple: %s" % str(state_tuple))
    logging.vlog(1, "cu_state_tuple: %s" % str(cu_state_tuple))
  return outputs, cu_outputs, state_tuple, cu_state_tuple


# Basic set of RNN configs to test. They can be further extended in relevant
# test (e.g. adding num_dirs).
NAMED_RNN_TESTCASES = ({
    "testcase_name": "xsmall",
    "num_units": 1,
    "input_size": 1,
    "batch_size": 1,
    "time": 1,
    "num_layers": 1,
}, {
    "testcase_name": "small",
    "num_units": 4,
    "input_size": 4,
    "batch_size": 4,
    "time": 4,
    "num_layers": 1,
}, {
    "testcase_name": "medium",
    "num_units": 128,
    "input_size": 64,
    "batch_size": 8,
    "time": 16,
    "num_layers": 1,
}, {
    "testcase_name": "large",
    "num_units": 128,
    "input_size": 128,
    "batch_size": 16,
    "time": 32,
    "num_layers": 1,
})


def ExpandNamedTestCases(inputs, *remove_keys, **extra_configs):
  """Expands testcase with new config dimensions.

  Example:
    inputs = (
      {'testcase_name': 'test1', 'gender': 'male'}
      {'testcase_name': 'test2', 'gender': 'female'}
    )
    remove_keys:  empty
    extra_configs = {
      'age': [40, 80]
      'height': [5, 6]
    }

    Returns:
      (
        {'testcase_name': 'test1_age_40_height_5','gender': 'male', 'age':
        40,'height': 5}
        {'testcase_name': 'test1_age_40_height_6', 'gender': 'male', 'age': 40,
        'height': 6}
        {'testcase_name': 'test1_age_80_height_5', 'gender': 'male', 'age': 80,
        'height': 5}
        {'testcase_name': 'test1_age_80_height_6', 'gender': 'male', 'age': 80,
        'height': 6}

        {'testcase_name': 'test2_age_40_height_5', 'gender': 'female', 'age':
        40,
        'height': 5}
        {'testcase_name': 'test2_age_40_height_6', 'gender': 'female', 'age':
        40,
        'height': 6}
        {'testcase_name': 'test2_age_80_height_5', 'gender': 'female', 'age':
        80,
        'height': 5}
        {'testcase_name': 'test2_age_80_height_6', 'gender': 'female', 'age':
        80,
        'height': 6}
      )

  Args:
    inputs: A list of dictionary, each being a testcase.
    *remove_keys: A list of keys into testcase which are not needed in new
      testcases.
    **extra_configs: A dict of new test dimension and applicable values in that
      dimension.

  Returns:
    A list of dictionary with expanded test cases.
  """
  res = []
  ordered_extra_configs = collections.OrderedDict(extra_configs)
  keys = ordered_extra_configs.keys()
  # A list of list of configs.
  # The outer loop is iterating keys, the innner is values of one key.
  combined_kv = [[(k, v) for v in ordered_extra_configs[k]] for k in keys]
  logging.info("combined_kv: %s", combined_kv)

  for inp in inputs:
    # Each inp is a dict
    for config in itertools.product(*combined_kv):
      new_inp = dict(inp)
      # config is a list in the form of [(k_i, v_j), (k_p, v_q), ...]
      suffix = ["%s_%s" % (p[0], str(p[1])) for p in config]
      suffix = "_".join(suffix)
      new_inp["testcase_name"] += "_" + suffix
      for k, v in config:
        new_inp[k] = v
      # Remove not used keys from the new test case.
      if remove_keys:
        if not isinstance(remove_keys, (list, tuple)):
          remove_keys = [remove_keys]
        for k in remove_keys:
          new_inp.pop(k, None)
      logging.info("new_inp: %s", new_inp)
      res.append(new_inp)
  # Dedup, necessary if `remove_keys` is set.
  return [dict(t) for t in {tuple(d.items()) for d in res}]


class CudnnLSTMTest(TensorFlowTestCase, parameterized.TestCase):

  def _test_training_helper(self,
                            num_units,
                            input_size,
                            batch_size,
                            time,
                            num_layers,
                            dtype,
                            rtol=2e-6,
                            atol=2e-6):
    with self.session(use_gpu=True) as sess:
      (outputs, cu_outputs, state_tuple, cu_state_tuple, inp_grad, cu_inp_grad,
       state_grad, cu_state_grad, wgrad, bgrad, cu_wgrad, cu_bgrad) = RunLSTM(
           sess, num_units, input_size, batch_size, time, num_layers)

      self.assertAllClose(outputs, cu_outputs, rtol=rtol, atol=atol)
      for s, cu_s in zip(state_tuple, cu_state_tuple):
        self.assertAllClose(s, cu_s, rtol=rtol, atol=atol)
      for sg, cu_sg in zip(state_grad, cu_state_grad):
        self.assertAllClose(sg, cu_sg, rtol=rtol, atol=atol)
      self.assertAllClose(inp_grad, cu_inp_grad, rtol=rtol, atol=atol)
      self.assertAllClose(bgrad, cu_bgrad, rtol=rtol, atol=atol)
      self.assertAllClose(wgrad, cu_wgrad, rtol=rtol, atol=atol)

  @parameterized.named_parameters(*NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_training(self, num_units, input_size, batch_size, time, num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    self._test_training_helper(num_units, input_size, batch_size, time,
                               num_layers, dtypes.float32)

  @parameterized.named_parameters(*NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_training_fp16(self, num_units, input_size, batch_size, time,
                         num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    self._test_training_helper(
        num_units,
        input_size,
        batch_size,
        time,
        num_layers,
        dtypes.float16,
        rtol=5e-3,
        atol=5e-4)

  @parameterized.named_parameters(*NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_inference(self, num_units, input_size, batch_size, time, num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    with self.session(use_gpu=True) as sess:
      (outputs, cu_outputs, state_tuple, cu_state_tuple) = RunLSTM(
          sess,
          num_units,
          input_size,
          batch_size,
          time,
          num_layers,
          is_training=False)

      self.assertAllClose(outputs, cu_outputs)
      # h
      self.assertAllClose(state_tuple.h, cu_state_tuple.h)
      # c
      self.assertAllClose(state_tuple.c, cu_state_tuple.c)

  @parameterized.named_parameters(*NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_inference_fp16(self, num_units, input_size, batch_size, time,
                          num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    with self.session(use_gpu=True) as sess:
      (outputs, cu_outputs, state_tuple, cu_state_tuple) = RunLSTM(
          sess,
          num_units,
          input_size,
          batch_size,
          time,
          num_layers,
          is_training=False,
          dtype=dtypes.float16)

      rtol, atol = 5e-3, 5e-4
      self.assertAllClose(outputs, cu_outputs, rtol=rtol, atol=atol)
      # h
      self.assertAllClose(
          state_tuple.h, cu_state_tuple.h, rtol=rtol, atol=atol)
      # c
      self.assertAllClose(
          state_tuple.c, cu_state_tuple.c, rtol=rtol, atol=atol)

  @parameterized.named_parameters(*NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_inference_with_dropout(self, num_units, input_size, batch_size, time,
                                  num_layers):
    """Validates that dropout does not affect Cudnn Rnn inference."""
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    # Hand-picked dropouts are used below (0. and 1.)
    with ops.Graph().as_default() as g:
      with self.session(use_gpu=True, graph=g) as sess:
        # 1st time w/o dropout.
        (_, cu_outputs, _, cu_state_tuple) = RunLSTM(
            sess,
            num_units,
            input_size,
            batch_size,
            time,
            num_layers,
            is_training=False,
            dropout=0.)

    with ops.Graph().as_default() as g:
      with self.session(use_gpu=True, graph=g) as sess:
        (_, cu_outputs2, _, cu_state_tuple2) = RunLSTM(
            sess,
            num_units,
            input_size,
            batch_size,
            time,
            num_layers,
            is_training=False,
            dropout=1.)

    self.assertAllClose(cu_outputs, cu_outputs2)
    # h
    self.assertAllClose(cu_state_tuple.h, cu_state_tuple2.h)
    # c
    self.assertAllClose(cu_state_tuple.c, cu_state_tuple2.c)


def RunGRU(sess,
           num_units,
           input_size,
           batch_size,
           time,
           num_layers=1,
           is_training=True,
           dropout=0.,
           num_dirs=True,
           dtype=dtypes.float32):
  # TODO(jamesqin): add multi-layer tests.
  # TODO(jamesqin): add multi-dir tests
  assert num_layers == 1
  assert num_dirs == 1
  if is_training and not np.isclose(dropout, 0):
    raise ValueError("dropout can not be 0. when test training.")

  # set graph level random seed and numpy random seed.
  random_seed.set_random_seed(0)
  np.random.seed(0)

  inputs = variable_scope.get_variable(
      "inputs",
      initializer=np.random.rand(time, batch_size,
                                 input_size).astype(dtype.as_numpy_dtype),
      dtype=dtype)
  initial_h_op = variable_scope.get_variable(
      "initial_h_op",
      initializer=np.random.rand(batch_size,
                                 num_units).astype(dtype.as_numpy_dtype),
      dtype=dtype)

  initializer = init_ops.random_uniform_initializer(
      -0.01, 0.01, dtype=dtype, seed=19980904)
  with variable_scope.variable_scope("test", initializer=initializer):
    gate_kernel = variable_scope.get_variable(
        "rnn/cudnn_compatible_gru_cell/gates/kernel",
        shape=[input_size + num_units, num_units * 2],
        dtype=dtype)
    gate_bias = variable_scope.get_variable(
        "rnn/cudnn_compatible_gru_cell/gates/bias",
        shape=[num_units * 2],
        dtype=dtype)
    candidate_inp_kernel = variable_scope.get_variable(
        "rnn/cudnn_compatible_gru_cell/candidate/input_projection/kernel",
        shape=[input_size, num_units],
        dtype=dtype)
    candidate_inp_bias = variable_scope.get_variable(
        "rnn/cudnn_compatible_gru_cell/candidate/input_projection/bias",
        shape=[num_units],
        dtype=dtype)
    candidate_hid_kernel = variable_scope.get_variable(
        "rnn/cudnn_compatible_gru_cell/candidate/hidden_projection/kernel",
        shape=[num_units, num_units],
        dtype=dtype)
    candidate_hid_bias = variable_scope.get_variable(
        "rnn/cudnn_compatible_gru_cell/candidate/hidden_projection/bias",
        shape=[num_units],
        dtype=dtype)

    cell = cudnn_rnn_ops.CudnnCompatibleGRUCell(num_units, reuse=True)
    outputs_op, h_op = rnn.dynamic_rnn(
        cell,
        inputs,
        initial_state=initial_h_op,
        dtype=dtype,
        time_major=True,
        scope=None)

  ws = [gate_kernel, candidate_inp_kernel, candidate_hid_kernel]
  bs = [gate_bias, candidate_inp_bias, candidate_hid_bias]
  # Convert to cudnn opaque param.
  format_converter = cudnn_rnn_ops.CudnnParamsFormatConverterGRU(
      num_layers, num_units, input_size)
  opaque_params = format_converter.tf_canonical_to_opaque(ws + bs)

  cu_initial_h_op = array_ops.expand_dims(initial_h_op, axis=0)
  cu_outputs_op, cu_h_op, _ = cudnn_rnn_ops._cudnn_rnn(
      inputs,
      cu_initial_h_op,
      array_ops.zeros_like(cu_initial_h_op),  # not used
      opaque_params,
      dropout=dropout,
      is_training=is_training,
      rnn_mode=cudnn_rnn_ops.CUDNN_GRU)

  if is_training:
    (inp_grad_op, hgrad_op, gk_grad_op, cik_grad_op, chk_grad_op, gb_grad_op,
     cib_grad_op, chb_grad_op) = gradients_impl.gradients(
         outputs_op, [inputs, initial_h_op] + ws + bs)

    (cu_inp_grad_op, cu_hgrad_op, opaque_grad_op) = gradients_impl.gradients(
        cu_outputs_op, [inputs, cu_initial_h_op, opaque_params])
    # Remove the trivial 1st dimension
    cu_hgrad_op = array_ops.squeeze(cu_hgrad_op, axis=0)

    cu_wgrad_op, cu_bgrad_op = format_converter.opaque_to_tf_canonical(
        opaque_grad_op)
    (cu_gk_grad_op, cu_cik_grad_op, cu_chk_grad_op) = cu_wgrad_op
    (cu_gb_grad_op, cu_cib_grad_op, cu_chb_grad_op) = cu_bgrad_op
    # cudnn gru has 2 biases for reset and update gates. When converting to tf
    # canonical format, the two biases are summed into one.  Thus here relevant
    # bias gradient should be halved before comparing with tf gru.
    cu_gb_grad_op *= 0.5

  init_op = variables.global_variables_initializer()
  sess.run(init_op)

  if is_training:
    outputs, h, inp_grad, hgrad, wgrad, bgrad = sess.run([
        outputs_op, h_op, inp_grad_op, hgrad_op,
        (gk_grad_op, cik_grad_op, chk_grad_op),
        (gb_grad_op, cib_grad_op, chb_grad_op)
    ])
    (cu_outputs, cu_h, cu_inp_grad, cu_hgrad, cu_wgrad, cu_bgrad) = sess.run([
        cu_outputs_op, cu_h_op, cu_inp_grad_op, cu_hgrad_op,
        (cu_gk_grad_op, cu_cik_grad_op, cu_chk_grad_op),
        (cu_gb_grad_op, cu_cib_grad_op, cu_chb_grad_op)
    ])
    # Remove the trivial 1st dimension
    cu_h = np.squeeze(cu_h, axis=0)

    logging.vlog(1, "outputs: %s" % outputs)
    logging.vlog(1, "cu_outputs: %s" % cu_outputs)
    logging.vlog(1, "h: %s" % h)
    logging.vlog(1, "cu_h: %s" % h)
    logging.vlog(1, "inp_grad: %s" % inp_grad)
    logging.vlog(1, "cu_inp_grad: %s" % cu_inp_grad)
    logging.vlog(1, "hgrad: %s" % hgrad)
    logging.vlog(1, "cu_hgrad: %s" % cu_hgrad)
    logging.vlog(1, "wgrad: %s" % str(wgrad))
    logging.vlog(1, "bgrad: %s" % str(bgrad))
    logging.vlog(1, "cu_wgrad: %s" % str(cu_wgrad))
    logging.vlog(1, "cu_bgrad: %s" % str(cu_bgrad))
    return (outputs, cu_outputs, h, cu_h, inp_grad, cu_inp_grad, hgrad,
            cu_hgrad, wgrad, bgrad, cu_wgrad, cu_bgrad)
  else:
    outputs, h = sess.run([outputs_op, h_op])
    cu_outputs, cu_h = sess.run([cu_outputs_op, cu_h_op])
    # Remove the trivial 1st dimension.
    cu_h = np.squeeze(cu_h, axis=0)

    logging.vlog(1, "outputs: %s" % outputs)
    logging.vlog(1, "cu_outputs: %s" % cu_outputs)
    logging.vlog(1, "h: %s" % h)
    logging.vlog(1, "cu_h: %s" % h)
  return outputs, cu_outputs, h, cu_h


class CudnnGRUTest(TensorFlowTestCase, parameterized.TestCase):

  def _test_training_helper(self,
                            num_units,
                            input_size,
                            batch_size,
                            time,
                            num_layers,
                            dtype,
                            rtol=2e-6,
                            atol=2e-6):
    with self.session(use_gpu=True) as sess:
      (outputs, cu_outputs, h, cu_h, inp_grad, cu_inp_grad, hgrad,
       cu_hgrad, wgrad, bgrad, cu_wgrad, cu_bgrad) = RunGRU(
           sess, num_units, input_size, batch_size, time, num_layers)

      self.assertAllClose(outputs, cu_outputs, rtol=rtol, atol=atol)
      self.assertAllClose(h, cu_h, rtol=rtol, atol=atol)
      self.assertAllClose(hgrad, cu_hgrad, rtol=rtol, atol=atol)
      self.assertAllClose(inp_grad, cu_inp_grad, rtol=rtol, atol=atol)
      for bg, cu_bg in zip(bgrad, cu_bgrad):
        self.assertAllClose(bg, cu_bg, rtol=rtol, atol=atol)
      for wg, cu_wg in zip(wgrad, cu_wgrad):
        self.assertAllClose(wg, cu_wg, rtol=rtol, atol=atol)

  @parameterized.named_parameters(*NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_training(self, num_units, input_size, batch_size, time, num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    self._test_training_helper(num_units, input_size, batch_size, time,
                               num_layers, dtypes.float32)

  @parameterized.named_parameters(*NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_training_fp16(self, num_units, input_size, batch_size, time,
                         num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    self._test_training_helper(
        num_units,
        input_size,
        batch_size,
        time,
        num_layers,
        dtypes.float16,
        rtol=5e-3,
        atol=5e-4)

  @parameterized.named_parameters(*NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_inference(self, num_units, input_size, batch_size, time, num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    with self.session(use_gpu=True) as sess:
      (outputs, cu_outputs, h, cu_h) = RunGRU(
          sess,
          num_units,
          input_size,
          batch_size,
          time,
          num_layers,
          is_training=False)
      self.assertAllClose(outputs, cu_outputs)
      self.assertAllClose(h, cu_h)

  @parameterized.named_parameters(*NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_inference_fp16(self, num_units, input_size, batch_size, time,
                          num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    with self.session(use_gpu=True) as sess:
      (outputs, cu_outputs, h, cu_h) = RunGRU(
          sess,
          num_units,
          input_size,
          batch_size,
          time,
          num_layers,
          is_training=False,
          dtype=dtypes.float16)

      rtol, atol = 5e-3, 5e-4
      self.assertAllClose(outputs, cu_outputs, rtol=rtol, atol=atol)
      self.assertAllClose(h, cu_h, rtol=rtol, atol=atol)

  @parameterized.named_parameters(*NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_inference_with_dropout(self, num_units, input_size, batch_size, time,
                                  num_layers):
    """Validates that dropout does not affect Cudnn Rnn inference."""
    # Hand-picked dropouts are used below (0. and 1.)
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    with ops.Graph().as_default() as g:
      with self.session(use_gpu=True, graph=g) as sess:
        # 1st time w/o dropout.
        (_, cu_outputs, _, cu_h) = RunGRU(
            sess,
            num_units,
            input_size,
            batch_size,
            time,
            num_layers,
            is_training=False,
            dropout=0.)

    with ops.Graph().as_default() as g:
      with self.session(use_gpu=True, graph=g) as sess:
        (_, cu_outputs2, _, cu_h2) = RunGRU(
            sess,
            num_units,
            input_size,
            batch_size,
            time,
            num_layers,
            is_training=False,
            dropout=1.)

    self.assertAllClose(cu_outputs, cu_outputs2)
    self.assertAllClose(cu_h[0], cu_h2[0])


class CudnnParamsFormatConverterTest(TensorFlowTestCase,
                                     parameterized.TestCase):
  """Class for testing various format converters."""

  def _test_lstm_helper(self, num_units, input_size, num_layers, direction):
    with self.session(use_gpu=True) as sess:
      random_seed.set_random_seed(0)
      np.random.seed(0)

      num_dirs = 1 if direction == cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION else 2
      format_converter = cudnn_rnn_ops.CudnnParamsFormatConverterLSTM(
          num_layers, num_units, input_size, direction=direction)

      ws, bs = [], []
      for _ in range(num_layers * num_dirs):
        w = constant_op.constant(
            np.random.rand(input_size + num_units, 4 * num_units),
            dtype=dtypes.float32)
        b = constant_op.constant(
            np.random.rand(4 * num_units), dtype=dtypes.float32)
        ws.append(w)
        bs.append(b)

      opaque_params = format_converter.tf_canonical_to_opaque(ws + bs)
      opaque_params_size = cudnn_rnn_ops.cudnn_rnn_opaque_params_size(
          cudnn_rnn_ops.CUDNN_LSTM,
          num_layers,
          num_units,
          input_size,
          direction=direction)

      ws_r, bs_r = format_converter.opaque_to_tf_canonical(opaque_params)

      # Test tf_canonical_to_opaque() followed by opaque_to_tf_canonical()
      # returns the original input.
      ws, ws_r, bs, bs_r = sess.run([ws, ws_r, bs, bs_r])
      for w, w_r in zip(ws, ws_r):
        self.assertAllClose(w, w_r)
      for b, b_r in zip(bs, bs_r):
        self.assertAllClose(b, b_r)

      # Test opaque_params size lower bound
      opaque_params_size_v = sess.run(opaque_params_size)
      min_params_size = sum(x.size for x in ws) + np.sum(x.size for x in bs)
      logging.info("min_parm_size: %d vs actual_opaque_param_size: %d",
                   min_params_size, opaque_params_size_v)
      self.assertLessEqual(min_params_size, opaque_params_size_v)

  @parameterized.named_parameters((c["testcase_name"], c["num_units"],
                                   c["input_size"], c["num_layers"])
                                  for c in NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_lstm(self, num_units, input_size, num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    self._test_lstm_helper(num_units, input_size, num_layers,
                           cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION)

  @parameterized.named_parameters((c["testcase_name"], c["num_units"],
                                   c["input_size"], c["num_layers"])
                                  for c in NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_lstm_bidi(self, num_units, input_size, num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    self._test_lstm_helper(num_units, input_size, num_layers,
                           cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION)

  def _test_gru_helper(self, num_units, input_size, num_layers, direction):
    with self.session(use_gpu=True) as sess:
      random_seed.set_random_seed(0)
      np.random.seed(0)

      num_dirs = 1 if direction == cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION else 2
      format_converter = cudnn_rnn_ops.CudnnParamsFormatConverterGRU(
          num_layers, num_units, input_size, direction=direction)

      ws, bs = [], []
      for _ in range(num_layers * num_dirs):
        gate_kernel = constant_op.constant(
            np.random.rand(input_size + num_units, num_units * 2),
            dtype=dtypes.float32)
        gate_bias = constant_op.constant(
            np.random.rand(num_units * 2), dtype=dtypes.float32)
        candidate_inp_kernel = constant_op.constant(
            np.random.rand(input_size, num_units), dtype=dtypes.float32)
        candidate_inp_bias = constant_op.constant(
            np.random.rand(num_units), dtype=dtypes.float32)
        candidate_hid_kernel = constant_op.constant(
            np.random.rand(num_units, num_units), dtype=dtypes.float32)
        candidate_hid_bias = constant_op.constant(
            np.random.rand(num_units), dtype=dtypes.float32)
        ws.extend([gate_kernel, candidate_inp_kernel, candidate_hid_kernel])
        bs.extend([gate_bias, candidate_inp_bias, candidate_hid_bias])

      opaque_params = format_converter.tf_canonical_to_opaque(ws + bs)
      opaque_params_size = cudnn_rnn_ops.cudnn_rnn_opaque_params_size(
          cudnn_rnn_ops.CUDNN_GRU,
          num_layers,
          num_units,
          input_size,
          direction=direction)

      ws_r, bs_r = format_converter.opaque_to_tf_canonical(opaque_params)

      # Test tf_canonical_to_opaque() followed by opaque_to_tf_canonical()
      # returns the original input.
      ws, ws_r, bs, bs_r = sess.run([ws, ws_r, bs, bs_r])
      for w, w_r in zip(ws, ws_r):
        self.assertAllClose(w, w_r)
      for b, b_r in zip(bs, bs_r):
        self.assertAllClose(b, b_r)

      # Test opaque_params size lower bound
      opaque_params_size_v = sess.run(opaque_params_size)
      min_params_size = sum(x.size for x in ws) + sum(x.size for x in bs)
      logging.info("min_parm_size: %d vs actual_opaque_param_size: %d",
                   min_params_size, opaque_params_size_v)
      self.assertLessEqual(min_params_size, opaque_params_size_v)

  @parameterized.named_parameters((c["testcase_name"], c["num_units"],
                                   c["input_size"], c["num_layers"])
                                  for c in NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_gru(self, num_units, input_size, num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    self._test_gru_helper(num_units, input_size, num_layers,
                          cudnn_rnn_ops.CUDNN_RNN_UNIDIRECTION)

  @parameterized.named_parameters((c["testcase_name"], c["num_units"],
                                   c["input_size"], c["num_layers"])
                                  for c in NAMED_RNN_TESTCASES)
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_gru_bidi(self, num_units, input_size, num_layers):
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    self._test_gru_helper(num_units, input_size, num_layers,
                          cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION)


class CudnnRnnSaveRestoreTest(TensorFlowTestCase, parameterized.TestCase):
  """Class for testing various Cudnn Rnn SaveableObjects."""

  def _create_opaque_param(self,
                           rnn_mode,
                           num_units,
                           input_size,
                           num_layers,
                           direction,
                           name=None):
    param_size_t = cudnn_rnn_ops.cudnn_rnn_opaque_params_size(
        rnn_mode, num_layers, num_units, input_size, direction=direction)
    init_val = random_ops.random_uniform([param_size_t])
    return variable_scope.get_variable(
        name or "opaque_param", initializer=init_val, validate_shape=False)

  def _create_saveable(self, opaque_param, rnn_mode, num_units, input_size,
                       num_layers, direction):
    if rnn_mode == CUDNN_LSTM:
      fn = cudnn_rnn_ops.CudnnLSTMSaveable
    elif rnn_mode == CUDNN_GRU:
      fn = cudnn_rnn_ops.CudnnGRUSaveable
    elif rnn_mode == CUDNN_RNN_TANH:
      fn = cudnn_rnn_ops.CudnnRNNTanhSaveable
    elif rnn_mode == CUDNN_RNN_RELU:
      fn = cudnn_rnn_ops.CudnnRNNReluSaveable
    saveable = fn(
        opaque_param, num_layers, num_units, input_size, direction=direction)
    return saveable

  def _compare_weights(self, lhs, rhs):
    self.assertLen(rhs, len(lhs))
    for lw, rw in zip(lhs, rhs):
      self.assertAllEqual(lw, rw)

  def _compare_biases(self, lhs, rhs):
    self.assertLen(rhs, len(lhs))
    for lf, rt in zip(lhs, rhs):
      self.assertAllEqual(lf, rt)

  @parameterized.named_parameters(
      ExpandNamedTestCases(
          NAMED_RNN_TESTCASES, "time", "batch_size", **{
              "rnn_mode": [
                  CUDNN_LSTM, CUDNN_GRU, CUDNN_RNN_RELU, CUDNN_RNN_TANH
              ],
              "direction": [CUDNN_RNN_UNIDIRECTION, CUDNN_RNN_BIDIRECTION]
          }))
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_save_restore_variable(self, rnn_mode, num_units, input_size,
                                 num_layers, direction):
    # Verify the restored opaque param, once converted to tf_canonical format,
    # is the same as the tf canonicals of the pre-restored param.
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    with self.session(use_gpu=True) as sess:
      opaque_param = self._create_opaque_param(rnn_mode, num_units, input_size,
                                               num_layers, direction)
      saveable = self._create_saveable(opaque_param, rnn_mode, num_units,
                                       input_size, num_layers, direction)
      ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
      weights_op, biases_op = saveable.format_converter.opaque_to_tf_canonical(
          saveable._variables)

      save_path = os.path.join(self.get_temp_dir(), "save_restore_var_test")
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

      init_op = variables.global_variables_initializer()
      reset_op = state_ops.assign(opaque_param,
                                  array_ops.zeros_like(opaque_param))
      sess.run(init_op)
      self.assertEqual(save_path, saver.save(sess, save_path))

      # Get the tf canonical vals before reset-restore
      weights, biases = sess.run([weights_op, biases_op])

      # Reset the opaque param value
      sess.run(reset_op)
      # Assert reset happened.
      weights_z, biases_z = sess.run([weights_op, biases_op])
      for w in weights_z:
        self.assertAllClose(w, np.zeros_like(w))
      for b in biases_z:
        self.assertAllClose(b, np.zeros_like(b))

      # Restore opaque param value from checkpoint.
      saver.restore(sess, save_path)
      weights_r, biases_r = sess.run([weights_op, biases_op])
      self._compare_weights(weights, weights_r)
      self._compare_biases(biases, biases_r)

  @parameterized.named_parameters(
      ExpandNamedTestCases(
          NAMED_RNN_TESTCASES, "time", "batch_size", **{
              "rnn_mode": [
                  CUDNN_LSTM, CUDNN_GRU, CUDNN_RNN_RELU, CUDNN_RNN_TANH
              ],
              "direction": [CUDNN_RNN_UNIDIRECTION, CUDNN_RNN_BIDIRECTION]
          }))
  @unittest.skipUnless(test.is_built_with_cuda(),
                       "Test only applicable when running on GPUs")
  def test_save_restore_multi_variables(self, rnn_mode, num_units, input_size,
                                        num_layers, direction):
    # Verify the restored opaque param, once converted to tf_canonical format,
    # is the same as the tf canonicals of the pre-restored param.
    if not context.context().num_gpus():
      self.skipTest("No GPUs found")
    with self.session(use_gpu=True) as sess:
      opaque_params = []
      saveables = []
      num_opaque_params = 2
      for i in range(num_opaque_params):
        opaque_params.append(
            self._create_opaque_param(
                rnn_mode,
                num_units,
                input_size,
                num_layers,
                direction,
                name="opaque_param_%d" % i))
        saveable = self._create_saveable(opaque_params[i], rnn_mode, num_units,
                                         input_size, num_layers, direction)
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)
        saveables.append(saveable)

      weights_ops, biases_ops = [], []
      for i in range(num_opaque_params):
        weights_op, biases_op = (
            saveables[i].format_converter.opaque_to_tf_canonical(
                saveables[i]._variables))
        weights_ops.append(weights_op)
        biases_ops.append(biases_op)

      save_path = os.path.join(self.get_temp_dir(), "save_restore_var_test")
      saver = saver_lib.Saver(write_version=saver_pb2.SaverDef.V2)

      init_op = variables.global_variables_initializer()
      reset_ops = []
      for i in range(num_opaque_params):
        reset_ops.append(
            state_ops.assign(opaque_params[i],
                             array_ops.zeros_like(opaque_params[i])))
      sess.run(init_op)
      self.assertEqual(save_path, saver.save(sess, save_path))

      # Get the tf canonical vals before reset-restore
      for i in range(num_opaque_params):
        weights, biases = sess.run([weights_ops[i], biases_ops[i]])

        # Reset the opaque param value
        sess.run(reset_ops[i])

        # Assert reset happened.
        weights_z, biases_z = sess.run([weights_ops[i], biases_ops[i]])
        for w in weights_z:
          self.assertAllClose(w, np.zeros_like(w))
        for b in biases_z:
          self.assertAllClose(b, np.zeros_like(b))

        # Restore opaque param value from checkpoint.
        saver.restore(sess, save_path)
        weights_r, biases_r = sess.run([weights_ops[i], biases_ops[i]])
        self._compare_weights(weights, weights_r)
        self._compare_biases(biases, biases_r)


if __name__ == "__main__":
  googletest.main()
