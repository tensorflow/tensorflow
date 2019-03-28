# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the LSTM cell and layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.client import session as session_lib
from tensorflow.compiler.plugin.poplar.ops import gen_popnn_ops
from tensorflow.python.platform import googletest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent

dataType = np.float32
batch_size = 1
seq_len = 3
input_size = 5
num_channels = 8


def _get_variable(name, shape, initializer):
  return variable_scope.get_variable(
      name, shape=shape, initializer=initializer, dtype=dataType)


def _createLSTMInput(value, batch_size, seq_len, input_size):
  return np.full(
      fill_value=value,
      shape=[seq_len, batch_size, input_size],
      dtype=dataType)


def _createLSTMInitialState(h_value, c_value, batch_size, num_channels):
  return (np.full(
      fill_value=h_value, shape=[batch_size, num_channels], dtype=dataType),
          np.full(
              fill_value=c_value,
              shape=[batch_size, num_channels],
              dtype=dataType))


class LSTMTest(test.TestCase):
  def _LSTMLayerCPU(self, inputs, weights_value, initial_state, forget_bias,
                    training, name):
    with ops.device("/device:CPU:0"):
      lstm_cell = rnn_cell.LSTMCell(
          num_channels,
          name='basic_lstm_cell',
          forget_bias=forget_bias,
          initializer=init_ops.constant_initializer(
              weights_value, dtype=dataType),
          reuse=variable_scope.AUTO_REUSE)
      state = rnn_cell.LSTMStateTuple(initial_state[1], initial_state[0])
      outputs, states = rnn.dynamic_rnn(
          lstm_cell,
          inputs,
          dtype=dataType,
          initial_state=state,
          time_major=True)
      return outputs

  def _LSTMLayer(self, inputs, weights_value, initial_state, forget_bias,
                 training, name):
    with ops.device("/device:IPU:0"):
      with variable_scope.variable_scope("lstm_layer", use_resource=True):
        kernel = _get_variable(
            "kernel",
            shape=[input_size + num_channels, 4 * num_channels],
            initializer=init_ops.constant_initializer(weights_value, dataType))
        biases = _get_variable(
            "biases",
            shape=[4, num_channels],
            initializer=init_ops.constant_initializer(0.0, dataType))
      outputs, _, _, _ = gen_popnn_ops.popnn_lstm_layer(
          inputs=inputs,
          num_channels=num_channels,
          kernel=kernel,
          biases=biases,
          input_h_state=initial_state[0],
          input_c_state=initial_state[1],
          is_training=training,
          name=name)
      return outputs

  def _RunLSTMLayerInference(self, name, input_value, forget_bias,
                             weights_value, h_value, c_value,
                             lstm_layer_function):
    pinputs = array_ops.placeholder(
        dataType, [seq_len, batch_size, input_size], name="inputs")
    pinitial_h_state = array_ops.placeholder(
        dataType, [batch_size, num_channels], name="init_h_state")
    pinitial_c_state = array_ops.placeholder(
        dataType, [batch_size, num_channels], name="init_c_state")
    lstm_output_seq = lstm_layer_function(
        inputs=pinputs,
        weights_value=weights_value,
        initial_state=(pinitial_h_state, pinitial_c_state),
        forget_bias=forget_bias,
        training=False,
        name=name)

    with session_lib.Session() as sess:
      inputs = _createLSTMInput(input_value, batch_size, seq_len, input_size)
      initial_state = _createLSTMInitialState(h_value, c_value, batch_size,
                                              num_channels)
      fd = {
          pinputs: inputs,
          pinitial_h_state: initial_state[0],
          pinitial_c_state: initial_state[1],
      }
      sess.run(variables.global_variables_initializer())
      return sess.run(lstm_output_seq, fd)

  def _RunInferenceComparison(self, name, input_value, forget_bias,
                              weights_value, h_value, c_value):
    ops.reset_default_graph()
    popnn_out = self._RunLSTMLayerInference(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        forget_bias=forget_bias,
        h_value=h_value,
        c_value=c_value,
        lstm_layer_function=self._LSTMLayer)
    ref_out = self._RunLSTMLayerInference(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        forget_bias=forget_bias,
        h_value=h_value,
        c_value=c_value,
        lstm_layer_function=self._LSTMLayerCPU)
    # Check that the whole outupt sequence matches
    self.assertAllClose(popnn_out, ref_out)

  def testLSTMLayerInference(self):
    np.random.seed(0)
    # Run with all-0 weights
    weight0 = 1.
    for h_init in [0., 1.]:
      for c_init in [0., 1.]:
        self._RunInferenceComparison(
            'ones',
            input_value=0.,
            forget_bias=0.,
            weights_value=weight0,
            h_value=h_init,
            c_value=c_init)

    # Run with all-1 weights
    weight1 = 1.
    for h_init in [0., 1.]:
      for c_init in [0., 1.]:
        self._RunInferenceComparison(
            'ones',
            input_value=0.,
            forget_bias=0.,
            weights_value=weight1,
            h_value=h_init,
            c_value=c_init)

    # Run with random weights
    for weight in np.random.rand(3):
      for h_init in [0., 1.]:
        for c_init in [0., 1.]:
          self._RunInferenceComparison(
              'rand',
              input_value=0.,
              forget_bias=0.,
              weights_value=weight,
              h_value=h_init,
              c_value=c_init)

  def _RunLSTMLayerTraining(self, name, input_value, forget_bias,
                            weights_value, h_value, c_value, training_steps,
                            labels_array, lstm_layer_function, device_string):
    pinputs = array_ops.placeholder(
        dataType, [seq_len, batch_size, input_size], name="inputs")
    plabels = array_ops.placeholder(
        dataType, [seq_len, batch_size, num_channels], name="labels")

    with ops.device(device_string):
      with variable_scope.variable_scope("lstm_layer", use_resource=True):
        initial_h_state = _get_variable(
            "initial_h_state",
            shape=[batch_size, num_channels],
            initializer=init_ops.constant_initializer(h_value, dataType))
        initial_c_state = _get_variable(
            "initial_c_state",
            shape=[batch_size, num_channels],
            initializer=init_ops.constant_initializer(c_value, dataType))
      logits = lstm_layer_function(
          inputs=pinputs,
          weights_value=weights_value,
          initial_state=(initial_h_state, initial_c_state),
          forget_bias=forget_bias,
          training=True,
          name=name)
      softmax = nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=plabels)
      loss = math_ops.reduce_mean(softmax)
      train = gradient_descent.GradientDescentOptimizer(0.01).minimize(loss)

    with session_lib.Session() as sess:
      sess.run(variables.global_variables_initializer())
      losses = []
      inputs = _createLSTMInput(input_value, batch_size, seq_len, input_size)
      fd = {
          pinputs: inputs,
          plabels: labels_array,
      }
      for _ in range(0, training_steps):
        l, _ = sess.run([loss, train], fd)
        losses.append(l)
      return losses

  def _RunTrainingComparison(self, name, input_value, forget_bias,
                             weights_value, h_value, c_value, training_steps):
    labels_array = np.ones(
        shape=[seq_len, batch_size, num_channels], dtype=np.float32)
    ops.reset_default_graph()
    popnn_losses = self._RunLSTMLayerTraining(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        forget_bias=forget_bias,
        h_value=h_value,
        c_value=c_value,
        training_steps=training_steps,
        labels_array=labels_array,
        lstm_layer_function=self._LSTMLayer,
        device_string="/device:IPU:0")
    ops.reset_default_graph()
    ref_losses = self._RunLSTMLayerTraining(
        name=name,
        input_value=input_value,
        weights_value=weights_value,
        forget_bias=forget_bias,
        h_value=h_value,
        c_value=c_value,
        training_steps=training_steps,
        labels_array=labels_array,
        lstm_layer_function=self._LSTMLayerCPU,
        device_string="/device:CPU:0")
    self.assertAllClose(popnn_losses, ref_losses)

  def testLSTMLayerTraining(self):
    np.random.seed(42)

    # Run with random weights
    for weight in np.random.rand(3):
      for h_init in [0., 1.]:
        for c_init in [0., 1.]:
          self._RunTrainingComparison(
              'rand',
              input_value=0.,
              forget_bias=0.,
              weights_value=weight,
              h_value=h_init,
              c_value=c_init,
              training_steps=3)


if __name__ == "__main__":
  googletest.main()
