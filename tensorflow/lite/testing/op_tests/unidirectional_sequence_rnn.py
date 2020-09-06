# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for unidirectional_sequence_rnn."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.framework import test_util


@register_make_test_function("make_unidirectional_sequence_rnn_tests")
@test_util.enable_control_flow_v2
def make_unidirectional_sequence_rnn_tests(options):
  """Make a set of tests to do unidirectional_sequence_rnn."""

  test_parameters = [{
      "batch_size": [2, 4, 6],
      "seq_length": [1, 3],
      "units": [4, 5],
      "is_dynamic_rnn": [False, True]
  }]

  def build_graph(parameters):
    """Build the graph for unidirectional_sequence_rnn."""
    input_values = []
    if parameters["is_dynamic_rnn"]:
      shape = [
          parameters["seq_length"], parameters["batch_size"],
          parameters["units"]
      ]
      input_value = tf.compat.v1.placeholder(
          dtype=tf.float32, name="input", shape=shape)
      input_values.append(input_value)
      rnn_cell = tf.lite.experimental.nn.TfLiteRNNCell(parameters["units"])
      outs, _ = tf.lite.experimental.nn.dynamic_rnn(
          rnn_cell, input_value, dtype=tf.float32, time_major=True)
      outs = tf.unstack(outs, axis=1)
    else:
      shape = [parameters["batch_size"], parameters["units"]]
      for i in range(parameters["seq_length"]):
        input_value = tf.compat.v1.placeholder(
            dtype=tf.float32, name=("input_%d" % i), shape=shape)
        input_values.append(input_value)
      rnn_cell = tf.lite.experimental.nn.TfLiteRNNCell(parameters["units"])
      outs, _ = tf.nn.static_rnn(rnn_cell, input_values, dtype=tf.float32)

    real_output = tf.zeros([1], dtype=tf.float32) + outs[-1]
    real_output = tf.identity(real_output)
    return input_values, [real_output]

  def build_inputs(parameters, sess, inputs, outputs):
    """Build the inputs for unidirectional_sequence_rnn."""
    input_values = []
    if parameters["is_dynamic_rnn"]:
      shape = [
          parameters["seq_length"], parameters["batch_size"],
          parameters["units"]
      ]
      input_value = create_tensor_data(tf.float32, shape)
      input_values.append(input_value)
    else:
      shape = [parameters["batch_size"], parameters["units"]]
      for _ in range(parameters["seq_length"]):
        input_value = create_tensor_data(tf.float32, shape)
        input_values.append(input_value)
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    # Tflite fused kernel takes input as [time, batch, input].
    # For static unidirectional sequence rnn, the input is an array sized of
    # time, and pack the array together, however, for time = 1, the input is
    # not packed.
    tflite_input_values = input_values
    if not parameters["is_dynamic_rnn"] and parameters["seq_length"] == 1:
      tflite_input_values = [
          input_values[0].reshape(
              (1, parameters["batch_size"], parameters["units"]))
      ]
    return tflite_input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      use_frozen_graph=True)
