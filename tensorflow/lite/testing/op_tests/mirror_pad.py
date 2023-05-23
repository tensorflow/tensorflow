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
"""Test configs for mirror_pad."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_mirror_pad_tests(options):
  """Make a set of tests to do mirror_pad."""

  test_parameters = [
      {
          "input_shape": [[2, 3]],
          "padding_matrix": [[[1, 1], [2, 1]]],
          "mode": ["REFLECT"],
          "type": ["const"],
          "fully_quantize": [True, False],
      },
      {
          "input_shape": [[2, 3]],
          "padding_matrix": [[[1, 1], [1, 1]]],
          "mode": ["REFLECT"],
          "type": ["const"],
          "fully_quantize": [False],
      },
      {
          "input_shape": [[2, 3]],
          "padding_matrix": [[[1, 1], [2, 1]]],
          "mode": ["SYMMETRIC"],
          "type": ["placeholder"],
          "fully_quantize": [False],
      },
      {
          "input_shape": [[2, 3]],
          "padding_matrix": [[[1, 1], [2, 1]]],
          "mode": ["REFLECT"],
          "type": ["placeholder"],
          "fully_quantize": [False],
      },
      {
          "input_shape": [[3]],
          "padding_matrix": [[[0, 2]]],
          "mode": ["SYMMETRIC"],
          "type": ["placeholder"],
          "fully_quantize": [False],
      },
      {
          "input_shape": [[3]],
          "padding_matrix": [[[0, 2]]],
          "mode": ["SYMMETRIC"],
          "type": ["const"],
          "fully_quantize": [False],
      },
      {
          "input_shape": [[3]],
          "padding_matrix": [[[0, 2]]],
          "mode": ["REFLECT"],
          "type": ["const"],
          "fully_quantize": [False, True],
          "quant_16x8": [False, True],
      },
      {
          "input_shape": [[3, 2, 4, 5]],
          "padding_matrix": [[[1, 1], [2, 2], [1, 1], [1, 1]]],
          "mode": ["SYMMETRIC"],
          "type": ["placeholder"],
          "fully_quantize": [False],
      },
  ]

  def build_graph(parameters):
    """Build the graph for the test case."""

    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    if parameters["type"] != "const" and not parameters["fully_quantize"]:
      padding_matrix = tf.compat.v1.placeholder(
          dtype=tf.int32,
          name="padding",
          shape=[len(parameters["input_shape"]), 2])
      input_tensors = [input_tensor, padding_matrix]
    else:
      padding_matrix = tf.constant(np.array(parameters["padding_matrix"]))
      input_tensors = [input_tensor]
    output = tf.pad(
        tensor=input_tensor, paddings=padding_matrix, mode=parameters["mode"])

    return input_tensors, [output]

  def build_inputs(parameters, sess, inputs, outputs):
    if not parameters["fully_quantize"]:
      input_values = [create_tensor_data(tf.float32, parameters["input_shape"])]
    else:
      input_values = [
          create_tensor_data(
              tf.float32, parameters["input_shape"], min_value=-1, max_value=1)
      ]
    if parameters["type"] != "const":
      input_values.append(np.array(parameters["padding_matrix"]))
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
