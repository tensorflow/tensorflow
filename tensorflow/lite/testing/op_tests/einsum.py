# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for einsum."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.framework import test_util


@register_make_test_function("make_einsum_tests")
@test_util.enable_control_flow_v2
def make_einsum_tests(options):
  """Make a set of tests to do basic einsum ops."""

  test_parameters = [
      {
          "dtype": [tf.float32],
          "shapes": [
              ((3, 4, 5), (3, 5, 6), "ijk,ikm->ijm"),
              ((3, 4, 5), (5, 6), "ijk,km->ijm"),
              ((2, 5, 7), (5, 2), "LBH,BL->BH"),
              ((2, 5, 7), (5, 3, 2), "LBH,BKL->BKH"),
              ((2, 5, 7, 3), (2, 4, 7, 3), "BFNH,BTNH->BNFT"),
              ((2, 5, 7, 3), (7, 3, 4), "BFND,NDH->BFH"),
              ((3, 4, 5), (5, 6, 2), "BFD,DNH->BFNH"),
              ((7, 11, 13), (7, 11, 13, 5), "BIN,BINJ->BIJ"),
              ((7, 11, 19), (7, 11, 13, 19), "BIJ,BINJ->BIN"),
              ((5, 13, 3, 11), (5, 11, 13, 8), "ACBE,AECD->ABCD"),
              ((5, 11, 7, 3), (5, 8, 7, 3), "AECD,ABCD->ACBE"),
              ((5, 4, 3), (3, 2, 1), "...ij,j...->i..."),
              ((5, 4, 3), (3, 2, 1), "...ij,j...->...i"),
              ((1, 11, 19), (7, 11, 13, 19), "...IJ,...INJ->...IN"),
              ((1, 11, 19), (7, 11, 13, 19), "...IJ,...INJ->IN..."),
              ((4, 3, 2, 5), (3, 6, 1), "ij...,jk...->ik..."),
              ((4, 3, 2, 5), (3, 6, 1), "ij...,jk...->...ik"),
          ],
      },
  ]

  def build_graph(parameters):
    """Build a simple graph with einsum Op."""
    input0_shape = parameters["shapes"][0]
    input1_shape = parameters["shapes"][1]
    equation = parameters["shapes"][2]

    input0_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], shape=input0_shape)
    input1_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], shape=input1_shape)
    out = tf.einsum(equation, input0_tensor, input1_tensor)
    return [input0_tensor, input1_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Feed inputs, assign variables, and freeze graph."""
    input0_shape = parameters["shapes"][0]
    input1_shape = parameters["shapes"][1]
    input0_value = create_tensor_data(parameters["dtype"], input0_shape)
    input1_value = create_tensor_data(parameters["dtype"], input1_shape)
    output_values = sess.run(
        outputs, feed_dict=dict(zip(inputs, [input0_value, input1_value])))
    return [input0_value, input1_value], output_values

  options.use_experimental_converter = True
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      use_frozen_graph=True)
