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
"""Test configs for batchmatmul."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function("make_batchmatmul_tests")
def make_batchmatmul_tests(options):
  """Make a set of tests to do basic batch matrix multiply."""

  test_parameters = [
      {
          "dtype": [tf.float32],
          "shapes": [((3, 4, 7), (7, 9), (3, 4, 7), (7, 9)),
                     ((None, 4, 5), (None, 5, 6), (3, 4, 5), (3, 5, 6)),
                     ((None, 1, 3, 4), (None, 4, 2), (2, 1, 3, 4), (5, 4, 2))],
          "adjoint_b": [False, True],
          "adjoint_a": [False, True],
          "rhs_constant": [False],
          "fully_quantize": [False, True],
      },
  ]

  def swap_last_two_dims(*args):
    """Return a tuple with the last two dimensions swapped."""
    return args[:-2] + (args[-1],) + (args[-2],)

  def build_graph(parameters):
    """Build a simple graph with BatchMatMul."""
    placeholder0_shape = parameters["shapes"][0]
    adj_a = parameters["adjoint_a"]
    adj_b = parameters["adjoint_b"]
    rhs_constant = parameters["rhs_constant"]
    if adj_a:
      placeholder0_shape = swap_last_two_dims(*placeholder0_shape)
    input0_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], shape=placeholder0_shape)
    if rhs_constant:
      if adj_b:
        constant1_shape = swap_last_two_dims(*parameters["shapes"][3])
      else:
        constant1_shape = parameters["shapes"][3]
      data = create_tensor_data(
          parameters["dtype"], constant1_shape, min_value=-1.0, max_value=1.0)
      input1_constant = tf.constant(
          data, shape=constant1_shape, dtype=parameters["dtype"])
      out = tf.matmul(
          input0_tensor, input1_constant, adjoint_a=adj_a, adjoint_b=adj_b)
      return [input0_tensor], [out]
    else:
      if adj_b:
        placeholder1_shape = swap_last_two_dims(*parameters["shapes"][1])
      else:
        placeholder1_shape = parameters["shapes"][1]
      input1_tensor = tf.compat.v1.placeholder(
          dtype=parameters["dtype"], shape=placeholder1_shape)
      out = tf.matmul(
          input0_tensor, input1_tensor, adjoint_a=adj_a, adjoint_b=adj_b)
      return [input0_tensor, input1_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Feed inputs, assign variables, and freeze graph."""
    input0_shape = parameters["shapes"][2]
    adj_a = parameters["adjoint_a"]
    adj_b = parameters["adjoint_b"]
    rhs_constant = parameters["rhs_constant"]
    if adj_a:
      input0_shape = swap_last_two_dims(*input0_shape)
    input0_value = create_tensor_data(
        parameters["dtype"], input0_shape, min_value=-1.0, max_value=1.0)
    if rhs_constant:
      output_values = sess.run(
          outputs, feed_dict=dict(zip(inputs, [input0_value])))
      return [input0_value], output_values
    else:
      input1_shape = parameters["shapes"][
          3] if not adj_b else swap_last_two_dims(*parameters["shapes"][3])
      input1_value = create_tensor_data(
          parameters["dtype"], input1_shape, min_value=-1.0, max_value=1.0)
      output_values = sess.run(
          outputs, feed_dict=dict(zip(inputs, [input0_value, input1_value])))
      return [input0_value, input1_value], output_values

  options.use_experimental_converter = True
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=0)
