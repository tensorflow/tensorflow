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
"""Test configs for add_n."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_add_n_tests(options):
  """Make a set of tests for AddN op."""

  test_parameters = [
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape": [[2, 5, 3, 1]],
          "num_inputs": [2, 3, 4, 5],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape": [[5]],
          "num_inputs": [2, 3, 4, 5],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32, tf.int32],
          "input_shape": [[]],
          "num_inputs": [2, 3, 4, 5],
          "dynamic_range_quantize": [False],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[]],
          "num_inputs": [2, 3, 4, 5],
          "dynamic_range_quantize": [True],
      },
  ]

  def build_graph(parameters):
    """Builds the graph given the current parameters."""
    input_tensors = []
    for i in range(parameters["num_inputs"]):
      input_tensors.append(
          tf.compat.v1.placeholder(
              dtype=parameters["dtype"],
              name="input_{}".format(i),
              shape=parameters["input_shape"]))
    out = tf.add_n(input_tensors)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Builds operand inputs for op."""
    input_data = []
    for _ in range(parameters["num_inputs"]):
      input_data.append(
          create_tensor_data(parameters["dtype"], parameters["input_shape"]))
    return input_data, sess.run(
        outputs, feed_dict={i: d for i, d in zip(inputs, input_data)})

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
