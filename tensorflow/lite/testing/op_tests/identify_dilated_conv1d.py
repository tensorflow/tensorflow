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
"""Test configs for identifying dilated Conv1D."""
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_identify_dilated_conv1d_tests(options):
  """Make a set of tests to do 1D dilated convolution."""

  test_parameters = [
      {
          "input_shape": [[1, 3, 3], [4, 6, 1]],
          "filter_size": [1, 2, 3],
          "stride": [1, 2],
          "dilations": [1, 2, 3],
          "padding": ["VALID", "SAME"],
          "num_filters": [1, 2],
      },
  ]

  def get_tensor_shapes(parameters):
    input_shape = parameters["input_shape"]
    filter_size = parameters["filter_size"]
    filter_shape = [filter_size, input_shape[2], parameters["num_filters"]]
    return [input_shape, filter_shape]

  def build_graph(parameters):
    """Build a conv graph given `parameters`."""
    input_shape, filter_shape = get_tensor_shapes(parameters)
    filter_input = tf.compat.v1.placeholder(
        dtype=tf.float32, name="filter", shape=filter_shape)
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=input_shape)
    input_tensors = [input_tensor, filter_input]

    out = tf.nn.conv1d(
        input_tensor,
        filter_input,
        stride=parameters["stride"],
        dilations=parameters["dilations"],
        padding=parameters["padding"])
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_shape, filter_shape = get_tensor_shapes(parameters)
    values = [
        create_tensor_data(np.float32, input_shape, min_value=-1, max_value=1),
        create_tensor_data(np.float32, filter_shape)
    ]

    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=16)
