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
"""Test configs for conv."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_conv_to_depthwiseconv_with_shared_weights_tests(options):
  """Make a test where 2 Conv ops shared the same constant weight tensor."""

  test_parameters = [{
      "input_shape": [[1, 10, 10, 1]],
      "filter_shape": [[3, 3]],
      "strides": [[1, 1, 1, 1]],
      "dilations": [[1, 1, 1, 1]],
      "padding": ["SAME"],
      "data_format": ["NHWC"],
      "channel_multiplier": [3],
  }]

  def get_tensor_shapes(parameters):
    input_shape = parameters["input_shape"]
    filter_size = parameters["filter_shape"]
    filter_shape = filter_size + [
        input_shape[3], parameters["channel_multiplier"]
    ]
    return [input_shape, filter_shape]

  def build_graph(parameters):
    """Build a conv graph given `parameters`."""
    input_shape, filter_shape = get_tensor_shapes(parameters)
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=input_shape)

    # Construct a constant weights tensor which will be used by both Conv2D.
    filter_tensor = tf.constant(
        create_tensor_data(np.float32, filter_shape), dtype=tf.float32)
    input_tensors = [input_tensor]

    # Construct 2 Conv2D operations which use exactly the same input and
    # weights.
    result1 = tf.nn.conv2d(
        input_tensor,
        filter_tensor,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    result2 = tf.nn.conv2d(
        input_tensor,
        filter_tensor,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"],
        data_format=parameters["data_format"])
    # Add the 2 results up.
    out = result1 + result2
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # Build list of input values either containing 1 tensor (input) or 2 tensors
    # (input, filter) based on whether filter is constant or variable input.
    input_shape, unused_filter_shape = get_tensor_shapes(parameters)
    values = [create_tensor_data(np.float32, input_shape)]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
