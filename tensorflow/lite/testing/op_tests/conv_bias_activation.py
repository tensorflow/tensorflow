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
"""Test configs for conv followed with bias Add and activations."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def make_conv_bias_activation_tests(activation_op):
  """Make a set of tests to do convolution with activation and bias.

  This test will create multiple consecutive convolutions with NCHW layout to
  make sure that the tranformations to NHWC works as expected. Note this
  doesn't check any performance so manual checking of the generated model is
  advised.

  Args:
    activation_op: The activation op to be used in the test.

  Returns:
    The function that creates the test.
  """

  def create_test(options):
    """Actual function that generates examples."""
    test_parameters = [
        {
            "input_shape": [[1, 3, 4, 3]],
            "filter_shape": [[2, 3], [3, 3]],
            "filter_2_shape": [[2, 1, 1, 3]],
            "strides": [[1, 1, 1, 1]],
            "dilations": [[1, 1, 1, 1]],
            "data_format": ["NCHW"],
            "channel_multiplier": [1, 2],
            "fully_quantize": [False],
            "dynamic_range_quantize": [False],
        },
    ]

    def get_tensor_shapes(parameters):
      input_shape = parameters["input_shape"]
      filter_size = parameters["filter_shape"]
      filter_shape = filter_size + [
          input_shape[3], parameters["channel_multiplier"]
      ]
      return [input_shape, filter_shape]

    # TF CPU doesn't support cases with NCHW. Instead
    # use XLA which doesn't have the same restrictions.
    @tf.function(jit_compile=True)
    def add_conv(input_tensor, filter_input, parameters):
      out = tf.nn.conv2d(
          input=input_tensor,
          filters=filter_input,
          strides=parameters["strides"],
          dilations=parameters["dilations"],
          padding="VALID",
          data_format=parameters["data_format"])
      return out

    def add_bias_add(data_input, filter_shape):
      bias_input = create_tensor_data(np.float32, (filter_shape[-1],))
      out = tf.nn.bias_add(data_input, bias_input, data_format="NHWC")
      return out

    def build_graph(parameters):
      """Build a conv graph given `parameters`."""
      input_shape, filter_shape = get_tensor_shapes(parameters)
      input_tensor = tf.compat.v1.placeholder(
          dtype=tf.float32, name="input", shape=input_shape)

      filter_input = create_tensor_data(
          np.float32, filter_shape, min_value=-10, max_value=10)
      input_tensors = [input_tensor]

      if parameters["data_format"] == "NCHW":
        out = add_conv(input_tensor, filter_input, parameters)
      else:
        out = tf.nn.conv2d(
            input=input_tensor,
            filters=filter_input,
            strides=parameters["strides"],
            dilations=parameters["dilations"],
            padding="VALID",
            data_format=parameters["data_format"])
      out = add_bias_add(out, filter_shape)
      out = activation_op(out)

      # Add another conv + bias_add + activation.

      # Create constant filter for the second conv2d.
      filter_input_2 = create_tensor_data(
          np.float32, parameters["filter_2_shape"], min_value=-10, max_value=10)
      if parameters["data_format"] == "NCHW":
        out = add_conv(out, filter_input_2, parameters)
      else:
        out = tf.nn.conv2d(
            input=out,
            filters=filter_input_2,
            strides=parameters["strides"],
            dilations=parameters["dilations"],
            padding="VALID",
            data_format=parameters["data_format"])
      out = add_bias_add(out, filter_shape)
      out = activation_op(out)
      return input_tensors, [out]

    def build_inputs(parameters, sess, inputs, outputs):
      """Build inputs for conv with activation."""

      input_shape, _ = get_tensor_shapes(parameters)
      values = [
          create_tensor_data(
              np.float32, input_shape, min_value=-1, max_value=1)
      ]
      return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

    make_zip_of_tests(
        options,
        test_parameters,
        build_graph,
        build_inputs,
        expected_tf_failures=2)

  return create_test


@register_make_test_function()
def make_conv_bias_relu6_tests(options):
  """Make a set of tests to do conv_bias_relu6."""
  return make_conv_bias_activation_tests(tf.nn.relu6)(options)
