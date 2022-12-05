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
"""Test configs for elementwise ops."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def _make_elementwise_tests(op, allow_fully_quantize=False, min_value=-100,
                            max_value=100):
  """Make a set of tests to do element-wise operations."""

  def f(options):
    """Actual function that generates examples."""
    test_parameters = [
        {
            "input_dtype": [tf.float32],
            "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
            "fully_quantize": [False],
            "input_range": [[min_value, max_value]],
        },
        {
            "input_dtype": [tf.float32],
            "input_shape": [[], [1], [1, 2], [5, 6, 7, 8], [3, 4, 5, 6]],
            "fully_quantize": [True],
            "input_range": [[min_value, max_value]],
        },
    ]

    if not allow_fully_quantize:
      test_parameters = [
          test_parameter for test_parameter in test_parameters
          if True not in test_parameter["fully_quantize"]
      ]

    def build_graph(parameters):
      """Build the unary op testing graph."""
      input_value = tf.compat.v1.placeholder(
          dtype=parameters["input_dtype"],
          name="input1",
          shape=parameters["input_shape"])
      out = op(input_value)
      return [input_value], [out]

    def build_inputs(parameters, sess, inputs, outputs):
      input_value = create_tensor_data(parameters["input_dtype"],
                                       parameters["input_shape"],
                                       min_value=min_value,
                                       max_value=max_value)
      return [input_value], sess.run(
          outputs, feed_dict={inputs[0]: input_value})

    make_zip_of_tests(options, test_parameters, build_graph, build_inputs)

  return f


@register_make_test_function()
def make_sin_tests(options):
  """Make a set of tests to do sin."""
  return _make_elementwise_tests(tf.sin)(options)


@register_make_test_function()
def make_log_tests(options):
  """Make a set of tests to do log."""
  return _make_elementwise_tests(tf.math.log)(options)


@register_make_test_function()
def make_sqrt_tests(options):
  """Make a set of tests to do sqrt."""
  return _make_elementwise_tests(tf.sqrt)(options)


@register_make_test_function()
def make_rsqrt_tests(options):
  """Make a set of tests to do 1/sqrt."""
  return _make_elementwise_tests(tf.math.rsqrt, allow_fully_quantize=True,
                                 min_value=.1, max_value=1)(options)


@register_make_test_function()
def make_square_tests(options):
  """Make a set of tests to do square."""
  return _make_elementwise_tests(tf.square)(options)
