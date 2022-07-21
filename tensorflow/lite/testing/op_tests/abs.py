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
"""Test configs for abs."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_abs_tests(options):
  """Make a set of tests to do abs."""

  # Chose a set of parameters
  test_parameters = [{
      "input_shape": [[], [1], [2, 3], [1, 1, 1, 1], [1, 3, 4, 3],
                      [3, 15, 14, 3], [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
      "dtype": [tf.float32],
      "dynamic_range_quantize": [False, True],
      "fully_quantize": [False],
      "input_range": [(-10, 10), (-10, 0)],
  }, {
      "input_shape": [[], [1], [2, 3], [1, 1, 1, 1], [1, 3, 4, 3],
                      [3, 15, 14, 3], [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
      "dtype": [tf.float32],
      "dynamic_range_quantize": [False],
      "fully_quantize": [True],
      "input_range": [(-10, 10)],
  }, {
      "input_shape": [[], [1], [2, 3], [1, 1, 1, 1],
                      [1, 3, 4, 3], [3, 15, 14, 3],
                      [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
      "dtype": [tf.int16],
  }]

  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.abs(input_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    min_value, max_value = (-10, 10)
    if "input_range" in parameters:
      min_value, max_value = parameters["input_range"]
    input_values = create_tensor_data(
        parameters["dtype"],
        parameters["input_shape"],
        min_value=min_value,
        max_value=max_value)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
