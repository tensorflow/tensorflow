# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for complex op."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_complex_tests(options):
  """Make a set of tests to do complex op."""

  # Chose a set of parameters
  test_parameters = [{
      "dtype": [tf.float32, tf.float64],
      "input_shape": [[], [1], [2, 3], [1, 3, 4, 3], [2, 2, 3, 4, 5, 6]],
  }]

  def build_graph(parameters):
    real_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    imag_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.complex(real_tensor, imag_tensor)
    return [real_tensor], [imag_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    real_values = create_tensor_data(
        parameters["dtype"].as_numpy_dtype,
        parameters["input_shape"],
        min_value=-10,
        max_value=10)
    imag_values = create_tensor_data(
        parameters["dtype"].as_numpy_dtype,
        parameters["input_shape"],
        min_value=-10,
        max_value=10)
    return [real_values, imag_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [real_values, imag_values])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
