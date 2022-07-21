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
"""Test configs for exp."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_conv3d_tests(options):
  """Make a set of tests to do conv3d."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[2, 3, 4, 5, 3], [2, 5, 6, 8, 3]],
      "filter_shape": [[2, 2, 2, 3, 2], [1, 2, 2, 3, 2]],
      "strides": [(1, 1, 1, 1, 1), (1, 1, 1, 2, 1), (1, 1, 2, 2, 1),
                  (1, 2, 1, 2, 1), (1, 2, 2, 2, 1)],
      "dilations": [(1, 1, 1, 1, 1)],
      "padding": ["SAME", "VALID"],
  }]

  def build_graph(parameters):
    """Build the exp op testing graph."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    filter_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="filter",
        shape=parameters["filter_shape"])

    out = tf.nn.conv3d(
        input_tensor,
        filter_tensor,
        strides=parameters["strides"],
        dilations=parameters["dilations"],
        padding=parameters["padding"])
    return [input_tensor, filter_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(
            parameters["input_dtype"],
            parameters["input_shape"],
            min_value=-100,
            max_value=9),
        create_tensor_data(
            parameters["input_dtype"],
            parameters["filter_shape"],
            min_value=-3,
            max_value=3)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
