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
"""Test configs for tile."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_tile_tests(options):
  """Make a set of tests to do tile."""
  test_parameters = [
      {
          "input_dtype": [tf.float32, tf.int32, tf.bool, tf.string],
          "input_shape": [[3, 2, 1], [2, 2, 2]],
          "multiplier_dtype": [tf.int32, tf.int64],
          "multiplier_shape": [[3]]
      },
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape": [[]],
          "multiplier_dtype": [tf.int32, tf.int64],
          "multiplier_shape": [[0]]
      },
      {
          "input_dtype": [tf.float32],
          "input_shape": [[3, 2, 1], [2, 2, 2]],
          "multiplier_dtype": [tf.int32, tf.int64],
          "multiplier_shape": [[3]],
          "fully_quantize": [True],
          "quant_16x8": [True, False],
          # The input range is used to create representative dataset for both
          # input and multiplier so it needs to be positive.
          "input_range": [(1, 10)],
      },
      {
          "input_dtype": [tf.float32],
          "input_shape": [[]],
          "multiplier_dtype": [tf.int32, tf.int64],
          "multiplier_shape": [[0]],
          "fully_quantize": [True],
          "quant_16x8": [True, False],
          "input_range": [(-10, 10)]
      }
  ]

  def build_graph(parameters):
    """Build the tile op testing graph."""
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        shape=parameters["input_shape"],
        name="input")
    multiplier_value = tf.compat.v1.placeholder(
        dtype=parameters["multiplier_dtype"],
        shape=parameters["multiplier_shape"],
        name="multiplier")
    out = tf.tile(input_value, multiplier_value)
    return [input_value, multiplier_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    min_value, max_value = parameters.get("input_range", (-10, 10))
    input_value = create_tensor_data(
        parameters["input_dtype"],
        parameters["input_shape"],
        min_value=min_value,
        max_value=max_value)
    multipliers_value = create_tensor_data(
        parameters["multiplier_dtype"],
        parameters["multiplier_shape"],
        min_value=0)
    return [input_value, multipliers_value], sess.run(
        outputs,
        feed_dict={
            inputs[0]: input_value,
            inputs[1]: multipliers_value
        })

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
