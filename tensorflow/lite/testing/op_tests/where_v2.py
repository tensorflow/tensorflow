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
"""Test configs for where_v2."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_where_v2_tests(options):
  """Make a set of tests to do where_v2."""

  test_parameters = [
      {
          "condition_dtype": [
              tf.float32, tf.bool, tf.int32, tf.uint32, tf.uint8
          ],
          "input_condition_shape": [[1, 2, 3, 4]],
          "input_dtype": [tf.float32, tf.int32, None],
          "input_shape_set": [([1, 2, 3, 4], [1, 1, 1, 1]),],
      },
      {
          "condition_dtype": [
              tf.float32, tf.bool, tf.int32, tf.uint32, tf.uint8
          ],
          "input_condition_shape": [[2], [1]],
          "input_dtype": [tf.float32, tf.int32, None],
          "input_shape_set": [([2, 1, 2, 1], [2, 1, 2, 1]),],
      },
      {
          "condition_dtype": [
              tf.float32, tf.bool, tf.int32, tf.uint32, tf.uint8
          ],
          "input_condition_shape": [[1, 4, 2]],
          "input_dtype": [tf.float32, tf.int32, None],
          "input_shape_set": [([1, 3, 4, 2], [1, 3, 4, 2]),],
      },
      {
          "condition_dtype": [
              tf.float32, tf.bool, tf.int32, tf.uint32, tf.uint8
          ],
          "input_condition_shape": [[1, 2]],
          "input_dtype": [tf.float32, tf.int32, None],
          "input_shape_set": [([1, 2, 2], [1, 2, 2]),],
      },
      {
          "condition_dtype": [tf.bool],
          "input_condition_shape": [[1, 1]],
          "input_dtype": [tf.float32, tf.int32, None],
          "input_shape_set": [([1, 1, 2, 2], [1, 1, 2, 2]),],
      },
      {
          "condition_dtype": [tf.bool],
          "input_condition_shape": [[4]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([4, 4], [4, 4]),],
      },
      {
          "condition_dtype": [tf.bool],
          "input_condition_shape": [[2]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([2, 3], [2, 3]),],
      },
      {
          "condition_dtype": [
              tf.float32, tf.bool, tf.int32, tf.uint32, tf.uint8
          ],
          "input_condition_shape": [[1, 2], None],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([1, 2, 2], [1, 2]),],
      },
  ]

  def build_graph(parameters):
    """Build the where op testing graph."""
    # To actually use where op, x, y params to where_v2 needs to be None.
    # This is needed when type is not bool, so we actually use where op.
    if parameters["condition_dtype"] != tf.bool and parameters[
        "input_dtype"] is not None:
      parameters["condition_dtype"] = tf.bool
    input_condition = tf.compat.v1.placeholder(
        dtype=parameters["condition_dtype"],
        name="input_condition",
        shape=parameters["input_condition_shape"])
    input_value1 = None
    input_value2 = None
    if parameters["input_dtype"] is not None:
      input_value1 = tf.compat.v1.placeholder(
          dtype=parameters["input_dtype"],
          name="input_x",
          shape=parameters["input_shape_set"][0])
      input_value2 = tf.compat.v1.placeholder(
          dtype=parameters["input_dtype"],
          name="input_y",
          shape=parameters["input_shape_set"][1])
    out = tf.where_v2(input_condition, input_value1, input_value2)
    return [input_condition, input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_condition = create_tensor_data(parameters["condition_dtype"],
                                         parameters["input_condition_shape"])
    input_value1 = None
    input_value2 = None
    if parameters["input_dtype"] is not None:
      input_value1 = create_tensor_data(parameters["input_dtype"],
                                        parameters["input_shape_set"][0])
      input_value2 = create_tensor_data(parameters["input_dtype"],
                                        parameters["input_shape_set"][1])
      return [input_condition, input_value1, input_value2], sess.run(
          outputs,
          feed_dict=dict(
              zip(inputs, [input_condition, input_value1, input_value2])))
    else:
      return [input_condition, input_value1, input_value2], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_condition])))

  options.use_experimental_converter = True
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=2)
