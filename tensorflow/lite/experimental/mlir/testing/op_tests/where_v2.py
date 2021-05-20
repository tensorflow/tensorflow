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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_where_v2_tests(options):
  """Make a set of tests to do where_v2."""

  test_parameters = [
      {
          "input_condition_shape": [[1, 2, 3, 4]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([1, 2, 3, 4], [1, 1, 1, 1]),],
      },
      {
          "input_condition_shape": [[2], [1]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([2, 1, 2, 1], [2, 1, 2, 1]),],
      },
      {
          "input_condition_shape": [[1, 4, 2]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([1, 3, 4, 2], [1, 3, 4, 2]),],
      },
      {
          "input_condition_shape": [[1, 2]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([1, 2, 2], [1, 2, 2]),],
      },
      {
          "input_condition_shape": [[1, 1]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([1, 1, 2, 2], [1, 1, 2, 2]),],
      },
      {
          "input_condition_shape": [[4]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([4, 4], [4, 4]),],
      },
      {
          "input_condition_shape": [[2]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([2, 3], [2, 3]),],
      },
      {
          "input_condition_shape": [[1, 2]],
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([1, 2, 2], [1, 2]),],
      },
  ]

  def build_graph(parameters):
    """Build the where op testing graph."""
    input_condition = tf.compat.v1.placeholder(
        dtype=tf.bool,
        name="input_condition",
        shape=parameters["input_condition_shape"])
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
    input_condition = create_tensor_data(tf.bool,
                                         parameters["input_condition_shape"])
    input_value1 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_set"][0])
    input_value2 = create_tensor_data(parameters["input_dtype"],
                                      parameters["input_shape_set"][1])
    return [input_condition, input_value1, input_value2], sess.run(
        outputs,
        feed_dict=dict(
            zip(inputs, [input_condition, input_value1, input_value2])))

  options.use_experimental_converter = True
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=2)
