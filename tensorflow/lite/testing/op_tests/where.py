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
"""Test configs for where."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_where_tests(options):
  """Make a set of tests to do where."""

  test_parameters = [
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([1, 2, 3, 4], [1, 2, 3, 4]),],
          "use_where_v2": [False, True],
          "fully_quantize": [False],
      },
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([], []),],
          "use_where_v2": [],
          "fully_quantize": [False],
      },
      {
          "input_dtype": [tf.float32],
          "input_shape_set": [
              ([1, 2, 3, 4], [1, 2, 3, 4]),
              ([], []),
          ],
          "use_where_v2": [False, True],
          "fully_quantize": [True],
      },
      # High dimension broadcasting support in MLIR converter.
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([8, 7, 6, 5, 4, 3, 2, 1], [4, 3, 2, 1]),
                              ([8, 7, 6, 5, 4, 3, 2, 1], [None, 3, 2, 1]),
                              ([8, 7, 6, 5, None, 3, 2, 1], [None, 3, 2, 1])],
          "use_where_v2": [True],
          "fully_quantize": [False],
          "dynamic_size_value": [4, 1],
      },
      {
          "input_dtype": [tf.float32],
          "input_shape_set": [([8, 7, 6, 5, 4, 3, 2, 1], [4, 3, 2, 1])],
          "use_where_v2": [True],
          "fully_quantize": [True],
          "dynamic_size_value": [4],
      },
      {
          "input_dtype": [tf.float32, tf.int32],
          "input_shape_set": [([], []), ([1], []), ([], [1])],
          "use_where_v2": [False, True],
          "fully_quantize": [False],
      },
  ]

  def populate_dynamic_shape(parameters, input_shape):
    return [
        parameters["dynamic_size_value"] if x is None else x
        for x in input_shape
    ]

  def build_graph(parameters):
    """Build the where op testing graph."""
    input_value1 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input2",
        shape=parameters["input_shape_set"][0])
    input_value2 = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input3",
        shape=parameters["input_shape_set"][1])
    less = tf.less(input_value1, input_value2)
    where = tf.where_v2 if parameters["use_where_v2"] else tf.where
    out = where(less, input_value1, input_value2)
    return [input_value1, input_value2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_shape_1 = populate_dynamic_shape(parameters,
                                           parameters["input_shape_set"][0])
    input_shape_2 = populate_dynamic_shape(parameters,
                                           parameters["input_shape_set"][1])

    input_value1 = create_tensor_data(
        parameters["input_dtype"], input_shape_1, min_value=-1, max_value=1)
    input_value2 = create_tensor_data(
        parameters["input_dtype"], input_shape_2, min_value=-1, max_value=1)
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=4)
