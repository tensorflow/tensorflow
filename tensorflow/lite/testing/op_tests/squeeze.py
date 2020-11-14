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
"""Test configs for squeeze."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_squeeze_tests(options):
  """Make a set of tests to do squeeze."""

  test_parameters = [{
      "dtype": [tf.int32, tf.float32, tf.int64],
      "input_shape": [[1, 2, 1, 3, 1, 4, 1, 1]],
      "axis": [
          None, [], [0, 2], [4, 7], [-1, 0, 2, 0, 7, -6], [1], [2, 3, 2],
          [-1, -2, -4, -6, -8], [0, 2, 4, 6, 7], [7, 6, 4, 2, 0], [6, 6],
          [0, 1, 2, 3, 4, 5, 6, 7], [-2, -3, 1, 0, 7, -5]
      ],
      "fully_quantize": [False],
  }, {
      "dtype": [tf.int32, tf.float32, tf.int64],
      "input_shape": [[1]],
      "axis": [None, [], [0], [-1]],
      "fully_quantize": [False],
  }, {
      "dtype": [tf.int32, tf.float32, tf.int64],
      "input_shape": [[1, 1, 1, 1, 1]],
      "axis": [None, [], [0], [3, 0], [-2, 0, 3, 2]],
      "fully_quantize": [False],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 2, 1, 3, 1, 4, 1, 1]],
      "axis": [
          None, [], [0, 2], [4, 7], [-1, 0, 2, 0, 7, -6], [1], [2, 3, 2],
          [-1, -2, -4, -6, -8], [0, 2, 4, 6, 7], [7, 6, 4, 2, 0], [6, 6],
          [0, 1, 2, 3, 4, 5, 6, 7], [-2, -3, 1, 0, 7, -5]
      ],
      "fully_quantize": [True],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 1, 1, 1, 1]],
      "axis": [[0], [3, 0], [-2, 0, 3, 2]],
      "fully_quantize": [True],
  }, {
      "dtype": [tf.float32],
      "input_shape": [[1, 1, 5, 10], [1, 5, 1, 10], [5, 1, 10]],
      "axis": [[0], [1], [3, 0], [-2, 0, 3, 2]],
      "fully_quantize": [True],
  }, {
      "dtype": [tf.string],
      "input_shape": [[1, 1, 5, 10], [1, 5, 1, 10]],
      "axis": [[0], []],
      "fully_quantize": [False],
  }]

  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    out = tf.squeeze(input_tensor, axis=parameters["axis"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        parameters["dtype"],
        parameters["input_shape"],
        min_value=-1,
        max_value=1)
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=24)
