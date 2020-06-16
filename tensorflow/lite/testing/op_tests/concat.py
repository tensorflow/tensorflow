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
"""Test configs for concat."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_concat_tests(options):
  """Make a set of tests to do concatenation."""

  test_parameters = [{
      "base_shape": [[1, 3, 4, 3], [3, 4]],
      "num_tensors": [1, 2, 3, 4, 5, 6],
      "axis": [0, 1, 2, 3, -3, -2, -1],
      "type": [tf.float32, tf.uint8, tf.int32, tf.int64],
      "fully_quantize": [False],
      "dynamic_range_quantize": [False],
  }, {
      "base_shape": [[1, 3, 4, 3], [3, 4], [2, 3, 4, 3]],
      "num_tensors": [1, 2, 3, 4, 5, 6],
      "axis": [1, 2, 3, -3, -2, -1],
      "type": [tf.float32],
      "fully_quantize": [True],
      "dynamic_range_quantize": [False],
  }, {
      "base_shape": [[1, 3, 4, 3]],
      "num_tensors": [6],
      "axis": [1],
      "type": [tf.float32],
      "fully_quantize": [False],
      "dynamic_range_quantize": [True],
  }]

  def get_shape(parameters, delta):
    """Return a tweaked version of 'base_shape'."""
    axis = parameters["axis"]
    shape = parameters["base_shape"][:]
    if axis < 0:
      axis += len(shape)
    if axis < len(shape):
      shape[axis] += delta
    return shape

  def build_graph(parameters):
    all_tensors = []
    for n in range(0, parameters["num_tensors"]):
      input_tensor = tf.compat.v1.placeholder(
          dtype=parameters["type"],
          name=("input%d" % n),
          shape=get_shape(parameters, n))
      all_tensors.append(input_tensor)
    out = tf.concat(all_tensors, parameters["axis"])
    return all_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    all_values = []
    for n in range(0, parameters["num_tensors"]):
      input_values = create_tensor_data(
          parameters["type"],
          get_shape(parameters, n),
          min_value=-1,
          max_value=1)
      all_values.append(input_values)
    return all_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, all_values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=75)
