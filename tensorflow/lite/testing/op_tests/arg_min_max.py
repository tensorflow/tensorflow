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
"""Test configs for arg_min_max."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_arg_min_max_tests(options):
  """Make a set of tests to do arg_max."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[], [1, 1, 1, 3], [2, 3, 4, 5], [2, 3, 3], [5, 5], [10]],
      "output_type": [tf.int32, tf.int64],
      "is_arg_max": [True],
      "dynamic_range_quantize": [False, True],
  }]

  def build_graph(parameters):
    """Build the topk op testing graph."""
    input_value = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    axis = random.randint(0, max(len(parameters["input_shape"]) - 1, 0))
    if parameters["is_arg_max"]:
      out = tf.math.argmax(
          input_value, axis, output_type=parameters["output_type"])
    else:
      out = tf.math.argmin(
          input_value, axis, output_type=parameters["output_type"])
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(parameters["input_dtype"],
                                     parameters["input_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=8)
