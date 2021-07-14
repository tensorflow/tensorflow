# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for broadcast_to."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function("make_broadcast_to_tests")
def make_broadcast_to_tests(options):
  """Make a set of tests to do broadcast_to."""

  # Chose a set of parameters
  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[1, 2], [2, 3, 4], [1], [2, 5, 2, 3, 4]],
      "output_shape": [[3, 1, 2], [5, 2, 3, 4], [10, 10],
                       [1, 2, 1, 2, 5, 2, 3, 4]],
  }, {
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[3, 2, 3, 4, 5, 6, 7, 8]],
      "output_shape": [[3, 2, 3, 4, 5, 6, 7, 8]],
  }, {
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[1, 3, 1, 2, 1, 4, 1, 1]],
      "output_shape": [[2, 3, 1, 2, 2, 4, 1, 1]],
  }, {
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[2, 1, 1, 2, 1, 4, 1, 1]],
      "output_shape": [[2, 3, 2, 2, 2, 4, 1, 1]],
  }, {
      "input_dtype": [tf.float32, tf.int32],
      "input_shape": [[3, 4, 1]],
      "output_shape": [[3, 4, 0]],
  }]

  def build_graph(parameters):
    """Build the graph for cond tests."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])

    out = tf.broadcast_to(input_tensor, shape=parameters["output_shape"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = [
        create_tensor_data(parameters["input_dtype"], parameters["input_shape"])
    ]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=16)
