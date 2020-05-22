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
"""Test configs for unroll_batch_matmul."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_unroll_batch_matmul_tests(options):
  """Make a set of tests to test unroll_batch_matmul."""

  # The test cases below requires broadcasting support (BatchMatMulV2 semantic),
  # whis isn't supported as of this change.
  broadcast_shape_params = [
      # Simple broadcast.
      [(1, 2, 3), (3, 5), False, False],
      # Empty batch broadcast.
      [(2, 5, 3), (3, 7), False, False],
      # Single batch with non-empty batch broadcast.
      [(1, 5, 3), (4, 3, 7), False, False],
      # Broadcast both operands
      [(3, 1, 5, 3), (1, 4, 3, 7), False, False],
  ]

  test_parameters = [{
      "dtype": [tf.float32],
      "shape": [[(2, 2, 3),
                 (2, 3, 2), False, False], [(2, 2, 3), (2, 3, 2), True, True],
                [(2, 2, 3),
                 (2, 2, 3), False, True], [(2, 2, 3), (2, 2, 3), True, False],
                [(4, 2, 2, 3), (4, 2, 3, 2), False, False],
                [(4, 2, 2, 3), (4, 2, 3, 2), True, True],
                [(4, 2, 2, 3), (4, 2, 2, 3), False, True],
                [(4, 2, 2, 3),
                 (4, 2, 2, 3), True, False]] + broadcast_shape_params,
  }]

  def build_graph(parameters):
    """Build the batch_matmul op testing graph."""

    def _build_graph():
      """Build the graph."""
      input_tensor1 = tf.compat.v1.placeholder(
          dtype=parameters["dtype"], shape=parameters["shape"][0])
      input_tensor2 = tf.compat.v1.placeholder(
          dtype=parameters["dtype"], shape=parameters["shape"][1])
      # Should be unrolled and replaced with fully_connected ops in the end.
      out = tf.matmul(
          input_tensor1,
          input_tensor2,
          transpose_a=parameters["shape"][2],
          transpose_b=parameters["shape"][3])
      return [input_tensor1, input_tensor2], [out]

    return _build_graph()

  def build_inputs(parameters, sess, inputs, outputs):
    input_value1 = create_tensor_data(
        parameters["dtype"], shape=parameters["shape"][0])
    input_value2 = create_tensor_data(
        parameters["dtype"], shape=parameters["shape"][1])
    return [input_value1, input_value2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value1, input_value2])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
