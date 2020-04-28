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
"""Test configs for space_to_batch_nd."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_space_to_batch_nd_tests(options):
  """Make a set of tests to do space_to_batch_nd."""

  # TODO(nupurgarg): Add test for uint8.
  test_parameters = [
      {
          "dtype": [tf.int32, tf.int64, tf.float32],
          "input_shape": [[1, 2, 2, 3], [2, 2, 4, 1]],
          "block_shape": [[1, 3], [2, 2]],
          "paddings": [[[0, 0], [0, 0]], [[0, 0], [2, 0]], [[1, 1], [1, 1]]],
          "constant_block_shape": [True, False],
          "constant_paddings": [True, False],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[2, 3, 7, 3]],
          "block_shape": [[1, 3], [2, 2]],
          "paddings": [[[0, 0], [2, 0]], [[1, 0], [1, 0]]],
          "constant_block_shape": [True, False],
          "constant_paddings": [True, False],
      },
      # Non-4D use case: 1 bath dimension, 3 spatial dimensions, 2 others.
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 4, 4, 4, 1, 1]],
          "block_shape": [[2, 2, 2]],
          "paddings": [[[0, 0], [0, 0], [0, 0]]],
          "constant_block_shape": [True, False],
          "constant_paddings": [True, False],
      },
      # 3D case.
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 4, 4]],
          "block_shape": [[2]],
          "paddings": [[[0, 0]]],
          "constant_block_shape": [True, False],
          "constant_paddings": [True, False],
      },
  ]

  def build_graph(parameters):
    """Build a space_to_batch graph given `parameters`."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input",
        shape=parameters["input_shape"])
    input_tensors = [input_tensor]

    # Get block_shape either as a const or as a placeholder (tensor).
    if parameters["constant_block_shape"]:
      block_shape = parameters["block_shape"]
    else:
      shape = [len(parameters["block_shape"])]
      block_shape = tf.compat.v1.placeholder(
          dtype=tf.int32, name="shape", shape=shape)
      input_tensors.append(block_shape)

    # Get paddings either as a const or as a placeholder (tensor).
    if parameters["constant_paddings"]:
      paddings = parameters["paddings"]
    else:
      shape = [len(parameters["paddings"]), 2]
      paddings = tf.compat.v1.placeholder(
          dtype=tf.int32, name="paddings", shape=shape)
      input_tensors.append(paddings)

    out = tf.space_to_batch_nd(input_tensor, block_shape, paddings)
    return input_tensors, [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(parameters["dtype"], parameters["input_shape"])
    ]
    if not parameters["constant_block_shape"]:
      values.append(np.array(parameters["block_shape"]))
    if not parameters["constant_paddings"]:
      values.append(np.array(parameters["paddings"]))
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=56)
