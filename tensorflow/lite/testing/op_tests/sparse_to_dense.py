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
"""Test configs for sparse_to_dense."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_scalar_data
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_sparse_to_dense_tests(options):
  """Make a set of tests to do sparse to dense."""

  test_parameters = [{
      "value_dtype": [tf.float32, tf.int32, tf.int64],
      "index_dtype": [tf.int32, tf.int64],
      "value_count": [1, 3, 6, 8],
      "dense_shape": [[15], [3, 10], [4, 4, 4, 4], [7, 10, 9]],
      "default_value": [0, -1],
      "value_is_scalar": [True, False],
  }]

  # Return a single value for 1-D dense shape, but a tuple for other shapes.
  def generate_index(dense_shape):
    if len(dense_shape) == 1:
      return np.random.randint(dense_shape[0])
    else:
      index = []
      for shape in dense_shape:
        index.append(np.random.randint(shape))
      return tuple(index)

  def build_graph(parameters):
    """Build the sparse_to_dense op testing graph."""
    dense_shape = parameters["dense_shape"]

    # Special handle for value_is_scalar case.
    # value_count must be 1.
    if parameters["value_is_scalar"] and parameters["value_count"] == 1:
      value = tf.compat.v1.placeholder(
          name="value", dtype=parameters["value_dtype"], shape=())
    else:
      value = tf.compat.v1.placeholder(
          name="value",
          dtype=parameters["value_dtype"],
          shape=[parameters["value_count"]])
    indices = set()
    while len(indices) < parameters["value_count"]:
      indices.add(generate_index(dense_shape))
    indices = tf.constant(tuple(indices), dtype=parameters["index_dtype"])
    # TODO(renjieliu): Add test for validate_indices case.
    out = tf.sparse_to_dense(
        indices,
        dense_shape,
        value,
        parameters["default_value"],
        validate_indices=False)

    return [value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    if parameters["value_is_scalar"] and parameters["value_count"] == 1:
      input_value = create_scalar_data(parameters["value_dtype"])
    else:
      input_value = create_tensor_data(parameters["value_dtype"],
                                       [parameters["value_count"]])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
