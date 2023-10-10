# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for TensorScatterAdd."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_tensor_scatter_add_tests(options):
  """Make a set of tests to do tensor_scatter_add."""

  test_parameters = [{
      "input_dtype": [tf.float32, tf.int32, tf.int64],
      "input_shape": [[14], [2, 4, 7]],
      "adds_count": [1, 3, 5],
  }]

  def build_graph(parameters):
    """Build the tensor_scatter_add op testing graph."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    # The indices will be a list of "input_shape".
    indices_tensor = tf.compat.v1.placeholder(
        dtype=tf.int32,
        name="indices",
        shape=([parameters["adds_count"],
                len(parameters["input_shape"])]))
    # The adds will be a list of scalar, shaped of "adds_count".
    adds_tensors = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="updates",
        shape=[parameters["adds_count"]])

    out = tf.tensor_scatter_nd_add(input_tensor, indices_tensor, adds_tensors)
    return [input_tensor, indices_tensor, adds_tensors], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    indices = set()
    while len(indices) < parameters["adds_count"]:
      loc = []
      for d in parameters["input_shape"]:
        loc.append(np.random.randint(0, d))
      indices.add(tuple(loc))

    values = [
        create_tensor_data(parameters["input_dtype"],
                           parameters["input_shape"]),
        np.array(list(indices), dtype=np.int32),
        create_tensor_data(
            parameters["input_dtype"],
            parameters["adds_count"],
            min_value=-3,
            max_value=3)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
