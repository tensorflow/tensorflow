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
"""Test configs for one_hot."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_one_hot_tests(options):
  """Make a set of tests to do one_hot."""

  test_parameters = [{
      "indices_type": [tf.int32, tf.int64],
      "indices_shape": [[3], [4, 4], [1, 5], [5, 1]],
      "axis": [0, 1],
      "dtype": [tf.int32, tf.int64, tf.float32],
      "provide_optional_inputs": [True, False],
  }]

  def build_graph(parameters):
    """Build the one_hot op testing graph."""
    indices = tf.compat.v1.placeholder(
        dtype=parameters["indices_type"],
        name="indices",
        shape=parameters["indices_shape"])
    depth = tf.compat.v1.placeholder(dtype=tf.int32, name="depth", shape=())

    if not parameters["provide_optional_inputs"]:
      out = tf.one_hot(indices=indices, depth=depth)
      return [indices, depth], [out]

    on_value = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="on_value", shape=())
    off_value = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="off_value", shape=())
    out = tf.one_hot(
        indices=indices,
        depth=depth,
        on_value=on_value,
        off_value=off_value,
        axis=parameters["axis"],
        dtype=parameters["dtype"])
    return [indices, depth, on_value, off_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    """Build the input for one_hot op."""
    input_values = [
        create_tensor_data(
            parameters["indices_type"],
            shape=parameters["indices_shape"],
            min_value=-1,
            max_value=10),
        create_tensor_data(tf.int32, shape=None, min_value=1, max_value=10),
    ]

    if parameters["provide_optional_inputs"]:
      input_values.append(
          create_tensor_data(
              parameters["dtype"], shape=None, min_value=1, max_value=10))
      input_values.append(
          create_tensor_data(
              parameters["dtype"], shape=None, min_value=-1, max_value=0))

    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
