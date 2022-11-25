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
"""Test configs for tensor_list_dynamic_shape."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.ops import list_ops


@register_make_test_function()
def make_tensor_list_dynamic_shape_tests(options):
  """Make a set of tests for tensorlists with dynamic shape."""

  test_parameters = [
      {
          "element_dtype": [tf.float32, tf.int32],
          "num_elements": [4, 5, 6],
          "element_shape": [[], [5], [3, 3]],
      },
  ]

  def build_graph(parameters):
    """Build the TensorListSetItem op testing graph."""
    item = tf.compat.v1.placeholder(
        dtype=parameters["element_dtype"], shape=parameters["element_shape"])
    tensor_list = list_ops.tensor_list_reserve(
        element_shape=None,
        num_elements=parameters["num_elements"],
        element_dtype=parameters["element_dtype"])

    init_state = (0, tensor_list)
    condition = lambda i, _: i < parameters["num_elements"]

    def loop_body(i, tensor_list):
      new_item = tf.add(
          tf.add(item, item),
          tf.constant(value=1, dtype=parameters["element_dtype"]))
      new_list = list_ops.tensor_list_set_item(tensor_list, i, new_item)
      return i + 1, new_list

    _, tensor_list = tf.while_loop(
        cond=condition, body=loop_body, loop_vars=init_state)
    out = list_ops.tensor_list_stack(
        tensor_list,
        num_elements=parameters["num_elements"],
        element_dtype=parameters["element_dtype"])
    return [item], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    item = create_tensor_data(parameters["element_dtype"],
                              parameters["element_shape"])
    return [item], sess.run(outputs, feed_dict=dict(zip(inputs, [item])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
