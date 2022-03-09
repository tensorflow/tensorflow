# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for tensor_list_set_item."""
import functools

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.ops import list_ops


def _tflite_convert_verify_op(tflite_convert_function, *args, **kwargs):
  """Verifies that the result of the conversion contains DynamicUpdateSlice op."""
  result = tflite_convert_function(*args, **kwargs)
  tflite_model_binary = result[0]
  if not result[0]:
    tf.compat.v1.logging.error(result[1])  # stderr from running tflite_convert.
    raise RuntimeError("Failed to build model: \n\n" + result[1])
  interpreter = tf.lite.Interpreter(model_content=tflite_model_binary)
  interpreter.allocate_tensors()
  for op in interpreter._get_ops_details():  # pylint: disable=protected-access
    if op["op_name"] == "DYNAMIC_UPDATE_SLICE":
      return result
  raise RuntimeError(
      "Expected to generate DYNAMIC_UPDATE_SLICE op node in graph.")


@register_make_test_function()
def make_dynamic_update_slice_tests(options):
  """Make a set of tests to do TensorListSetItem."""

  test_parameters = [
      {
          "element_dtype": [tf.float32, tf.int32, tf.bool],
          "num_elements": [4, 5, 6],
          "element_shape": [[], [5], [3, 3]],
          "index": [0, 1, 2, 3],
      },
  ]

  def build_graph(parameters):
    """Build the TensorListSetItem op testing graph."""
    data = tf.placeholder(
        dtype=parameters["element_dtype"],
        shape=[parameters["num_elements"]] + parameters["element_shape"])
    item = tf.placeholder(
        dtype=parameters["element_dtype"], shape=parameters["element_shape"])
    tensor_list = list_ops.tensor_list_from_tensor(data,
                                                   parameters["element_shape"])
    tensor_list = list_ops.tensor_list_set_item(tensor_list,
                                                parameters["index"], item)
    out = list_ops.tensor_list_stack(
        tensor_list,
        num_elements=parameters["num_elements"],
        element_dtype=parameters["element_dtype"])
    return [data, item], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    data = create_tensor_data(parameters["element_dtype"],
                              [parameters["num_elements"]] +
                              parameters["element_shape"])
    item = create_tensor_data(parameters["element_dtype"],
                              parameters["element_shape"])
    return [data, item], sess.run(
        outputs, feed_dict=dict(zip(inputs, [data, item])))

  options.enable_dynamic_update_slice = True
  options.tflite_convert_function = functools.partial(
      _tflite_convert_verify_op, options.tflite_convert_function)
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
