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
"""Test configs for gelu."""
import functools

import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def _tflite_convert_verify_op(tflite_convert_function, *args, **kwargs):
  """Verifies that the result of the conversion contains Gelu op."""
  result = tflite_convert_function(*args, **kwargs)
  tflite_model_binary = result[0]
  if not result[0]:
    tf.compat.v1.logging.error(result[1])  # stderr from running tflite_convert.
    raise RuntimeError("Failed to build model: \n\n" + result[1])
  interpreter = tf.lite.Interpreter(model_content=tflite_model_binary)
  interpreter.allocate_tensors()
  for op in interpreter._get_ops_details():  # pylint: disable=protected-access
    if op["op_name"] == "GELU":
      return result
  raise RuntimeError("Expected to generate GELU op node in graph.")


@register_make_test_function()
def make_gelu_tests(options):
  """Makes a set of tests for gelu."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[], [1], [2, 3], [1, 1, 1, 1], [1, 3, 4, 3],
                      [3, 15, 14, 3], [3, 1, 2, 4, 6], [2, 2, 3, 4, 5, 6]],
      "fully_quantize": [False, True],
      "input_range": [(-10, 10)],
      "approximate": [True, False],
  }]

  def build_graph(parameters):
    """Builds the gelu op testing graph."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])

    out = tf.nn.gelu(input_tensor, approximate=parameters["approximate"])
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(
            parameters["input_dtype"],
            parameters["input_shape"],
            min_value=-8,
            max_value=8)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  if not options.run_with_flex:
    options.tflite_convert_function = functools.partial(
        _tflite_convert_verify_op,
        options.tflite_convert_function)
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
