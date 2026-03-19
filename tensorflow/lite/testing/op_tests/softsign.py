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
"""Test configs for softsign."""
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_softsign_tests(options):
  """Make a set of tests to do softsign."""
  test_parameters = [
      {
          "dtype": [tf.float32],
          "input_shape": [[1, 3, 4, 3], [2, 3], [3], [1, 4], [1, 1, 5],
                          [1, 1, 1, 6]],
          "fully_quantize": [False, True],
          "input_range": [(-1000, 1000)],
      },
      {
          "dtype": [tf.float32],
          "input_shape": [[4, 7]],
          "fully_quantize": [False, True],
          "input_range": [(-1000, 1000)],
      },
  ]
  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="input", shape=parameters["input_shape"]
    )
    out = tf.nn.softsign(input_tensor)
    return [input_tensor], [out]
  def build_inputs(parameters, sess, inputs, outputs):
    input_values = create_tensor_data(
        parameters["dtype"],
        parameters["input_shape"],
        min_value=-1000,
        max_value=1000,
    )
    return [input_values], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_values]))
    )
  # Enable MLIR quantizer.
  options.mlir_quantizer = True
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
