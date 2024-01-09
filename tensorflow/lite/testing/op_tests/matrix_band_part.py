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
"""Test configs for matrix_band_part."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_matrix_band_part_tests(options):
  """Make a set of tests for tf.linalg.band_part op."""

  test_parameters = [{
      "input_dtype": [tf.float32],
      "input_shape": [[1, 2], [3, 4, 5], [6, 7, 8, 9], [None, None], [10, None],
                      [None, 10], [3, None, 10], [None, None, None], None],
      # TFL range doesn't accept I64.
      "index_dtype": [tf.int32],
  }]

  def get_static_shape(shape):
    """Randomly assign static number for dynamic dimension."""
    if not shape:
      return np.random.randint(
          low=1, high=10, size=np.random.randint(low=2, high=10,
                                                 size=())).tolist()
    return [x or np.random.randint(low=5, high=10, size=()) for x in shape]

  def build_graph(parameters):
    """Build the sign op testing graph."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["input_dtype"],
        name="input",
        shape=parameters["input_shape"])
    num_lower = tf.compat.v1.placeholder(
        dtype=parameters["index_dtype"], name="num_lower", shape=())
    num_upper = tf.compat.v1.placeholder(
        dtype=parameters["index_dtype"], name="num_upper", shape=())
    out = tf.linalg.band_part(input_tensor, num_lower, num_upper)
    return [input_tensor, num_lower, num_upper], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    static_input_shape = get_static_shape(parameters["input_shape"])
    input_value = create_tensor_data(parameters["input_dtype"],
                                     static_input_shape)
    num_lower = create_tensor_data(
        parameters["index_dtype"],
        shape=(),
        min_value=-1,
        max_value=static_input_shape[-2])
    num_upper = create_tensor_data(
        parameters["index_dtype"],
        shape=(),
        min_value=-1,
        max_value=static_input_shape[-1])
    return [input_value, num_lower, num_upper
           ], sess.run(outputs,
                       dict(zip(inputs, [input_value, num_lower, num_upper])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
