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
"""Test configs for constant ops."""
import numpy as np
import tensorflow as tf

from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import MAP_TF_TO_NUMPY_TYPE
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


# This function tests various TensorFLow functions that generates Const op,
# including `tf.ones`, `tf.zeros` and random functions.
@register_make_test_function()
def make_constant_tests(options):
  """Make a set of tests to do constant ops."""

  test_parameters = [{
      "dtype": [tf.float32, tf.int32],
      "input_shape": [[], [1], [2], [1, 1, 1, 1], [2, 2, 2, 2]],
      "constant_is_also_output": [True, False],
      # Models should not be rejected regardless whether it has unread inputs.
      "has_unread_input": [True, False],
  }]

  def build_graph(parameters):
    """Build a constant graph given `parameters`."""
    dummy_input = tf.compat.v1.placeholder(
        dtype=parameters["dtype"],
        name="input1",
        shape=parameters["input_shape"])
    constant = tf.constant(
        create_tensor_data(parameters["dtype"], parameters["input_shape"]))
    outputs = [tf.maximum(dummy_input, constant)]
    if parameters["constant_is_also_output"]:
      outputs.append(constant)
    inputs = [dummy_input]
    if parameters["has_unread_input"]:
      unread_input = tf.compat.v1.placeholder(
          dtype=parameters["dtype"],
          name="unread_input",
          shape=parameters["input_shape"])
      inputs.append(unread_input)

    return inputs, outputs

  def build_inputs(parameters, sess, inputs, outputs):
    dummy_input = np.zeros(
        parameters["input_shape"],
        dtype=MAP_TF_TO_NUMPY_TYPE[parameters["dtype"]])
    return [dummy_input], sess.run(outputs, feed_dict={inputs[0]: dummy_input})

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
