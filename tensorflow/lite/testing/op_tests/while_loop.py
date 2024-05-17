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
"""Test configs for while_loop."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing import zip_test_utils
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function
from tensorflow.python.framework import test_util


@register_make_test_function("make_while_tests")
@test_util.enable_control_flow_v2
def make_while_tests(options):
  """Make a set of tests to do while."""
  # Chose a set of parameters
  test_parameters = [{
      "num_iterations": range(20),
      "increment_value": [[1]],
      "dtype": [tf.int32],
  }, {
      "num_iterations": range(20),
      "increment_value": [["a"]],
      "dtype": [tf.string],
  }]

  def build_graph(parameters):
    """Build the graph for while tests."""
    # MLIR TFLite converter can't handle scalar inputs. This is a workaround
    # to input (1,) tensors and then reshape to scalar.
    # TODO(b/129003347): Remove the workaround after scalar inputs are
    # supported.
    num_iterations = tf.compat.v1.placeholder(
        dtype=tf.int32, name="num_iterations", shape=(1,))
    increment_value = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="increment_value", shape=(1,))
    num_iterations_scalar = tf.reshape(num_iterations, ())

    # For intger inputs, this simple model calucates i-th number of triangular
    # sequence. For string inputs, the model returns the string value, filled
    # with the given increment value times the given num_iterations.
    # The model also returns the counter variable and increment value in the
    # outputs. The counter and increment value are passed to the result to make
    # sure the necessary control depenecy of the model is generated for testing
    # the dynamic tensor cases.
    def cond_fn(counter, value, increment_value):
      del value
      del increment_value
      return counter < num_iterations_scalar

    def body_fn(counter, value, increment_value):
      new_counter = counter + 1
      if parameters["dtype"] == tf.string:
        # Use fill op to create new string value with the given counter value.
        del value
        new_value = tf.fill([1], tf.reshape(increment_value, ()))
      else:
        new_value = value + increment_value
      return [new_counter, new_value, increment_value]

    counter, value, result_increment_value = tf.while_loop(
        cond=cond_fn, body=body_fn,
        loop_vars=[1, increment_value, increment_value])
    return [num_iterations,
            increment_value], [counter, value, result_increment_value]

  def build_inputs(parameters, sess, inputs, outputs):
    numpy_type = zip_test_utils.MAP_TF_TO_NUMPY_TYPE[parameters["dtype"]]
    input_values = [
        np.array([parameters["num_iterations"]], dtype=np.int32),
        np.array(parameters["increment_value"], dtype=numpy_type)
    ]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
