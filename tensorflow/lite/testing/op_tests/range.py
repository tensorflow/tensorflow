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
"""Test configs for range."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_scalar_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_range_tests(options):
  """Make a set of tests to do range."""

  test_parameters = [{
      "dtype": [tf.int32, tf.float32],
      "offset": [10, 100, 1000, 0],
      "delta": [1, 2, 3, 4, -1, -2, -3, -4],
  }]

  def build_graph(parameters):
    """Build the range op testing graph."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name=("start"), shape=[])
    if parameters["delta"] < 0:
      offset = parameters["offset"] * -1
    else:
      offset = parameters["offset"]
    delta = parameters["delta"]
    limit_tensor = input_tensor + offset
    delta_tensor = tf.constant(delta, dtype=parameters["dtype"])
    out = tf.range(input_tensor, limit_tensor, delta_tensor)
    return [input_tensor], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_scalar_data(parameters["dtype"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
