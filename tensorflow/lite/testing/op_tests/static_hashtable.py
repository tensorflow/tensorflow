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
"""Test configs for static hashtable."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraTocoOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function("make_static_hashtable_tests")
def make_static_hashtable_tests(options):
  """Make a set of tests to use static hashtable."""

  # Chose a set of parameters
  test_parameters = [{
      "table": [(tf.string, tf.int64, ["1", "2", "3"], [4, 5, 6], -1),
                (tf.int64, tf.string, [1, 2, 3], ["4", "5", "6"], "-1")],
      "input_shape": [[], [3], [1], [10]],
  }]

  def build_graph(parameters):
    """Build the graph for static hashtable tests."""
    (key_dtype, value_dtype, keys, values, default_value) = parameters["table"]

    key_tensor = tf.constant(keys, dtype=key_dtype)
    value_tensor = tf.constant(values, dtype=value_dtype)

    initializer = tf.lookup.KeyValueTensorInitializer(key_tensor, value_tensor)
    table = tf.lookup.StaticHashTable(initializer, default_value)

    with tf.control_dependencies([tf.initializers.tables_initializer()]):
      input_value = tf.compat.v1.placeholder(
          dtype=key_dtype, name="input", shape=parameters["input_shape"])
      out = table.lookup(key_tensor)
    return [input_value], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    (key_dtype, _, _, _, _) = parameters["table"]
    input_values = [create_tensor_data(key_dtype, parameters["input_shape"])]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  extra_toco_options = ExtraTocoOptions()
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs,
                    extra_toco_options)
