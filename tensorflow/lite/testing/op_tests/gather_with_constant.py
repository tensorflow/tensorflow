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
"""Test configs for gather_with_constant."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_gather_with_constant_tests(options):
  """Make a set of test which feed a constant to gather."""

  test_parameters = [{
      "input_shape": [[3]],
      "reference_shape": [[2]],
  }, {
      "input_shape": [[2, 3]],
      "reference_shape": [[2, 3]],
  }]

  def build_graph(parameters):
    """Build a graph where the inputs to Gather are constants."""
    reference = tf.compat.v1.placeholder(
        dtype=tf.int32, shape=parameters["reference_shape"])
    gather_input = tf.constant(
        create_tensor_data(tf.int32, parameters["input_shape"]))
    gather_indices = tf.constant([0, 1], tf.int32)
    out = tf.equal(reference, tf.gather(gather_input, gather_indices))
    return [reference], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    reference_values = np.zeros(parameters["reference_shape"], dtype=np.int32)
    return [reference_values], sess.run(
        outputs, feed_dict={inputs[0]: reference_values})

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
