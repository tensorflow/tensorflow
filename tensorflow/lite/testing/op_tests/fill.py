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
"""Test configs for fill."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_scalar_data
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_fill_tests(options):
  """Make a set of tests to do fill."""

  test_parameters = [{
      "dims_dtype": [tf.int32, tf.int64],
      "dims_shape": [[], [1], [3], [3, 3]],
      "value_dtype": [tf.int32, tf.int64, tf.float32, tf.bool, tf.string],
  }]

  def build_graph(parameters):
    """Build the fill op testing graph."""
    input1 = tf.compat.v1.placeholder(
        dtype=parameters["dims_dtype"],
        name="dims",
        shape=parameters["dims_shape"])
    input2 = tf.compat.v1.placeholder(
        dtype=parameters["value_dtype"], name="value", shape=[])
    out = tf.fill(input1, input2)
    return [input1, input2], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input1 = create_tensor_data(parameters["dims_dtype"],
                                parameters["dims_shape"], 1)
    input2 = create_scalar_data(parameters["value_dtype"])
    return [input1, input2], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input1, input2])))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=20)
