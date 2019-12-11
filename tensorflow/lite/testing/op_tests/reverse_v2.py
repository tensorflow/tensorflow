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
"""Test configs for reverse_v2."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_reverse_v2_tests(options):
  """Make a set of tests to do reverse_v2."""

  test_parameters = [{
      "base_shape": [[3, 4, 3], [3, 4], [5, 6, 7, 8]],
      "axis": [0, 1, 2, 3],
  }]

  def get_valid_axis(parameters):
    """Return a tweaked version of 'axis'."""
    axis = parameters["axis"]
    shape = parameters["base_shape"][:]
    while axis > len(shape) - 1:
      axis -= 1
    return axis

  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name=("input"), shape=parameters["base_shape"])
    outs = tf.reverse(input_tensor, axis=[get_valid_axis(parameters)])
    return [input_tensor], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value = create_tensor_data(np.float32, shape=parameters["base_shape"])
    return [input_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [input_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
