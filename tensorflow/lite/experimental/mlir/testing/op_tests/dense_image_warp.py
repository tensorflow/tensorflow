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
"""Test configs for dense_image_warp."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
# Placeholder for internal API

from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraTocoOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_dense_image_warp_tests(options):
  """Make a set of tests to do dense_image_warp."""

  test_parameters = [{
      'input_size': [[2, 4, 4, 1], [2, 4, 3, 3], [3, 7, 9, 2]],
      'flow_size': [[2, 4, 4, 2], [2, 4, 3, 2], [3, 7, 9, 2]],
  }]

  def build_graph(parameters):
    """Build the exp op testing graph."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name='input', shape=parameters['input_size'])
    flow_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name='flow', shape=parameters['flow_size'])
    output = dense_image_warp_annotated(input_tensor, flow_tensor)
    return [input_tensor, flow_tensor], [output]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(
            tf.float32, parameters['input_size'], min_value=-10, max_value=10),
        create_tensor_data(
            tf.float32, parameters['flow_size'], min_value=-10, max_value=10)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  extra_toco_options = ExtraTocoOptions()
  extra_toco_options.allow_custom_ops = True
  options.expected_ops_in_converted_model = ['DenseImageWarp']
  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      extra_toco_options,
      expected_tf_failures=6)
