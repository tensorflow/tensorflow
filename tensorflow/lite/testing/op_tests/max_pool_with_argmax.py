# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for max_pool_with_argmax."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import ExtraTocoOptions
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_max_pool_with_argmax_tests(options):
  """Make a set of tests to do max_pool_with_argmax."""

  test_parameters = [{
      'input_size': [[2, 4, 2, 2], [2, 4, 3, 2]],
      'pool_size': [(2, 2), (2, 1)],
      'strides': [(2, 2)],
      'padding': ['SAME', 'VALID'],
  }, {
      'input_size': [[2, 4, 10, 2], [2, 4, 11, 2], [2, 4, 12, 2]],
      'pool_size': [(2, 2)],
      'strides': [(2, 3)],
      'padding': ['SAME', 'VALID'],
  }]

  def build_graph(parameters):
    """Build the exp op testing graph."""
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name='input', shape=parameters['input_size'])
    updates, indices = tf.nn.max_pool_with_argmax(
        input_tensor,
        ksize=parameters['pool_size'],
        strides=parameters['strides'],
        padding=parameters['padding'],
        output_dtype=tf.dtypes.int32)
    return [input_tensor], [updates, indices]

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(
            tf.float32, parameters['input_size'], min_value=-10, max_value=10)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  extra_toco_options = ExtraTocoOptions()
  extra_toco_options.allow_custom_ops = True
  make_zip_of_tests(options, test_parameters, build_graph, build_inputs,
                    extra_toco_options)
