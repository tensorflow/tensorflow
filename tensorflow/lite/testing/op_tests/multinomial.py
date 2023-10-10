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
"""Test configs for multinomial."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_multinomial_tests(options):
  """Make a set of tests to do multinomial."""
  test_parameters = [{
      "logits_shape": [[1, 2], [2, 5]],
      "dtype": [tf.int64, tf.int32],
      "seed": [None, 1234],
      "seed2": [5678],
  }, {
      "logits_shape": [[1, 2]],
      "dtype": [tf.int64, tf.int32],
      "seed": [1234],
      "seed2": [None]
  }]

  def build_graph(parameters):
    """Build the op testing graph."""
    tf.compat.v1.set_random_seed(seed=parameters["seed"])
    logits_tf = tf.compat.v1.placeholder(
        name="logits", dtype=tf.float32, shape=parameters["logits_shape"])
    num_samples_tf = tf.compat.v1.placeholder(
        name="num_samples", dtype=tf.int32, shape=None)
    out = tf.random.categorical(
        logits=logits_tf,
        num_samples=num_samples_tf,
        dtype=parameters["dtype"],
        seed=parameters["seed2"])
    return [logits_tf, num_samples_tf], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    input_values = [
        create_tensor_data(
            dtype=tf.float32, shape=parameters["logits_shape"], min_value=-2,
            max_value=-1),
        create_tensor_data(
            dtype=tf.int32, shape=None, min_value=10, max_value=100)
    ]
    return input_values, sess.run(
        outputs, feed_dict=dict(zip(inputs, input_values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
