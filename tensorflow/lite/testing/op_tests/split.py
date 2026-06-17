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
"""Test configs for split."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_split_tests(options):
  """Make a set of tests to do tf.split."""

  test_parameters = [{
      "input_shape": [[1, 3, 4, 6], [2, 4, 1], [6, 4], [8]],
      "num_or_size_splits": [1, 2, 3, 4, 5],
      "axis": [0, 1, 2, 3, -4, -3, -2, -1],
      "fully_quantize": [True, False],
  }]

  def build_graph(parameters):
    input_tensor = tf.compat.v1.placeholder(
        dtype=tf.float32, name="input", shape=parameters["input_shape"])
    out = tf.split(input_tensor, parameters["num_or_size_splits"],
                   parameters["axis"])
    return [input_tensor], out

  def build_inputs(parameters, sess, inputs, outputs):
    values = [
        create_tensor_data(
            np.float32, parameters["input_shape"], min_value=-1, max_value=1)
    ]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(
      options,
      test_parameters,
      build_graph,
      build_inputs,
      expected_tf_failures=224)
