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
"""Test configs for eye."""
import numpy as np
import tensorflow as tf
from tensorflow.lite.testing.zip_test_utils import create_scalar_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_eye_tests(options):
  """Make a set of tests for tf.eye op."""

  test_parameters = [{
      "num_rows_shape": [[]],
      "num_cols_shape": [[]],
      "batch_shape": [[3], [2, 4], [4, 5, 6], None],
      "use_num_cols": [True, False],
      "dtype": [tf.float32, tf.int32],
  }]

  def build_graph(parameters):
    """Make a set of tests to do eye."""

    input_tensor0 = tf.compat.v1.placeholder(
        dtype=tf.int32, name="num_rows", shape=parameters["num_rows_shape"])
    input_tensor1 = tf.compat.v1.placeholder(
        dtype=tf.int32, name="num_columns", shape=parameters["num_cols_shape"])
    if parameters["use_num_cols"]:
      outs = tf.eye(
          num_rows=input_tensor0,
          num_columns=input_tensor1,
          batch_shape=parameters["batch_shape"],
          dtype=parameters["dtype"])
      return [input_tensor0, input_tensor1], [outs]
    else:
      outs = tf.eye(num_rows=input_tensor0, dtype=parameters["dtype"])
      return [input_tensor0], [outs]

  def build_inputs(parameters, sess, inputs, outputs):
    input_value0 = create_scalar_data(dtype=np.int32, min_value=1)
    input_value1 = create_scalar_data(dtype=np.int32, min_value=1)
    if parameters["use_num_cols"]:
      return [input_value0, input_value1], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_value0, input_value1])))
    else:
      return [input_value0], sess.run(
          outputs, feed_dict=dict(zip(inputs, [input_value0])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
