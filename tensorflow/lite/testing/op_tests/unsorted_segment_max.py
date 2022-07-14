# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for unsorted_segment_max."""

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_unsorted_segment_max_tests(options):
  """Make a set of tests for unsorted_segment_max op."""
  test_parameters = [{
      "data": [[5]],
      "segment_id": [[0, 1, 1, 0, 1]],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "segment_dtype": [tf.int32, tf.int64]
  }, {
      "data": [[2, 3, 4], [2, 5, 2]],
      "segment_id": [[0, 1], [-1, -1]],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "segment_dtype": [tf.int32, tf.int64]
  }]

  def build_graph(parameters):
    data_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="data", shape=parameters["data"])
    segment_ids_tensor = tf.constant(
        parameters["segment_id"],
        dtype=parameters["segment_dtype"],
        name="segment_ids")
    num_segments = tf.constant(
        parameters["num_segments"],
        dtype=parameters["segment_dtype"],
        shape=[],
        name="num_segments")
    output = tf.math.unsorted_segment_max(data_tensor, segment_ids_tensor,
                                          num_segments)
    return [data_tensor], [output]

  def build_inputs(parameters, sess, inputs, outputs):
    data_value = create_tensor_data(
        parameters["dtype"], shape=parameters["data"])
    return [data_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [data_value])))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
