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
"""Test configs for unsorted_segment ops."""

import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


def make_unsorted_segment_tests(options, unsorted_segment_op):
  """Make a set of tests for given unsorted_segment op."""
  test_parameters = [{
      "data_shape": [[5]],
      "segment_id": [[0, 1, 1, 0, 1]],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[2, 3, 4], [2, 5, 2]],
      "segment_id": [[0, 1]],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[4]],
      "segment_id": [[0, 0, 1, 8]],
      "num_segments": [9],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[3]],
      "segment_id": [[-1, -2, -1]],
      "num_segments": [1],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[3]],
      "segment_id": [[-1, 0, 1]],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[3, 2]],
      "segment_id": [[-1, 0, 0]],
      "num_segments": [1],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[3, 2]],
      "segment_id": [[-1, -2, -1]],
      "num_segments": [1],
      "dtype": [tf.int32, tf.float32],
      "multi_node": [0]
  }, {
      "data_shape": [[4]],
      "segment_id_shape": [[4]],
      "segment_id_min": [0],
      "segment_id_max": [1],
      "num_segments": [2],
      "dtype": [tf.int32, tf.float32],
      "segment_id_2": [[0, 0]],
      "num_segments_2": [1],
      "multi_node": [1]
  }]

  def build_graph_one_node(parameters):
    data_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="data", shape=parameters["data_shape"])
    segment_ids_tensor = tf.constant(
        parameters["segment_id"], dtype=tf.int32, name="segment_ids")
    num_segments_tensor = tf.constant(
        parameters["num_segments"],
        dtype=tf.int32,
        shape=[],
        name="num_segments")
    output = unsorted_segment_op(data_tensor, segment_ids_tensor,
                                 num_segments_tensor)
    return [data_tensor], [output]


# test cases for handling dynamically shaped input tensor
  def build_graph_multi_node(parameters):
    data_tensor = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="data", shape=parameters["data_shape"])
    segment_ids_tensor = tf.compat.v1.placeholder(
        dtype=tf.int32,
        name="segment_ids",
        shape=parameters["segment_id_shape"])
    num_segments_tensor = tf.constant(
        parameters["num_segments"],
        dtype=tf.int32,
        shape=[],
        name="num_segments")
    intermediate_tensor = unsorted_segment_op(data_tensor, segment_ids_tensor,
                                              num_segments_tensor)
    segment_ids_tensor_2 = tf.constant(
        parameters["segment_id_2"], dtype=tf.int32, name="segment_ids_2")
    num_segments_tensor_2 = tf.constant(
        parameters["num_segments_2"],
        dtype=tf.int32,
        shape=[],
        name="num_segments_2")
    output = unsorted_segment_op(intermediate_tensor, segment_ids_tensor_2,
                                 num_segments_tensor_2)
    return [data_tensor, segment_ids_tensor], [output]

  def build_graph(parameters):
    multi_node = parameters["multi_node"]
    if multi_node:
      return build_graph_multi_node(parameters)

    return build_graph_one_node(parameters)

  def build_inputs_one_node(parameters, sess, inputs, outputs):
    data_value = create_tensor_data(
        parameters["dtype"], shape=parameters["data_shape"])
    return [data_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [data_value])))

  def build_inputs_multi_node(parameters, sess, inputs, outputs):
    data_value = create_tensor_data(
        dtype=parameters["dtype"], shape=parameters["data_shape"])
    segment_id_value = create_tensor_data(
        dtype=tf.int32,
        shape=parameters["segment_id_shape"],
        min_value=parameters["segment_id_min"],
        max_value=parameters["segment_id_max"])
    return [data_value, segment_id_value], sess.run(
        outputs, feed_dict=dict(zip(inputs, [data_value, segment_id_value])))

  def build_inputs(parameters, sess, inputs, outputs):
    multi_node = parameters["multi_node"]
    if multi_node:
      return build_inputs_multi_node(parameters, sess, inputs, outputs)

    return build_inputs_one_node(parameters, sess, inputs, outputs)

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)


@register_make_test_function()
def make_unsorted_segment_prod_tests(options):
  make_unsorted_segment_tests(options, tf.math.unsorted_segment_prod)


@register_make_test_function()
def make_unsorted_segment_max_tests(options):
  make_unsorted_segment_tests(options, tf.math.unsorted_segment_max)


@register_make_test_function()
def make_unsorted_segment_sum_tests(options):
  make_unsorted_segment_tests(options, tf.math.unsorted_segment_sum)
