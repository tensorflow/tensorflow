# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests the graph quantization script.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.contrib.quantization.tools import quantize_graph
from tensorflow.python.framework import graph_util

flags = tf.app.flags
FLAGS = flags.FLAGS


def run_graph_def(graph_def, input_map, outputs):
  graph = tf.Graph()
  with graph.as_default():
    tf.import_graph_def(graph_def, input_map={}, name="")
  with tf.Session(graph=graph) as sess:
    results = sess.run(outputs, feed_dict=input_map)
  return results


def test_mat_mul(m, n, k, a, b):
  """Tests a MatMul replacement."""
  a_constant_name = "a_constant"
  b_constant_name = "b_constant"
  mat_mul_name = "mat_mul"

  float_graph_def = tf.GraphDef()
  a_constant = quantize_graph.create_constant_node(a_constant_name,
                                                   value=a,
                                                   dtype=tf.float32,
                                                   shape=[m, k])
  float_graph_def.node.extend([a_constant])
  b_constant = quantize_graph.create_constant_node(b_constant_name,
                                                   value=b,
                                                   dtype=tf.float32,
                                                   shape=[k, n])
  float_graph_def.node.extend([b_constant])
  mat_mul_node = quantize_graph.create_node("MatMul", mat_mul_name,
                                            [a_constant_name, b_constant_name])
  quantize_graph.set_attr_dtype(mat_mul_node, "T", tf.float32)
  quantize_graph.set_attr_bool(mat_mul_node, "transpose_a", False)
  quantize_graph.set_attr_bool(mat_mul_node, "transpose_b", False)
  float_graph_def.node.extend([mat_mul_node])

  test_graph(float_graph_def, {}, [mat_mul_name])


def test_conv(depth, image_width, image_height, image_batch_count, filter_size,
              filter_count, stride, padding, input_values, filter_values):
  """Tests a Conv replacement."""
  input_constant_name = "input_constant"
  filter_constant_name = "filter_constant"
  conv_name = "conv"

  float_graph_def = tf.GraphDef()
  input_constant = quantize_graph.create_constant_node(
      input_constant_name,
      value=input_values,
      dtype=tf.float32,
      shape=[
          image_batch_count, image_height, image_width, depth
      ])
  float_graph_def.node.extend([input_constant])
  filter_constant = quantize_graph.create_constant_node(
      filter_constant_name,
      value=filter_values,
      dtype=tf.float32,
      shape=[
          filter_size, filter_size, depth, filter_count
      ])
  float_graph_def.node.extend([filter_constant])
  conv_node = quantize_graph.create_node("Conv2D", conv_name,
                                         [input_constant_name,
                                          filter_constant_name])
  quantize_graph.set_attr_dtype(conv_node, "T", tf.float32)
  quantize_graph.set_attr_int_list(conv_node, "strides", [1, stride, stride, 1])
  quantize_graph.set_attr_string(conv_node, "padding", padding)
  float_graph_def.node.extend([conv_node])

  test_graph(float_graph_def, {}, [conv_name])


def are_tensors_near(a, b, tolerance):
  """Tests whether two tensors are nearly identical.

  This is a specialized comparison function designed to help debug problems with
  quantization. It prints out information about the differences between tensors
  on failure, paying special attention to possible biases by looking at the mean
  and absolute average errors.

  Args:
    a: First comparison tensor.
    b: Second comparison tensor.
    tolerance: Float value indicating how large an error between values is ok.

  Returns:
    Boolean indicating whether the two inputs were close enough.
  """
  flat_a = a.flatten()
  flat_b = b.flatten()
  if len(flat_a) != len(flat_b):
    print("Tensors are different sizes: " + str(len(flat_a)) + " vs " +
          str(len(flat_b)))
    return False
  value_count = len(flat_a)
  how_many_different = 0
  total_difference = 0
  total_abs_difference = 0
  for index in range(value_count):
    a_value = flat_a[index]
    b_value = flat_b[index]
    difference = a_value - b_value
    total_difference += difference
    total_abs_difference += abs(difference)
    if abs(difference) > tolerance:
      how_many_different += 1
  mean_difference = total_difference / value_count
  mean_abs_difference = total_abs_difference / value_count
  proportion_different = (how_many_different * 1.0) / value_count
  if how_many_different == 0:
    return True
  else:
    print("Tensors have {0} different values ({1}%), with mean difference"
          " {2} and mean absolute difference {3}".format(
              how_many_different, proportion_different * 100, mean_difference,
              mean_abs_difference))
    return False


def get_top_value(input_values):
  max_value = None
  max_index = None
  for index, value in enumerate(input_values.flatten()):
    if max_value is None or value > max:
      max_value = value
      max_index = index
  return max_index, max_value


def test_graph(float_graph_def, input_map, output_names):
  """Runs the float graph through the rewriter and tests the results."""
  float_results = run_graph_def(float_graph_def, input_map,
                                [output_name + ":0"
                                 for output_name in output_names])
  # TODO(petewarden): round test is currently failing because there is no
  # RoundToSteps op available.
  # round_rewriter = quantize_graph.GraphRewriter(float_graph_def, "round")
  # round_graph_def = round_rewriter.rewrite(output_name)
  # round_results = run_graph_def(round_graph_def, input_map,
  #                               [output_name + ":0"])
  # assert are_tensors_near(expected, round_results[0], 1.0)
  #
  # TODO(petewarden): Add test for "quantize" mode.

  eightbit_rewriter = quantize_graph.GraphRewriter(float_graph_def, "eightbit")
  eightbit_graph_def = eightbit_rewriter.rewrite(output_names)
  eightbit_results = run_graph_def(eightbit_graph_def, input_map,
                                   [output_name + ":0"
                                    for output_name in output_names])
  for expected, result in zip(float_results, eightbit_results):
    assert are_tensors_near(expected, result, 1.0)

  # Test the weights_rounded mode. This uses the default bit_depth.
  weights_rounded_rewriter = quantize_graph.GraphRewriter(
      float_graph_def, "weights_rounded")
  weights_rounded_graph_def = weights_rounded_rewriter.rewrite(output_names)
  weights_rounded_results = run_graph_def(weights_rounded_graph_def, input_map,
                                          [output_name + ":0"
                                           for output_name in output_names])
  for expected, result in zip(float_results, weights_rounded_results):
    assert are_tensors_near(expected, result, 1.0)


class QuantizeGraphTest(tf.test.TestCase):

  def test_negative_const_problem(self):
    shape_constant_name = "shape_constant"
    shape_constant = quantize_graph.create_constant_node(
        shape_constant_name, value=-0.8, dtype=tf.float32, shape=[1])
    quantization_result = quantize_graph.quantize_weight_eightbit(
        shape_constant, b"MIN_COMBINED")
    self.assertEqual(4, len(quantization_result))

  def test_odd_padding_problem(self):
    """Tests one error case we ran into in a real graph."""
    test_conv(1, 4, 4, 1, 3, 1, 2, b"SAME",
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
              [1, 2, 3, 4, 5, 6, 7, 8, 9])

  def test_mat_mul_tiny(self):
    # These tests are added to test the generate case where
    # min(matrix) == max(matrix), which used to cause problems.
    test_mat_mul(1, 1, 1, [2], [3])
    test_mat_mul(1, 2, 1, [1], [2, 3])
    test_mat_mul(1, 1, 2, [1, 1], [1, 1])
    test_mat_mul(1, 1, 2, [0, 0], [1, 1])
    # The general case.
    test_mat_mul(1, 1, 2, [1, 2], [1, 2])

  def test_mat_mul_small(self):
    test_mat_mul(2, 4, 3, [1, 2, 3, 4, 5, 6],
                 [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

  def test_conv(self):
    test_conv(1, 4, 3, 1, 3, 1, 1, b"SAME",
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
              [1, 4, 7, 2, 5, 8, 3, 6, 9])

  def test_quantize_array(self):
    # Test invalid parameters (empty array, or 0 buckets.
    self.assertRaises(ValueError, quantize_graph.quantize_array,
                      np.array([]), 2)
    self.assertRaises(ValueError, quantize_graph.quantize_array,
                      np.array([1, 2]), 0)
    # Test input array of length 1.
    arr = np.array([1])
    qarr = quantize_graph.quantize_array(arr, 1)
    self.assertEqual(arr, qarr)
    qarr = quantize_graph.quantize_array(arr, 2)
    self.assertEqual(arr, qarr)
    # Test input array with all elements equal.
    arr = np.array([1, 1, 1])
    qarr = quantize_graph.quantize_array(arr, 10)
    self.assertTrue((np.array([1, 1, 1]) == qarr).all())
    # Test "normal" input arrays.
    arr = np.array([0, 0.3, 0.6, 1])
    qarr = quantize_graph.quantize_array(arr, 1)
    self.assertTrue((np.array([0.5, 0.5, 0.5, 0.5]) == qarr).all())
    qarr = quantize_graph.quantize_array(arr, 2)
    self.assertTrue((np.array([0.25, 0.25, 0.75, 0.75]) == qarr).all())
    qarr = quantize_graph.quantize_array(arr.reshape((2, 2)), 2)
    self.assertTrue((np.array([[0.25, 0.25], [0.75, 0.75]]) == qarr).all())

  def test_concat(self):
    shape_constant_name = "shape_constant"
    a_constant_name = "a_constant"
    b_constant_name = "b_constant"
    concat_name = "concat"

    float_graph_def = tf.GraphDef()
    shape_constant = quantize_graph.create_constant_node(shape_constant_name,
                                                         value=0,
                                                         dtype=tf.int32,
                                                         shape=[])
    float_graph_def.node.extend([shape_constant])
    a_constant = quantize_graph.create_constant_node(a_constant_name,
                                                     value=[1, 2, 3, 4, 5, 6, 7,
                                                            8, 9, 10, 11, 12],
                                                     dtype=tf.float32,
                                                     shape=[2, 2, 3])
    float_graph_def.node.extend([a_constant])
    b_constant = quantize_graph.create_constant_node(b_constant_name,
                                                     value=[13, 14, 15, 16, 17,
                                                            18, 19, 20, 21, 22,
                                                            23, 24],
                                                     dtype=tf.float32,
                                                     shape=[2, 2, 3])
    float_graph_def.node.extend([b_constant])
    concat_node = quantize_graph.create_node("Concat", concat_name,
                                             [shape_constant_name,
                                              a_constant_name, b_constant_name])
    quantize_graph.set_attr_int(concat_node, "N", 2)
    quantize_graph.set_attr_dtype(concat_node, "T", tf.float32)
    float_graph_def.node.extend([concat_node])

    test_graph(float_graph_def, {}, [concat_name])

  def test_multiple_outputs(self):
    input_constant_name = "input_constant"
    split_constant_name = "split_constant"
    split_name = "split"
    concat_constant_name = "concat_constant"
    concat_name = "concat"

    float_graph_def = tf.GraphDef()
    input_constant = quantize_graph.create_constant_node(input_constant_name,
                                                         value=[1, 2, 3, 4, 5,
                                                                6, 7, 8, 9, 10,
                                                                11, 12],
                                                         dtype=tf.float32,
                                                         shape=[2, 6])
    float_graph_def.node.extend([input_constant])
    split_constant = quantize_graph.create_constant_node(split_constant_name,
                                                         value=1,
                                                         dtype=tf.int32,
                                                         shape=[])
    float_graph_def.node.extend([split_constant])
    split_node = quantize_graph.create_node("Split", split_name,
                                            [split_constant_name,
                                             input_constant_name])
    quantize_graph.set_attr_int(split_node, "num_split", 2)
    quantize_graph.set_attr_dtype(split_node, "T", tf.float32)
    float_graph_def.node.extend([split_node])
    concat_constant = quantize_graph.create_constant_node(concat_constant_name,
                                                          value=1,
                                                          dtype=tf.int32,
                                                          shape=[])
    float_graph_def.node.extend([concat_constant])
    concat_node = quantize_graph.create_node("Concat", concat_name,
                                             [concat_constant_name,
                                              split_name + ":0",
                                              split_name + ":1"])
    quantize_graph.set_attr_int(concat_node, "N", 2)
    quantize_graph.set_attr_dtype(concat_node, "T", tf.float32)
    float_graph_def.node.extend([concat_node])

    test_graph(float_graph_def, {}, [concat_name])

  def test_node_name_from_input(self):
    self.assertEqual("SomeName",
                     quantize_graph.node_name_from_input("^SomeName:2"))

  def test_unique_node_name_from_input(self):
    self.assertEqual("__hat__SomeName__port__2",
                     quantize_graph.unique_node_name_from_input("^SomeName:2"))

  def test_identity(self):
    input_constant_name = "input_constant"
    identity_name = "identity"
    float_graph_def = tf.GraphDef()
    input_constant = quantize_graph.create_constant_node(input_constant_name,
                                                         value=[1, 2, 3, 4, 5,
                                                                6, 7, 8, 9, 10,
                                                                11, 12],
                                                         dtype=tf.float32,
                                                         shape=[2, 6])
    float_graph_def.node.extend([input_constant])
    identity_node = quantize_graph.create_node("Identity", identity_name,
                                               [input_constant_name])
    quantize_graph.set_attr_dtype(identity_node, "T", tf.float32)
    float_graph_def.node.extend([identity_node])
    test_graph(float_graph_def, {}, [identity_name])

  def test_keep_control_edges(self):
    no_op_name = "no_op"
    a_constant_name = "a_constant"
    b_constant_name = "b_constant"
    a_check_name = "a_check"
    b_check_name = "b_check"
    a_identity_name = "a_identity"
    b_identity_name = "b_identity"
    add_name = "add"
    graph_def = tf.GraphDef()
    no_op = quantize_graph.create_node("NoOp", no_op_name, [])
    graph_def.node.extend([no_op])
    a_constant = quantize_graph.create_constant_node(a_constant_name,
                                                     value=1,
                                                     dtype=tf.float32,
                                                     shape=[])
    graph_def.node.extend([a_constant])
    a_check_node = quantize_graph.create_node("CheckNumerics", a_check_name,
                                              [a_constant_name])
    graph_def.node.extend([a_check_node])
    a_identity_node = quantize_graph.create_node("Identity", a_identity_name,
                                                 [a_constant_name,
                                                  "^" + a_check_name,
                                                  "^" + no_op_name])
    graph_def.node.extend([a_identity_node])
    b_constant = quantize_graph.create_constant_node(b_constant_name,
                                                     value=1,
                                                     dtype=tf.float32,
                                                     shape=[])
    graph_def.node.extend([b_constant])
    b_check_node = quantize_graph.create_node("CheckNumerics", b_check_name,
                                              [b_constant_name])
    graph_def.node.extend([b_check_node])
    b_identity_node = quantize_graph.create_node("Identity", b_identity_name,
                                                 [b_constant_name,
                                                  "^" + b_check_name])
    graph_def.node.extend([b_identity_node])
    add_node = quantize_graph.create_node("Add", add_name,
                                          [a_identity_name,
                                           b_identity_name])
    quantize_graph.set_attr_dtype(add_node, "T", tf.float32)
    graph_def.node.extend([add_node])

    expected_output = tf.GraphDef()
    no_op = quantize_graph.create_node("NoOp", no_op_name, [])
    expected_output.node.extend([no_op])
    a_constant = quantize_graph.create_constant_node(a_constant_name,
                                                     value=1,
                                                     dtype=tf.float32,
                                                     shape=[])
    expected_output.node.extend([a_constant])
    a_identity_node = quantize_graph.create_node("Identity", a_identity_name,
                                                 [a_constant_name,
                                                  "^" + no_op_name])
    expected_output.node.extend([a_identity_node])
    b_constant = quantize_graph.create_constant_node(b_constant_name,
                                                     value=1,
                                                     dtype=tf.float32,
                                                     shape=[])
    expected_output.node.extend([b_constant])
    add_node = quantize_graph.create_node("Add", add_name,
                                          [a_identity_name,
                                           b_constant_name])
    quantize_graph.set_attr_dtype(add_node, "T", tf.float32)
    expected_output.node.extend([add_node])

    output = graph_util.remove_training_nodes(graph_def)
    stripped_output = graph_util.extract_sub_graph(output, [add_name])
    self.assertProtoEquals(expected_output, stripped_output)

  def test_batch_norm(self):
    input_constant_name = "input_constant"
    mean_constant_name = "mean_constant"
    variance_constant_name = "variance_constant"
    beta_constant_name = "beta_constant"
    gamma_constant_name = "gamma_constant"
    batch_norm_name = "batch_norm"
    float_graph_def = tf.GraphDef()
    input_constant = quantize_graph.create_constant_node(input_constant_name,
                                                         value=[1, 4, 2, 5, 3,
                                                                6, -1, -4, -2,
                                                                -5, -3, -6],
                                                         dtype=tf.float32,
                                                         shape=[1, 1, 6, 2])
    float_graph_def.node.extend([input_constant])
    mean_constant = quantize_graph.create_constant_node(mean_constant_name,
                                                        value=[10, 20],
                                                        dtype=tf.float32,
                                                        shape=[2])
    float_graph_def.node.extend([mean_constant])
    variance_constant = quantize_graph.create_constant_node(
        variance_constant_name, value=[0.25, 0.5], dtype=tf.float32, shape=[2])
    float_graph_def.node.extend([variance_constant])
    beta_constant = quantize_graph.create_constant_node(beta_constant_name,
                                                        value=[0.1, 0.6],
                                                        dtype=tf.float32,
                                                        shape=[2])
    float_graph_def.node.extend([beta_constant])
    gamma_constant = quantize_graph.create_constant_node(gamma_constant_name,
                                                         value=[0, 0],
                                                         dtype=tf.float32,
                                                         shape=[2])
    float_graph_def.node.extend([gamma_constant])
    batch_norm_node = quantize_graph.create_node(
        "BatchNormWithGlobalNormalization", batch_norm_name,
        [input_constant_name, mean_constant_name, variance_constant_name,
         beta_constant_name, gamma_constant_name])
    quantize_graph.set_attr_dtype(batch_norm_node, "T", tf.float32)
    quantize_graph.set_attr_bool(batch_norm_node, "scale_after_normalization",
                                 False)
    quantize_graph.set_attr_float(batch_norm_node, "variance_epsilon", 0.001)
    float_graph_def.node.extend([batch_norm_node])
    test_graph(float_graph_def, {}, [batch_norm_name])

  def test_max_pool(self):
    input_constant_name = "input_constant"
    max_pool_name = "max_pool"
    float_graph_def = tf.GraphDef()
    input_constant = quantize_graph.create_constant_node(input_constant_name,
                                                         value=[1, 2, 3, 4, 5,
                                                                6, 7, 8, 9, 10,
                                                                11, 12],
                                                         dtype=tf.float32,
                                                         shape=[1, 2, 6, 1])
    float_graph_def.node.extend([input_constant])
    max_pool_node = quantize_graph.create_node("MaxPool", max_pool_name,
                                               [input_constant_name])
    quantize_graph.set_attr_int_list(max_pool_node, "ksize", [1, 2, 2, 1])
    quantize_graph.set_attr_int_list(max_pool_node, "strides", [1, 1, 1, 1])
    quantize_graph.set_attr_string(max_pool_node, "padding", b"SAME")
    float_graph_def.node.extend([max_pool_node])
    test_graph(float_graph_def, {}, [max_pool_name])

  def test_avg_pool(self):
    input_constant_name = "input_constant"
    avg_pool_name = "avg_pool"
    float_graph_def = tf.GraphDef()
    input_constant = quantize_graph.create_constant_node(input_constant_name,
                                                         value=[1, 2, 3, 4, 5,
                                                                6, 7, 8, 9, 10,
                                                                11, 12],
                                                         dtype=tf.float32,
                                                         shape=[1, 2, 6, 1])
    float_graph_def.node.extend([input_constant])
    avg_pool_node = quantize_graph.create_node("AvgPool", avg_pool_name,
                                               [input_constant_name])
    quantize_graph.set_attr_dtype(avg_pool_node, "T", tf.float32)
    quantize_graph.set_attr_int_list(avg_pool_node, "ksize", [1, 2, 2, 1])
    quantize_graph.set_attr_int_list(avg_pool_node, "strides", [1, 1, 1, 1])
    quantize_graph.set_attr_string(avg_pool_node, "padding", b"SAME")
    float_graph_def.node.extend([avg_pool_node])
    test_graph(float_graph_def, {}, [avg_pool_name])

  def test_relu(self):
    input_constant_name = "input_constant"
    relu_name = "relu"
    float_graph_def = tf.GraphDef()
    input_constant = quantize_graph.create_constant_node(input_constant_name,
                                                         value=[1, 2, 3, 4, 5,
                                                                6, 7, 8, 9, 10,
                                                                11, 12],
                                                         dtype=tf.float32,
                                                         shape=[1, 2, 6, 1])
    float_graph_def.node.extend([input_constant])
    relu_node = quantize_graph.create_node("Relu", relu_name,
                                           [input_constant_name])
    quantize_graph.set_attr_dtype(relu_node, "T", tf.float32)
    float_graph_def.node.extend([relu_node])
    test_graph(float_graph_def, {}, [relu_name])

  def test_relu6(self):
    input_constant_name = "input_constant"
    relu6_name = "relu6"
    float_graph_def = tf.GraphDef()
    input_constant = quantize_graph.create_constant_node(input_constant_name,
                                                         value=[1, 2, 3, 4, 5,
                                                                6, 7, 8, 9, 10,
                                                                11, 12],
                                                         dtype=tf.float32,
                                                         shape=[1, 2, 6, 1])
    float_graph_def.node.extend([input_constant])
    relu6_node = quantize_graph.create_node("Relu6", relu6_name,
                                            [input_constant_name])
    quantize_graph.set_attr_dtype(relu6_node, "T", tf.float32)
    float_graph_def.node.extend([relu6_node])
    test_graph(float_graph_def, {}, [relu6_name])

  def test_bias_add(self):
    input_constant_name = "input_constant"
    offset_constant_name = "offset_constant"
    bias_add_name = "bias_add"
    float_graph_def = tf.GraphDef()
    input_constant = quantize_graph.create_constant_node(input_constant_name,
                                                         value=[1, 2, 3, 4, 5,
                                                                6, 7, 8, 9, 10,
                                                                11, 12],
                                                         dtype=tf.float32,
                                                         shape=[1, 1, 2, 6])
    float_graph_def.node.extend([input_constant])
    offset_constant = quantize_graph.create_constant_node(offset_constant_name,
                                                          value=[1, 2, 3, 4, 5,
                                                                 6],
                                                          dtype=tf.float32,
                                                          shape=[6])
    float_graph_def.node.extend([offset_constant])
    bias_add_node = quantize_graph.create_node("BiasAdd", bias_add_name,
                                               [input_constant_name,
                                                offset_constant_name])
    quantize_graph.set_attr_dtype(bias_add_node, "T", tf.float32)
    float_graph_def.node.extend([bias_add_node])
    test_graph(float_graph_def, {}, [bias_add_name])

  def test_remove_redundant_quantization(self):
    a_constant_name = "a_constant"
    a_constant_min_name = "a_constant_min"
    a_constant_max_name = "a_constant_max"
    a_dequantize_name = "a_dequantize"
    a_quantize_name = "a_quantize"
    b_constant_name = "b_constant"
    b_constant_min_name = "b_constant_min"
    b_constant_max_name = "b_constant_max"
    b_dequantize_name = "b_dequantize"
    b_quantize_name = "b_quantize"
    mat_mul_name = "mat_mul"
    graph_def = tf.GraphDef()
    a_constant = quantize_graph.create_constant_node(a_constant_name,
                                                     value=(0,),
                                                     dtype=tf.quint8,
                                                     shape=[])
    graph_def.node.extend([a_constant])
    a_constant_min = quantize_graph.create_constant_node(a_constant_min_name,
                                                         value=2,
                                                         dtype=tf.float32,
                                                         shape=[])
    graph_def.node.extend([a_constant_min])
    a_constant_max = quantize_graph.create_constant_node(a_constant_max_name,
                                                         value=2,
                                                         dtype=tf.float32,
                                                         shape=[])
    graph_def.node.extend([a_constant_max])
    a_dequantize_node = quantize_graph.create_node("Dequantize",
                                                   a_dequantize_name,
                                                   [a_constant_name,
                                                    a_constant_min_name,
                                                    a_constant_max_name])
    quantize_graph.set_attr_dtype(a_dequantize_node, "T", tf.uint8)
    graph_def.node.extend([a_dequantize_node])
    a_quantize_node = quantize_graph.create_node("QuantizeV2",
                                                 a_quantize_name,
                                                 [a_dequantize_name,
                                                  a_dequantize_name + ":1",
                                                  a_dequantize_name + ":2"])
    quantize_graph.set_attr_dtype(a_quantize_node, "T", tf.uint8)
    graph_def.node.extend([a_quantize_node])
    b_constant = quantize_graph.create_constant_node(b_constant_name,
                                                     value=(0,),
                                                     dtype=tf.quint8,
                                                     shape=[])
    graph_def.node.extend([b_constant])
    b_constant_min = quantize_graph.create_constant_node(b_constant_min_name,
                                                         value=3,
                                                         dtype=tf.float32,
                                                         shape=[])
    graph_def.node.extend([b_constant_min])
    b_constant_max = quantize_graph.create_constant_node(b_constant_max_name,
                                                         value=3,
                                                         dtype=tf.float32,
                                                         shape=[])
    graph_def.node.extend([b_constant_max])
    b_dequantize_node = quantize_graph.create_node("Dequantize",
                                                   b_dequantize_name,
                                                   [b_constant_name,
                                                    b_constant_min_name,
                                                    b_constant_max_name])
    quantize_graph.set_attr_dtype(b_dequantize_node, "T", tf.uint8)
    graph_def.node.extend([b_dequantize_node])
    b_quantize_node = quantize_graph.create_node("QuantizeV2",
                                                 b_quantize_name,
                                                 [b_dequantize_name,
                                                  b_dequantize_name + ":1",
                                                  b_dequantize_name + ":2"])
    quantize_graph.set_attr_dtype(b_quantize_node, "T", tf.uint8)
    graph_def.node.extend([b_quantize_node])
    mat_mul_node = quantize_graph.create_node("QuantizedMatMul", mat_mul_name,
                                              [a_quantize_name,
                                               b_quantize_name,
                                               a_quantize_name + ":1",
                                               a_quantize_name + ":2",
                                               b_quantize_name + ":1",
                                               b_quantize_name + ":2"])
    quantize_graph.set_attr_dtype(mat_mul_node, "T1", tf.uint8)
    quantize_graph.set_attr_dtype(mat_mul_node, "T2", tf.int32)
    graph_def.node.extend([mat_mul_node])

    expected_output = tf.GraphDef()
    a_constant = quantize_graph.create_constant_node(a_constant_name,
                                                     value=(0,),
                                                     dtype=tf.quint8,
                                                     shape=[])
    expected_output.node.extend([a_constant])
    a_constant_min = quantize_graph.create_constant_node(a_constant_min_name,
                                                         value=2,
                                                         dtype=tf.float32,
                                                         shape=[])
    expected_output.node.extend([a_constant_min])
    a_constant_max = quantize_graph.create_constant_node(a_constant_max_name,
                                                         value=2,
                                                         dtype=tf.float32,
                                                         shape=[])
    expected_output.node.extend([a_constant_max])
    b_constant = quantize_graph.create_constant_node(b_constant_name,
                                                     value=(0,),
                                                     dtype=tf.quint8,
                                                     shape=[])
    expected_output.node.extend([b_constant])
    b_constant_min = quantize_graph.create_constant_node(b_constant_min_name,
                                                         value=3,
                                                         dtype=tf.float32,
                                                         shape=[])
    expected_output.node.extend([b_constant_min])
    b_constant_max = quantize_graph.create_constant_node(b_constant_max_name,
                                                         value=3,
                                                         dtype=tf.float32,
                                                         shape=[])
    expected_output.node.extend([b_constant_max])
    mat_mul_node = quantize_graph.create_node("QuantizedMatMul", mat_mul_name,
                                              [a_constant_name,
                                               b_constant_name,
                                               a_constant_min_name,
                                               a_constant_max_name,
                                               b_constant_min_name,
                                               b_constant_max_name])
    quantize_graph.set_attr_dtype(mat_mul_node, "T1", tf.uint8)
    quantize_graph.set_attr_dtype(mat_mul_node, "T2", tf.int32)
    expected_output.node.extend([mat_mul_node])

    rewriter = quantize_graph.GraphRewriter(graph_def, [mat_mul_name])
    output = rewriter.remove_redundant_quantization(graph_def)
    stripped_output = graph_util.extract_sub_graph(output, [mat_mul_name])
    self.assertProtoEquals(expected_output, stripped_output)


if __name__ == "__main__":
  tf.test.main()
