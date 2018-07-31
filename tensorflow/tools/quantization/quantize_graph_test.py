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

import sys
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.tools.quantization import quantize_graph

flags = flags_lib
FLAGS = flags.FLAGS


def run_graph_def(graph_def, input_map, outputs):
  graph = ops_lib.Graph()
  with graph.as_default():
    importer.import_graph_def(graph_def, input_map={}, name="")
  with session.Session(graph=graph) as sess:
    results = sess.run(outputs, feed_dict=input_map)
  return results


def test_mat_mul(m, n, k, a, b):
  """Tests a MatMul replacement."""
  a_constant_name = "a_constant"
  b_constant_name = "b_constant"
  mat_mul_name = "mat_mul"

  float_graph_def = graph_pb2.GraphDef()
  a_constant = quantize_graph.create_constant_node(
      a_constant_name, value=a, dtype=dtypes.float32, shape=[m, k])
  float_graph_def.node.extend([a_constant])
  b_constant = quantize_graph.create_constant_node(
      b_constant_name, value=b, dtype=dtypes.float32, shape=[k, n])
  float_graph_def.node.extend([b_constant])
  mat_mul_node = quantize_graph.create_node("MatMul", mat_mul_name,
                                            [a_constant_name, b_constant_name])
  quantize_graph.set_attr_dtype(mat_mul_node, "T", dtypes.float32)
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

  float_graph_def = graph_pb2.GraphDef()
  input_constant = quantize_graph.create_constant_node(
      input_constant_name,
      value=input_values,
      dtype=dtypes.float32,
      shape=[image_batch_count, image_height, image_width, depth])
  float_graph_def.node.extend([input_constant])
  filter_constant = quantize_graph.create_constant_node(
      filter_constant_name,
      value=filter_values,
      dtype=dtypes.float32,
      shape=[filter_size, filter_size, depth, filter_count])
  float_graph_def.node.extend([filter_constant])
  conv_node = quantize_graph.create_node(
      "Conv2D", conv_name, [input_constant_name, filter_constant_name])
  quantize_graph.set_attr_dtype(conv_node, "T", dtypes.float32)
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
    tf_logging.info("Tensors are different sizes: " + str(len(flat_a)) + " vs "
                    + str(len(flat_b)))
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
    tf_logging.info("Tensors have {0} different values ({1}%), with mean"
                    " difference {2} and mean absolute difference {3}".format(
                        how_many_different, proportion_different * 100,
                        mean_difference, mean_abs_difference))
    return False


def get_top_value(input_values):
  max_value = None
  max_index = None
  for index, value in enumerate(input_values.flatten()):
    if max_value is None or value > max:
      max_value = value
      max_index = index
  return max_index, max_value


def test_graph(float_graph_def, input_map, output_names, log_graph=False):
  """Runs the float graph through the rewriter and tests the results."""
  float_results = run_graph_def(
      float_graph_def, input_map,
      [output_name + ":0" for output_name in output_names])
  # TODO(petewarden): round test is currently failing because there is no
  # RoundToSteps op available.
  # round_rewriter = quantize_graph.GraphRewriter(float_graph_def, "round")
  # round_graph_def = round_rewriter.rewrite(output_name)
  # round_results = run_graph_def(round_graph_def, input_map,
  #                               [output_name + ":0"])
  # assert are_tensors_near(expected, round_results[0], 1.0)
  #
  # TODO(petewarden): Add test for "quantize" mode.

  eightbit_rewriter = quantize_graph.GraphRewriter(
      float_graph_def, "eightbit", quantized_input_range=None)
  eightbit_graph_def = eightbit_rewriter.rewrite(output_names)
  eightbit_results = run_graph_def(
      eightbit_graph_def, input_map,
      [output_name + ":0" for output_name in output_names])
  for expected, result in zip(float_results, eightbit_results):
    assert are_tensors_near(expected, result, 1.0)

  if log_graph:
    tf_logging.info("8bit:\n%s", str(eightbit_graph_def))

  # Test the weights_rounded mode. This uses the default bit_depth.
  weights_rounded_rewriter = quantize_graph.GraphRewriter(
      float_graph_def, "weights_rounded", quantized_input_range=None)
  weights_rounded_graph_def = weights_rounded_rewriter.rewrite(output_names)
  weights_rounded_results = run_graph_def(
      weights_rounded_graph_def, input_map,
      [output_name + ":0" for output_name in output_names])
  for expected, result in zip(float_results, weights_rounded_results):
    assert are_tensors_near(expected, result, 1.0)


class QuantizeGraphTest(test.TestCase):

  def test_negative_const_problem(self):
    shape_constant_name = "shape_constant"
    shape_constant = quantize_graph.create_constant_node(
        shape_constant_name, value=-0.8, dtype=dtypes.float32, shape=[1])
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

  def test_reshape(self):
    """Tests that MatMul->Reshape->MatMul avoids extra quantize/dequantize."""

    def make_matmul(name, a, b):
      n = quantize_graph.create_node("MatMul", name, [a.name, b.name])
      quantize_graph.set_attr_dtype(n, "T", dtypes.float32)
      quantize_graph.set_attr_bool(n, "transpose_a", False)
      quantize_graph.set_attr_bool(n, "transpose_b", False)
      return n

    # matmul_1 = input*weight_1
    input_node = quantize_graph.create_constant_node(
        "input", value=[0, 1, 2, 3], dtype=dtypes.float32, shape=[4, 1])
    weight_1_node = quantize_graph.create_constant_node(
        "weight_1",
        value=[.5, .6, .7, .8, .9],
        dtype=dtypes.float32,
        shape=[1, 5])
    matmul_1_node = make_matmul("matmul_1", input_node, weight_1_node)

    # Reshape 4x5 to 10x2.
    new_shape_node = quantize_graph.create_constant_node(
        "new_shape_node", value=[10, 2], dtype=dtypes.int32, shape=[2])
    reshape_node = quantize_graph.create_node(
        "Reshape", "reshape", [matmul_1_node.name, new_shape_node.name])
    quantize_graph.set_attr_dtype(reshape_node, "T", dtypes.float32)

    # matmul_2_node = reshape*weight_2
    weight_2_node = quantize_graph.create_constant_node(
        "weight_2", value=[1.5, 2.5], dtype=dtypes.float32, shape=[2, 1])
    matmul_2_node = make_matmul("matmul_2", reshape_node, weight_2_node)

    g = graph_pb2.GraphDef()
    g.node.extend([
        input_node, weight_1_node, matmul_1_node, new_shape_node, reshape_node,
        weight_2_node, matmul_2_node
    ])

    # Test the graph
    test_graph(g, {}, ["matmul_2"])

    # Verify there is only one Quantize and one Requantize op.
    eightbit_rewriter = quantize_graph.GraphRewriter(
        g, "eightbit", quantized_input_range=None)
    eightbit_graph_def = eightbit_rewriter.rewrite(["matmul_2"])

    ops = [node.op for node in eightbit_graph_def.node]
    # No quantize since all inputs are const and can be quantized up-front.
    self.assertEqual(0, ops.count("QuantizeV2") + ops.count("Quantize"))
    self.assertEqual(1, ops.count("QuantizedReshape"))

    # One dequantize at the end.
    self.assertEqual(1, ops.count("Dequantize"))

  def test_quantize_array(self):
    # Test invalid parameters (empty array, or 0 buckets.
    self.assertRaises(ValueError, quantize_graph.quantize_array, np.array([]),
                      2)
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

  def test_non_float_concat(self):
    concat_dim = quantize_graph.create_constant_node(
        "concat_dim", value=0, dtype=dtypes.int32, shape=[])
    a = quantize_graph.create_constant_node(
        "a",
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.int32,
        shape=[2, 2, 3])
    b = quantize_graph.create_constant_node(
        "b",
        value=[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        dtype=dtypes.int32,
        shape=[2, 2, 3])
    concat = quantize_graph.create_node("Concat", "concat",
                                        [concat_dim.name, a.name, b.name])
    quantize_graph.set_attr_int(concat, "N", 2)
    quantize_graph.set_attr_dtype(concat, "T", dtypes.int32)

    g = graph_pb2.GraphDef()
    g.node.extend([concat_dim, a, b, concat])
    test_graph(g, {}, [concat.name])

  def test_non_float_reshape(self):
    a = quantize_graph.create_constant_node(
        "a",
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.int32,
        shape=[2, 2, 3])
    shape = quantize_graph.create_constant_node(
        "shape", value=[12], dtype=dtypes.int32, shape=[1])
    reshape = quantize_graph.create_node("Reshape", "reshape",
                                         [a.name, shape.name])
    quantize_graph.set_attr_dtype(reshape, "T", dtypes.int32)

    g = graph_pb2.GraphDef()
    g.node.extend([a, shape, reshape])
    test_graph(g, {}, [reshape.name])

  def test_concat(self):
    shape_constant_name = "shape_constant"
    a_constant_name = "a_constant"
    b_constant_name = "b_constant"
    concat_name = "concat"

    float_graph_def = graph_pb2.GraphDef()
    shape_constant = quantize_graph.create_constant_node(
        shape_constant_name, value=0, dtype=dtypes.int32, shape=[])
    float_graph_def.node.extend([shape_constant])
    a_constant = quantize_graph.create_constant_node(
        a_constant_name,
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.float32,
        shape=[2, 2, 3])
    float_graph_def.node.extend([a_constant])
    b_constant = quantize_graph.create_constant_node(
        b_constant_name,
        value=[13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        dtype=dtypes.float32,
        shape=[2, 2, 3])
    float_graph_def.node.extend([b_constant])
    concat_node = quantize_graph.create_node(
        "Concat", concat_name,
        [shape_constant_name, a_constant_name, b_constant_name])
    quantize_graph.set_attr_int(concat_node, "N", 2)
    quantize_graph.set_attr_dtype(concat_node, "T", dtypes.float32)
    float_graph_def.node.extend([concat_node])

    test_graph(float_graph_def, {}, [concat_name])

    # Verify the concat is quantized.
    eightbit_rewriter = quantize_graph.GraphRewriter(
        float_graph_def, "eightbit", quantized_input_range=None)
    eightbit_graph_def = eightbit_rewriter.rewrite([concat_name])

    ops = [node.op for node in eightbit_graph_def.node]
    self.assertEqual(1, ops.count("QuantizedConcat"))

  def test_multiple_outputs(self):
    input_constant_name = "input_constant"
    split_constant_name = "split_constant"
    split_name = "split"
    concat_constant_name = "concat_constant"
    concat_name = "concat"

    float_graph_def = graph_pb2.GraphDef()
    input_constant = quantize_graph.create_constant_node(
        input_constant_name,
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.float32,
        shape=[2, 6])
    float_graph_def.node.extend([input_constant])
    split_constant = quantize_graph.create_constant_node(
        split_constant_name, value=1, dtype=dtypes.int32, shape=[])
    float_graph_def.node.extend([split_constant])
    split_node = quantize_graph.create_node(
        "Split", split_name, [split_constant_name, input_constant_name])
    quantize_graph.set_attr_int(split_node, "num_split", 2)
    quantize_graph.set_attr_dtype(split_node, "T", dtypes.float32)
    float_graph_def.node.extend([split_node])
    concat_constant = quantize_graph.create_constant_node(
        concat_constant_name, value=1, dtype=dtypes.int32, shape=[])
    float_graph_def.node.extend([concat_constant])
    concat_node = quantize_graph.create_node(
        "Concat", concat_name,
        [concat_constant_name, split_name + ":0", split_name + ":1"])
    quantize_graph.set_attr_int(concat_node, "N", 2)
    quantize_graph.set_attr_dtype(concat_node, "T", dtypes.float32)
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
    float_graph_def = graph_pb2.GraphDef()
    input_constant = quantize_graph.create_constant_node(
        input_constant_name,
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.float32,
        shape=[2, 6])
    float_graph_def.node.extend([input_constant])
    identity_node = quantize_graph.create_node("Identity", identity_name,
                                               [input_constant_name])
    quantize_graph.set_attr_dtype(identity_node, "T", dtypes.float32)
    float_graph_def.node.extend([identity_node])

    mul_name = "mul"
    mul_node = quantize_graph.create_node("Mul", mul_name,
                                          [identity_name, identity_name])
    quantize_graph.set_attr_dtype(mul_node, "T", dtypes.float32)
    float_graph_def.node.extend([mul_node])

    test_graph(float_graph_def, {}, [mul_name])

  def test_keep_control_edges(self):
    no_op_name = "no_op"
    a_constant_name = "a_constant"
    b_constant_name = "b_constant"
    a_check_name = "a_check"
    b_check_name = "b_check"
    a_identity_name = "a_identity"
    b_identity_name = "b_identity"
    add_name = "add"
    graph_def = graph_pb2.GraphDef()
    no_op = quantize_graph.create_node("NoOp", no_op_name, [])
    graph_def.node.extend([no_op])
    a_constant = quantize_graph.create_constant_node(
        a_constant_name, value=1, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([a_constant])
    a_check_node = quantize_graph.create_node("CheckNumerics", a_check_name,
                                              [a_constant_name])
    graph_def.node.extend([a_check_node])
    a_identity_node = quantize_graph.create_node(
        "Identity", a_identity_name,
        [a_constant_name, "^" + a_check_name, "^" + no_op_name])
    graph_def.node.extend([a_identity_node])
    b_constant = quantize_graph.create_constant_node(
        b_constant_name, value=1, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([b_constant])
    b_check_node = quantize_graph.create_node("CheckNumerics", b_check_name,
                                              [b_constant_name])
    graph_def.node.extend([b_check_node])
    b_identity_node = quantize_graph.create_node(
        "Identity", b_identity_name, [b_constant_name, "^" + b_check_name])
    graph_def.node.extend([b_identity_node])
    add_node = quantize_graph.create_node("Add", add_name,
                                          [a_identity_name, b_identity_name])
    quantize_graph.set_attr_dtype(add_node, "T", dtypes.float32)
    graph_def.node.extend([add_node])

    expected_output = graph_pb2.GraphDef()
    no_op = quantize_graph.create_node("NoOp", no_op_name, [])
    expected_output.node.extend([no_op])
    a_constant = quantize_graph.create_constant_node(
        a_constant_name, value=1, dtype=dtypes.float32, shape=[])
    expected_output.node.extend([a_constant])
    a_identity_node = quantize_graph.create_node(
        "Identity", a_identity_name, [a_constant_name, "^" + no_op_name])
    expected_output.node.extend([a_identity_node])
    b_constant = quantize_graph.create_constant_node(
        b_constant_name, value=1, dtype=dtypes.float32, shape=[])
    expected_output.node.extend([b_constant])
    add_node = quantize_graph.create_node("Add", add_name,
                                          [a_identity_name, b_constant_name])
    quantize_graph.set_attr_dtype(add_node, "T", dtypes.float32)
    expected_output.node.extend([add_node])
    expected_output.versions.CopyFrom(graph_def.versions)
    expected_output.library.CopyFrom(graph_def.library)

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
    float_graph_def = graph_pb2.GraphDef()
    input_constant = quantize_graph.create_constant_node(
        input_constant_name,
        value=[1, 4, 2, 5, 3, 6, -1, -4, -2, -5, -3, -6],
        dtype=dtypes.float32,
        shape=[1, 1, 6, 2])
    float_graph_def.node.extend([input_constant])
    mean_constant = quantize_graph.create_constant_node(
        mean_constant_name, value=[10, 20], dtype=dtypes.float32, shape=[2])
    float_graph_def.node.extend([mean_constant])
    variance_constant = quantize_graph.create_constant_node(
        variance_constant_name,
        value=[0.25, 0.5],
        dtype=dtypes.float32,
        shape=[2])
    float_graph_def.node.extend([variance_constant])
    beta_constant = quantize_graph.create_constant_node(
        beta_constant_name, value=[0.1, 0.6], dtype=dtypes.float32, shape=[2])
    float_graph_def.node.extend([beta_constant])
    gamma_constant = quantize_graph.create_constant_node(
        gamma_constant_name, value=[0, 0], dtype=dtypes.float32, shape=[2])
    float_graph_def.node.extend([gamma_constant])
    batch_norm_node = quantize_graph.create_node(
        "BatchNormWithGlobalNormalization", batch_norm_name, [
            input_constant_name, mean_constant_name, variance_constant_name,
            beta_constant_name, gamma_constant_name
        ])
    quantize_graph.set_attr_dtype(batch_norm_node, "T", dtypes.float32)
    quantize_graph.set_attr_bool(batch_norm_node, "scale_after_normalization",
                                 False)
    quantize_graph.set_attr_float(batch_norm_node, "variance_epsilon", 0.001)
    float_graph_def.node.extend([batch_norm_node])
    test_graph(float_graph_def, {}, [batch_norm_name])

  def test_max_pool(self):
    input_constant_name = "input_constant"
    max_pool_name = "max_pool"
    float_graph_def = graph_pb2.GraphDef()
    input_constant = quantize_graph.create_constant_node(
        input_constant_name,
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.float32,
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
    float_graph_def = graph_pb2.GraphDef()
    input_constant = quantize_graph.create_constant_node(
        input_constant_name,
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.float32,
        shape=[1, 2, 6, 1])
    float_graph_def.node.extend([input_constant])
    avg_pool_node = quantize_graph.create_node("AvgPool", avg_pool_name,
                                               [input_constant_name])
    quantize_graph.set_attr_dtype(avg_pool_node, "T", dtypes.float32)
    quantize_graph.set_attr_int_list(avg_pool_node, "ksize", [1, 2, 2, 1])
    quantize_graph.set_attr_int_list(avg_pool_node, "strides", [1, 1, 1, 1])
    quantize_graph.set_attr_string(avg_pool_node, "padding", b"SAME")
    float_graph_def.node.extend([avg_pool_node])
    test_graph(float_graph_def, {}, [avg_pool_name])

  def test_relu(self):
    input_constant_name = "input_constant"
    relu_name = "relu"
    float_graph_def = graph_pb2.GraphDef()
    input_constant = quantize_graph.create_constant_node(
        input_constant_name,
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.float32,
        shape=[1, 2, 6, 1])
    float_graph_def.node.extend([input_constant])
    relu_node = quantize_graph.create_node("Relu", relu_name,
                                           [input_constant_name])
    quantize_graph.set_attr_dtype(relu_node, "T", dtypes.float32)
    float_graph_def.node.extend([relu_node])
    test_graph(float_graph_def, {}, [relu_name])

  def test_relu_w_fake_quant_w_min_max_vars(self):
    input_node = quantize_graph.create_constant_node(
        "input",
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.float32,
        shape=[1, 2, 6, 1])
    relu_node = quantize_graph.create_node("Relu", "relu", [input_node.name])
    quantize_graph.set_attr_dtype(relu_node, "T", dtypes.float32)

    min_node = quantize_graph.create_constant_node(
        "min_bias_add", value=0, dtype=dtypes.float32, shape=[])
    max_node = quantize_graph.create_constant_node(
        "max_bias_add", value=12, dtype=dtypes.float32, shape=[])
    fake_quant_node = quantize_graph.create_node(
        "FakeQuantWithMinMaxVars", "fake_quant",
        [relu_node.name, min_node.name, max_node.name])

    float_graph_def = graph_pb2.GraphDef()
    float_graph_def.node.extend(
        [input_node, relu_node, min_node, max_node, fake_quant_node])
    test_graph(float_graph_def, {}, [fake_quant_node.name], log_graph=True)

    # Verify there is only one Quantize and one Requantize op.
    eightbit_rewriter = quantize_graph.GraphRewriter(
        float_graph_def, "eightbit", quantized_input_range=None)
    eightbit_graph_def = eightbit_rewriter.rewrite([fake_quant_node.name])

    ops = [node.op for node in eightbit_graph_def.node]
    # No quantize since all inputs are const and can be quantized up-front.
    self.assertEqual(0, ops.count("QuantizeV2") + ops.count("Quantize"))

    # One dequantize at the end.
    self.assertEqual(1, ops.count("Dequantize"))

  def test_relu6(self):
    input_constant_name = "input_constant"
    relu6_name = "relu6"
    float_graph_def = graph_pb2.GraphDef()
    input_constant = quantize_graph.create_constant_node(
        input_constant_name,
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.float32,
        shape=[1, 2, 6, 1])
    float_graph_def.node.extend([input_constant])
    relu6_node = quantize_graph.create_node("Relu6", relu6_name,
                                            [input_constant_name])
    quantize_graph.set_attr_dtype(relu6_node, "T", dtypes.float32)
    float_graph_def.node.extend([relu6_node])
    test_graph(float_graph_def, {}, [relu6_name])

  def test_bias_add(self):
    input_constant_name = "input_constant"
    offset_constant_name = "offset_constant"
    bias_add_name = "bias_add"
    float_graph_def = graph_pb2.GraphDef()
    input_constant = quantize_graph.create_constant_node(
        input_constant_name,
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        dtype=dtypes.float32,
        shape=[1, 1, 2, 6])
    float_graph_def.node.extend([input_constant])
    offset_constant = quantize_graph.create_constant_node(
        offset_constant_name,
        value=[1, 2, 3, 4, 5, 6],
        dtype=dtypes.float32,
        shape=[6])
    float_graph_def.node.extend([offset_constant])
    bias_add_node = quantize_graph.create_node(
        "BiasAdd", bias_add_name, [input_constant_name, offset_constant_name])
    quantize_graph.set_attr_dtype(bias_add_node, "T", dtypes.float32)
    float_graph_def.node.extend([bias_add_node])
    test_graph(float_graph_def, {}, [bias_add_name])

  def test_quantized_input_range_errors(self):
    with self.assertRaises(ValueError):
      # Invalid mode.
      quantize_graph.GraphRewriter(graph_pb2.GraphDef(), "weights_rounded",
                                   [0, 1])
    with self.assertRaises(ValueError):
      # Invalid range.
      quantize_graph.GraphRewriter(graph_pb2.GraphDef(), "eightbit", [0, -1])

  def test_quantized_input_range_bias_add(self):
    input_shape = [1, 1, 2, 6]
    input_n = quantize_graph.create_node("Placeholder", "input", [])
    quantize_graph.set_attr_dtype(input_n, "dtype", dtypes.float32)
    quantize_graph.set_attr_shape(input_n, "shape", input_shape)
    offset_n = quantize_graph.create_constant_node(
        "offset", value=[1, 2, 3, 4, 5, 6], dtype=dtypes.float32, shape=[6])
    bias_add_n = quantize_graph.create_node("BiasAdd", "bias_add",
                                            [input_n.name, offset_n.name])
    quantize_graph.set_attr_dtype(bias_add_n, "T", dtypes.float32)

    float_graph_def = graph_pb2.GraphDef()
    float_graph_def.node.extend([input_n, offset_n, bias_add_n])

    input_map = {
        input_n.name + ":0":
            np.reshape([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], input_shape)
    }
    self._RunTestsForQuantizedInputRange(float_graph_def, input_map,
                                         [bias_add_n.name], [-1, 20.])
    self._RunTestsForQuantizedInputRange(float_graph_def, input_map,
                                         [bias_add_n.name], [0, 12.])

  def test_quantized_input_range_mat_mul(self):
    shapes = [[3, 2], [2, 4]]
    inputs = []
    for i, shape in enumerate(shapes):
      node = quantize_graph.create_node("Placeholder", "input_%s" % i, [])
      quantize_graph.set_attr_dtype(node, "dtype", dtypes.float32)
      quantize_graph.set_attr_shape(node, "shape", shape)
      inputs.append(node)
    mat_mul_node = quantize_graph.create_node("MatMul", "mat_mul",
                                              [n.name for n in inputs])
    quantize_graph.set_attr_dtype(mat_mul_node, "T", dtypes.float32)

    float_graph_def = graph_pb2.GraphDef()
    float_graph_def.node.extend(inputs + [mat_mul_node])

    input_map = {
        inputs[0].name + ":0":
            np.reshape([1, 2, 3, 4, 5, 6], shapes[0]),
        inputs[1].name + ":0":
            np.reshape([.8, .7, .6, .5, .4, .3, .2, .1], shapes[1])
    }
    self._RunTestsForQuantizedInputRange(float_graph_def, input_map,
                                         [mat_mul_node.name], [-1, 20.])
    self._RunTestsForQuantizedInputRange(float_graph_def, input_map,
                                         [mat_mul_node.name], [0, 6.])

  def _RunTestsForQuantizedInputRange(self, float_graph_def, input_map,
                                      output_names, input_range):
    if sys.version_info[0] == 3:
      # uint8->quint8 conversion for numpy is not working currently.
      return

    quantized_input_map = {}
    for k, v in input_map.items():
      arr = [
          int(
              round((n - input_range[0]) * 255 / (input_range[1] - input_range[
                  0]))) for n in v.flat
      ]
      arr = np.array(arr, np.uint8)
      arr = arr.reshape(v.shape)
      arr = arr.astype(dtypes.quint8.as_numpy_dtype)
      quantized_input_map[k] = arr
    output_tensors = [output_name + ":0" for output_name in output_names]
    float_results = run_graph_def(float_graph_def, input_map, output_tensors)

    # Quantize treating the input as quantized in range <input_range>.
    rewriter = quantize_graph.GraphRewriter(float_graph_def, "eightbit",
                                            input_range)
    graph_def = rewriter.rewrite(output_names)
    results = run_graph_def(graph_def, quantized_input_map, output_tensors)
    for expected, result in zip(float_results, results):
      assert are_tensors_near(expected, result, .5)
    ops = [node.op for node in graph_def.node]
    self.assertEqual(0, ops.count("QuantizeV2") + ops.count("Quantize"))
    self.assertEqual(len(output_names), ops.count("Dequantize"))

    # Quantize without treating input as quantized.
    rewriter = quantize_graph.GraphRewriter(
        float_graph_def, "eightbit", quantized_input_range=None)
    graph_def = rewriter.rewrite(output_names)
    results = run_graph_def(graph_def, input_map, output_tensors)
    for expected, result in zip(float_results, results):
      assert are_tensors_near(expected, result, .5)
    ops = [node.op for node in graph_def.node]
    self.assertEqual(
        len(input_map), ops.count("QuantizeV2") + ops.count("Quantize"))
    self.assertEqual(len(output_names), ops.count("Dequantize"))

  def test_bias_add_w_fake_quant_w_min_max_vars(self):
    input_node = quantize_graph.create_constant_node(
        "input",
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        dtype=dtypes.float32,
        shape=[1, 1, 2, 5])
    offset_node = quantize_graph.create_constant_node(
        "offset", value=[1, 2, 3, 4, 5], dtype=dtypes.float32, shape=[5])
    bias_add_node = quantize_graph.create_node(
        "BiasAdd", "bias_add", [input_node.name, offset_node.name])
    quantize_graph.set_attr_dtype(bias_add_node, "T", dtypes.float32)

    min_node = quantize_graph.create_constant_node(
        "min_bias_add", value=-.5, dtype=dtypes.float32, shape=[])
    max_node = quantize_graph.create_constant_node(
        "max_bias_add", value=15.5, dtype=dtypes.float32, shape=[])
    fake_quant_node = quantize_graph.create_node(
        "FakeQuantWithMinMaxVars", "fake_quant",
        [bias_add_node.name, min_node.name, max_node.name])

    float_graph_def = graph_pb2.GraphDef()
    float_graph_def.node.extend([
        input_node, offset_node, bias_add_node, min_node, max_node,
        fake_quant_node
    ])
    test_graph(float_graph_def, {}, [fake_quant_node.name], log_graph=True)

    # Verify there is only one Quantize and one Requantize op.
    # Pass in fallback_quantization_range, although it will have no effect
    # because the FakeQuantWithMinMaxVars are used instead.
    eightbit_rewriter = quantize_graph.GraphRewriter(
        float_graph_def,
        "eightbit",
        quantized_input_range=None,
        fallback_quantization_range=[-100, 100])
    eightbit_graph_def = eightbit_rewriter.rewrite([fake_quant_node.name])

    ops = [node.op for node in eightbit_graph_def.node]
    node_names = [node.name for node in eightbit_graph_def.node]
    # No quantize since all inputs are const and can be quantized up-front.
    self.assertEqual(0, ops.count("QuantizeV2") + ops.count("Quantize"))

    # One dequantize at the end.
    self.assertEqual(1, ops.count("Dequantize"))

    # The fallback constants are not in the graph.
    self.assertEqual(0, node_names.count("fallback_quantization_min_value"))
    self.assertEqual(0, node_names.count("fallback_quantization_max_value"))

  def test_bias_add_w_fallback_min_max_vars(self):
    input_node = quantize_graph.create_constant_node(
        "input",
        value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        dtype=dtypes.float32,
        shape=[1, 1, 2, 5])
    offset_node = quantize_graph.create_constant_node(
        "offset", value=[1, 2, 3, 4, 5], dtype=dtypes.float32, shape=[5])
    bias_add_node = quantize_graph.create_node(
        "BiasAdd", "bias_add", [input_node.name, offset_node.name])
    quantize_graph.set_attr_dtype(bias_add_node, "T", dtypes.float32)

    float_graph_def = graph_pb2.GraphDef()
    float_graph_def.node.extend([input_node, offset_node, bias_add_node])
    test_graph(float_graph_def, {}, [bias_add_node.name], log_graph=True)

    # Verify there is only one Quantize, one Requantize op, and no
    # RequantizationRange op.
    eightbit_rewriter = quantize_graph.GraphRewriter(
        float_graph_def,
        "eightbit",
        quantized_input_range=None,
        fallback_quantization_range=[-.5, 15.5])
    eightbit_graph_def = eightbit_rewriter.rewrite([bias_add_node.name])

    ops = [node.op for node in eightbit_graph_def.node]
    node_names = [node.name for node in eightbit_graph_def.node]
    # No quantize since all inputs are const and can be quantized up-front.
    self.assertEqual(0, ops.count("QuantizeV2") + ops.count("Quantize"))

    # One dequantize at the end.
    self.assertEqual(1, ops.count("Dequantize"))

    # No RequantizationRange
    self.assertEqual(0, ops.count("RequantizationRange"))

    # The fallback constants are in the graph.
    self.assertEqual(1, node_names.count("fallback_quantization_min_value"))
    self.assertEqual(1, node_names.count("fallback_quantization_max_value"))

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
    graph_def = graph_pb2.GraphDef()
    a_constant = quantize_graph.create_constant_node(
        a_constant_name, value=(0,), dtype=dtypes.quint8, shape=[])
    graph_def.node.extend([a_constant])
    a_constant_min = quantize_graph.create_constant_node(
        a_constant_min_name, value=2, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([a_constant_min])
    a_constant_max = quantize_graph.create_constant_node(
        a_constant_max_name, value=2, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([a_constant_max])
    a_dequantize_node = quantize_graph.create_node(
        "Dequantize", a_dequantize_name,
        [a_constant_name, a_constant_min_name, a_constant_max_name])
    quantize_graph.set_attr_dtype(a_dequantize_node, "T", dtypes.uint8)
    graph_def.node.extend([a_dequantize_node])
    a_quantize_node = quantize_graph.create_node(
        "QuantizeV2", a_quantize_name,
        [a_dequantize_name, a_dequantize_name + ":1", a_dequantize_name + ":2"])
    quantize_graph.set_attr_dtype(a_quantize_node, "T", dtypes.uint8)
    graph_def.node.extend([a_quantize_node])
    b_constant = quantize_graph.create_constant_node(
        b_constant_name, value=(0,), dtype=dtypes.quint8, shape=[])
    graph_def.node.extend([b_constant])
    b_constant_min = quantize_graph.create_constant_node(
        b_constant_min_name, value=3, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([b_constant_min])
    b_constant_max = quantize_graph.create_constant_node(
        b_constant_max_name, value=3, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([b_constant_max])
    b_dequantize_node = quantize_graph.create_node(
        "Dequantize", b_dequantize_name,
        [b_constant_name, b_constant_min_name, b_constant_max_name])
    quantize_graph.set_attr_dtype(b_dequantize_node, "T", dtypes.uint8)
    graph_def.node.extend([b_dequantize_node])
    b_quantize_node = quantize_graph.create_node(
        "QuantizeV2", b_quantize_name,
        [b_dequantize_name, b_dequantize_name + ":1", b_dequantize_name + ":2"])
    quantize_graph.set_attr_dtype(b_quantize_node, "T", dtypes.uint8)
    graph_def.node.extend([b_quantize_node])
    mat_mul_node = quantize_graph.create_node("QuantizedMatMul", mat_mul_name, [
        a_quantize_name, b_quantize_name, a_quantize_name + ":1",
        a_quantize_name + ":2", b_quantize_name + ":1", b_quantize_name + ":2"
    ])
    quantize_graph.set_attr_dtype(mat_mul_node, "T1", dtypes.uint8)
    quantize_graph.set_attr_dtype(mat_mul_node, "T2", dtypes.int32)
    graph_def.node.extend([mat_mul_node])

    expected_output = graph_pb2.GraphDef()
    a_constant = quantize_graph.create_constant_node(
        a_constant_name, value=(0,), dtype=dtypes.quint8, shape=[])
    expected_output.node.extend([a_constant])
    a_constant_min = quantize_graph.create_constant_node(
        a_constant_min_name, value=2, dtype=dtypes.float32, shape=[])
    expected_output.node.extend([a_constant_min])
    a_constant_max = quantize_graph.create_constant_node(
        a_constant_max_name, value=2, dtype=dtypes.float32, shape=[])
    expected_output.node.extend([a_constant_max])
    b_constant = quantize_graph.create_constant_node(
        b_constant_name, value=(0,), dtype=dtypes.quint8, shape=[])
    expected_output.node.extend([b_constant])
    b_constant_min = quantize_graph.create_constant_node(
        b_constant_min_name, value=3, dtype=dtypes.float32, shape=[])
    expected_output.node.extend([b_constant_min])
    b_constant_max = quantize_graph.create_constant_node(
        b_constant_max_name, value=3, dtype=dtypes.float32, shape=[])
    expected_output.node.extend([b_constant_max])
    mat_mul_node = quantize_graph.create_node("QuantizedMatMul", mat_mul_name, [
        a_constant_name, b_constant_name, a_constant_min_name,
        a_constant_max_name, b_constant_min_name, b_constant_max_name
    ])
    quantize_graph.set_attr_dtype(mat_mul_node, "T1", dtypes.uint8)
    quantize_graph.set_attr_dtype(mat_mul_node, "T2", dtypes.int32)
    expected_output.node.extend([mat_mul_node])
    expected_output.versions.CopyFrom(graph_def.versions)
    expected_output.library.CopyFrom(graph_def.library)

    rewriter = quantize_graph.GraphRewriter(
        graph_def, [mat_mul_name], quantized_input_range=None)
    output = rewriter.remove_redundant_quantization(graph_def)
    stripped_output = graph_util.extract_sub_graph(output, [mat_mul_name])
    self.assertProtoEquals(expected_output, stripped_output)


if __name__ == "__main__":
  test.main()
