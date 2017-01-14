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
r"""Transforms a float-trained graph into an equivalent quantized version.

An example of command-line usage is:
bazel build tensorflow/contrib/quantization/tools:quantize_graph \
&& bazel-bin/tensorflow/contrib/quantization/tools/quantize_graph \
--input=tensorflow_inception_graph.pb
--output_node_names="softmax2" --print_nodes --output=/tmp/quantized_graph.pb \
--mode=eightbit --logtostderr

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util

# TODO(petewarden) - Remove this ugly hack to get around Python linking problems
# with Bazel.
# pylint: disable=g-bad-import-order
from tensorflow.contrib.quantization import load_quantized_ops_so
from tensorflow.contrib.quantization.kernels import load_quantized_kernels_so


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean("print_nodes", False, """Lists all nodes in the model.""")
flags.DEFINE_string("input", "", """TensorFlow 'GraphDef' file to load.""")
flags.DEFINE_string("output_node_names", "",
                    """Output node names, comma separated.""")
flags.DEFINE_string("output", "", """File to save the output graph to.""")
flags.DEFINE_integer("bitdepth", 8,
                     """How many bits to quantize the graph to.""")
flags.DEFINE_string("mode", "round",
                    """What transformation to apply (round, quantize,"""
                    """ eightbit, weights, or weights_rounded).""")
flags.DEFINE_string("test_input_dims", "1,224,224,3",
                    """The size of the input tensor to use when testing a"""
                    """ graph loaded from a file.""")
flags.DEFINE_boolean("strip_redundant_quantization", True,
                     """Removes redundant dequantize/quantize pairs.""")
flags.DEFINE_boolean("load_quantization_so", True,
                     """Explicitly load the quantization ops library""")


def print_input_nodes(current_node, nodes_map, indent, already_visited):
  print(" " * indent + current_node.op + ":" + current_node.name)
  for input_node_name in current_node.input:
    if input_node_name in already_visited:
      continue
    input_node = nodes_map[input_node_name]
    print_input_nodes(input_node, nodes_map, indent + 1, already_visited)
  already_visited[current_node.name] = True


def create_node(op, name, inputs):
  new_node = tf.NodeDef()
  new_node.op = op
  new_node.name = name
  for input_name in inputs:
    new_node.input.extend([input_name])
  return new_node


def create_constant_node(name, value, dtype, shape=None):
  node = create_node("Const", name, [])
  set_attr_dtype(node, "dtype", dtype)
  set_attr_tensor(node, "value", value, dtype, shape)
  return node


def copy_attr(node, key, attr_value):
  try:
    node.attr[key].CopyFrom(attr_value)
  except KeyError:
    pass


def set_attr_dtype(node, key, value):
  try:
    node.attr[key].CopyFrom(tf.AttrValue(type=value.as_datatype_enum))
  except KeyError:
    pass


def set_attr_tensor(node, key, value, dtype, shape=None):
  try:
    node.attr[key].CopyFrom(tf.AttrValue(
        tensor=tensor_util.make_tensor_proto(value,
                                             dtype=dtype,
                                             shape=shape)))
  except KeyError:
    pass


def set_attr_string(node, key, value):
  try:
    node.attr[key].CopyFrom(tf.AttrValue(s=value))
  except KeyError:
    pass


def set_attr_int_list(node, key, value):
  list_value = tf.AttrValue.ListValue(i=value)
  try:
    node.attr[key].CopyFrom(tf.AttrValue(list=list_value))
  except KeyError:
    pass


def set_attr_bool(node, key, value):
  try:
    node.attr[key].CopyFrom(tf.AttrValue(b=value))
  except KeyError:
    pass


def set_attr_int(node, key, value):
  try:
    node.attr[key].CopyFrom(tf.AttrValue(i=value))
  except KeyError:
    pass


def set_attr_float(node, key, value):
  try:
    node.attr[key].CopyFrom(tf.AttrValue(f=value))
  except KeyError:
    pass


def node_name_from_input(node_name):
  """Strips off ports and other decorations to get the underlying node name."""
  if node_name.startswith("^"):
    node_name = node_name[1:]
  m = re.search(r"(.*):\d+$", node_name)
  if m:
    node_name = m.group(1)
  return node_name


def ensure_tensor_name_has_port(node_name):
  """Makes sure that a tensor name has :0 if no explicit port exists."""
  m = re.search(r"(.*):\d+$", node_name)
  if m:
    name_with_port = node_name
  else:
    name_with_port = node_name + ":0"
  return name_with_port


def unique_node_name_from_input(node_name):
  """Replaces invalid characters in input names to get a unique node name."""
  return node_name.replace(":", "__port__").replace("^", "__hat__")


def quantize_array(arr, num_buckets):
  """Quantizes a numpy array.

  This function maps each scalar in arr to the center of one of num_buckets
  buckets. For instance,
  quantize_array([0, 0.3, 0.6, 1], 2) => [0.25, 0.25, 0.75, 0.75]

  Args:
    arr: The numpy array to quantize.
    num_buckets: The number of buckets to map "var" to.
  Returns:
    The quantized numpy array.
  Raises:
    ValueError: when num_buckets < 1.
  """
  if num_buckets < 1:
    raise ValueError("num_buckets must be >= 1")
  arr_max = arr.max()
  arr_min = arr.min()
  if arr_max == arr_min:
    return arr
  bucket_width = (arr_max - arr_min) / num_buckets
  # Map scalars to bucket indices. Take special care of max(arr).
  bucket_indices = np.floor((arr - arr_min) / bucket_width)
  bucket_indices[bucket_indices == num_buckets] = num_buckets - 1
  # Map each scalar to the center of a bucket.
  arr = arr_min + bucket_width * (bucket_indices + 0.5)
  return arr


def quantize_weight_rounded(input_node):
  """Returns a replacement node for input_node containing bucketed floats."""
  input_tensor = input_node.attr["value"].tensor
  tensor_value = tensor_util.MakeNdarray(input_tensor)
  tensor_shape = input_tensor.tensor_shape
  # Currently, the parameter FLAGS.bitdepth is used to compute the
  # number of buckets as 1 << FLAGS.bitdepth, meaning the number of
  # buckets can only be a power of 2.
  # This could be fixed by intorducing a new parameter, num_buckets,
  # which would allow for more flexibility in chosing the right model
  # size/accuracy tradeoff. But I didn't want to add more parameters
  # to this script than absolutely necessary.
  num_buckets = 1 << FLAGS.bitdepth
  tensor_value_rounded = quantize_array(tensor_value, num_buckets)
  tensor_shape_list = tensor_util.TensorShapeProtoToList(tensor_shape)
  return [create_constant_node(input_node.name, tensor_value_rounded,
                               tf.float32, shape=tensor_shape_list)]


def quantize_weight_eightbit(input_node, quantization_mode):
  """Returns replacement nodes for input_node using the Dequantize op."""
  base_name = input_node.name + "_"
  quint8_const_name = base_name + "quint8_const"
  min_name = base_name + "min"
  max_name = base_name + "max"
  float_tensor = tensor_util.MakeNdarray(
      input_node.attr["value"].tensor)
  min_value = np.min(float_tensor.flatten())
  max_value = np.max(float_tensor.flatten())
  # min_value == max_value is a tricky case. It can occur for general
  # tensors, and of course for scalars. The quantized ops cannot deal
  # with this case, so we set max_value to something else.
  # It's a tricky question what is the numerically best solution to
  # deal with this degeneracy.
  # TODO(petewarden): Better use a tolerance than a hard comparison?
  if min_value == max_value:
    if abs(min_value) < 0.000001:
      max_value = min_value + 1.0
    elif min_value > 0:
      max_value = 2 * min_value
    else:
      max_value = min_value / 2.0

  sess = tf.Session()
  with sess.as_default():
    quantize_op = tf.contrib.quantization.python.quantize_v2(
        float_tensor,
        min_value,
        max_value,
        tf.quint8,
        mode=quantization_mode)
    quint8_tensor = quantize_op[0].eval()
  shape = tensor_util.TensorShapeProtoToList(input_node.attr[
      "value"].tensor.tensor_shape)
  quint8_const_node = create_constant_node(quint8_const_name,
                                           quint8_tensor,
                                           tf.quint8,
                                           shape=shape)
  min_node = create_constant_node(min_name, min_value, tf.float32)
  max_node = create_constant_node(max_name, max_value, tf.float32)
  dequantize_node = create_node("Dequantize", input_node.name,
                                [quint8_const_name, min_name, max_name])
  set_attr_dtype(dequantize_node, "T", tf.quint8)
  set_attr_string(dequantize_node, "mode", quantization_mode)
  return [quint8_const_node, min_node, max_node, dequantize_node]


class GraphRewriter(object):
  """Takes a float graph, and rewrites it in quantized form."""

  def __init__(self, input_graph, mode):
    """Sets up the class to rewrite a float graph.

    Args:
      input_graph: A float graph to transform.
      mode: A string controlling how quantization is performed -
        round, quantize, eightbit, or weights.

    Raises:
      ValueError: Two nodes with the same name were found in the graph.
    """
    self.input_graph = input_graph
    self.nodes_map = self.create_nodes_map(input_graph)
    self.output_graph = None
    self.mode = mode
    if FLAGS.load_quantization_so:
      load_quantized_ops_so.Load()
      load_quantized_kernels_so.Load()

  def create_nodes_map(self, graph):
    """Builds a mapping of node names to their defs from the graph."""
    nodes_map = {}
    for node in graph.node:
      if node.name not in nodes_map.keys():
        nodes_map[node.name] = node
      else:
        raise ValueError("Duplicate node names detected.")
    return nodes_map

  def rewrite(self, output_node_names):
    """Triggers rewriting of the float graph.

    Args:
      output_node_names: A list of names of the nodes that produce the final
        results.

    Returns:
      A quantized version of the float graph.
    """
    self.output_graph = tf.GraphDef()
    output_nodes = [self.nodes_map[output_node_name]
                    for output_node_name in output_node_names]
    if self.mode == "round":
      self.already_visited = {}
      for output_node in output_nodes:
        self.round_nodes_recursively(output_node)
    elif self.mode == "quantize":
      self.already_visited = {}
      self.already_quantized = {}
      for output_node in output_nodes:
        self.quantize_nodes_recursively(output_node)
    elif self.mode == "eightbit":
      self.set_input_graph(self.remove_unneeded_nodes(self.input_graph))
      self.already_visited = {}
      self.layers_eightbitized = []
      for output_node in output_nodes:
        self.eightbitize_nodes_recursively(output_node)
      self.output_graph = self.quantize_weights(self.output_graph, b"MIN_FIRST")
      if FLAGS.strip_redundant_quantization:
        self.output_graph = self.remove_redundant_quantization(
            self.output_graph)
        self.remove_dead_nodes(output_node_names)
    elif self.mode == "weights":
      self.output_graph = self.quantize_weights(self.input_graph,
                                                b"MIN_COMBINED")
      self.remove_dead_nodes(output_node_names)
    elif self.mode == "weights_rounded":
      self.output_graph = self.quantize_weights(self.input_graph, self.mode)
      self.remove_dead_nodes(output_node_names)
    else:
      print("Bad mode - " + self.mode + ".")
    return self.output_graph

  def round_nodes_recursively(self, current_node):
    """The entry point for simple rounding quantization."""
    for input_node_name in current_node.input:
      input_node_name = node_name_from_input(input_node_name)
      if input_node_name in self.already_visited:
        continue
      input_node = self.nodes_map[input_node_name]
      self.round_nodes_recursively(input_node)
    self.already_visited[current_node.name] = True
    nodes_to_quantize = ["Conv2D", "BiasAdd", "MatMul"]
    if any(current_node.op in s for s in nodes_to_quantize):
      new_node = tf.NodeDef()
      new_node.CopyFrom(current_node)
      new_node.name = current_node.name + "_original"
      self.add_output_graph_node(new_node)
      levels = 1 << FLAGS.bitdepth
      constant_name = current_node.name + "_round_depth"
      constant_tensor = tf.constant(levels, dtype=tf.int32, name=constant_name)
      constant_node = constant_tensor.op.node_def
      self.add_output_graph_node(constant_node)
      quantize_node = tf.NodeDef()
      quantize_node.op = "RoundToSteps"
      quantize_node.name = current_node.name
      quantize_node.input.extend([current_node.name + "_original"])
      quantize_node.input.extend([constant_node.name])
      self.add_output_graph_node(quantize_node)
    else:
      new_node = tf.NodeDef()
      new_node.CopyFrom(current_node)
      self.add_output_graph_node(new_node)

  def quantize_nodes_recursively(self, current_node):
    """The entry point for quantizing nodes to eight bit and back."""
    for input_node_name in current_node.input:
      input_node_name = node_name_from_input(input_node_name)
      if input_node_name in self.already_visited:
        continue
      input_node = self.nodes_map[input_node_name]
      self.quantize_nodes_recursively(input_node)
    self.already_visited[current_node.name] = True
    nodes_to_quantize = ["Conv2D", "BiasAdd", "MatMul"]
    if any(current_node.op in s for s in nodes_to_quantize):
      for input_name in current_node.input:
        input_name = node_name_from_input(input_name)
        input_node = self.nodes_map[input_name]
        self.quantize_node(input_node)
      self.quantize_node(current_node)
    else:
      new_node = tf.NodeDef()
      new_node.CopyFrom(current_node)
      self.add_output_graph_node(new_node)

  def quantize_node(self, input_node):
    """Handles quantizing a single node."""
    input_name = input_node.name
    if input_name in self.already_quantized:
      return
    self.already_quantized[input_name] = True
    original_input_name = input_name + "_original"
    reshape_name = input_name + "_reshape"
    reshape_dims_name = input_name + "_reshape_dims"
    max_name = input_name + "_max"
    min_name = input_name + "_min"
    dims_name = input_name + "_dims"
    quantize_name = input_name + "_quantize"
    dequantize_name = input_name
    original_input_node = tf.NodeDef()
    original_input_node.CopyFrom(input_node)
    original_input_node.name = original_input_name
    self.add_output_graph_node(original_input_node)
    reshape_dims_node = create_constant_node(reshape_dims_name, -1, tf.int32,
                                             [1])
    self.add_output_graph_node(reshape_dims_node)
    reshape_node = create_node("Reshape", reshape_name, [original_input_name,
                                                         reshape_dims_name])
    set_attr_dtype(reshape_node, "T", tf.float32)
    self.add_output_graph_node(reshape_node)
    dims_node = create_constant_node(dims_name, 0, tf.int32, [1])
    self.add_output_graph_node(dims_node)
    max_node = create_node("Max", max_name, [reshape_name, dims_name])
    set_attr_dtype(max_node, "T", tf.float32)
    set_attr_bool(max_node, "keep_dims", False)
    self.add_output_graph_node(max_node)
    min_node = create_node("Min", min_name, [reshape_name, dims_name])
    set_attr_dtype(min_node, "T", tf.float32)
    set_attr_bool(min_node, "keep_dims", False)
    self.add_output_graph_node(min_node)
    quantize_node = create_node("Quantize", quantize_name, [original_input_name,
                                                            min_name, max_name])
    set_attr_dtype(quantize_node, "T", tf.quint8)
    set_attr_string(quantize_node, "mode", b"MIN_FIRST")
    self.add_output_graph_node(quantize_node)
    dequantize_node = create_node("Dequantize", dequantize_name,
                                  [quantize_name, min_name, max_name])
    set_attr_dtype(dequantize_node, "T", tf.quint8)
    set_attr_string(dequantize_node, "mode", b"MIN_FIRST")
    self.add_output_graph_node(dequantize_node)

  def eightbitize_nodes_recursively(self, current_node):
    """The entry point for transforming a graph into full eight bit."""
    for input_node_name in current_node.input:
      input_node_name = node_name_from_input(input_node_name)
      if input_node_name in self.already_visited:
        continue
      input_node = self.nodes_map[input_node_name]
      self.eightbitize_nodes_recursively(input_node)
    self.already_visited[current_node.name] = True
    if current_node.op == "MatMul":
      self.eightbitize_mat_mul_node(current_node)
    elif current_node.op == "Conv2D":
      self.eightbitize_conv_node(current_node)
      self.layers_eightbitized.append(current_node.name)
    elif current_node.op == "BiasAdd":
      self.eightbitize_bias_add_node(current_node)
    elif current_node.op == "MaxPool" or current_node.op == "AvgPool":
      self.eightbitize_single_input_tensor_node(current_node,
                                                self.add_pool_function)
    elif current_node.op == "Relu" or current_node.op == "Relu6":
      self.eightbitize_single_input_tensor_node(current_node,
                                                self.add_relu_function)
    elif current_node.op == "Concat":
      self.eightbitize_concat_node(current_node)
    elif current_node.op == "BatchNormWithGlobalNormalization":
      self.eightbitize_batch_norm_node(current_node)
    else:
      new_node = tf.NodeDef()
      new_node.CopyFrom(current_node)
      self.add_output_graph_node(new_node)

  def add_eightbit_prologue_nodes(self, original_node):
    """Adds input conversion nodes to handle quantizing the underlying node."""
    namespace_prefix = original_node.name + "_eightbit"
    reshape_dims_name, reduction_dims_name = self.add_common_quantization_nodes(
        namespace_prefix)
    input_names = []
    min_max_names = []
    for original_input_name in original_node.input:
      quantize_input_name, min_input_name, max_input_name = (
          self.eightbitize_input_to_node(namespace_prefix, original_input_name,
                                         reshape_dims_name,
                                         reduction_dims_name))
      input_names.append(quantize_input_name)
      min_max_names.append(min_input_name)
      min_max_names.append(max_input_name)
    all_input_names = []
    all_input_names.extend(input_names)
    all_input_names.extend(min_max_names)
    return all_input_names

  def add_common_quantization_nodes(self, namespace_prefix):
    """Builds constant nodes needed for quantization of inputs."""
    reshape_dims_name = namespace_prefix + "_reshape_dims"
    reduction_dims_name = namespace_prefix + "_reduction_dims"

    reshape_dims_node = create_constant_node(reshape_dims_name, -1, tf.int32,
                                             [1])
    self.add_output_graph_node(reshape_dims_node)
    reduction_dims_node = create_constant_node(reduction_dims_name, 0, tf.int32,
                                               [1])
    self.add_output_graph_node(reduction_dims_node)
    return reshape_dims_name, reduction_dims_name

  def eightbitize_input_to_node(self, namespace_prefix, original_input_name,
                                reshape_dims_name, reduction_dims_name):
    """Takes one float input to an op, and converts it to quantized form."""
    unique_input_name = unique_node_name_from_input(original_input_name)
    reshape_input_name = namespace_prefix + "_reshape_" + unique_input_name
    min_input_name = namespace_prefix + "_min_" + unique_input_name
    max_input_name = namespace_prefix + "_max_" + unique_input_name
    quantize_input_name = namespace_prefix + "_quantize_" + unique_input_name
    reshape_input_node = create_node("Reshape", reshape_input_name,
                                     [original_input_name, reshape_dims_name])
    set_attr_dtype(reshape_input_node, "T", tf.float32)
    self.add_output_graph_node(reshape_input_node)
    min_input_node = create_node("Min", min_input_name, [reshape_input_name,
                                                         reduction_dims_name])
    set_attr_dtype(min_input_node, "T", tf.float32)
    set_attr_bool(min_input_node, "keep_dims", False)
    self.add_output_graph_node(min_input_node)
    max_input_node = create_node("Max", max_input_name, [reshape_input_name,
                                                         reduction_dims_name])
    set_attr_dtype(max_input_node, "T", tf.float32)
    set_attr_bool(max_input_node, "keep_dims", False)
    self.add_output_graph_node(max_input_node)
    quantize_input_node = create_node("QuantizeV2", quantize_input_name,
                                      [original_input_name, min_input_name,
                                       max_input_name])
    set_attr_dtype(quantize_input_node, "T", tf.quint8)
    set_attr_string(quantize_input_node, "mode", b"MIN_FIRST")
    self.add_output_graph_node(quantize_input_node)
    min_output_name = quantize_input_name + ":1"
    max_output_name = quantize_input_name + ":2"
    return quantize_input_name, min_output_name, max_output_name

  def add_quantize_down_node(self, original_node, quantized_output_name):
    quantize_down_name = original_node.name + "_eightbit_quantize_down"
    quantize_down_node = create_node(
        "QuantizeDownAndShrinkRange", quantize_down_name,
        [quantized_output_name, quantized_output_name + ":1",
         quantized_output_name + ":2"])
    set_attr_dtype(quantize_down_node, "Tinput", tf.qint32)
    set_attr_dtype(quantize_down_node, "out_type", tf.quint8)
    self.add_output_graph_node(quantize_down_node)
    return quantize_down_name

  def add_dequantize_result_node(self, quantized_output_name,
                                 original_node_name):
    dequantize_name = original_node_name
    dequantize_node = create_node("Dequantize", dequantize_name,
                                  [quantized_output_name,
                                   quantized_output_name + ":1",
                                   quantized_output_name + ":2"])
    set_attr_dtype(dequantize_node, "T", tf.quint8)
    set_attr_string(dequantize_node, "mode", b"MIN_FIRST")
    self.add_output_graph_node(dequantize_node)

  def eightbitize_mat_mul_node(self, original_node):
    """Replaces a MatMul node with the eight bit equivalent sub-graph."""
    quantized_mat_mul_name = original_node.name + "_eightbit_quantized_bias_add"
    all_input_names = self.add_eightbit_prologue_nodes(original_node)
    quantized_mat_mul_node = create_node(
        "QuantizedMatMul", quantized_mat_mul_name,
        all_input_names)
    set_attr_dtype(quantized_mat_mul_node, "T1", tf.quint8)
    set_attr_dtype(quantized_mat_mul_node, "T2", tf.quint8)
    set_attr_dtype(quantized_mat_mul_node, "Toutput", tf.qint32)
    copy_attr(quantized_mat_mul_node, "transpose_a",
              original_node.attr["transpose_a"])
    copy_attr(quantized_mat_mul_node, "transpose_b",
              original_node.attr["transpose_b"])
    self.add_output_graph_node(quantized_mat_mul_node)
    quantize_down_name = self.add_quantize_down_node(original_node,
                                                     quantized_mat_mul_name)
    self.add_dequantize_result_node(quantize_down_name, original_node.name)

  def eightbitize_conv_node(self, original_node):
    """Replaces a Conv2D node with the eight bit equivalent sub-graph."""
    all_input_names = self.add_eightbit_prologue_nodes(original_node)
    quantized_conv_name = original_node.name + "_eightbit_quantized_conv"
    quantized_conv_node = create_node("QuantizedConv2D", quantized_conv_name,
                                      all_input_names)
    copy_attr(quantized_conv_node, "strides", original_node.attr["strides"])
    copy_attr(quantized_conv_node, "padding", original_node.attr["padding"])
    set_attr_dtype(quantized_conv_node, "Tinput", tf.quint8)
    set_attr_dtype(quantized_conv_node, "Tfilter", tf.quint8)
    set_attr_dtype(quantized_conv_node, "out_type", tf.qint32)
    self.add_output_graph_node(quantized_conv_node)
    quantize_down_name = self.add_quantize_down_node(original_node,
                                                     quantized_conv_name)
    self.add_dequantize_result_node(quantize_down_name, original_node.name)

  def eightbitize_bias_add_node(self, original_node):
    """Replaces a BiasAdd node with the eight bit equivalent sub-graph."""
    quantized_bias_add_name = (original_node.name +
                               "_eightbit_quantized_bias_add")
    all_input_names = self.add_eightbit_prologue_nodes(original_node)
    quantized_bias_add_node = create_node(
        "QuantizedBiasAdd", quantized_bias_add_name,
        all_input_names)
    set_attr_dtype(quantized_bias_add_node, "T1", tf.quint8)
    set_attr_dtype(quantized_bias_add_node, "T2", tf.quint8)
    set_attr_dtype(quantized_bias_add_node, "out_type", tf.qint32)
    self.add_output_graph_node(quantized_bias_add_node)
    quantize_down_name = self.add_quantize_down_node(original_node,
                                                     quantized_bias_add_name)
    self.add_dequantize_result_node(quantize_down_name, original_node.name)

  def eightbitize_single_input_tensor_node(self, original_node,
                                           add_op_function):
    """Replaces a single-tensor node with the eight bit equivalent sub-graph.

    Converts a node like this:

       Shape(f)   Input(f)
         |          |
         +--------v v
                Operation
                    |
                    v
                   (f)

     Into a quantized equivalent:

                    Input(f)              ReshapeDims
                       +------v v-------------+
                       |    Reshape
                       |      |
                       |      |          ReductionDims
                       |      +-----+         |
                       |      | +---c---------+
                       |      v v   v v-------+
                       |      Min   Max
                       |  +----+      |
                       v  v  v--------+
                      Quantize
                          |
                          v
                   QuantizedOperation
                      |   |   |
                      v   v   v
                      Dequantize
                          |
                          v
                         (f)


    Args:
      original_node: Float node to be converted.
      add_op_function: Function to create the actual node.

    Returns:
      Subgraph representing the quantized version of the original node.

    """
    quantized_op_name = original_node.name + "_eightbit_quantized"
    quantized_op_type = "Quantized" + original_node.op
    all_input_names = self.add_eightbit_prologue_nodes(original_node)
    quantized_op_node = create_node(
        quantized_op_type, quantized_op_name, all_input_names)
    add_op_function(original_node, quantized_op_node)
    self.add_output_graph_node(quantized_op_node)
    self.add_dequantize_result_node(quantized_op_name, original_node.name)

  def add_pool_function(self, original_node, quantized_op_node):
    set_attr_dtype(quantized_op_node, "T", tf.quint8)
    copy_attr(quantized_op_node, "ksize", original_node.attr["ksize"])
    copy_attr(quantized_op_node, "strides", original_node.attr["strides"])
    copy_attr(quantized_op_node, "padding", original_node.attr["padding"])

  def add_relu_function(self, unused_arg_node, quantized_op_node):
    set_attr_dtype(quantized_op_node, "Tinput", tf.quint8)

  def eightbitize_concat_node(self, original_node):
    """Replaces a Concat node with the eight bit equivalent sub-graph.

    Converts a node like this:

       Shape(f)   Input0(f)   Input1(f)
         |          |            |
         +--------v v v----------+
                  Concat
                    |
                    v
                   (f)

     Into a quantized equivalent:

       Shape(f)     Input0(f)             ReshapeDims                  Input1(f)
         |             +------v v--------------+------------------v v------+
         |             |    Reshape                             Reshape    |
         |             |      |                                     |      |
         |             |      |           ReductionDims             |      |
         |             |      +------+         |           +--------+      |
         |             |      |  +---c---------+-----------c-----+  |      |
         |             |      +v v   v v-------+---------v v     v v+      |
         |             |       Min   Max                 Min     Max       |
         |             |  +----+      |                   |       +-----+  |
         |             v  v  v--------+                   +----------v  v  v
         |            Quantize                                       Quantize
         |                +------------------+   +----------------------+
         +-------------------------------+   |   |
                                         v   v   v
                                      QuantizedConcat
                                         |   |   |
                                         v   v   v
                                        Dequantize
                                             |
                                             v
                                            (f)
    Args:
      original_node: Float node to be converted.

    Returns:
      Subgraph representing the quantized version of the original node.

    """
    namespace_prefix = original_node.name + "_eightbit"
    quantized_concat_name = namespace_prefix + "_quantized_concat"
    reshape_dims_name, reduction_dims_name = self.add_common_quantization_nodes(
        namespace_prefix)
    shape_input_name = original_node.input[0]
    original_inputs = original_node.input[1:]
    input_names = []
    min_names = []
    max_names = []
    for original_input_name in original_inputs:
      quantize_input_name, min_input_name, max_input_name = (
          self.eightbitize_input_to_node(namespace_prefix, original_input_name,
                                         reshape_dims_name,
                                         reduction_dims_name))
      input_names.append(quantize_input_name)
      min_names.append(min_input_name)
      max_names.append(max_input_name)
    all_input_names = [shape_input_name]
    all_input_names.extend(input_names)
    all_input_names.extend(min_names)
    all_input_names.extend(max_names)
    quantized_concat_node = create_node(
        "QuantizedConcat", quantized_concat_name, all_input_names)
    set_attr_int(quantized_concat_node, "N", len(original_inputs))
    set_attr_dtype(quantized_concat_node, "T", tf.quint8)
    self.add_output_graph_node(quantized_concat_node)
    self.add_dequantize_result_node(quantized_concat_name, original_node.name)

  def eightbitize_batch_norm_node(self, original_node):
    """Replaces a MatMul node with the eight bit equivalent sub-graph."""
    namespace_prefix = original_node.name + "_eightbit"
    original_input_name = original_node.input[0]
    original_mean_name = original_node.input[1]
    original_variance_name = original_node.input[2]
    original_beta_name = original_node.input[3]
    original_gamma_name = original_node.input[4]
    quantized_batch_norm_name = namespace_prefix + "_quantized_batch_norm"

    reshape_dims_name, reduction_dims_name = self.add_common_quantization_nodes(
        namespace_prefix)
    quantize_input_name, min_input_name, max_input_name = (
        self.eightbitize_input_to_node(namespace_prefix, original_input_name,
                                       reshape_dims_name, reduction_dims_name))
    quantize_mean_name, min_mean_name, max_mean_name = (
        self.eightbitize_input_to_node(namespace_prefix, original_mean_name,
                                       reshape_dims_name, reduction_dims_name))
    quantize_variance_name, min_variance_name, max_variance_name = (
        self.eightbitize_input_to_node(namespace_prefix, original_variance_name,
                                       reshape_dims_name, reduction_dims_name))
    quantize_beta_name, min_beta_name, max_beta_name = (
        self.eightbitize_input_to_node(namespace_prefix, original_beta_name,
                                       reshape_dims_name, reduction_dims_name))
    quantize_gamma_name, min_gamma_name, max_gamma_name = (
        self.eightbitize_input_to_node(namespace_prefix, original_gamma_name,
                                       reshape_dims_name, reduction_dims_name))
    quantized_batch_norm_node = create_node(
        "QuantizedBatchNormWithGlobalNormalization", quantized_batch_norm_name,
        [quantize_input_name, min_input_name, max_input_name,
         quantize_mean_name, min_mean_name, max_mean_name,
         quantize_variance_name, min_variance_name, max_variance_name,
         quantize_beta_name, min_beta_name, max_beta_name, quantize_gamma_name,
         min_gamma_name, max_gamma_name])
    set_attr_dtype(quantized_batch_norm_node, "Tinput", tf.quint8)
    set_attr_dtype(quantized_batch_norm_node, "out_type", tf.qint32)
    copy_attr(quantized_batch_norm_node, "scale_after_normalization",
              original_node.attr["scale_after_normalization"])
    copy_attr(quantized_batch_norm_node, "variance_epsilon",
              original_node.attr["variance_epsilon"])
    self.add_output_graph_node(quantized_batch_norm_node)
    quantize_down_name = self.add_quantize_down_node(original_node,
                                                     quantized_batch_norm_name)
    self.add_dequantize_result_node(quantize_down_name, original_node.name)

  def add_output_graph_node(self, output_node):
    """Inserts one node into the new graph."""
    self.output_graph.node.extend([output_node])

  def remove_redundant_quantization(self, old_graph):
    """Removes unneeded pairs of quantize/dequantize ops from the graph.

    This is a bit of a tricky function, because it's attempting to spot the
    pattern of dequantizing from eight-bit up to float, and then immediately
    quantizing back down to eight bits again, that's introduced by previous
    passes that do 'key-hole' conversions of individual nodes but have to
    convert back to float to match the previous output interface, since they
    don't know that the next op can handle quantized tensors.
    It works by:
     - Looking for Quantize nodes.
     - Checking to see if their first input is a Dequantize node.
     - Seeing if their min/max inputs come from Min/Max nodes.
     - Making sure those Min/Max nodes are being fed from the same Dequantize.
     - Or that the Min is indirectly being fed from the same Dequantize as Max.
     - Making sure the Dequantize is going through a Reshape (which we add
       during the previous pass when we create the quantize sub-graph).
     - Looking for the dims Const op for the Min/Max dims.
    If all of these conditions are met, then it's a sub-graph pattern that
    we know how to optimize out (and is likely the common one we've introduced).
    We then rewire the graph to skip it entirely, and then rely on the dead node
    removal pass to get rid of any nodes that are no longer needed.

    Args:
      old_graph: The model we'll be stripping redundant nodes from.

    Returns:
      A graph with the unnecessary nodes removed.

    Raises:
      ValueError: Two nodes with the same name were found in the graph.
    """
    old_nodes_map = self.create_nodes_map(old_graph)
    self.output_graph = tf.GraphDef()
    inputs_to_rename = {}
    # We go through all the nodes, looking for any that match the patterns we
    # know how to optimize away.
    for node in old_graph.node:
      # We always start with a Quantize node, and examine its inputs to see if
      # they are in a form that can be removed.
      if node.op not in ["Quantize", "QuantizeV2"]:
        continue
      dequantize_node_name = node_name_from_input(node.input[0])
      if dequantize_node_name not in old_nodes_map:
        raise ValueError("Input node name '" + dequantize_node_name +
                         "' not found in node '" + node.name + "'")
      dequantize_node = old_nodes_map[dequantize_node_name]
      # Do we have a Dequantize feeding in, with the same type as the Quantize?
      if dequantize_node.op != "Dequantize":
        continue
      if node.attr["T"] != dequantize_node.attr["T"]:
        continue
      # Now look at the other inputs, and ensure they're Min/Max nodes.
      min_node_name = node_name_from_input(node.input[1])
      max_node_name = node_name_from_input(node.input[2])
      min_node = old_nodes_map[min_node_name]
      max_node = old_nodes_map[max_node_name]
      is_min_right_type = (min_node.op in ["Min", "Dequantize"])
      is_max_right_type = (max_node.op in ["Max", "Dequantize"])
      if not is_min_right_type or not is_max_right_type:
        print("Didn't find expected types on inputs : %s, %s." % (
            min_node.op, max_node.op))
        continue
      min_node_input_name = node_name_from_input(min_node.input[0])
      max_node_input_name = node_name_from_input(max_node.input[0])
      # There are two different patterns for Min nodes we can recognize, one
      # where the input comes directly from the same one as the Max, and
      # another where we run it through another Min first, so check for both.
      is_same_input = False
      if min_node_input_name == max_node_input_name:
        is_same_input = True
      else:
        first_min_node_input = old_nodes_map[min_node_input_name]
        if first_min_node_input.op == "Concat":
          second_min_node_name = node_name_from_input(
              first_min_node_input.input[1])
          second_min_node = old_nodes_map[second_min_node_name]
          if second_min_node.op == "Min":
            second_min_node_input_name = node_name_from_input(
                second_min_node.input[0])
            is_same_input = (second_min_node_input_name == max_node_input_name)
      if not is_same_input:
        print("Different min/max inputs: " + min_node_input_name)
        continue
      # We recognize this pattern, so mark the graph edges to be rewired to
      # route around it entirely, since we know it's a no-op.
      dequantize_source_name = node_name_from_input(dequantize_node.input[0])
      node_tensor_name = ensure_tensor_name_has_port(node.name)
      min_tensor_name = node.name + ":1"
      max_tensor_name = node.name + ":2"
      inputs_to_rename[node_tensor_name] = dequantize_source_name
      inputs_to_rename[min_tensor_name] = dequantize_node.input[1]
      inputs_to_rename[max_tensor_name] = dequantize_node.input[2]
    # Finally we apply all the rewiring we've marked to the graph.
    for node in old_graph.node:
      for index, input_full_name in enumerate(node.input):
        input_name = ensure_tensor_name_has_port(input_full_name)
        if input_name in inputs_to_rename:
          node.input[index] = inputs_to_rename[input_name]
      self.add_output_graph_node(node)
    return self.output_graph

  def remove_dead_nodes(self, output_names):
    """Removes nodes that are no longer needed for inference from the graph."""
    old_output_graph = self.output_graph
    self.output_graph = graph_util.extract_sub_graph(old_output_graph,
                                                     output_names)

  def quantize_weights(self, input_graph, quantization_mode):
    """Quantize float Const ops.

    There are two modes of operations, both replace float Const ops with
    quantized values.
    1. If quantization_mode is "weights_rounded", this function replaces float
    Const ops with quantized float Const ops - same as the original op, but
    float values being mapped to the center of one of 1<<FLAGS.bitdepth buckets.
    This does not change the raw model size, but compression algorithms such as
    zip (as used for compressing apks) or bzip2 will achieve a very good
    compression ratio.
    2. For other quantization modes ("MIN_COMBINED" or "MIN_FIRST"), float
    Const ops are quantized and replaced by a tuple of four ops to perform
    the dequantization at runtime:
    * eight-bit Const (bucket indices, same shape as original float Const op
    * two float Const ops (min and max value of original float Const op)
    * Dequantize op to convert the eight-bit consts to float tensors.
    The quantization mode is important because we see accuracy problems when
    quantizing weights for different situations depending on the algorithm
    used. We haven't figured out exactly what the underlying cause is yet,
    unfortunately.

    Args:
      input_graph: A GraphDef of the model containing float Const ops.
      quantization_mode: How to quantize and dequantize the values.

    Returns:
      A GraphDef of the converted graph.

    Raises:
      ValueError: If quantization_mode is unsupported.
    """
    output_graph = tf.GraphDef()
    for input_node in input_graph.node:
      should_quantize = False
      if input_node.op == "Const":
        dtype = tf.as_dtype(input_node.attr["dtype"].type)
        if dtype == tf.float32:
          should_quantize = True
      if should_quantize:
        if quantization_mode == "weights_rounded":
          output_graph.node.extend(quantize_weight_rounded(input_node))
        elif quantization_mode in (b"MIN_COMBINED", b"MIN_FIRST"):
          output_graph.node.extend(quantize_weight_eightbit(input_node,
                                                            quantization_mode))
        else:
          raise ValueError("Unsupported quantization mode %s." %
                           quantization_mode)
      else:
        output_node = tf.NodeDef()
        output_node.CopyFrom(input_node)
        output_graph.node.extend([output_node])
    return output_graph

  def remove_unneeded_nodes(self, input_graph):
    """Prunes out nodes that aren't needed for inference.

    There are nodes like Identity and CheckNumerics that are only useful
    during training, and can be removed in graphs that will be used for
    nothing but inference. Here we identify and remove them, returning an
    equivalent graph.

    Args:
      input_graph: Model to analyze and prune.

    Returns:
    A list of nodes with the unnecessary ones removed.
    """

    types_to_remove = {"CheckNumerics": True}

    input_nodes = input_graph.node
    names_to_remove = {}
    for node in input_nodes:
      if node.op in types_to_remove:
        names_to_remove[node.name] = True

    nodes_after_removal = []
    for node in input_nodes:
      if node.name in names_to_remove:
        continue
      new_node = tf.NodeDef()
      new_node.CopyFrom(node)
      input_before_removal = node.input
      del new_node.input[:]
      for full_input_name in input_before_removal:
        input_name = re.sub(r"^\^", "", full_input_name)
        if input_name in names_to_remove:
          continue
        new_node.input.append(full_input_name)
      nodes_after_removal.append(new_node)

    types_to_splice = {"Identity": True}
    names_to_splice = {}
    for node in nodes_after_removal:
      if node.op in types_to_splice:
        # We don't want to remove nodes that have control edge inputs, because
        # they might be involved in subtle dependency issues that removing them
        # will jeopardize.
        has_control_edge = False
        for input_name in node.input:
          if re.match(r"^\^", input_name):
            has_control_edge = True
        if not has_control_edge:
          names_to_splice[node.name] = node.input[0]

    nodes_after_splicing = []
    for node in nodes_after_removal:
      if node.name in names_to_splice:
        continue
      new_node = tf.NodeDef()
      new_node.CopyFrom(node)
      input_before_removal = node.input
      del new_node.input[:]
      for full_input_name in input_before_removal:
        input_name = re.sub(r"^\^", "", full_input_name)
        if input_name in names_to_splice:
          new_node.input.append(names_to_splice[input_name])
        else:
          new_node.input.append(full_input_name)
      nodes_after_splicing.append(new_node)

    output_graph = tf.GraphDef()
    output_graph.node.extend(nodes_after_splicing)
    return output_graph

  def set_input_graph(self, new_input_graph):
    self.input_graph = new_input_graph
    self.nodes_map = self.create_nodes_map(self.input_graph)


def main(unused_args):
  if not tf.gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  known_modes = ["round", "quantize", "eightbit", "weights", "test",
                 "weights_rounded"]
  if not any(FLAGS.mode in s for s in known_modes):
    print("mode is '" + FLAGS.mode + "', not in " + ", ".join(known_modes) +
          ".")
    return -1

  tf_graph = tf.GraphDef()
  with tf.gfile.Open(FLAGS.input, "r") as f:
    data = f.read()
    tf_graph.ParseFromString(data)

  graph = tf.Graph()
  with graph.as_default():
    tf.import_graph_def(tf_graph, input_map={}, name="")

  rewriter = GraphRewriter(tf_graph, FLAGS.mode)

  output_graph = rewriter.rewrite(FLAGS.output_node_names.split(","))

  f = tf.gfile.FastGFile(FLAGS.output, "w")
  f.write(output_graph.SerializeToString())

  return 0


if __name__ == "__main__":
  tf.app.run()
