# pylint: disable=g-bad-file-header
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
r"""Removes parts of a graph that are only needed for training.

There are several common transformations that can be applied to GraphDefs
created to train a model, that help reduce the amount of computation needed when
the network is used only for inference. These include:

 - Removing training-only operations like checkpoint saving.

 - Stripping out parts of the graph that are never reached.

 - Removing debug operations like CheckNumerics.

 - Folding batch normalization ops into the pre-calculated weights.

This script takes a frozen GraphDef file (where the weight variables have been
converted into constants by the freeze_graph script) and outputs a new GraphDef
with the optimizations applied.

An example of command-line usage is:

bazel build tensorflow/python/tools:optimize_for_inference && \
bazel-bin/tensorflow/python/tools/optimize_for_inference \
--input_graph=some_graph_def.pb \
--output_graph=/tmp/optimized_graph.pb \
--input_node_names=Mul
--output_node_names=softmax

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import re
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_util
from tensorflow.python.tools import strip_unused_lib

flags = tf.app.flags
FLAGS = flags.FLAGS


def optimize_for_inference(input_graph_def, input_node_names,
                           output_node_names, placeholder_type_enum):
  """Applies a series of inference optimizations on the input graph.

  Args:
    input_graph_def: A GraphDef containing a training model.
    input_node_names: A list of names of the nodes that are fed inputs during
      inference.
    output_node_names: A list of names of the nodes that produce the final
      results.
    placeholder_type_enum: Data type of the placeholders used for inputs.

  Returns:
    An optimized version of the input graph.
  """
  stripped_graph_def = strip_unused_lib.strip_unused(input_graph_def,
                                                     input_node_names,
                                                     output_node_names,
                                                     placeholder_type_enum)
  detrained_graph_def = graph_util.remove_training_nodes(stripped_graph_def)
  folded_graph_def = fold_batch_norms(detrained_graph_def)
  return folded_graph_def


def node_name_from_input(node_name):
  """Strips off ports and other decorations to get the underlying node name."""
  if node_name.startswith("^"):
    node_name = node_name[1:]
  m = re.search(r"(.*):\d+$", node_name)
  if m:
    node_name = m.group(1)
  return node_name


def node_from_map(node_map, name):
  """Pulls a node def from a dictionary for a given name.

  Args:
    node_map: Dictionary containing an entry indexed by name for every node.
    name: Identifies the node we want to find.

  Returns:
    NodeDef of the node with the given name.

  Raises:
    ValueError: If the node isn't present in the dictionary.
  """
  stripped_name = node_name_from_input(name)
  if stripped_name not in node_map:
    raise ValueError("No node named '%s' found in map." % name)
  return node_map[stripped_name]


def values_from_const(node_def):
  """Extracts the values from a const NodeDef as a numpy ndarray.

  Args:
    node_def: Const NodeDef that has the values we want to access.

  Returns:
    Numpy ndarray containing the values.

  Raises:
    ValueError: If the node isn't a Const.
  """
  if node_def.op != "Const":
    raise ValueError(
        "Node named '%s' should be a Const op for values_from_const." %
        node_def.name)
  input_tensor = node_def.attr["value"].tensor
  tensor_value = tensor_util.MakeNdarray(input_tensor)
  return tensor_value


def fold_batch_norms(input_graph_def):
  """Removes batch normalization ops by folding them into convolutions.

  Batch normalization during training has multiple dynamic parameters that are
  updated, but once the graph is finalized these become constants. That means
  there's an opportunity to reduce the computations down to a scale and
  addition, rather than the more expensive multiple ops, and even bake the
  scaling into the convolution weights. This function identifies the typical
  pattern of batch normalization subgraphs, and performs the transformation to
  fold the computations down into a simpler form. It currently only spots batch
  normalization that's performed by the BatchNormWithGlobalNormalization op, and
  will need to be extended in the future to handle the newer style.

  Args:
    input_graph_def: A GraphDef containing a model.

  Returns:
    Modified graph with BN ops removed, and modified weights.

  Raises:
    ValueError: If the graph is badly formed with duplicate node names.
  """

  input_node_map = {}
  for node in input_graph_def.node:
    if node.name not in input_node_map.keys():
      input_node_map[node.name] = node
    else:
      raise ValueError("Duplicate node names detected.")

  nodes_to_skip = {}
  new_ops = []
  for node in input_graph_def.node:
    if node.op != "BatchNormWithGlobalNormalization":
      continue

    conv_op = node_from_map(input_node_map, node.input[0])
    if conv_op.op != "Conv2D":
      tf.logging.warning("Didn't find expected Conv2D input to '%s'" %
                         node.name)
      continue

    weights_op = node_from_map(input_node_map, conv_op.input[1])
    if weights_op.op != "Const":
      tf.logging.warning("Didn't find expected conv Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" %
                         (conv_op.name, weights_op))
      continue
    weights = values_from_const(weights_op)
    channel_count = weights.shape[3]

    mean_op = node_from_map(input_node_map, node.input[1])
    if mean_op.op != "Const":
      tf.logging.warning("Didn't find expected mean Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" %
                         (node.name, mean_op))
      continue
    mean_value = values_from_const(mean_op)
    if mean_value.shape != (channel_count,):
      tf.logging.warning("Incorrect shape for mean, found %s, expected %s,"
                         " for node %s" % (str(mean_value.shape),
                                           str((channel_count,)),
                                           node.name))
      continue

    var_op = node_from_map(input_node_map, node.input[2])
    if var_op.op != "Const":
      tf.logging.warning("Didn't find expected var Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" %
                         (node.name, var_op))
      continue
    var_value = values_from_const(var_op)
    if var_value.shape != (channel_count,):
      tf.logging.warning("Incorrect shape for var, found %s, expected %s,"
                         " for node %s" % (str(var_value.shape),
                                           str((channel_count,)),
                                           node.name))
      continue

    beta_op = node_from_map(input_node_map, node.input[3])
    if beta_op.op != "Const":
      tf.logging.warning("Didn't find expected beta Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" %
                         (node.name, beta_op))
      continue
    beta_value = values_from_const(beta_op)
    if beta_value.shape != (channel_count,):
      tf.logging.warning("Incorrect shape for beta, found %s, expected %s,"
                         " for node %s" % (str(beta_value.shape),
                                           str((channel_count,)),
                                           node.name))
      continue

    gamma_op = node_from_map(input_node_map, node.input[4])
    if gamma_op.op != "Const":
      tf.logging.warning("Didn't find expected gamma Constant input to '%s',"
                         " found %s instead. Maybe because freeze_graph wasn't"
                         " run first?" %
                         (node.name, gamma_op))
      continue
    gamma_value = values_from_const(gamma_op)
    if gamma_value.shape != (channel_count,):
      tf.logging.warning("Incorrect shape for gamma, found %s, expected %s,"
                         " for node %s" % (str(gamma_value.shape),
                                           str((channel_count,)),
                                           node.name))
      continue

    variance_epsilon_value = node.attr["variance_epsilon"].f
    scale_after_normalization = node.attr["scale_after_normalization"].b
    nodes_to_skip[node.name] = True
    nodes_to_skip[weights_op.name] = True
    nodes_to_skip[mean_op.name] = True
    nodes_to_skip[var_op.name] = True
    nodes_to_skip[beta_op.name] = True
    nodes_to_skip[gamma_op.name] = True
    nodes_to_skip[conv_op.name] = True

    if scale_after_normalization:
      scale_value = ((1.0 / np.vectorize(math.sqrt)
                      (var_value + variance_epsilon_value)) *
                     gamma_value)
    else:
      scale_value = (1.0 / np.vectorize(math.sqrt)
                     (var_value + variance_epsilon_value))
    offset_value = (-mean_value * scale_value) + beta_value
    scaled_weights = np.copy(weights)
    it = np.nditer(scaled_weights, flags=["multi_index"],
                   op_flags=["readwrite"])
    while not it.finished:
      current_scale = scale_value[it.multi_index[3]]
      it[0] *= current_scale
      it.iternext()
    scaled_weights_op = tf.NodeDef()
    scaled_weights_op.op = "Const"
    scaled_weights_op.name = weights_op.name
    scaled_weights_op.attr["dtype"].CopyFrom(weights_op.attr["dtype"])
    scaled_weights_op.attr["value"].CopyFrom(tf.AttrValue(
        tensor=tensor_util.make_tensor_proto(
            scaled_weights, weights.dtype.type, weights.shape)))
    new_conv_op = tf.NodeDef()
    new_conv_op.CopyFrom(conv_op)
    offset_op = tf.NodeDef()
    offset_op.op = "Const"
    offset_op.name = conv_op.name + "_bn_offset"
    offset_op.attr["dtype"].CopyFrom(mean_op.attr["dtype"])
    offset_op.attr["value"].CopyFrom(tf.AttrValue(
        tensor=tensor_util.make_tensor_proto(
            offset_value, mean_value.dtype.type, offset_value.shape)))
    bias_add_op = tf.NodeDef()
    bias_add_op.op = "BiasAdd"
    bias_add_op.name = node.name
    bias_add_op.attr["T"].CopyFrom(conv_op.attr["T"])
    bias_add_op.input.extend([new_conv_op.name, offset_op.name])
    new_ops.extend([scaled_weights_op, new_conv_op, offset_op, bias_add_op])

  result_graph_def = tf.GraphDef()
  for node in input_graph_def.node:
    if node.name in nodes_to_skip:
      continue
    new_node = tf.NodeDef()
    new_node.CopyFrom(node)
    result_graph_def.node.extend([new_node])

  result_graph_def.node.extend(new_ops)
  return result_graph_def
