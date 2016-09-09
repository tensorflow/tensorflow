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

"""Utilities to remove unneeded nodes from a GraphDefs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy

import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.framework import graph_util


def strip_unused(input_graph_def, input_node_names, output_node_names,
                 placeholder_type_enum):
  """Removes unused nodes from a GraphDef.

  Args:
    input_graph_def: A graph with nodes we want to prune.
    input_node_names: A list of the nodes we use as inputs.
    output_node_names: A list of the output nodes.
    placeholder_type_enum: The AttrValue enum for the placeholder data type.

  Returns:
    A GraphDef with all unnecessary ops removed.
  """
  # Here we replace the nodes we're going to override as inputs with
  # placeholders so that any unused nodes that are inputs to them are
  # automatically stripped out by extract_sub_graph().
  inputs_replaced_graph_def = tf.GraphDef()
  for node in input_graph_def.node:
    if node.name in input_node_names:
      placeholder_node = tf.NodeDef()
      placeholder_node.op = "Placeholder"
      placeholder_node.name = node.name
      placeholder_node.attr["dtype"].CopyFrom(tf.AttrValue(
          type=placeholder_type_enum))
      if "_output_shapes" in node.attr:
        placeholder_node.attr["_output_shapes"].CopyFrom(
            node.attr["_output_shapes"])
      inputs_replaced_graph_def.node.extend([placeholder_node])
    else:
      inputs_replaced_graph_def.node.extend([copy.deepcopy(node)])

  output_graph_def = graph_util.extract_sub_graph(inputs_replaced_graph_def,
                                                  output_node_names)
  return output_graph_def


def strip_unused_from_files(input_graph, input_binary, output_graph,
                            output_binary, input_node_names, output_node_names,
                            placeholder_type_enum):
  """Removes unused nodes from a graph file."""

  if not tf.gfile.Exists(input_graph):
    print("Input graph file '" + input_graph + "' does not exist!")
    return -1

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  input_graph_def = tf.GraphDef()
  mode = "rb" if input_binary else "r"
  with tf.gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read().decode("utf-8"), input_graph_def)

  output_graph_def = strip_unused(input_graph_def, input_node_names.split(","),
                                  output_node_names.split(","),
                                  placeholder_type_enum)

  if output_binary:
    with tf.gfile.GFile(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString())
  else:
    with tf.gfile.GFile(output_graph, "w") as f:
      f.write(text_format.MessageToString(output_graph_def))
  print("%d ops in the final graph." % len(output_graph_def.node))
