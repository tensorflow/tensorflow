# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Helpers to manipulate a tensor graph in python.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import six

# pylint: disable=unused-import
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework.graph_util_impl import _assert_nodes_are_present
from tensorflow.python.framework.graph_util_impl import _bfs_for_reachable_nodes
from tensorflow.python.framework.graph_util_impl import _extract_graph_summary
from tensorflow.python.framework.graph_util_impl import _node_name


__all__ = ["fuse_op", "get_placeholders"]


def fuse_op(graph_def, input_nodes, output_nodes, output_dtypes,
            output_quantized, op_name, op_type):
  """Fuse subgraph between input_nodes and output_nodes into a single custom op.

  Args:
    graph_def: A graph_pb2.GraphDef proto.
    input_nodes: input nodes to the subgraph to be fused.
    output_nodes: output nodes to the subgraph to be fused.
    output_dtypes: A list of output datatypes for the custom op
    output_quantized: A boolean flag that indicates if output is quantized
    op_name: fused op name.
    op_type: fused op type.
  Returns:
    The GraphDef of the new graph.

  Raises:
    TypeError: If 'graph_def' is not a graph_pb2.GraphDef proto.
  """

  if not isinstance(graph_def, graph_pb2.GraphDef):
    raise TypeError("graph_def must be a graph_pb2.GraphDef proto.")

  if isinstance(input_nodes, six.string_types):
    raise TypeError("input_nodes must be a list.")

  if isinstance(output_nodes, six.string_types):
    raise TypeError("output_nodes must be a list.")

  name_to_input_name, name_to_node, name_to_seq_num = _extract_graph_summary(
      graph_def)
  _assert_nodes_are_present(name_to_node, input_nodes + output_nodes)

  # Nodes upto and including input_nodes
  reachable_by_input = _bfs_for_reachable_nodes(input_nodes, name_to_input_name)
  # Nodes upto and including output_nodes
  reachable_by_output = _bfs_for_reachable_nodes(output_nodes,
                                                 name_to_input_name)

  # Set of nodes in the list input_nodes
  input_nodes_set = set(input_nodes)

  # Set of nodes in the list output_nodes
  output_nodes_set = set(output_nodes)

  nodes_post_output = []
  for node in graph_def.node:
    n = _node_name(node.name)
    if n in reachable_by_output:
      if n not in reachable_by_input and n not in output_nodes_set:
        # n is between input and output, i.e., part of the fused op
        next_to_visit = [n]
        visited = set()
        while next_to_visit:
          cur_node = next_to_visit[0]
          visited.add(cur_node)
          del next_to_visit[0]
          if cur_node in reachable_by_input and cur_node not in input_nodes_set:
            raise TypeError("Node %s uses input %s not in input_nodes." %
                            (n, cur_node))
          if cur_node not in input_nodes_set:
            next_to_visit += [
                input_node for input_node in name_to_input_name[cur_node]
                if input_node not in visited
            ]
    elif n not in reachable_by_input:
      nodes_post_output.append(n)

  # Add all nodes upto the input nodes
  out = graph_pb2.GraphDef()
  reachable_by_input_sorted = sorted(
      list(reachable_by_input), key=lambda n: name_to_seq_num[n])
  for node in reachable_by_input_sorted:
    out.node.extend([copy.deepcopy(name_to_node[node])])

  # Add the custom op
  new_node = node_def_pb2.NodeDef()
  for node in input_nodes:
    new_node.input.append(node)
  new_node.attr["_output_types"].list.type[:] = output_dtypes
  new_node.attr["_output_quantized"].b = output_quantized
  new_node.op = op_type
  new_node.name = op_name
  out.node.extend([new_node])

  # Add the nodes in the output of the custom op
  for index, n in enumerate(output_nodes):
    assert len(name_to_node[n].input) == 1
    new_node = copy.deepcopy(name_to_node[n])
    del new_node.input[:]
    new_node.input.append(op_name + (":" + str(index) if index != 0 else ""))
    out.node.extend([new_node])

  # Add the nodes post output_nodes
  for n in nodes_post_output:
    out.node.extend([copy.deepcopy(name_to_node[n])])

  out.library.CopyFrom(graph_def.library)
  out.versions.CopyFrom(graph_def.versions)
  return out


def get_placeholders(graph):
  """Get placeholders of a graph.

  For example:

  ```python
  a = tf.placeholder(dtype=tf.float32, shape=[2, 2], name='a')
  a = tf.placeholder(dtype=tf.int32, shape=[3, 2], name='b')

  tf.contrib.framework.get_placeholders(tf.get_default_graph())
  # Returns:
  #  [<tf.Tensor 'a:0' shape=(2, 2) dtype=float32>,
  #   <tf.Tensor 'b:0' shape=(3, 2) dtype=int32>]
  ```

  Args:
    graph: A tf.Graph.
  Returns:
    A list contains all placeholders of given graph.

  Raises:
    TypeError: If `graph` is not a tensorflow graph.
  """

  if not isinstance(graph, ops.Graph):
    raise TypeError("Input graph needs to be a Graph: %s" % graph)

  # For each placeholder() call, there is a corresponding
  # operation of type 'Placeholder' registered to the graph.
  # The return value (a Tensor) of placeholder() is the
  # first output of this operation in fact.
  operations = graph.get_operations()
  result = [i.outputs[0] for i in operations if i.type == "Placeholder"]
  return result
