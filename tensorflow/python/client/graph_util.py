# Copyright 2015 Google Inc. All Rights Reserved.
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

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.platform import logging

_VARIABLE_OPS = {
    "Assign",
    "AssignAdd",
    "AssignSub",
    "Queue",
    "ScatterAdd",
    "ScatterSub",
    "ScatterUpdate",
    "TruncatedNormal",
    "Variable",
}


def _is_variable_op(op):
  """Returns true if 'op' refers to a Variable node."""
  return op in _VARIABLE_OPS


def set_cpu0(device_string):
  """Creates a new device string based on `device_string' but using /CPU:0.

   If the device is already on /CPU:0, this is a no-op.

   Args:
     device_string: A device string.

   Returns:
     A device string.
  """
  parsed_device = pydev.from_string(device_string)
  parsed_device.device_type = "CPU"
  parsed_device.device_index = 0
  return parsed_device.to_string()


def must_run_on_cpu(node, pin_variables_on_cpu=False):
  """Returns True if the given node_def must run on CPU, otherwise False.

  Args:
    node: The node to be assigned to a device. Could be either an ops.Operation
      or NodeDef.
    pin_variables_on_cpu: If True, this function will return False if node_def
      represents a variable-related op.

  Returns:
    True if the given node must run on CPU, otherwise False.
  """

  if isinstance(node, ops.Operation):
    node_def = node.node_def
  else:
    assert isinstance(node, graph_pb2.NodeDef)
    node_def = node

  # If the op is a variable-related op, should we pin it on CPU?
  if pin_variables_on_cpu and _is_variable_op(node_def.op):
    return True

  # Constant operations producing a string or int32 must run on CPU.
  if node_def.op == "Const":
    # Get the value of the 'dtype' attr
    dtype = node_def.attr["dtype"].type
    if dtype == dtypes.string or dtype == dtypes.int32:
      return True

  if node_def.op == "DynamicStitch":
    dtype = node_def.attr["T"].type
    if dtype == dtypes.int32:
      # DynamicStitch on GPU only works for int32 values.
      return True

  if node_def.op in ["Cast"]:
    dtype = node_def.attr["SrcT"].type
    if dtype == dtypes.int32:
      # Cast on GPU does not works for int32 values.
      return True
  return False


################################################################################
#
# device functions for use in with g.device(...)
#
################################################################################


def pin_variables_on_cpu(op):
  """Returns a CPU device for Variable nodes if the device is not specified.

  Args:
    op: The ops.Operation object describing the node for which a device
      should be chosen. The op.device field is respected.

  Returns:
    A device containing "/device:CPU:0" if the node is related to a variable.
  """
  device = op.device if op.device is not None else ""
  dev = pydev.from_string(device)

  # If a device type exists already, do not override.
  if dev.device_type:
    return device

  if isinstance(op, ops.Operation):
    node_def = op.node_def
  else:
    assert isinstance(op, graph_pb2.NodeDef)
    node_def = op

  if _is_variable_op(node_def.op):
    return set_cpu0(device)
  return device


def pin_to_cpu(op):
  """Returns a CPU device for the given node."""
  device = op.device if op.device is not None else ""
  dev = pydev.from_string(device)

  if not dev.device_type:
    return set_cpu0(device)
  if dev.device_type == "CPU":
    return device

  logging.info("Operation %s has been assigned to a non-CPU (%s), so "
               "it will not be pinned to the CPU.", op.name, dev.device_type)
  return device


def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]


def extract_sub_graph(graph_def, dest_nodes):
  """Extract the subgraph that can reach any of the nodes in 'dest_nodes'.

  Args:
    graph_def: A graph_pb2.GraphDef proto.
    dest_nodes: A list of strings specifying the destination node names.
  Returns:
    The GraphDef of the sub-graph.

  Raises:
    TypeError: If 'graph_def' is not a graph_pb2.GraphDef proto.
  """

  if not isinstance(graph_def, graph_pb2.GraphDef):
    raise TypeError("graph_def must be a graph_pb2.GraphDef proto.")

  edges = {}  # Keyed by the dest node name.
  name_to_node_map = {}  # Keyed by node name.

  # Keeps track of node sequences. It is important to still output the
  # operations in the original order.
  node_seq = {}  # Keyed by node name.
  seq = 0
  for node in graph_def.node:
    n = _node_name(node.name)
    name_to_node_map[n] = node
    edges[n] = [_node_name(x) for x in node.input]
    node_seq[n] = seq
    seq += 1

  for d in dest_nodes:
    assert d in name_to_node_map, "%d is not in graph" % d

  nodes_to_keep = set()
  # Breadth first search to find all the nodes that we should keep.
  next_to_visit = dest_nodes[:]
  while next_to_visit:
    n = next_to_visit[0]
    del next_to_visit[0]
    if n in nodes_to_keep:
      # Already visited this node.
      continue
    nodes_to_keep.add(n)
    next_to_visit += edges[n]

  nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
  # Now construct the output GraphDef
  out = graph_pb2.GraphDef()
  for n in nodes_to_keep_list:
    out.node.extend([copy.deepcopy(name_to_node_map[n])])

  return out


def tensor_shape_from_node_def_name(graph, input_name):
  """Convenience function to get a shape from a NodeDef's input string."""
  # To get a tensor, the name must be in the form <input>:<port>, for example
  # 'Mul:0'. The GraphDef input strings don't always have the port specified
  # though, so if there isn't a colon we need to add a default ':0' to the end.
  if ":" not in input_name:
    canonical_name = input_name + ":0"
  else:
    canonical_name = input_name
  tensor = graph.get_tensor_by_name(canonical_name)
  shape = tensor.get_shape()
  return shape
