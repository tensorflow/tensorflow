# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Utlity to convert FunctionDef to GraphDef and Graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import versions


def function_def_to_graph(fdef, input_shapes=None):
  """Converts a FunctionDef to a function._FuncGraph (sub-class Graph).

  The returned _FuncGraph's `name`, `inputs` and `outputs` fields will be set.
  The input tensors are represented as placeholders.

  Note: `_FuncGraph.inputs` and `_FuncGraph._captured` are not set and may be
  set by the caller.

  Args:
    fdef: FunctionDef.
    input_shapes: Optional. A list of TensorShape objects of the shapes of
      function inputs. If specified, its length must match length of
      `fdef.signature.input_arg`. If a shape is None, the corresponding input
      placeholder will have unknown shape.

  Returns:
    A _FuncGraph.
  """
  func_graph = function._FuncGraph(fdef.signature.name, capture_by_value=False)  # pylint: disable=protected-access
  graph_def, nested_to_flat_tensor_name = function_def_to_graph_def(
      fdef, input_shapes)

  with func_graph.as_default():
    # Add all function nodes to the graph.
    importer.import_graph_def(graph_def, name="")

    # Initialize fields specific to _FuncGraph.

    # inputs
    input_tensor_names = [
        nested_to_flat_tensor_name[arg.name] for arg in fdef.signature.input_arg
    ]
    func_graph.inputs = [
        func_graph.get_tensor_by_name(name) for name in input_tensor_names
    ]

    # outputs
    output_tensor_names = [
        nested_to_flat_tensor_name[fdef.ret[arg.name]]
        for arg in fdef.signature.output_arg
    ]
    func_graph.outputs = [
        func_graph.get_tensor_by_name(name) for name in output_tensor_names
    ]

  return func_graph


def function_def_to_graph_def(fdef, input_shapes=None):
  """Convert a FunctionDef to a GraphDef.

  Steps:
  1. Creates placeholder nodes corresponding to inputs in
     `FunctionDef.signature.input_arg`.
  2. Adds NodeDefs in `FunctionDef.node_def` to `GraphDef.node`.
  3. Renames inputs of all nodes to use the convention of GraphDef instead of
     FunctionDef. See comment on `FunctionDef.node_def` on how the tensor naming
     in FunctionDefs is different from GraphDefs.

  Args:
    fdef: FunctionDef.
    input_shapes: Optional. A list of TensorShape objects of the shapes of
      function inputs. If specified, its length must match length of
      `fdef.signature.input_arg`. If a shape is None, the corresponding input
      placeholder will have unknown shape.

  Returns:
    A tuple of (GraphDef, dict<string, string>). The dict contains a mapping
    from nested tensor names (in FunctionDef) to flattened names (in GraphDef).

  Raises:
    ValueError: If the length of input_shapes does not match the number of
      input_args or if the FunctionDef is invalid.
  """
  graph_def = graph_pb2.GraphDef()
  graph_def.versions.CopyFrom(
      versions_pb2.VersionDef(
          producer=versions.GRAPH_DEF_VERSION,
          min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER))

  if input_shapes and len(input_shapes) != len(fdef.signature.input_arg):
    raise ValueError("Length of input_shapes must match the number of " +
                     "input_args. len(input_shapes): {} len(input_arg): {}".
                     format(len(input_shapes), len(fdef.signature.input_arg)))

  # 1. Create placeholders for input nodes.
  for i, arg_def in enumerate(fdef.signature.input_arg):
    node_def = graph_def.node.add()
    node_def.name = arg_def.name
    node_def.op = "Placeholder"
    node_def.attr["dtype"].type = arg_def.type
    if input_shapes and input_shapes[i] is not None:
      node_def.attr["shape"].shape.CopyFrom(input_shapes[i].as_proto())

  # 2. Copy all body NodeDefs to the GraphDef.
  graph_def.node.extend(fdef.node_def)

  # 3. Perform the renaming.

  # Build the tensor name mapping then flatten the tensor names.
  # See comment on `FunctionDef.node_def` on how the tensor naming in
  # FunctionDefs is different from GraphDefs.
  nested_to_flat_tensor_name = {}

  for arg_def in fdef.signature.input_arg:
    nested_to_flat_tensor_name[arg_def.name] = "{}:0".format(arg_def.name)

  for node_def in fdef.node_def:
    op_def = op_def_registry.get_registered_ops().get(node_def.op)
    if not op_def:
      # TODO(b/80470245): Support functions which refer other functions.
      raise NotImplementedError(
          "No op registered for {},".format(node_def.op) +
          " it may be a function. function_def_to_graph_def " +
          "currently does not support converting functions with " +
          "references to other graph functions.")

    for attr in op_def.attr:
      if attr.type in ("func", "list(func)"):
        # TODO(b/80470245): Support functions which refer other functions.
        raise NotImplementedError("Unsupported attr {} ".format(attr.name) +
                                  " with type {}".format(attr.type) +
                                  " in op {}. ".format(op_def.name) +
                                  "function_def_to_graph_def currently does " +
                                  "not support converting functions with " +
                                  "references to other graph functions.")

    # Iterate over output_args in op_def to build the map.
    # Index of the output tensor in the flattened list of *all* output
    # tensors of the op.
    flattened_index = 0
    for arg_def in op_def.output_arg:
      num_args = _get_num_args(arg_def, node_def)
      for i in range(num_args):
        # Map tensor names from "node_name:output_arg_name:index" to
        # "node_name:flattened_index".
        nested_name = "{}:{}:{}".format(node_def.name, arg_def.name, i)
        flat_name = "{}:{}".format(node_def.name, flattened_index)
        nested_to_flat_tensor_name[nested_name] = flat_name
        flattened_index += 1

  # Update inputs of all nodes in graph.
  for node_def in graph_def.node:
    for i in range(len(node_def.input)):
      node_def.input[i] = nested_to_flat_tensor_name[node_def.input[i]]

  return graph_def, nested_to_flat_tensor_name


# Based on implementation in core/framework/node_def_util.cc::ComputeArgRange.
def _get_num_args(arg_def, node_def):
  if arg_def.number_attr:
    return node_def.attr[arg_def.number_attr].i
  elif arg_def.type_list_attr:
    return len(node_def.attr[arg_def.type_list_attr].list.type)
  elif arg_def.type_attr or arg_def.type != types_pb2.DT_INVALID:
    return 1
  else:
    raise ValueError("Invalid arg_def:\n\n{}".format(str(arg_def)))
