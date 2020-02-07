# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Helpers to convert variables to constants in TensorFlow 2.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.util import object_identity
from tensorflow.python.training.saver import export_meta_graph


_CONDITIONAL_OPS = set(["If", "StatelessIf"])
_LOOP_OPS = set(["While", "StatelessWhile"])
_CONTROL_FLOW_OPS = _CONDITIONAL_OPS.union(_LOOP_OPS)


def disable_lower_using_switch_merge(graph_def):
  """Set '_lower_using_switch_merge' attributes to False.

  Sets the attribute to False in the NodeDefs in the main graph and the NodeDefs
  in each function's graph.

  Args:
    graph_def: GraphDef proto.

  Returns:
    GraphDef
  """
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.CopyFrom(graph_def)

  def disable_control_flow_lowering(node):
    if node.op in _CONTROL_FLOW_OPS:
      node.attr["_lower_using_switch_merge"].b = False

  for node in output_graph_def.node:
    disable_control_flow_lowering(node)

  if output_graph_def.library:
    for func in output_graph_def.library.function:
      for node in func.node_def:
        disable_control_flow_lowering(node)
  return output_graph_def


def _run_inline_graph_optimization(func, lower_control_flow,
                                   aggressive_inlining):
  """Apply function inline optimization to the graph.

  Returns the GraphDef after Grappler's function inlining optimization is
  applied. This optimization does not work on models with control flow.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to to aggressive
      function inlining (might be unsafe if function has stateful ops not
      properly connected to control outputs).

  Returns:
    GraphDef
  """
  graph_def = func.graph.as_graph_def()
  if not lower_control_flow:
    graph_def = disable_lower_using_switch_merge(graph_def)

  # In some cases, a secondary implementation of the function (e.g. for GPU) is
  # written to the "api_implements" attribute. (e.g. `tf.keras.layers.LSTM` in
  # TF2 produces a CuDNN-based RNN for GPU).
  # This function suppose to inline all functions calls, but "api_implements"
  # prevents this from happening. Removing the attribute solves the problem.
  # To learn more about "api_implements", see:
  #   tensorflow/core/grappler/optimizers/implementation_selector.h
  for function in graph_def.library.function:
    if "api_implements" in function.attr:
      del function.attr["api_implements"]

  meta_graph = export_meta_graph(graph_def=graph_def, graph=func.graph)

  # Clear the initializer_name for the variables collections, since they are not
  # needed after saved to saved_model.
  for name in [
      "variables", "model_variables", "trainable_variables", "local_variables"
  ]:
    raw_list = []
    for raw in meta_graph.collection_def["variables"].bytes_list.value:
      variable = variable_pb2.VariableDef()
      variable.ParseFromString(raw)
      variable.ClearField("initializer_name")
      raw_list.append(variable.SerializeToString())
    meta_graph.collection_def[name].bytes_list.value[:] = raw_list

  # Add a collection 'train_op' so that Grappler knows the outputs.
  fetch_collection = meta_graph_pb2.CollectionDef()
  for array in func.inputs + func.outputs:
    fetch_collection.node_list.value.append(array.name)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

  # Initialize RewriterConfig with everything disabled except function inlining.
  config = config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  rewrite_options.min_graph_nodes = -1  # do not skip small graphs
  rewrite_options.optimizers.append("function")
  if aggressive_inlining:
    rewrite_options.function_optimization =\
      rewriter_config_pb2.RewriterConfig.AGGRESSIVE
  return tf_optimizer.OptimizeGraph(config, meta_graph)


def _get_tensor_name(name):
  """Returns the name of the input tensor.

  Args:
    name: str

  Returns:
    str
  """
  return name.split(":")[0]


def _get_new_function_name(name):
  """Returns the function name with '_frozen' appended.

  Args:
    name: str

  Returns:
    str
  """
  return name + "_frozen"


def _get_node_defs_list(graph_def):
  """Returns a list of NodeDefs in the GraphDef.

  This list consists of all NodeDefs in the main graph as well as all control
  flow NodeDefs in the functions.

  The remaining NodeDefs in the functions are not included because the op names
  are not unique and the variables are handled differently than the main graph.
  The control flow ops need to be extracted because they are need their
  attributes to be updated similar to the control flow ops in the main graph.

  Args:
    graph_def: GraphDef proto.

  Returns:
    [NodeDef]
  """
  node_defs = list(graph_def.node)

  if graph_def.library:
    for func in graph_def.library.function:
      node_defs.extend(
          [node for node in func.node_def if node.op in _CONTROL_FLOW_OPS])
  return node_defs


def _get_tensor_data(func):
  """Gets the tensor data for all Placeholders in the model.

  Returns a dictionary that maps the tensor name to a dictionary containing:
    data: numpy data
    index: int index in func.graph.captures
    is_variable: bool indicating whether the tensor is a variable or not

  Args:
    func: ConcreteFunction.

  Returns:
    Dict
  """
  tensor_data = {}
  map_index_to_variable = {}
  for var in func.graph.variables:
    for idx, captured_input in enumerate(func.captured_inputs):
      if var.handle is captured_input:  # pylint: disable=protected-access
        map_index_to_variable[idx] = var
        break

  # Iterates through all captures which are represented as Placeholders.
  for idx, (val_tensor, name_tensor) in enumerate(func.graph.captures):
    tensor_name = _get_tensor_name(name_tensor.name)
    is_variable = idx in map_index_to_variable
    if is_variable:
      data = map_index_to_variable[idx].numpy()
    else:
      data = val_tensor.numpy()
    tensor_data[tensor_name] = {
        "data": data,
        "index": idx,
        "is_variable": is_variable,
    }
  return tensor_data


def _get_control_flow_function_data(node_defs, tensor_data, name_to_node):
  """Gets the types and shapes for the parameters to the function.

  Creates a map from function name to a list of types and a list of shapes that
  correspond with the function arguments. The data is primarily determined from
  the corresponding "If" or "While" op. If the argument is a resource variable,
  then the type is determined from the type of the data contained within the
  Tensor. The shape data is only determined in the case of the "While" op.

  `is_also_output_type` is used to identify the "While" bodies that require the
  output types to be updated at the same time the input types are updated.

  Args:
    node_defs: List of NodeDefs.
    tensor_data: {str name : Tensor}.
    name_to_node: Dictionary mapping node name to node object.

  Returns:
    {str function name : {"types" : [int representing DataType],
                          "shapes" : [[int] representing TensorShape]],
                          "is_also_output_type" : bool}
  """
  func_data = {}

  def get_source_node_name_through_identities(node_name):
    # Trace the source node along with a chain of Identity nodes.
    # For example, given Plaecholder -> Identity -> Identity -> node_name
    # The function will return the name of the Placeholder.
    while name_to_node[node_name].op == "Identity":
      node_name = _get_tensor_name(name_to_node[node_name].input[0])
    return node_name

  def get_resource_type(node_name):
    node_name = get_source_node_name_through_identities(node_name)

    numpy_type = tensor_data[node_name]["data"].dtype
    return dtypes.as_dtype(numpy_type).as_datatype_enum

  def get_resource_shape(node_name):
    node_name = get_source_node_name_through_identities(node_name)

    return tensor_shape_pb2.TensorShapeProto(dim=[
        tensor_shape_pb2.TensorShapeProto.Dim(size=dim)
        for dim in tensor_data[node_name]["data"].shape
    ])

  def add_value(func_name, arg_types, output_shapes, is_also_output_type):
    func_data[func_name] = {
        "types": arg_types,
        "shapes": output_shapes,
        "is_also_output_type": is_also_output_type
    }

  for node in node_defs:
    if node.op in _CONDITIONAL_OPS:
      arg_types = [dtype for dtype in node.attr["Tin"].list.type]

      for idx in range(len(arg_types)):
        if arg_types[idx] == dtypes.resource:
          # Skip first index which represents the condition.
          arg_types[idx] = get_resource_type(node.input[idx + 1])

      add_value(node.attr["then_branch"].func.name, arg_types, None, False)
      add_value(node.attr["else_branch"].func.name, arg_types, None, False)
    elif node.op in _LOOP_OPS:
      arg_types = [dtype for dtype in node.attr["T"].list.type]
      output_shapes = [shape for shape in node.attr["output_shapes"].list.shape]

      for idx in range(len(arg_types)):
        if arg_types[idx] == dtypes.resource:
          input_name = node.input[idx]
          arg_types[idx] = get_resource_type(input_name)
          output_shapes[idx] = get_resource_shape(input_name)

      add_value(node.attr["body"].func.name, arg_types, output_shapes, True)
      add_value(node.attr["cond"].func.name, arg_types, output_shapes, False)
  return func_data


def _populate_const_op(output_node, node_name, dtype, data, data_shape):
  """Creates a Const op.

  Args:
    output_node: TensorFlow NodeDef.
    node_name: str node name.
    dtype: AttrValue with a populated .type field.
    data: numpy data value.
    data_shape: Tuple of integers containing data shape.
  """
  output_node.op = "Const"
  output_node.name = node_name
  output_node.attr["dtype"].CopyFrom(dtype)
  tensor = tensor_util.make_tensor_proto(
      data, dtype=dtype.type, shape=data_shape)
  output_node.attr["value"].tensor.CopyFrom(tensor)


def _populate_identity_op(output_node, input_node):
  """Creates an Identity op from a ReadVariable op.

  Args:
    output_node: TensorFlow NodeDef.
    input_node: TensorFlow NodeDef.
  """
  output_node.op = "Identity"
  output_node.name = input_node.name
  output_node.input.append(input_node.input[0])
  output_node.attr["T"].CopyFrom(input_node.attr["dtype"])
  if "_class" in input_node.attr:
    output_node.attr["_class"].CopyFrom(input_node.attr["_class"])


def _populate_if_op(output_node, input_node, function_data):
  """Updates the type attributes and function names of If or StatelessIf.

  Args:
    output_node: TensorFlow NodeDef.
    input_node: TensorFlow NodeDef.
    function_data: Map of function names to the list of types and shapes that
      correspond with the function arguments.
  """
  output_node.CopyFrom(input_node)
  then_func = input_node.attr["then_branch"].func.name
  output_node.attr["then_branch"].func.name = _get_new_function_name(then_func)
  output_node.attr["else_branch"].func.name = _get_new_function_name(
      input_node.attr["else_branch"].func.name)
  output_node.attr["Tin"].list.CopyFrom(
      attr_value_pb2.AttrValue.ListValue(
          type=function_data[then_func]["types"]))


def _populate_while_op(output_node, input_node, function_data):
  """Updates the type attributes and function names of While or StatelessWhile.

  Args:
    output_node: TensorFlow NodeDef.
    input_node: TensorFlow NodeDef.
    function_data: Map of function names to the list of types and shapes that
      correspond with the function arguments.
  """
  output_node.CopyFrom(input_node)
  cond_func = input_node.attr["cond"].func.name
  output_node.attr["cond"].func.name = _get_new_function_name(cond_func)
  output_node.attr["body"].func.name = _get_new_function_name(
      input_node.attr["body"].func.name)
  output_node.attr["T"].list.CopyFrom(
      attr_value_pb2.AttrValue.ListValue(
          type=function_data[cond_func]["types"]))
  output_node.attr["output_shapes"].list.CopyFrom(
      attr_value_pb2.AttrValue.ListValue(
          shape=function_data[cond_func]["shapes"]))


def _construct_concrete_function(func, output_graph_def,
                                 converted_input_indices):
  """Constructs a concrete function from the `output_graph_def`.

  Args:
    func: ConcreteFunction
    output_graph_def: GraphDef proto.
    converted_input_indices: Set of integers of input indices that were
      converted to constants.

  Returns:
    ConcreteFunction.
  """
  # Create a ConcreteFunction from the new GraphDef.
  input_tensors = func.graph.internal_captures
  converted_inputs = object_identity.ObjectIdentitySet(
      [input_tensors[index] for index in converted_input_indices])
  not_converted_inputs = [
      tensor for tensor in func.inputs if tensor not in converted_inputs]
  not_converted_inputs_map = {
      tensor.name: tensor for tensor in not_converted_inputs
  }

  new_input_names = [tensor.name for tensor in not_converted_inputs]
  new_output_names = [tensor.name for tensor in func.outputs]
  new_func = wrap_function.function_from_graph_def(output_graph_def,
                                                   new_input_names,
                                                   new_output_names)

  # Manually propagate shape for input tensors where the shape is not correctly
  # propagated. Scalars shapes are lost when wrapping the function.
  for input_tensor in new_func.inputs:
    input_tensor.set_shape(not_converted_inputs_map[input_tensor.name].shape)
  return new_func


def _convert_variables_to_constants_v2_impl(func,
                                            lower_control_flow=True,
                                            aggressive_inlining=False):
  """Replaces all the variables in a graph with constants of the same values.

  TensorFlow 2.0 function for converting all Variable ops into Const ops holding
  the same values. This makes it possible to describe the network fully with a
  single GraphDef file, and allows the removal of a lot of ops related to
  loading and saving the variables. This function runs Grappler's function
  inlining optimization in order to return a single subgraph.

  The current implementation only works for graphs that do not contain any
  control flow or embedding related ops.

  Note that the NodeDefs in the returned GraphDef contains the original node
  names if they are created by the graph optimization. Converting the GraphDef
  to concrete function will lose these debug information.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Inlining functions with stateful ops might lead to
      undefined execution if function call doesn't have an outgoing control
      edge and control outputs (they should be added automatically in TFv2).
      Aggressive mode disables safety checks in Grappler function optimizer.

  Returns:
    GraphDef containing a simplified version of the original and converted
    input indices that were converted to constants.
  """
  # Inline the graph in order to remove functions when possible.
  graph_def = _run_inline_graph_optimization(func, lower_control_flow,
                                             aggressive_inlining)

  # Gets list of all node defs include those in the library.
  node_defs = _get_node_defs_list(graph_def)

  # Get mapping from node name to node.
  name_to_node = {_get_tensor_name(node.name): node for node in node_defs}

  # Get mapping from node name to variable value.
  tensor_data = _get_tensor_data(func)

  # Get mapping from function name to argument types.
  function_data = _get_control_flow_function_data(
      node_defs, tensor_data, name_to_node)

  # Get variable data for all nodes in `node_defs`.
  reference_variables = {}
  resource_identities = {}
  placeholders = {}
  converted_input_indices = set()

  def _save_placeholder(node_name, dtype):
    placeholders[node_name] = {
        "dtype": dtype,
        "data": tensor_data[node_name]["data"],
    }
    converted_input_indices.add(tensor_data[node_name]["index"])

  for node in node_defs:
    if node.op in _CONDITIONAL_OPS:
      # Get dtype and data for resource Placeholders.
      then_func = node.attr["then_branch"].func.name
      arg_types = function_data[then_func]["types"]
      for idx, input_tensor in enumerate(node.input[1:]):
        input_name = _get_tensor_name(input_tensor)
        if input_name in tensor_data:
          dtype = attr_value_pb2.AttrValue(type=arg_types[idx])
          _save_placeholder(_get_tensor_name(input_tensor), dtype)
    elif node.op in _LOOP_OPS:
      # Get dtype and data for resource Placeholders.
      cond_func = node.attr["cond"].func.name
      arg_types = function_data[cond_func]["types"]
      for idx, input_tensor in enumerate(node.input):
        input_name = _get_tensor_name(input_tensor)
        if input_name in tensor_data:
          dtype = attr_value_pb2.AttrValue(type=arg_types[idx])
          _save_placeholder(_get_tensor_name(input_tensor), dtype)
    elif (node.op == "Identity" and node.attr["T"].type == dtypes.resource and
          name_to_node[_get_tensor_name(node.input[0])].op in _LOOP_OPS):
      # Store the dtype for Identity resource ops that are outputs of While ops.
      while_node = name_to_node[_get_tensor_name(node.input[0])]
      body_func = while_node.attr["body"].func.name
      input_data = node.input[0].split(":")
      idx = 0 if len(input_data) == 1 else int(input_data[1])

      dtype = attr_value_pb2.AttrValue(
          type=function_data[body_func]["types"][idx])
      resource_identities[node.name] = dtype
    elif node.op == "VariableV2":
      # Get data for VariableV2 ops (reference variables) that cannot be lifted.
      with func.graph.as_default():
        identity_node = array_ops.identity(
            func.graph.as_graph_element(node.name + ":0"))
      reference_variables[node.name] = (
          func.prune([], [identity_node.name])()[0])
    elif node.name in tensor_data and not tensor_data[node.name]["is_variable"]:
      # Get dtype and data for non-variable Placeholders (ex. values for 1.X
      # Const ops that are loaded as Placeholders in 2.0)
      _save_placeholder(node.name, node.attr["dtype"])
    elif node.op in ["ReadVariableOp", "ResourceGather", "ResourceGatherNd"]:
      # Get dtype and data for Placeholder ops associated with ReadVariableOp
      # and ResourceGather ops. There can be an Identity in between the
      # resource op and Placeholder. Store the dtype for the Identity ops.
      input_name = _get_tensor_name(node.input[0])
      while name_to_node[input_name].op == "Identity":
        resource_identities[input_name] = node.attr["dtype"]
        input_name = _get_tensor_name(name_to_node[input_name].input[0])
      if name_to_node[input_name].op != "Placeholder":
        raise ValueError("Cannot find the Placeholder op that is an input "
                         "to the ReadVariableOp.")
      _save_placeholder(input_name, node.attr["dtype"])

  # Reconstruct the graph with constants in place of variables.
  output_graph_def = graph_pb2.GraphDef()

  for input_node in graph_def.node:
    output_node = output_graph_def.node.add()
    # Convert VariableV2 ops to Const ops.
    if input_node.name in reference_variables:
      data = reference_variables[input_node.name]
      dtype = attr_value_pb2.AttrValue(type=data.dtype.as_datatype_enum)
      _populate_const_op(output_node, input_node.name, dtype, data.numpy(),
                         data.shape)
    # Convert Placeholder ops to Const ops.
    elif input_node.name in placeholders:
      data = placeholders[input_node.name]["data"]
      dtype = placeholders[input_node.name]["dtype"]
      _populate_const_op(output_node, input_node.name, dtype, data, data.shape)
    # Update the dtype for Identity ops that are inputs to ReadVariableOps.
    elif input_node.name in resource_identities:
      output_node.CopyFrom(input_node)
      output_node.attr["T"].CopyFrom(resource_identities[input_node.name])
    # Convert ReadVariableOps to Identity ops.
    elif input_node.op == "ReadVariableOp":
      _populate_identity_op(output_node, input_node)
    # Convert ResourceGather to Gather ops with a Const axis feeding into it.
    elif input_node.op == "ResourceGather":
      if input_node.attr["batch_dims"].i != 0:
        raise ValueError("batch_dims != 0 is not supported by freeze_graph.")
      output_axis_node = output_graph_def.node.add()
      axis_node_name = input_node.name + "/axis"
      axis_dtype = input_node.attr["Tindices"]
      axis_data = np.array(input_node.attr["batch_dims"].i)
      _populate_const_op(output_axis_node, axis_node_name, axis_dtype,
                         axis_data, axis_data.shape)

      output_node.op = "GatherV2"
      output_node.name = input_node.name
      output_node.input.extend(
          [input_node.input[0], input_node.input[1], axis_node_name])
      output_node.attr["Tparams"].CopyFrom(input_node.attr["dtype"])
      output_node.attr["Tindices"].CopyFrom(input_node.attr["Tindices"])
      output_node.attr["Taxis"].CopyFrom(axis_dtype)
      if "_class" in input_node.attr:
        output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
    elif input_node.op == "ResourceGatherNd":
      output_node.op = "GatherNd"
      output_node.name = input_node.name
      output_node.input.extend(
          [input_node.input[0], input_node.input[1]])
      output_node.attr["Tparams"].CopyFrom(input_node.attr["dtype"])
      output_node.attr["Tindices"].CopyFrom(input_node.attr["Tindices"])
      if "_class" in input_node.attr:
        output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
    # Update the function names and argument types for the conditional ops.
    elif input_node.op in _CONDITIONAL_OPS:
      _populate_if_op(output_node, input_node, function_data)
    elif input_node.op in _LOOP_OPS:
      _populate_while_op(output_node, input_node, function_data)
    else:
      output_node.CopyFrom(input_node)

  # Add functions to reconstructed graph.
  if graph_def.library:
    library = output_graph_def.library

    for input_library_func in graph_def.library.function:
      orig_func_name = input_library_func.signature.name
      new_func_name = _get_new_function_name(orig_func_name)

      # Do not copy any functions that aren't being used in the graph. Any
      # functions that are not used by control flow should have been inlined.
      if orig_func_name not in function_data:
        continue

      output_library_func = library.function.add()
      for key, value in input_library_func.ret.items():
        output_library_func.ret[key] = value
      for key, value in input_library_func.control_ret.items():
        output_library_func.control_ret[key] = value

      # Update the input types in the function signature. Update the output
      # types for functions that are while loop bodies.
      output_library_func.signature.CopyFrom(input_library_func.signature)
      output_library_func.signature.name = new_func_name
      for dtype, arg in zip(function_data[orig_func_name]["types"],
                            output_library_func.signature.input_arg):
        arg.type = dtype
      if function_data[orig_func_name]["is_also_output_type"]:
        for dtype, arg in zip(function_data[orig_func_name]["types"],
                              output_library_func.signature.output_arg):
          arg.type = dtype

      # Update the NodeDefs.
      func_variables = {
          node.name: node.input[0]
          for node in input_library_func.node_def
          if node.op == "ReadVariableOp"
      }

      for input_node in input_library_func.node_def:
        output_node = output_library_func.node_def.add()
        # Convert ReadVariableOps to Identity ops.
        if input_node.op == "ReadVariableOp":
          _populate_identity_op(output_node, input_node)
        # Update the function names and argument types for the conditional ops.
        elif input_node.op in _CONDITIONAL_OPS:
          _populate_if_op(output_node, input_node, function_data)
        elif input_node.op in _LOOP_OPS:
          _populate_while_op(output_node, input_node, function_data)
        else:
          output_node.CopyFrom(input_node)
          # Convert :value to :output for ops that use the ReadVariableOp.
          for idx, full_name in enumerate(input_node.input):
            input_name = _get_tensor_name(full_name)
            if input_name in func_variables:
              full_name_parts = full_name.split(":")
              full_name_parts[1] = "output"
              input_name = ":".join(full_name_parts)
              output_node.input[idx] = input_name

  output_graph_def.versions.CopyFrom(graph_def.versions)
  return (output_graph_def, converted_input_indices)


def convert_variables_to_constants_v2(func,
                                      lower_control_flow=True,
                                      aggressive_inlining=False):
  """Replaces all the variables in a graph with constants of the same values.

  TensorFlow 2.0 function for converting all Variable ops into Const ops holding
  the same values. This makes it possible to describe the network fully with a
  single GraphDef file, and allows the removal of a lot of ops related to
  loading and saving the variables. This function runs Grappler's function
  inlining optimization in order to return a single subgraph.

  The current implementation only works for graphs that do not contain any
  control flow or embedding related ops.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to to aggressive
      function inlining (might be unsafe if function has stateful ops, not
      properly connected to control outputs). (default False)

  Returns:
    ConcreteFunction containing a simplified version of the original.
  """
  output_graph_def, converted_inputs = _convert_variables_to_constants_v2_impl(
      func, lower_control_flow, aggressive_inlining)
  return _construct_concrete_function(func, output_graph_def, converted_inputs)


def convert_variables_to_constants_v2_as_graph(func,
                                               lower_control_flow=True,
                                               aggressive_inlining=False):
  """Replaces all the variables in a graph with constants of the same values.

  This function works as same as convert_variables_to_constants_v2, but it
  returns the intermediate `GraphDef` as well. This `GraphDef` contains all the
  debug information after all the transformations in the frozen phase.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)
    aggressive_inlining: Boolean indicating whether or not to to aggressive
      function inlining (might be unsafe if function has stateful ops, not
      properly connected to control outputs).

  Returns:
    ConcreteFunction containing a simplified version of the original, and also
    the intermediate GraphDef containing the node debug information for the
    transformations in the frozen phase.
  """
  graph_def, converted_inputs = _convert_variables_to_constants_v2_impl(
      func, lower_control_flow, aggressive_inlining)
  frozen_func = _construct_concrete_function(func, graph_def, converted_inputs)
  return frozen_func, graph_def
