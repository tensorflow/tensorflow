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

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import variable_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.training.saver import export_meta_graph


def _disable_lower_using_switch_merge(graph_def):
  """Set '_lower_using_switch_merge' attributes to False in If and While ops.

  Args:
    graph_def: GraphDef proto.

  Returns:
    GraphDef
  """
  output_graph_def = graph_pb2.GraphDef()
  output_graph_def.library.CopyFrom(graph_def.library)
  output_graph_def.versions.CopyFrom(graph_def.versions)

  for input_node in graph_def.node:
    output_node = output_graph_def.node.add()
    output_node.CopyFrom(input_node)
    if output_node.op in ("If", "While"):
      output_node.attr["_lower_using_switch_merge"].b = False
  return output_graph_def


def _run_inline_graph_optimization(func, lower_control_flow):
  """Apply function inline optimization to the graph.

  Returns the GraphDef after Grappler's function inlining optimization is
  applied. This optimization does not work on models with control flow.

  Args:
    func: ConcreteFunction.
    lower_control_flow: Boolean indicating whether or not to lower control flow
      ops such as If and While. (default True)

  Returns:
    GraphDef
  """
  graph_def = func.graph.as_graph_def()
  if not lower_control_flow:
    graph_def = _disable_lower_using_switch_merge(graph_def)
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
  return tf_optimizer.OptimizeGraph(config, meta_graph)


def _get_tensor_name(name):
  """Returns the name of the input tensor.

  Args:
    name: str

  Returns:
    str
  """
  return name.split(":")[0]


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
  map_index_to_variable = {
      func.captured_inputs.index(var.handle): var
      for var in func.graph.variables
  }

  # Iterates through all captures which are represented as Placeholders.
  for idx, (val_tensor, name_tensor) in enumerate(func.graph.captures.items()):
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


def _get_control_flow_function_types(graph_def, tensor_data):
  """Gets the types for the parameters to the function.

  Creates a map from function name to a list of types that correspond with the
  function arguments. The type is primarily determined from the corresponding
  "If" op. If the argument is a resource variable, then the type is determined
  from the type of the data contained within the Tensor.

  Args:
    graph_def: GraphDef proto.
    tensor_data: {str name : Tensor}.

  Returns:
    {str function name : [int representing DataType]}
  """
  # TODO(b/133793620): Support the "While" op.
  func_types = {}
  for node in graph_def.node:
    if node.op == "If":
      arg_types = [dtype for dtype in node.attr["Tin"].list.type]

      for idx in range(len(arg_types)):
        if arg_types[idx] == dtypes.resource:
          # Skip first index which represents the condition.
          input_name = node.input[idx + 1]
          numpy_type = tensor_data[input_name]["data"].dtype
          arg_types[idx] = dtypes.as_dtype(numpy_type).as_datatype_enum

      func_types[node.attr["then_branch"].func.name] = arg_types
      func_types[node.attr["else_branch"].func.name] = arg_types
  return func_types


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
  input_tensors = list(func.graph.captures.values())
  converted_inputs = set(
      [input_tensors[index] for index in converted_input_indices])
  not_converted_inputs = set(func.inputs).difference(converted_inputs)
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


def convert_variables_to_constants_v2(func, lower_control_flow=True):
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

  Returns:
    ConcreteFunction containing a simplified version of the original.
  """
  # TODO(nupurgarg): Replace ResourceGather with Gather.
  # Inline the graph in order to remove functions when possible.
  graph_def = _run_inline_graph_optimization(func, lower_control_flow)

  # Get mapping from node name to node.
  name_to_node = {_get_tensor_name(node.name): node for node in graph_def.node}

  # Get mapping from node name to variable value.
  tensor_data = _get_tensor_data(func)

  # Get mapping from function name to argument types.
  get_new_func_name = lambda func_name: func_name + "_frozen"
  function_types = _get_control_flow_function_types(graph_def, tensor_data)

  # Get variable data.
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

  for node in graph_def.node:
    if node.op == "If":
      # Get dtype and data for resource Placeholders.
      then_func = node.attr["then_branch"].func.name
      arg_types = function_types[then_func]
      for idx, input_tensor in enumerate(node.input[1:]):
        input_name = _get_tensor_name(input_tensor)
        if input_name in tensor_data:
          dtype = attr_value_pb2.AttrValue(type=arg_types[idx])
          _save_placeholder(_get_tensor_name(input_tensor), dtype)
    if node.op == "VariableV2":
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
    elif node.op == "ReadVariableOp":
      # Get dtype and data for Placeholder ops associated with ReadVariableOp.
      # There can be an Identity in between the ReadVariableOp and Placeholder.
      # Store the dtype for the Identity ops.
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
  how_many_converted = 0

  for input_node in graph_def.node:
    output_node = output_graph_def.node.add()
    # Convert VariableV2 ops to Const ops.
    if input_node.name in reference_variables:
      data = reference_variables[input_node.name]
      dtype = attr_value_pb2.AttrValue(type=data.dtype.as_datatype_enum)
      _populate_const_op(output_node, input_node.name, dtype, data.numpy(),
                         data.shape)
      how_many_converted += 1
    # Convert Placeholder ops to Const ops.
    elif input_node.name in placeholders:
      data = placeholders[input_node.name]["data"]
      dtype = placeholders[input_node.name]["dtype"]
      _populate_const_op(output_node, input_node.name, dtype, data, data.shape)
      how_many_converted += 1
    # Update the dtype for Identity ops that are inputs to ReadVariableOps.
    elif input_node.name in resource_identities:
      output_node.CopyFrom(input_node)
      output_node.attr["T"].CopyFrom(resource_identities[input_node.name])
    # Convert ReadVariableOps to Identity ops.
    elif input_node.op == "ReadVariableOp":
      _populate_identity_op(output_node, input_node)
    # Update the function names and function's arguments types for the If ops.
    elif input_node.op == "If":
      output_node.CopyFrom(input_node)
      then_func = input_node.attr["then_branch"].func.name
      output_node.attr["then_branch"].func.name = get_new_func_name(then_func)
      output_node.attr["else_branch"].func.name = get_new_func_name(
          input_node.attr["else_branch"].func.name)
      output_node.attr["Tin"].list.CopyFrom(
          attr_value_pb2.AttrValue.ListValue(type=function_types[then_func]))
    else:
      output_node.CopyFrom(input_node)

  # Add functions to reconstructed graph.
  if graph_def.library:
    library = output_graph_def.library

    for input_library_func in graph_def.library.function:
      orig_func_name = input_library_func.signature.name
      new_func_name = get_new_func_name(orig_func_name)

      # Do not copy any functions that aren't being used in the graph. Any
      # functions that are not used by control flow should have been inlined.
      if orig_func_name not in function_types:
        continue

      output_library_func = library.function.add()
      for key, value in input_library_func.ret.items():
        output_library_func.ret[key] = value
      for key, value in input_library_func.control_ret.items():
        output_library_func.control_ret[key] = value

      # Update the input types in the function signature.
      output_library_func.signature.CopyFrom(input_library_func.signature)
      output_library_func.signature.name = new_func_name
      for dtype, arg in zip(function_types[orig_func_name],
                            output_library_func.signature.input_arg):
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
  return _construct_concrete_function(func, output_graph_def,
                                      converted_input_indices)
