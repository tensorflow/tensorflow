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

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.eager import function
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.saver import export_meta_graph


def _run_inline_graph_optimization(func):
  """Apply function inline optimization to the graph.

  Returns the GraphDef after Grappler's function inlining optimization is
  applied. This optimization does not work on models with control flow.

  Args:
    func: ConcreteFunction.

  Returns:
    GraphDef
  """
  meta_graph = export_meta_graph(
      graph_def=func.graph.as_graph_def(), graph=func.graph)

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


def _get_tensors_from_graph(graph, tensors):
  """Gets the Tensors in `graph` with the name of the tensors in `tensors`.

  Args:
    graph: TensorFlow Graph.
    tensors: List of Tensors.

  Returns:
    List of Tensors.
  """
  new_tensors = []
  for orig_tensor in tensors:
    new_tensor = graph.get_tensor_by_name(orig_tensor.name)
    if new_tensor.shape.rank is None:
      new_tensor.set_shape(orig_tensor.shape)
    new_tensors.append(new_tensor)
  return new_tensors


def _construct_concrete_function(input_func, graph_def):
  """Creates a ConcreteFunction from the input function and frozen graph.

  Args:
    input_func: ConcreteFunction.
    graph_def: TensorFlow GraphDef.

  Returns:
    ConcreteFunction containing the graph_def.
  """
  output_graph = func_graph.FuncGraph(input_func.graph.name)
  with output_graph.as_default():
    importer.import_graph_def(graph_def, name="")
    output_graph.inputs = _get_tensors_from_graph(output_graph,
                                                  input_func.inputs)
    output_graph.outputs = _get_tensors_from_graph(output_graph,
                                                   input_func.outputs)

  output_graph.structured_outputs = input_func.graph.structured_outputs
  output_graph.structured_input_signature = (
      input_func.graph.structured_input_signature)

  # pylint: disable=protected-access
  # Create the ConcreteFunction and add it to the global context.
  output_func = function.ConcreteFunction(
      output_graph, attrs=input_func._attrs, signature=input_func._signature)
  output_func.add_to_graph()

  # Inject the captured inputs into the ConcreteFunction.
  output_func._captured_inputs = input_func.captured_inputs
  output_func.graph.variables = input_func.graph.variables
  output_func._arg_keywords = input_func._arg_keywords
  output_func._num_positional_args = input_func._num_positional_args
  # pylint: enable=protected-access

  # Register the gradients in the current root context.
  with ops.init_scope():
    output_func._register_gradient()  # pylint: disable=protected-access
  return output_func


def convert_variables_to_constants_v2(func):
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

  Returns:
    ConcreteFunction containing a simplified version of the original.
  """
  # TODO(nupurgarg): Replace ResourceGather with Gather.
  # TODO(nupurgarg): Change attr for Variables in control flow and functions.
  graph_def = _run_inline_graph_optimization(func)

  # Identify the ReadVariableOps.
  get_name = lambda name: name.split(":")[0]
  map_name_to_node = {get_name(node.name): node for node in graph_def.node}

  # TODO(b/125838789): Use `func.graph.captures`.
  # Get mapping from input name to variable value.
  tensor_data = {}
  input_tensors = func.inputs[-len(func.captured_inputs):]
  for var in func.graph.variables:
    index = func.captured_inputs.index(var.handle)
    tensor = input_tensors[index]
    tensor_data[get_name(tensor.name)] = var.numpy()

  resource_identities = {}
  resource_placeholders = {}
  for node in graph_def.node:
    if node.op == "ReadVariableOp":
      # Get name of Placeholder op associated with ReadVariableOp. There can be
      # an Identity in between the ReadVariableOp and Placeholder. Store the
      # Identity ops with the associated dtypes.
      input_name = get_name(node.input[0])
      while map_name_to_node[input_name].op == "Identity":
        resource_identities[input_name] = node.attr["dtype"]
        input_name = get_name(map_name_to_node[input_name].input[0])
      if map_name_to_node[input_name].op != "Placeholder":
        raise ValueError("Cannot find the Placeholder op that is an input "
                         "to the ReadVariableOp.")
      # Build a map of Placeholder ops that are inputs to ReadVariableOps to the
      # variable's dtype and data.
      resource_placeholders[input_name] = {
          "dtype": node.attr["dtype"],
          "data": tensor_data[input_name],
      }

  # Reconstruct the graph with constants in place of variables.
  output_graph_def = graph_pb2.GraphDef()
  how_many_converted = 0

  for input_node in graph_def.node:
    output_node = output_graph_def.node.add()
    # Convert Placeholder ops that are inputs to ReadVariableOps into Const ops.
    if input_node.name in resource_placeholders:
      dtype = resource_placeholders[input_node.name]["dtype"]
      data = resource_placeholders[input_node.name]["data"]

      output_node.op = "Const"
      output_node.name = input_node.name
      output_node.attr["dtype"].CopyFrom(dtype)
      output_node.attr["value"].tensor.CopyFrom(
          tensor_util.make_tensor_proto(
              data, dtype=dtype.type, shape=data.shape))
      how_many_converted += 1
    # Change the dtype for Identity ops that are inputs to ReadVariableOps.
    elif input_node.name in resource_identities:
      output_node.CopyFrom(input_node)
      output_node.attr["T"].CopyFrom(resource_identities[input_node.name])
    # Convert ReadVariableOps into Identity ops.
    elif input_node.op == "ReadVariableOp":
      output_node.op = "Identity"
      output_node.name = input_node.name
      output_node.input.extend([input_node.input[0]])
      output_node.attr["T"].CopyFrom(input_node.attr["dtype"])
      if "_class" in input_node.attr:
        output_node.attr["_class"].CopyFrom(input_node.attr["_class"])
    else:
      output_node.CopyFrom(input_node)

  logging.info("Converted %d variables to const ops.", how_many_converted)
  # TODO(b/126613403): Use wrap_function.function_from_graph_def.
  return _construct_concrete_function(func, output_graph_def)
