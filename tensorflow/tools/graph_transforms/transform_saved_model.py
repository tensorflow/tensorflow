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
"""Tools for invoking the Graph Transform Tool on SavedModel files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import graph_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save
from tensorflow.python.training.tracking import tracking
from tensorflow.tools.graph_transforms import TransformGraph

import os
import shutil
import tempfile


def _find_function_by_name(func_graph_def, name):
  """Linear search for a function's FunctionDef in the graph that backs a
  ConcreteFunction."""
  for f in func_graph_def.library.function:
    if f.signature.name == name:
      return f
  raise ValueError("Graph does not contain function with expected name "
                   "'{}'".format(name))


def _node_part_of_tensor_name(tensor_name):
  return tensor_name.split(":")[0]


def _is_fn_call(n):
  return n.op in ("PartitionedCall", "StatefulPartitionedCall")


def _get_node_by_name(nodes, node_name):
  """Retrieve a node by name from an iterable of NodeDef protos."""
  for n in nodes:
    if n.name == node_name:
      return n
  raise ValueError("Graph does not contain a node that corresponds to result "
                   "node name '{}'. Node names are: {}"
                   "".format(node_name,
                             [n.name for n in nodes][:100]))


def _core_function_name(func_graph_def, result_node_name):
  """
  Search through a ConcreteFunction's graph to identify the core C++-layer
  function that implements the ConcreteFuncion.

  func_graph_def: GraphDef of a ConcreteFunction. This GraphDef may nest the
      implementation of the function inside multiple generated function calls
      for argument renaming, type labeling, and so on; but it should not
      include any actual processing outside of the core function.
  result_tensor_name: Name of a node in the "top-level graph" part of the
      GraphDef that produces part of the result of the ConcreteFunction's
      `call` method.
  """
  # Define some macros for dealing with ConcreteFunction GraphDefs.
  def func_call_to_fn_name(func_call_node):
    return func_call_node.attr["f"].func.name

  def first_fn_call_or_none(nodes):
    return next((n for n in nodes if _is_fn_call(n)), None)

  # Find the outermost function call in the chain. Assume that we
  # can reach this function call by tracing back through the first
  # input of each op.
  outer_call_node = _get_node_by_name(func_graph_def.node, result_node_name)
  while not _is_fn_call(outer_call_node):
    if 0 == len(outer_call_node.input):
      raise ValueError("Graph does not appear to contain a function call.")
    outer_call_node = _get_node_by_name(
        func_graph_def.node,
        _node_part_of_tensor_name(outer_call_node.input[0]))

  # Follow the rest of the chain. Assume that each function's graph
  # only contains one function call.
  cur_fn_call = outer_call_node
  while cur_fn_call is not None:
    cur_fn_def = _find_function_by_name(func_graph_def,
                                        func_call_to_fn_name(cur_fn_call))
    cur_fn_call = first_fn_call_or_none(cur_fn_def.node_def)
  return cur_fn_def.signature.name


def _copy_and_convert_tensors(orig, orig_graph, new_graph):
  """
  Copy an arbitrarily nested set of Python structures, converting Tensors from
  orig_graph to Tensors from new_graph along the way. Shallow-copies elements
  that are not Tensors, lists, or dictionaries."""
  if isinstance(orig, ops.Tensor):
    return new_graph.get_tensor_by_name(orig.name)
  elif isinstance(orig, list):
    return [_copy_and_convert_tensors(elem, orig_graph, new_graph)
            for elem in orig]
  elif isinstance(orig, dict):
    return {_copy_and_convert_tensors(key, orig_graph, new_graph):
            _copy_and_convert_tensors(val, orig_graph, new_graph)
            for key, val in orig.items()}
  else:
    return orig


def _rename_function(graph_def, old_name, new_name):
  """
  Rename a function embedded inside a GraphDef. Modifies the GraphDef's
  function library in place and applies the renaming to any function
  calls made from either the main graph or any functions in the library.

  Args:
  graph_def: GraphDef of the sort that FuncGraph.as_graph_def() produces.
  old_name: Current name of the target function
  new_name: Desired name for the function

  No return value. Modifies `graph_def` in place.
  """
  def _rename_calls(node_list):
    for n in node_list:
      if _is_fn_call(n) and n.attr["f"].func.name == old_name:
        n.attr["f"].func.name = new_name

  for f in graph_def.library.function:
    if f.signature.name == new_name:
      raise ValueError("GraphDef already has a function called '{}'"
                       "".format(new_name))

  # Pass 1: Function signature. Also validates old_name param
  func_def_ptr = _find_function_by_name(graph_def, old_name)
  func_def_ptr.signature.name = new_name

  # Pass 2: Function calls in main agraph
  _rename_calls(graph_def.node)

  # Pass 3: Function calls from inside FunctionDefs
  for f in graph_def.library.function:
    _rename_calls(f.node_def)


def _renumber_function_name(old_name):
  """
  Given a possibly generated function name, strip off any generated
  numerical ID from the name and add a new generated ID that should
  be unique to this process.

  Args:
      old_name: Function name to transform

  Returns a transformed version of old_name
  """
  elems = old_name.split("_")
  if len(elems) > 1:
    prefix = "_".join(elems[:-1])
  else:
    prefix = old_name
  return "{}_{}".format(prefix, ops.uid())


def _copy_and_replace_graph(func, replacement_graph_def,
                            input_tensor_names, output_tensor_names):
  """
  Create a copy of a ConcreteFunction, replacing the original function's
  internal graph with the contents of the specified GraphDef.

  If the original function has a gradient, the gradient will be passed
  through to the new copy.

  Args:
  func: A ConcreteFunction object. Will be used as a donor of metadata
      about input and output arguments.
  replacement_graph_def: GraphDef describing the new function body.
      Will replace the internals of `func`'s FunctionDef
      For every input tensor of `func`, this graph must contain a
      Placeholder op with the same name.
      For every output tensor of `func`, this graph must contain a
      tensor with the same name.
  input_tensor_names: List of the names of the placeholder tensors in
      `replacement_graph_def` that correspond to the function arguments
  output_tensor_names: List of the names of output tensors of
      `replacement_graph_def` that correspond to the function's return
      values


  Returns a copy of `func` with its internal `FuncGraph` replaced with
  the contents of `replacement_graph_def`.
  """
  if not isinstance(func, function.ConcreteFunction):
    raise ValueError("This function only works on ConcreteFunctions; got "
                     "{}".format(func))

  # Use the GraphDef of the original function as a metadata donor.
  new_func_graph_def = graph_pb2.GraphDef()
  new_func_graph_def.CopyFrom(func.graph.as_graph_def())

  # The function's graph may contain auxiliary functions for argument
  # translation and gradient computation. Find the core function.
  first_output_node_name = func.outputs[0].op.name
  core_func_name = _core_function_name(new_func_graph_def,
                                       first_output_node_name)
  func_def_ptr = _find_function_by_name(new_func_graph_def, core_func_name)

  # Generate a new replacement FunctionDef.
  replacement_graph = ops.Graph()
  with replacement_graph.as_default():
    importer.import_graph_def(replacement_graph_def, name="")
  replacement_function_def = graph_to_function_def.graph_to_function_def(
      replacement_graph, replacement_graph.get_operations(),
      [replacement_graph.get_tensor_by_name(n) for n in input_tensor_names],
      [replacement_graph.get_tensor_by_name(n) for n in output_tensor_names],
      [a.name for a in func_def_ptr.signature.output_arg]
  )
  replacement_function_def.signature.name = core_func_name

  # Patch the function library of the rewritten GraphDef
  func_def_ptr.CopyFrom(replacement_function_def)

  # Rename all functions in the patch graph so that it can be loaded
  # into an EagerContext that may have a copy of the original functions.
  orig_names = list([f.signature.name
                     for f in new_func_graph_def.library.function])
  for n in orig_names:
    _rename_function(new_func_graph_def, n, _renumber_function_name(n))

  # Wrap the GraphDef in a FuncGraph, then wrap that in a ConcreteFunction
  rewritten_func_graph = func_graph.FuncGraph(func.graph.name)
  with rewritten_func_graph.as_default():
    importer.import_graph_def(new_func_graph_def, name="")

  rewritten_func_graph.inputs = _copy_and_convert_tensors(
      func.graph.inputs, func.graph, rewritten_func_graph)
  rewritten_func_graph.outputs = _copy_and_convert_tensors(
      func.graph.outputs, func.graph, rewritten_func_graph)
  rewritten_func_graph.structured_outputs = _copy_and_convert_tensors(
      func.graph.structured_outputs, func.graph, rewritten_func_graph)
  rewritten_func = function.ConcreteFunction(rewritten_func_graph)

  # Copy Python-only metadata from the original ConcreteFunction.
  rewritten_func._arg_keywords = func._arg_keywords  # pylint: disable=protected-access
  rewritten_func._num_positional_args = func._num_positional_args  # pylint: disable=protected-access

  # Make the returned function callable.
  if context.executing_eagerly():
    for f in new_func_graph_def.library.function:
      context.context().add_function_def(f)
  rewritten_func.add_to_graph()
  return rewritten_func


def TransformSavedModel(input_dir, output_dir, transforms,
                        signature_name="serving_default",
                        tags=None):
  """Front-end for the Graph Transform Tool that reads and writes SavedModels.

  This front-end uses TensorFlow V2 APIs to read and write SavedModels,
  so it requires that [eager execution](
  https://www.tensorflow.org/guide/eager) be enabled.

  Limitations of the current implementation of this function:
  * The output SavedModel will be in the format of the current version of 
    TensorFlow and may not work with older versions, even if the input
    SavedModel was compatible with those versions.
  * The current implementation can only extract a single concrete function
    from the input SavedModel. The `tags` and `signature_name` arguments
    identify the function used.
  * The current implementation only stores the output function as a
    signature, not an attribute of a Trackable object. If the original
    SavedModel had a Trackable with named attributes, you can regenerate these
    attributes manually by loading and re-saving the output model.
    For example:
    ```python
    trackable = tf.saved_model.load(result_dir)
    trackable.my_attr = trackable.signatures["serving_default"]
    tf.saved_model.save(trackable, result_dir_2)
    ```
  * The current implementation does not support SavedModels that contain
    variables. Convert all variables to constants before calling this function.
  * The current implementation does not support functions with control outputs.

  Args:
    input_dir: Root directory of the input SavedModel. This directory is the one 
      that contains a file called `saved_model.pb` or `saved_model.pbtext`.
    output_dir: Directory at which to write the results of the transformation
    transforms: List of strings containing transform names and parameters. See
      the "Transform Reference" section of
      https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
      for a list of all available transforms and their parameters.
    signature_name: Name under which the target function has been stored
      inside the SavedMode; see the `signatures` argument of
      `tf.saved_model.save()` for more information.
      Note that the default value of "serving_default" is the key that
      TensorFlow uses when you save a `Trackable` with one function and do
      not specify a
      signature.
    tags: Optional argument specifying a set of string tags that tell which
      MetaGraph to load from the input SavedModel.
      If this parameter is set to None and there are multiple MetaGraphs, this
      function will raise an error.
  """
  if not loader_impl.contains_saved_model(input_dir):
    raise ValueError("Input directory '{}' does not appear to be the root "
                     "directory of a SavedModel. The directory should contain "
                     "a file called `saved_model.pb` or `saved_model.pbtext`"
                     ".".format(input_dir))
  if loader_impl.contains_saved_model(output_dir):
    raise ValueError("Output directory '{}' appears to contain an existing "
                     "SavedModel".format(output_dir))

  trackable = load.load(input_dir, tags)
  if hasattr(trackable, "variables") and len(trackable.variables) > 0:
    raise NotImplementedError("SavedModel at '{}' contains variables, "
                              "and this function does not currently support "
                              "SavedModels that contain variables"
                              "".format(input_dir))
  if signature_name not in trackable.signatures:
    raise ValueError("The SavedModel at {} does not contain any signatures "
                     "with the key '{}' under tag set '{}'."
                     "".format(input_dir, signature_name, tags))
  signature = trackable.signatures[signature_name]

  # With the current version of load.load(), models in v1 format end up only
  # partially converted to functions. Send the model on a round-trip through the
  # save-load cycle to ensure it's a function all the way down to the C++ layer.
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)
  tmp_dir = tempfile.mkdtemp(dir=output_dir)
  save.save(tracking.AutoTrackable(), tmp_dir,
            signatures={"serving_default": signature})
  clean_trackable = load.load(tmp_dir)
  shutil.rmtree(tmp_dir)

  func = clean_trackable.signatures["serving_default"]

  if not isinstance(func, function.ConcreteFunction):
    raise ValueError("Entry point at signature '{}' is not a concrete function"
                     "".format(signature_name))

  # Now extract the graph from the function. The graph is stored in a protobuf
  # nested a few levels deep, in a slightly obfuscated format.
  func_graph_def = func.graph.as_graph_def()
  core_func_name = _core_function_name(func_graph_def,
                                       func.outputs[0].op.name)
  func_def = _find_function_by_name(func_graph_def, core_func_name)
  graphdef_before, name_map = function_def_to_graph.function_def_to_graph_def(
      func_def)  # Deobfuscate node input strings
  input_names = [a.name for a in func_def.signature.input_arg]
  output_names = [name_map[func_def.ret[a.name]]
                  for a in func_def.signature.output_arg]

  # Invoke the Python front end to the Graph Transform Tool.
  graphdef_after = TransformGraph(graphdef_before,
                                  inputs=[t.name for t in signature.inputs],
                                  outputs=[t.name for t in signature.outputs],
                                  transforms=transforms)

  # Turn the resulting graph back into a function so we can save in V2 format.
  func_after = _copy_and_replace_graph(func, graphdef_after,
                                       [n + ":0" for n in input_names],
                                       output_names)
  save.save(tracking.AutoTrackable(), output_dir,
            signatures={signature_name: func_after})
