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
# ==============================================================================
"""Tools for deserializing `Function`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from tensorflow.core.framework import function_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import function_def_to_graph as function_def_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest


def _is_tensor(t):
  return isinstance(t, (ops.Tensor, resource_variable_ops.ResourceVariable))


def _inputs_compatible(args, stored_inputs):
  """Checks whether function arguments are compatible with parameters."""
  # TODO(vbardiovsky): The compatibility check should be about the signature,
  # not the flattened version of it.
  if len(args) != len(stored_inputs):
    return False
  for a, b in zip(args, stored_inputs):
    if _is_tensor(a):
      if not isinstance(b, tensor_spec.TensorSpec):
        return False
      if a.dtype != b.dtype or not b.shape.is_compatible_with(a.shape):
        return False
    else:
      if a != b:
        return False
  return True


def recreate_function(saved_function, concrete_functions):
  """Creates a `Function` from a `SavedFunction`.

  Args:
    saved_function: `SavedFunction` proto.
    concrete_functions: map from function name to `ConcreteFunction`.

  Returns:
    A `Function`.
  """
  # TODO(andresp): Construct a `Function` with the cache populated
  # instead of creating a new `Function` backed by a Python layer to
  # glue things together. Current approach is nesting functions deeper for each
  # serialization cycle.

  coder = nested_structure_coder.StructureCoder()
  function_spec_tuple = coder.decode_proto(
      saved_function.function_spec_tuple)
  function_spec = function_lib.FunctionSpec.from_tuple(function_spec_tuple)

  # TODO(mdan): We may enable autograph once exceptions are supported.
  @def_function.function(autograph=False)
  def restored_function(*args, **kwargs):
    """Calls a restored function."""
    # TODO(allenl): Functions saved with input_signatures should revive with
    # input_signatures.
    for concrete_function in saved_function.concrete_function:
      function_obj = concrete_functions[concrete_function.name]
      canonicalized_original_inputs = coder.decode_proto(
          concrete_function.canonicalized_input)

      try:
        can_args, can_kwargs = function_spec.canonicalize_function_inputs(
            *args, **kwargs)
        if can_kwargs:
          # TODO(vbardiovsky): Enable this along with the structured input and
          # structured output.
          raise ValueError(
              "Received keywords arguments that could not be bound: %s" %
              kwargs)
      except ValueError:
        continue

      canonicalized_inputs = nest.flatten(can_args)

      if _inputs_compatible(canonicalized_inputs,
                            canonicalized_original_inputs):
        filtered_inputs = [t for t in canonicalized_inputs if _is_tensor(t)]
        flattened_outputs = function_obj._call_flat(filtered_inputs)  # pylint: disable=protected-access
        # TODO(vbardiovsky): Rebuild output structure.
        single_output, = flattened_outputs
        return single_output

    raise AssertionError(
        "Could not find matching function to call for arguments: %s" % (args,))
  return restored_function


def load_function_def_library(library):
  """Load a set of functions as concrete functions without captured inputs.

  Functions names are manipulated during load such that they do not overlap
  with previously created ones.

  Args:
    library: FunctionDefLibrary proto message.

  Returns:
    Map of original function names in the library to instances of
    `ConcreteFunction` without captured inputs.

  Raises:
    ValueError: if functions dependencies have a cycle.
  """
  # TODO(andresp): Look into restoring gradient function information.
  functions = {}
  name_mapping = {}
  # Note: Use a new graph to allow function_def_to_graph to help validating
  # that the functions are loaded correctly. This is not possible to do
  # just in eager mode as there is no python API to find if a function has
  # been registered in eager. Note also that despite this the created
  # func_graphs can still be used in eager or in other graphs.
  with ops.Graph().as_default() as import_graph:
    for fdef in _sort_function_defs(library):
      copy = _fix_fdef(fdef, name_mapping)

      func_graph = function_def_lib.function_def_to_graph(copy)
      func = function_lib.ConcreteFunction(func_graph)
      func.add_to_graph(import_graph)

      name_mapping[fdef.signature.name] = func.name
      functions[fdef.signature.name] = func
  return functions


def _sort_function_defs(library):
  """Return a topologic sort of FunctionDefs in a library."""
  edges = collections.defaultdict(list)
  in_count = collections.defaultdict(lambda: 0)

  for fdef in library.function:
    for dep in _list_function_deps(fdef):
      edges[dep].append(fdef.signature.name)
      in_count[fdef.signature.name] += 1

  ready = [
      fdef.signature.name
      for fdef in library.function
      if in_count[fdef.signature.name] == 0
  ]
  output = []
  while ready:
    node = ready.pop()
    output.append(node)
    for dest in edges[node]:
      in_count[dest] -= 1
      if not in_count[dest]:
        ready.append(dest)

  if len(output) != len(library.function):
    loaded = set([x.signature.name for x in output])
    failed_to_resolve = sorted(set(in_count.keys()) - loaded)
    raise ValueError("There is a cyclic-dependency between functions. ",
                     "Could not resolve %r." % (failed_to_resolve,))

  reverse = {fdef.signature.name: fdef for fdef in library.function}
  return [reverse[x] for x in output]


def _fix_fdef(orig_fdef, name_map):
  fdef = function_pb2.FunctionDef()
  fdef.CopyFrom(orig_fdef)
  fdef.signature.name = _clean_function_name(fdef.signature.name)
  for node_def in fdef.node_def:
    for _, attr_value in node_def.attr.items():
      if attr_value.func.name:
        attr_value.func.name = name_map[attr_value.func.name]
  return fdef


def _list_function_deps(fdef):
  # TODO(andresp): Recurse into list attributes and into NameAttrList attrs both
  # when listing deps and when fixing them. `function_def_to_graph` also
  # requires fixes.
  deps = set()
  for node_def in fdef.node_def:
    for _, attr_value in node_def.attr.items():
      if attr_value.WhichOneof("value") == "func":
        deps.add(attr_value.func.name)
  return deps


def _clean_function_name(name):
  """Vanity function to keep the function names comprehensible."""
  # Note: each time a function is wrapped into `function_lib.ConcreteFunction`
  # its name becomes "__inference_<orig>_xyz".
  match = re.search(r"^__inference_(.*)_\d+$", name)
  if match:
    return match.group(1)
  else:
    return name
