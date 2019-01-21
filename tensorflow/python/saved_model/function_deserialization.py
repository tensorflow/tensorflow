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
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import compat
from tensorflow.python.util import nest


def _is_tensor(t):
  return isinstance(t, (ops.Tensor, resource_variable_ops.ResourceVariable))


def _inputs_compatible(args, stored_inputs):
  """Checks whether function arguments are compatible with parameters."""
  if len(args) != len(stored_inputs):
    return False

  for arg, stored_input in zip(args, stored_inputs):
    if not function_lib.is_same_structure(arg, stored_input):
      return False

    flattened_arg = nest.flatten(arg)
    flattened_stored_input = nest.flatten(stored_input)

    for a, b in zip(flattened_arg, flattened_stored_input):
      if _is_tensor(a):
        if not isinstance(b, tensor_spec.TensorSpec):
          return False
        if a.dtype != b.dtype or not b.shape.is_compatible_with(a.shape):
          return False
      else:
        if a != b:
          return False
  return True


def _deserialize_function_spec(function_spec_proto, coder):
  """Deserialize a FunctionSpec object from its proto representation."""
  fullargspec = coder.decode_proto(function_spec_proto.fullargspec)
  is_method = function_spec_proto.is_method
  args_to_prepend = coder.decode_proto(function_spec_proto.args_to_prepend)
  kwargs_to_include = coder.decode_proto(function_spec_proto.kwargs_to_include)
  input_signature = coder.decode_proto(function_spec_proto.input_signature)
  return function_lib.FunctionSpec(fullargspec, is_method, args_to_prepend,
                                   kwargs_to_include, input_signature)


def recreate_concrete_function(saved_concrete_function, concrete_functions):
  """Recreates a user-facing concrete function."""
  coder = nested_structure_coder.StructureCoder()

  concrete_function = concrete_functions[saved_concrete_function.name]
  input_signature = coder.decode_proto(
      saved_concrete_function.canonicalized_input_signature)
  input_signature_args, input_signature_kwargs = input_signature
  if input_signature_kwargs:
    raise ValueError("Restoring concrete function with non-empty kwargs (%s)." %
                     input_signature_kwargs)

  # pylint: disable=protected-access
  # Set metadata required for the concrete function to accept keyword and
  # positional arguments in __call__. Normally this is set in
  # get_concrete_function.
  concrete_function._arg_keywords = [spec.name for spec in input_signature_args]
  # TODO(allenl): Should we preserve the number of allowed positional arguments?
  concrete_function._num_positional_args = len(input_signature_args)
  # pylint: enable=protected-access
  concrete_function.add_to_graph()
  return concrete_function


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
  function_spec = _deserialize_function_spec(saved_function.function_spec,
                                             coder)

  # TODO(mdan): We may enable autograph once exceptions are supported.
  @def_function.function(autograph=False)
  def restored_function(*args, **kwargs):
    """Calls a restored function."""
    # TODO(allenl): Functions saved with input_signatures should revive with
    # input_signatures.
    for concrete_function in saved_function.concrete_function:
      function_obj = concrete_functions[concrete_function.name]
      canonicalized_original_inputs = coder.decode_proto(
          concrete_function.canonicalized_input_signature)

      try:
        canonicalized_inputs = function_spec.canonicalize_function_inputs(
            *args, **kwargs)
      except ValueError:
        continue

      if _inputs_compatible(canonicalized_inputs,
                            canonicalized_original_inputs):
        flattened_inputs = nest.flatten(canonicalized_inputs)
        filtered_inputs = [t for t in flattened_inputs if _is_tensor(t)]
        return function_obj._call_flat(filtered_inputs)  # pylint: disable=protected-access

    raise AssertionError(
        "Could not find matching function to call for args %r and kwargs %r" %
        (args, kwargs))

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
  functions = {}

  for fdef in _sort_function_defs(library):
    copy = _fix_fdef(fdef, functions)

    func_graph = function_def_lib.function_def_to_graph(copy)
    func = function_lib.ConcreteFunction(func_graph)
    func.add_to_graph()

    functions[fdef.signature.name] = func

    # Also register the gradients in the current root context.
    with ops.init_scope():
      func._register_gradient()  # pylint: disable=protected-access

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


def _fix_fdef(orig_fdef, functions):
  """Fixes a FunctionDef proto to be loaded in current context.

  In particular, when loading a function library into an eager context, one
  must rename the functions to avoid conflicts with existent functions.

  Args:
    orig_fdef: FunctionDef proto to fix. It is not modified.
    functions: map from function name to a ConcreteFunction instance.

  Returns:
    A fixed copy of the original FunctionDef.
  """
  fdef = function_pb2.FunctionDef()
  fdef.CopyFrom(orig_fdef)
  for node_def in fdef.node_def:
    if "_gradient_op_type" in node_def.attr:
      if node_def.op in ["StatefulPartitionedCall", "PartitionedCall"]:
        # TODO(andresp): This code assumes that the gradient registered for this
        # function call is the default gradient for the function and not a
        # custom one.
        fname = node_def.attr["f"].func.name
        node_def.attr["_gradient_op_type"].s = compat.as_bytes(
            functions[fname]._gradient_name)  # pylint: disable=protected-access
      else:
        logging.warning("Importing a function (%s) with ops with custom "
                        "gradients. Will likely fail if a gradient is "
                        "requested.", fdef.signature.name)
    for _, attr_value in node_def.attr.items():
      if attr_value.func.name:
        attr_value.func.name = functions[attr_value.func.name].name

  fdef.signature.name = _clean_function_name(fdef.signature.name)
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
