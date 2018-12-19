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
"""Tools for deserializing PolymorphicFunctions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as function_lib
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


def recreate_polymorphic_function(
    saved_polymorphic_function, functions):
  """Creates a PolymorphicFunction from a SavedPolymorphicFunction.

  Args:
    saved_polymorphic_function: SavedPolymorphicFunction proto.
    functions: map from function name to Function.

  Returns:
    A PolymorphicFunction.
  """
  # TODO(andresp): Construct a PolymorphicFunction with the cache populated
  # instead of creating a new PolymorphicFunction backed by a Python layer to
  # glue things together. Current approach is nesting functions deeper for each
  # serialization cycle.

  coder = nested_structure_coder.StructureCoder()
  function_spec_tuple = coder.decode_proto(
      saved_polymorphic_function.function_spec_tuple)
  function_spec = function_lib.FunctionSpec.from_tuple(function_spec_tuple)

  # TODO(mdan): We may enable autograph once exceptions are supported.
  @def_function.function(autograph=False)
  def restored_function(*args, **kwargs):
    """Calls a restored function."""
    # TODO(allenl): Functions saved with input_signatures should revive with
    # input_signatures.
    for monomorphic_function in saved_polymorphic_function.monomorphic_function:
      function_obj = functions[monomorphic_function.concrete_function]
      canonicalized_original_inputs = coder.decode_proto(
          monomorphic_function.canonicalized_input)

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
