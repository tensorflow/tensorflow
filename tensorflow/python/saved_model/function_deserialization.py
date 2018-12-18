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
from tensorflow.python.util import nest


def _inputs_compatible(args, function):
  """Check if args are compatible with a concrete function."""
  # TODO(vbardiovsky): The compatibility check should be about the signature,
  # not the flattened version of it.
  flattened_inputs = nest.flatten(args)
  expected_input_count = len(function.inputs) - len(function.captured_inputs)
  if len(flattened_inputs) != expected_input_count:
    return False
  for a, b in zip(flattened_inputs, function.inputs):
    if a.dtype != b.dtype or not b.shape.is_compatible_with(a.shape):
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
  # TODO(mdan): We may enable autograph once exceptions are supported.
  @def_function.function(autograph=False)
  def restored_function(*args):
    """Calls a restored function."""
    # TODO(allenl): Functions saved with input_signatures should revive with
    # input_signatures.
    for monomorphic_function in saved_polymorphic_function.monomorphic_function:
      function_obj = functions[monomorphic_function.concrete_function]
      if _inputs_compatible(args, function_obj):
        flattened_inputs = nest.flatten(args)
        flattened_outputs = function_obj._call_flat(flattened_inputs)  # pylint: disable=protected-access
        # TODO(vbardiovsky): rebuild output structure.
        single_output, = flattened_outputs
        return single_output

    raise AssertionError(
        "Could not find matching function to call for arguments: %s" % (args,))
  return restored_function
