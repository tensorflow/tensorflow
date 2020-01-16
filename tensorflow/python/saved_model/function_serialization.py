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
"""Tools for serializing `Function`s."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import saved_object_graph_pb2
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.saved_model import nested_structure_coder


def _serialize_function_spec(function_spec, coder):
  """Serialize a FunctionSpec object into its proto representation."""
  if function_spec.is_method and not function_spec.fullargspec.args:
    raise NotImplementedError(
        "Missing support to serialize a method function without a named "
        "'self' argument.")
  proto = saved_object_graph_pb2.FunctionSpec()

  # Intentionally skip encoding annotations of a function because function
  # annotations are mainly for optional type checking during development
  # and does not affect runtime behavior.
  # https://www.python.org/dev/peps/pep-3107/
  # https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
  proto.fullargspec.CopyFrom(
      coder.encode_structure(
          function_spec.fullargspec._replace(annotations={})))

  proto.is_method = function_spec.is_method
  proto.input_signature.CopyFrom(
      coder.encode_structure(function_spec.input_signature))
  return proto


def serialize_concrete_function(concrete_function, node_ids, coder):
  """Build a SavedConcreteFunction."""
  bound_inputs = []
  try:
    for capture in concrete_function.captured_inputs:
      bound_inputs.append(node_ids[capture])
  except KeyError:
    raise KeyError(
        "Failed to add concrete function %s to object based saved model as it "
        "captures tensor %s which is unsupported or not reachable from root. "
        "One reason could be that a stateful object or a variable that the "
        "function depends on is not assigned to an attribute of the serialized "
        "trackable object "
        "(see SaveTest.test_captures_unreachable_variable)."
        % (concrete_function.name, capture))
  concrete_function_proto = saved_object_graph_pb2.SavedConcreteFunction()
  structured_outputs = func_graph_module.convert_structure_to_signature(
      concrete_function.structured_outputs)
  concrete_function_proto.canonicalized_input_signature.CopyFrom(
      coder.encode_structure(concrete_function.structured_input_signature))
  concrete_function_proto.output_signature.CopyFrom(
      coder.encode_structure(structured_outputs))
  concrete_function_proto.bound_inputs.extend(bound_inputs)
  return concrete_function_proto


def serialize_bare_concrete_function(concrete_function):
  """Build a SavedBareConcreteFunction."""
  # pylint: disable=protected-access
  return saved_object_graph_pb2.SavedBareConcreteFunction(
      concrete_function_name=concrete_function.name,
      allowed_positional_arguments=concrete_function._num_positional_args,
      argument_keywords=concrete_function._arg_keywords)
  # pylint: enable=protected-access


def serialize_function(function):
  """Build a SavedFunction proto."""
  coder = nested_structure_coder.StructureCoder()
  proto = saved_object_graph_pb2.SavedFunction()

  function_spec_proto = _serialize_function_spec(function.function_spec, coder)
  proto.function_spec.CopyFrom(function_spec_proto)
  all_concrete_functions = \
      function._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
  for concrete_function in all_concrete_functions:
    proto.concrete_functions.append(concrete_function.name)
  return proto
