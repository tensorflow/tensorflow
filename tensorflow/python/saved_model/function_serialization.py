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

from tensorflow.python.eager import function as defun_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import saved_object_graph_pb2


def serialize_function(function, node_ids):
  """Build a SavedPolymorphicFunction proto."""
  coder = nested_structure_coder.StructureCoder()
  proto = saved_object_graph_pb2.SavedPolymorphicFunction()

  proto.function_spec_tuple.CopyFrom(
      coder.encode_structure(function.function_spec.as_tuple()))  # pylint: disable=protected-access
  for signature, concrete_function in list_all_concrete_functions(function):
    bound_inputs = []
    try:
      for capture in concrete_function.captured_inputs:
        bound_inputs.append(node_ids[capture])
    except KeyError:
      # TODO(andresp): Would it better to throw an exception?
      logging.warning(
          "Concrete function %s not added to object based saved model as it "
          "captures tensor %s which is unsupported or not reachable from root.",
          concrete_function.name, capture)
      continue
    function_proto = proto.monomorphic_function.add()
    function_proto.concrete_function = concrete_function.name
    function_proto.canonicalized_input.CopyFrom(
        coder.encode_structure(signature))
    function_proto.bound_inputs.extend(bound_inputs)
  return proto


def list_all_concrete_functions(function):
  """Given a `Function`, returns all of its concrete functions.

  Args:
    function: Instance of `Function`.

  Returns:
    A list of tuples in the form (signature, concrete_function), where
    `concrete_function` is an instance of `ConcreteFunction`.
  """
  input_signature = function._input_signature  # pylint: disable=protected-access
  if input_signature is not None:
    function.get_concrete_function()
  concrete_functions = []
  for signature in function._cached_input_signatures:  # pylint: disable=protected-access
    if any(isinstance(arg, defun_lib.UnknownArgument) for arg in signature):
      continue
    concrete_function = function.get_concrete_function(*signature)
    concrete_functions.append((signature, concrete_function))
  return concrete_functions
