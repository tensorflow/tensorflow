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


def _serialize_function_spec(function_spec, coder):
  """Serialize a FunctionSpec object into its proto representation."""
  proto = saved_object_graph_pb2.FunctionSpec()
  proto.fullargspec.CopyFrom(coder.encode_structure(function_spec.fullargspec))
  proto.is_method = function_spec.is_method
  proto.args_to_prepend.CopyFrom(
      coder.encode_structure(function_spec.args_to_prepend))
  proto.kwargs_to_include.CopyFrom(
      coder.encode_structure(function_spec.kwargs_to_include))
  proto.input_signature.CopyFrom(
      coder.encode_structure(function_spec.input_signature))
  return proto


def serialize_function(function, node_ids):
  """Build a SavedFunction proto."""
  coder = nested_structure_coder.StructureCoder()
  proto = saved_object_graph_pb2.SavedFunction()

  function_spec_proto = _serialize_function_spec(
      function.function_spec, coder)
  proto.function_spec.CopyFrom(function_spec_proto)
  all_concrete_functions = \
      function._list_all_concrete_functions_for_serialization()  # pylint: disable=protected-access
  for signature, concrete_function in all_concrete_functions:
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
    function_proto = proto.concrete_function.add()
    function_proto.name = concrete_function.name
    function_proto.canonicalized_input_signature.CopyFrom(
        coder.encode_structure(signature))
    structured_outputs = defun_lib.convert_structure_to_signature(
        concrete_function.structured_outputs)
    function_proto.output_signature.CopyFrom(
        coder.encode_structure(structured_outputs))
    function_proto.bound_inputs.extend(bound_inputs)
  return proto
