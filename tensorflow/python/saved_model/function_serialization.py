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
"""Tools for serializing PolymorphicFunctions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as defun_lib
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import saved_object_graph_pb2


def _serialize_polymorphic_function(polymorphic_function, node_ids):
  """Build a SavedPolymorphicProto."""
  proto = saved_object_graph_pb2.SavedPolymorphicFunction()
  for concrete_function in list_all_concrete_functions(polymorphic_function):
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
    function_proto.bound_inputs.extend(bound_inputs)
  return proto


def list_all_concrete_functions(polymorphic_function):
  """Given a polymorphic function, returns all of its concrete functions."""
  input_signature = polymorphic_function._input_signature  # pylint: disable=protected-access
  if input_signature is not None:
    polymorphic_function.get_concrete_function()
  concrete_functions = []
  for signature in polymorphic_function._cached_input_signatures:  # pylint: disable=protected-access
    if any(isinstance(arg, defun_lib.UnknownArgument) for arg in signature):
      continue
    concrete_function = polymorphic_function.get_concrete_function(*signature)
    concrete_functions.append(concrete_function)
  return concrete_functions


def list_all_polymorphic_functions(checkpointable_object):
  """Given a checkpointable object, returns all of its polymorphic functions."""
  polymorphic_functions = dict()
  for attribute_name in dir(checkpointable_object):
    try:
      attribute_value = getattr(checkpointable_object, attribute_name, None)
    except:  # pylint: disable=bare-except
      # We really don't want to throw an exception just because some object's
      # attribute accessor is broken.
      attribute_value = None
    # TODO(allenl): Consider de-duplicating functions which are referenced
    # from multiple attributes.
    if isinstance(attribute_value, def_function.PolymorphicFunction):
      polymorphic_functions[attribute_name] = attribute_value
  return polymorphic_functions


def add_polymorphic_functions_to_object_graph_proto(checkpointable_objects,
                                                    saved_object_graph,
                                                    node_ids):
  """Finds PolymorphicFunctions attached to objects and saves them."""
  existing_objects = list(zip(checkpointable_objects, saved_object_graph.nodes))
  for obj, obj_proto in existing_objects:
    for name, polymorphic_function in list_all_polymorphic_functions(
        obj).items():
      function_node_id = len(saved_object_graph.nodes)
      function_node = saved_object_graph.nodes.add()
      function_node.function.CopyFrom(
          _serialize_polymorphic_function(polymorphic_function, node_ids))
      reference = obj_proto.children.add()
      reference.node_id = function_node_id
      reference.local_name = name
