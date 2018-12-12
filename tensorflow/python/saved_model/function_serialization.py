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
from tensorflow.python.saved_model import saved_object_graph_pb2


def _serialize_polymorphic_function(function):
  """Represents a PolymorphicFunction in a SavedModel.

  Adds `function`'s concrete functions to the current graph.

  Args:
    function: A `PolymorphicFunction` to serialize.

  Returns:
    An unserialized `SavedPolymorphicFunction` protocol buffer object.
  """
  monomorphic_functions = []
  for signature in function._cached_input_signatures:  # pylint: disable=protected-access
    if any(isinstance(arg, defun_lib.UnknownArgument) for arg in signature):
      continue
    concrete_function = function.get_concrete_function(*signature)
    concrete_function.add_to_graph()
    monomorphic_functions.append(
        saved_object_graph_pb2.SavedMonomorphicFunction(
            concrete_function=concrete_function.name))
  return saved_object_graph_pb2.SavedPolymorphicFunction(
      monomorphic_function=monomorphic_functions)


def add_polymorphic_functions_to_object_graph_proto(
    checkpointable_objects, saved_object_graph):
  """Finds PolymorphicFunctions attached to objects and saves them."""
  existing_objects = list(zip(checkpointable_objects, saved_object_graph.nodes))
  for obj, obj_proto in existing_objects:
    for attribute_name in dir(obj):
      try:
        attribute_value = getattr(obj, attribute_name, None)
      except:  # pylint: disable=bare-except
        # We really don't want to throw an exception just because some object's
        # attribute accessor is broken.
        attribute_value = None
      # TODO(allenl): Consider de-duplicating functions which are referenced
      # from multiple attributes.
      if isinstance(attribute_value, def_function.PolymorphicFunction):
        function_node_id = len(saved_object_graph.nodes)
        function_node = saved_object_graph.nodes.add()
        function_node.function.CopyFrom(
            _serialize_polymorphic_function(attribute_value))
        reference = obj_proto.children.add()
        reference.node_id = function_node_id
        reference.local_name = attribute_name
