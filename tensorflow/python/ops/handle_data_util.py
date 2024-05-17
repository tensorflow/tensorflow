# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Decorator to overrides the gradient for a function."""

from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import compat


def get_resource_handle_data(graph_op):
  assert (isinstance(graph_op, core.Symbol)
          and not isinstance(graph_op, core.Value))

  with graph_op.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
    handle_data = pywrap_tf_session.GetHandleShapeAndType(
        c_graph, graph_op._as_tf_output())  # pylint: disable=protected-access

  return cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData.FromString(
      compat.as_bytes(handle_data))


def get_handle_data(source_t):
  """Obtains HandleData from a tensor."""
  if isinstance(source_t, core.Value):
    return source_t._handle_data  # pylint: disable=protected-access
  return get_resource_handle_data(source_t)


def copy_handle_data(source_t, target_t):
  """Copies HandleData for variant and resource type tensors if available.

  The CppShapeInferenceResult::HandleData proto contains information about the
  shapes and types of the element tensors of resource/variant type tensors.
  We need to copy this across function boundaries, i.e., when capturing a
  placeholder or when returning a function tensor as output. If we don't do this
  the element tensors will have unknown shapes, e.g., if a TensorList variant
  tensor is captured as a placeholder, elements popped from that list would have
  unknown shape.

  Args:
    source_t: The tensor to copy HandleData from.
    target_t: The tensor to copy HandleData to.
  """
  if (target_t.dtype == dtypes.resource or
      target_t.dtype == dtypes.variant):
    handle_data = get_handle_data(source_t)
    set_handle_data(target_t, handle_data)


def set_handle_data(target_t, handle_data):
  """Sets handle data on the giver tensor."""
  if (
      handle_data is None
      or not handle_data.is_set
      or not handle_data.shape_and_type
  ):
    return

  # pylint: disable=protected-access
  if isinstance(target_t, core.Value):
    target_t._handle_data = handle_data
    return
  with target_t.graph._c_graph.get() as c_graph:
    pywrap_tf_session.SetHandleShapeAndType(c_graph, target_t._as_tf_output(),
                                            handle_data.SerializeToString())
  # pylint: enable=protected-access


def create_handle_data(shape, dtype):
  handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
  handle_data.is_set = True
  handle_data.shape_and_type.append(
      cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
          shape=shape.as_proto(), dtype=dtype.as_datatype_enum))
  return handle_data
