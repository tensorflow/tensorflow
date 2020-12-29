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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops


get_resource_handle_data = ops.get_resource_handle_data


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
    if isinstance(source_t, ops.EagerTensor):
      handle_data = source_t._handle_data  # pylint: disable=protected-access
    else:
      handle_data = get_resource_handle_data(source_t)
    if (handle_data is not None
        and handle_data.is_set
        and handle_data.shape_and_type):
      set_handle_data(target_t, handle_data)


def set_handle_data(target_t, handle_data):
  # pylint: disable=protected-access
  if isinstance(target_t, ops.EagerTensor):
    target_t._handle_data = handle_data
    return
  pywrap_tf_session.SetHandleShapeAndType(target_t.graph._c_graph,
                                          target_t._as_tf_output(),
                                          handle_data.SerializeToString())
  # pylint: enable=protected-access
