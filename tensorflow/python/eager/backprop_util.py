# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Shared utilities related to backprop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import handle_data_util


def _DTypeFromTensor(tensor):
  """Extract either `tensor.dtype` or the unanimous sub-type of a variant."""
  dtype = tensor.dtype
  if dtype.base_dtype == dtypes.variant:
    # If we know statically that the data a variant points to is non-trainable
    # then the variant itself is non-trainable.
    if isinstance(tensor, ops.EagerTensor):
      handle_data = tensor._handle_data  # pylint: disable=protected-access
    else:
      handle_data = handle_data_util.get_resource_handle_data(tensor)
    if (handle_data is not None
        and handle_data.is_set
        and handle_data.shape_and_type):
      first_type = handle_data.shape_and_type[0].dtype
      # Some variants have statically unknown dtypes; we can't make inferences
      # about trainability, so we conservatively assume they're trainable
      # (which may waste memory passing zeros around, but will be correct).
      if (first_type != types_pb2.DT_INVALID
          and all(shape_and_type.dtype == first_type
                  for shape_and_type in handle_data.shape_and_type)):
        return first_type
  return dtype


def IsTrainable(tensor_or_dtype):
  """Determines whether a tensor or dtype supports infinitesimal changes."""
  if tensor_util.is_tf_type(tensor_or_dtype):
    dtype = _DTypeFromTensor(tensor_or_dtype)
  else:
    dtype = tensor_or_dtype
  dtype = dtypes.as_dtype(dtype)
  return dtype.base_dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                              dtypes.complex64, dtypes.complex128,
                              dtypes.resource, dtypes.variant, dtypes.bfloat16)
