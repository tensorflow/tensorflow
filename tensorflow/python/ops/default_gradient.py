# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for computing default gradients."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import resource_variable_ops


def get_zeros_dtype(t):
  """Return the dtype for the default gradient for a Tensor."""
  if t.dtype == dtypes.resource:
    handle_data = resource_variable_ops.get_eager_safe_handle_data(t)
    if (handle_data is None or not handle_data.is_set or
        len(handle_data.shape_and_type) != 1):
      # TODO(srbs): Ideally we should raise an error here but returning float32
      # for backwards compatibility.
      return dtypes.float32
    else:
      return handle_data.shape_and_type[0].dtype
  return t.dtype
