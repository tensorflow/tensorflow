# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""A typed list in Python."""

from tensorflow.python.framework import tensor
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops


def dynamic_list_append(target, element):
  """Converts a list append call inline."""
  if isinstance(target, tensor_array_ops.TensorArray):
    return target.write(target.size(), element)
  # TODO(mdan): What's the right way to check this?
  # TODO(mdan): We may not need this branch.
  # It may be possible to use TensorList alone if the loop body will not
  # require wrapping it, although we'd have to think about an autoboxing
  # mechanism for lists received as parameter.
  if isinstance(target, tensor.Tensor):
    return list_ops.tensor_list_push_back(target, element)

  # Python targets (including TensorList): fallback to their original append.
  target.append(element)
  return target


class TensorList(object):
  """Tensor list wrapper API-compatible with Python built-in list."""

  def __init__(self, shape, dtype):
    self.dtype = dtype
    self.shape = shape
    self.clear()

  def append(self, value):
    self.list_ = list_ops.tensor_list_push_back(self.list_, value)

  def pop(self):
    self.list_, value = list_ops.tensor_list_pop_back(self.list_, self.dtype)
    return value

  def clear(self):
    self.list_ = list_ops.empty_tensor_list(self.shape, self.dtype)

  def count(self):
    return list_ops.tensor_list_length(self.list_)

  def __getitem__(self, key):
    return list_ops.tensor_list_get_item(self.list_, key, self.dtype)

  def __setitem__(self, key, value):
    self.list_ = list_ops.tensor_list_set_item(self.list_, key, value)
