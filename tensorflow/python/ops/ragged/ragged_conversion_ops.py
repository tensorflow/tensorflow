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
"""Ops to convert between RaggedTensors and other tensor types."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.ragged import ragged_tensor


def from_tensor(tensor, lengths=None, padding=None, ragged_rank=1, name=None):
  if ragged_tensor.is_ragged(tensor):
    return tensor
  else:
    return ragged_tensor.RaggedTensor.from_tensor(tensor, lengths, padding,
                                                  ragged_rank, name)


def to_tensor(rt_input, default_value=None, name=None):
  if ragged_tensor.is_ragged(rt_input):
    return rt_input.to_tensor(default_value, name)
  else:
    return rt_input


def to_sparse(rt_input, name=None):
  return rt_input.to_sparse(name)


def from_sparse(st_input, name=None):
  return ragged_tensor.RaggedTensor.from_sparse(st_input, name)
