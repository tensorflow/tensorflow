# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Data structures and utilities for checkpoint sharding."""

import dataclasses
from typing import Callable, Mapping, Sequence

from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.training.saving import saveable_object


TensorSlice = Mapping[tensor_spec.TensorSpec, tensor_lib.Tensor]
TensorSliceDict = Mapping[str, TensorSlice]


@dataclasses.dataclass(frozen=True)
class ShardableTensor:
  """Tensor wrapper containing data necessary for sharding."""
  _tensor_save_spec: saveable_object.SaveSpec
  tensor: tensor_lib.Tensor
  dtype: dtypes.DType
  device: device_lib.DeviceSpec
  name: str
  shape: tensor_shape.TensorShape
  slice_spec: variables.Variable.SaveSliceInfo
  checkpoint_key: str
  trackable: base.Trackable

  def __hash__(self) -> int:
    return hash((self.name, self.dtype, str(self.device), self.checkpoint_key))


@dataclasses.dataclass(frozen=True)
class ShardingCallback:
  """Checkpoint sharding callback function, along with a text description."""
  callback: Callable[
      [Sequence[ShardableTensor], ...],
      Sequence[Mapping[
          str, Mapping[tensor_spec.TensorSpec, saveable_object.SaveSpec]]]]
  description: str

  def __hash__(self) -> int:
    if hasattr(self.callback, "__name__"):
      callback_hash = hash((self.callback.__module__, self.callback.__name__))
    else:
      callback_hash = id(self.callback)
    return hash((callback_hash, self.description))
