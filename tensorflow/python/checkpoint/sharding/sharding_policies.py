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
"""Checkpoint policies that determine how tensors are split into shards."""

from typing import Sequence

from tensorflow.python.checkpoint.sharding import sharding_util
from tensorflow.python.training.saving import saveable_object_util


class ShardByDevicePolicy(sharding_util.ShardingCallback):
  """Policy that splits tensors into shards based on their device spec."""

  def __init__(self):
    super().__init__(
        self._device_callback_impl,
        "Split tensors into shards based on their device spec.")

  def _device_callback_impl(
      self,
      shardable_tensors: Sequence[sharding_util.ShardableTensor]
  ) -> Sequence[sharding_util.TensorSlice]:
    """Callback to split tensors into shards based on their device spec.

    Args:
      shardable_tensors: A list of ShardableTensors.

    Returns:
      List of shard dicts containing tensors.
        [ {checkpoint key: {slice_spec: tensor} } ]
    """
    tensors_by_device = {}

    for shardable_tensor in shardable_tensors:
      tensor = shardable_tensor.tensor
      checkpoint_key = shardable_tensor.checkpoint_key
      slice_spec = shardable_tensor.slice_spec
      device = saveable_object_util.set_cpu0(shardable_tensor.device)

      (tensors_by_device
       .setdefault(device, {})
       .setdefault(checkpoint_key, {})[slice_spec]) = tensor

    return list(tensors_by_device.values())

  def __call__(
      self,
      shardable_tensors: Sequence[sharding_util.ShardableTensor]
  ) -> Sequence[sharding_util.TensorSlice]:
    return self.callback(shardable_tensors)  # pylint: disable=no-value-for-parameter
