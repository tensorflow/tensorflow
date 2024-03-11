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

import abc
import dataclasses
import inspect
from typing import Hashable, MutableMapping, Sequence

from tensorflow.python.framework import device as device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util import tf_export


TensorSlice = MutableMapping[tensor_spec.TensorSpec, tensor_lib.Tensor]
TensorSliceDict = MutableMapping[str, TensorSlice]


@tf_export.tf_export("train.experimental.ShardableTensor")
@dataclasses.dataclass(frozen=True)
class ShardableTensor:
  """Tensor wrapper containing data necessary for sharding.

  The tensor representation used as inputs to pre-made and custom
  `tf.train.experiemental.ShardingCallback`s, which can be specified using the
  `experimental_sharding_callback` option in `tf.train.CheckpointOptions`.

  """
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

  def __repr__(self) -> str:
    return (f"\n{self.__class__.__name__}:\n"
            f"  _tensor_save_spec={self._tensor_save_spec!r}\n"
            f"  tensor={self.tensor!r}\n"
            f"  dtype={self.dtype!r}\n"
            f"  device={self.device!r}\n"
            f"  name={self.name!r}\n"
            f"  shape={self.shape!r}\n"
            f"  slice_spec={self.slice_spec!r}\n"
            f"  checkpoint_key={self.checkpoint_key!r}\n"
            f"  trackable={self.trackable!r}")


@tf_export.tf_export("train.experimental.ShardingCallback")
class ShardingCallback(abc.ABC):
  """Checkpoint sharding callback function, along with a text description.

  A callback function wrapper that will be executed to determine how tensors
  will be split into shards when the saver writes the checkpoint shards to disk.

  The callback takes a list of `tf.train.experimental.ShardableTensor`s as input
  (as well as any kwargs defined by the `tf.train.experimental.ShardingCallback`
  subclass), and organizes the input tensors into different shards. Tensors are
  first organized by device task (see `tf.DeviceSpec`), then the callback will
  be called for each collection of tensors.

  There are a few restrictions to keep in mind when creating a custom callback:
    - Tensors must not be removed from the checkpoint.
    - Tensors must not be reshaped.
    - Tensor dtypes must not change.
    - Tensors within a shard must belong to the same task.
  Validation checks will be performed after the callback function is executed to
  ensure these restrictions aren't violated.

  Here's an example of a simple custom callback:

  ```
  # Place all tensors in a single shard.
  class AllInOnePolicy(tf.train.experimental.ShardingCallback):
    @property
    def description(self):
      return "Place all tensors in a single shard."

    def __call__(self, shardable_tensors):
      tensors = {}
      for shardable_tensor in shardable_tensors:
        tensor = shardable_tensor.tensor_save_spec.tensor
        checkpoint_key = shardable_tensor.checkpoint_key
        slice_spec = shardable_tensor.slice_spec

        tensors.set_default(checkpoint_key, {})[slice_spec] = tensor
      return [tensors]

  ckpt.save(
      "path",
      options=tf.train.CheckpointOptions(
          experimental_sharding_callback=AllInOnePolicy()))
  ```

  The `description` attribute is used to identify the callback and to aid
  debugging during saving and restoration.

  To take in kwargs, simply define the constructor and pass them in:

  ```
  class ParameterPolicy(tf.train.experimental.ShardingCallback):
    def __init__(self, custom_param):
      self.custom_param = custom_param
    ...

  ckpt.save(
      "path",
      options=tf.train.CheckpointOptions(
          experimental_sharding_callback=ParameterPolicy(custom_param=...)))
  ```

  """
  description: str

  @property
  @abc.abstractmethod
  def description(self) -> str:
    pass

  @abc.abstractmethod
  def __call__(
      self, shardable_tensors: Sequence[ShardableTensor]
  ) -> Sequence[TensorSliceDict]:
    pass

  def __hash__(self) -> int:
    hash_val = hash(self.description)
    # vars() only includes user-defined attributes.
    for attr_name, attr_val in vars(self).items():
      if not (inspect.ismethod(attr_val) or inspect.isfunction(attr_val)):
        hash_val ^= hash(attr_name)
        if isinstance(attr_val, Hashable):
          hash_val ^= hash(attr_val)
    return hash_val


def validate_shards(
    shards: Sequence[TensorSliceDict],
    shardable_tensors: Sequence[ShardableTensor],
    callback_description: str
) -> None:
  """Validates shards generated by the sharding_callback."""
  unseen_tensor_dict = {}
  for shardable_tensor in shardable_tensors:
    unseen_tensor_dict.setdefault(
        shardable_tensor.checkpoint_key, {}
        )[shardable_tensor.slice_spec] = shardable_tensor.tensor
  seen_tensor_set = set()

  for shard_tensors in shards:
    task_tensor = None
    for checkpoint_key, tensor_slice_dict in shard_tensors.items():
      for slice_spec, shard_tensor in tensor_slice_dict.items():
        slice_spec = slice_spec.strip()

        # Validate uniqueness.
        if (checkpoint_key, slice_spec) in seen_tensor_set:
          raise RuntimeError(
              "After executing the checkpoint sharding callback, multiple "
              "tensors with the same checkpoint key and slice spec were "
              "found:\n"
              f"  callback_description: {callback_description}\n"
              f"  checkpoint_key: {checkpoint_key}\n"
              f"  slice_spec: {slice_spec}\n")

        # Validate no added tensors.
        if checkpoint_key not in unseen_tensor_dict:
          raise RuntimeError(
              "After executing the checkpoint sharding callback, a tensor "
              "not originally in the object graph was found in the "
              "checkpoint shards:\n"
              f"  callback_description: {callback_description}\n"
              f"  checkpoint_key: {checkpoint_key}\n"
              f"  slice_spec: {slice_spec}\n")

        # Validate no shape change.
        target_shape = unseen_tensor_dict[checkpoint_key][slice_spec].shape
        if shard_tensor.shape != target_shape:
          raise RuntimeError(
              "After executing the checkpoint sharding callback, a tensor "
              "was found with an altered shape:\n"
              f"  callback_description: {callback_description}\n"
              f"  checkpoint_key: {checkpoint_key}\n"
              f"  slice_spec: {slice_spec}\n"
              f"  original tensor_shape: {target_shape}\n"
              f"  new tensor_shape: {shard_tensor.shape}\n")

        # Validate no dtype change.
        target_dtype = unseen_tensor_dict[checkpoint_key][slice_spec].dtype
        if shard_tensor.dtype != target_dtype:
          raise RuntimeError(
              "After executing the checkpoint sharding callback, a tensor "
              "was found with an altered dtype:\n"
              f"  callback_description: {callback_description}\n"
              f"  checkpoint_key: {checkpoint_key}\n"
              f"  slice_spec: {slice_spec}\n"
              f"  original tensor_dtype: {target_dtype}\n"
              f"  new tensor_dtype: {shard_tensor.dtype}\n")

        # Validate no task change.
        target_task = device_lib.DeviceSpec.from_string(
            unseen_tensor_dict[checkpoint_key][slice_spec].device).task
        shard_tensor_task = device_lib.DeviceSpec.from_string(
            shard_tensor.device).task
        if shard_tensor_task != target_task:
          raise RuntimeError(
              "After executing the checkpoint sharding callback, a tensor "
              "was found with an altered task:\n"
              f"  callback_description: {callback_description}\n"
              f"  checkpoint_key: {checkpoint_key}\n"
              f"  slice_spec: {slice_spec}\n"
              f"  original tensor_task: {target_task}\n"
              f"  new tensor_task: {shard_tensor_task}\n")

        # Validate tensors in shard have the same task.
        if task_tensor is None:
          task_tensor = ShardableTensor(
              _tensor_save_spec=None,
              tensor=None,
              dtype=None,
              device=shard_tensor.device,
              name=None,
              shape=None,
              slice_spec=slice_spec,
              checkpoint_key=checkpoint_key,
              trackable=None)
        else:
          task1 = device_lib.DeviceSpec.from_string(task_tensor.device).task
          task2 = device_lib.DeviceSpec.from_string(shard_tensor.device).task
          if task1 is not None and task2 is not None and task1 != task2:
            raise RuntimeError(
                "After executing the checkpoint sharding callback, tensors "
                "with different tasks were found in the same shard:\n"
                f"  callback_description: {callback_description}\n"
                "  tensor #1:"
                f"    checkpoint_key: {task_tensor.checkpoint_key}\n"
                f"    slice_spec: {task_tensor.slice_spec}\n"
                f"    task: {task1}\n"
                "  tensor #2:"
                f"    checkpoint_key: {checkpoint_key}\n"
                f"    slice_spec: {slice_spec}\n"
                f"    task: {task2}\n")

        del unseen_tensor_dict[checkpoint_key][slice_spec]
        if not unseen_tensor_dict[checkpoint_key]:
          del unseen_tensor_dict[checkpoint_key]
        seen_tensor_set.add((checkpoint_key, slice_spec))

  # validate no tensor removal
  if unseen_tensor_dict:
    tensors_info = ""
    for ckpt_key, slice_spec in unseen_tensor_dict.items():
      tensors_info += "  tensor:\n"
      tensors_info += f"    checkpoint_key: {ckpt_key}\n"
      tensors_info += f"    slice_spec: {slice_spec}\n"
    raise RuntimeError(
        "After executing the checkpoint sharding callback, tensors in the "
        "object graph were not found in the checkpoint shards:\n"
        f"  callback_description: {callback_description}\n"
        f"{tensors_info}")
