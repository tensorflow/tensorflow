# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Experimental API for checkpoint adapter."""
import abc
from typing import List, Optional

from tensorflow.python.framework import tensor
from tensorflow.python.trackable import base


class ReshardCallback:
  """API to reshard a checkpoint value during restore.

  When a ReshardCallback is attached to a CheckpointPosition, the restored value
  of the checkpoint position is resharded based on this callback.
  """

  def object_name(self) -> str:
    """Returns the local name of the object being restored.

    Override this method when the local name of object is different than in the
    checkpoint.
    """
    return None

  def reshard(
      self,
      checkpoint_values: List[tensor.Tensor],
      shape_and_slice_spec: List[str],
  ) -> tensor.Tensor:
    """Reshards the checkpoint values as read from the checkpoint file.

    Override this to reshard/modify the restored values
    Args:
      checkpoint_values: The values returned by the restore op, as read from
        file.
      shape_and_slice_spec: The shape and slice spec required by the caller.

    Returns:
      List of restored Tensor values after being resharded.
    """
    del shape_and_slice_spec  # unused
    # Default reshard is a trivial one.
    if len(checkpoint_values) != 1:
      raise ValueError("Default reshard expects a single checkpoint value.")
    return checkpoint_values[0]

  def update_restore_inputs(
      self, checkpoint_key, shape_and_slice_spec
  ) -> tuple[List[str], List[str]]:
    """Updates the specs to restore op.

    Override this method if the arguments to restore op need to be updated as
    per the resharding required.
    Args:
      checkpoint_key: The cehckpopoint key as requested by the caller
      shape_and_slice_spec: The shape and slice spec as requested by caller

    Returns:
    Tuple of list of checkpoint_keys and specs that the restore op should fetch
    as per the resharding requirement. The length of checkpoint keys returned by
    this method will match the length of checkpoint_values that are input to
    `reshard`.
    """
    return ([checkpoint_key], [shape_and_slice_spec])


class AbstractCheckpointAdapter(abc.ABC):
  """Abstract API for checkpoint adapter.

  This is an experimental API that specifies how checkpoint restore should be
  adapted for specific trackable objects.
  """

  @classmethod
  @abc.abstractmethod
  def create_from_checkpoint(cls, path: str):
    """Create factory to create an Adapter from checkpoint.

    Args:
      path: Path to checkpoint.
    """

  @abc.abstractmethod
  def is_applicable(self, trackable: base.Trackable) -> bool:
    """Returns whether the adapter is applicable to trackable for resharding.

    Args:
      trackable: A Trackable object that is being restored.

    Returns:
      A Boolean indicating if the checkpoint value for this Trackable should be
      resharded.
    """

  @abc.abstractmethod
  def get_reshard_callback(self, name: str) -> Optional[ReshardCallback]:
    """Returns the reshard callback for the trackable with `name`."""

  def maybe_reshard(self, name: str) -> tuple[str, Optional[ReshardCallback]]:
    """Returns the updated name and ReshardCallback applicable to it."""
    callback = self.get_reshard_callback(name)
    if callback is None:
      return name, None
    if callback.object_name():
      return callback.object_name(), callback
    return name, callback
