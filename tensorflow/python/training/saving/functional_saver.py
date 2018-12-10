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
"""Saves and restore variables inside traced @tf.functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util


class Saver(object):
  """A minimal utility class for saving and restoring checkpoints.

  Note that this is a low-level utility which stores Tensors in the keys
  specified by `SaveableObject`s. Higher-level utilities for object-based
  checkpointing are built on top of it.
  """

  def __init__(self, saveable_objects):
    """Specify a list of `SaveableObject`s to save and restore.

    Args:
      saveable_objects: A list of `SaveableObject`s.
    """
    saveable_objects = list(saveable_objects)
    for saveable in saveable_objects:
      if not isinstance(saveable, saveable_object.SaveableObject):
        raise ValueError(
            "Saver expected a list of SaveableObjects, got %s." % (saveable,))
    self._saveable_objects = saveable_objects

  # TODO(b/120569892): Use tf.function here
  def save(self, file_prefix):
    """Save the saveable objects to a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix to
        save under.
    Returns:
      A scalar string Tensor containing `file_prefix` with control dependencies
      on the save ops.
    """
    tensor_names = []
    tensors = []
    tensor_slices = []
    for saveable in self._saveable_objects:
      for spec in saveable.specs:
        tensor_names.append(spec.name)
        tensors.append(spec.tensor)
        tensor_slices.append(spec.slice_spec)
    with ops.control_dependencies(
        [io_ops.save_v2(file_prefix, tensor_names, tensor_slices, tensors)]):
      return array_ops.identity(file_prefix)

  # TODO(b/120569892): Use tf.function here
  def restore(self, file_prefix):
    """Restore the saveable objects from a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix for
        files to read from.

    Returns:
      An operation which restores the `Saver`'s `SaveableObject`s when run, or
      None if executing eagerly.
    """
    restore_ops = []
    for saveable in self._saveable_objects:
      if saveable.device:
        device = saveable_object_util.set_cpu0(saveable.device)
      else:
        device = None
      with ops.device(device):
        tensors = []
        for spec in saveable.specs:
          tensors.append(
              io_ops.restore_v2(
                  file_prefix,
                  [spec.name],
                  [spec.slice_spec],
                  [spec.dtype])[0])
        restore_ops.append(saveable.restore(tensors, restored_shapes=None))
    return control_flow_ops.group(restore_ops)
