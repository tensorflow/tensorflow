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
"""Types for specifying saving and loading behavior."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class SaveSpec(object):
  """Class used to describe tensor slices that need to be saved."""

  def __init__(self, tensor, slice_spec, name, dtype=None):
    """Creates a `SaveSpec` object.

    Args:
      tensor: the tensor to save or callable that produces a tensor to save.
      slice_spec: the slice to be saved. See `Variable.SaveSliceInfo`.
      name: the name to save the tensor under.
      dtype: The data type of the Tensor. Required if `tensor` is callable.
        Used for error checking in the restore op.
    """
    self._tensor = tensor
    self.slice_spec = slice_spec
    self.name = name
    if callable(self._tensor):
      if dtype is None:
        raise AssertionError(
            "When passing a callable `tensor` to a SaveSpec, an explicit "
            "dtype must be provided.")
      self.dtype = dtype
    else:
      self.dtype = tensor.dtype

  @property
  def tensor(self):
    return self._tensor() if callable(self._tensor) else self._tensor


class SaveableObject(object):
  """Base class for saving and restoring saveable objects."""

  def __init__(self, op, specs, name):
    """Creates a `SaveableObject` object.

    Args:
      op: the "producer" object that this class wraps; it produces a list of
        tensors to save.  E.g., a "Variable" object saving its backing tensor.
      specs: a list of SaveSpec, each element of which describes one tensor to
        save under this object. All Tensors must be on the same device.
      name: the name to save the object under.
    """
    self.op = op
    self.specs = specs
    self.name = name
    self._device = None

  @property
  def optional_restore(self):
    """A hint to restore assertions that this object is optional."""
    return False  # Default to required

  @property
  def device(self):
    """The device for SaveSpec Tensors."""
    # Note that SaveSpec.tensor runs Tensor-gathering ops when executing
    # eagerly, making this call potentially very expensive.
    #
    # TODO(allenl): Consider another way to gather device information. Lower
    # priority since this property isn't part of the normal save()/restore()
    # workflow, but does come up when some alternative builders are passed to
    # the Saver.
    if self._device is None:
      self._device = self.specs[0].tensor.device
    return self._device

  def restore(self, restored_tensors, restored_shapes):
    """Restores this object from 'restored_tensors'.

    Args:
      restored_tensors: the tensors that were loaded from a checkpoint
      restored_shapes: the shapes this object should conform to after
        restore, or None.

    Returns:
      An operation that restores the state of the object.

    Raises:
      ValueError: If the object cannot be restored using the provided
        parameters.
    """
    # pylint: disable=unused-argument
    raise ValueError("Calling an abstract method.")
