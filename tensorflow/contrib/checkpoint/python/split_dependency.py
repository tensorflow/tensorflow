"""Utility for creating multiple dependencies with synchronized save/restore."""
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training.tracking import base as trackable


class _CallbackSaveable(saver_lib.BaseSaverBuilder.SaveableObject):
  """Wraps save and restore callbacks as a `SaveableObject`."""

  def __init__(self, name, dtype, device, save_callback, restore_callback):
    self._restore_callback = restore_callback
    spec = saver_lib.BaseSaverBuilder.SaveSpec(
        tensor=save_callback,
        slice_spec="",
        name=name,
        dtype=dtype,
        device=device)
    super(_CallbackSaveable, self).__init__(
        save_callback, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    """Restore the same value into both variables."""
    tensor, = restored_tensors
    return self._restore_callback(tensor)


class _SplitDependency(trackable.Trackable):
  """Looks like a regular variable while synchronizing save/restores."""

  def __init__(self, save_buffer, restore_buffer, name, dtype, device,
               num_components, fill_save_buffer_fn, consume_restore_buffer_fn):
    self._save_buffer = save_buffer
    self._restore_buffer = restore_buffer
    self._name = name
    self._dtype = dtype
    self._device = device
    self._num_components = num_components
    self._fill_save_buffer_fn = fill_save_buffer_fn
    self._consume_restore_buffer_fn = consume_restore_buffer_fn

  def _save(self):
    """Pull from the shared buffer, populating it if necessary."""
    if self._name not in self._save_buffer:
      if self._save_buffer:
        raise AssertionError(
            ("Split dependency %s (%s) unsynchronized. Split dependencies must "
             "be saved together.") % (self._name, self))
      self._fill_save_buffer_fn(self._save_buffer)
    return self._save_buffer.pop(self._name)

  def _restore(self, tensor):
    """Push into the shared buffer, flushing it if necessary."""
    if self._name in self._restore_buffer:
      raise AssertionError(
          ("Split dependency %s (%s) unsynchronized. Split dependencies must "
           "be restored together.") % (self._name, self))
    self._restore_buffer[self._name] = tensor
    if len(self._restore_buffer) == self._num_components:
      op = self._consume_restore_buffer_fn(self._restore_buffer)
      self._restore_buffer.clear()
      return op
    else:
      return control_flow_ops.no_op()

  def _gather_saveables_for_checkpoint(self):
    """Looks to Trackable like a regular variable."""
    return {
        trackable.VARIABLE_VALUE_KEY:
        functools.partial(_CallbackSaveable,
                          dtype=self._dtype,
                          device=self._device,
                          save_callback=self._save,
                          restore_callback=self._restore)
    }


def split_dependency(component_names, component_dtypes,
                     fill_save_buffer_fn, consume_restore_buffer_fn,
                     device):
  """Creates multiple dependencies with a synchronized save/restore.

  Useful when a single op produces `Tensor`s which should each be saved under
  different objects, or when `Tensor`s saved with many different objects need to
  be restored together as inputs to a single op (i.e. an object which uses a
  single fused op may be swapped out for a subgraph of objects, and these two
  programs are checkpoint compatible).

  Args:
    component_names: A sequence of names for the split
      dependencies. `fill_save_buffer_fn` must add these keys to the dictionary
      it is passed, and `consume_restore_buffer_fn` will receive a dictionary
      with these keys.
    component_dtypes: Data types for the `Tensor`s being saved and restored, a
      sequence corresponding to `component_names`.
    fill_save_buffer_fn: A function which takes an empty dictionary as an
      argument and adds `Tensor`s with `component_names` as keys. These
      `Tensor`s will be saved as if they were individual variables.
    consume_restore_buffer_fn: A function which takes a dictionary with
      `component_names` as keys mapping to restored individual `Tensor`s and
      returns a restore op (or if executing eagerly, runs the restoration and
      may return `None`).
    device: The device on which to run save and restore operations.

  Returns:
    A dictionary mapping from names to Trackable objects. If one is
    reachable from an object as a dependency, the others should be too; adding
    dependencies on some but not all of the objects will result in errors.
  """
  save_buffer = {}
  restore_buffer = {}
  split_dependencies = {}
  for name, dtype in zip(component_names, component_dtypes):
    split_dependencies[name] = _SplitDependency(
        save_buffer=save_buffer,
        restore_buffer=restore_buffer,
        name=name,
        dtype=dtype,
        device=device,
        num_components=len(component_names),
        fill_save_buffer_fn=fill_save_buffer_fn,
        consume_restore_buffer_fn=consume_restore_buffer_fn)
  return split_dependencies
