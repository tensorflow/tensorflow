"""Utilities for including Python state in TensorFlow checkpoints."""
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools
import six

import numpy

from tensorflow.python.training.checkpointable import base

# pylint: disable=g-import-not-at-top
try:
  # In Python 2.x, use the faster string buffering option.
  from cStringIO import StringIO as BytesIO
except ImportError:
  from io import BytesIO
# pylint: enable=g-import-not-at-top


class NumpyState(base.CheckpointableBase):
  """A checkpointable object whose NumPy array attributes are saved/restored.

  Example usage:

  ```python
  arrays = tf.contrib.checkpoint.NumpyState()
  checkpoint = tf.train.Checkpoint(numpy_arrays=arrays)
  arrays.x = numpy.zeros([3, 4])
  save_path = checkpoint.save("/tmp/ckpt")
  arrays.x[1, 1] = 4.
  checkpoint.restore(save_path)
  assert (arrays.x == numpy.zeros([3, 4])).all()

  second_checkpoint = tf.train.Checkpoint(
      numpy_arrays=tf.contrib.checkpoint.NumpyState())
  # Attributes of NumpyState objects are created automatically by restore()
  second_checkpoint.restore(save_path)
  assert (second_checkpoint.numpy_arrays.x == numpy.zeros([3, 4])).all()
  ```

  Note that `NumpyState` objects re-create the attributes of the previously
  saved object on `restore()`. This is in contrast to TensorFlow variables, for
  which a `Variable` object must be created and assigned to an attribute.

  This snippet works both when graph building and when executing eagerly. On
  save, the NumPy array(s) are fed as strings to be saved in the checkpoint (via
  a placeholder when graph building, or as a string constant when executing
  eagerly). When restoring they skip the TensorFlow graph entirely, and so no
  restore ops need be run. This means that restoration always happens eagerly,
  rather than waiting for `checkpoint.restore(...).run_restore_ops()` like
  TensorFlow variables when graph building.
  """

  def _lookup_dependency(self, name):
    """Create placeholder NumPy arrays for to-be-restored attributes.

    Typically `_lookup_dependency` is used to check by name whether a dependency
    exists. We cheat slightly by creating a checkpointable object for `name` if
    we don't already have one, giving us attribute re-creation behavior when
    loading a checkpoint.

    Args:
      name: The name of the dependency being checked.
    Returns:
      An existing dependency if one exists, or a new `_NumpyWrapper` placeholder
      dependency (which will generally be restored immediately).
    """
    value = super(NumpyState, self)._lookup_dependency(name)
    if value is None:
      value = _NumpyWrapper(numpy.array([]))
      new_reference = base.CheckpointableReference(name=name, ref=value)
      self._unconditional_checkpoint_dependencies.append(new_reference)
      self._unconditional_dependency_names[name] = value
      super(NumpyState, self).__setattr__(name, value)
    return value

  def __getattribute__(self, name):
    """Un-wrap `_NumpyWrapper` objects when accessing attributes."""
    value = super(NumpyState, self).__getattribute__(name)
    if isinstance(value, _NumpyWrapper):
      return value.array
    return value

  def __setattr__(self, name, value):
    """Automatically wrap NumPy arrays assigned to attributes."""
    # TODO(allenl): Consider supporting lists/tuples, either ad-hoc or by making
    # ndarrays checkpointable natively and using standard checkpointable list
    # tracking.
    if isinstance(value, (numpy.ndarray, numpy.generic)):
      try:
        existing = super(NumpyState, self).__getattribute__(name)
        existing.array = value
        return
      except AttributeError:
        value = _NumpyWrapper(value)
        self._track_checkpointable(value, name=name, overwrite=True)
    elif (name not in ("_setattr_tracking", "_update_uid")
          and getattr(self, "_setattr_tracking", True)):
      # Mixing restore()-created attributes with user-added checkpointable
      # objects is tricky, since we can't use the `_lookup_dependency` trick to
      # re-create attributes (we might accidentally steal the restoration for
      # another checkpointable object). For now `NumpyState` objects must be
      # leaf nodes. Theoretically we could add some extra arguments to
      # `_lookup_dependency` to figure out whether we should create a NumPy
      # array for the attribute or not.
      raise NotImplementedError(
          ("Assigned %s to the %s property of %s, which is not a NumPy array. "
           "Currently mixing NumPy arrays and other checkpointable objects is "
           "not supported. File a feature request if this limitation bothers "
           "you.")
          % (value, name, self))
    super(NumpyState, self).__setattr__(name, value)


@six.add_metaclass(abc.ABCMeta)
class PythonStateWrapper(base.CheckpointableBase):
  """Wraps a Python object for storage in an object-based checkpoint."""

  @abc.abstractmethod
  def _serialize(self):
    """Callback for `PythonStringStateSaveable` to serialize the object."""

  @abc.abstractmethod
  def _deserialize(self, string_value):
    """Callback for `PythonStringStateSaveable` to deserialize the object."""

  def _gather_saveables_for_checkpoint(self):
    """Specify callbacks for saving and restoring `array`."""
    return {
        "py_state": functools.partial(
            base.PythonStringStateSaveable,
            state_callback=self._serialize,
            restore_callback=self._deserialize)
        }


class _NumpyWrapper(PythonStateWrapper):
  """Wraps a NumPy array for storage in an object-based checkpoint."""

  def __init__(self, array):
    """Specify a NumPy array to wrap.

    Args:
      array: The NumPy array to save and restore (may be overwritten).
    """
    self.array = array

  def _serialize(self):
    """Callback to serialize the array."""
    string_file = BytesIO()
    try:
      numpy.save(string_file, self.array, allow_pickle=False)
      serialized = string_file.getvalue()
    finally:
      string_file.close()
    return serialized

  def _deserialize(self, string_value):
    """Callback to deserialize the array."""
    string_file = BytesIO(string_value)
    try:
      self.array = numpy.load(string_file, allow_pickle=False)
    finally:
      string_file.close()

