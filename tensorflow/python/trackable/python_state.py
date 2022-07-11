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
import abc
import functools

import six

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.trackable import base
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.util.tf_export import tf_export


@tf_export("train.experimental.PythonState")
@six.add_metaclass(abc.ABCMeta)
class PythonState(base.Trackable):
  """A mixin for putting Python state in an object-based checkpoint.

  This is an abstract class which allows extensions to TensorFlow's object-based
  checkpointing (see `tf.train.Checkpoint`). For example a wrapper for NumPy
  arrays:

  ```python
  import io
  import numpy

  class NumpyWrapper(tf.train.experimental.PythonState):

    def __init__(self, array):
      self.array = array

    def serialize(self):
      string_file = io.BytesIO()
      try:
        numpy.save(string_file, self.array, allow_pickle=False)
        serialized = string_file.getvalue()
      finally:
        string_file.close()
      return serialized

    def deserialize(self, string_value):
      string_file = io.BytesIO(string_value)
      try:
        self.array = numpy.load(string_file, allow_pickle=False)
      finally:
        string_file.close()
  ```

  Instances of `NumpyWrapper` are checkpointable objects, and will be saved and
  restored from checkpoints along with TensorFlow state like variables.

  ```python
  root = tf.train.Checkpoint(numpy=NumpyWrapper(numpy.array([1.])))
  save_path = root.save(prefix)
  root.numpy.array *= 2.
  assert [2.] == root.numpy.array
  root.restore(save_path)
  assert [1.] == root.numpy.array
  ```
  """

  @abc.abstractmethod
  def serialize(self):
    """Callback to serialize the object. Returns a string."""

  @abc.abstractmethod
  def deserialize(self, string_value):
    """Callback to deserialize the object."""

  def _gather_saveables_for_checkpoint(self):
    """Specify callbacks for saving and restoring `array`."""
    return {
        "py_state": functools.partial(
            PythonStringStateSaveable,
            state_callback=self.serialize,
            restore_callback=self.deserialize)
        }


@six.add_metaclass(abc.ABCMeta)
class PythonStateSaveable(saveable_object.SaveableObject):
  """An interface for saving/restoring volatile Python state."""

  @abc.abstractmethod
  def feed_dict_additions(self):
    """When running a graph, indicates fresh state to feed.

    Returns:
      A dictionary mapping `Tensor`s to current Python state.
    """

  @abc.abstractmethod
  def freeze(self):
    """Create a new `SaveableObject` which freezes current state as a constant.

    Used when executing eagerly to embed the current state as a constant, or
    when creating a static tf.compat.v1.train.Saver with the frozen current
    Python state.

    Returns:
      A `SaveableObject` which is not a `PythonStateSaveable` instance (i.e. has
      no Python state associated with it).
    """


class PythonStringStateSaveable(PythonStateSaveable):
  """Saves Python state in a checkpoint."""

  def __init__(self, name, state_callback, restore_callback):
    """Configure saving.

    Args:
      name: The checkpoint key to write to.
      state_callback: A function taking no arguments which returns a string.
        This function is run every time a checkpoint is written.
      restore_callback: A function taking a Python string, used to restore
        state.
    """
    def _state_callback_wrapper():
      with ops.init_scope():
        return state_callback()

    self._state_callback = _state_callback_wrapper
    self._restore_callback = restore_callback
    with ops.device("/cpu:0"):
      self._save_string = constant_op.constant("", dtype=dtypes.string)
    spec = saveable_object.SaveSpec(
        self._save_string, "", name, dtype=dtypes.string)
    super(PythonStringStateSaveable, self).__init__(self._save_string, [spec],
                                                    name)

  def feed_dict_additions(self):
    """When running a graph, indicates fresh state to feed."""
    return {self._save_string: self._state_callback()}

  def freeze(self):
    """Create a frozen `SaveableObject` which saves the current state."""

    def _constant_state():
      return constant_op.constant(self._state_callback(), dtype=dtypes.string)

    return base.NoRestoreSaveable(
        tensor=_constant_state,
        dtype=dtypes.string,
        name=self.name,
        device="cpu:0")

  def python_restore(self, restored_strings):
    """Called to restore Python state."""
    if self._restore_callback:
      restored, = restored_strings
      self._restore_callback(restored)

  def restore(self, restored_tensors, restored_shapes):
    """Called to restore TensorFlow state (nothing to do)."""
    return control_flow_ops.no_op()

