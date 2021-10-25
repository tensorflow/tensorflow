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

from tensorflow.python.training.tracking import base
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
            base.PythonStringStateSaveable,
            state_callback=self.serialize,
            restore_callback=self.deserialize)
        }
