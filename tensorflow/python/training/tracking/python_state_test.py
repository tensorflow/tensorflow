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

import io
import os

import numpy

from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.tracking import base
from tensorflow.python.training.tracking import python_state
from tensorflow.python.training.tracking import util


class _NumpyState(base.Trackable):
  """A checkpointable object whose NumPy array attributes are saved/restored.

  Example usage:

  ```python
  arrays = _NumpyState()
  checkpoint = tf.train.Checkpoint(numpy_arrays=arrays)
  arrays.x = numpy.zeros([3, 4])
  save_path = checkpoint.save("/tmp/ckpt")
  arrays.x[1, 1] = 4.
  checkpoint.restore(save_path)
  assert (arrays.x == numpy.zeros([3, 4])).all()

  second_checkpoint = tf.train.Checkpoint(
      numpy_arrays=_NumpyState())
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
    value = super(_NumpyState, self)._lookup_dependency(name)
    if value is None:
      value = _NumpyWrapper(numpy.array([]))
      new_reference = base.TrackableReference(name=name, ref=value)
      self._unconditional_checkpoint_dependencies.append(new_reference)
      self._unconditional_dependency_names[name] = value
      super(_NumpyState, self).__setattr__(name, value)
    return value

  def __getattribute__(self, name):
    """Un-wrap `_NumpyWrapper` objects when accessing attributes."""
    value = super(_NumpyState, self).__getattribute__(name)
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
        existing = super(_NumpyState, self).__getattribute__(name)
        existing.array = value
        return
      except AttributeError:
        value = _NumpyWrapper(value)
        self._track_trackable(value, name=name, overwrite=True)
    elif (name not in ("_setattr_tracking", "_update_uid")
          and getattr(self, "_setattr_tracking", True)):
      # Mixing restore()-created attributes with user-added checkpointable
      # objects is tricky, since we can't use the `_lookup_dependency` trick to
      # re-create attributes (we might accidentally steal the restoration for
      # another checkpointable object). For now `_NumpyState` objects must be
      # leaf nodes. Theoretically we could add some extra arguments to
      # `_lookup_dependency` to figure out whether we should create a NumPy
      # array for the attribute or not.
      raise NotImplementedError(
          ("Assigned %s to the %s property of %s, which is not a NumPy array. "
           "Currently mixing NumPy arrays and other checkpointable objects is "
           "not supported. File a feature request if this limitation bothers "
           "you.")
          % (value, name, self))
    super(_NumpyState, self).__setattr__(name, value)


class _NumpyWrapper(python_state.PythonState):
  """Wraps a NumPy array for storage in an object-based checkpoint."""

  def __init__(self, array):
    """Specify a NumPy array to wrap.

    Args:
      array: The NumPy array to save and restore (may be overwritten).
    """
    self.array = array

  def serialize(self):
    """Callback to serialize the array."""
    string_file = io.BytesIO()
    try:
      numpy.save(string_file, self.array, allow_pickle=False)
      serialized = string_file.getvalue()
    finally:
      string_file.close()
    return serialized

  def deserialize(self, string_value):
    """Callback to deserialize the array."""
    string_file = io.BytesIO(string_value)
    try:
      self.array = numpy.load(string_file, allow_pickle=False)
    finally:
      string_file.close()


class NumpyStateTests(test.TestCase):

  def testWrapper(self):
    directory = self.get_temp_dir()
    prefix = os.path.join(directory, "ckpt")
    root = util.Checkpoint(numpy=_NumpyWrapper(numpy.array([1.])))
    save_path = root.save(prefix)
    root.numpy.array *= 2.
    self.assertEqual([2.], root.numpy.array)
    root.restore(save_path)
    self.assertEqual([1.], root.numpy.array)

  @test_util.run_in_graph_and_eager_modes
  def testSaveRestoreNumpyState(self):
    directory = self.get_temp_dir()
    prefix = os.path.join(directory, "ckpt")
    save_state = _NumpyState()
    saver = util.Checkpoint(numpy=save_state)
    save_state.a = numpy.ones([2, 2])
    save_state.b = numpy.ones([2, 2])
    save_state.b = numpy.zeros([2, 2])
    save_state.c = numpy.int64(3)
    self.assertAllEqual(numpy.ones([2, 2]), save_state.a)
    self.assertAllEqual(numpy.zeros([2, 2]), save_state.b)
    self.assertEqual(3, save_state.c)
    first_save_path = saver.save(prefix)
    save_state.a[1, 1] = 2.
    save_state.c = numpy.int64(4)
    second_save_path = saver.save(prefix)

    load_state = _NumpyState()
    loader = util.Checkpoint(numpy=load_state)
    loader.restore(first_save_path).initialize_or_restore()
    self.assertAllEqual(numpy.ones([2, 2]), load_state.a)
    self.assertAllEqual(numpy.zeros([2, 2]), load_state.b)
    self.assertEqual(3, load_state.c)
    load_state.a[0, 0] = 42.
    self.assertAllEqual([[42., 1.], [1., 1.]], load_state.a)
    loader.restore(first_save_path).run_restore_ops()
    self.assertAllEqual(numpy.ones([2, 2]), load_state.a)
    loader.restore(second_save_path).run_restore_ops()
    self.assertAllEqual([[1., 1.], [1., 2.]], load_state.a)
    self.assertAllEqual(numpy.zeros([2, 2]), load_state.b)
    self.assertEqual(4, load_state.c)

  def testNoGraphPollution(self):
    graph = ops.Graph()
    with graph.as_default(), session.Session():
      directory = self.get_temp_dir()
      prefix = os.path.join(directory, "ckpt")
      save_state = _NumpyState()
      saver = util.Checkpoint(numpy=save_state)
      save_state.a = numpy.ones([2, 2])
      save_path = saver.save(prefix)
      saver.restore(save_path)
      graph.finalize()
      saver.save(prefix)
      save_state.a = numpy.zeros([2, 2])
      saver.save(prefix)
      saver.restore(save_path)

  @test_util.run_in_graph_and_eager_modes
  def testNoMixedNumpyStateTF(self):
    save_state = _NumpyState()
    save_state.a = numpy.ones([2, 2])
    with self.assertRaises(NotImplementedError):
      save_state.v = variables.Variable(1.)

  @test_util.run_in_graph_and_eager_modes
  def testDocstringExample(self):
    arrays = _NumpyState()
    checkpoint = util.Checkpoint(numpy_arrays=arrays)
    arrays.x = numpy.zeros([3, 4])
    save_path = checkpoint.save(os.path.join(self.get_temp_dir(), "ckpt"))
    arrays.x[1, 1] = 4.
    checkpoint.restore(save_path)
    self.assertAllEqual(numpy.zeros([3, 4]), arrays.x)

    second_checkpoint = util.Checkpoint(numpy_arrays=_NumpyState())
    second_checkpoint.restore(save_path)
    self.assertAllEqual(numpy.zeros([3, 4]), second_checkpoint.numpy_arrays.x)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
