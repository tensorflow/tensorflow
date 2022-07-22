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
import io
import os

import numpy
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.platform import test
from tensorflow.python.trackable import python_state


class _NumpyState(module.Module):
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

  def __init__(self):
    super(_NumpyState, self).__setattr__("_arrays", module.Module())

  def __getattribute__(self, name):
    """Un-wrap `_NumpyWrapper` objects when accessing attributes."""
    try:
      arrays = super(_NumpyState, self).__getattribute__("_arrays")
    except AttributeError:
      # _arrays hasn't been assigned yet
      return super(_NumpyState, self).__getattribute__(name)
    try:
      value = getattr(arrays, name)
    except AttributeError:
      dummy_array = numpy.array([])
      setattr(arrays, name, _NumpyWrapper(dummy_array))
      value = getattr(arrays, name)
      if value.array is dummy_array:
        # No set or restored attribute with this name
        delattr(arrays, name)
        return super(_NumpyState, self).__getattribute__(name)

    if isinstance(value, _NumpyWrapper):
      return value.array
    return super(_NumpyState, self).__getattribute__(name)

  def __setattr__(self, name, value):
    """Automatically wrap NumPy arrays assigned to attributes."""
    if isinstance(value, (numpy.ndarray, numpy.generic)):
      try:
        existing = getattr(self._arrays, name)
        existing.array = value
        return
      except AttributeError:
        value = _NumpyWrapper(value)
      setattr(self._arrays, name, value)
      return
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
      self.array = numpy.load(string_file, allow_pickle=False)  # pylint: disable=unexpected-keyword-arg
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
