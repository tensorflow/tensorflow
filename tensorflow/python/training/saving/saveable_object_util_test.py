# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for saveable_object_util."""

import os

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.trackable import base
from tensorflow.python.trackable import resource
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util


class _VarSaveable(saveable_object.SaveableObject):

  def __init__(self, var, slice_spec, name):
    specs = [saveable_object.SaveSpec(var.read_value(), slice_spec, name)]
    super().__init__(var, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    return self.op.assign(restored_tensors[0])


def _create_converted_trackable(obj):
  saveable_factories = saveable_object_util.saveable_objects_from_trackable(obj)
  saveables = [factory(name) for name, factory in saveable_factories.items()]
  return saveable_object_util.SaveableCompatibilityConverter(obj, saveables)


class SaveableCompatibilityConverterTest(test.TestCase):

  def test_convert_no_saveable(self):
    t = base.Trackable()
    converter = _create_converted_trackable(t)
    self.assertEmpty(converter._serialize_to_tensors())
    converter._restore_from_tensors({})

    with self.assertRaisesRegex(ValueError, "Could not restore object"):
      converter._restore_from_tensors({"": 0})

  def test_convert_single_saveable(self):

    class MyTrackable(base.Trackable):

      def __init__(self):
        self.a = variables.Variable(5.0)

      def _gather_saveables_for_checkpoint(self):
        return {"a": lambda name: _VarSaveable(self.a, "", name)}

    t = MyTrackable()
    converter = _create_converted_trackable(t)

    serialized_tensors = converter._serialize_to_tensors()
    self.assertLen(serialized_tensors, 1)
    self.assertIn("a", serialized_tensors)
    self.assertEqual(5, self.evaluate(serialized_tensors["a"]))

    with self.assertRaisesRegex(ValueError, "Could not restore object"):
      converter._restore_from_tensors({})
    with self.assertRaisesRegex(ValueError, "Could not restore object"):
      converter._restore_from_tensors({"not_a": 1.})

    self.assertEqual(5, self.evaluate(t.a))
    converter._restore_from_tensors({"a": 123.})
    self.assertEqual(123, self.evaluate(t.a))

  def test_convert_single_saveable_renamed(self):

    class MyTrackable(base.Trackable):

      def __init__(self):
        self.a = variables.Variable(15.0)

      def _gather_saveables_for_checkpoint(self):
        return {"a": lambda name: _VarSaveable(self.a, "", name + "-value")}

    t = MyTrackable()
    converter = _create_converted_trackable(t)

    serialized_tensors = converter._serialize_to_tensors()

    self.assertLen(serialized_tensors, 1)
    self.assertEqual(15, self.evaluate(serialized_tensors["a-value"]))

    with self.assertRaisesRegex(ValueError, "Could not restore object"):
      converter._restore_from_tensors({"a": 1.})

    self.assertEqual(15, self.evaluate(t.a))
    converter._restore_from_tensors({"a-value": 456.})
    self.assertEqual(456, self.evaluate(t.a))


class _MultiSpecSaveable(saveable_object.SaveableObject):

  def __init__(self, obj, name):
    self.obj = obj
    specs = [
        saveable_object.SaveSpec(obj.a, "", name + "-a"),
        saveable_object.SaveSpec(obj.b, "", name + "-b")]
    super(_MultiSpecSaveable, self).__init__(None, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    del restored_shapes  # Unused.
    self.obj.a.assign(restored_tensors[0])
    self.obj.b.assign(restored_tensors[1])


class State(resource.TrackableResource):

  def __init__(self, initial_value):
    super().__init__()
    self._initial_value = initial_value
    self._initialize()

  def _create_resource(self):
    return gen_resource_variable_ops.var_handle_op(
        shape=[],
        dtype=dtypes.float32,
        shared_name=context.anonymous_name(),
        name="StateVar",
        container="")

  def _initialize(self):
    gen_resource_variable_ops.assign_variable_op(self.resource_handle,
                                                 self._initial_value)

  def _destroy_resource(self):
    gen_resource_variable_ops.destroy_resource_op(self.resource_handle,
                                                  ignore_lookup_error=True)

  def read(self):
    return gen_resource_variable_ops.read_variable_op(self.resource_handle,
                                                      dtypes.float32)

  def assign(self, value):
    gen_resource_variable_ops.assign_variable_op(self.resource_handle, value)


class _StateSaveable(saveable_object.SaveableObject):

  def __init__(self, obj, name):
    spec = saveable_object.SaveSpec(obj.read(), "", name)
    self.obj = obj
    super(_StateSaveable, self).__init__(obj, [spec], name)

  def restore(self, restored_tensors, restored_shapes):
    del restored_shapes  # Unused.
    self.obj.assign(restored_tensors[0])


class SaveableState(State):

  def _gather_saveables_for_checkpoint(self):
    return {
        "value": lambda name: _StateSaveable(self, name)
    }


class TrackableState(State):

  def _serialize_to_tensors(self):
    return {
        "value": self.read()
    }

  def _restore_from_tensors(self, restored_tensors):
    self.assign(restored_tensors["value"])


class SaveableCompatibilityEndToEndTest(test.TestCase):

  def test_checkpoint_comparison(self):
    saveable_state = SaveableState(5.)
    trackable_state = TrackableState(10.)

    # First test that SaveableState and TrackableState are equivalent by
    # saving a checkpoint with both objects and swapping values.

    self.assertEqual(5, self.evaluate(saveable_state.read()))
    self.assertEqual(10, self.evaluate(trackable_state.read()))

    ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
    checkpoint.Checkpoint(a=saveable_state, b=trackable_state).write(ckpt_path)

    status = checkpoint.Checkpoint(b=saveable_state,
                                   a=trackable_state).read(ckpt_path)
    status.assert_consumed()

    self.assertEqual(10, self.evaluate(saveable_state.read()))
    self.assertEqual(5, self.evaluate(trackable_state.read()))

    # Test that the converted SaveableState is compatible with the checkpoint
    # saved above.
    to_convert = SaveableState(0.0)

    converted_saveable_state = _create_converted_trackable(to_convert)

    checkpoint.Checkpoint(a=converted_saveable_state).read(
        ckpt_path).assert_existing_objects_matched().expect_partial()
    self.assertEqual(5, self.evaluate(to_convert.read()))

    checkpoint.Checkpoint(b=converted_saveable_state).read(
        ckpt_path).assert_existing_objects_matched().expect_partial()
    self.assertEqual(10, self.evaluate(to_convert.read()))


if __name__ == "__main__":
  test.main()
