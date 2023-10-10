# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests saving with registered Trackable classes and checkpoint functions."""

import os
import tempfile

from absl.testing import parameterized

from google.protobuf import wrappers_pb2
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import autotrackable


@registration.register_serializable()
class Part(resource_variable_ops.ResourceVariable):

  def __init__(self, value):
    self._init_from_args(value)

  @classmethod
  def _deserialize_from_proto(cls, **kwargs):
    return cls([0, 0])

  def _export_to_saved_model_graph(self, object_map, tensor_map, **kwargs):
    p = Part(array_ops.zeros(self.shape, self.dtype))
    object_map[self] = p
    tensor_map[self.handle] = p.handle
    return [self.handle]


@registration.register_serializable()
class Stack(autotrackable.AutoTrackable):

  def __init__(self, parts=None):
    self.parts = parts

  @def_function.function(input_signature=[])
  def value(self):
    return array_ops_stack.stack(self.parts)


def get_tensor_slices(trackables):
  tensor_names = []
  shapes_and_slices = []
  tensors = []
  restored_trackables = []
  for obj_prefix, obj in trackables.items():
    if isinstance(obj, Part):
      continue  # only save stacks
    tensor_names.append(obj_prefix + "/value")
    shapes_and_slices.append("")
    x = obj.value()
    with ops.device("/device:CPU:0"):
      tensors.append(array_ops.identity(x))
    restored_trackables.append(obj)

  return tensor_names, shapes_and_slices, tensors, restored_trackables


def save_stacks_and_parts(trackables, file_prefix):
  """Save stack and part objects to a checkpoint shard."""
  tensor_names, shapes_and_slices, tensors, _ = get_tensor_slices(trackables)
  io_ops.save_v2(file_prefix, tensor_names, shapes_and_slices, tensors)
  return file_prefix


def restore_stacks_and_parts(trackables, merged_prefix):
  tensor_names, shapes_and_slices, tensors, restored_trackables = (
      get_tensor_slices(trackables))
  dtypes = [t.dtype for t in tensors]
  restored_tensors = io_ops.restore_v2(merged_prefix, tensor_names,
                                       shapes_and_slices, dtypes)
  for trackable, restored_tensor in zip(restored_trackables, restored_tensors):
    expected_shape = trackable.value().get_shape()
    restored_tensor = array_ops.reshape(restored_tensor, expected_shape)
    parts = array_ops_stack.unstack(restored_tensor)
    for part, restored_part in zip(trackable.parts, parts):
      part.assign(restored_part)


registration.register_checkpoint_saver(
    name="stacks",
    predicate=lambda x: isinstance(x, (Stack, Part)),
    save_fn=save_stacks_and_parts,
    restore_fn=restore_stacks_and_parts)


def cycle(obj, cycles, signatures=None, options=None):
  to_save = obj
  for _ in range(cycles):
    path = tempfile.mkdtemp(prefix=test.get_temp_dir())
    # If available, we'll run the save and restore preferring the GPU. This
    # just makes sure we aren't throwing errors and have enough
    # device("CPU") blocks to satisfy the placer.
    with test_util.use_gpu():
      save.save(to_save, path, signatures, options=options)
      loaded = load.load(path)
      signatures = loaded.signatures
    to_save = loaded
  return loaded


@parameterized.named_parameters(
    dict(testcase_name="ReloadOnce", cycles=1),
    dict(testcase_name="ReloadTwice", cycles=2),
    dict(testcase_name="ReloadThrice", cycles=3))
class SavedModelTest(test.TestCase, parameterized.TestCase):

  def test_registered_serializable(self, cycles):

    @registration.register_serializable(name=f"SaveAndLoad{cycles}")
    class Module(autotrackable.AutoTrackable):

      def __init__(self, name="module"):
        self.v = variables.Variable(1.)
        self.name = name

      def _serialize_to_proto(self, **unused_kwargs):
        return wrappers_pb2.StringValue(value=self.name)

      @classmethod
      def _deserialize_from_proto(cls, proto, **unused_kwargs):
        if proto.Is(wrappers_pb2.StringValue.DESCRIPTOR):
          unpacked = wrappers_pb2.StringValue()
          proto.Unpack(unpacked)
          return cls(name=unpacked.value)
        raise AssertionError(
            "Did not receive proto of correct type during deserialization. "
            f"Expected type {wrappers_pb2.StringValue.DESCRIPTOR.full_name}, "
            f"got {proto.TypeName()}")

    m = Module("a")
    m.v.assign(5)
    loaded = cycle(m, cycles)
    self.assertIsInstance(loaded, Module)
    self.assertEqual(5, loaded.v.numpy())
    self.assertEqual("a", loaded.name)

  def test_none_proto(self, cycles):

    @registration.register_serializable(name=f"NoneProto{cycles}")
    class Module(autotrackable.AutoTrackable):

      def __init__(self, name="module"):
        self.v = variables.Variable(1.)
        self.name = name

      # Leave _serialize_to_proto as the default (returns `None`).

      @classmethod
      def _deserialize_from_proto(cls, proto, **unused_kwargs):
        self.assertEqual(proto.ByteSize(), 0)
        return cls("deserialized")

    m = Module("a")
    m.v.assign(5)
    loaded = cycle(m, cycles)
    self.assertIsInstance(loaded, Module)
    self.assertEqual(5, loaded.v.numpy())
    self.assertEqual("deserialized", loaded.name)

  def test_deserialization_dependencies(self, cycles):
    @registration.register_serializable(name=f"Dependency{cycles}")
    class Module(autotrackable.AutoTrackable):

      def __init__(self, v=None):
        self.v = v if v is not None else variables.Variable(1.)

      def _deserialization_dependencies(self, children):
        del children  # Unused.
        return {"v": self.v}

      @classmethod
      def _deserialize_from_proto(cls, dependencies, **unused_kwargs):
        self.assertIn("v", dependencies)
        return cls(v=dependencies["v"])

    m = Module()
    m.v.assign(5)
    loaded = cycle(m, cycles)
    self.assertIsInstance(loaded, Module)
    self.assertEqual(5, loaded.v.numpy())

  def test_registered_saver(self, cycles):
    p1 = Part([1, 4])
    p2 = Part([2, 5])
    p3 = Part([3, 6])
    s = Stack([p1, p2, p3])
    loaded = cycle(s, cycles)
    self.assertAllEqual(s.value(), loaded.value())


class SingleCycleTest(test.TestCase):

  @test_util.deprecated_graph_mode_only()
  def test_registered_saver_fails_in_saved_model_graph_mode(self):
    with context.eager_mode():
      p1 = Part([1, 4])
      p2 = Part([2, 5])
      p3 = Part([3, 6])
      s = Stack([p1, p2, p3])
      save_dir = os.path.join(self.get_temp_dir(), "save_dir")
      save.save(s, save_dir)

    with self.assertRaisesRegex(
        NotImplementedError,
        "registered checkpoint saver is not supported in graph mode"):
      load.load(save_dir)

  def test_registered_saver_checkpoint(self):
    p1 = Part([1, 4])
    p2 = Part([2, 5])
    p3 = Part([3, 6])
    s = Stack([p1, p2, p3])
    s2 = Stack([p3, p1, p2])

    expected_value_s = s.value()
    expected_value_s2 = s2.value()

    ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
    util.Checkpoint(s=s, s2=s2).write(ckpt_path)

    del s, s2, p1, p2, p3

    restore_s = Stack([Part([0, 0]) for _ in range(3)])
    util.Checkpoint(s=restore_s).read(ckpt_path).expect_partial()
    self.assertAllEqual(expected_value_s, restore_s.value())
    util.Checkpoint(s2=restore_s).read(ckpt_path).expect_partial()
    self.assertAllEqual(expected_value_s2, restore_s.value())

  def test_compatible_with_v1_savedmodel(self):
    p1 = Part([1, 4])
    p2 = Part([2, 5])
    p3 = Part([3, 6])
    s = Stack([p1, p2, p3])
    save_path = os.path.join(self.get_temp_dir(), "savedmodel")

    @def_function.function(input_signature=[])
    def serve():
      return {"value": s.value()}

    exported_value = serve()["value"]

    save.save(s, save_path, signatures=serve)
    with ops.Graph().as_default(), session.Session() as sess:
      metagraph = loader.load(sess, ["serve"], save_path)
      value_output = metagraph.signature_def["serving_default"].outputs["value"]
      self.assertAllEqual(exported_value, sess.run(value_output.name))

  def test_non_strict_predicate(self):
    class NonStrictPredicateClass(autotrackable.AutoTrackable):
      pass
    registration.register_checkpoint_saver(
        name="NonStrictPredicate",
        predicate=lambda x: isinstance(x, NonStrictPredicateClass),
        save_fn=lambda **kwargs: [],
        restore_fn=lambda **kwargs: None,
        strict_predicate_restore=False)

    root = NonStrictPredicateClass()
    ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
    util.Checkpoint(root).write(ckpt_path)

    root2 = autotrackable.AutoTrackable()
    # This should run without throwing an error.
    util.Checkpoint(root2).read(ckpt_path)

  def test_strict_predicate(self):
    class StrictPredicateClass(autotrackable.AutoTrackable):
      pass
    registration.register_checkpoint_saver(
        name="StrictPredicate",
        predicate=lambda x: isinstance(x, StrictPredicateClass),
        save_fn=lambda **kwargs: [],
        restore_fn=lambda **kwargs: None,
        strict_predicate_restore=True)

    root = StrictPredicateClass()
    ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
    util.Checkpoint(root).write(ckpt_path)

    root2 = autotrackable.AutoTrackable()
    with self.assertRaisesRegex(ValueError, "saver cannot be used"):
      util.Checkpoint(root2).read(ckpt_path)

  def test_registered_saver_is_called_before_save_after_load(self):
    if not context.executing_eagerly():
      self.skipTest("This test must run under eager mode.")

    class RestoreClass(autotrackable.AutoTrackable):
      pass
    def save_fn(trackables, file_prefix):
      del trackables  # Unused.
      # Check that directory is empty
      files = gfile.ListDirectory(os.path.dirname(file_prefix.numpy()))
      self.assertEmpty(files)

    def restore_fn(trackables, merged_prefix):
      del merged_prefix  # Unused.
      root = next(trackables.values())
      self.assertEqual(root.v.numpy(), 123)

    registration.register_checkpoint_saver(
        name="OptionalRestore",
        predicate=lambda x: isinstance(x, RestoreClass),
        save_fn=save_fn,
        restore_fn=restore_fn)

    root = RestoreClass()
    root.v = variables.Variable(123.0)

    ckpt_path = os.path.join(self.get_temp_dir(), "ckpt")
    util.Checkpoint(root).write(ckpt_path)

  def test_migration_backwards_compatibility(self):
    # Tests that objects migrated to using the advanced saver registration can
    # use pre-migration checkpoints.

    class NoRegisteredSaver(autotrackable.AutoTrackable):

      def __init__(self, name):
        self.name = name

      def _serialize_to_tensors(self):
        return {"name": constant_op.constant(self.name)}

    class RegisteredSaver(autotrackable.AutoTrackable):

      def __init__(self, name):
        self.name = name

    def _get_tensors(trackables, append_name=True):
      tensor_names = []
      shapes_and_slices = []
      tensors = []
      restored_trackables = []
      for obj_prefix, obj in trackables.items():
        tensor_names.append(obj_prefix + "name" if append_name else obj_prefix)
        shapes_and_slices.append("")
        tensors.append(constant_op.constant(obj.name))
        restored_trackables.append(obj)
      return tensor_names, shapes_and_slices, tensors, restored_trackables

    def save_fn(trackables, file_prefix):
      tensor_names, shapes_and_slices, tensors, _ = _get_tensors(trackables)
      io_ops.save_v2(file_prefix, tensor_names, shapes_and_slices, tensors)
      return file_prefix

    def restore_fn(trackables, merged_prefix):
      tensor_names, shapes_and_slices, tensors, restored_trackables = (
          _get_tensors(trackables))
      dtypes = [t.dtype for t in tensors]
      try:
        restored_tensors = io_ops.restore_v2(merged_prefix, tensor_names,
                                             shapes_and_slices, dtypes)
      except errors_impl.NotFoundError:
        # If a NotFoundError is caught, then it means that the checkpoint
        # was written prior to the saver registration migration.
        tensor_names, shapes_and_slices, tensors, restored_trackables = (
            _get_tensors(trackables, append_name=False))
        restored_tensors = io_ops.restore_v2(merged_prefix, tensor_names,
                                             shapes_and_slices, dtypes)
      for trackable, name_tensor in zip(restored_trackables, restored_tensors):
        trackable.name = name_tensor

    registration.register_checkpoint_saver(
        name="MigratedSaver",
        predicate=lambda x: isinstance(x, RegisteredSaver),
        save_fn=save_fn,
        restore_fn=restore_fn,
    )

    before = NoRegisteredSaver("before")
    after = RegisteredSaver("after")
    before_ckpt_path = os.path.join(self.get_temp_dir(), "before_ckpt")
    util.Checkpoint(before).write(before_ckpt_path)

    after_ckpt = util.Checkpoint(after)
    after_ckpt_path = os.path.join(self.get_temp_dir(), "after_ckpt")
    after_ckpt.write(after_ckpt_path)

    # Try loading the pre-migrated checkpoint to the migrated object.
    after_ckpt.read(before_ckpt_path)
    self.assertEqual(b"before", self.evaluate(after.name))


if __name__ == "__main__":
  test.main()
