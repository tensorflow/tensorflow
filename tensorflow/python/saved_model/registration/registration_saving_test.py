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
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save
from tensorflow.python.training.tracking import tracking
from tensorflow.python.training.tracking import util


@registration.register_serializable()
class Part(resource_variable_ops.ResourceVariable):

  def __init__(self, value):
    self._init_from_args(value)

  @classmethod
  def _deserialize_from_proto(cls, **kwargs):
    return cls([0, 0])


@registration.register_serializable()
class Stack(tracking.AutoTrackable):

  def __init__(self, parts=None):
    self.parts = parts

  @def_function.function(input_signature=[])
  def value(self):
    return array_ops.stack(self.parts)


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
    parts = array_ops.unstack(restored_tensor)
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
    class Module(tracking.AutoTrackable):

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
    class Module(tracking.AutoTrackable):

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
    class Module(tracking.AutoTrackable):

      def __init__(self, v=None):
        self.v = v if v is not None else variables.Variable(1.)

      def _deserialization_dependencies(self):
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


if __name__ == "__main__":
  test.main()
