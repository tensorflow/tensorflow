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

import tempfile
from absl.testing import parameterized

from google.protobuf import wrappers_pb2
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import save
from tensorflow.python.training.tracking import tracking


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
    dict(testcase_name="ReloadThrice", cycles=3)
)
class SavedModelTest(test.TestCase, parameterized.TestCase):

  def test_save_and_load(self, cycles):

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

if __name__ == "__main__":
  test.main()
