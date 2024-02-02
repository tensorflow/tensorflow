# Copyright 2023 The OpenXLA Authors.
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
import collections

from absl.testing import absltest

from xla.python import xla_client

pytree = xla_client._xla.pytree


ExampleType = collections.namedtuple("ExampleType", "field0 field1")

registry = pytree.PyTreeRegistry()


class ExampleType2:

  def __init__(self, field0, field1):
    self.field0 = field0
    self.field1 = field1

  def to_iterable(self):
    return [self.field0, self.field1], (None,)


def from_iterable(state, values):
  del state
  return ExampleType2(field0=values[0], field1=values[1])


registry.register_node(ExampleType2, ExampleType2.to_iterable, from_iterable)


class PyTreeTest(absltest.TestCase):

  def roundtrip(self, example):
    original = registry.flatten(example)[1]
    self.assertEqual(
        pytree.PyTreeDef.deserialize_using_proto(
            registry, original.serialize_using_proto()
        ),
        original,
    )

  def testSerializeDeserializeNoPickle(self):
    o = object()
    self.roundtrip(({"a": o, "b": o}, [o, (o, o), None]))

  def testSerializeWithFallback(self):
    o = object()
    with self.assertRaises(ValueError):
      self.roundtrip({"a": ExampleType(field0=o, field1=o)})

  def testRegisteredType(self):
    o = object()
    with self.assertRaises(ValueError):
      self.roundtrip({"a": ExampleType2(field0=o, field1=o)})

  def roundtrip_node_data(self, example):
    original = registry.flatten(example)[1]
    restored = pytree.PyTreeDef.make_from_node_data_and_children(
        registry, original.node_data(), original.children()
    )
    self.assertEqual(restored, original)

  def testRoundtripNodeData(self):
    o = object()
    self.roundtrip_node_data([o, o, o])
    self.roundtrip_node_data((o, o, o))
    self.roundtrip_node_data({"a": o, "b": o})
    self.roundtrip_node_data({22: o, 88: o})
    self.roundtrip_node_data(None)
    self.roundtrip_node_data(o)
    self.roundtrip_node_data(ExampleType(field0=o, field1=o))
    self.roundtrip_node_data(ExampleType2(field0=o, field1=o))

  def testCompose(self):
    x = registry.flatten(0)[1]
    y = registry.flatten((0, 0))[1]
    self.assertEqual((x.compose(y)).num_leaves, 2)


if __name__ == "__main__":
  absltest.main()
