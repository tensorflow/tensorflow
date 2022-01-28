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
"""Tests for type_dispatch."""

from typing import Optional

from tensorflow.core.function.polymorphism import type_dispatch
from tensorflow.python.platform import test
from tensorflow.python.types import trace


class MockShape(trace.TraceType):

  def __init__(self, *shape: Optional[int]):
    self.shape = shape

  def is_subtype_of(self, other: "MockShape") ->bool:
    if len(self.shape) != len(other.shape):
      return False

    if any(s is not None and s != o for s, o in zip(self.shape, other.shape)):
      return False

    return True

  def most_specific_common_supertype(self, _):
    raise NotImplementedError

  def __str__(self):
    return str(self.shape)

  def __repr__(self):
    return str(self)

  def __hash__(self) -> int:
    return hash(self.shape)

  def __eq__(self, other: "MockShape") -> bool:
    return self.shape == other.shape


class TypeDispatchTableTest(test.TestCase):

  def testVertical(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, None, 1))
    table.add_target(MockShape(None, 1, 1))
    table.add_target(MockShape(1, 1, 1))
    self.assertEqual(
        table.all_targets(), [
            MockShape(None, None, None),
            MockShape(None, None, 1),
            MockShape(None, 1, 1),
            MockShape(1, 1, 1)
        ])

  def testHorizontal(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(1,))
    table.add_target(MockShape(1, 2))
    table.add_target(MockShape(1, 2, 3))
    self.assertEqual(
        table.all_targets(), [
            MockShape(1,),
            MockShape(1, 2),
            MockShape(1, 2, 3)
        ])

  def testDuplicateNodes(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None))
    table.add_target(MockShape(1, None))
    table.add_target(MockShape(None, 2))
    table.add_target(MockShape(None, None))
    self.assertEqual(
        table.all_targets(), [
            MockShape(None, None),
            MockShape(1, None),
            MockShape(None, 2)
        ])

  def testDeletion(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None))
    table.add_target(MockShape(None, 1))
    table.add_target(MockShape(None, 2))

    self.assertEqual(
        table.all_targets(), [
            MockShape(None, None),
            MockShape(None, 1),
            MockShape(None, 2)
        ])

    table.delete(MockShape(None, 2))  # Should remove the target

    self.assertEqual(
        table.all_targets(), [
            MockShape(None, None),
            MockShape(None, 1),
        ])

    table.delete(MockShape(None, 2))  # Should have no effect

    self.assertEqual(
        table.all_targets(), [
            MockShape(None, None),
            MockShape(None, 1),
        ])

  def testContains(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, 1))
    table.add_target(MockShape(1, 1))
    table.add_target(MockShape(None, 2, 1))

    self.assertTrue(table.contains(MockShape(None, None, None)))
    self.assertTrue(table.contains(MockShape(None, 1)))
    self.assertTrue(table.contains(MockShape(1, 1)))
    self.assertTrue(table.contains(MockShape(None, 2, 1)))

    self.assertFalse(table.contains(MockShape(None, None, 1)))
    self.assertFalse(table.contains(MockShape(1, None)))
    self.assertFalse(table.contains(MockShape(1, 2)))
    self.assertFalse(table.contains(MockShape(None, 2, None)))

  def testDispatchExactMatches(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    table.add_target(MockShape(None, 2, 2))

    self.assertEqual(
        table.dispatch(MockShape(None, 1, 2)), MockShape(None, 1, 2))
    self.assertEqual(
        table.dispatch(MockShape(None, 1, None)), MockShape(None, 1, None))
    self.assertEqual(
        table.dispatch(MockShape(None, None, None)),
        MockShape(None, None, None))
    self.assertEqual(
        table.dispatch(MockShape(None, 2, 2)), MockShape(None, 2, 2))

  def testDispatchMoreSpecific(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    table.add_target(MockShape(None, 2, 2))

    self.assertEqual(table.dispatch(MockShape(1, 1, 2)), MockShape(None, 1, 2))
    self.assertEqual(
        table.dispatch(MockShape(1, 1, 3)), MockShape(None, 1, None))
    self.assertEqual(
        table.dispatch(MockShape(1, 3, 3)), MockShape(None, None, None))
    self.assertEqual(table.dispatch(MockShape(1, 2, 2)), MockShape(None, 2, 2))

  def testDispatchNoMatches(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    table.add_target(MockShape(None, 2, 2))

    self.assertIsNone(table.dispatch(MockShape(1, 2)))
    self.assertIsNone(table.dispatch(MockShape(1, 2, 3)))
    self.assertIsNone(table.dispatch(MockShape(1, 2, 3, 4)))

  def testDispatchCachedAddUpdates(self):
    table = type_dispatch.TypeDispatchTable()

    table.add_target(MockShape(None, None, None))
    self.assertEqual(
        table.dispatch(MockShape(1, 1, 2)), MockShape(None, None, None))

    table.add_target(MockShape(None, 1, None))
    self.assertEqual(
        table.dispatch(MockShape(1, 1, 2)), MockShape(None, 1, None))

    table.add_target(MockShape(None, 1, 2))
    self.assertEqual(table.dispatch(MockShape(1, 1, 2)), MockShape(None, 1, 2))

    table.add_target(MockShape(1, 1, 2))
    self.assertEqual(table.dispatch(MockShape(1, 1, 2)), MockShape(1, 1, 2))

  def testDispatchCachedDeleteUpdates(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(MockShape(None, None, None))
    table.add_target(MockShape(None, 1, None))
    table.add_target(MockShape(None, 1, 2))
    table.add_target(MockShape(1, 1, 2))

    self.assertEqual(table.dispatch(MockShape(1, 1, 2)), MockShape(1, 1, 2))

    table.delete(MockShape(1, 1, 2))
    self.assertEqual(table.dispatch(MockShape(1, 1, 2)), MockShape(None, 1, 2))

    table.delete(MockShape(None, 1, 2))
    self.assertEqual(
        table.dispatch(MockShape(1, 1, 2)), MockShape(None, 1, None))

    table.delete(MockShape(None, 1, None))
    self.assertEqual(
        table.dispatch(MockShape(1, 1, 2)), MockShape(None, None, None))

  def testDispatchCacheOrderingDeterminism(self):
    table_1 = type_dispatch.TypeDispatchTable()
    table_1.add_target(MockShape(1, None, None))
    table_1.add_target(MockShape(None, 2, None))
    table_1.add_target(MockShape(None, None, 3))

    table_2 = type_dispatch.TypeDispatchTable()
    table_2.add_target(MockShape(None, 2, None))
    table_2.add_target(MockShape(1, None, None))
    table_2.add_target(MockShape(None, None, 3))

    table_3 = type_dispatch.TypeDispatchTable()
    table_3.add_target(MockShape(None, None, 3))
    table_3.add_target(MockShape(1, None, None))
    table_3.add_target(MockShape(None, 2, None))

    # table_1, table_2, table_3 have the same targets
    self.assertEqual(set(table_1.all_targets()), set(table_2.all_targets()))
    self.assertEqual(set(table_2.all_targets()), set(table_3.all_targets()))

    # But they dispatch to the first target they find which does not have any
    # more specific viable target.
    shape = MockShape(1, 2, 3)
    self.assertEqual(table_1.dispatch(shape), MockShape(1, None, None))
    self.assertEqual(table_2.dispatch(shape), MockShape(None, 2, None))
    self.assertEqual(table_3.dispatch(shape), MockShape(None, None, 3))

if __name__ == "__main__":
  test.main()
