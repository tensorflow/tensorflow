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

from tensorflow.core.function.polymorphism import function_type
from tensorflow.core.function.polymorphism import type_dispatch
from tensorflow.python.platform import test
from tensorflow.python.types import trace


class MockShape(trace.TraceType):

  def __init__(self, *shape: Optional[int]):
    self.shape = shape

  def is_subtype_of(self, other: "MockShape") -> bool:
    if len(self.shape) != len(other.shape):
      return False

    return all(o is None or s == o for s, o in zip(self.shape, other.shape))

  def most_specific_common_supertype(self, others):
    if any(len(other.shape) != len(self.shape) for other in others):
      return None

    dims = [
        dim if all(dim == other.shape[i]
                   for other in others) else None
        for i, dim in enumerate(self.shape)
    ]
    return MockShape(*dims)

  def placeholder_value(self, placeholder_context=None):
    raise NotImplementedError

  def __str__(self):
    return str(self.shape)

  def __repr__(self):
    return str(self)

  def __hash__(self) -> int:
    return hash(self.shape)

  def __eq__(self, other: "MockShape") -> bool:
    return self.shape == other.shape


def make_shape_function_type(*shape):
  return function_type.FunctionType([
      function_type.Parameter("x", function_type.Parameter.POSITIONAL_ONLY,
                              False, MockShape(*shape))
  ])


class TypeDispatchTableTest(test.TestCase):

  def testVertical(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, None, 1))
    table.add_target(make_shape_function_type(None, 1, 1))
    table.add_target(make_shape_function_type(1, 1, 1))
    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(None, None, None),
            make_shape_function_type(None, None, 1),
            make_shape_function_type(None, 1, 1),
            make_shape_function_type(1, 1, 1)
        ])

  def testHorizontal(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(1,))
    table.add_target(make_shape_function_type(1, 2))
    table.add_target(make_shape_function_type(1, 2, 3))
    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(1,),
            make_shape_function_type(1, 2),
            make_shape_function_type(1, 2, 3)
        ])

  def testDuplicateNodes(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None))
    table.add_target(make_shape_function_type(1, None))
    table.add_target(make_shape_function_type(None, 2))
    table.add_target(make_shape_function_type(None, None))
    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(None, None),
            make_shape_function_type(1, None),
            make_shape_function_type(None, 2)
        ])

  def testDeletion(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None))
    table.add_target(make_shape_function_type(None, 1))
    table.add_target(make_shape_function_type(None, 2))

    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(None, None),
            make_shape_function_type(None, 1),
            make_shape_function_type(None, 2)
        ])

    table.delete(make_shape_function_type(None, 2))  # Should remove the target

    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(None, None),
            make_shape_function_type(None, 1),
        ])

    table.delete(make_shape_function_type(None, 2))  # Should have no effect

    self.assertEqual(
        list(table.targets), [
            make_shape_function_type(None, None),
            make_shape_function_type(None, 1),
        ])

  def testContains(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, 1))
    table.add_target(make_shape_function_type(1, 1))
    table.add_target(make_shape_function_type(None, 2, 1))

    self.assertIn(make_shape_function_type(None, None, None), table.targets)
    self.assertIn(make_shape_function_type(None, 1), table.targets)
    self.assertIn(make_shape_function_type(1, 1), table.targets)
    self.assertIn(make_shape_function_type(None, 2, 1), table.targets)

    self.assertNotIn(make_shape_function_type(None, None, 1), table.targets)
    self.assertNotIn(make_shape_function_type(1, None), table.targets)
    self.assertNotIn(make_shape_function_type(1, 2), table.targets)
    self.assertNotIn(make_shape_function_type(None, 2, None), table.targets)

  def testDispatchExactMatches(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    table.add_target(make_shape_function_type(None, 2, 2))

    self.assertEqual(
        table.dispatch(make_shape_function_type(None, 1, 2)),
        make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(None, 1, None)),
        make_shape_function_type(None, 1, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(None, None, None)),
        make_shape_function_type(None, None, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(None, 2, 2)),
        make_shape_function_type(None, 2, 2))

  def testDispatchMoreSpecific(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    table.add_target(make_shape_function_type(None, 2, 2))

    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 3)),
        make_shape_function_type(None, 1, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 3, 3)),
        make_shape_function_type(None, None, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 2, 2)),
        make_shape_function_type(None, 2, 2))

  def testDispatchNoMatches(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    table.add_target(make_shape_function_type(None, 2, 2))

    self.assertIsNone(table.dispatch(make_shape_function_type(1, 2)))
    self.assertIsNone(table.dispatch(make_shape_function_type(1, 2, 3)))
    self.assertIsNone(table.dispatch(make_shape_function_type(1, 2, 3, 4)))

  def testDispatchCachedAddUpdates(self):
    table = type_dispatch.TypeDispatchTable()

    table.add_target(make_shape_function_type(None, None, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, None, None))

    table.add_target(make_shape_function_type(None, 1, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, 1, None))

    table.add_target(make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, 1, 2))

    table.add_target(make_shape_function_type(1, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(1, 1, 2))

  def testDispatchCachedDeleteUpdates(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    table.add_target(make_shape_function_type(1, 1, 2))

    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(1, 1, 2))

    table.delete(make_shape_function_type(1, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, 1, 2))

    table.delete(make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, 1, None))

    table.delete(make_shape_function_type(None, 1, None))
    self.assertEqual(
        table.dispatch(make_shape_function_type(1, 1, 2)),
        make_shape_function_type(None, None, None))

  def testDispatchCacheOrderingDeterminism(self):
    table_1 = type_dispatch.TypeDispatchTable()
    table_1.add_target(make_shape_function_type(1, None, None))
    table_1.add_target(make_shape_function_type(None, 2, None))
    table_1.add_target(make_shape_function_type(None, None, 3))

    table_2 = type_dispatch.TypeDispatchTable()
    table_2.add_target(make_shape_function_type(None, 2, None))
    table_2.add_target(make_shape_function_type(1, None, None))
    table_2.add_target(make_shape_function_type(None, None, 3))

    table_3 = type_dispatch.TypeDispatchTable()
    table_3.add_target(make_shape_function_type(None, None, 3))
    table_3.add_target(make_shape_function_type(1, None, None))
    table_3.add_target(make_shape_function_type(None, 2, None))

    # table_1, table_2, table_3 have the same targets
    self.assertEqual(set(table_1.targets), set(table_2.targets))
    self.assertEqual(set(table_2.targets), set(table_3.targets))

    # But they dispatch to the first target they find which does not have any
    # more specific viable target.
    shape = make_shape_function_type(1, 2, 3)
    self.assertEqual(
        table_1.dispatch(shape), make_shape_function_type(1, None, None))
    self.assertEqual(
        table_2.dispatch(shape), make_shape_function_type(None, 2, None))
    self.assertEqual(
        table_3.dispatch(shape), make_shape_function_type(None, None, 3))

  def testGeneralizedExisting(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, None, None))
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.try_generalizing_function_type(
            make_shape_function_type(None, 1, 3)),
        make_shape_function_type(None, None, None))

  def testGeneralizedNovel(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, 1, None))
    table.add_target(make_shape_function_type(None, 1, 2))
    self.assertEqual(
        table.try_generalizing_function_type(
            make_shape_function_type(None, 2, 3)),
        make_shape_function_type(None, None, None))

  def testGeneralizedUnknown(self):
    table = type_dispatch.TypeDispatchTable()
    table.add_target(make_shape_function_type(None, 1))
    table.add_target(make_shape_function_type(None, 2))
    table.add_target(make_shape_function_type(None, 3))
    self.assertEqual(
        table.try_generalizing_function_type(
            make_shape_function_type(None, 4, 3)),
        make_shape_function_type(None, 4, 3))


if __name__ == "__main__":
  test.main()
