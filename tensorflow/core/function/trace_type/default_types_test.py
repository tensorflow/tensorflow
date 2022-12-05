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
"""Tests for default_types."""

import collections

from tensorflow.core.function.trace_type import default_types
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.platform import test
from tensorflow.python.types import trace


class MockSupertypes2With3(trace.TraceType):

  def __init__(self, obj):
    self._object = obj

  def is_subtype_of(self, other):
    return self._object == 2 and other._object == 3

  def most_specific_common_supertype(self, others):
    if not others:
      return self

    if self._object == 2 and isinstance(others[0]._object, int):
      return MockSupertypes2With3(3)
    else:
      return None

  def __eq__(self, other) -> bool:
    return isinstance(other, type(self)) and self._object == other._object

  def __hash__(self) -> int:
    return self._object_hash


class Mock2AsTopType(MockSupertypes2With3):

  def is_subtype_of(self, other):
    return other._object == 2

  def most_specific_common_supertype(self, others):
    if not all(isinstance(other, Mock2AsTopType) for other in others):
      return None
    return self if all(self._object == other._object
                       for other in others) else Mock2AsTopType(2)


class TestAttr:
  """Helps test attrs collections."""

  def __init__(self, name):
    self.name = name


class TestAttrsClass:
  """Helps test attrs collections."""

  __attrs_attrs__ = (TestAttr('a'), TestAttr('b'))

  def __init__(self, a, b):
    self.a = a
    self.b = b

  def __eq__(self, other):
    return isinstance(
        other, TestAttrsClass) and self.a == other.a and self.b == other.b


class DefaultTypesTest(test.TestCase):

  def testLiteralSupertypes(self):
    literal_a = default_types.Literal(1)
    literal_b = default_types.Literal(2)
    literal_c = default_types.Literal(1)

    self.assertEqual(literal_a, literal_a.most_specific_common_supertype([]))
    self.assertEqual(literal_a,
                     literal_a.most_specific_common_supertype([literal_a]))
    self.assertEqual(literal_a,
                     literal_a.most_specific_common_supertype([literal_c]))
    self.assertIsNone(literal_a.most_specific_common_supertype([literal_b]))

  def testLiteralSerialization(self):
    literal_bool = default_types.Literal(True)
    literal_int = default_types.Literal(1)
    literal_float = default_types.Literal(1.2)
    literal_str = default_types.Literal('a')
    literal_none = default_types.Literal(None)

    self.assertEqual(
        serialization.deserialize(serialization.serialize(literal_bool)),
        literal_bool)
    self.assertEqual(
        serialization.deserialize(serialization.serialize(literal_int)),
        literal_int)
    self.assertEqual(
        serialization.deserialize(serialization.serialize(literal_float)),
        literal_float)
    self.assertEqual(
        serialization.deserialize(serialization.serialize(literal_str)),
        literal_str)
    self.assertEqual(
        serialization.deserialize(serialization.serialize(literal_none)),
        literal_none)

  def testListSupertype(self):
    list_a = default_types.List(
        MockSupertypes2With3(1), MockSupertypes2With3(2),
        MockSupertypes2With3(3))
    list_b = default_types.List(
        MockSupertypes2With3(2), MockSupertypes2With3(2),
        MockSupertypes2With3(2))

    self.assertEqual(list_a, list_a.most_specific_common_supertype([]))
    self.assertIsNone(list_a.most_specific_common_supertype([list_b]))
    self.assertEqual(
        list_b.most_specific_common_supertype([list_a]),
        default_types.List(
            MockSupertypes2With3(3), MockSupertypes2With3(3),
            MockSupertypes2With3(3)))

  def testListSerialization(self):
    list_original = default_types.List(
        default_types.Literal(1), default_types.Literal(2),
        default_types.Literal(3))

    self.assertEqual(
        serialization.deserialize(serialization.serialize(list_original)),
        list_original)

  def testTupleSupertype(self):
    tuple_a = default_types.Tuple(
        MockSupertypes2With3(1), MockSupertypes2With3(2),
        MockSupertypes2With3(3))
    tuple_b = default_types.Tuple(
        MockSupertypes2With3(2), MockSupertypes2With3(2),
        MockSupertypes2With3(2))

    self.assertEqual(tuple_a, tuple_a.most_specific_common_supertype([]))
    self.assertIsNone(tuple_a.most_specific_common_supertype([tuple_b]))
    self.assertEqual(
        tuple_b.most_specific_common_supertype([tuple_a]),
        default_types.Tuple(
            MockSupertypes2With3(3), MockSupertypes2With3(3),
            MockSupertypes2With3(3)))

  def testTupleSerialization(self):
    tuple_original = default_types.Tuple(
        default_types.Literal(1), default_types.Literal(2),
        default_types.Literal(3))

    self.assertEqual(
        serialization.deserialize(serialization.serialize(tuple_original)),
        tuple_original)

  def testNamedTupleSupertype(self):
    named_tuple_type = collections.namedtuple('MyNamedTuple', 'x y z')
    tuple_a = default_types.NamedTuple.from_type_and_attributes(
        named_tuple_type, (MockSupertypes2With3(1), MockSupertypes2With3(2),
                           MockSupertypes2With3(3)))
    tuple_b = default_types.NamedTuple.from_type_and_attributes(
        named_tuple_type, (MockSupertypes2With3(2), MockSupertypes2With3(2),
                           MockSupertypes2With3(2)))

    self.assertEqual(tuple_a, tuple_a.most_specific_common_supertype([]))
    self.assertIsNone(tuple_a.most_specific_common_supertype([tuple_b]))
    self.assertEqual(
        tuple_b.most_specific_common_supertype([tuple_a]),
        default_types.NamedTuple.from_type_and_attributes(
            named_tuple_type, (MockSupertypes2With3(3), MockSupertypes2With3(3),
                               MockSupertypes2With3(3))))

  def testAttrsSupertype(self):
    attrs_a = default_types.Attrs.from_type_and_attributes(
        TestAttrsClass, (MockSupertypes2With3(1), MockSupertypes2With3(2),
                         MockSupertypes2With3(3)))
    attrs_b = default_types.Attrs.from_type_and_attributes(
        TestAttrsClass, (MockSupertypes2With3(2), MockSupertypes2With3(2),
                         MockSupertypes2With3(2)))

    self.assertEqual(attrs_a, attrs_a.most_specific_common_supertype([]))
    self.assertIsNone(attrs_a.most_specific_common_supertype([attrs_b]))
    self.assertEqual(
        attrs_b.most_specific_common_supertype([attrs_a]),
        default_types.Attrs.from_type_and_attributes(
            TestAttrsClass, (MockSupertypes2With3(3), MockSupertypes2With3(3),
                             MockSupertypes2With3(3))))

  def testDictTypeSubtype(self):
    dict_type = default_types.Dict
    dict_a = dict_type({
        'a': Mock2AsTopType(1),
        'b': Mock2AsTopType(1),
        'c': Mock2AsTopType(1)
    })
    dict_b = dict_type({
        'a': Mock2AsTopType(2),
        'b': Mock2AsTopType(2),
        'c': Mock2AsTopType(2)
    })
    dict_c = dict_type({'a': Mock2AsTopType(1), 'b': Mock2AsTopType(1)})

    self.assertTrue(dict_a.is_subtype_of(dict_b))
    self.assertFalse(dict_c.is_subtype_of(dict_b))
    self.assertFalse(dict_c.is_subtype_of(dict_a))

  def testDictTypeSupertype(self):
    dict_type = default_types.Dict
    dict_a = dict_type({
        'a': MockSupertypes2With3(1),
        'b': MockSupertypes2With3(2),
        'c': MockSupertypes2With3(3)
    })
    dict_b = dict_type({
        'a': MockSupertypes2With3(2),
        'b': MockSupertypes2With3(2),
        'c': MockSupertypes2With3(2)
    })

    self.assertEqual(dict_a, dict_a.most_specific_common_supertype([]))
    self.assertIsNone(dict_a.most_specific_common_supertype([dict_b]))
    self.assertEqual(
        dict_b.most_specific_common_supertype([dict_a]),
        dict_type({
            'a': MockSupertypes2With3(3),
            'b': MockSupertypes2With3(3),
            'c': MockSupertypes2With3(3)
        }))

  def testDictSerialization(self):
    dict_original = default_types.Dict({
        'a': default_types.Literal(1),
        'b': default_types.Literal(2),
        'c': default_types.Literal(3)
    })

    self.assertEqual(
        serialization.deserialize(serialization.serialize(dict_original)),
        dict_original)

  def testListTupleInequality(self):
    literal = default_types.Literal

    list_a = default_types.List(literal(1), literal(2), literal(3))
    list_b = default_types.List(literal(1), literal(2), literal(3))

    tuple_a = default_types.Tuple(literal(1), literal(2), literal(3))
    tuple_b = default_types.Tuple(literal(1), literal(2), literal(3))

    self.assertEqual(list_a, list_b)
    self.assertEqual(tuple_a, tuple_b)
    self.assertNotEqual(list_a, tuple_a)
    self.assertNotEqual(tuple_a, list_a)

  def testDictTypeEquality(self):
    dict_type = default_types.Dict
    literal = default_types.Literal

    dict_a = dict_type({literal(1): literal(2), literal(3): literal(4)})
    dict_b = dict_type({literal(1): literal(2)})
    dict_c = dict_type({literal(3): literal(4), literal(1): literal(2)})

    self.assertEqual(dict_a, dict_c)
    self.assertNotEqual(dict_a, dict_b)


if __name__ == '__main__':
  test.main()
