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
from tensorflow.python.platform import test


class DefaultTypesTest(test.TestCase):

  def testGenericSupertypes(self):
    generic_a = default_types.Generic(1)
    generic_b = default_types.Generic(2)
    generic_c = default_types.Generic(1)

    self.assertEqual(generic_a, generic_a.most_specific_common_supertype([]))
    self.assertEqual(generic_a,
                     generic_a.most_specific_common_supertype([generic_a]))
    self.assertEqual(generic_a,
                     generic_a.most_specific_common_supertype([generic_c]))
    self.assertIsNone(generic_a.most_specific_common_supertype([generic_b]))

  def testListSupertype(self):

    class Supertypable(default_types.Generic):

      def most_specific_common_supertype(self, others):
        if not others:
          return self

        if self._object == 2 and isinstance(others[0]._object, int):
          return Supertypable(3)
        else:
          return None

    list_a = default_types.List(
        Supertypable(1), Supertypable(2), Supertypable(3))
    list_b = default_types.List(
        Supertypable(2), Supertypable(2), Supertypable(2))

    self.assertEqual(list_a, list_a.most_specific_common_supertype([]))
    self.assertIsNone(list_a.most_specific_common_supertype([list_b]))
    self.assertEqual(
        list_b.most_specific_common_supertype([list_a]),
        default_types.List(Supertypable(3), Supertypable(3), Supertypable(3)))

  def testTupleSupertype(self):

    class Supertypable(default_types.Generic):

      def most_specific_common_supertype(self, others):
        if not others:
          return self

        if self._object == 2 and isinstance(others[0]._object, int):
          return Supertypable(3)
        else:
          return None

    tuple_a = default_types.Tuple(
        Supertypable(1), Supertypable(2), Supertypable(3))
    tuple_b = default_types.Tuple(
        Supertypable(2), Supertypable(2), Supertypable(2))

    self.assertEqual(tuple_a, tuple_a.most_specific_common_supertype([]))
    self.assertIsNone(tuple_a.most_specific_common_supertype([tuple_b]))
    self.assertEqual(
        tuple_b.most_specific_common_supertype([tuple_a]),
        default_types.Tuple(Supertypable(3), Supertypable(3), Supertypable(3)))

  def testNamedTupleSupertype(self):

    class Supertypable(default_types.Generic):

      def most_specific_common_supertype(self, others):
        if not others:
          return self

        if self._object == 2 and isinstance(others[0]._object, int):
          return Supertypable(3)
        else:
          return None

    named_tuple_type = collections.namedtuple('MyNamedTuple', 'x y z')
    tuple_a = default_types.NamedTuple(
        named_tuple_type, (Supertypable(1), Supertypable(2), Supertypable(3)))
    tuple_b = default_types.NamedTuple(
        named_tuple_type, (Supertypable(2), Supertypable(2), Supertypable(2)))

    self.assertEqual(tuple_a, tuple_a.most_specific_common_supertype([]))
    self.assertIsNone(tuple_a.most_specific_common_supertype([tuple_b]))
    self.assertEqual(
        tuple_b.most_specific_common_supertype([tuple_a]),
        default_types.NamedTuple(
            named_tuple_type,
            (Supertypable(3), Supertypable(3), Supertypable(3))))

  def testDictTypeSubtype(self):

    class MockSubtypeOf2(default_types.Generic):

      def is_subtype_of(self, other):
        return other._object == 2

    dict_type = default_types.Dict
    dict_a = dict_type({
        'a': MockSubtypeOf2(1),
        'b': MockSubtypeOf2(1),
        'c': MockSubtypeOf2(1)
    })
    dict_b = dict_type({
        'a': MockSubtypeOf2(2),
        'b': MockSubtypeOf2(2),
        'c': MockSubtypeOf2(2)
    })
    dict_c = dict_type({'a': MockSubtypeOf2(1), 'b': MockSubtypeOf2(1)})

    self.assertTrue(dict_a.is_subtype_of(dict_b))
    self.assertFalse(dict_c.is_subtype_of(dict_b))
    self.assertFalse(dict_c.is_subtype_of(dict_a))

  def testDictTypeSupertype(self):

    class MockSupertypes2With3(default_types.Generic):

      def most_specific_common_supertype(self, others):
        if not others:
          return self

        if self._object == 2 and isinstance(others[0]._object, int):
          return MockSupertypes2With3(3)
        else:
          return None

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

    self.assertEqual(dict_a,
                     dict_a.most_specific_common_supertype([]))
    self.assertIsNone(dict_a.most_specific_common_supertype([dict_b]))
    self.assertEqual(
        dict_b.most_specific_common_supertype([dict_a]),
        dict_type({
            'a': MockSupertypes2With3(3),
            'b': MockSupertypes2With3(3),
            'c': MockSupertypes2With3(3)
        }))

  def testListTupleInequality(self):
    generic = default_types.Generic

    list_a = default_types.List(generic(1), generic(2), generic(3))
    list_b = default_types.List(generic(1), generic(2), generic(3))

    tuple_a = default_types.Tuple(generic(1), generic(2), generic(3))
    tuple_b = default_types.Tuple(generic(1), generic(2), generic(3))

    self.assertEqual(list_a, list_b)
    self.assertEqual(tuple_a, tuple_b)
    self.assertNotEqual(list_a, tuple_a)
    self.assertNotEqual(tuple_a, list_a)

  def testDictTypeEquality(self):
    dict_type = default_types.Dict
    generic = default_types.Generic

    dict_a = dict_type({generic(1): generic(2), generic(3): generic(4)})
    dict_b = dict_type({generic(1): generic(2)})
    dict_c = dict_type({generic(3): generic(4), generic(1): generic(2)})

    self.assertEqual(dict_a, dict_c)
    self.assertNotEqual(dict_a, dict_b)

  def testReferenceSubtype(self):

    class MockSubtypeOf2(default_types.Generic):

      def is_subtype_of(self, other):
        return other._object == 2

    original = default_types.Reference(MockSubtypeOf2(3), 1)
    clone = default_types.Reference(MockSubtypeOf2(3), 1)
    different_id = default_types.Reference(MockSubtypeOf2(3), 2)
    supertype = default_types.Reference(MockSubtypeOf2(2), 1)
    different_type = default_types.Generic(1)

    self.assertEqual(original, clone)
    self.assertFalse(original.is_subtype_of(different_id))
    self.assertTrue(original.is_subtype_of(supertype))
    self.assertFalse(supertype.is_subtype_of(original))
    self.assertFalse(original.is_subtype_of(different_type))

  def testReferenceSupertype(self):

    class Mock2AsTopType(default_types.Generic):

      def most_specific_common_supertype(self, types):
        if not all(isinstance(other, Mock2AsTopType) for other in types):
          return None
        return self if all(self._object == other._object
                           for other in types) else Mock2AsTopType(2)

    original = default_types.Reference(Mock2AsTopType(3), 1)
    clone = default_types.Reference(Mock2AsTopType(3), 1)
    different_id = default_types.Reference(Mock2AsTopType(3), 2)
    supertype = default_types.Reference(Mock2AsTopType(2), 1)
    different_type = default_types.Generic(1)

    self.assertEqual(supertype.most_specific_common_supertype([]), supertype)
    self.assertEqual(original.most_specific_common_supertype([clone]), original)
    self.assertIsNone(original.most_specific_common_supertype([different_id]))
    self.assertIsNone(original.most_specific_common_supertype([different_type]))

if __name__ == '__main__':
  test.main()
