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

from tensorflow.core.function.trace_type import default_types
from tensorflow.python.platform import test


class DefaultTypesTest(test.TestCase):

  def testOrderedCollectionTypeEquality(self):
    collection = default_types.OrderedCollection
    generic = default_types.Generic
    collection_a = collection(generic(1), generic(2), generic(3))
    collection_b = collection(generic(1), generic(2), generic(1))
    collection_c = collection(generic(1), generic(2), generic(3))

    self.assertNotEqual(collection_a, collection_b)
    self.assertEqual(collection_a, collection_c)
    self.assertEqual(hash(collection_a), hash(collection_c))

  def testOrderedCollectionTypeSubtype(self):

    class Subtypable(default_types.Generic):

      def is_subtype_of(self, other):
        return self._object == 2 or other._object == 3

    collection = default_types.OrderedCollection
    collection_a = collection(Subtypable(1), Subtypable(2), Subtypable(3))
    collection_b = collection(Subtypable(2), Subtypable(1), Subtypable(2))
    collection_c = collection(Subtypable(1), Subtypable(3), Subtypable(3))

    self.assertTrue(collection_b.is_subtype_of(collection_c))
    self.assertFalse(collection_a.is_subtype_of(collection_b))
    self.assertFalse(collection_c.is_subtype_of(collection_a))

  def testOrderedCollectionTypeSupertype(self):

    class Supertypable(default_types.Generic):

      def most_specific_common_supertype(self, others):
        if self._object == 2 and isinstance(others[0]._object, int):
          return Supertypable(3)
        else:
          return None

    collection = default_types.OrderedCollection
    collection_a = collection(Supertypable(1), Supertypable(2), Supertypable(3))
    collection_b = collection(Supertypable(2), Supertypable(2), Supertypable(2))

    self.assertIsNone(
        collection_a.most_specific_common_supertype([collection_b]))
    self.assertEqual(
        collection_b.most_specific_common_supertype([collection_a]),
        collection(Supertypable(3), Supertypable(3), Supertypable(3)))

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


if __name__ == '__main__':
  test.main()
