# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for utilities working with arbitrarily nested structures."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

from tensorflow.python.util import nest


class NestTest(tf.test.TestCase):

  def testFlattenAndPack(self):
    structure = ((3, 4), 5, (6, 7, (9, 10), 8))
    flat = ["a", "b", "c", "d", "e", "f", "g", "h"]
    self.assertEqual(nest.flatten(structure), [3, 4, 5, 6, 7, 9, 10, 8])
    self.assertEqual(nest.pack_sequence_as(structure, flat),
                     (("a", "b"), "c", ("d", "e", ("f", "g"), "h")))
    point = collections.namedtuple("Point", ["x", "y"])
    structure = (point(x=4, y=2), ((point(x=1, y=0),),))
    flat = [4, 2, 1, 0]
    self.assertEqual(nest.flatten(structure), flat)
    restructured_from_flat = nest.pack_sequence_as(structure, flat)
    self.assertEqual(restructured_from_flat, structure)
    self.assertEqual(restructured_from_flat[0].x, 4)
    self.assertEqual(restructured_from_flat[0].y, 2)
    self.assertEqual(restructured_from_flat[1][0][0].x, 1)
    self.assertEqual(restructured_from_flat[1][0][0].y, 0)

    self.assertEqual([5], nest.flatten(5))
    self.assertEqual([np.array([5])], nest.flatten(np.array([5])))

    self.assertEqual("a", nest.pack_sequence_as(5, ["a"]))
    self.assertEqual(
        np.array([5]), nest.pack_sequence_as("scalar", [np.array([5])]))

    with self.assertRaisesRegexp(ValueError, "Structure is a scalar"):
      nest.pack_sequence_as("scalar", [4, 5])

    with self.assertRaisesRegexp(TypeError, "flat_sequence"):
      nest.pack_sequence_as([4, 5], "bad_sequence")

    with self.assertRaises(ValueError):
      nest.pack_sequence_as([5, 6, [7, 8]], ["a", "b", "c"])

  def testIsSequence(self):
    self.assertFalse(nest.is_sequence("1234"))
    self.assertTrue(nest.is_sequence([1, 3, [4, 5]]))
    self.assertTrue(nest.is_sequence(((7, 8), (5, 6))))
    self.assertTrue(nest.is_sequence([]))
    self.assertFalse(nest.is_sequence(set([1, 2])))
    ones = tf.ones([2, 3])
    self.assertFalse(nest.is_sequence(ones))
    self.assertFalse(nest.is_sequence(tf.tanh(ones)))
    self.assertFalse(nest.is_sequence(np.ones((4, 5))))

  def testFlattenDictItems(self):
    dictionary = {(4, 5, (6, 8)): ("a", "b", ("c", "d"))}
    flat = {4: "a", 5: "b", 6: "c", 8: "d"}
    self.assertEqual(nest.flatten_dict_items(dictionary), flat)

    with self.assertRaises(TypeError):
      nest.flatten_dict_items(4)

    bad_dictionary = {(4, 5, (4, 8)): ("a", "b", ("c", "d"))}
    with self.assertRaisesRegexp(ValueError, "not unique"):
      nest.flatten_dict_items(bad_dictionary)

    another_bad_dictionary = {(4, 5, (6, 8)): ("a", "b", ("c", ("d", "e")))}
    with self.assertRaisesRegexp(
        ValueError, "Key had [0-9]* elements, but value had [0-9]* elements"):
      nest.flatten_dict_items(another_bad_dictionary)

  def testAssertSameStructure(self):
    structure1 = (((1, 2), 3), 4, (5, 6))
    structure2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
    structure_different_num_elements = ("spam", "eggs")
    structure_different_nesting = (((1, 2), 3), 4, 5, (6,))
    nest.assert_same_structure(structure1, structure2)
    nest.assert_same_structure("abc", 1.0)
    nest.assert_same_structure("abc", np.array([0, 1]))
    nest.assert_same_structure("abc", tf.constant([0, 1]))

    with self.assertRaisesRegexp(
        ValueError, "don't have the same number of elements"):
      nest.assert_same_structure(structure1, structure_different_num_elements)

    with self.assertRaisesRegexp(
        ValueError, "don't have the same number of elements"):
      nest.assert_same_structure([0, 1], np.array([0, 1]))

    with self.assertRaisesRegexp(
        ValueError, "don't have the same number of elements"):
      nest.assert_same_structure(0, [0, 1])

    self.assertRaises(TypeError, nest.assert_same_structure, (0, 1), [0, 1])

    with self.assertRaisesRegexp(
        ValueError, "don't have the same nested structure"):
      nest.assert_same_structure(structure1, structure_different_nesting)

    named_type_0 = collections.namedtuple("named_0", ("a", "b"))
    named_type_1 = collections.namedtuple("named_1", ("a", "b"))
    self.assertRaises(TypeError, nest.assert_same_structure,
                      (0, 1), named_type_0("a", "b"))

    nest.assert_same_structure(named_type_0(3, 4), named_type_0("a", "b"))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      named_type_0(3, 4), named_type_1(3, 4))

    with self.assertRaisesRegexp(
        ValueError, "don't have the same nested structure"):
      nest.assert_same_structure(named_type_0(3, 4), named_type_0([3], 4))

    with self.assertRaisesRegexp(
        ValueError, "don't have the same nested structure"):
      nest.assert_same_structure([[3], 4], [3, [4]])


if __name__ == "__main__":
  tf.test.main()
