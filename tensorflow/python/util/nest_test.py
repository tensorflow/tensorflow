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
import time

from absl.testing import parameterized
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class _CustomMapping(collections.Mapping):

  def __init__(self, *args, **kwargs):
    self._wrapped = dict(*args, **kwargs)

  def __getitem__(self, key):
    return self._wrapped[key]

  def __iter__(self):
    return iter(self._wrapped)

  def __len__(self):
    return len(self._wrapped)


class NestTest(parameterized.TestCase, test.TestCase):

  PointXY = collections.namedtuple("Point", ["x", "y"])  # pylint: disable=invalid-name

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testFlattenAndPack(self):
    structure = ((3, 4), 5, (6, 7, (9, 10), 8))
    flat = ["a", "b", "c", "d", "e", "f", "g", "h"]
    self.assertEqual(nest.flatten(structure), [3, 4, 5, 6, 7, 9, 10, 8])
    self.assertEqual(
        nest.pack_sequence_as(structure, flat), (("a", "b"), "c",
                                                 ("d", "e", ("f", "g"), "h")))
    structure = (NestTest.PointXY(x=4, y=2),
                 ((NestTest.PointXY(x=1, y=0),),))
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

  @parameterized.parameters({"mapping_type": collections.OrderedDict},
                            {"mapping_type": _CustomMapping})
  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testFlattenDictOrder(self, mapping_type):
    """`flatten` orders dicts by key, including OrderedDicts."""
    ordered = mapping_type([("d", 3), ("b", 1), ("a", 0), ("c", 2)])
    plain = {"d": 3, "b": 1, "a": 0, "c": 2}
    ordered_flat = nest.flatten(ordered)
    plain_flat = nest.flatten(plain)
    self.assertEqual([0, 1, 2, 3], ordered_flat)
    self.assertEqual([0, 1, 2, 3], plain_flat)

  @parameterized.parameters({"mapping_type": collections.OrderedDict},
                            {"mapping_type": _CustomMapping})
  def testPackDictOrder(self, mapping_type):
    """Packing orders dicts by key, including OrderedDicts."""
    custom = mapping_type([("d", 0), ("b", 0), ("a", 0), ("c", 0)])
    plain = {"d": 0, "b": 0, "a": 0, "c": 0}
    seq = [0, 1, 2, 3]
    custom_reconstruction = nest.pack_sequence_as(custom, seq)
    plain_reconstruction = nest.pack_sequence_as(plain, seq)
    self.assertIsInstance(custom_reconstruction, mapping_type)
    self.assertIsInstance(plain_reconstruction, dict)
    self.assertEqual(
        mapping_type([("d", 3), ("b", 1), ("a", 0), ("c", 2)]),
        custom_reconstruction)
    self.assertEqual({"d": 3, "b": 1, "a": 0, "c": 2}, plain_reconstruction)

  Abc = collections.namedtuple("A", ("b", "c"))  # pylint: disable=invalid-name

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testFlattenAndPack_withDicts(self):
    # A nice messy mix of tuples, lists, dicts, and `OrderedDict`s.
    mess = [
        "z",
        NestTest.Abc(3, 4), {
            "d": _CustomMapping({
                41: 4
            }),
            "c": [
                1,
                collections.OrderedDict([
                    ("b", 3),
                    ("a", 2),
                ]),
            ],
            "b": 5
        }, 17
    ]

    flattened = nest.flatten(mess)
    self.assertEqual(flattened, ["z", 3, 4, 5, 1, 2, 3, 4, 17])

    structure_of_mess = [
        14,
        NestTest.Abc("a", True),
        {
            "d": _CustomMapping({
                41: 42
            }),
            "c": [
                0,
                collections.OrderedDict([
                    ("b", 9),
                    ("a", 8),
                ]),
            ],
            "b": 3
        },
        "hi everybody",
    ]

    unflattened = nest.pack_sequence_as(structure_of_mess, flattened)
    self.assertEqual(unflattened, mess)

    # Check also that the OrderedDict was created, with the correct key order.
    unflattened_ordered_dict = unflattened[2]["c"][1]
    self.assertIsInstance(unflattened_ordered_dict, collections.OrderedDict)
    self.assertEqual(list(unflattened_ordered_dict.keys()), ["b", "a"])

    unflattened_custom_mapping = unflattened[2]["d"]
    self.assertIsInstance(unflattened_custom_mapping, _CustomMapping)
    self.assertEqual(list(unflattened_custom_mapping.keys()), [41])

  def testFlatten_numpyIsNotFlattened(self):
    structure = np.array([1, 2, 3])
    flattened = nest.flatten(structure)
    self.assertEqual(len(flattened), 1)

  def testFlatten_stringIsNotFlattened(self):
    structure = "lots of letters"
    flattened = nest.flatten(structure)
    self.assertEqual(len(flattened), 1)
    unflattened = nest.pack_sequence_as("goodbye", flattened)
    self.assertEqual(structure, unflattened)

  def testPackSequenceAs_notIterableError(self):
    with self.assertRaisesRegexp(TypeError,
                                 "flat_sequence must be a sequence"):
      nest.pack_sequence_as("hi", "bye")

  def testPackSequenceAs_wrongLengthsError(self):
    with self.assertRaisesRegexp(
        ValueError,
        "Structure had 2 elements, but flat_sequence had 3 elements."):
      nest.pack_sequence_as(["hello", "world"],
                            ["and", "goodbye", "again"])

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testIsSequence(self):
    self.assertFalse(nest.is_sequence("1234"))
    self.assertTrue(nest.is_sequence([1, 3, [4, 5]]))
    self.assertTrue(nest.is_sequence(((7, 8), (5, 6))))
    self.assertTrue(nest.is_sequence([]))
    self.assertTrue(nest.is_sequence({"a": 1, "b": 2}))
    self.assertFalse(nest.is_sequence(set([1, 2])))
    ones = array_ops.ones([2, 3])
    self.assertFalse(nest.is_sequence(ones))
    self.assertFalse(nest.is_sequence(math_ops.tanh(ones)))
    self.assertFalse(nest.is_sequence(np.ones((4, 5))))

  @parameterized.parameters({"mapping_type": _CustomMapping},
                            {"mapping_type": dict})
  def testFlattenDictItems(self, mapping_type):
    dictionary = mapping_type({(4, 5, (6, 8)): ("a", "b", ("c", "d"))})
    flat = {4: "a", 5: "b", 6: "c", 8: "d"}
    self.assertEqual(nest.flatten_dict_items(dictionary), flat)

    with self.assertRaises(TypeError):
      nest.flatten_dict_items(4)

    bad_dictionary = mapping_type({(4, 5, (4, 8)): ("a", "b", ("c", "d"))})
    with self.assertRaisesRegexp(ValueError, "not unique"):
      nest.flatten_dict_items(bad_dictionary)

    another_bad_dictionary = mapping_type({
        (4, 5, (6, 8)): ("a", "b", ("c", ("d", "e")))
    })
    with self.assertRaisesRegexp(
        ValueError, "Key had [0-9]* elements, but value had [0-9]* elements"):
      nest.flatten_dict_items(another_bad_dictionary)

  # pylint does not correctly recognize these as class names and
  # suggests to use variable style under_score naming.
  # pylint: disable=invalid-name
  Named0ab = collections.namedtuple("named_0", ("a", "b"))
  Named1ab = collections.namedtuple("named_1", ("a", "b"))
  SameNameab = collections.namedtuple("same_name", ("a", "b"))
  SameNameab2 = collections.namedtuple("same_name", ("a", "b"))
  SameNamexy = collections.namedtuple("same_name", ("x", "y"))
  SameName1xy = collections.namedtuple("same_name_1", ("x", "y"))
  SameName1xy2 = collections.namedtuple("same_name_1", ("x", "y"))
  NotSameName = collections.namedtuple("not_same_name", ("a", "b"))
  # pylint: enable=invalid-name

  class SameNamedType1(SameNameab):
    pass

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testAssertSameStructure(self):
    structure1 = (((1, 2), 3), 4, (5, 6))
    structure2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
    structure_different_num_elements = ("spam", "eggs")
    structure_different_nesting = (((1, 2), 3), 4, 5, (6,))
    nest.assert_same_structure(structure1, structure2)
    nest.assert_same_structure("abc", 1.0)
    nest.assert_same_structure("abc", np.array([0, 1]))
    nest.assert_same_structure("abc", constant_op.constant([0, 1]))

    with self.assertRaisesRegexp(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         "More specifically: Substructure "
         r'"type=tuple str=\(\(1, 2\), 3\)" is a sequence, while '
         'substructure "type=str str=spam" is not')):
      nest.assert_same_structure(structure1, structure_different_num_elements)

    with self.assertRaisesRegexp(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         r'More specifically: Substructure "type=list str=\[0, 1\]" '
         r'is a sequence, while substructure "type=ndarray str=\[0 1\]" '
         "is not")):
      nest.assert_same_structure([0, 1], np.array([0, 1]))

    with self.assertRaisesRegexp(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         r'More specifically: Substructure "type=list str=\[0, 1\]" '
         'is a sequence, while substructure "type=int str=0" '
         "is not")):
      nest.assert_same_structure(0, [0, 1])

    self.assertRaises(TypeError, nest.assert_same_structure, (0, 1), [0, 1])

    with self.assertRaisesRegexp(
        ValueError,
        ("don't have the same nested structure\\.\n\n"
         "First structure: .*?\n\nSecond structure: ")):
      nest.assert_same_structure(structure1, structure_different_nesting)

    self.assertRaises(TypeError, nest.assert_same_structure, (0, 1),
                      NestTest.Named0ab("a", "b"))

    nest.assert_same_structure(NestTest.Named0ab(3, 4),
                               NestTest.Named0ab("a", "b"))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      NestTest.Named0ab(3, 4), NestTest.Named1ab(3, 4))

    with self.assertRaisesRegexp(
        ValueError,
        ("don't have the same nested structure\\.\n\n"
         "First structure: .*?\n\nSecond structure: ")):
      nest.assert_same_structure(NestTest.Named0ab(3, 4),
                                 NestTest.Named0ab([3], 4))

    with self.assertRaisesRegexp(
        ValueError,
        ("don't have the same nested structure\\.\n\n"
         "First structure: .*?\n\nSecond structure: ")):
      nest.assert_same_structure([[3], 4], [3, [4]])

    structure1_list = [[[1, 2], 3], 4, [5, 6]]
    with self.assertRaisesRegexp(TypeError,
                                 "don't have the same sequence type"):
      nest.assert_same_structure(structure1, structure1_list)
    nest.assert_same_structure(structure1, structure2, check_types=False)
    nest.assert_same_structure(structure1, structure1_list, check_types=False)

    with self.assertRaisesRegexp(ValueError,
                                 "don't have the same set of keys"):
      nest.assert_same_structure({"a": 1}, {"b": 1})

    nest.assert_same_structure(NestTest.SameNameab(0, 1),
                               NestTest.SameNameab2(2, 3))

    # This assertion is expected to pass: two namedtuples with the same
    # name and field names are considered to be identical.
    nest.assert_same_structure(
        NestTest.SameNameab(NestTest.SameName1xy(0, 1), 2),
        NestTest.SameNameab2(NestTest.SameName1xy2(2, 3), 4))

    expected_message = "The two structures don't have the same.*"
    with self.assertRaisesRegexp(ValueError, expected_message):
      nest.assert_same_structure(
          NestTest.SameNameab(0, NestTest.SameNameab2(1, 2)),
          NestTest.SameNameab2(NestTest.SameNameab(0, 1), 2))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      NestTest.SameNameab(0, 1), NestTest.NotSameName(2, 3))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      NestTest.SameNameab(0, 1), NestTest.SameNamexy(2, 3))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      NestTest.SameNameab(0, 1), NestTest.SameNamedType1(2, 3))

  EmptyNT = collections.namedtuple("empty_nt", "")  # pylint: disable=invalid-name

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testMapStructure(self):
    structure1 = (((1, 2), 3), 4, (5, 6))
    structure2 = (((7, 8), 9), 10, (11, 12))
    structure1_plus1 = nest.map_structure(lambda x: x + 1, structure1)
    nest.assert_same_structure(structure1, structure1_plus1)
    self.assertAllEqual(
        [2, 3, 4, 5, 6, 7],
        nest.flatten(structure1_plus1))
    structure1_plus_structure2 = nest.map_structure(
        lambda x, y: x + y, structure1, structure2)
    self.assertEqual(
        (((1 + 7, 2 + 8), 3 + 9), 4 + 10, (5 + 11, 6 + 12)),
        structure1_plus_structure2)

    self.assertEqual(3, nest.map_structure(lambda x: x - 1, 4))

    self.assertEqual(7, nest.map_structure(lambda x, y: x + y, 3, 4))

    # Empty structures
    self.assertEqual((), nest.map_structure(lambda x: x + 1, ()))
    self.assertEqual([], nest.map_structure(lambda x: x + 1, []))
    self.assertEqual({}, nest.map_structure(lambda x: x + 1, {}))
    self.assertEqual(NestTest.EmptyNT(), nest.map_structure(lambda x: x + 1,
                                                            NestTest.EmptyNT()))

    # This is checking actual equality of types, empty list != empty tuple
    self.assertNotEqual((), nest.map_structure(lambda x: x + 1, []))

    with self.assertRaisesRegexp(TypeError, "callable"):
      nest.map_structure("bad", structure1_plus1)

    with self.assertRaisesRegexp(ValueError, "at least one structure"):
      nest.map_structure(lambda x: x)

    with self.assertRaisesRegexp(ValueError, "same number of elements"):
      nest.map_structure(lambda x, y: None, (3, 4), (3, 4, 5))

    with self.assertRaisesRegexp(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, 3, (3,))

    with self.assertRaisesRegexp(TypeError, "same sequence type"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), [(3, 4), 5])

    with self.assertRaisesRegexp(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)))

    structure1_list = [[[1, 2], 3], 4, [5, 6]]
    with self.assertRaisesRegexp(TypeError, "same sequence type"):
      nest.map_structure(lambda x, y: None, structure1, structure1_list)

    nest.map_structure(lambda x, y: None, structure1, structure1_list,
                       check_types=False)

    with self.assertRaisesRegexp(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)),
                         check_types=False)

    with self.assertRaisesRegexp(ValueError, "Only valid keyword argument"):
      nest.map_structure(lambda x: None, structure1, foo="a")

    with self.assertRaisesRegexp(ValueError, "Only valid keyword argument"):
      nest.map_structure(lambda x: None, structure1, check_types=False, foo="a")

  ABTuple = collections.namedtuple("ab_tuple", "a, b")  # pylint: disable=invalid-name

  @test_util.assert_no_new_pyobjects_executing_eagerly
  def testMapStructureWithStrings(self):
    inp_a = NestTest.ABTuple(a="foo", b=("bar", "baz"))
    inp_b = NestTest.ABTuple(a=2, b=(1, 3))
    out = nest.map_structure(lambda string, repeats: string * repeats,
                             inp_a,
                             inp_b)
    self.assertEqual("foofoo", out.a)
    self.assertEqual("bar", out.b[0])
    self.assertEqual("bazbazbaz", out.b[1])

    nt = NestTest.ABTuple(a=("something", "something_else"),
                          b="yet another thing")
    rev_nt = nest.map_structure(lambda x: x[::-1], nt)
    # Check the output is the correct structure, and all strings are reversed.
    nest.assert_same_structure(nt, rev_nt)
    self.assertEqual(nt.a[0][::-1], rev_nt.a[0])
    self.assertEqual(nt.a[1][::-1], rev_nt.a[1])
    self.assertEqual(nt.b[::-1], rev_nt.b)

  def testMapStructureOverPlaceholders(self):
    inp_a = (array_ops.placeholder(dtypes.float32, shape=[3, 4]),
             array_ops.placeholder(dtypes.float32, shape=[3, 7]))
    inp_b = (array_ops.placeholder(dtypes.float32, shape=[3, 4]),
             array_ops.placeholder(dtypes.float32, shape=[3, 7]))

    output = nest.map_structure(lambda x1, x2: x1 + x2, inp_a, inp_b)

    nest.assert_same_structure(output, inp_a)
    self.assertShapeEqual(np.zeros((3, 4)), output[0])
    self.assertShapeEqual(np.zeros((3, 7)), output[1])

    feed_dict = {
        inp_a: (np.random.randn(3, 4), np.random.randn(3, 7)),
        inp_b: (np.random.randn(3, 4), np.random.randn(3, 7))
    }

    with self.test_session() as sess:
      output_np = sess.run(output, feed_dict=feed_dict)
    self.assertAllClose(output_np[0],
                        feed_dict[inp_a][0] + feed_dict[inp_b][0])
    self.assertAllClose(output_np[1],
                        feed_dict[inp_a][1] + feed_dict[inp_b][1])

  def testAssertShallowStructure(self):
    inp_ab = ["a", "b"]
    inp_abc = ["a", "b", "c"]
    expected_message = (
        "The two structures don't have the same sequence length. Input "
        "structure has length 2, while shallow structure has length 3.")
    with self.assertRaisesRegexp(ValueError, expected_message):
      nest.assert_shallow_structure(inp_abc, inp_ab)

    inp_ab1 = [(1, 1), (2, 2)]
    inp_ab2 = [[1, 1], [2, 2]]
    expected_message = (
        "The two structures don't have the same sequence type. Input structure "
        "has type <(type|class) 'tuple'>, while shallow structure has type "
        "<(type|class) 'list'>.")
    with self.assertRaisesRegexp(TypeError, expected_message):
      nest.assert_shallow_structure(inp_ab2, inp_ab1)
    nest.assert_shallow_structure(inp_ab2, inp_ab1, check_types=False)

    inp_ab1 = {"a": (1, 1), "b": {"c": (2, 2)}}
    inp_ab2 = {"a": (1, 1), "b": {"d": (2, 2)}}
    expected_message = (
        r"The two structures don't have the same keys. Input "
        r"structure has keys \['c'\], while shallow structure has "
        r"keys \['d'\].")

    with self.assertRaisesRegexp(ValueError, expected_message):
      nest.assert_shallow_structure(inp_ab2, inp_ab1)

    inp_ab = collections.OrderedDict([("a", 1), ("b", (2, 3))])
    inp_ba = collections.OrderedDict([("b", (2, 3)), ("a", 1)])
    nest.assert_shallow_structure(inp_ab, inp_ba)

    # This assertion is expected to pass: two namedtuples with the same
    # name and field names are considered to be identical.
    inp_shallow = NestTest.SameNameab(1, 2)
    inp_deep = NestTest.SameNameab2(1, [1, 2, 3])
    nest.assert_shallow_structure(inp_shallow, inp_deep, check_types=False)
    nest.assert_shallow_structure(inp_shallow, inp_deep, check_types=True)

  def testFlattenUpTo(self):
    # Shallow tree ends at scalar.
    input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
    shallow_tree = [[True, True], [False, True]]
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [[2, 2], [3, 3], [4, 9], [5, 5]])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

    # Shallow tree ends at string.
    input_tree = [[("a", 1), [("b", 2), [("c", 3), [("d", 4)]]]]]
    shallow_tree = [["level_1", ["level_2", ["level_3", ["level_4"]]]]]
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    input_tree_flattened = nest.flatten(input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    self.assertEqual(input_tree_flattened, ["a", 1, "b", 2, "c", 3, "d", 4])

    # Make sure dicts are correctly flattened, yielding values, not keys.
    input_tree = {"a": 1, "b": {"c": 2}, "d": [3, (4, 5)]}
    shallow_tree = {"a": 0, "b": 0, "d": [0, 0]}
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [1, {"c": 2}, 3, (4, 5)])

    # Namedtuples.
    ab_tuple = NestTest.ABTuple
    input_tree = ab_tuple(a=[0, 1], b=2)
    shallow_tree = ab_tuple(a=0, b=1)
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [[0, 1], 2])

    # Nested dicts, OrderedDicts and namedtuples.
    input_tree = collections.OrderedDict(
        [("a", ab_tuple(a=[0, {"b": 1}], b=2)),
         ("c", {"d": 3, "e": collections.OrderedDict([("f", 4)])})])
    shallow_tree = input_tree
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree, [0, 1, 2, 3, 4])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", {"d": 3, "e": 1})])
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      3,
                      collections.OrderedDict([("f", 4)])])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", 0)])
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      {"d": 3, "e": collections.OrderedDict([("f", 4)])}])

    ## Shallow non-list edge-case.
    # Using iterable elements.
    input_tree = ["input_tree"]
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = ["input_tree_0", "input_tree_1"]
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = [0]
    shallow_tree = 9
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = [0, 1]
    shallow_tree = 9
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Both non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = 0
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Input non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = ["shallow_tree"]
    expected_message = ("If shallow structure is a sequence, input must also "
                        "be a sequence. Input has type: <(type|class) 'str'>.")
    with self.assertRaisesRegexp(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = "input_tree"
    shallow_tree = ["shallow_tree_9", "shallow_tree_8"]
    with self.assertRaisesRegexp(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = [9]
    expected_message = ("If shallow structure is a sequence, input must also "
                        "be a sequence. Input has type: <(type|class) 'int'>.")
    with self.assertRaisesRegexp(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = 0
    shallow_tree = [9, 8]
    with self.assertRaisesRegexp(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

  def testMapStructureUpTo(self):
    # Named tuples.
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    op_tuple = collections.namedtuple("op_tuple", "add, mul")
    inp_val = ab_tuple(a=2, b=3)
    inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val, lambda val, ops: (val + ops.add) * ops.mul, inp_val, inp_ops)
    self.assertEqual(out.a, 6)
    self.assertEqual(out.b, 15)

    # Lists.
    data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
    name_list = ["evens", ["odds", "primes"]]
    out = nest.map_structure_up_to(
        name_list, lambda name, sec: "first_{}_{}".format(len(sec), name),
        name_list, data_list)
    self.assertEqual(out, ["first_4_evens", ["first_5_odds", "first_3_primes"]])

  def testGetTraverseShallowStructure(self):
    scalar_traverse_input = [3, 4, (1, 2, [0]), [5, 6], {"a": (7,)}, []]
    scalar_traverse_r = nest.get_traverse_shallow_structure(
        lambda s: not isinstance(s, tuple),
        scalar_traverse_input)
    self.assertEqual(scalar_traverse_r,
                     [True, True, False, [True, True], {"a": False}, []])
    nest.assert_shallow_structure(scalar_traverse_r,
                                  scalar_traverse_input)

    structure_traverse_input = [(1, [2]), ([1], 2)]
    structure_traverse_r = nest.get_traverse_shallow_structure(
        lambda s: (True, False) if isinstance(s, tuple) else True,
        structure_traverse_input)
    self.assertEqual(structure_traverse_r,
                     [(True, False), ([True], False)])
    nest.assert_shallow_structure(structure_traverse_r,
                                  structure_traverse_input)

    with self.assertRaisesRegexp(TypeError, "returned structure"):
      nest.get_traverse_shallow_structure(lambda _: [True], 0)

    with self.assertRaisesRegexp(TypeError, "returned a non-bool scalar"):
      nest.get_traverse_shallow_structure(lambda _: 1, [1])

    with self.assertRaisesRegexp(
        TypeError, "didn't return a depth=1 structure of bools"):
      nest.get_traverse_shallow_structure(lambda _: [1], [1])

  def testYieldFlatStringPaths(self):
    for inputs_expected in ({"inputs": [], "expected": []},
                            {"inputs": 3, "expected": [()]},
                            {"inputs": [3], "expected": [(0,)]},
                            {"inputs": {"a": 3}, "expected": [("a",)]},
                            {"inputs": {"a": {"b": 4}},
                             "expected": [("a", "b")]},
                            {"inputs": [{"a": 2}], "expected": [(0, "a")]},
                            {"inputs": [{"a": [2]}], "expected": [(0, "a", 0)]},
                            {"inputs": [{"a": [(23, 42)]}],
                             "expected": [(0, "a", 0, 0), (0, "a", 0, 1)]},
                            {"inputs": [{"a": ([23], 42)}],
                             "expected": [(0, "a", 0, 0), (0, "a", 1)]},
                            {"inputs": {"a": {"a": 2}, "c": [[[4]]]},
                             "expected": [("a", "a"), ("c", 0, 0, 0)]},
                            {"inputs": {"0": [{"1": 23}]},
                             "expected": [("0", 0, "1")]}):
      inputs = inputs_expected["inputs"]
      expected = inputs_expected["expected"]
      self.assertEqual(list(nest.yield_flat_paths(inputs)), expected)

  def testFlattenWithStringPaths(self):
    for inputs_expected in (
        {"inputs": [], "expected": []},
        {"inputs": [23, "42"], "expected": [("0", 23), ("1", "42")]},
        {"inputs": [[[[108]]]], "expected": [("0/0/0/0", 108)]}):
      inputs = inputs_expected["inputs"]
      expected = inputs_expected["expected"]
      self.assertEqual(
          nest.flatten_with_joined_string_paths(inputs, separator="/"),
          expected)

  # Need a separate test for namedtuple as we can't declare tuple definitions
  # in the @parameterized arguments.
  def testFlattenNamedTuple(self):
    # pylint: disable=invalid-name
    Foo = collections.namedtuple("Foo", ["a", "b"])
    Bar = collections.namedtuple("Bar", ["c", "d"])
    # pylint: enable=invalid-name
    test_cases = [
        (Foo(a=3, b=Bar(c=23, d=42)),
         [("a", 3), ("b/c", 23), ("b/d", 42)]),
        (Foo(a=Bar(c=23, d=42), b=Bar(c=0, d="something")),
         [("a/c", 23), ("a/d", 42), ("b/c", 0), ("b/d", "something")]),
        (Bar(c=42, d=43),
         [("c", 42), ("d", 43)]),
        (Bar(c=[42], d=43),
         [("c/0", 42), ("d", 43)]),
    ]
    for inputs, expected in test_cases:
      self.assertEqual(
          list(nest.flatten_with_joined_string_paths(inputs)), expected)


class NestBenchmark(test.Benchmark):

  def run_and_report(self, s1, s2, name):
    burn_iter, test_iter = 100, 30000

    for _ in xrange(burn_iter):
      nest.assert_same_structure(s1, s2)

    t0 = time.time()
    for _ in xrange(test_iter):
      nest.assert_same_structure(s1, s2)
    t1 = time.time()

    self.report_benchmark(iters=test_iter, wall_time=(t1 - t0) / test_iter,
                          name=name)

  def benchmark_assert_structure(self):
    s1 = (((1, 2), 3), 4, (5, 6))
    s2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
    self.run_and_report(s1, s2, "assert_same_structure_6_elem")

    s1 = (((1, 2), 3), 4, (5, 6)) * 10
    s2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6")) * 10
    self.run_and_report(s1, s2, "assert_same_structure_60_elem")


if __name__ == "__main__":
  test.main()
