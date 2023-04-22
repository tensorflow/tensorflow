# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


class NestTest(test_base.DatasetTestBase, parameterized.TestCase):

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenAndPack(self):
    structure = ((3, 4), 5, (6, 7, (9, 10), 8))
    flat = ["a", "b", "c", "d", "e", "f", "g", "h"]
    self.assertEqual(nest.flatten(structure), [3, 4, 5, 6, 7, 9, 10, 8])
    self.assertEqual(
        nest.pack_sequence_as(structure, flat), (("a", "b"), "c",
                                                 ("d", "e", ("f", "g"), "h")))
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

    with self.assertRaisesRegex(ValueError, "Structure is a scalar"):
      nest.pack_sequence_as("scalar", [4, 5])

    with self.assertRaisesRegex(TypeError, "flat_sequence"):
      nest.pack_sequence_as([4, 5], "bad_sequence")

    with self.assertRaises(ValueError):
      nest.pack_sequence_as([5, 6, [7, 8]], ["a", "b", "c"])

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenDictOrder(self):
    """`flatten` orders dicts by key, including OrderedDicts."""
    ordered = collections.OrderedDict([("d", 3), ("b", 1), ("a", 0), ("c", 2)])
    plain = {"d": 3, "b": 1, "a": 0, "c": 2}
    ordered_flat = nest.flatten(ordered)
    plain_flat = nest.flatten(plain)
    self.assertEqual([0, 1, 2, 3], ordered_flat)
    self.assertEqual([0, 1, 2, 3], plain_flat)

  @combinations.generate(test_base.default_test_combinations())
  def testPackDictOrder(self):
    """Packing orders dicts by key, including OrderedDicts."""
    ordered = collections.OrderedDict([("d", 0), ("b", 0), ("a", 0), ("c", 0)])
    plain = {"d": 0, "b": 0, "a": 0, "c": 0}
    seq = [0, 1, 2, 3]
    ordered_reconstruction = nest.pack_sequence_as(ordered, seq)
    plain_reconstruction = nest.pack_sequence_as(plain, seq)
    self.assertEqual(
        collections.OrderedDict([("d", 3), ("b", 1), ("a", 0), ("c", 2)]),
        ordered_reconstruction)
    self.assertEqual({"d": 3, "b": 1, "a": 0, "c": 2}, plain_reconstruction)

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenAndPackWithDicts(self):
    # A nice messy mix of tuples, lists, dicts, and `OrderedDict`s.
    named_tuple = collections.namedtuple("A", ("b", "c"))
    mess = (
        "z",
        named_tuple(3, 4),
        {
            "c": (
                1,
                collections.OrderedDict([
                    ("b", 3),
                    ("a", 2),
                ]),
            ),
            "b": 5
        },
        17
    )

    flattened = nest.flatten(mess)
    self.assertEqual(flattened, ["z", 3, 4, 5, 1, 2, 3, 17])

    structure_of_mess = (
        14,
        named_tuple("a", True),
        {
            "c": (
                0,
                collections.OrderedDict([
                    ("b", 9),
                    ("a", 8),
                ]),
            ),
            "b": 3
        },
        "hi everybody",
    )

    unflattened = nest.pack_sequence_as(structure_of_mess, flattened)
    self.assertEqual(unflattened, mess)

    # Check also that the OrderedDict was created, with the correct key order.
    unflattened_ordered_dict = unflattened[2]["c"][1]
    self.assertIsInstance(unflattened_ordered_dict, collections.OrderedDict)
    self.assertEqual(list(unflattened_ordered_dict.keys()), ["b", "a"])

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenSparseValue(self):
    st = sparse_tensor.SparseTensorValue([[0]], [0], [1])
    single_value = st
    list_of_values = [st, st, st]
    nest_of_values = ((st), ((st), (st)))
    dict_of_values = {"foo": st, "bar": st, "baz": st}
    self.assertEqual([st], nest.flatten(single_value))
    self.assertEqual([[st, st, st]], nest.flatten(list_of_values))
    self.assertEqual([st, st, st], nest.flatten(nest_of_values))
    self.assertEqual([st, st, st], nest.flatten(dict_of_values))

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenRaggedValue(self):
    rt = ragged_factory_ops.constant_value([[[0]], [[1]]])
    single_value = rt
    list_of_values = [rt, rt, rt]
    nest_of_values = ((rt), ((rt), (rt)))
    dict_of_values = {"foo": rt, "bar": rt, "baz": rt}
    self.assertEqual([rt], nest.flatten(single_value))
    self.assertEqual([[rt, rt, rt]], nest.flatten(list_of_values))
    self.assertEqual([rt, rt, rt], nest.flatten(nest_of_values))
    self.assertEqual([rt, rt, rt], nest.flatten(dict_of_values))

  @combinations.generate(test_base.default_test_combinations())
  def testIsSequence(self):
    self.assertFalse(nest.is_sequence("1234"))
    self.assertFalse(nest.is_sequence([1, 3, [4, 5]]))
    self.assertTrue(nest.is_sequence(((7, 8), (5, 6))))
    self.assertFalse(nest.is_sequence([]))
    self.assertFalse(nest.is_sequence(set([1, 2])))
    ones = array_ops.ones([2, 3])
    self.assertFalse(nest.is_sequence(ones))
    self.assertFalse(nest.is_sequence(math_ops.tanh(ones)))
    self.assertFalse(nest.is_sequence(np.ones((4, 5))))
    self.assertTrue(nest.is_sequence({"foo": 1, "bar": 2}))
    self.assertFalse(
        nest.is_sequence(sparse_tensor.SparseTensorValue([[0]], [0], [1])))
    self.assertFalse(
        nest.is_sequence(ragged_factory_ops.constant_value([[[0]], [[1]]])))

  @combinations.generate(test_base.default_test_combinations())
  def testAssertSameStructure(self):
    structure1 = (((1, 2), 3), 4, (5, 6))
    structure2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
    structure_different_num_elements = ("spam", "eggs")
    structure_different_nesting = (((1, 2), 3), 4, 5, (6,))
    structure_dictionary = {"foo": 2, "bar": 4, "baz": {"foo": 5, "bar": 6}}
    structure_dictionary_diff_nested = {
        "foo": 2,
        "bar": 4,
        "baz": {
            "foo": 5,
            "baz": 6
        }
    }
    nest.assert_same_structure(structure1, structure2)
    nest.assert_same_structure("abc", 1.0)
    nest.assert_same_structure("abc", np.array([0, 1]))
    nest.assert_same_structure("abc", constant_op.constant([0, 1]))

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure(structure1, structure_different_num_elements)

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure((0, 1), np.array([0, 1]))

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure(0, (0, 1))

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure(structure1, structure_different_nesting)

    named_type_0 = collections.namedtuple("named_0", ("a", "b"))
    named_type_1 = collections.namedtuple("named_1", ("a", "b"))
    self.assertRaises(TypeError, nest.assert_same_structure, (0, 1),
                      named_type_0("a", "b"))

    nest.assert_same_structure(named_type_0(3, 4), named_type_0("a", "b"))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      named_type_0(3, 4), named_type_1(3, 4))

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure(named_type_0(3, 4), named_type_0((3,), 4))

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure(((3,), 4), (3, (4,)))

    structure1_list = {"a": ((1, 2), 3), "b": 4, "c": (5, 6)}
    structure2_list = {"a": ((1, 2), 3), "b": 4, "d": (5, 6)}
    with self.assertRaisesRegex(TypeError, "don't have the same sequence type"):
      nest.assert_same_structure(structure1, structure1_list)
    nest.assert_same_structure(structure1, structure2, check_types=False)
    nest.assert_same_structure(structure1, structure1_list, check_types=False)
    with self.assertRaisesRegex(ValueError, "don't have the same set of keys"):
      nest.assert_same_structure(structure1_list, structure2_list)
    with self.assertRaisesRegex(ValueError, "don't have the same set of keys"):
      nest.assert_same_structure(structure_dictionary,
                                 structure_dictionary_diff_nested)
    nest.assert_same_structure(
        structure_dictionary,
        structure_dictionary_diff_nested,
        check_types=False)
    nest.assert_same_structure(
        structure1_list, structure2_list, check_types=False)

  @combinations.generate(test_base.default_test_combinations())
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

    with self.assertRaisesRegex(TypeError, "callable"):
      nest.map_structure("bad", structure1_plus1)

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, 3, (3,))

    with self.assertRaisesRegex(TypeError, "same sequence type"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), {"a": (3, 4), "b": 5})

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)))

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)),
                         check_types=False)

    with self.assertRaisesRegex(ValueError, "Only valid keyword argument"):
      nest.map_structure(lambda x: None, structure1, foo="a")

    with self.assertRaisesRegex(ValueError, "Only valid keyword argument"):
      nest.map_structure(lambda x: None, structure1, check_types=False, foo="a")

  @combinations.generate(test_base.default_test_combinations())
  def testAssertShallowStructure(self):
    inp_ab = ("a", "b")
    inp_abc = ("a", "b", "c")
    expected_message = (
        "The two structures don't have the same sequence length. Input "
        "structure has length 2, while shallow structure has length 3.")
    with self.assertRaisesRegex(ValueError, expected_message):
      nest.assert_shallow_structure(inp_abc, inp_ab)

    inp_ab1 = ((1, 1), (2, 2))
    inp_ab2 = {"a": (1, 1), "b": (2, 2)}
    expected_message = (
        "The two structures don't have the same sequence type. Input structure "
        "has type <(type|class) 'tuple'>, while shallow structure has type "
        "<(type|class) 'dict'>.")
    with self.assertRaisesRegex(TypeError, expected_message):
      nest.assert_shallow_structure(inp_ab2, inp_ab1)
    nest.assert_shallow_structure(inp_ab2, inp_ab1, check_types=False)

    inp_ab1 = {"a": (1, 1), "b": {"c": (2, 2)}}
    inp_ab2 = {"a": (1, 1), "b": {"d": (2, 2)}}
    expected_message = (
        r"The two structures don't have the same keys. Input "
        r"structure has keys \['c'\], while shallow structure has "
        r"keys \['d'\].")
    with self.assertRaisesRegex(ValueError, expected_message):
      nest.assert_shallow_structure(inp_ab2, inp_ab1)

    inp_ab = collections.OrderedDict([("a", 1), ("b", (2, 3))])
    inp_ba = collections.OrderedDict([("b", (2, 3)), ("a", 1)])
    nest.assert_shallow_structure(inp_ab, inp_ba)

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenUpTo(self):
    input_tree = (((2, 2), (3, 3)), ((4, 9), (5, 5)))
    shallow_tree = ((True, True), (False, True))
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [(2, 2), (3, 3), (4, 9), (5, 5)])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

    input_tree = ((("a", 1), (("b", 2), (("c", 3), (("d", 4))))))
    shallow_tree = (("level_1", ("level_2", ("level_3", ("level_4")))))
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    input_tree_flattened = nest.flatten(input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    self.assertEqual(input_tree_flattened, ["a", 1, "b", 2, "c", 3, "d", 4])

    ## Shallow non-list edge-case.
    # Using iterable elements.
    input_tree = ["input_tree"]
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = ("input_tree_0", "input_tree_1")
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = (0,)
    shallow_tree = 9
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = (0, 1)
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
    shallow_tree = ("shallow_tree",)
    expected_message = ("If shallow structure is a sequence, input must also "
                        "be a sequence. Input has type: <(type|class) 'str'>.")
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, list(shallow_tree))

    input_tree = "input_tree"
    shallow_tree = ("shallow_tree_9", "shallow_tree_8")
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, list(shallow_tree))

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = (9,)
    expected_message = ("If shallow structure is a sequence, input must also "
                        "be a sequence. Input has type: <(type|class) 'int'>.")
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, list(shallow_tree))

    input_tree = 0
    shallow_tree = (9, 8)
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, list(shallow_tree))

    # Using dict.
    input_tree = {"a": ((2, 2), (3, 3)), "b": ((4, 9), (5, 5))}
    shallow_tree = {"a": (True, True), "b": (False, True)}
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [(2, 2), (3, 3), (4, 9), (5, 5)])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

  @combinations.generate(test_base.default_test_combinations())
  def testMapStructureUpTo(self):
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    op_tuple = collections.namedtuple("op_tuple", "add, mul")
    inp_val = ab_tuple(a=2, b=3)
    inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val, lambda val, ops: (val + ops.add) * ops.mul, inp_val, inp_ops)
    self.assertEqual(out.a, 6)
    self.assertEqual(out.b, 15)

    data_list = ((2, 4, 6, 8), ((1, 3, 5, 7, 9), (3, 5, 7)))
    name_list = ("evens", ("odds", "primes"))
    out = nest.map_structure_up_to(
        name_list, lambda name, sec: "first_{}_{}".format(len(sec), name),
        name_list, data_list)
    self.assertEqual(out, ("first_4_evens", ("first_5_odds", "first_3_primes")))


if __name__ == "__main__":
  test.main()
