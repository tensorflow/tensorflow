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

import collections
import collections.abc
import dataclasses
import time
from typing import NamedTuple

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.util import nest
from tensorflow.python.util.nest_util import CustomNestProtocol

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


class _CustomMapping(collections.abc.Mapping):

  def __init__(self, *args, **kwargs):
    self._wrapped = dict(*args, **kwargs)

  def __getitem__(self, key):
    return self._wrapped[key]

  def __iter__(self):
    return iter(self._wrapped)

  def __len__(self):
    return len(self._wrapped)


class _CustomList(list):
  pass


class _CustomSequenceThatRaisesException(collections.abc.Sequence):

  def __len__(self):
    return 1

  def __getitem__(self, item):
    raise ValueError("Cannot get item: %s" % item)


@dataclasses.dataclass
class MaskedTensor:
  mask: bool
  value: tensor.Tensor

  def __tf_flatten__(self):
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, components):
    mask = metadata[0]
    value = components[0]
    return MaskedTensor(mask=mask, value=value)

  def __eq__(self, other):
    return self.mask == other.mask and math_ops.reduce_all(
        self.value == other.value
    )

  def __len__(self):
    # Used by `nest.map_structure_up_to` and releatd functions to verify the
    # arity compatibility.
    return 1


class MaskedTensor2(MaskedTensor):
  pass


@dataclasses.dataclass
class NestedMaskedTensor:
  mask: bool
  value: MaskedTensor

  @classmethod
  def nested_masked_tensor_with_opposite_masks(cls, mask, inner_value):
    return NestedMaskedTensor(
        mask=mask, value=MaskedTensor(mask=not mask, value=inner_value)
    )

  def __tf_flatten__(self):
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, components):
    mask = metadata[0]
    value = components[0]
    return NestedMaskedTensor(mask=mask, value=value)

  def __eq__(self, other):
    return self.mask == other.mask and self.value == other.value

  def __len__(self):
    return 1


class NestTest(parameterized.TestCase, test.TestCase):

  PointXY = collections.namedtuple("Point", ["x", "y"])  # pylint: disable=invalid-name
  unsafe_map_pattern = ("nest cannot guarantee that it is safe to map one to "
                        "the other.")
  bad_pack_pattern = ("Attempted to pack value:\n  .+\ninto a structure, but "
                      "found incompatible type `<(type|class) 'str'>` instead.")

  if attr:
    class BadAttr(object):
      """Class that has a non-iterable __attrs_attrs__."""
      __attrs_attrs__ = None

    @attr.s
    class SampleAttr(object):
      field1 = attr.ib()
      field2 = attr.ib()

    @attr.s
    class UnsortedSampleAttr(object):
      field3 = attr.ib()
      field1 = attr.ib()
      field2 = attr.ib()

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassCustomProtocol(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    self.assertIsInstance(mt, CustomNestProtocol)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassIsNested(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    self.assertTrue(nest.is_nested(mt))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlatten(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    leaves = nest.flatten(mt)
    self.assertLen(leaves, 1)
    self.assertAllEqual(leaves[0], [1])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenUpToCompatible(self):
    simple_list = [2]
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    flattened_mt = nest.flatten_up_to(
        shallow_tree=simple_list, input_tree=mt, check_types=False
    )
    # Expected flat_path_mt = [Tensor([1])]
    self.assertAllEqual(flattened_mt[0], [1])
    flattened_list = nest.flatten_up_to(
        shallow_tree=mt, input_tree=simple_list, check_types=False
    )
    self.assertEqual(flattened_list, [2])

    nested_list = [[2]]
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([3])
    )
    flattened_nmt = nest.flatten_up_to(
        shallow_tree=nested_list, input_tree=nmt, check_types=False
    )
    # Expected flattened_nmt = [Tensor([3])]
    self.assertAllEqual(flattened_nmt[0], [3])

    flat_path_nested_list = nest.flatten_up_to(
        shallow_tree=nmt, input_tree=nested_list, check_types=False
    )
    self.assertAllEqual(flat_path_nested_list, [2])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenUpToIncompatible(self):
    simple_list = [2]
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))

    # When `check_types=True` is set, `flatten_up_to` would fail when input_tree
    # and shallow_tree args don't have the same type
    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(simple_list), input_type=type(mt)
        ),
    ):
      nest.flatten_up_to(
          shallow_tree=simple_list, input_tree=mt, check_types=True
      )

    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(mt), input_type=type(simple_list)
        ),
    ):
      nest.flatten_up_to(
          shallow_tree=mt, input_tree=simple_list, check_types=True
      )

    nested_list = [[1]]
    # Although `check_types=False` is set, this assertion would fail because the
    # shallow_tree component has a deeper structure than the input_tree
    # component.
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        "If shallow structure is a sequence, input must also be a sequence",
    ):
      nest.flatten_up_to(
          shallow_tree=nested_list, input_tree=mt, check_types=False
      )

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenWithTuplePathsUpToCompatible(self):
    simple_list = [2]
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    flat_path_mt = nest.flatten_with_tuple_paths_up_to(
        shallow_tree=simple_list, input_tree=mt, check_types=False
    )
    # Expected flat_path_mt = [((0,), Tensor([1]))]
    self.assertEqual(flat_path_mt[0][0], (0,))
    self.assertAllEqual(flat_path_mt[0][1], [1])

    flat_path_list = nest.flatten_with_tuple_paths_up_to(
        shallow_tree=mt, input_tree=simple_list, check_types=False
    )
    self.assertAllEqual(flat_path_list, [[(0,), 2]])

    nested_list = [[2]]
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([3])
    )
    flat_path_nmt = nest.flatten_with_tuple_paths_up_to(
        shallow_tree=nested_list, input_tree=nmt, check_types=False
    )
    # Expected flat_path_nmt = [((0,), Tensor([3]))]
    self.assertAllEqual(flat_path_nmt[0][0], [0, 0])
    self.assertAllEqual(flat_path_nmt[0][1], [3])

    flat_path_nested_list = nest.flatten_with_tuple_paths_up_to(
        shallow_tree=nmt, input_tree=nested_list, check_types=False
    )
    self.assertAllEqual(flat_path_nested_list, [[(0, 0), 2]])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenWithTuplePathsUpToIncompatible(self):
    simple_list = [2]
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(simple_list), input_type=type(mt)
        ),
    ):
      nest.flatten_with_tuple_paths_up_to(
          shallow_tree=simple_list, input_tree=mt, check_types=True
      )

    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(mt), input_type=type(simple_list)
        ),
    ):
      nest.flatten_with_tuple_paths_up_to(
          shallow_tree=mt, input_tree=simple_list, check_types=True
      )

    nested_list2 = [[[2]]]
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([3])
    )

    # Although `check_types=False` is set, this assertion would fail because the
    # shallow_tree component has a deeper structure than the input_tree
    # component.
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        "If shallow structure is a sequence, input must also be a sequence",
    ):
      nest.flatten_up_to(
          shallow_tree=nested_list2, input_tree=nmt, check_types=False
      )

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenAndPack(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    leaves = nest.flatten(mt)
    reconstructed_mt = nest.pack_sequence_as(mt, leaves)
    self.assertIsInstance(reconstructed_mt, MaskedTensor)
    self.assertEqual(reconstructed_mt, mt)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassMapStructure(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt_doubled = nest.map_structure(lambda x: x * 2, mt)
    self.assertIsInstance(mt_doubled, MaskedTensor)
    self.assertEqual(mt_doubled.mask, True)
    self.assertAllEqual(mt_doubled.value, [2])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassMapStructureWithPaths(self):
    mt = MaskedTensor(mask=False, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2]))
    mt3 = MaskedTensor(mask=True, value=constant_op.constant([3]))

    def path_sum(path, *tensors):
      return (path, sum(tensors))

    mt_combined_with_path = nest.map_structure_with_paths(
        path_sum, mt, mt2, mt3
    )
    self.assertIsInstance(mt_combined_with_path, MaskedTensor)
    # metadata uses the one from the first input (mt).
    self.assertEqual(mt_combined_with_path.mask, False)
    # Tesnor index is '0' for the only compoenent in MaskedTensor.
    self.assertAllEqual(mt_combined_with_path.value[0], "0")
    # sum of all input tensors.
    self.assertAllEqual(mt_combined_with_path.value[1], [6])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([4])
    )
    nmt2 = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=False, inner_value=constant_op.constant([5])
    )
    nmt_combined_with_path = nest.map_structure_with_paths(path_sum, nmt, nmt2)
    self.assertIsInstance(nmt_combined_with_path, NestedMaskedTensor)
    self.assertEqual(nmt_combined_with_path.mask, True)
    self.assertEqual(nmt_combined_with_path.value.mask, False)
    self.assertAllEqual(nmt_combined_with_path.value.value[0], "0/0")
    self.assertAllEqual(nmt_combined_with_path.value.value[1], [9])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassMapStructureWithTuplePaths(self):
    mt = MaskedTensor(mask=False, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2]))
    mt3 = MaskedTensor(mask=True, value=constant_op.constant([3]))

    def tuple_path_sum(tuple_path, *tensors):
      return (tuple_path, sum(tensors))

    mt_combined_with_path = nest.map_structure_with_tuple_paths(
        tuple_path_sum, mt, mt2, mt3
    )
    self.assertIsInstance(mt_combined_with_path, MaskedTensor)
    # metadata uses the one from the first input (mt).
    self.assertEqual(mt_combined_with_path.mask, False)
    # Tesnor index is 0 for the only compoenent in MaskedTensor.
    self.assertAllEqual(mt_combined_with_path.value[0], (0,))
    # sum of all input tensors.
    self.assertAllEqual(mt_combined_with_path.value[1], [6])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([4])
    )
    nmt2 = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=False, inner_value=constant_op.constant([5])
    )
    nmt_combined_with_path = nest.map_structure_with_tuple_paths(
        tuple_path_sum, nmt, nmt2
    )
    self.assertIsInstance(nmt_combined_with_path, NestedMaskedTensor)
    self.assertEqual(nmt_combined_with_path.mask, True)
    self.assertEqual(nmt_combined_with_path.value.mask, False)
    self.assertAllEqual(nmt_combined_with_path.value.value[0], (0, 0))
    self.assertAllEqual(nmt_combined_with_path.value.value[1], [9])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassMapStructureUpTo(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2]))
    mt3 = MaskedTensor(mask=True, value=constant_op.constant([3]))
    mt_out_template = MaskedTensor(mask=False, value=constant_op.constant([4]))

    def sum_tensors(*tensors):
      return sum(tensors)

    mt_combined_with_path = nest.map_structure_up_to(
        mt_out_template, sum_tensors, mt, mt2, mt3
    )
    self.assertIsInstance(mt_combined_with_path, MaskedTensor)
    # metadata uses the one from the first arg (mt_out_template).
    self.assertEqual(mt_combined_with_path.mask, False)
    # sum of all input tensors.
    self.assertAllEqual(mt_combined_with_path.value, [6])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([4])
    )
    nmt2 = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([5])
    )
    nmt_out = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=False, inner_value=constant_op.constant([6])
    )
    nmt_combined_with_path = nest.map_structure_up_to(
        nmt_out, sum_tensors, nmt, nmt2
    )
    self.assertIsInstance(nmt_combined_with_path, NestedMaskedTensor)
    self.assertEqual(nmt_combined_with_path.mask, False)
    self.assertEqual(nmt_combined_with_path.value.mask, True)
    self.assertAllEqual(nmt_combined_with_path.value.value, [9])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassMapStructureWithTuplePathsUoTo(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2]))
    mt3 = MaskedTensor(mask=True, value=constant_op.constant([3]))
    mt_out_template = MaskedTensor(mask=False, value=constant_op.constant([4]))

    def tuple_path_sum(tuple_path, *tensors):
      return (tuple_path, sum(tensors))

    mt_combined_with_path = nest.map_structure_with_tuple_paths_up_to(
        mt_out_template, tuple_path_sum, mt, mt2, mt3
    )
    self.assertIsInstance(mt_combined_with_path, MaskedTensor)
    # metadata uses the one from the first arg (mt_out_template).
    self.assertEqual(mt_combined_with_path.mask, False)
    # Tesnor index is 0 for the only compoenent in MaskedTensor.
    self.assertAllEqual(mt_combined_with_path.value[0], (0,))
    # sum of all input tensors.
    self.assertAllEqual(mt_combined_with_path.value[1], [6])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([4])
    )
    nmt2 = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([5])
    )
    nmt_out = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=False, inner_value=constant_op.constant([6])
    )
    nmt_combined_with_path = nest.map_structure_with_tuple_paths_up_to(
        nmt_out, tuple_path_sum, nmt, nmt2
    )
    self.assertIsInstance(nmt_combined_with_path, NestedMaskedTensor)
    self.assertEqual(nmt_combined_with_path.mask, False)
    self.assertEqual(nmt_combined_with_path.value.mask, True)
    self.assertAllEqual(nmt_combined_with_path.value.value[0], (0, 0))
    self.assertAllEqual(nmt_combined_with_path.value.value[1], [9])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNestedDataclassIsNested(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    self.assertTrue(nest.is_nested(mt))

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )
    self.assertTrue(nest.is_nested(nmt))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassAssertShallowStructure(self):
    # These assertions are expected to pass: two dataclasses with the same
    # component size are considered to have the same shallow structure.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=False, value=constant_op.constant([2, 3]))
    nest.assert_shallow_structure(
        shallow_tree=mt, input_tree=mt2, check_types=True
    )
    nest.assert_shallow_structure(
        shallow_tree=mt2, input_tree=mt, check_types=True
    )

    mt3 = MaskedTensor2(mask=True, value=constant_op.constant([1]))
    # These assertions are expected to pass: two dataclasses with the same
    # component size are considered to have the same shallow structure.
    nest.assert_shallow_structure(
        shallow_tree=mt, input_tree=mt3, check_types=False
    )
    nest.assert_shallow_structure(
        shallow_tree=mt3, input_tree=mt, check_types=False
    )

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )

    # This assertion is expected to fail, when `check_types=True`, because the
    # shallow_tree type is not the same as input_tree.
    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(mt), input_type=type(nmt)
        ),
    ):
      nest.assert_shallow_structure(
          shallow_tree=mt, input_tree=nmt, check_types=True
      )

    # This assertion is expected to pass: the shallow_tree component contains
    # the shallow structure of the input_tree component.
    nest.assert_shallow_structure(
        shallow_tree=mt, input_tree=nmt, check_types=False
    )

    # This assertion is expected to fail: the shallow_tree component has
    # a deeper structure than the input_tree component.
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        "If shallow structure is a sequence, input must also be a sequence",
    ):
      nest.assert_shallow_structure(
          shallow_tree=nmt, input_tree=mt, check_types=False
      )

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassGetTraverseShallowStructure(self):
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )
    traverse_result = nest.get_traverse_shallow_structure(
        lambda s: isinstance(s, (NestedMaskedTensor, MaskedTensor)), nmt
    )
    self.assertIsInstance(traverse_result, NestedMaskedTensor)
    self.assertEqual(traverse_result.mask, nmt.mask)
    self.assertIsInstance(traverse_result.value, MaskedTensor)
    self.assertEqual(traverse_result.value.value, False)
    nest.assert_shallow_structure(traverse_result, nmt)

    traverse_result2 = nest.get_traverse_shallow_structure(
        lambda s: not isinstance(s, list), nmt
    )
    self.assertIsInstance(traverse_result2, NestedMaskedTensor)
    self.assertEqual(traverse_result2.mask, nmt.mask)
    self.assertIsInstance(traverse_result2.value, MaskedTensor)
    # Expected traverse_result2.value.value is True since it can pass the
    # traverse function, but there is no more flattening for the Tensor value.
    self.assertEqual(traverse_result2.value.value, True)
    nest.assert_shallow_structure(traverse_result2, nmt)

    traverse_result3 = nest.get_traverse_shallow_structure(
        lambda s: isinstance(s, tensor.Tensor), nmt
    )
    # Expected `traverse_result3 = False` because `nmt` doesn't pass the
    # traverse function.
    self.assertEqual(traverse_result3, False)
    nest.assert_shallow_structure(traverse_result3, nmt)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNestedDataclassFlatten(self):
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )
    leaves = nest.flatten(nmt)
    self.assertLen(leaves, 1)
    self.assertAllEqual(leaves[0], [1])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNestedDataclassFlattenAndPack(self):
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )
    leaves = nest.flatten(nmt)
    reconstructed_mt = nest.pack_sequence_as(nmt, leaves)
    self.assertIsInstance(reconstructed_mt, NestedMaskedTensor)
    self.assertEqual(reconstructed_mt, nmt)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNestedDataclassMapStructure(self):
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )
    mt_doubled = nest.map_structure(lambda x: x * 2, nmt)
    expected = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([2])
    )

    self.assertIsInstance(mt_doubled, NestedMaskedTensor)
    self.assertEqual(mt_doubled.mask, expected.mask)
    self.assertEqual(mt_doubled.value.mask, expected.value.mask)
    self.assertAllEqual(mt_doubled.value.value, expected.value.value)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassYieldFlatPaths(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt_flat_paths = list(nest.yield_flat_paths(mt))
    self.assertEqual(mt_flat_paths, [(0,)])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([2])
    )
    nmt_flat_paths = list(nest.yield_flat_paths(nmt))
    self.assertEqual(nmt_flat_paths, [(0, 0)])

    dict_mt_nmt = {"mt": mt, "nmt": nmt, "mt_nmt_list": [mt, nmt]}
    dict_mt_nmt_flat_paths = list(nest.yield_flat_paths(dict_mt_nmt))
    self.assertEqual(
        dict_mt_nmt_flat_paths,
        [
            ("mt", 0),
            ("mt_nmt_list", 0, 0),
            ("mt_nmt_list", 1, 0, 0),
            ("nmt", 0, 0),
        ],
    )

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenWithStringPaths(self):
    sep = "/"
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt_flat_paths = nest.flatten_with_joined_string_paths(mt, separator=sep)
    self.assertEqual(mt_flat_paths[0][0], "0")
    self.assertAllEqual(mt_flat_paths[0][1], [1])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([2])
    )
    nmt_flat_paths = nest.flatten_with_joined_string_paths(nmt, separator=sep)
    self.assertEqual(nmt_flat_paths[0][0], "0/0")
    self.assertAllEqual(nmt_flat_paths[0][1], [2])

    dict_mt_nmt = {"mt": mt, "nmt": nmt}
    dict_mt_nmt_flat_paths = nest.flatten_with_joined_string_paths(
        dict_mt_nmt, separator=sep
    )
    self.assertEqual(dict_mt_nmt_flat_paths[0][0], "mt/0")
    self.assertAllEqual(dict_mt_nmt_flat_paths[0][1], [1])
    self.assertEqual(dict_mt_nmt_flat_paths[1][0], "nmt/0/0")
    self.assertAllEqual(dict_mt_nmt_flat_paths[1][1], [2])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenWithTuplePaths(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt_flat_paths = nest.flatten_with_tuple_paths(mt)
    self.assertEqual(mt_flat_paths[0][0], (0,))
    self.assertAllEqual(mt_flat_paths[0][1], [1])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([2])
    )
    nmt_flat_paths = nest.flatten_with_tuple_paths(nmt)
    self.assertEqual(nmt_flat_paths[0][0], (0, 0))
    self.assertAllEqual(nmt_flat_paths[0][1], [2])

    dict_mt_nmt = {"mt": mt, "nmt": nmt}
    dict_mt_nmt_flat_paths = nest.flatten_with_tuple_paths(dict_mt_nmt)
    self.assertEqual(dict_mt_nmt_flat_paths[0][0], ("mt", 0))
    self.assertAllEqual(dict_mt_nmt_flat_paths[0][1], [1])
    self.assertEqual(dict_mt_nmt_flat_paths[1][0], ("nmt", 0, 0))
    self.assertAllEqual(dict_mt_nmt_flat_paths[1][1], [2])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassListToTuple(self):
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([2])
    )
    input_sequence = [mt, (nmt, {"a": [mt, nmt, (mt,)]}, None, nmt, [[[mt]]])]

    mt2 = MaskedTensor(mask=True, value=constant_op.constant([3]))
    nmt2 = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=False, inner_value=constant_op.constant([2])
    )
    results = nest.list_to_tuple(input_sequence)
    expected = (
        mt2,
        (nmt2, {"a": (mt2, nmt2, (mt2,))}, None, nmt2, (((mt2,),),)),
    )
    nest.assert_same_structure(results, expected)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testAttrsFlattenAndPack(self):
    if attr is None:
      self.skipTest("attr module is unavailable.")

    field_values = [1, 2]
    sample_attr = NestTest.SampleAttr(*field_values)
    self.assertFalse(nest._is_attrs(field_values))
    self.assertTrue(nest._is_attrs(sample_attr))
    flat = nest.flatten(sample_attr)
    self.assertEqual(field_values, flat)
    restructured_from_flat = nest.pack_sequence_as(sample_attr, flat)
    self.assertIsInstance(restructured_from_flat, NestTest.SampleAttr)
    self.assertEqual(restructured_from_flat, sample_attr)

    # Check that flatten fails if attributes are not iterable
    with self.assertRaisesRegex(TypeError, "object is not iterable"):
      flat = nest.flatten(NestTest.BadAttr())

  @parameterized.parameters(
      {"values": [1, 2, 3]},
      {"values": [{"B": 10, "A": 20}, [1, 2], 3]},
      {"values": [(1, 2), [3, 4], 5]},
      {"values": [PointXY(1, 2), 3, 4]},
  )
  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testAttrsMapStructure(self, values):
    if attr is None:
      self.skipTest("attr module is unavailable.")

    structure = NestTest.UnsortedSampleAttr(*values)
    new_structure = nest.map_structure(lambda x: x, structure)
    self.assertEqual(structure, new_structure)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
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

    with self.assertRaisesRegex(ValueError, self.unsafe_map_pattern):
      nest.pack_sequence_as("scalar", [4, 5])

    with self.assertRaisesRegex(TypeError, self.bad_pack_pattern):
      nest.pack_sequence_as([4, 5], "bad_sequence")

    with self.assertRaises(ValueError):
      nest.pack_sequence_as([5, 6, [7, 8]], ["a", "b", "c"])

  @parameterized.parameters({"mapping_type": collections.OrderedDict},
                            {"mapping_type": _CustomMapping})
  @test_util.assert_no_new_pyobjects_executing_eagerly()
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

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testFlattenAndPackMappingViews(self):
    """`flatten` orders dicts by key, including OrderedDicts."""
    ordered = collections.OrderedDict([("d", 3), ("b", 1), ("a", 0), ("c", 2)])

    # test flattening
    ordered_keys_flat = nest.flatten(ordered.keys())
    ordered_values_flat = nest.flatten(ordered.values())
    ordered_items_flat = nest.flatten(ordered.items())
    self.assertEqual([3, 1, 0, 2], ordered_values_flat)
    self.assertEqual(["d", "b", "a", "c"], ordered_keys_flat)
    self.assertEqual(["d", 3, "b", 1, "a", 0, "c", 2], ordered_items_flat)

    # test packing
    self.assertEqual([("d", 3), ("b", 1), ("a", 0), ("c", 2)],
                     nest.pack_sequence_as(ordered.items(), ordered_items_flat))

  Abc = collections.namedtuple("A", ("b", "c"))  # pylint: disable=invalid-name

  @test_util.assert_no_new_pyobjects_executing_eagerly()
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
    self.assertLen(flattened, 1)

  def testFlatten_stringIsNotFlattened(self):
    structure = "lots of letters"
    flattened = nest.flatten(structure)
    self.assertLen(flattened, 1)
    unflattened = nest.pack_sequence_as("goodbye", flattened)
    self.assertEqual(structure, unflattened)

  def testPackSequenceAs_notIterableError(self):
    with self.assertRaisesRegex(TypeError, self.bad_pack_pattern):
      nest.pack_sequence_as("hi", "bye")

  def testPackSequenceAs_wrongLengthsError(self):
    with self.assertRaisesRegex(
        ValueError, "Structure had 2 atoms, but flat_sequence had 3 items."):
      nest.pack_sequence_as(["hello", "world"],
                            ["and", "goodbye", "again"])

  def testPackSequenceAs_CompositeTensor(self):
    val = ragged_tensor.RaggedTensor.from_row_splits(values=[1],
                                                     row_splits=[0, 1])
    with self.assertRaisesRegex(
        ValueError, "Structure had 2 atoms, but flat_sequence had 1 items."):
      nest.pack_sequence_as(val, [val], expand_composites=True)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testIsNested(self):
    self.assertFalse(nest.is_nested("1234"))
    self.assertTrue(nest.is_nested([1, 3, [4, 5]]))
    self.assertTrue(nest.is_nested(((7, 8), (5, 6))))
    self.assertTrue(nest.is_nested([]))
    self.assertTrue(nest.is_nested({"a": 1, "b": 2}))
    self.assertTrue(nest.is_nested({"a": 1, "b": 2}.keys()))
    self.assertTrue(nest.is_nested({"a": 1, "b": 2}.values()))
    self.assertTrue(nest.is_nested({"a": 1, "b": 2}.items()))
    self.assertFalse(nest.is_nested(set([1, 2])))
    ones = array_ops.ones([2, 3])
    self.assertFalse(nest.is_nested(ones))
    self.assertFalse(nest.is_nested(math_ops.tanh(ones)))
    self.assertFalse(nest.is_nested(np.ones((4, 5))))

  @parameterized.parameters({"mapping_type": _CustomMapping},
                            {"mapping_type": dict})
  def testFlattenDictItems(self, mapping_type):
    dictionary = mapping_type({(4, 5, (6, 8)): ("a", "b", ("c", "d"))})
    flat = {4: "a", 5: "b", 6: "c", 8: "d"}
    self.assertEqual(nest.flatten_dict_items(dictionary), flat)

    with self.assertRaises(TypeError):
      nest.flatten_dict_items(4)

    bad_dictionary = mapping_type({(4, 5, (4, 8)): ("a", "b", ("c", "d"))})
    with self.assertRaisesRegex(ValueError, "not unique"):
      nest.flatten_dict_items(bad_dictionary)

    another_bad_dictionary = mapping_type({
        (4, 5, (6, 8)): ("a", "b", ("c", ("d", "e")))
    })
    with self.assertRaisesRegex(
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

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testAssertSameStructure(self):
    structure1 = (((1, 2), 3), 4, (5, 6))
    structure2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
    structure_different_num_elements = ("spam", "eggs")
    structure_different_nesting = (((1, 2), 3), 4, 5, (6,))
    nest.assert_same_structure(structure1, structure2)
    nest.assert_same_structure("abc", 1.0)
    nest.assert_same_structure("abc", np.array([0, 1]))
    nest.assert_same_structure("abc", constant_op.constant([0, 1]))

    with self.assertRaisesRegex(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         "More specifically: Substructure "
         r'"type=tuple str=\(\(1, 2\), 3\)" is a sequence, while '
         'substructure "type=str str=spam" is not\n'
         "Entire first structure:\n"
         r"\(\(\(\., \.\), \.\), \., \(\., \.\)\)\n"
         "Entire second structure:\n"
         r"\(\., \.\)")):
      nest.assert_same_structure(structure1, structure_different_num_elements)

    with self.assertRaisesRegex(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         r'More specifically: Substructure "type=list str=\[0, 1\]" '
         r'is a sequence, while substructure "type=ndarray str=\[0 1\]" '
         "is not")):
      nest.assert_same_structure([0, 1], np.array([0, 1]))

    with self.assertRaisesRegex(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         r'More specifically: Substructure "type=list str=\[0, 1\]" '
         'is a sequence, while substructure "type=int str=0" '
         "is not")):
      nest.assert_same_structure(0, [0, 1])

    self.assertRaises(TypeError, nest.assert_same_structure, (0, 1), [0, 1])

    with self.assertRaisesRegex(ValueError,
                                ("don't have the same nested structure\\.\n\n"
                                 "First structure: .*?\n\nSecond structure: ")):
      nest.assert_same_structure(structure1, structure_different_nesting)

    self.assertRaises(TypeError, nest.assert_same_structure, (0, 1),
                      NestTest.Named0ab("a", "b"))

    nest.assert_same_structure(NestTest.Named0ab(3, 4),
                               NestTest.Named0ab("a", "b"))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      NestTest.Named0ab(3, 4), NestTest.Named1ab(3, 4))

    with self.assertRaisesRegex(ValueError,
                                ("don't have the same nested structure\\.\n\n"
                                 "First structure: .*?\n\nSecond structure: ")):
      nest.assert_same_structure(NestTest.Named0ab(3, 4),
                                 NestTest.Named0ab([3], 4))

    with self.assertRaisesRegex(ValueError,
                                ("don't have the same nested structure\\.\n\n"
                                 "First structure: .*?\n\nSecond structure: ")):
      nest.assert_same_structure([[3], 4], [3, [4]])

    structure1_list = [[[1, 2], 3], 4, [5, 6]]
    with self.assertRaisesRegex(TypeError, "don't have the same sequence type"):
      nest.assert_same_structure(structure1, structure1_list)
    nest.assert_same_structure(structure1, structure2, check_types=False)
    nest.assert_same_structure(structure1, structure1_list, check_types=False)

    with self.assertRaisesRegex(ValueError, "don't have the same set of keys"):
      nest.assert_same_structure({"a": 1}, {"b": 1})

    nest.assert_same_structure(NestTest.SameNameab(0, 1),
                               NestTest.SameNameab2(2, 3))

    # This assertion is expected to pass: two namedtuples with the same
    # name and field names are considered to be identical.
    nest.assert_same_structure(
        NestTest.SameNameab(NestTest.SameName1xy(0, 1), 2),
        NestTest.SameNameab2(NestTest.SameName1xy2(2, 3), 4))

    expected_message = "The two structures don't have the same.*"
    with self.assertRaisesRegex(ValueError, expected_message):
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

  def testHeterogeneousComparison(self):
    nest.assert_same_structure({"a": 4}, _CustomMapping(a=3))
    nest.assert_same_structure(_CustomMapping(b=3), {"b": 4})

  @test_util.assert_no_new_pyobjects_executing_eagerly()
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

    structure3 = collections.defaultdict(list)
    structure3["a"] = [1, 2, 3, 4]
    structure3["b"] = [2, 3, 4, 5]

    expected_structure3 = collections.defaultdict(list)
    expected_structure3["a"] = [2, 3, 4, 5]
    expected_structure3["b"] = [3, 4, 5, 6]
    self.assertEqual(expected_structure3,
                     nest.map_structure(lambda x: x + 1, structure3))

    # Empty structures
    self.assertEqual((), nest.map_structure(lambda x: x + 1, ()))
    self.assertEqual([], nest.map_structure(lambda x: x + 1, []))
    self.assertEqual({}, nest.map_structure(lambda x: x + 1, {}))
    self.assertEqual(NestTest.EmptyNT(), nest.map_structure(lambda x: x + 1,
                                                            NestTest.EmptyNT()))

    # This is checking actual equality of types, empty list != empty tuple
    self.assertNotEqual((), nest.map_structure(lambda x: x + 1, []))

    with self.assertRaisesRegex(TypeError, "callable"):
      nest.map_structure("bad", structure1_plus1)

    with self.assertRaisesRegex(ValueError, "at least one structure"):
      nest.map_structure(lambda x: x)

    with self.assertRaisesRegex(ValueError, "same number of elements"):
      nest.map_structure(lambda x, y: None, (3, 4), (3, 4, 5))

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, 3, (3,))

    with self.assertRaisesRegex(TypeError, "same sequence type"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), [(3, 4), 5])

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)))

    structure1_list = [[[1, 2], 3], 4, [5, 6]]
    with self.assertRaisesRegex(TypeError, "same sequence type"):
      nest.map_structure(lambda x, y: None, structure1, structure1_list)

    nest.map_structure(lambda x, y: None, structure1, structure1_list,
                       check_types=False)

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)),
                         check_types=False)

    with self.assertRaisesRegex(ValueError, "Only valid keyword argument.*foo"):
      nest.map_structure(lambda x: None, structure1, foo="a")

    with self.assertRaisesRegex(ValueError, "Only valid keyword argument.*foo"):
      nest.map_structure(lambda x: None, structure1, check_types=False, foo="a")

  ABTuple = collections.namedtuple("ab_tuple", "a, b")  # pylint: disable=invalid-name

  @test_util.assert_no_new_pyobjects_executing_eagerly()
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
    # Test requires placeholders and thus requires graph mode
    with ops.Graph().as_default():
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

      with self.cached_session() as sess:
        output_np = sess.run(output, feed_dict=feed_dict)
      self.assertAllClose(output_np[0],
                          feed_dict[inp_a][0] + feed_dict[inp_b][0])
      self.assertAllClose(output_np[1],
                          feed_dict[inp_a][1] + feed_dict[inp_b][1])

  def testAssertShallowStructure(self):
    inp_ab = ["a", "b"]
    inp_abc = ["a", "b", "c"]
    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        nest.STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
            input_length=len(inp_ab), shallow_length=len(inp_abc)
        ),
    ):
      nest.assert_shallow_structure(inp_abc, inp_ab)

    inp_ab1 = [(1, 1), (2, 2)]
    inp_ab2 = [[1, 1], [2, 2]]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(inp_ab2[0]), input_type=type(inp_ab1[0])
        ),
    ):
      nest.assert_shallow_structure(inp_ab2, inp_ab1)
    nest.assert_shallow_structure(inp_ab2, inp_ab1, check_types=False)

    inp_ab1 = {"a": (1, 1), "b": {"c": (2, 2)}}
    inp_ab2 = {"a": (1, 1), "b": {"d": (2, 2)}}
    with self.assertRaisesWithLiteralMatch(
        ValueError, nest.SHALLOW_TREE_HAS_INVALID_KEYS.format(["d"])
    ):
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

    # This assertion is expected to pass: two list-types with same number
    # of fields are considered identical.
    inp_shallow = _CustomList([1, 2])
    inp_deep = [1, 2]
    nest.assert_shallow_structure(inp_shallow, inp_deep, check_types=False)
    nest.assert_shallow_structure(inp_shallow, inp_deep, check_types=True)

    # This assertion is expected to pass: a VariableSpec with alias_id and
    # a Variable are considered identical.
    inp_shallow = resource_variable_ops.VariableSpec(None, alias_id=0)
    inp_deep = resource_variable_ops.ResourceVariable(1.)
    nest.assert_shallow_structure(inp_shallow, inp_deep,
                                  expand_composites=False)
    nest.assert_shallow_structure(inp_shallow, inp_deep,
                                  expand_composites=True)

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
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = "input_tree"
    shallow_tree = ["shallow_tree_9", "shallow_tree_8"]
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = [9]
    expected_message = ("If shallow structure is a sequence, input must also "
                        "be a sequence. Input has type: <(type|class) 'int'>.")
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = 0
    shallow_tree = [9, 8]
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = [(1,), (2,), 3]
    shallow_tree = [(1,), (2,)]
    expected_message = nest.STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
        input_length=len(input_tree), shallow_length=len(shallow_tree)
    )
    with self.assertRaisesRegex(ValueError, expected_message):  # pylint: disable=g-error-prone-assert-raises
      nest.assert_shallow_structure(shallow_tree, input_tree)

  def testFlattenWithTuplePathsUpTo(self):
    def get_paths_and_values(shallow_tree, input_tree):
      path_value_pairs = nest.flatten_with_tuple_paths_up_to(
          shallow_tree, input_tree)
      paths = [p for p, _ in path_value_pairs]
      values = [v for _, v in path_value_pairs]
      return paths, values

    # Shallow tree ends at scalar.
    input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
    shallow_tree = [[True, True], [False, True]]
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths,
                     [(0, 0), (0, 1), (1, 0), (1, 1)])
    self.assertEqual(flattened_input_tree, [[2, 2], [3, 3], [4, 9], [5, 5]])
    self.assertEqual(flattened_shallow_tree_paths,
                     [(0, 0), (0, 1), (1, 0), (1, 1)])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

    # Shallow tree ends at string.
    input_tree = [[("a", 1), [("b", 2), [("c", 3), [("d", 4)]]]]]
    shallow_tree = [["level_1", ["level_2", ["level_3", ["level_4"]]]]]
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    input_tree_flattened_paths = [p for p, _ in
                                  nest.flatten_with_tuple_paths(input_tree)]
    input_tree_flattened = nest.flatten(input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [(0, 0), (0, 1, 0), (0, 1, 1, 0), (0, 1, 1, 1, 0)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [("a", 1), ("b", 2), ("c", 3), ("d", 4)])

    self.assertEqual(input_tree_flattened_paths,
                     [(0, 0, 0), (0, 0, 1),
                      (0, 1, 0, 0), (0, 1, 0, 1),
                      (0, 1, 1, 0, 0), (0, 1, 1, 0, 1),
                      (0, 1, 1, 1, 0, 0), (0, 1, 1, 1, 0, 1)])
    self.assertEqual(input_tree_flattened, ["a", 1, "b", 2, "c", 3, "d", 4])

    # Make sure dicts are correctly flattened, yielding values, not keys.
    input_tree = {"a": 1, "b": {"c": 2}, "d": [3, (4, 5)]}
    shallow_tree = {"a": 0, "b": 0, "d": [0, 0]}
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",), ("b",), ("d", 0), ("d", 1)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [1, {"c": 2}, 3, (4, 5)])

    # Namedtuples.
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    input_tree = ab_tuple(a=[0, 1], b=2)
    shallow_tree = ab_tuple(a=0, b=1)
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",), ("b",)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [[0, 1], 2])

    # Nested dicts, OrderedDicts and namedtuples.
    input_tree = collections.OrderedDict(
        [("a", ab_tuple(a=[0, {"b": 1}], b=2)),
         ("c", {"d": 3, "e": collections.OrderedDict([("f", 4)])})])
    shallow_tree = input_tree
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a", "a", 0),
                      ("a", "a", 1, "b"),
                      ("a", "b"),
                      ("c", "d"),
                      ("c", "e", "f")])
    self.assertEqual(input_tree_flattened_as_shallow_tree, [0, 1, 2, 3, 4])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", {"d": 3, "e": 1})])
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",),
                      ("c", "d"),
                      ("c", "e")])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      3,
                      collections.OrderedDict([("f", 4)])])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", 0)])
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",), ("c",)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      {"d": 3, "e": collections.OrderedDict([("f", 4)])}])

    ## Shallow non-list edge-case.
    # Using iterable elements.
    input_tree = ["input_tree"]
    shallow_tree = "shallow_tree"
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = ["input_tree_0", "input_tree_1"]
    shallow_tree = "shallow_tree"
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Test case where len(shallow_tree) < len(input_tree)
    input_tree = {"a": "A", "b": "B", "c": "C"}
    shallow_tree = {"a": 1, "c": 2}

    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        nest.STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
            input_length=len(input_tree), shallow_length=len(shallow_tree)
        ),
    ):
      get_paths_and_values(shallow_tree, input_tree)

    # Using non-iterable elements.
    input_tree = [0]
    shallow_tree = 9
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = [0, 1]
    shallow_tree = 9
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Both non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = "shallow_tree"
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = 0
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Input non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = ["shallow_tree"]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        nest.IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree)),
    ):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = "input_tree"
    shallow_tree = ["shallow_tree_9", "shallow_tree_8"]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        nest.IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree)),
    ):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,), (1,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = [9]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        nest.IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree)),
    ):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = 0
    shallow_tree = [9, 8]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        nest.IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree)),
    ):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,), (1,)])
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

    # Dicts.
    inp_val = dict(a=2, b=3)
    inp_ops = dict(a=dict(add=1, mul=2), b=dict(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val,
        lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)
    self.assertEqual(out["a"], 6)
    self.assertEqual(out["b"], 15)

    # Non-equal dicts.
    inp_val = dict(a=2, b=3)
    inp_ops = dict(a=dict(add=1, mul=2), c=dict(add=2, mul=3))
    with self.assertRaisesWithLiteralMatch(
        ValueError, nest.SHALLOW_TREE_HAS_INVALID_KEYS.format(["b"])
    ):
      nest.map_structure_up_to(
          inp_val,
          lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)

    # Dict+custom mapping.
    inp_val = dict(a=2, b=3)
    inp_ops = _CustomMapping(a=dict(add=1, mul=2), b=dict(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val,
        lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)
    self.assertEqual(out["a"], 6)
    self.assertEqual(out["b"], 15)

    # Non-equal dict/mapping.
    inp_val = dict(a=2, b=3)
    inp_ops = _CustomMapping(a=dict(add=1, mul=2), c=dict(add=2, mul=3))
    with self.assertRaisesWithLiteralMatch(
        ValueError, nest.SHALLOW_TREE_HAS_INVALID_KEYS.format(["b"])
    ):
      nest.map_structure_up_to(
          inp_val,
          lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)

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

    with self.assertRaisesRegex(TypeError, "returned structure"):
      nest.get_traverse_shallow_structure(lambda _: [True], 0)

    with self.assertRaisesRegex(TypeError, "returned a non-bool scalar"):
      nest.get_traverse_shallow_structure(lambda _: 1, [1])

    with self.assertRaisesRegex(TypeError,
                                "didn't return a depth=1 structure of bools"):
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

  # We cannot define namedtuples within @parameterized argument lists.
  # pylint: disable=invalid-name
  Foo = collections.namedtuple("Foo", ["a", "b"])
  Bar = collections.namedtuple("Bar", ["c", "d"])
  # pylint: enable=invalid-name

  @parameterized.parameters([
      dict(inputs=[], expected=[]),
      dict(inputs=[23, "42"], expected=[("0", 23), ("1", "42")]),
      dict(inputs=[[[[108]]]], expected=[("0/0/0/0", 108)]),
      dict(inputs=Foo(a=3, b=Bar(c=23, d=42)),
           expected=[("a", 3), ("b/c", 23), ("b/d", 42)]),
      dict(inputs=Foo(a=Bar(c=23, d=42), b=Bar(c=0, d="thing")),
           expected=[("a/c", 23), ("a/d", 42), ("b/c", 0), ("b/d", "thing")]),
      dict(inputs=Bar(c=42, d=43),
           expected=[("c", 42), ("d", 43)]),
      dict(inputs=Bar(c=[42], d=43),
           expected=[("c/0", 42), ("d", 43)]),
  ])
  def testFlattenWithStringPaths(self, inputs, expected):
    self.assertEqual(
        nest.flatten_with_joined_string_paths(inputs, separator="/"),
        expected)

  @parameterized.parameters([
      dict(inputs=[], expected=[]),
      dict(inputs=[23, "42"], expected=[((0,), 23), ((1,), "42")]),
      dict(inputs=[[[[108]]]], expected=[((0, 0, 0, 0), 108)]),
      dict(inputs=Foo(a=3, b=Bar(c=23, d=42)),
           expected=[(("a",), 3), (("b", "c"), 23), (("b", "d"), 42)]),
      dict(inputs=Foo(a=Bar(c=23, d=42), b=Bar(c=0, d="thing")),
           expected=[(("a", "c"), 23), (("a", "d"), 42), (("b", "c"), 0),
                     (("b", "d"), "thing")]),
      dict(inputs=Bar(c=42, d=43),
           expected=[(("c",), 42), (("d",), 43)]),
      dict(inputs=Bar(c=[42], d=43),
           expected=[(("c", 0), 42), (("d",), 43)]),
  ])
  def testFlattenWithTuplePaths(self, inputs, expected):
    self.assertEqual(nest.flatten_with_tuple_paths(inputs), expected)

  @parameterized.named_parameters(
      ("tuples", (1, 2), (3, 4), True, (("0", 4), ("1", 6))),
      ("dicts", {"a": 1, "b": 2}, {"b": 4, "a": 3}, True,
       {"a": ("a", 4), "b": ("b", 6)}),
      ("mixed", (1, 2), [3, 4], False, (("0", 4), ("1", 6))),
      ("nested",
       {"a": [2, 3], "b": [1, 2, 3]}, {"b": [5, 6, 7], "a": [8, 9]}, True,
       {"a": [("a/0", 10), ("a/1", 12)],
        "b": [("b/0", 6), ("b/1", 8), ("b/2", 10)]}))
  def testMapWithPathsCompatibleStructures(self, s1, s2, check_types, expected):
    def format_sum(path, *values):
      return (path, sum(values))
    result = nest.map_structure_with_paths(format_sum, s1, s2,
                                           check_types=check_types)
    self.assertEqual(expected, result)

  @parameterized.named_parameters(
      ("tuples", (1, 2, 3), (4, 5), ValueError),
      ("dicts", {"a": 1}, {"b": 2}, ValueError),
      ("mixed", (1, 2), [3, 4], TypeError),
      ("nested",
       {"a": [2, 3, 4], "b": [1, 3]},
       {"b": [5, 6], "a": [8, 9]},
       ValueError
      ))
  def testMapWithPathsIncompatibleStructures(self, s1, s2, error_type):
    with self.assertRaises(error_type):
      nest.map_structure_with_paths(lambda path, *s: 0, s1, s2)

  @parameterized.named_parameters([
      dict(testcase_name="Tuples", s1=(1, 2), s2=(3, 4),
           check_types=True, expected=(((0,), 4), ((1,), 6))),
      dict(testcase_name="Dicts", s1={"a": 1, "b": 2}, s2={"b": 4, "a": 3},
           check_types=True, expected={"a": (("a",), 4), "b": (("b",), 6)}),
      dict(testcase_name="Mixed", s1=(1, 2), s2=[3, 4],
           check_types=False, expected=(((0,), 4), ((1,), 6))),
      dict(testcase_name="Nested",
           s1={"a": [2, 3], "b": [1, 2, 3]},
           s2={"b": [5, 6, 7], "a": [8, 9]},
           check_types=True,
           expected={"a": [(("a", 0), 10), (("a", 1), 12)],
                     "b": [(("b", 0), 6), (("b", 1), 8), (("b", 2), 10)]}),
  ])
  def testMapWithTuplePathsCompatibleStructures(
      self, s1, s2, check_types, expected):
    def path_and_sum(path, *values):
      return path, sum(values)
    result = nest.map_structure_with_tuple_paths(
        path_and_sum, s1, s2, check_types=check_types)
    self.assertEqual(expected, result)

  @parameterized.named_parameters([
      dict(testcase_name="Tuples", s1=(1, 2, 3), s2=(4, 5),
           error_type=ValueError),
      dict(testcase_name="Dicts", s1={"a": 1}, s2={"b": 2},
           error_type=ValueError),
      dict(testcase_name="Mixed", s1=(1, 2), s2=[3, 4], error_type=TypeError),
      dict(testcase_name="Nested",
           s1={"a": [2, 3, 4], "b": [1, 3]},
           s2={"b": [5, 6], "a": [8, 9]},
           error_type=ValueError)
  ])
  def testMapWithTuplePathsIncompatibleStructures(self, s1, s2, error_type):
    with self.assertRaises(error_type):
      nest.map_structure_with_tuple_paths(lambda path, *s: 0, s1, s2)

  def testFlattenCustomSequenceThatRaisesException(self):  # b/140746865
    seq = _CustomSequenceThatRaisesException()
    with self.assertRaisesRegex(ValueError, "Cannot get item"):
      nest.flatten(seq)

  def testListToTuple(self):
    input_sequence = [1, (2, {3: [4, 5, (6,)]}, None, 7, [[[8]]])]
    expected = (1, (2, {3: (4, 5, (6,))}, None, 7, (((8,),),)))
    nest.assert_same_structure(
        nest.list_to_tuple(input_sequence),
        expected,
    )

  def testInvalidCheckTypes(self):
    with self.assertRaises((ValueError, TypeError)):
      nest.assert_same_structure(
          nest1=array_ops.zeros((1)),
          nest2=array_ops.ones((1, 1, 1)),
          check_types=array_ops.ones((2)))
    with self.assertRaises((ValueError, TypeError)):
      nest.assert_same_structure(
          nest1=array_ops.zeros((1)),
          nest2=array_ops.ones((1, 1, 1)),
          expand_composites=array_ops.ones((2)))

  def testIsNamedtuple(self):
    # A classic namedtuple.
    Foo = collections.namedtuple("Foo", ["a", "b"])
    self.assertTrue(nest.is_namedtuple(Foo(1, 2)))

    # A subclass of it.
    class SubFoo(Foo):

      def extra_method(self, x):
        return self.a + x

    self.assertTrue(nest.is_namedtuple(SubFoo(1, 2)))

    # A typing.NamedTuple.
    class TypedFoo(NamedTuple):
      a: int
      b: int
    self.assertTrue(nest.is_namedtuple(TypedFoo(1, 2)))

    # Their types are not namedtuple values themselves.
    self.assertFalse(nest.is_namedtuple(Foo))
    self.assertFalse(nest.is_namedtuple(SubFoo))
    self.assertFalse(nest.is_namedtuple(TypedFoo))

    # These values don't have namedtuple types.
    self.assertFalse(nest.is_namedtuple(123))
    self.assertFalse(nest.is_namedtuple("abc"))
    self.assertFalse(nest.is_namedtuple((123, "abc")))

    class SomethingElseWithFields(tuple):

      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fields = [1, 2, 3]  # Not str, as expected for a namedtuple.

    self.assertFalse(nest.is_namedtuple(SomethingElseWithFields()))

  def testSameNamedtuples(self):
    # A classic namedtuple and an equivalent cppy.
    Foo1 = collections.namedtuple("Foo", ["a", "b"])
    Foo2 = collections.namedtuple("Foo", ["a", "b"])
    self.assertTrue(nest.same_namedtuples(Foo1(1, 2), Foo1(3, 4)))
    self.assertTrue(nest.same_namedtuples(Foo1(1, 2), Foo2(3, 4)))

    # Non-equivalent namedtuples.
    Bar = collections.namedtuple("Bar", ["a", "b"])
    self.assertFalse(nest.same_namedtuples(Foo1(1, 2), Bar(1, 2)))
    FooXY = collections.namedtuple("Foo", ["x", "y"])
    self.assertFalse(nest.same_namedtuples(Foo1(1, 2), FooXY(1, 2)))

    # An equivalent subclass from the typing module
    class Foo(NamedTuple):
      a: int
      b: int
    self.assertTrue(nest.same_namedtuples(Foo1(1, 2), Foo(3, 4)))


class NestBenchmark(test.Benchmark):

  def run_and_report(self, s1, s2, name):
    burn_iter, test_iter = 100, 30000

    for _ in range(burn_iter):
      nest.assert_same_structure(s1, s2)

    t0 = time.time()
    for _ in range(test_iter):
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
