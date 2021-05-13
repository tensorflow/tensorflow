# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
import functools

import numpy as np
import wrapt
from absl.testing import parameterized

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.platform import test
from tensorflow.python.util.compat import collections_abc


# NOTE(mrry): Arguments of parameterized tests are lifted into lambdas to make
# sure they are not executed before the (eager- or graph-mode) test environment
# has been set up.
#


def _test_flat_structure_combinations():
  cases = [
      ("Tensor", lambda: constant_op.constant(37.0),
       lambda: tensor_spec.TensorSpec, lambda: [dtypes.float32], lambda: [[]]),
      ("TensorArray", lambda: tensor_array_ops.TensorArray(
          dtype=dtypes.float32, element_shape=(3,), size=0),
       lambda: tensor_array_ops.TensorArraySpec, lambda: [dtypes.variant],
       lambda: [[]]),
      ("SparseTensor", lambda: sparse_tensor.SparseTensor(
          indices=[[3, 4]], values=[-1], dense_shape=[4, 5]),
       lambda: sparse_tensor.SparseTensorSpec, lambda: [dtypes.variant],
       lambda: [None]),
      ("RaggedTensor", lambda: ragged_factory_ops.constant([[1, 2], [], [4]]),
       lambda: ragged_tensor.RaggedTensorSpec, lambda: [dtypes.variant],
       lambda: [None]),
      ("Nested_0", lambda:
       (constant_op.constant(37.0), constant_op.constant([1, 2, 3])),
       lambda: tuple, lambda: [dtypes.float32, dtypes.int32],
       lambda: [[], [3]]),
      ("Nested_1", lambda: {
          "a": constant_op.constant(37.0),
          "b": constant_op.constant([1, 2, 3])
      }, lambda: dict, lambda: [dtypes.float32, dtypes.int32],
       lambda: [[], [3]]),
      ("Nested_2", lambda: {
          "a":
              constant_op.constant(37.0),
          "b": (sparse_tensor.SparseTensor(
              indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
                sparse_tensor.SparseTensor(
                    indices=[[3, 4]], values=[-1], dense_shape=[4, 5]))
      }, lambda: dict, lambda: [dtypes.float32, dtypes.variant, dtypes.variant],
       lambda: [[], None, None]),
  ]

  def reduce_fn(x, y):
    # workaround for long line
    name, value_fn = y[:2]
    expected_structure_fn, expected_types_fn, expected_shapes_fn = y[2:]
    return x + combinations.combine(
        value_fn=combinations.NamedObject("value_fn.{}".format(name), value_fn),
        expected_structure_fn=combinations.NamedObject(
            "expected_structure_fn.{}".format(name), expected_structure_fn),
        expected_types_fn=combinations.NamedObject(
            "expected_types_fn.{}".format(name), expected_types_fn),
        expected_shapes_fn=combinations.NamedObject(
            "expected_shapes_fn.{}".format(name), expected_shapes_fn))

  return functools.reduce(reduce_fn, cases, [])


def _test_is_compatible_with_structure_combinations():
  cases = [
      ("Tensor", lambda: constant_op.constant(37.0), lambda: [
          constant_op.constant(38.0),
          array_ops.placeholder(dtypes.float32),
          variables.Variable(100.0), 42.0,
          np.array(42.0, dtype=np.float32)
      ], lambda: [constant_op.constant([1.0, 2.0]),
                  constant_op.constant(37)]),
      ("TensorArray", lambda: tensor_array_ops.TensorArray(
          dtype=dtypes.float32, element_shape=(3,), size=0), lambda: [
              tensor_array_ops.TensorArray(
                  dtype=dtypes.float32, element_shape=(3,), size=0),
              tensor_array_ops.TensorArray(
                  dtype=dtypes.float32, element_shape=(3,), size=10)
          ], lambda: [
              tensor_array_ops.TensorArray(
                  dtype=dtypes.int32, element_shape=(3,), size=0),
              tensor_array_ops.TensorArray(
                  dtype=dtypes.float32, element_shape=(), size=0)
          ]),
      ("SparseTensor", lambda: sparse_tensor.SparseTensor(
          indices=[[3, 4]], values=[-1], dense_shape=[4, 5]),
       lambda: [
           sparse_tensor.SparseTensor(
               indices=[[1, 1], [3, 4]], values=[10, -1], dense_shape=[4, 5]),
           sparse_tensor.SparseTensorValue(
               indices=[[1, 1], [3, 4]], values=[10, -1], dense_shape=[4, 5]),
           array_ops.sparse_placeholder(dtype=dtypes.int32),
           array_ops.sparse_placeholder(dtype=dtypes.int32, shape=[None, None])
       ], lambda: [
           constant_op.constant(37, shape=[4, 5]),
           sparse_tensor.SparseTensor(
               indices=[[3, 4]], values=[-1], dense_shape=[5, 6]),
           array_ops.sparse_placeholder(
               dtype=dtypes.int32, shape=[None, None, None]),
           sparse_tensor.SparseTensor(
               indices=[[3, 4]], values=[-1.0], dense_shape=[4, 5])
       ]),
      ("RaggedTensor", lambda: ragged_factory_ops.constant([[1, 2], [], [3]]),
       lambda: [
           ragged_factory_ops.constant([[1, 2], [3, 4], []]),
           ragged_factory_ops.constant([[1], [2, 3, 4], [5]]),
       ], lambda: [
           ragged_factory_ops.constant(1),
           ragged_factory_ops.constant([1, 2]),
           ragged_factory_ops.constant([[1], [2]]),
           ragged_factory_ops.constant([["a", "b"]]),
       ]),
      ("Nested", lambda: {
          "a": constant_op.constant(37.0),
          "b": constant_op.constant([1, 2, 3])
      }, lambda: [{
          "a": constant_op.constant(15.0),
          "b": constant_op.constant([4, 5, 6])
      }], lambda: [{
          "a": constant_op.constant(15.0),
          "b": constant_op.constant([4, 5, 6, 7])
      }, {
          "a": constant_op.constant(15),
          "b": constant_op.constant([4, 5, 6])
      }, {
          "a":
              constant_op.constant(15),
          "b":
              sparse_tensor.SparseTensor(
                  indices=[[0], [1], [2]], values=[4, 5, 6], dense_shape=[3])
      }, (constant_op.constant(15.0), constant_op.constant([4, 5, 6]))]),
  ]

  def reduce_fn(x, y):
    name, original_value_fn, compatible_values_fn, incompatible_values_fn = y
    return x + combinations.combine(
        original_value_fn=combinations.NamedObject(
            "original_value_fn.{}".format(name), original_value_fn),
        compatible_values_fn=combinations.NamedObject(
            "compatible_values_fn.{}".format(name), compatible_values_fn),
        incompatible_values_fn=combinations.NamedObject(
            "incompatible_values_fn.{}".format(name), incompatible_values_fn))

  return functools.reduce(reduce_fn, cases, [])


def _test_structure_from_value_equality_combinations():
  cases = [
      ("Tensor", lambda: constant_op.constant(37.0),
       lambda: constant_op.constant(42.0), lambda: constant_op.constant([5])),
      ("TensorArray", lambda: tensor_array_ops.TensorArray(
          dtype=dtypes.float32, element_shape=(3,), size=0),
       lambda: tensor_array_ops.TensorArray(
           dtype=dtypes.float32, element_shape=(3,), size=0),
       lambda: tensor_array_ops.TensorArray(
           dtype=dtypes.int32, element_shape=(), size=0)),
      ("SparseTensor", lambda: sparse_tensor.SparseTensor(
          indices=[[3, 4]], values=[-1], dense_shape=[4, 5]),
       lambda: sparse_tensor.SparseTensor(
           indices=[[1, 2]], values=[42], dense_shape=[4, 5]), lambda:
       sparse_tensor.SparseTensor(indices=[[3]], values=[-1], dense_shape=[5]),
       lambda: sparse_tensor.SparseTensor(
           indices=[[3, 4]], values=[1.0], dense_shape=[4, 5])),
      ("RaggedTensor", lambda: ragged_factory_ops.constant([[[1, 2]], [[3]]]),
       lambda: ragged_factory_ops.constant([[[5]], [[8], [3, 2]]]),
       lambda: ragged_factory_ops.constant([[[1]], [[2], [3]]], ragged_rank=1),
       lambda: ragged_factory_ops.constant([[[1.0, 2.0]], [[3.0]]]),
       lambda: ragged_factory_ops.constant([[[1]], [[2]], [[3]]])),
      ("Nested", lambda: {
          "a": constant_op.constant(37.0),
          "b": constant_op.constant([1, 2, 3])
      }, lambda: {
          "a": constant_op.constant(42.0),
          "b": constant_op.constant([4, 5, 6])
      }, lambda: {
          "a": constant_op.constant([1, 2, 3]),
          "b": constant_op.constant(37.0)
      }),
  ]

  def reduce_fn(x, y):
    name, value1_fn, value2_fn, *not_equal_value_fns = y
    return x + combinations.combine(
        value1_fn=combinations.NamedObject("value1_fn.{}".format(name),
                                           value1_fn),
        value2_fn=combinations.NamedObject("value2_fn.{}".format(name),
                                           value2_fn),
        not_equal_value_fns=combinations.NamedObject(
            "not_equal_value_fns.{}".format(name), not_equal_value_fns))

  return functools.reduce(reduce_fn, cases, [])


def _test_ragged_structure_inequality_combinations():
  cases = [
      (ragged_tensor.RaggedTensorSpec(None, dtypes.int32, 1),
       ragged_tensor.RaggedTensorSpec(None, dtypes.int32, 2)),
      (ragged_tensor.RaggedTensorSpec([3, None], dtypes.int32, 1),
       ragged_tensor.RaggedTensorSpec([5, None], dtypes.int32, 1)),
      (ragged_tensor.RaggedTensorSpec(None, dtypes.int32, 1),
       ragged_tensor.RaggedTensorSpec(None, dtypes.float32, 1)),
  ]

  def reduce_fn(x, y):
    spec1, spec2 = y
    return x + combinations.combine(spec1=spec1, spec2=spec2)

  return functools.reduce(reduce_fn, cases, [])


def _test_hash_combinations():
  cases = [
      ("Tensor", lambda: constant_op.constant(37.0),
       lambda: constant_op.constant(42.0), lambda: constant_op.constant([5])),
      ("TensorArray", lambda: tensor_array_ops.TensorArray(
          dtype=dtypes.float32, element_shape=(3,), size=0),
       lambda: tensor_array_ops.TensorArray(
           dtype=dtypes.float32, element_shape=(3,), size=0),
       lambda: tensor_array_ops.TensorArray(
           dtype=dtypes.int32, element_shape=(), size=0)),
      ("SparseTensor", lambda: sparse_tensor.SparseTensor(
          indices=[[3, 4]], values=[-1], dense_shape=[4, 5]),
       lambda: sparse_tensor.SparseTensor(
           indices=[[1, 2]], values=[42], dense_shape=[4, 5]), lambda:
       sparse_tensor.SparseTensor(indices=[[3]], values=[-1], dense_shape=[5])),
      ("Nested", lambda: {
          "a": constant_op.constant(37.0),
          "b": constant_op.constant([1, 2, 3])
      }, lambda: {
          "a": constant_op.constant(42.0),
          "b": constant_op.constant([4, 5, 6])
      }, lambda: {
          "a": constant_op.constant([1, 2, 3]),
          "b": constant_op.constant(37.0)
      }),
  ]

  def reduce_fn(x, y):
    name, value1_fn, value2_fn, value3_fn = y
    return x + combinations.combine(
        value1_fn=combinations.NamedObject("value1_fn.{}".format(name),
                                           value1_fn),
        value2_fn=combinations.NamedObject("value2_fn.{}".format(name),
                                           value2_fn),
        value3_fn=combinations.NamedObject("value3_fn.{}".format(name),
                                           value3_fn))

  return functools.reduce(reduce_fn, cases, [])


def _test_round_trip_conversion_combinations():
  cases = [
      (
          "Tensor",
          lambda: constant_op.constant(37.0),
      ),
      (
          "SparseTensor",
          lambda: sparse_tensor.SparseTensor(
              indices=[[3, 4]], values=[-1], dense_shape=[4, 5]),
      ),
      ("TensorArray", lambda: tensor_array_ops.TensorArray(
          dtype=dtypes.float32, element_shape=(), size=1).write(0, 7)),
      (
          "RaggedTensor",
          lambda: ragged_factory_ops.constant([[1, 2], [], [3]]),
      ),
      (
          "Nested_0",
          lambda: {
              "a": constant_op.constant(37.0),
              "b": constant_op.constant([1, 2, 3])
          },
      ),
      (
          "Nested_1",
          lambda: {
              "a":
                  constant_op.constant(37.0),
              "b": (sparse_tensor.SparseTensor(
                  indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
                    sparse_tensor.SparseTensor(
                        indices=[[3, 4]], values=[-1], dense_shape=[4, 5]))
          },
      ),
  ]

  def reduce_fn(x, y):
    name, value_fn = y
    return x + combinations.combine(
        value_fn=combinations.NamedObject("value_fn.{}".format(name), value_fn))

  return functools.reduce(reduce_fn, cases, [])


def _test_convert_legacy_structure_combinations():
  cases = [
      (dtypes.float32, tensor_shape.TensorShape([]), ops.Tensor,
       tensor_spec.TensorSpec([], dtypes.float32)),
      (dtypes.int32, tensor_shape.TensorShape([2,
                                               2]), sparse_tensor.SparseTensor,
       sparse_tensor.SparseTensorSpec([2, 2], dtypes.int32)),
      (dtypes.int32, tensor_shape.TensorShape([None, True, 2, 2]),
       tensor_array_ops.TensorArray,
       tensor_array_ops.TensorArraySpec([2, 2],
                                        dtypes.int32,
                                        dynamic_size=None,
                                        infer_shape=True)),
      (dtypes.int32, tensor_shape.TensorShape([True, None, 2, 2]),
       tensor_array_ops.TensorArray,
       tensor_array_ops.TensorArraySpec([2, 2],
                                        dtypes.int32,
                                        dynamic_size=True,
                                        infer_shape=None)),
      (dtypes.int32, tensor_shape.TensorShape([True, False, 2, 2]),
       tensor_array_ops.TensorArray,
       tensor_array_ops.TensorArraySpec([2, 2],
                                        dtypes.int32,
                                        dynamic_size=True,
                                        infer_shape=False)),
      (dtypes.int32, tensor_shape.TensorShape([2, None]),
       ragged_tensor.RaggedTensorSpec([2, None], dtypes.int32, 1),
       ragged_tensor.RaggedTensorSpec([2, None], dtypes.int32, 1)),
      ({
          "a": dtypes.float32,
          "b": (dtypes.int32, dtypes.string)
      }, {
          "a": tensor_shape.TensorShape([]),
          "b": (tensor_shape.TensorShape([2, 2]), tensor_shape.TensorShape([]))
      }, {
          "a": ops.Tensor,
          "b": (sparse_tensor.SparseTensor, ops.Tensor)
      }, {
          "a":
              tensor_spec.TensorSpec([], dtypes.float32),
          "b": (sparse_tensor.SparseTensorSpec([2, 2], dtypes.int32),
                tensor_spec.TensorSpec([], dtypes.string))
      })
  ]

  def reduce_fn(x, y):
    output_types, output_shapes, output_classes, expected_structure = y
    return x + combinations.combine(
        output_types=output_types,
        output_shapes=output_shapes,
        output_classes=output_classes,
        expected_structure=expected_structure)

  return functools.reduce(reduce_fn, cases, [])


def _test_batch_combinations():
  cases = [
      (tensor_spec.TensorSpec([], dtypes.float32), 32,
       tensor_spec.TensorSpec([32], dtypes.float32)),
      (tensor_spec.TensorSpec([], dtypes.float32), None,
       tensor_spec.TensorSpec([None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([None], dtypes.float32), 32,
       sparse_tensor.SparseTensorSpec([32, None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([4], dtypes.float32), None,
       sparse_tensor.SparseTensorSpec([None, 4], dtypes.float32)),
      (ragged_tensor.RaggedTensorSpec([2, None], dtypes.float32, 1), 32,
       ragged_tensor.RaggedTensorSpec([32, 2, None], dtypes.float32, 2)),
      (ragged_tensor.RaggedTensorSpec([4, None], dtypes.float32, 1), None,
       ragged_tensor.RaggedTensorSpec([None, 4, None], dtypes.float32, 2)),
      ({
          "a":
              tensor_spec.TensorSpec([], dtypes.float32),
          "b": (sparse_tensor.SparseTensorSpec([2, 2], dtypes.int32),
                tensor_spec.TensorSpec([], dtypes.string))
      }, 128, {
          "a":
              tensor_spec.TensorSpec([128], dtypes.float32),
          "b": (sparse_tensor.SparseTensorSpec([128, 2, 2], dtypes.int32),
                tensor_spec.TensorSpec([128], dtypes.string))
      }),
  ]

  def reduce_fn(x, y):
    element_structure, batch_size, expected_batched_structure = y
    return x + combinations.combine(
        element_structure=element_structure,
        batch_size=batch_size,
        expected_batched_structure=expected_batched_structure)

  return functools.reduce(reduce_fn, cases, [])


def _test_unbatch_combinations():
  cases = [
      (tensor_spec.TensorSpec([32], dtypes.float32),
       tensor_spec.TensorSpec([], dtypes.float32)),
      (tensor_spec.TensorSpec([None], dtypes.float32),
       tensor_spec.TensorSpec([], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([32, None], dtypes.float32),
       sparse_tensor.SparseTensorSpec([None], dtypes.float32)),
      (sparse_tensor.SparseTensorSpec([None, 4], dtypes.float32),
       sparse_tensor.SparseTensorSpec([4], dtypes.float32)),
      (ragged_tensor.RaggedTensorSpec([32, None, None], dtypes.float32, 2),
       ragged_tensor.RaggedTensorSpec([None, None], dtypes.float32, 1)),
      (ragged_tensor.RaggedTensorSpec([None, None, None], dtypes.float32, 2),
       ragged_tensor.RaggedTensorSpec([None, None], dtypes.float32, 1)),
      ({
          "a":
              tensor_spec.TensorSpec([128], dtypes.float32),
          "b": (sparse_tensor.SparseTensorSpec([128, 2, 2], dtypes.int32),
                tensor_spec.TensorSpec([None], dtypes.string))
      }, {
          "a":
              tensor_spec.TensorSpec([], dtypes.float32),
          "b": (sparse_tensor.SparseTensorSpec([2, 2], dtypes.int32),
                tensor_spec.TensorSpec([], dtypes.string))
      }),
  ]

  def reduce_fn(x, y):
    element_structure, expected_unbatched_structure = y
    return x + combinations.combine(
        element_structure=element_structure,
        expected_unbatched_structure=expected_unbatched_structure)

  return functools.reduce(reduce_fn, cases, [])


def _test_to_batched_tensor_list_combinations():
  cases = [
      ("Tensor", lambda: constant_op.constant([[1.0, 2.0], [3.0, 4.0]]),
       lambda: constant_op.constant([1.0, 2.0])),
      ("SparseTensor", lambda: sparse_tensor.SparseTensor(
          indices=[[0, 0], [1, 1]], values=[13, 27], dense_shape=[2, 2]),
       lambda: sparse_tensor.SparseTensor(
           indices=[[0]], values=[13], dense_shape=[2])),
      ("RaggedTensor", lambda: ragged_factory_ops.constant([[[1]], [[2]]]),
       lambda: ragged_factory_ops.constant([[1]])),
      ("Nest", lambda:
       (constant_op.constant([[1.0, 2.0], [3.0, 4.0]]),
        sparse_tensor.SparseTensor(
            indices=[[0, 0], [1, 1]], values=[13, 27], dense_shape=[2, 2])),
       lambda:
       (constant_op.constant([1.0, 2.0]),
        sparse_tensor.SparseTensor(indices=[[0]], values=[13], dense_shape=[2]))
      ),
  ]

  def reduce_fn(x, y):
    name, value_fn, element_0_fn = y
    return x + combinations.combine(
        value_fn=combinations.NamedObject("value_fn.{}".format(name), value_fn),
        element_0_fn=combinations.NamedObject("element_0_fn.{}".format(name),
                                              element_0_fn))

  return functools.reduce(reduce_fn, cases, [])

# TODO(jsimsa): Add tests for OptionalStructure and DatasetStructure.
class StructureTest(test_base.DatasetTestBase, parameterized.TestCase):

  # pylint: disable=g-long-lambda,protected-access
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_flat_structure_combinations()))
  def testFlatStructure(self, value_fn, expected_structure_fn,
                        expected_types_fn, expected_shapes_fn):
    value = value_fn()
    expected_structure = expected_structure_fn()
    expected_types = expected_types_fn()
    expected_shapes = expected_shapes_fn()
    s = structure.type_spec_from_value(value)
    self.assertIsInstance(s, expected_structure)
    flat_types = structure.get_flat_tensor_types(s)
    self.assertEqual(expected_types, flat_types)
    flat_shapes = structure.get_flat_tensor_shapes(s)
    self.assertLen(flat_shapes, len(expected_shapes))
    for expected, actual in zip(expected_shapes, flat_shapes):
      if expected is None:
        self.assertEqual(actual.ndims, None)
      else:
        self.assertEqual(actual.as_list(), expected)

  @combinations.generate(
      combinations.times(test_base.graph_only_combinations(),
                         _test_is_compatible_with_structure_combinations()))
  def testIsCompatibleWithStructure(self, original_value_fn,
                                    compatible_values_fn,
                                    incompatible_values_fn):
    original_value = original_value_fn()
    compatible_values = compatible_values_fn()
    incompatible_values = incompatible_values_fn()

    s = structure.type_spec_from_value(original_value)
    for compatible_value in compatible_values:
      self.assertTrue(
          structure.are_compatible(
              s, structure.type_spec_from_value(compatible_value)))
    for incompatible_value in incompatible_values:
      self.assertFalse(
          structure.are_compatible(
              s, structure.type_spec_from_value(incompatible_value)))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_structure_from_value_equality_combinations()))
  def testStructureFromValueEquality(self, value1_fn, value2_fn,
                                     not_equal_value_fns):
    # pylint: disable=g-generic-assert
    not_equal_value_fns = not_equal_value_fns._obj
    s1 = structure.type_spec_from_value(value1_fn())
    s2 = structure.type_spec_from_value(value2_fn())
    self.assertEqual(s1, s1)  # check __eq__ operator.
    self.assertEqual(s1, s2)  # check __eq__ operator.
    self.assertFalse(s1 != s1)  # check __ne__ operator.
    self.assertFalse(s1 != s2)  # check __ne__ operator.
    for c1, c2 in zip(nest.flatten(s1), nest.flatten(s2)):
      self.assertEqual(hash(c1), hash(c1))
      self.assertEqual(hash(c1), hash(c2))
    for value_fn in not_equal_value_fns:
      s3 = structure.type_spec_from_value(value_fn())
      self.assertNotEqual(s1, s3)  # check __ne__ operator.
      self.assertNotEqual(s2, s3)  # check __ne__ operator.
      self.assertFalse(s1 == s3)  # check __eq_ operator.
      self.assertFalse(s2 == s3)  # check __eq_ operator.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_ragged_structure_inequality_combinations()))
  def testRaggedStructureInequality(self, spec1, spec2):
    # pylint: disable=g-generic-assert
    self.assertNotEqual(spec1, spec2)  # check __ne__ operator.
    self.assertFalse(spec1 == spec2)  # check __eq__ operator.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_hash_combinations()))
  def testHash(self, value1_fn, value2_fn, value3_fn):
    s1 = structure.type_spec_from_value(value1_fn())
    s2 = structure.type_spec_from_value(value2_fn())
    s3 = structure.type_spec_from_value(value3_fn())
    for c1, c2, c3 in zip(nest.flatten(s1), nest.flatten(s2), nest.flatten(s3)):
      self.assertEqual(hash(c1), hash(c1))
      self.assertEqual(hash(c1), hash(c2))
      self.assertNotEqual(hash(c1), hash(c3))
      self.assertNotEqual(hash(c2), hash(c3))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_round_trip_conversion_combinations()))
  def testRoundTripConversion(self, value_fn):
    value = value_fn()
    s = structure.type_spec_from_value(value)

    def maybe_stack_ta(v):
      if isinstance(v, tensor_array_ops.TensorArray):
        return v.stack()
      return v

    before = self.evaluate(maybe_stack_ta(value))
    after = self.evaluate(
        maybe_stack_ta(
            structure.from_tensor_list(s, structure.to_tensor_list(s, value))))

    flat_before = nest.flatten(before)
    flat_after = nest.flatten(after)
    for b, a in zip(flat_before, flat_after):
      if isinstance(b, sparse_tensor.SparseTensorValue):
        self.assertAllEqual(b.indices, a.indices)
        self.assertAllEqual(b.values, a.values)
        self.assertAllEqual(b.dense_shape, a.dense_shape)
      elif isinstance(
          b,
          (ragged_tensor.RaggedTensor, ragged_tensor_value.RaggedTensorValue)):
        self.assertAllEqual(b, a)
      else:
        self.assertAllEqual(b, a)

  # pylint: enable=g-long-lambda

  def preserveStaticShape(self):
    rt = ragged_factory_ops.constant([[1, 2], [], [3]])
    rt_s = structure.type_spec_from_value(rt)
    rt_after = structure.from_tensor_list(rt_s,
                                          structure.to_tensor_list(rt_s, rt))
    self.assertEqual(rt_after.row_splits.shape.as_list(),
                     rt.row_splits.shape.as_list())
    self.assertEqual(rt_after.values.shape.as_list(), [None])

    st = sparse_tensor.SparseTensor(
        indices=[[3, 4]], values=[-1], dense_shape=[4, 5])
    st_s = structure.type_spec_from_value(st)
    st_after = structure.from_tensor_list(st_s,
                                          structure.to_tensor_list(st_s, st))
    self.assertEqual(st_after.indices.shape.as_list(), [None, 2])
    self.assertEqual(st_after.values.shape.as_list(), [None])
    self.assertEqual(st_after.dense_shape.shape.as_list(),
                     st.dense_shape.shape.as_list())

  @combinations.generate(test_base.default_test_combinations())
  def testPreserveTensorArrayShape(self):
    ta = tensor_array_ops.TensorArray(
        dtype=dtypes.int32, size=1, element_shape=(3,))
    ta_s = structure.type_spec_from_value(ta)
    ta_after = structure.from_tensor_list(ta_s,
                                          structure.to_tensor_list(ta_s, ta))
    self.assertEqual(ta_after.element_shape.as_list(), [3])

  @combinations.generate(test_base.default_test_combinations())
  def testPreserveInferredTensorArrayShape(self):
    ta = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=1)
    # Shape is inferred from the write.
    ta = ta.write(0, [1, 2, 3])
    ta_s = structure.type_spec_from_value(ta)
    ta_after = structure.from_tensor_list(ta_s,
                                          structure.to_tensor_list(ta_s, ta))
    self.assertEqual(ta_after.element_shape.as_list(), [3])

  @combinations.generate(test_base.default_test_combinations())
  def testIncompatibleStructure(self):
    # Define three mutually incompatible values/structures, and assert that:
    # 1. Using one structure to flatten a value with an incompatible structure
    #    fails.
    # 2. Using one structure to restructure a flattened value with an
    #    incompatible structure fails.
    value_tensor = constant_op.constant(42.0)
    s_tensor = structure.type_spec_from_value(value_tensor)
    flat_tensor = structure.to_tensor_list(s_tensor, value_tensor)

    value_sparse_tensor = sparse_tensor.SparseTensor(
        indices=[[0, 0]], values=[1], dense_shape=[1, 1])
    s_sparse_tensor = structure.type_spec_from_value(value_sparse_tensor)
    flat_sparse_tensor = structure.to_tensor_list(s_sparse_tensor,
                                                  value_sparse_tensor)

    value_nest = {
        "a": constant_op.constant(37.0),
        "b": constant_op.constant([1, 2, 3])
    }
    s_nest = structure.type_spec_from_value(value_nest)
    flat_nest = structure.to_tensor_list(s_nest, value_nest)

    with self.assertRaisesRegex(
        ValueError, r"SparseTensor.* is not convertible to a tensor with "
        r"dtype.*float32.* and shape \(\)"):
      structure.to_tensor_list(s_tensor, value_sparse_tensor)
    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure."):
      structure.to_tensor_list(s_tensor, value_nest)

    with self.assertRaisesRegex(TypeError,
                                "Neither a SparseTensor nor SparseTensorValue"):
      structure.to_tensor_list(s_sparse_tensor, value_tensor)

    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure."):
      structure.to_tensor_list(s_sparse_tensor, value_nest)

    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure."):
      structure.to_tensor_list(s_nest, value_tensor)

    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure."):
      structure.to_tensor_list(s_nest, value_sparse_tensor)

    with self.assertRaisesRegex(ValueError, r"Incompatible input:"):
      structure.from_tensor_list(s_tensor, flat_sparse_tensor)

    with self.assertRaisesRegex(ValueError, "Expected 1 tensors but got 2."):
      structure.from_tensor_list(s_tensor, flat_nest)

    with self.assertRaisesRegex(ValueError, "Incompatible input: "):
      structure.from_tensor_list(s_sparse_tensor, flat_tensor)

    with self.assertRaisesRegex(ValueError, "Expected 1 tensors but got 2."):
      structure.from_tensor_list(s_sparse_tensor, flat_nest)

    with self.assertRaisesRegex(ValueError, "Expected 2 tensors but got 1."):
      structure.from_tensor_list(s_nest, flat_tensor)

    with self.assertRaisesRegex(ValueError, "Expected 2 tensors but got 1."):
      structure.from_tensor_list(s_nest, flat_sparse_tensor)

  @combinations.generate(test_base.default_test_combinations())
  def testIncompatibleNestedStructure(self):
    # Define three mutually incompatible nested values/structures, and assert
    # that:
    # 1. Using one structure to flatten a value with an incompatible structure
    #    fails.
    # 2. Using one structure to restructure a flattened value with an
    #    incompatible structure fails.

    value_0 = {
        "a": constant_op.constant(37.0),
        "b": constant_op.constant([1, 2, 3])
    }
    s_0 = structure.type_spec_from_value(value_0)
    flat_s_0 = structure.to_tensor_list(s_0, value_0)

    # `value_1` has compatible nested structure with `value_0`, but different
    # classes.
    value_1 = {
        "a":
            constant_op.constant(37.0),
        "b":
            sparse_tensor.SparseTensor(
                indices=[[0, 0]], values=[1], dense_shape=[1, 1])
    }
    s_1 = structure.type_spec_from_value(value_1)
    flat_s_1 = structure.to_tensor_list(s_1, value_1)

    # `value_2` has incompatible nested structure with `value_0` and `value_1`.
    value_2 = {
        "a":
            constant_op.constant(37.0),
        "b": (sparse_tensor.SparseTensor(
            indices=[[0, 0]], values=[1], dense_shape=[1, 1]),
              sparse_tensor.SparseTensor(
                  indices=[[3, 4]], values=[-1], dense_shape=[4, 5]))
    }
    s_2 = structure.type_spec_from_value(value_2)
    flat_s_2 = structure.to_tensor_list(s_2, value_2)

    with self.assertRaisesRegex(
        ValueError, r"SparseTensor.* is not convertible to a tensor with "
        r"dtype.*int32.* and shape \(3,\)"):
      structure.to_tensor_list(s_0, value_1)

    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure."):
      structure.to_tensor_list(s_0, value_2)

    with self.assertRaisesRegex(TypeError,
                                "Neither a SparseTensor nor SparseTensorValue"):
      structure.to_tensor_list(s_1, value_0)

    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure."):
      structure.to_tensor_list(s_1, value_2)

    # NOTE(mrry): The repr of the dictionaries is not sorted, so the regexp
    # needs to account for "a" coming before or after "b". It might be worth
    # adding a deterministic repr for these error messages (among other
    # improvements).
    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure."):
      structure.to_tensor_list(s_2, value_0)

    with self.assertRaisesRegex(
        ValueError, "The two structures don't have the same nested structure."):
      structure.to_tensor_list(s_2, value_1)

    with self.assertRaisesRegex(ValueError, r"Incompatible input:"):
      structure.from_tensor_list(s_0, flat_s_1)

    with self.assertRaisesRegex(ValueError, "Expected 2 tensors but got 3."):
      structure.from_tensor_list(s_0, flat_s_2)

    with self.assertRaisesRegex(ValueError, "Incompatible input: "):
      structure.from_tensor_list(s_1, flat_s_0)

    with self.assertRaisesRegex(ValueError, "Expected 2 tensors but got 3."):
      structure.from_tensor_list(s_1, flat_s_2)

    with self.assertRaisesRegex(ValueError, "Expected 3 tensors but got 2."):
      structure.from_tensor_list(s_2, flat_s_0)

    with self.assertRaisesRegex(ValueError, "Expected 3 tensors but got 2."):
      structure.from_tensor_list(s_2, flat_s_1)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_convert_legacy_structure_combinations()))
  def testConvertLegacyStructure(self, output_types, output_shapes,
                                 output_classes, expected_structure):
    actual_structure = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)
    self.assertEqual(actual_structure, expected_structure)

  @combinations.generate(test_base.default_test_combinations())
  def testNestedNestedStructure(self):
    s = (tensor_spec.TensorSpec([], dtypes.int64),
         (tensor_spec.TensorSpec([], dtypes.float32),
          tensor_spec.TensorSpec([], dtypes.string)))

    int64_t = constant_op.constant(37, dtype=dtypes.int64)
    float32_t = constant_op.constant(42.0)
    string_t = constant_op.constant("Foo")

    nested_tensors = (int64_t, (float32_t, string_t))

    tensor_list = structure.to_tensor_list(s, nested_tensors)
    for expected, actual in zip([int64_t, float32_t, string_t], tensor_list):
      self.assertIs(expected, actual)

    (actual_int64_t,
     (actual_float32_t,
      actual_string_t)) = structure.from_tensor_list(s, tensor_list)
    self.assertIs(int64_t, actual_int64_t)
    self.assertIs(float32_t, actual_float32_t)
    self.assertIs(string_t, actual_string_t)

    (actual_int64_t, (actual_float32_t, actual_string_t)) = (
        structure.from_compatible_tensor_list(s, tensor_list))
    self.assertIs(int64_t, actual_int64_t)
    self.assertIs(float32_t, actual_float32_t)
    self.assertIs(string_t, actual_string_t)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_batch_combinations()))
  def testBatch(self, element_structure, batch_size,
                expected_batched_structure):
    batched_structure = nest.map_structure(
        lambda component_spec: component_spec._batch(batch_size),
        element_structure)
    self.assertEqual(batched_structure, expected_batched_structure)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_unbatch_combinations()))
  def testUnbatch(self, element_structure, expected_unbatched_structure):
    unbatched_structure = nest.map_structure(
        lambda component_spec: component_spec._unbatch(), element_structure)
    self.assertEqual(unbatched_structure, expected_unbatched_structure)

  # pylint: disable=g-long-lambda
  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         _test_to_batched_tensor_list_combinations()))
  def testToBatchedTensorList(self, value_fn, element_0_fn):
    batched_value = value_fn()
    s = structure.type_spec_from_value(batched_value)
    batched_tensor_list = structure.to_batched_tensor_list(s, batched_value)

    # The batch dimension is 2 for all of the test cases.
    # NOTE(mrry): `tf.shape()` does not currently work for the DT_VARIANT
    # tensors in which we store sparse tensors.
    for t in batched_tensor_list:
      if t.dtype != dtypes.variant:
        self.assertEqual(2, self.evaluate(array_ops.shape(t)[0]))

    # Test that the 0th element from the unbatched tensor is equal to the
    # expected value.
    expected_element_0 = self.evaluate(element_0_fn())
    unbatched_s = nest.map_structure(
        lambda component_spec: component_spec._unbatch(), s)
    actual_element_0 = structure.from_tensor_list(
        unbatched_s, [t[0] for t in batched_tensor_list])

    for expected, actual in zip(
        nest.flatten(expected_element_0), nest.flatten(actual_element_0)):
      self.assertValuesEqual(expected, actual)

  # pylint: enable=g-long-lambda

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSpecConstructor(self):
    rt_spec = ragged_tensor.RaggedTensorSpec([10, None], dtypes.int32)
    st_spec = sparse_tensor.SparseTensorSpec([10, 20], dtypes.float32)
    t_spec = tensor_spec.TensorSpec([10, 8], dtypes.string)
    element_spec = {"rt": rt_spec, "st": st_spec, "t": t_spec}
    ds_struct = dataset_ops.DatasetSpec(element_spec, [5])
    self.assertEqual(ds_struct._element_spec, element_spec)
    # Note: shape was automatically converted from a list to a TensorShape.
    self.assertEqual(ds_struct._dataset_shape, tensor_shape.TensorShape([5]))

  @combinations.generate(test_base.default_test_combinations())
  def testCustomMapping(self):
    elem = CustomMap(foo=constant_op.constant(37.))
    spec = structure.type_spec_from_value(elem)
    self.assertIsInstance(spec, CustomMap)
    self.assertEqual(spec["foo"], tensor_spec.TensorSpec([], dtypes.float32))

  @combinations.generate(test_base.default_test_combinations())
  def testObjectProxy(self):
    nt_type = collections.namedtuple("A", ["x", "y"])
    proxied = wrapt.ObjectProxy(nt_type(1, 2))
    proxied_spec = structure.type_spec_from_value(proxied)
    self.assertEqual(structure.type_spec_from_value(nt_type(1, 2)),
                     proxied_spec)


class CustomMap(collections_abc.Mapping):
  """Custom, immutable map."""

  def __init__(self, *args, **kwargs):
    self.__dict__.update(dict(*args, **kwargs))

  def __getitem__(self, x):
    return self.__dict__[x]

  def __iter__(self):
    return iter(self.__dict__)

  def __len__(self):
    return len(self.__dict__)


if __name__ == "__main__":
  test.main()
