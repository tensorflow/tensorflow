# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for StructuredTensor."""

import textwrap

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import row_partition
from tensorflow.python.ops.ragged.dynamic_ragged_shape import DynamicRaggedShape

# TODO(b/173144447): remove when structured_array_ops is included in init.
from tensorflow.python.ops.structured import structured_array_ops  # pylint: disable=unused-import

from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.ops.structured import structured_tensor_dynamic
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import dispatch


class _PrivateSpecialType(extension_type.ExtensionType):
  ragged: ragged_tensor.RaggedTensor


@dispatch.dispatch_for_types(array_ops.shape_v2, _PrivateSpecialType)
def shape_v2_special(input: _PrivateSpecialType, out_type=dtypes.int32,  # pylint: disable=redefined-builtin
                     name=None):
  """Returns a DynamicRaggedShape containing the shape of the input."""
  del name
  return array_ops.shape_v2(input.ragged, out_type)  # pylint: disable=protected-access


class _PrivateBrokenType(extension_type.ExtensionType):
  ragged: ragged_tensor.RaggedTensor


@dispatch.dispatch_for_types(array_ops.shape_v2, _PrivateBrokenType)
def shape_v2_broken(input: _PrivateBrokenType, out_type=dtypes.int32,  # pylint: disable=redefined-builtin
                    name=None):
  """Returns a DynamicRaggedShape containing the shape of the input."""
  del name
  del input
  del out_type
  return {
      "foo": "This is not a shape",
      "bar": "But if I put a string here, it becomes a vector"
  }


# pylint: disable=g-long-lambda
@test_util.run_all_in_graph_and_eager_modes
class StructuredTensorTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  def assertAllEqual(self, a, b, msg=None):
    if not (isinstance(a, structured_tensor.StructuredTensor) or
            isinstance(b, structured_tensor.StructuredTensor)):
      return super(StructuredTensorTest, self).assertAllEqual(a, b, msg)
    if not isinstance(a, structured_tensor.StructuredTensor):
      a = structured_tensor.StructuredTensor.from_pyval(a)
      self._assertStructuredEqual(a, b, msg, False)
    elif not isinstance(b, structured_tensor.StructuredTensor):
      b = structured_tensor.StructuredTensor.from_pyval(b)
      self._assertStructuredEqual(a, b, msg, False)
    else:
      self._assertStructuredEqual(a, b, msg, True)

  def _assertStructuredEqual(self, a, b, msg, check_shape):
    if check_shape:
      self.assertEqual(repr(a.shape), repr(b.shape))
    self.assertEqual(set(a.field_names()), set(b.field_names()))
    for field in a.field_names():
      a_value = a.field_value(field)
      b_value = b.field_value(field)
      self.assertIs(type(a_value), type(b_value))
      if isinstance(a_value, structured_tensor.StructuredTensor):
        self._assertStructuredEqual(a_value, b_value, msg, check_shape)
      else:
        self.assertAllEqual(a_value, b_value, msg)

  @parameterized.named_parameters([
      # Scalar (rank=0) StructuredTensors.
      {
          "testcase_name": "Rank0_WithTensorFields",
          "rank": 0,
          "fields": {"Foo": 5, "Bar": [1, 2, 3]},
          "expected_shape": []
      },
      {
          "testcase_name": "Rank0_WithRaggedFields",
          "fields": {
              # note: fields have varying rank & ragged_rank.
              "p": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "q": ragged_factory_ops.constant_value([[[4]], [], [[5, 6]]]),
              "r": ragged_factory_ops.constant_value([[[4]], [], [[5]]],
                                                     ragged_rank=1),
              "s": ragged_factory_ops.constant_value([[[4]], [], [[5]]],
                                                     ragged_rank=2),
          },
          "rank": 0,
          "expected_shape": [],
      },
      {
          "testcase_name": "Rank0_WithStructuredFields",
          "fields": lambda: {
              "foo": StructuredTensor.from_pyval({"a": 1, "b": [1, 2, 3]}),
              "bar": StructuredTensor.from_pyval(
                  [[{"x": 12}], [{"x": 13}, {"x": 14}]]),
              },
          "rank": 0,
          "expected_shape": [],
      },
      {
          "testcase_name": "Rank0_WithMixedFields",
          "fields": lambda: {
              # TODO(martinz): should handle this, but can't.
              "f1": 5,
              "f2": [1, 2, 3],
              "f3": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "f4": StructuredTensor.from_pyval({"a": 1, "b": [1, 2, 3]}),
          },
          "rank": 0,
          "expected_shape": [],
      },
      # Vector (rank=1) StructuredTensors.
      {
          "testcase_name": "Rank1_WithExplicitNrows",
          "fields": {"x": [1, 2], "y": [[1, 2], [3, 4]]},
          "rank": 1,
          "expected_shape": [2],
      },
      {
          "testcase_name": "Rank1_WithTensorFields",
          "fields": {"x": [1, 2], "y": [[1, 2], [3, 4]]},
          "rank": 1,
          "expected_shape": [2],

      },
      {
          "testcase_name": "Rank1_WithRaggedFields",
          "fields": {
              # note: fields have varying rank & ragged_rank.
              "p": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "q": ragged_factory_ops.constant_value([[[4]], [[5, 6], [7]]]),
              "r": ragged_factory_ops.constant_value([[], [[[12]], [[13]]]]),
              "s": ragged_factory_ops.constant_value([[], [[[12]], [[13]]]],
                                                     ragged_rank=1),
              "t": ragged_factory_ops.constant_value([[], [[[12]], [[13]]]],
                                                     ragged_rank=2),
          },
          "rank": 1,
          "expected_shape": [2],
      },
      {
          "testcase_name": "Rank1_WithStructuredFields",
          "fields": lambda: {
              "foo": StructuredTensor.from_pyval(
                  [{"a": 1, "b": [1, 2, 3]}, {"a": 2, "b": []}]),
              "bar": StructuredTensor.from_pyval(
                  [[{"x": 12}], [{"x": 13}, {"x": 14}]]),
          },
          "rank": 1,
          "expected_shape": [2],
      },
      {
          "testcase_name": "Rank1_WithMixedFields",
          "fields": lambda: {
              "x": [1, 2],
              "y": [[1, 2], [3, 4]],
              "r": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "s": StructuredTensor.from_pyval(
                  [[{"x": 12}], [{"x": 13}, {"x": 14}]]),
          },
          "rank": 1,
          "expected_shape": [2],
      },
      {
          "testcase_name": "Rank1_WithNoElements",
          "fields": lambda: {
              "x": [],
              "y": np.zeros([0, 8]),
              "r": ragged_factory_ops.constant([], ragged_rank=1),
              "s": StructuredTensor.from_pyval([]),
          },
          "rank": 1,
          "expected_shape": [0],  # Note: could also be [None] (?)
      },
      {
          "testcase_name": "Rank1_InferDimSize",
          "fields": lambda: {
              "x": [1, 2],
              "y": [[1, 2], [3, 4]],
              "r": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "p": ragged_factory_ops.constant_value([[4], [5, 6, 7]]),
              "foo": StructuredTensor.from_pyval(
                  [{"a": 1, "b": [1, 2, 3]}, {"a": 2, "b": []}]),
              "bar": StructuredTensor.from_pyval(
                  [[{"x": 12}], [{"x": 13}, {"x": 14}]]),
          },
          "rank": 1,
          "expected_shape": [2],  # inferred from field values.
      },
      # Matrix (rank=2) StructuredTensors.
      {
          "testcase_name": "Rank2_WithTensorFields",
          "fields": {
              "x": [[1, 2, 3], [4, 5, 6]],
              "y": np.ones([2, 3, 8])
          },
          "rank": 2,
          "expected_shape": [2, 3],  # inferred from field values.
      },
      {
          "testcase_name": "Rank2_WithRaggedFields",
          "fields": {
              # Note: fields must have identical row_splits.
              "a": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "b": ragged_factory_ops.constant_value([[4, 5], [6]]),
              "c": ragged_factory_ops.constant_value([[[1, 2], [3]], [[4, 5]]]),
              "d": ragged_factory_ops.constant_value(
                  [[[[1, 2], [3]], [[4], [], [5]]], [[[6, 7, 8], []]]]),
          },
          "rank": 2,
          "expected_shape": [2, None],
      },
      {
          "testcase_name": "Rank2_WithStructuredFields",
          "fields": lambda: {
              # Note: fields must have identical row_splits.
              "a": StructuredTensor.from_pyval(
                  [[{"x": 1}], [{"x": 2}, {"x": 3}]]),
              "b": StructuredTensor.from_pyval(
                  [[[{"y": 1}]], [[], [{"y": 2}, {"y": 3}]]]),
          },
          "rank": 2,
          "expected_shape": [2, None],  # ragged shape = [[*], [*, *]]
      },
      {
          "testcase_name": "Rank2_WithMixedFields",
          "fields": lambda: {
              "a": [[1, 2], [3, 4]],
              "b": ragged_factory_ops.constant_value([[1, 2], [3, 4]]),
              "c": StructuredTensor.from_pyval(
                  [[[{"y": 1}], []], [[], [{"y": 2}, {"y": 3}]]]),
              "d": ragged_factory_ops.constant_value(
                  [[[1, 2], []], [[3], [4]]]),
          },
          "rank": 2,
          "expected_shape": [2, 2],
      },
      # Rank=4 StructuredTensors.
      {
          "testcase_name": "Rank4_WithMixedFields",
          "fields": lambda: {
              "a": np.ones([1, 2, 3, 1]),
              "b": np.ones([1, 2, 3, 1, 5]),
              "c": ragged_factory_ops.constant(np.zeros([1, 2, 3, 1])),
              "d": ragged_factory_ops.constant(
                  np.zeros([1, 2, 3, 1, 3]).tolist(), ragged_rank=1),
              "e": ragged_factory_ops.constant(
                  np.zeros([1, 2, 3, 1, 2, 2]).tolist(), ragged_rank=2),
              "f": ragged_factory_ops.constant(np.zeros([1, 2, 3, 1, 3])),
              "g": StructuredTensor.from_pyval(
                  [[[[{"x": j, "y": k}] for k in range(3)]
                    for j in range(2)]]),
              "h": StructuredTensor.from_pyval(
                  [[[[[{"x": j, "y": k, "z": z} for z in range(j)]]
                     for k in range(3)]
                    for j in range(2)]]),
          },
          "rank": 4,
          "expected_shape": [1, 2, 3, 1],  # inferred from field values.
      },
  ])  # pyformat: disable
  def testFromFieldsAndRank(self, fields, rank, expected_shape):
    if callable(fields):
      fields = fields()  # deferred construction: fields may include tensors.

    struct = StructuredTensor.from_fields_and_rank(fields, rank)
    self.assertEqual(struct.shape.as_list(), expected_shape)

  @parameterized.named_parameters([
      {
          "testcase_name": "NoFields",
          "rank": 1,
          "fields": {},
          "msg": "Must provide at least one field"
      },
      {
          "testcase_name": "IntegerRank",
          "rank": 0.5,
          "fields": {
              "foo": [1]
          },
          "msg": "rank must be an integer"
      },
      {
          "testcase_name": "NonNegativeRank",
          "rank": -1,
          "fields": {
              "bar": [1, 2, 3]
          },
          "msg": "rank must be nonnegative"
      },
  ])
  def testFromFieldsAndRankError(self, fields, rank, msg):
    if callable(fields):
      fields = fields()  # deferred construction: fields may include tensors.
    with self.assertRaisesRegex(ValueError, msg):
      StructuredTensor.from_fields_and_rank(fields, rank)

  @parameterized.named_parameters([
      # Scalar (rank=0) StructuredTensors.
      {
          "testcase_name": "Rank0_WithNoFields",
          "shape": [],
          "fields": {},
      },
      {
          "testcase_name": "Rank0_WithTensorFields",
          "shape": [],
          "fields": {"Foo": 5, "Bar": [1, 2, 3]},
      },
      {
          "testcase_name": "Rank0_WithRaggedFields",
          "shape": [],
          "fields": {
              # note: fields have varying rank & ragged_rank.
              "p": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "q": ragged_factory_ops.constant_value([[[4]], [], [[5, 6]]]),
              "r": ragged_factory_ops.constant_value([[[4]], [], [[5]]],
                                                     ragged_rank=1),
              "s": ragged_factory_ops.constant_value([[[4]], [], [[5]]],
                                                     ragged_rank=2),
          },
      },
      {
          "testcase_name": "Rank0_WithStructuredFields",
          "shape": [],
          "fields": lambda: {
              "foo": StructuredTensor.from_pyval({"a": 1, "b": [1, 2, 3]}),
              "bar": StructuredTensor.from_pyval(
                  [[{"x": 12}], [{"x": 13}, {"x": 14}]]),
              },
      },
      {
          "testcase_name": "Rank0_WithMixedFields",
          "shape": [],
          "fields": lambda: {
              "f1": 5,
              "f2": [1, 2, 3],
              "f3": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "f4": StructuredTensor.from_pyval({"a": 1, "b": [1, 2, 3]}),
          },
      },
      # Vector (rank=1) StructuredTensors.
      {
          "testcase_name": "Rank1_WithNoFields",
          "shape": [2],
          "fields": {},
      },
      {
          "testcase_name": "Rank1_WithExplicitNrows",
          "shape": [None],
          "nrows": 2,
          "fields": {"x": [1, 2], "y": [[1, 2], [3, 4]]},
          "expected_shape": [2],
      },
      {
          "testcase_name": "Rank1_WithTensorFields",
          "shape": [2],
          "fields": {"x": [1, 2], "y": [[1, 2], [3, 4]]},
      },
      {
          "testcase_name": "Rank1_WithRaggedFields",
          "shape": [2],
          "fields": {
              # note: fields have varying rank & ragged_rank.
              "p": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "q": ragged_factory_ops.constant_value([[[4]], [[5, 6], [7]]]),
              "r": ragged_factory_ops.constant_value([[], [[[12]], [[13]]]]),
              "s": ragged_factory_ops.constant_value([[], [[[12]], [[13]]]],
                                                     ragged_rank=1),
              "t": ragged_factory_ops.constant_value([[], [[[12]], [[13]]]],
                                                     ragged_rank=2),
          },
      },
      {
          "testcase_name": "Rank1_WithStructuredFields",
          "shape": [2],
          "fields": lambda: {
              "foo": StructuredTensor.from_pyval(
                  [{"a": 1, "b": [1, 2, 3]}, {"a": 2, "b": []}]),
              "bar": StructuredTensor.from_pyval(
                  [[{"x": 12}], [{"x": 13}, {"x": 14}]]),
          },
      },
      {
          "testcase_name": "Rank1_WithMixedFields",
          "shape": [2],
          "fields": lambda: {
              "x": [1, 2],
              "y": [[1, 2], [3, 4]],
              "r": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "s": StructuredTensor.from_pyval(
                  [[{"x": 12}], [{"x": 13}, {"x": 14}]]),
          },
      },
      {
          "testcase_name": "Rank1_WithNoElements",
          "shape": [0],
          "fields": lambda: {
              "x": [],
              "y": np.zeros([0, 8]),
              "r": ragged_factory_ops.constant([], ragged_rank=1),
              "s": StructuredTensor.from_pyval([]),
          },
      },
      {
          "testcase_name": "Rank1_InferDimSize",
          "shape": [None],
          "fields": lambda: {
              "x": [1, 2],
              "y": [[1, 2], [3, 4]],
              "r": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "p": ragged_factory_ops.constant_value([[4], [5, 6, 7]]),
              "foo": StructuredTensor.from_pyval(
                  [{"a": 1, "b": [1, 2, 3]}, {"a": 2, "b": []}]),
              "bar": StructuredTensor.from_pyval(
                  [[{"x": 12}], [{"x": 13}, {"x": 14}]]),
          },
          "expected_shape": [2],  # inferred from field values.
      },
      # Matrix (rank=2) StructuredTensors.
      {
          "testcase_name": "Rank2_WithNoFields",
          "shape": [2, 8],
          "fields": {},
      },
      {
          "testcase_name": "Rank2_WithNoFieldsAndExplicitRowPartitions",
          "shape": [2, None],
          "row_partitions":
              lambda: [row_partition.RowPartition.from_row_lengths([3, 7])],
          "fields": {},
      },
      {
          "testcase_name": "Rank2_WithTensorFields",
          "shape": [None, None],
          "fields": {
              "x": [[1, 2, 3], [4, 5, 6]],
              "y": np.ones([2, 3, 8])
          },
          "expected_shape": [2, 3],  # inferred from field values.
      },
      {
          "testcase_name": "Rank2_WithRaggedFields",
          "shape": [2, None],  # ragged shape = [[*, *], [*]]
          "fields": {
              # Note: fields must have identical row_splits.
              "a": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "b": ragged_factory_ops.constant_value([[4, 5], [6]]),
              "c": ragged_factory_ops.constant_value([[[1, 2], [3]], [[4, 5]]]),
              "d": ragged_factory_ops.constant_value(
                  [[[[1, 2], [3]], [[4], [], [5]]], [[[6, 7, 8], []]]]),
          },
      },
      {
          "testcase_name": "Rank2_WithStructuredFields",
          "shape": [2, None],  # ragged shape = [[*], [*, *]]
          "fields": lambda: {
              # Note: fields must have identical row_splits.
              "a": StructuredTensor.from_pyval(
                  [[{"x": 1}], [{"x": 2}, {"x": 3}]]),
              "b": StructuredTensor.from_pyval(
                  [[[{"y": 1}]], [[], [{"y": 2}, {"y": 3}]]]),
          },
      },
      {
          "testcase_name": "Rank2_WithMixedFields",
          "shape": [2, None],
          "fields": lambda: {
              "a": [[1, 2], [3, 4]],
              "b": ragged_factory_ops.constant_value([[1, 2], [3, 4]]),
              "c": StructuredTensor.from_pyval(
                  [[[{"y": 1}], []], [[], [{"y": 2}, {"y": 3}]]]),
              "d": ragged_factory_ops.constant_value(
                  [[[1, 2], []], [[3], [4]]]),
          },
          "expected_shape": [2, 2],
      },
      # Rank=4 StructuredTensors.
      {
          "testcase_name": "Rank4_WithNoFields",
          "shape": [1, None, None, 3],
          "fields": {},
          "row_partitions": lambda: [
              row_partition.RowPartition.from_row_lengths([3]),
              row_partition.RowPartition.from_row_lengths([2, 0, 1]),
              row_partition.RowPartition.from_uniform_row_length(3, nvals=9)
          ]
      },
      {
          "testcase_name": "Rank4_WithMixedFields",
          "shape": [1, None, None, 1],
          "fields": lambda: {
              "a": np.ones([1, 2, 3, 1]),
              "b": np.ones([1, 2, 3, 1, 5]),
              "c": ragged_factory_ops.constant(np.zeros([1, 2, 3, 1])),
              "d": ragged_factory_ops.constant(
                  np.zeros([1, 2, 3, 1, 3]).tolist(), ragged_rank=1),
              "e": ragged_factory_ops.constant(
                  np.zeros([1, 2, 3, 1, 2, 2]).tolist(), ragged_rank=2),
              "f": ragged_factory_ops.constant(np.zeros([1, 2, 3, 1, 3])),
              "g": StructuredTensor.from_pyval(
                  [[[[{"x": j, "y": k}] for k in range(3)]
                    for j in range(2)]]),
              "h": StructuredTensor.from_pyval(
                  [[[[[{"x": j, "y": k, "z": z} for z in range(j)]]
                     for k in range(3)]
                    for j in range(2)]]),
          },
          "expected_shape": [1, 2, 3, 1],  # inferred from field values.
      },
  ])  # pyformat: disable
  def testFromFields(self,
                     shape,
                     fields,
                     expected_shape=None,
                     nrows=None,
                     row_partitions=None):
    if callable(fields):
      fields = fields()  # deferred construction: fields may include tensors.
    if callable(nrows):
      nrows = nrows()  # deferred construction.
    if callable(row_partitions):
      row_partitions = row_partitions()  # deferred construction.
    for validate in (True, False):
      struct = StructuredTensor.from_fields(
          fields,
          shape,
          nrows=nrows,
          row_partitions=row_partitions,
          validate=validate)
      if expected_shape is None:
        expected_shape = shape
      self.assertEqual(struct.shape.as_list(), expected_shape)
      self.assertLen(expected_shape, struct.rank)
      self.assertCountEqual(struct.field_names(), tuple(fields.keys()))
      for field, value in fields.items():
        self.assertIsInstance(
            struct.field_value(field),
            (ops.Tensor, structured_tensor.StructuredTensor,
             ragged_tensor.RaggedTensor))
        self.assertAllEqual(struct.field_value(field), value)

  @parameterized.parameters([
      dict(fields={}, shape=object(), err=TypeError),
      dict(
          fields=object(),
          shape=[],
          err=TypeError,
          msg="fields must be a dictionary"),
      dict(
          fields={1: 2}, shape=[], err=TypeError,
          msg="Unexpected type for key"),
      dict(
          fields={"x": object()},
          shape=[],
          err=(TypeError, ValueError),
          msg="Error with shape of x|Unexpected type for value"),
      dict(
          fields={},
          shape=None,
          err=ValueError,
          msg="StructuredTensor's shape must have known rank"),
      dict(
          fields={"f": 5},
          shape=[5],
          err=ValueError,
          msg=r"Field f has shape \(\), which is incompatible with the shape "
          r"that was specified or inferred from other fields: \(5,\)|Shapes"),
      dict(
          fields=dict(x=[1], y=[]),
          shape=[None],
          err=ValueError,
          msg=r"Error in shape of y"),
      dict(
          fields={"": 5},
          shape=[],
          err=ValueError,
          msg="Field name '' is not currently allowed."),
      dict(
          fields={"_": 5},
          shape=[],
          err=ValueError,
          msg="Field name '_' is not currently allowed."),
      dict(
          fields={
              "r1": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "r2": ragged_factory_ops.constant_value([[1, 2, 3], [4]])
          },
          shape=[2, None],
          validate=True,
          err=ValueError,
          msg=r"Error in shape of r2",
      ),
      dict(
          fields={},
          shape=(),
          nrows=5,
          err=ValueError,
          msg="nrows must be None if shape.rank==0"),
      dict(
          fields={},
          shape=(),
          row_partitions=[0],
          err=ValueError,
          msg=r"row_partitions must be None or \[\] if shape.rank<2"),
      dict(
          fields={},
          shape=(None, None, None),
          row_partitions=[],
          err=ValueError,
          msg=r"len\(row_partitions\) must be shape.rank-1"),
      dict(
          fields={},
          shape=[None],
          err=ValueError,
          msg="Must specify `nrows`, a fully specified `shape`, "
          "or have `fields` if `rank=1`"),
      dict(
          fields={},
          shape=[None, None],
          err=ValueError,
          msg="Must specify row_partitions, a fully specified shape, "
          "or have fields if rank > 1"),
      dict(
          fields={},
          shape=[None, None],
          nrows=lambda: constant_op.constant(2, dtypes.int32),
          row_partitions=lambda:
          [row_partition.RowPartition.from_row_lengths([3, 4])],
          err=ValueError,
          msg="row_partition dtypes are inconsistent"),
      dict(
          fields=lambda: {
              "a":
                  ragged_factory_ops.constant([[1]],
                                              row_splits_dtype=dtypes.int32),
              "b":
                  ragged_factory_ops.constant([[1]],
                                              row_splits_dtype=dtypes.int64)
          },
          shape=[None, None],
          err=ValueError,
          msg="field values have incompatible row_partition dtypes"),
  ])
  def testFromFieldsErrors(self,
                           fields,
                           shape,
                           nrows=None,
                           row_partitions=None,
                           validate=False,
                           err=ValueError,
                           msg=None,
                           test_in_eager=True):
    if not test_in_eager and context.executing_eagerly():
      return
    if callable(fields):
      fields = fields()  # deferred construction.
    if callable(nrows):
      nrows = nrows()  # deferred construction.
    if callable(row_partitions):
      row_partitions = row_partitions()  # deferred construction.
    with self.assertRaisesRegex(err, msg):
      struct = StructuredTensor.from_fields(
          fields=fields,
          shape=shape,
          nrows=nrows,
          row_partitions=row_partitions,
          validate=validate)
      for field_name in struct.field_names():
        self.evaluate(struct.field_value(field_name))
      self.evaluate(struct.nrows())

  def testMergeNrowsErrors(self):
    nrows = constant_op.constant(5)
    static_nrows = tensor_shape.Dimension(5)
    value = constant_op.constant([1, 2, 3])
    with self.assertRaisesRegex(ValueError, "fields have incompatible nrows"):
      structured_tensor._merge_nrows(
          nrows, static_nrows, value, dtypes.int32, validate=False)

  def testNestedStructConstruction(self):
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    struct1 = StructuredTensor.from_fields(shape=[], fields={"x": [1, 2]})
    struct2 = StructuredTensor.from_fields(shape=[2], fields={"x": [1, 2]})
    struct3 = StructuredTensor.from_fields(
        shape=[], fields={
            "r": rt,
            "s": struct1
        })
    struct4 = StructuredTensor.from_fields(
        shape=[2], fields={
            "r": rt,
            "s": struct2
        })

    self.assertEqual(struct3.shape.as_list(), [])
    self.assertEqual(struct3.rank, 0)
    self.assertEqual(set(struct3.field_names()), set(["r", "s"]))
    self.assertAllEqual(struct3.field_value("r"), rt)
    self.assertAllEqual(struct3.field_value("s"), struct1)

    self.assertEqual(struct4.shape.as_list(), [2])
    self.assertEqual(struct4.rank, 1)
    self.assertEqual(set(struct4.field_names()), set(["r", "s"]))
    self.assertAllEqual(struct4.field_value("r"), rt)
    self.assertAllEqual(struct4.field_value("s"), struct2)

  def testPartitionOuterDims(self):
    a = dict(x=1, y=[1, 2])
    b = dict(x=2, y=[3, 4])
    c = dict(x=3, y=[5, 6])
    d = dict(x=4, y=[7, 8])
    st1 = StructuredTensor.from_pyval([a, b, c, d])

    st2 = st1.partition_outer_dimension(
        row_partition.RowPartition.from_row_splits([0, 2, 2, 3, 4]))
    self.assertAllEqual(st2, [[a, b], [], [c], [d]])

    st3 = st2.partition_outer_dimension(
        row_partition.RowPartition.from_row_lengths([1, 0, 3, 0]))
    self.assertAllEqual(st3, [[[a, b]], [], [[], [c], [d]], []])

    # If we partition with uniform_row_lengths, then `x` is partitioned into
    # a Tensor (not a RaggedTensor).
    st4 = st1.partition_outer_dimension(
        row_partition.RowPartition.from_uniform_row_length(
            uniform_row_length=2, nvals=4, nrows=2))
    self.assertAllEqual(
        st4,
        structured_tensor.StructuredTensor.from_pyval(
            [[a, b], [c, d]],
            structured_tensor.StructuredTensor.Spec(
                _ragged_shape=DynamicRaggedShape.Spec(
                    row_partitions=[],
                    static_inner_shape=[2, 2],
                    dtype=dtypes.int64),
                _fields={
                    "x":
                        tensor_spec.TensorSpec([2, 2], dtypes.int32),
                    "y":
                        ragged_tensor.RaggedTensorSpec([2, 2, None],
                                                       dtypes.int32)
                })))

  def testPartitionOuterDimension3(self):
    rt = ragged_tensor.RaggedTensor.from_value_rowids(
        array_ops.constant([[1, 2], [3, 4], [5, 6]]), [0, 0, 1])
    struct = structured_tensor.StructuredTensor.from_fields({"r": rt}, [2])
    struct_2 = struct.partition_outer_dimension(
        row_partition.RowPartition.from_row_splits([0, 1, 2]))
    struct_3 = struct_2.partition_outer_dimension(
        row_partition.RowPartition.from_row_splits([0, 1, 2]))
    self.assertEqual(3, struct_3.rank)

  def testWithPrivateSpecialType(self):
    rt = ragged_tensor.RaggedTensor.from_value_rowids(
        array_ops.constant([[1, 2], [3, 4], [5, 6]]), [0, 0, 1])
    pst = _PrivateSpecialType(rt)
    pst_shape = array_ops.shape_v2(pst)
    st = structured_tensor.StructuredTensor.from_fields_and_rank({"r": pst}, 1)
    st_shape = st._ragged_shape
    self.assertEqual(1, st.rank)
    self.assertAllEqual(pst_shape[0], st_shape[0])

  def testWithPrivateBrokenType(self):
    rt = ragged_tensor.RaggedTensor.from_value_rowids(
        array_ops.constant([[1, 2], [3, 4], [5, 6]]), [0, 0, 1])
    pbt = _PrivateBrokenType(rt)

    with self.assertRaisesRegex(ValueError, "Error in shape of r"):
      structured_tensor.StructuredTensor.from_fields_and_rank({"r": pbt}, 1)

  def testPartitionOuterDimsErrors(self):
    st = StructuredTensor.from_fields({})
    partition = row_partition.RowPartition.from_row_splits([0])
    with self.assertRaisesRegex(ValueError,
                                r"Shape \(\) must have rank at least 1"):
      st.partition_outer_dimension(partition)

    with self.assertRaisesRegex(TypeError,
                                "row_partition must be a RowPartition"):
      st.partition_outer_dimension(10)

  @parameterized.named_parameters([
      {
          "testcase_name": "ScalarEmpty",
          "pyval": {},
          "expected": lambda: StructuredTensor.from_fields(shape=[], fields={})
      },
      {
          "testcase_name": "ScalarSimple",
          "pyval": {"a": 12, "b": [1, 2, 3], "c": [[1, 2], [3]]},
          "expected": lambda: StructuredTensor.from_fields(shape=[], fields={
              "a": 12,
              "b": [1, 2, 3],
              "c": ragged_factory_ops.constant([[1, 2], [3]])})
      },
      {
          "testcase_name": "ScalarSimpleWithTypeSpec",
          "pyval": {"a": 12, "b": [1, 2, 3], "c": [[1, 2], [3]]},
          "type_spec": StructuredTensor.Spec._from_fields_and_rank(
              fields={
                  "a": tensor_spec.TensorSpec([], dtypes.int32),
                  "b": tensor_spec.TensorSpec([None], dtypes.int32),
                  "c": ragged_tensor.RaggedTensorSpec([None, None],
                                                      dtypes.int32)},
              rank=0),
          "expected": lambda: StructuredTensor.from_fields(shape=[], fields={
              "a": 12,
              "b": [1, 2, 3],
              "c": ragged_factory_ops.constant([[1, 2], [3]])})
      },
      {
          "testcase_name": "ScalarWithNestedStruct",
          "pyval": {"a": 12, "b": [1, 2, 3], "c": {"x": b"Z", "y": [10, 20]}},
          "expected": lambda: StructuredTensor.from_fields(shape=[], fields={
              "a": 12,
              "b": [1, 2, 3],
              "c": StructuredTensor.from_fields(shape=[], fields={
                  "x": "Z",
                  "y": [10, 20]})})
      },
      {
          "testcase_name": "EmptyList",
          "pyval": [],
          "expected": lambda: [],
      },
      {
          "testcase_name": "ListOfEmptyList",
          "pyval": [[], []],
          "expected": lambda: [[], []],
      },
      {
          "testcase_name": "EmptyListWithTypeSpecAndFields",
          "pyval": [],
          "type_spec": structured_tensor.StructuredTensor.Spec._from_fields_and_rank(
              fields={"a": tensor_spec.TensorSpec([0], dtypes.int32)},
              rank=1),
          "expected": lambda: StructuredTensor.from_fields(shape=[0], fields={
              "a": []})
      },
      {
          "testcase_name": "EmptyListWithTypeSpecNoFieldsShape0_5",
          "pyval": [],
          "type_spec": StructuredTensor.Spec._from_shape(DynamicRaggedShape.Spec(
              row_partitions=[],
              static_inner_shape=[0, 5],
              dtype=dtypes.int64)),
          "expected": lambda: StructuredTensor.from_fields(shape=[0, 5],
                                                           fields={})
      },
      {
          "testcase_name": "EmptyListWithTypeSpecNoFieldsShape1_0",
          "pyval": [[]],
          "type_spec": StructuredTensor.Spec._from_shape(
              DynamicRaggedShape.Spec(
                  row_partitions=[],
                  static_inner_shape=[1, 0],
                  dtype=dtypes.int64)),
          "expected": lambda: StructuredTensor.from_shape(
              DynamicRaggedShape.from_lengths([1, 0]))
      },
      {
          "testcase_name": "VectorOfDict",
          "pyval": [{"a": 1}, {"a": 2}],
          "expected": lambda: StructuredTensor.from_fields(shape=[2], fields={
              "a": [1, 2]})
      },
      {
          "testcase_name": "VectorOfDictWithNestedStructScalar",
          "pyval": [{"a": 1, "b": {"x": [1, 2]}},
                    {"a": 2, "b": {"x": [3]}}],
          "expected": lambda: StructuredTensor.from_fields(shape=[2], fields={
              "a": [1, 2],
              "b": StructuredTensor.from_fields(shape=[2], fields={
                  "x": ragged_factory_ops.constant([[1, 2], [3]])})}),
      },
      {
          "testcase_name": "VectorOfDictWithNestedStructVector",
          "pyval": [{"a": 1, "b": [{"x": [1, 2]}, {"x": [5]}]},
                    {"a": 2, "b": [{"x": [3]}]}],
          "expected": lambda: StructuredTensor.from_fields(shape=[2], fields={
              "a": [1, 2],
              "b": StructuredTensor.from_fields(shape=[2, None], fields={
                  "x": ragged_factory_ops.constant([[[1, 2], [5]], [[3]]])})}),
      },
      {
          "testcase_name": "Ragged2DOfDict",
          "pyval": [[{"a": 1}, {"a": 2}, {"a": 3},],
                    [{"a": 4}, {"a": 5}]],
          "expected": lambda: StructuredTensor.from_fields(
              shape=[2, None],
              fields={
                  "a": ragged_factory_ops.constant([[1, 2, 3], [4, 5]])})
      },
      {
          # With no type-spec, all tensors>1D are encoded as ragged:
          "testcase_name": "MatrixOfDictWithoutTypeSpec",
          "pyval": [[{"a": 1}, {"a": 2}, {"a": 3},],
                    [{"a": 4}, {"a": 5}, {"a": 6}]],
          "expected": lambda: StructuredTensor.from_fields(
              shape=[2, None], fields={
                  "a": ragged_factory_ops.constant([[1, 2, 3], [4, 5, 6]])})
      },
      {
          # TypeSpec can be used to specify StructuredTensor shape.
          "testcase_name": "MatrixOfDictWithTypeSpec",
          "pyval": [[{"a": 1}, {"a": 2}, {"a": 3},],
                    [{"a": 4}, {"a": 5}, {"a": 6}]],
          "type_spec": structured_tensor.StructuredTensorSpec([2, 3], {
              "a": tensor_spec.TensorSpec(None, dtypes.int32)}),
          "expected": lambda: StructuredTensor.from_fields(
              shape=[2, 3], fields={"a": [[1, 2, 3], [4, 5, 6]]})
      },
  ])  # pyformat: disable
  def testPyvalConversion(self, pyval, expected, type_spec=None):
    expected = expected()  # Deferred init because it creates tensors.
    actual = structured_tensor.StructuredTensor.from_pyval(pyval, type_spec)
    self.assertAllEqual(actual, expected)
    if isinstance(actual, structured_tensor.StructuredTensor):
      if context.executing_eagerly():  # to_pyval only available in eager.
        self.assertEqual(actual.to_pyval(), pyval)

  def testStructuredTensorSpecFactory(self):
    spec = StructuredTensor.Spec._from_fields_and_rank(
        fields={
            "a": tensor_spec.TensorSpec([], dtypes.int32),
            "b": tensor_spec.TensorSpec([None], dtypes.int32),
            "c": ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)},
        rank=0)
    self.assertEqual(spec.rank, 0)

  @parameterized.named_parameters([
      dict(
          testcase_name="NoFieldsRaggedRank0",
          st=lambda: StructuredTensor.from_fields({}, (3,)),
          expected=[{}, {}, {}]),
      dict(
          testcase_name="NoFieldsRaggedRank1",
          st=lambda: StructuredTensor.from_fields(
              {}, (2, None),
              row_partitions=[
                  row_partition.RowPartition.from_row_lengths([3, 2])]),
          expected=[[{}, {}, {}], [{}, {}]]),
      dict(
          testcase_name="NoFieldsRaggedRank2",
          st=lambda: StructuredTensor.from_fields(
              {}, (2, None, None),
              row_partitions=[
                  row_partition.RowPartition.from_row_lengths([2, 1]),
                  row_partition.RowPartition.from_row_lengths([2, 3, 1])]),
          expected=[[[{}, {}], [{}, {}, {}]], [[{}]]]),
      dict(
          testcase_name="NoFieldsRaggedRank2NoDicts",
          st=lambda: StructuredTensor.from_fields(
              {}, (1, None, None),
              row_partitions=[
                  row_partition.RowPartition.from_row_lengths([2]),
                  row_partition.RowPartition.from_row_lengths([0, 0])]),
          expected=[[[], []]]),
      dict(
          testcase_name="NestedStructTensorWithNoFields",
          st=lambda: StructuredTensor.from_fields(
              {
                  "foo": ragged_factory_ops.constant([[[], []]]),
                  "bar": StructuredTensor.from_fields(
                      {}, (1, None, None, None), row_partitions=[
                          row_partition.RowPartition.from_row_lengths([2]),
                          row_partition.RowPartition.from_row_lengths([0, 0]),
                          row_partition.RowPartition.from_row_lengths([]),
                      ])

              }, (1, None, None),),
          expected=[[[], []]]),
  ])  # pyformat: disable
  def testToPyval(self, st, expected):
    if context.executing_eagerly():  # to_pyval only available in eager.
      st = st()  # Deferred init because it creates tensors.
      self.assertEqual(st.to_pyval(), expected)

  @parameterized.named_parameters([
      dict(testcase_name="MissingKeys",
           pyval=[{"a": [1, 2]}, {"b": [3, 4]}],
           err=KeyError,
           msg="'b'"),
      dict(testcase_name="TypeSpecMismatch_DictKey",
           pyval={"a": 1},
           type_spec=StructuredTensor.Spec._from_fields_and_rank(
               fields={"b": tensor_spec.TensorSpec([1], dtypes.int32)},
               rank=1),
           msg=r"Value at \(\) does not match typespec"),
      dict(testcase_name="TypeSpecMismatch_ListDictKey",
           pyval=[{"a": 1}],
           type_spec=StructuredTensor.Spec._from_fields_and_rank(
               fields={"b": tensor_spec.TensorSpec([1], dtypes.int32)},
               rank=1),
           msg=r"Value at \(\) does not match typespec"),
      dict(testcase_name="TypeSpecMismatch_RankMismatch",
           pyval=[{"a": 1}],
           type_spec=StructuredTensor.Spec._from_fields_and_rank(
               fields={"a": tensor_spec.TensorSpec([], dtypes.int32)},
               rank=0),
           msg=r"Value at \(\) does not match typespec \(rank mismatch\)"),
      dict(testcase_name="TypeSpecMismatch_Scalar",
           pyval=0,
           type_spec=StructuredTensor.Spec._from_shape(
               DynamicRaggedShape.Spec(
                   row_partitions=[],
                   static_inner_shape=[],
                   dtype=dtypes.int64)),
           msg=r"Value at \(\) does not match typespec"),
      dict(testcase_name="TypeSpecMismatch_ListTensor",
           pyval={"a": [[1]]},
           type_spec=StructuredTensor.Spec._from_fields_and_rank(
               fields={"a": tensor_spec.TensorSpec([], dtypes.int32)},
               rank=0),
           msg=r"Value at \('a',\) does not match typespec"),
      dict(testcase_name="TypeSpecMismatch_ListTensorDeep",
           pyval={"a": {"b": [[1]]}},
           type_spec=StructuredTensor.Spec._from_fields_and_rank(
               fields={"a": StructuredTensor.Spec._from_fields_and_rank(
                   fields={"b": tensor_spec.TensorSpec([], dtypes.int32)},
                   rank=0
               )},
               rank=0),
           msg=r"Value at \('a', 'b'\) does not match typespec"),
      dict(testcase_name="TypeSpecMismatch_ListTensorDeep_infer",
           pyval={"a": [{"b": [[1]]}, {"b": [["c"]]}]},
           type_spec=None,
           msg=r"Error parsing path \('a', 'b'\)"),
      dict(testcase_name="TypeSpecMismatch_ListTensorDeep_infer2",
           pyval=[{"a": 1}, {"a": "c"}],
           type_spec=None,
           msg=r"Error parsing path \('a',\)"),
      dict(testcase_name="TypeSpecMismatch_ListSparse",
           pyval=[1, 2],
           type_spec=sparse_tensor.SparseTensorSpec([None], dtypes.int32),
           msg=r"Value at \(\) does not match typespec"),
      dict(testcase_name="TypeSpecMismatch_ListStruct",
           pyval=[[1]],
           type_spec=StructuredTensor.Spec._from_fields_and_rank(
               fields={"a": tensor_spec.TensorSpec([1, 1], dtypes.int32)},
               rank=2),
           msg=r"Value at \(\) does not match typespec"),
      dict(testcase_name="InconsistentDictionaryDepth",
           pyval=[{}, [{}]],
           msg="Inconsistent depth of dictionaries"),
      dict(testcase_name="FOO",
           pyval=[[{}], 5],
           msg="Expected dict or nested list/tuple of dict"),

  ])  # pyformat: disable
  def testFromPyvalError(self, pyval, err=ValueError, type_spec=None, msg=None):
    with self.assertRaisesRegex(err, msg):
      structured_tensor.StructuredTensor.from_pyval(pyval, type_spec)

  def testToPyvalRequiresEagerMode(self):
    st = structured_tensor.StructuredTensor.from_pyval({"a": 5})
    if not context.executing_eagerly():
      with self.assertRaisesRegex(ValueError, "only supported in eager mode."):
        st.to_pyval()

  @parameterized.named_parameters([
      (
          "Rank0",
          [],
      ),
      (
          "Rank1",
          [5, 3],
      ),
      (
          "Rank2",
          [5, 8, 3],
      ),
      (
          "Rank5",
          [1, 2, 3, 4, 5],
      ),
  ])
  def testRowPartitionsFromUniformShape(self, shape):
    for rank in range(len(shape)):
      partitions = structured_tensor._row_partitions_for_uniform_shape(
          ops.convert_to_tensor(shape), rank)
      self.assertLen(partitions, max(0, rank - 1))
      if partitions:
        self.assertAllEqual(shape[0], partitions[0].nrows())
      for (dim, partition) in enumerate(partitions):
        self.assertAllEqual(shape[dim + 1], partition.uniform_row_length())

  @parameterized.named_parameters([
      # For shapes: U = uniform dimension; R = ragged dimension.
      dict(
          testcase_name="Shape_UR_Rank2",
          rt=[[1, 2], [], [3]],
          rt_ragged_rank=1,
          rank=2,
          expected_row_lengths=[[2, 0, 1]]),
      dict(
          testcase_name="Shape_URR_Rank2",
          rt=[[[1, 2], []], [[3]]],
          rt_ragged_rank=2,
          rank=2,
          expected_row_lengths=[[2, 1]]),
      dict(
          testcase_name="Shape_URU_Rank2",
          rt=[[[1], [2]], [[3]]],
          rt_ragged_rank=1,
          rank=2,
          expected_row_lengths=[[2, 1]]),
      dict(
          testcase_name="Shape_URR_Rank3",
          rt=[[[1, 2], []], [[3]]],
          rt_ragged_rank=2,
          rank=3,
          expected_row_lengths=[[2, 1], [2, 0, 1]]),
      dict(
          testcase_name="Shape_URU_Rank3",
          rt=[[[1], [2]], [[3]]],
          rt_ragged_rank=1,
          rank=3,
          expected_row_lengths=[[2, 1], [1, 1, 1]]),
      dict(
          testcase_name="Shape_URRUU_Rank2",
          rt=[[[[[1, 2]]]]],
          rt_ragged_rank=2,
          rank=2,
          expected_row_lengths=[[1]]),
      dict(
          testcase_name="Shape_URRUU_Rank3",
          rt=[[[[[1, 2]]]]],
          rt_ragged_rank=2,
          rank=3,
          expected_row_lengths=[[1], [1]]),
      dict(
          testcase_name="Shape_URRUU_Rank4",
          rt=[[[[[1, 2]]]]],
          rt_ragged_rank=2,
          rank=4,
          expected_row_lengths=[[1], [1], [1]]),
      dict(
          testcase_name="Shape_URRUU_Rank5",
          rt=[[[[[1, 2]]]]],
          rt_ragged_rank=2,
          rank=5,
          expected_row_lengths=[[1], [1], [1], [2]]),
  ])
  def testRowPartitionsForRaggedTensor(self, rt, rt_ragged_rank, rank,
                                       expected_row_lengths):
    rt = ragged_factory_ops.constant(rt, rt_ragged_rank)
    partitions = structured_tensor._row_partitions_for_ragged_tensor(
        rt, rank, dtypes.int64)
    self.assertLen(partitions, rank - 1)
    self.assertLen(partitions, len(expected_row_lengths))
    for partition, expected in zip(partitions, expected_row_lengths):
      self.assertAllEqual(partition.row_lengths(), expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="2D_0_1",
          st=[[{"x": 1}, {"x": 2}], [{"x": 3}]],
          outer_axis=0, inner_axis=1,
          expected=[{"x": 1}, {"x": 2}, {"x": 3}]),
      dict(
          testcase_name="3D_0_1",
          st=[[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]],
          outer_axis=0, inner_axis=1,
          expected=[[{"x": 1}, {"x": 2}], [{"x": 3}], [{"x": 4}]]),
      dict(
          testcase_name="3D_1_2",
          st=[[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]],
          outer_axis=1, inner_axis=2,
          expected=[[{"x": 1}, {"x": 2}, {"x": 3}], [{"x": 4}]]),
      dict(
          testcase_name="3D_0_2",
          st=[[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]],
          outer_axis=0, inner_axis=2,
          expected=[{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}]),
      dict(
          testcase_name="4D_0_1",
          st=[[[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]],
              [[[{"x": 5}]], [[{"x": 6}], [{"x": 7}]]]],
          outer_axis=0, inner_axis=1,
          expected=[[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]],
                    [[{"x": 5}]], [[{"x": 6}], [{"x": 7}]]]),
      dict(
          testcase_name="4D_0_2",
          st=[[[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]],
              [[[{"x": 5}]], [[{"x": 6}], [{"x": 7}]]]],
          outer_axis=0, inner_axis=2,
          expected=[[{"x": 1}, {"x": 2}], [{"x": 3}], [{"x": 4}],
                    [{"x": 5}], [{"x": 6}], [{"x": 7}]]),
      dict(
          testcase_name="4D_0_3",
          st=[[[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]],
              [[[{"x": 5}]], [[{"x": 6}], [{"x": 7}]]]],
          outer_axis=0, inner_axis=3,
          expected=[{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4},
                    {"x": 5}, {"x": 6}, {"x": 7}]),
      dict(
          testcase_name="4D_1_2",
          st=[[[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]],
              [[[{"x": 5}]], [[{"x": 6}], [{"x": 7}]]]],
          outer_axis=1, inner_axis=2,
          expected=[[[{"x": 1}, {"x": 2}], [{"x": 3}], [{"x": 4}]],
                    [[{"x": 5}], [{"x": 6}], [{"x": 7}]]]),
      dict(
          testcase_name="4D_1_3",
          st=[[[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]],
              [[[{"x": 5}]], [[{"x": 6}], [{"x": 7}]]]],
          outer_axis=1, inner_axis=3,
          expected=[[{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}],
                    [{"x": 5}, {"x": 6}, {"x": 7}]]),
      dict(
          testcase_name="4D_2_3",
          st=[[[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]],
              [[[{"x": 5}]], [[{"x": 6}], [{"x": 7}]]]],
          outer_axis=2, inner_axis=3,
          expected=[[[{"x": 1}, {"x": 2}, {"x": 3}], [{"x": 4}]],
                    [[{"x": 5}], [{"x": 6}, {"x": 7}]]]),
  ])  # pyformat: disable
  def testMergeDims(self, st, outer_axis, inner_axis, expected):
    st = StructuredTensor.from_pyval(st)
    result = st.merge_dims(outer_axis, inner_axis)
    self.assertAllEqual(result, expected)

  def testMergeDimsDetail_3D_0_1(self):
    st = StructuredTensor.from_pyval(
        [[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]])
    result = st.merge_dims(0, 1)
    expected_shape = tensor_shape.TensorShape([3, None])
    self.assertTrue(expected_shape.is_compatible_with(result.shape))

  def testMergeDims_0_1(self):
    rt = ragged_tensor.RaggedTensor.from_value_rowids(
        array_ops.constant([[1, 2], [3, 4], [5, 6]]), [0, 0, 1])
    struct = StructuredTensor.from_fields({"r": rt}, [2])
    struct_2 = struct.partition_outer_dimension(
        row_partition.RowPartition.from_row_splits([0, 1, 2]))
    struct_3 = struct_2.partition_outer_dimension(
        row_partition.RowPartition.from_row_splits([0, 1, 2]))
    self.assertLen(struct_3.row_partitions, 2)
    merged = struct_3.merge_dims(0, 1)
    self.assertLen(merged.row_partitions, 1)

  def testMergeDimsError(self):
    st = StructuredTensor.from_pyval([[[{"a": 5}]]])
    with self.assertRaisesRegex(
        ValueError, r"Expected outer_axis \(2\) to be less than "
        r"or equal to inner_axis \(1\)"):
      st.merge_dims(2, 1)

  def testTupleFieldValue(self):
    st = StructuredTensor.from_pyval({"a": 5, "b": {"c": [1, 2, 3]}})
    self.assertAllEqual(st.field_value(("a",)), 5)
    self.assertAllEqual(st.field_value(("b", "c")), [1, 2, 3])
    expected = r"Field path \(.*a.*,.*b.*\) not found in .*"
    with self.assertRaisesRegex(KeyError, expected):
      st.field_value(("a", "b"))

  @parameterized.named_parameters([
      dict(
          testcase_name="scalar_scalar_scalar",
          st={"b": {"a": 5}},
          source_path=("b", "a"),
          new_field_name="new_field",
          expected={"b": {"a": 5}, "new_field": 5},),
      dict(
          testcase_name="scalar_scalar_repeated",
          st={"b": {"a": [5, 3]}},
          source_path=("b", "a"),
          new_field_name="new_field",
          expected={"b": {"a": [5, 3]}, "new_field": [5, 3]}),
      dict(
          testcase_name="scalar_scalar_repeated2",
          st={"b": {"a": [[7], [5, 3]]}},
          source_path=("b", "a"),
          new_field_name="new_field",
          expected={"b": {"a": [[7], [5, 3]]}, "new_field": [[7], [5, 3]]}),
      dict(
          testcase_name="repeated_scalar_repeated",
          st=[{"b": {"a": [7]}},
              {"b": {"a": [5, 3]}}],
          source_path=("b", "a"),
          new_field_name="new_field",
          expected=[{"b": {"a": [7]}, "new_field": [7]},
                    {"b": {"a": [5, 3]}, "new_field": [5, 3]}]),
      dict(
          testcase_name="repeated_scalar_repeated2",
          st=[{"b": {"a": [[5, 7], []]}},
              {"b": {"a": [[5, 1], [3]]}}],
          source_path=("b", "a"),
          new_field_name="new_field",
          expected=[{"b": {"a": [[5, 7], []]},
                     "new_field": [[5, 7], []]},
                    {"b": {"a": [[5, 1], [3]]},
                     "new_field": [[5, 1], [3]]}]),
      dict(
          testcase_name="scalar_scalar_scalar_scalar",
          st={"a": {"b": {"c": 7}}},
          source_path=("a", "b", "c"),
          new_field_name="new_field",
          expected={"a": {"b": {"c": 7}, "new_field": 7}}),
      dict(
          testcase_name="repeated_scalar_scalar_scalar",
          st=[{"a": {"b": {"c": 7}}},
              {"a": {"b": {"c": 5}}}],
          source_path=("a", "b", "c"),
          new_field_name="new_field",
          expected=[{"a": {"b": {"c": 7}, "new_field": 7}},
                    {"a": {"b": {"c": 5}, "new_field": 5}}],),
      dict(
          testcase_name="repeated_repeated_scalar_scalar",
          st=[{"a": [{"b": {"c": 7}}, {"b": {"c": 3}}]},
              {"a": [{"b": {"c": 5}}]}],
          source_path=("a", "b", "c"),
          new_field_name="new_field",
          expected=[{"a": [{"b": {"c": 7}, "new_field": 7},
                           {"b": {"c": 3}, "new_field": 3}]},
                    {"a": [{"b": {"c": 5}, "new_field": 5}]}]),
      dict(
          testcase_name="docs_tokens",
          st=[{"docs": [{"tokens": [7, 17]}, {"tokens": [3, 13]}]},
              {"docs": [{"tokens": [5, 15]}]}],
          source_path=("docs", "tokens"),
          new_field_name="docs_tokens",
          expected=[{"docs": [{"tokens": [7, 17]}, {"tokens": [3, 13]}],
                     "docs_tokens": [7, 17, 3, 13]},
                    {"docs": [{"tokens": [5, 15]}],
                     "docs_tokens": [5, 15]}],
          ),
      dict(
          testcase_name="repeated_repeated_scalar_repeated",
          st=[{"a": [{"b": {"c": [7, 17]}}, {"b": {"c": [3, 13]}}]},
              {"a": [{"b": {"c": [5, 15]}}]}],
          source_path=("a", "b", "c"),
          new_field_name="new_field",
          expected=[{"a": [{"b": {"c": [7, 17]}, "new_field": [7, 17]},
                           {"b": {"c": [3, 13]}, "new_field": [3, 13]}]},
                    {"a": [{"b": {"c": [5, 15]}, "new_field": [5, 15]}]}]),
      dict(
          testcase_name="scalar_scalar_scalar_repeated",
          st={"a": {"b": {"c": [7, 3, 5]}}},
          source_path=("a", "b", "c"),
          new_field_name="new_field",
          expected={"a": {"b": {"c": [7, 3, 5]}, "new_field": [7, 3, 5]}}),
      dict(
          testcase_name="repeated_repeated_scalar_repeated2",
          st=[{"a": [{"b": {"c": [[7, 3], [17]]}}, {"b": {"c": [[3, 13]]}}]},
              {"a": [{"b": {"c": [[5, 15]]}}]}],
          source_path=("a", "b", "c"),
          new_field_name="new_field",
          expected=[{"a": [{"b": {"c": [[7, 3], [17]]},
                            "new_field": [[7, 3], [17]]},
                           {"b": {"c": [[3, 13]]},
                            "new_field": [[3, 13]]}]},
                    {"a": [{"b": {"c": [[5, 15]]},
                            "new_field": [[5, 15]]}]}]),
      dict(testcase_name="example_4_promote_of_labeled_vector",
           st=[{"user_info": [{"gaia_id": {"vec": [0, 1, 2]}}]},
               {"user_info": [{"gaia_id": {"vec": [3, 4, 5]}}]}],
           source_path=("user_info", "gaia_id"),
           new_field_name="user_info_gaia_id",
           expected=[{"user_info": [{"gaia_id": {"vec": [0, 1, 2]}}],
                      "user_info_gaia_id": [{"vec": [0, 1, 2]}]},
                     {"user_info": [{"gaia_id": {"vec": [3, 4, 5]}}],
                      "user_info_gaia_id": [{"vec": [3, 4, 5]}]}]),
      dict(
          testcase_name="promote_structure",
          st=[{"a": [{"aa": [{"b": {"c": 1}}, {"b": {"c": 8}}]}],},
              {"a": [{"aa": [{"b": {"c": 12}}]}],}],
          source_path=("a", "aa", "b"),
          new_field_name="new_field",
          expected=[{"a": [{"aa": [{"b": {"c": 1}}, {"b": {"c": 8}}],
                            "new_field": [{"c": 1}, {"c": 8}]}]},
                    {"a": [{"aa": [{"b": {"c": 12}}],
                            "new_field": [{"c": 12}]}]}])])  # pyformat: disable
  def testPromote(self, st, source_path, new_field_name, expected):
    st2 = StructuredTensor.from_pyval(st)
    expected2 = StructuredTensor.from_pyval(expected)
    result = st2.promote(source_path, new_field_name)
    self.assertAllEqual(result, expected2)

  def testPromoteDense(self):
    st = StructuredTensor.from_fields(
        {
            "a":
                StructuredTensor.from_fields(
                    {"b": [[[1, 11], [2, 12]], [[3, 13], [4, 14]]]},
                    shape=[2, 2, 2])
        },
        shape=[2])
    result = st.promote(("a", "b"), "new_field")
    self.assertEqual(st.rank, 1)
    self.assertEqual(st.field_value("a").rank, 3)
    self.assertAllEqual(
        result.field_value("new_field"), [[1, 11, 2, 12], [3, 13, 4, 14]])

  def testMergeDimsGeneric(self):
    """This is an example of a dense tensor being merged, when outer=rank.

    Note that outer=rank is equivalent to outer=rank - 1. And yet, from the
    perspective of promote, it is nice to be able to have this functionality
    directly available, because sometimes the rank of the parent equals the
    rank of the child.

    Finally, note that merge_dims for Ragged and StructuredTensor would not
    accept this as a valid argument.

    Note: _merge_dims_generic is private, but these unit tests help to
    discuss the proper API definition.
    """
    t = array_ops.constant([[[1, 11], [2, 12]], [[3, 13], [4, 14]]])
    t2 = structured_tensor._merge_dims_generic(t, 1, 3)
    self.assertAllEqual(t2, [[1, 11, 2, 12], [3, 13, 4, 14]])

  def testMergeDimsGenericNoop(self):
    """This is an example of a dense tensor being merged, when outer=inner.

    Sometimes, when promoting, the parent and grandparent ranks are equal.
    Finally, note that merge_dims for Ragged and StructuredTensor would not
    accept this as a valid argument. This should be aligned.
    """
    t = array_ops.constant([[[1, 11], [2, 12]], [[3, 13], [4, 14]]])
    t2 = structured_tensor._merge_dims_generic(t, 2, 2)
    self.assertAllEqual(t2, [[[1, 11], [2, 12]], [[3, 13], [4, 14]]])

  def testRepr(self):
    st = StructuredTensor.from_pyval({"a": 5, "b": {"c": [1, 2, 3]}})
    if context.executing_eagerly():
      expected = textwrap.dedent("""
          <StructuredTensor(
              fields={
                  "a": tf.Tensor(5, shape=(), dtype=int32),
                  "b": <StructuredTensor(
                          fields={
                              "c": tf.Tensor([1 2 3], shape=(3,), dtype=int32)},
                          shape=())>},
              shape=())>""")[1:]
    else:
      expected = textwrap.dedent("""
          <StructuredTensor(
              fields={
                  "a": Tensor("Const:0", shape=(), dtype=int32),
                  "b": <StructuredTensor(
                          fields={
                              "c": Tensor("RaggedConstant/Const:0", shape=(3,), dtype=int32)},
                          shape=())>},
              shape=())>""")[1:]
    self.assertEqual(repr(st), expected)

  def testPartitionOuterDimension2DDenseField(self):
    struct = structured_tensor.StructuredTensor.from_fields(
        fields={"r": array_ops.constant([[1, 2], [3, 4]])}, shape=[2])

    result = struct.partition_outer_dimension(
        row_partition.RowPartition.from_uniform_row_length(2, 2))
    r = result.field_value("r")
    self.assertAllEqual(r, [[[1, 2], [3, 4]]])

  @parameterized.parameters([
      # Simple example.
      (
          {"a": 12, "b": 23},
          {"a": 7},
      ),
      # New field.
      (
          {"a": 12},
          {("b",): 13},
      ),
      # Nested example.
      (
          {"a": 12, "b": {"c": 23}},
          {("b", "c"): 7},
      ),
      # Multipe updates.
      (
          {"a": 12, "b": {"c": 23}},
          {"a": 3, ("b", "c"): 7},
      ),
      # Deep updates.
      (
          {"a": 12, "b": {"c": 23, "d": {"e": 11}}},
          {("b", "c"): 7, ("b", "d", "e"): 13},
      ),
      # Multiple updates to the same substructure.
      (
          {"a": 12, "b": {"c": 23, "d": {"e": 11}}},
          {("b", "c"): 7, ("b", "f"): 13},
      ),
      # Scalar to non-scalar elements. Shape remains unchanged.
      (
          {"a": 5},
          {"a": ragged_factory_ops.constant_value([[51, 52], [61, 62, 63]])},
      ),
      # Non-scalar element to scalar.
      (
          {"c": {"a": [5, 3], "b": 2}},
          {("c", "a"): 5},
      ),
      # Rank-1 StructuredTensor: shape is preserved and an item is added.
      (
          [{"a": 5}, {"a": 6}],
          {"a": [15, 16], "b": np.array([0.9, 1.1])},
      ),
      # Non-scalar ragged elements, within a rank-2 StructuredTensor: elements
      # rows (inner dimensions) are changed, but StructuredTensor shape
      # (outer dimensions) are preserved.
      (
          [[{"a": [5]}], [{"a": [3, 4]}, {"a": [8]}]],
          {"a": ragged_factory_ops.constant_value([[[50, 60]], [[30], []]])},
      ),
  ])  # pyformat: disable
  def testWithUpdatesValues(self, pyval, updates):
    st = StructuredTensor.from_pyval(pyval)
    updated_st = st.with_updates(updates, validate=False)
    for key, value in updates.items():
      got = updated_st.field_value(key)
      self.assertAllEqual(
          value, got,
          "Update failed: key={}, value={}, got={}".format(key, value, got))

  def testWithUpdatesFunctions(self):
    pyval = {"a": 12, "b": {"c": 23, "d": {"e": 11}}}
    st = StructuredTensor.from_pyval(pyval)
    st_updated = st.with_updates(
        {
            "a": lambda x: x + 1,
            ("b", "d", "e"): lambda x: x + 7
        }, validate=True)
    # Updated values.
    self.assertAllEqual(st_updated.field_value("a"), 13)
    self.assertAllEqual(st_updated.field_value(("b", "d", "e")), 18)
    # Unchanged value.
    self.assertAllEqual(st_updated.field_value(("b", "c")), 23)

  def test_from_pyval_list_of_empty(self):
    """See b/183245576."""
    st = structured_tensor.StructuredTensor.from_pyval([{}])
    self.assertAllEqual([1], st.shape.as_list())

  def test_from_pyval_list_of_empty_three(self):
    """See b/183245576."""
    st = structured_tensor.StructuredTensor.from_pyval([{}, {}, {}])
    self.assertAllEqual([3], st.shape.as_list())
    self.assertEmpty(st.field_names())

  def test_from_pyval_deep_list_of_empty(self):
    """See b/183245576."""
    st = structured_tensor.StructuredTensor.from_pyval([[{
        "a": {},
        "b": [3, 4]
    }, {
        "a": {},
        "b": [5]
    }], [{
        "a": {},
        "b": [7, 8, 9]
    }]])
    self.assertAllEqual(2, st.rank)
    self.assertEqual(2, st.shape[0])
    self.assertEmpty(st.field_value("a").field_names())

  def testWithUpdatesChecks(self):
    pyval = {"a": 12, "b": {"c": 23, "d": {"e": 11}}}
    st = StructuredTensor.from_pyval(pyval)

    # Try to set non-existant sub-structure.
    with self.assertRaisesRegex(
        ValueError, r"cannot create new sub-field.*\('b', 'x'\).*is not set"):
      st.with_updates({("b", "x", "e"): 5})

    # Try to set with path to a non-sub-structure.
    with self.assertRaisesRegex(
        ValueError, r"cannot create new sub-field.*\('b', 'c'\).*is not a "
        r"`StructuredTensor`"):
      st.with_updates({("b", "c", "e"): 5})

    # Try to apply function to non-existing value.
    with self.assertRaisesRegex(
        ValueError, r"cannot update.*\('b', 'd', 'x'\).*does not already "
        r"exist"):
      st.with_updates({("b", "d", "x"): lambda x: x + 1})

    # Empty names not allowed.
    with self.assertRaisesRegex(ValueError, r"does not allow empty names"):
      st.with_updates({(): lambda x: x + 1})
    with self.assertRaisesRegex(ValueError, r"does not allow empty names"):
      st.with_updates({("b", ""): lambda x: x + 1})

    # Parent and child nodes cannot be updated simultaneously.
    with self.assertRaisesRegex(
        ValueError, r"does not allow both parent and child nodes.*"
        r"parent=\('b'.*child=\('b', 'd'"):
      st.with_updates({("b", "d"): lambda x: x + 1, "a": 3, "b": 10})

    # Invalid shape change.
    with self.assertRaisesRegex(
        ValueError,
        r"`StructuredTensor.with_updates` failed for field \('c',\)"):
      st_with_shape = StructuredTensor.from_pyval([[{
          "c": {
              "a": 5,
              "b": 2
          }
      }], [{
          "c": {
              "a": 3,
              "b": 1
          }
      }, {
          "c": {
              "a": 8,
              "b": 18
          }
      }]])
      st_with_shape.with_updates({("c", "a"): 3})

  def testWithUpdatesDelete(self):
    pyval = {"a": 12, "b": {"c": 23, "d": {"e": 11}}}
    st = StructuredTensor.from_pyval(pyval)
    updated_st = st.with_updates({("b", "c"): None}, validate=True)
    self.assertNotIn("c", updated_st.field_value("b").field_names())
    with self.assertRaisesRegex(ValueError,
                                r"cannot delete.*\('b', 'x'\).*not present"):
      st.with_updates({("b", "x"): None}, validate=True)
    with self.assertRaisesRegex(ValueError,
                                r"cannot delete.*\'x'.*not present"):
      st.with_updates({"x": None}, validate=False)

    # Test that nrows() and rowpartitions() is preserved after removal.
    pyval = [[{"a": 1}, {"a": 2}], [{"a": 3}]]
    st = StructuredTensor.from_pyval(pyval)
    self.assertLen(st.row_partitions, 1)
    self.assertAllEqual(st.nrows(), 2)
    self.assertAllEqual(st.row_partitions[0].row_lengths(), [2, 1])
    updated_st = st.with_updates({("a",): None}, validate=True)
    self.assertLen(updated_st.row_partitions, 1)
    self.assertAllEqual(updated_st.nrows(), 2)
    self.assertAllEqual(updated_st.row_partitions[0].row_lengths(), [2, 1])

    # Test that it works also for rank-1 and rank-0 empty results.
    pyval = [{"a": 1}, {"a": 2}]
    st = StructuredTensor.from_pyval(pyval)
    self.assertEqual(st.rank, 1)
    updated_st = st.with_updates({("a",): None}, validate=True)
    self.assertEqual(updated_st.rank, 1)

    # assertEqual won't work because nrows() returns a tensor, and
    # assertEqual doesn't do the magic to convert them to numbers in a
    # way that works in eager/non-eager mode.
    self.assertAllEqual(updated_st.nrows(), 2)
    pyval = {"a": [0, 1]}
    st = StructuredTensor.from_pyval(pyval)
    self.assertEqual(st.rank, 0)
    updated_st = st.with_updates({("a",): None}, validate=True)
    self.assertEqual(updated_st.rank, 0)
    self.assertFalse(updated_st.row_partitions)
    self.assertIsNone(updated_st.nrows())

  def test_from_pyval_deep_row_partitions(self):
    """See b/179195750."""
    st = structured_tensor.StructuredTensor.from_pyval([{
        "foo": [{
            "bar": [{
                "baz": [b"FW"]
            }]
        }]
    }])
    st2 = st.field_value(("foo", "bar"))
    self.assertLen(st2.row_partitions, st2.rank - 1)

  def test_from_fields_deep_row_partitions(self):
    """Test a field with its own row_partition. See b/179195750."""
    st = structured_tensor.StructuredTensor.from_pyval([[[{"baz": [b"FW"]}]]])
    self.assertLen(st.row_partitions, st.rank - 1)
    st2 = structured_tensor.StructuredTensor.from_fields(
        fields={"bar": st}, shape=(None, None), validate=False)
    st3 = st2.field_value("bar")
    self.assertLen(st3.row_partitions, st3.rank - 1)

  def test_structured_tensor_spec_shape_property(self):
    spec = StructuredTensor.Spec._from_shape(DynamicRaggedShape.Spec(
        row_partitions=[],
        static_inner_shape=[1, 2],
        dtype=dtypes.int64))
    self.assertEqual(spec.shape.as_list(), [1, 2])
    spec = StructuredTensor.Spec._from_shape(DynamicRaggedShape.Spec(
        row_partitions=[],
        static_inner_shape=[None],
        dtype=dtypes.int64))
    self.assertEqual(spec.shape.as_list(), [None])

  def test_dynamic_ragged_shape_init_vector(self):
    x = constant_op.constant([1, 2, 3, 4])
    y = constant_op.constant([[1, 2], [3, 4], [5, 6], [7, 8]])
    fields = {"x": x, "y": y}
    nrows = constant_op.constant(4)
    shape = tensor_shape.TensorShape((4,))
    row_partitions = ()
    rs = structured_tensor_dynamic._dynamic_ragged_shape_init(
        fields, shape, nrows, row_partitions)
    self.assertEqual(
        repr(rs._to_tensor_shape()), repr(tensor_shape.TensorShape((4,))))

  def test_dynamic_ragged_shape_init_scalar(self):
    x = constant_op.constant([1, 2, 3, 4])
    y = constant_op.constant([[1, 2], [3, 4], [5, 6], [7, 8]])
    fields = {"x": x, "y": y}
    nrows = None
    shape = tensor_shape.TensorShape(())
    row_partitions = ()

    rs = structured_tensor_dynamic._dynamic_ragged_shape_init(
        fields, shape, nrows, row_partitions)
    self.assertEqual(
        repr(rs._to_tensor_shape()), repr(tensor_shape.TensorShape(())))

  def test_dynamic_ragged_shape_init_ragged(self):
    x = ragged_factory_ops.constant_value([[1, 2, 3], [4]])
    fields = {"x": x}
    nrows = constant_op.constant(2, dtype=dtypes.int64)
    shape = tensor_shape.TensorShape([2, None])
    row_partitions = tuple(x._nested_row_partitions)
    rs = structured_tensor_dynamic._dynamic_ragged_shape_init(
        fields, shape, nrows, row_partitions)
    self.assertEqual(
        repr(rs._to_tensor_shape()), repr(tensor_shape.TensorShape((2, None))))


if __name__ == "__main__":
  googletest.main()
