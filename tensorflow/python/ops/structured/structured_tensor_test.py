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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.platform import googletest


# pylint: disable=g-long-lambda
@test_util.run_all_in_graph_and_eager_modes
class StructuredTensorTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):

  def assertAllEqual(self, a, b, msg=None):
    if not (isinstance(a, structured_tensor.StructuredTensor) or
            isinstance(b, structured_tensor.StructuredTensor)):
      return super(StructuredTensorTest, self).assertAllEqual(a, b, msg)
    if not (isinstance(a, structured_tensor.StructuredTensor) and
            isinstance(b, structured_tensor.StructuredTensor)):
      # TODO(edloper) Add support for this once structured_factory_ops is added.
      raise ValueError("Not supported yet")

    self.assertEqual(repr(a.shape), repr(b.shape))
    self.assertEqual(set(a.field_names()), set(b.field_names()))
    for field in a.field_names():
      self.assertAllEqual(a.field_value(field), b.field_value(field))

  @parameterized.parameters([
      {
          "shape": [],
          "fields": {},
      },
      {
          "shape": [None],
          "fields": {},
      },
      {
          "shape": [1, 5, 3],
          "fields": {},
      },
      {
          "shape": [],
          "fields": {"Foo": 5, "Bar": [1, 2, 3]},
      },
      {
          "shape": [2],
          "fields": {"x": [1, 2], "y": [[1, 2], [3, 4]]},
      },
      {
          "shape": [None],
          "fields": {"x": [1, 2], "y": [[1, 2], [3, 4]]},
          "expected_shape": [2],  # inferred from field values.
      },
      {
          "shape": [],
          "fields": {
              "r": ragged_factory_ops.constant_value([[1, 2], [3]]),
          },
      },
      {
          "shape": [2],
          "fields": {
              "r": ragged_factory_ops.constant_value([[1, 2], [3]]),
          },
      },
      {
          "shape": [2, None],
          "fields": {
              "r": ragged_factory_ops.constant_value(
                  [[[1, 2], [3]], [[4, 5, 6], [7], [8, 9]]]),
          },
          "expected_ragged_rank": 1,
      },
      {
          # Note: fields must have identical row_splits.
          "shape": [2, None],
          "fields": {
              "a": ragged_factory_ops.constant_value([[1, 2], [3]]),
              "b": ragged_factory_ops.constant_value([[4, 5], [6]]),
          },
          "expected_ragged_rank": 1,
      },
      {
          # Note: fields must have identical outer row_splits.
          "shape": [2, None],
          "fields": {
              "a": ragged_factory_ops.constant_value(
                  [[[1, 2], [3]], [[4, 5, 6], [7], [8, 9]]]),
              "b": ragged_factory_ops.constant_value(
                  [[[1], []], [[2, 3], [4, 5, 6], [7, 8]]]),
          },
          "expected_ragged_rank": 1,
      },
  ])  # pyformat: disable
  def testConstructor(self, shape, fields, expected_shape=None,
                      expected_ragged_rank=0):
    struct = structured_tensor.StructuredTensor(shape, fields)
    if expected_shape is None:
      expected_shape = shape
    self.assertEqual(struct.shape.as_list(), expected_shape)
    self.assertLen(expected_shape, struct.rank)
    self.assertEqual(struct.field_names(), tuple(fields.keys()))
    self.assertEqual(struct.ragged_rank, expected_ragged_rank)
    for field, value in fields.items():
      self.assertIsInstance(
          struct.field_value(field),
          (ops.Tensor, structured_tensor.StructuredTensor,
           ragged_tensor.RaggedTensor))
      self.assertAllEqual(struct.field_value(field), value)

  def testNestedStructConstruction(self):
    rt = ragged_factory_ops.constant([[1, 2], [3]])
    struct1 = structured_tensor.StructuredTensor([], {"x": [1, 2]})
    struct2 = structured_tensor.StructuredTensor([2], {"x": [1, 2]})
    struct3 = structured_tensor.StructuredTensor([], {"r": rt, "s": struct1})
    struct4 = structured_tensor.StructuredTensor([2], {"r": rt, "s": struct2})

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

  @parameterized.parameters([
      (object(), {}, TypeError),
      ([], object(), TypeError, "fields must be a dictionary"),
      ([], {1: 2}, TypeError, "Unexpected type for key"),
      ([], {"x": object()}, TypeError, "Unexpected type for value"),
      (None, {}, ValueError, "StructuredTensor's shape must have known rank"),
      ([5], {"f": 5}, ValueError, r"Shapes \(5,\) and \(\) are not compatible"),
      ([None], {"x": [1], "y": []}, ValueError,
       r"Shapes \([01],\) and \([01],\) are not compatible"),
      ([], {"": 5}, ValueError, "Field name '' is not currently allowed."),
      ([], {"_": 5}, ValueError, "Field name '_' is not currently allowed."),
      {
          # Note: fields must have identical outer row_splits.
          "shape": [2, None],
          "fields": {
              "r1": ragged_factory_ops.constant_value(
                  [[1, 2], [3]]),
              "r2": ragged_factory_ops.constant_value(
                  [[1, 2, 3], [4]]),
          },
          "err": errors.InvalidArgumentError,
          "msg": r"Inputs must have identical ragged splits"
      },
  ])  # pyformat: disable
  def testConstructorErrors(self, shape, fields, err, msg=None):
    with self.assertRaisesRegexp(err, msg):
      struct = structured_tensor.StructuredTensor(shape, fields)
      self.evaluate(struct.field_value(struct.field_names()[0]))

  @parameterized.parameters([
      {
          "shape": [3],
          "fields": {"x": [1, 2, 3], "y": [[1, 2], [3, 4], [5, 6]]},
          "row_splits": [0, 2, 3],
      },
  ])  # pyformat: disable
  def testFromRowSplits(self, shape, fields, row_splits, expected_shape=None):
    values = structured_tensor.StructuredTensor(shape, fields)
    struct = structured_tensor.StructuredTensor.from_row_splits(
        values, row_splits)
    if expected_shape is None:
      expected_shape = (
          tensor_shape.TensorShape([None, None]).concatenate(shape[1:]))
      struct.shape.assert_is_compatible_with(expected_shape)
    else:
      self.assertEqual(struct.shape.as_list(), expected_shape)
    self.assertEqual(struct.shape.rank, struct.rank)
    self.assertEqual(struct.field_names(), tuple(fields.keys()))
    for field, value in fields.items():
      self.assertIsInstance(
          struct.field_value(field),
          (ops.Tensor, structured_tensor.StructuredTensor,
           ragged_tensor.RaggedTensor))
      self.assertAllEqual(
          struct.field_value(field),
          ragged_tensor.RaggedTensor.from_row_splits(value, row_splits))

  @parameterized.parameters([
      ([], {}, ["x"], ValueError,
       r"Shape \(\) must have rank at least 1"),
      ([0], {}, ["x"], ValueError,
       r"Row-partitioning tensors must have dtype int32 or int64"),
      ([0], {}, [[0]], ValueError,
       r"Shape \(1, 1\) must have rank 1"),
      ([0], {}, np.array([], np.int32), ValueError,
       r"row_splits may not be empty"),
  ])  # pyformat: disable
  def testFromRowSplitsErrors(self, shape, fields, row_splits, err, msg=None):
    with self.assertRaisesRegexp(err, msg):
      values = structured_tensor.StructuredTensor(shape, fields)
      structured_tensor.StructuredTensor.from_row_splits(values, row_splits)

  def testFromRowSplitsBadValueType(self):
    with self.assertRaisesRegexp(TypeError,
                                 "values must be a StructuredTensor"):
      structured_tensor.StructuredTensor.from_row_splits([1, 2], [0, 2])

  @parameterized.named_parameters([
      {
          "testcase_name": "ScalarEmpty",
          "pyval": {},
          "expected": lambda: structured_tensor.StructuredTensor([], {})
      },
      {
          "testcase_name": "ScalarSimple",
          "pyval": {"a": 12, "b": [1, 2, 3], "c": [[1, 2], [3]]},
          "expected": lambda: structured_tensor.StructuredTensor([], {
              "a": 12,
              "b": [1, 2, 3],
              "c": ragged_factory_ops.constant([[1, 2], [3]])})
      },
      {
          "testcase_name": "ScalarSimpleWithTypeSpec",
          "pyval": {"a": 12, "b": [1, 2, 3], "c": [[1, 2], [3]]},
          "type_spec": structured_tensor.StructuredTensorSpec([], {
              "a": tensor_spec.TensorSpec([], dtypes.int32),
              "b": tensor_spec.TensorSpec([None], dtypes.int32),
              "c": ragged_tensor.RaggedTensorSpec([None, None], dtypes.int32)}),
          "expected": lambda: structured_tensor.StructuredTensor([], {
              "a": 12,
              "b": [1, 2, 3],
              "c": ragged_factory_ops.constant([[1, 2], [3]])})
      },
      {
          "testcase_name": "ScalarWithNestedStruct",
          "pyval": {"a": 12, "b": [1, 2, 3], "c": {"x": b"Z", "y": [10, 20]}},
          "expected": lambda: structured_tensor.StructuredTensor([], {
              "a": 12,
              "b": [1, 2, 3],
              "c": structured_tensor.StructuredTensor([], {
                  "x": "Z",
                  "y": [10, 20]})})
      },
      {
          "testcase_name": "EmptyList",
          "pyval": [],
          "expected": lambda: [],
      },
      {
          "testcase_name": "EmptyListWithTypeSpec",
          "pyval": [],
          "type_spec": structured_tensor.StructuredTensorSpec([0], {
              "a": tensor_spec.TensorSpec(None, dtypes.int32)}),
          "expected": lambda: structured_tensor.StructuredTensor([0], {
              "a": []})
      },
      {
          "testcase_name": "VectorOfDict",
          "pyval": [{"a": 1}, {"a": 2}],
          "expected": lambda: structured_tensor.StructuredTensor([2], {
              "a": [1, 2]})
      },
      {
          "testcase_name": "VectorOfDictWithNestedStructScalar",
          "pyval": [{"a": 1, "b": {"x": [1, 2]}},
                    {"a": 2, "b": {"x": [3]}}],
          "expected": lambda: structured_tensor.StructuredTensor([2], {
              "a": [1, 2],
              "b": structured_tensor.StructuredTensor([2], {
                  "x": ragged_factory_ops.constant([[1, 2], [3]])})}),
      },
      {
          "testcase_name": "VectorOfDictWithNestedStructVector",
          "pyval": [{"a": 1, "b": [{"x": [1, 2]}, {"x": [5]}]},
                    {"a": 2, "b": [{"x": [3]}]}],
          "expected": lambda: structured_tensor.StructuredTensor([2], {
              "a": [1, 2],
              "b": structured_tensor.StructuredTensor([2, None], {
                  "x": ragged_factory_ops.constant([[[1, 2], [5]], [[3]]])})}),
      },
      {
          "testcase_name": "Ragged2DOfDict",
          "pyval": [[{"a": 1}, {"a": 2}, {"a": 3},],
                    [{"a": 4}, {"a": 5}]],
          "expected": lambda: structured_tensor.StructuredTensor([2, None], {
              "a": ragged_factory_ops.constant([[1, 2, 3], [4, 5]])})
      },
      {
          # With no type-spec, all tensors>1D are encoded as ragged:
          "testcase_name": "MatrixOfDictWithoutTypeSpec",
          "pyval": [[{"a": 1}, {"a": 2}, {"a": 3},],
                    [{"a": 4}, {"a": 5}, {"a": 6}]],
          "expected": lambda: structured_tensor.StructuredTensor([2, None], {
              "a": ragged_factory_ops.constant([[1, 2, 3], [4, 5, 6]])})
      },
      {
          # TypeSpec can be used to specify StructuredTensor shape.
          "testcase_name": "MatrixOfDictWithTypeSpec",
          "pyval": [[{"a": 1}, {"a": 2}, {"a": 3},],
                    [{"a": 4}, {"a": 5}, {"a": 6}]],
          "type_spec": structured_tensor.StructuredTensorSpec([2, 3], {
              "a": tensor_spec.TensorSpec(None, dtypes.int32)}),
          "expected": lambda: structured_tensor.StructuredTensor([2, 3], {
              "a": [[1, 2, 3], [4, 5, 6]]})
      },
  ])  # pyformat: disable
  def testPyvalConversion(self, pyval, expected, type_spec=None):
    expected = expected()  # Deferred init because it creates tensors.
    actual = structured_tensor.StructuredTensor.from_pyval(pyval, type_spec)
    self.assertAllEqual(actual, expected)
    if isinstance(actual, structured_tensor.StructuredTensor):
      if context.executing_eagerly():  # to_pyval only available in eager.
        self.assertEqual(actual.to_pyval(), pyval)

  @parameterized.named_parameters([
      {
          "testcase_name": "MissingKeys",
          "pyval": [{"a": [1, 2]}, {"b": [3, 4]}],
          "err": KeyError,
          "msg": "'b'"
      }
  ])  # pyformat: disable
  def testFromPyvalError(self, pyval, err, type_spec=None, msg=None):
    with self.assertRaisesRegexp(err, msg):
      structured_tensor.StructuredTensor.from_pyval(pyval, type_spec)


if __name__ == "__main__":
  googletest.main()
