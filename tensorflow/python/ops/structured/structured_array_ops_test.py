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
"""Tests for structured_array_ops."""

from absl.testing import parameterized

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import row_partition
from tensorflow.python.ops.ragged.dynamic_ragged_shape import DynamicRaggedShape
from tensorflow.python.ops.structured import structured_array_ops
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


# TODO(martinz):create StructuredTensorTestCase.
# pylint: disable=g-long-lambda
@test_util.run_all_in_graph_and_eager_modes
class StructuredArrayOpsTest(test_util.TensorFlowTestCase,
                             parameterized.TestCase):

  def assertAllEqual(self, a, b, msg=None):
    if not (isinstance(a, structured_tensor.StructuredTensor) or
            isinstance(b, structured_tensor.StructuredTensor)):
      return super(StructuredArrayOpsTest, self).assertAllEqual(a, b, msg)

    if not isinstance(a, structured_tensor.StructuredTensor):
      a = structured_tensor.StructuredTensor.from_pyval(a)
    elif not isinstance(b, structured_tensor.StructuredTensor):
      b = structured_tensor.StructuredTensor.from_pyval(b)

    try:
      nest.assert_same_structure(a, b, expand_composites=True)
    except (TypeError, ValueError) as e:
      self.assertIsNone(e, (msg + ": " if msg else "") + str(e))
    a_tensors = [
        x for x in nest.flatten(a, expand_composites=True)
        if isinstance(x, ops.Tensor)
    ]
    b_tensors = [
        x for x in nest.flatten(b, expand_composites=True)
        if isinstance(x, ops.Tensor)
    ]
    self.assertLen(a_tensors, len(b_tensors))
    a_arrays, b_arrays = self.evaluate((a_tensors, b_tensors))
    for a_array, b_array in zip(a_arrays, b_arrays):
      self.assertAllEqual(a_array, b_array, msg)

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
      dict(
          testcase_name="0D_0",
          st={"x": 1},
          axis=0,
          expected=[{"x": 1}]),
      dict(
          testcase_name="0D_minus_1",
          st={"x": 1},
          axis=-1,
          expected=[{"x": 1}]),
      dict(
          testcase_name="1D_0",
          st=[{"x": [1, 3]}, {"x": [2, 7, 9]}],
          axis=0,
          expected=[[{"x": [1, 3]}, {"x": [2, 7, 9]}]]),
      dict(
          testcase_name="1D_1",
          st=[{"x": [1]}, {"x": [2, 10]}],
          axis=1,
          expected=[[{"x": [1]}], [{"x": [2, 10]}]]),
      dict(
          testcase_name="2D_0",
          st=[[{"x": [1]}, {"x": [2]}], [{"x": [3, 4]}]],
          axis=0,
          expected=[[[{"x": [1]}, {"x": [2]}], [{"x": [3, 4]}]]]),
      dict(
          testcase_name="2D_1",
          st=[[{"x": 1}, {"x": 2}], [{"x": 3}]],
          axis=1,
          expected=[[[{"x": 1}, {"x": 2}]], [[{"x": 3}]]]),
      dict(
          testcase_name="2D_2",
          st=[[{"x": [1]}, {"x": [2]}], [{"x": [3, 4]}]],
          axis=2,
          expected=[[[{"x": [1]}], [{"x": [2]}]], [[{"x": [3, 4]}]]]),
      dict(
          testcase_name="3D_0",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=0,
          expected=[[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]],
                     [[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_minus_4",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=-4,  # same as zero
          expected=[[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]],
                     [[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_1",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=1,
          expected=[[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]]],
                    [[[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_minus_3",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=-3,  # same as 1
          expected=[[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]]],
                    [[[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_2",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=2,
          expected=[[[[{"x": [1]}, {"x": [2]}]], [[{"x": [3]}]]],
                    [[[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_minus_2",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=-2,  # same as 2
          expected=[[[[{"x": [1]}, {"x": [2]}]], [[{"x": [3]}]]],
                    [[[{"x": [4, 5]}]]]]),
      dict(
          testcase_name="3D_3",
          st=[[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]],
          axis=3,
          expected=[[[[{"x": [1]}], [{"x": [2]}]], [[{"x": [3]}]]],
                    [[[{"x": [4, 5]}]]]]),
  ])  # pyformat: disable
  def testExpandDims(self, st, axis, expected):
    st = StructuredTensor.from_pyval(st)
    result = array_ops.expand_dims(st, axis)
    self.assertAllEqual(result, expected)

  def testExpandDimsAxisTooBig(self):
    st = [[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]]
    st = StructuredTensor.from_pyval(st)
    with self.assertRaisesRegex(ValueError,
                                "axis=4 out of bounds: expected -4<=axis<4"):
      array_ops.expand_dims(st, 4)

  def testExpandDimsAxisTooSmall(self):
    st = [[[{"x": [1]}, {"x": [2]}], [{"x": [3]}]], [[{"x": [4, 5]}]]]
    st = StructuredTensor.from_pyval(st)
    with self.assertRaisesRegex(ValueError,
                                "axis=-5 out of bounds: expected -4<=axis<4"):
      array_ops.expand_dims(st, -5)

  def testExpandDimsScalar(self):
    # Note that if we expand_dims for the final dimension and there are scalar
    # fields, then the shape is (2, None, None, 1), whereas if it is constructed
    # from pyval it is (2, None, None, None).
    st = [[[{"x": 1}, {"x": 2}], [{"x": 3}]], [[{"x": 4}]]]
    st = StructuredTensor.from_pyval(st)
    result = array_ops.expand_dims(st, 3)
    expected_shape = tensor_shape.TensorShape([2, None, None, 1])
    self.assertEqual(repr(expected_shape), repr(result.shape))

  @parameterized.named_parameters([
      dict(
          testcase_name="scalar_int32",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int32,
          expected=1),
      dict(
          testcase_name="scalar_int64",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int64,
          expected=1),
      dict(
          testcase_name="list_0_int32",
          row_partitions=None,
          shape=(0),
          dtype=dtypes.int32,
          expected=0),
      dict(
          testcase_name="list_0_0_int32",
          row_partitions=None,
          shape=(0, 0),
          dtype=dtypes.int32,
          expected=0),
      dict(
          testcase_name="list_int32",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int32,
          expected=7),
      dict(
          testcase_name="list_int64",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int64,
          expected=7),
      dict(
          testcase_name="matrix_int32",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          dtype=dtypes.int32,
          expected=6),
      dict(
          testcase_name="tensor_int32",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          dtype=dtypes.int32,
          expected=6),
      dict(
          testcase_name="ragged_1_int32",
          row_partitions=[[0, 3, 4]],
          shape=(2, None),
          dtype=dtypes.int32,
          expected=4),
      dict(
          testcase_name="ragged_2_float32",
          row_partitions=[[0, 3, 4], [0, 2, 3, 5, 7]],
          shape=(2, None, None),
          dtype=dtypes.float32,
          expected=7),
  ])  # pyformat: disable
  def testSizeObject(self, row_partitions, shape, dtype, expected):
    if row_partitions is not None:
      row_partitions = [
          row_partition.RowPartition.from_row_splits(r) for r in row_partitions
      ]
    st = StructuredTensor.from_fields({},
                                      shape=shape,
                                      row_partitions=row_partitions)
    # NOTE: size is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.size(st, out_type=dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.size_v2(st, out_type=dtype)
    self.assertAllEqual(actual2, expected)

  def test_shape_v2(self):
    rt = ragged_tensor.RaggedTensor.from_row_lengths(["a", "b", "c"], [1, 2])
    st = StructuredTensor.from_fields_and_rank({"r": rt}, rank=2)
    actual = array_ops.shape_v2(st, out_type=dtypes.int64)
    actual_static_lengths = actual.static_lengths()
    self.assertAllEqual([2, (1, 2)], actual_static_lengths)

  def test_shape(self):
    rt = ragged_tensor.RaggedTensor.from_row_lengths(["a", "b", "c"], [1, 2])
    st = StructuredTensor.from_fields_and_rank({"r": rt}, rank=2)
    actual = array_ops.shape(st, out_type=dtypes.int64).static_lengths()
    actual_v2 = array_ops.shape_v2(st, out_type=dtypes.int64).static_lengths()
    expected = [2, (1, 2)]
    self.assertAllEqual(expected, actual)
    self.assertAllEqual(expected, actual_v2)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty_2_1",
          values=[[{}, {}], [{}]],
          dtype=dtypes.int32,
          expected=3),
      dict(
          testcase_name="list_empty_2",
          values=[{}, {}],
          dtype=dtypes.int32,
          expected=2),
      dict(
          testcase_name="list_empty_1",
          values=[{}],
          dtype=dtypes.int32,
          expected=1),
      dict(
          testcase_name="list_example_1",
          values=[{"x": [3]}, {"x": [4, 5]}],
          dtype=dtypes.int32,
          expected=2),
      dict(
          testcase_name="list_example_2",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=dtypes.float32,
          expected=3),
      dict(
          testcase_name="list_example_2_None",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=None,
          expected=3),
  ])  # pyformat: disable
  def testSizeAlt(self, values, dtype, expected):
    st = StructuredTensor.from_pyval(values)
    # NOTE: size is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.size(st, out_type=dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.size_v2(st, out_type=dtype)
    self.assertAllEqual(actual2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="scalar_int32",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int32,
          expected=0),
      dict(
          testcase_name="scalar_bool",
          row_partitions=None,
          shape=(),
          dtype=dtypes.bool,
          expected=False),
      dict(
          testcase_name="scalar_int64",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int64,
          expected=0),
      dict(
          testcase_name="scalar_float32",
          row_partitions=None,
          shape=(),
          dtype=dtypes.float32,
          expected=0.0),
      dict(
          testcase_name="list_0_int32",
          row_partitions=None,
          shape=(0),
          dtype=dtypes.int32,
          expected=[]),
      dict(
          testcase_name="list_0_0_int32",
          row_partitions=None,
          shape=(0, 0),
          dtype=dtypes.int32,
          expected=[]),
      dict(
          testcase_name="list_int32",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int32,
          expected=[0, 0, 0, 0, 0, 0, 0]),
      dict(
          testcase_name="list_int64",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int64,
          expected=[0, 0, 0, 0, 0, 0, 0]),
      dict(
          testcase_name="list_float32",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.float32,
          expected=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
      dict(
          testcase_name="matrix_int32",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          dtype=dtypes.int32,
          expected=[[0, 0, 0], [0, 0, 0]]),
      dict(
          testcase_name="matrix_float64",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          dtype=dtypes.float64,
          expected=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
      dict(
          testcase_name="tensor_int32",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          dtype=dtypes.int32,
          expected=[[[0], [0], [0]], [[0], [0], [0]]]),
      dict(
          testcase_name="tensor_float32",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          dtype=dtypes.float32,
          expected=[[[0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0]]]),
      dict(
          testcase_name="ragged_1_float32",
          row_partitions=[[0, 3, 4]],
          shape=(2, None),
          dtype=dtypes.float32,
          expected=[[0.0, 0.0, 0.0], [0.0]]),
      dict(
          testcase_name="ragged_2_float32",
          row_partitions=[[0, 3, 4], [0, 2, 3, 5, 7]],
          shape=(2, None, None),
          dtype=dtypes.float32,
          expected=[[[0.0, 0.0], [0.0], [0.0, 0.0]], [[0.0, 0.0]]]),
  ])  # pyformat: disable
  def testZerosLikeObject(self, row_partitions, shape, dtype, expected):
    if row_partitions is not None:
      row_partitions = [
          row_partition.RowPartition.from_row_splits(r) for r in row_partitions
      ]
    st = StructuredTensor.from_fields({},
                                      shape=shape,
                                      row_partitions=row_partitions)
    # NOTE: zeros_like is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.zeros_like(st, dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.zeros_like_v2(st, dtype)
    self.assertAllEqual(actual2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty_2_1",
          values=[[{}, {}], [{}]],
          dtype=dtypes.int32,
          expected=[[0, 0], [0]]),
      dict(
          testcase_name="list_empty_2",
          values=[{}, {}],
          dtype=dtypes.int32,
          expected=[0, 0]),
      dict(
          testcase_name="list_empty_1",
          values=[{}],
          dtype=dtypes.int32,
          expected=[0]),
      dict(
          testcase_name="list_example_1",
          values=[{"x": [3]}, {"x": [4, 5]}],
          dtype=dtypes.int32,
          expected=[0, 0]),
      dict(
          testcase_name="list_example_2",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=dtypes.float32,
          expected=[[0.0], [0.0, 0.0]]),
      dict(
          testcase_name="list_example_2_None",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=None,
          expected=[[0.0], [0.0, 0.0]]),
  ])  # pyformat: disable
  def testZerosLikeObjectAlt(self, values, dtype, expected):
    st = StructuredTensor.from_pyval(values)
    # NOTE: zeros_like is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.zeros_like(st, dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.zeros_like_v2(st, dtype)
    self.assertAllEqual(actual2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="scalar_int32",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int32,
          expected=1),
      dict(
          testcase_name="scalar_bool",
          row_partitions=None,
          shape=(),
          dtype=dtypes.bool,
          expected=True),
      dict(
          testcase_name="scalar_int64",
          row_partitions=None,
          shape=(),
          dtype=dtypes.int64,
          expected=1),
      dict(
          testcase_name="scalar_float32",
          row_partitions=None,
          shape=(),
          dtype=dtypes.float32,
          expected=1.0),
      dict(
          testcase_name="list_0_int32",
          row_partitions=None,
          shape=(0),
          dtype=dtypes.int32,
          expected=[]),
      dict(
          testcase_name="list_0_0_int32",
          row_partitions=None,
          shape=(0, 0),
          dtype=dtypes.int32,
          expected=[]),
      dict(
          testcase_name="list_int32",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int32,
          expected=[1, 1, 1, 1, 1, 1, 1]),
      dict(
          testcase_name="list_int64",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.int64,
          expected=[1, 1, 1, 1, 1, 1, 1]),
      dict(
          testcase_name="list_float32",
          row_partitions=None,
          shape=(7),
          dtype=dtypes.float32,
          expected=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
      dict(
          testcase_name="matrix_int32",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          dtype=dtypes.int32,
          expected=[[1, 1, 1], [1, 1, 1]]),
      dict(
          testcase_name="matrix_float64",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          dtype=dtypes.float64,
          expected=[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
      dict(
          testcase_name="tensor_int32",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          dtype=dtypes.int32,
          expected=[[[1], [1], [1]], [[1], [1], [1]]]),
      dict(
          testcase_name="tensor_float32",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          dtype=dtypes.float32,
          expected=[[[1.0], [1.0], [1.0]], [[1.0], [1.0], [1.0]]]),
      dict(
          testcase_name="ragged_1_float32",
          row_partitions=[[0, 3, 4]],
          shape=(2, None),
          dtype=dtypes.float32,
          expected=[[1.0, 1.0, 1.0], [1.0]]),
      dict(
          testcase_name="ragged_2_float32",
          row_partitions=[[0, 3, 4], [0, 2, 3, 5, 7]],
          shape=(2, None, None),
          dtype=dtypes.float32,
          expected=[[[1.0, 1.0], [1.0], [1.0, 1.0]], [[1.0, 1.0]]]),
  ])  # pyformat: disable
  def testOnesLikeObject(self, row_partitions, shape, dtype, expected):
    if row_partitions is not None:
      row_partitions = [
          row_partition.RowPartition.from_row_splits(r) for r in row_partitions
      ]
    st = StructuredTensor.from_fields({},
                                      shape=shape,
                                      row_partitions=row_partitions)
    # NOTE: ones_like is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.ones_like(st, dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.ones_like_v2(st, dtype)
    self.assertAllEqual(actual2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty_2_1",
          values=[[{}, {}], [{}]],
          dtype=dtypes.int32,
          expected=[[1, 1], [1]]),
      dict(
          testcase_name="list_empty_2",
          values=[{}, {}],
          dtype=dtypes.int32,
          expected=[1, 1]),
      dict(
          testcase_name="list_empty_1",
          values=[{}],
          dtype=dtypes.int32,
          expected=[1]),
      dict(
          testcase_name="list_example_1",
          values=[{"x": [3]}, {"x": [4, 5]}],
          dtype=dtypes.int32,
          expected=[1, 1]),
      dict(
          testcase_name="list_example_2",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=dtypes.float32,
          expected=[[1.0], [1.0, 1.0]]),
      dict(
          testcase_name="list_example_2_None",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          dtype=None,
          expected=[[1.0], [1.0, 1.0]]),
  ])  # pyformat: disable
  def testOnesLikeObjectAlt(self, values, dtype, expected):
    st = StructuredTensor.from_pyval(values)
    # NOTE: ones_like is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.ones_like(st, dtype)
    self.assertAllEqual(actual, expected)

    actual2 = array_ops.ones_like_v2(st, dtype)
    self.assertAllEqual(actual2, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="scalar",
          row_partitions=None,
          shape=(),
          expected=0),
      dict(
          testcase_name="list_0",
          row_partitions=None,
          shape=(0,),
          expected=1),
      dict(
          testcase_name="list_0_0",
          row_partitions=None,
          shape=(0, 0),
          expected=2),
      dict(
          testcase_name="list_7",
          row_partitions=None,
          shape=(7,),
          expected=1),
      dict(
          testcase_name="matrix",
          row_partitions=[[0, 3, 6]],
          shape=(2, 3),
          expected=2),
      dict(
          testcase_name="tensor",
          row_partitions=[[0, 3, 6], [0, 1, 2, 3, 4, 5, 6]],
          shape=(2, 3, 1),
          expected=3),
      dict(
          testcase_name="ragged_1",
          row_partitions=[[0, 3, 4]],
          shape=(2, None),
          expected=2),
      dict(
          testcase_name="ragged_2",
          row_partitions=[[0, 3, 4], [0, 2, 3, 5, 7]],
          shape=(2, None, None),
          expected=3),
  ])  # pyformat: disable
  def testRank(self, row_partitions, shape, expected):
    if row_partitions is not None:
      row_partitions = [
          row_partition.RowPartition.from_row_splits(r) for r in row_partitions
      ]
    st = StructuredTensor.from_fields({},
                                      shape=shape,
                                      row_partitions=row_partitions)

    # NOTE: rank is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = structured_array_ops.rank(st)
    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty_2_1",
          values=[[{}, {}], [{}]],
          expected=2),
      dict(
          testcase_name="list_empty_2",
          values=[{}, {}],
          expected=1),
      dict(
          testcase_name="list_empty_1",
          values=[{}],
          expected=1),
      dict(
          testcase_name="list_example_1",
          values=[{"x": [3]}, {"x": [4, 5]}],
          expected=1),
      dict(
          testcase_name="list_example_2",
          values=[[{"x": [3]}], [{"x": [4, 5]}, {"x": []}]],
          expected=2),
  ])  # pyformat: disable
  def testRankAlt(self, values, expected):
    st = StructuredTensor.from_pyval(values)
    # NOTE: rank is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = array_ops.rank(st)
    self.assertAllEqual(expected, actual)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty",
          values=[[{}], [{}]],
          axis=0,
          expected=[{}, {}]),
      dict(
          testcase_name="list_empty_2_1",
          values=[[{}, {}], [{}]],
          axis=0,
          expected=[{}, {}, {}]),
      dict(
          testcase_name="list_with_fields",
          values=[[{"a": 4, "b": [3, 4]}], [{"a": 5, "b": [5, 6]}]],
          axis=0,
          expected=[{"a": 4, "b": [3, 4]}, {"a": 5, "b": [5, 6]}]),
      dict(
          testcase_name="list_with_submessages",
          values=[[{"a": {"foo": 3}, "b": [3, 4]}],
                  [{"a": {"foo": 4}, "b": [5, 6]}]],
          axis=0,
          expected=[{"a": {"foo": 3}, "b": [3, 4]},
                    {"a": {"foo": 4}, "b": [5, 6]}]),
      dict(
          testcase_name="list_with_empty_submessages",
          values=[[{"a": {}, "b": [3, 4]}],
                  [{"a": {}, "b": [5, 6]}]],
          axis=0,
          expected=[{"a": {}, "b": [3, 4]},
                    {"a": {}, "b": [5, 6]}]),
      dict(
          testcase_name="lists_of_lists",
          values=[[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                   [{"a": {}, "b": [7, 8, 9]}]],
                  [[{"a": {}, "b": [10]}]]],
          axis=0,
          expected=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                    [{"a": {}, "b": [7, 8, 9]}],
                    [{"a": {}, "b": [10]}]]),
      dict(
          testcase_name="lists_of_lists_axis_1",
          values=[[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                   [{"a": {}, "b": [7, 8, 9]}]],
                  [[{"a": {}, "b": []}], [{"a": {}, "b": [3]}]]],
          axis=1,
          expected=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]},
                     {"a": {}, "b": []}],
                    [{"a": {}, "b": [7, 8, 9]}, {"a": {}, "b": [3]}]]),
      dict(
          testcase_name="lists_of_lists_axis_minus_2",
          values=[[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                   [{"a": {}, "b": [7, 8, 9]}]],
                  [[{"a": {}, "b": [10]}]]],
          axis=-2,  # Same as axis=0.
          expected=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                    [{"a": {}, "b": [7, 8, 9]}],
                    [{"a": {}, "b": [10]}]]),
      dict(
          testcase_name="from_structured_tensor_util_test",
          values=[[{"x0": 0, "y": {"z": [[3, 13]]}},
                   {"x0": 1, "y": {"z": [[3], [4, 13]]}},
                   {"x0": 2, "y": {"z": [[3, 5], [4]]}}],
                  [{"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                   {"x0": 4, "y": {"z": [[3], [4]]}}]],
          axis=0,
          expected=[{"x0": 0, "y": {"z": [[3, 13]]}},
                    {"x0": 1, "y": {"z": [[3], [4, 13]]}},
                    {"x0": 2, "y": {"z": [[3, 5], [4]]}},
                    {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                    {"x0": 4, "y": {"z": [[3], [4]]}}]),
  ])  # pyformat: disable
  def testConcat(self, values, axis, expected):
    values = [StructuredTensor.from_pyval(v) for v in values]
    actual = array_ops.concat(values, axis)
    self.assertAllEqual(actual, expected)

  def testConcatTuple(self):
    values = (StructuredTensor.from_pyval([{
        "a": 3
    }]), StructuredTensor.from_pyval([{
        "a": 4
    }]))
    actual = array_ops.concat(values, axis=0)
    self.assertAllEqual(actual, [{"a": 3}, {"a": 4}])

  @parameterized.named_parameters([
      dict(
          testcase_name="field_dropped",
          values=[[{"a": [2]}], [{}]],
          axis=0,
          error_type=ValueError,
          error_regex="a"),
      dict(
          testcase_name="field_added",
          values=[[{"b": [3]}], [{"b": [3], "a": [7]}]],
          axis=0,
          error_type=ValueError,
          error_regex="b"),
      dict(testcase_name="rank_submessage_change",
           values=[[{"a": [{"b": [[3]]}]}],
                   [{"a": [[{"b": [3]}]]}]],
           axis=0,
           error_type=ValueError,
           error_regex="Ranks of sub-message do not match",
          ),
      dict(testcase_name="rank_message_change",
           values=[[{"a": [3]}],
                   [[{"a": 3}]]],
           axis=0,
           error_type=ValueError,
           error_regex="Ranks of sub-message do not match",
          ),
      dict(testcase_name="concat_scalar",
           values=[{"a": [3]}, {"a": [4]}],
           axis=0,
           error_type=ValueError,
           error_regex="axis=0 out of bounds",
          ),
      dict(testcase_name="concat_axis_large",
           values=[[{"a": [3]}], [{"a": [4]}]],
           axis=1,
           error_type=ValueError,
           error_regex="axis=1 out of bounds",
          ),
      dict(testcase_name="concat_axis_large_neg",
           values=[[{"a": [3]}], [{"a": [4]}]],
           axis=-2,
           error_type=ValueError,
           error_regex="axis=-2 out of bounds",
          ),
      dict(testcase_name="concat_deep_rank_wrong",
           values=[[{"a": [3]}], [{"a": [[4]]}]],
           axis=0,
           error_type=ValueError,
           error_regex="must have rank",
          ),
  ])  # pyformat: disable
  def testConcatError(self, values, axis, error_type, error_regex):
    values = [StructuredTensor.from_pyval(v) for v in values]
    with self.assertRaisesRegex(error_type, error_regex):
      array_ops.concat(values, axis)

  def testConcatWithRagged(self):
    values = [StructuredTensor.from_pyval({}), array_ops.constant(3)]
    with self.assertRaisesRegex(ValueError,
                                "values must be a list of StructuredTensors"):
      array_ops.concat(values, 0)

  def testConcatNotAList(self):
    values = StructuredTensor.from_pyval({})
    with self.assertRaisesRegex(ValueError,
                                "values must be a list of StructuredTensors"):
      structured_array_ops.concat(values, 0)

  def testConcatEmptyList(self):
    with self.assertRaisesRegex(ValueError, "values must not be an empty list"):
      structured_array_ops.concat([], 0)

  def testExtendOpErrorNotList(self):
    # Should be a list.
    values = StructuredTensor.from_pyval({})

    def leaf_op(values):
      return values[0]

    with self.assertRaisesRegex(ValueError, "Expected a list"):
      structured_array_ops._extend_op(values, leaf_op)

  def testExtendOpErrorEmptyList(self):

    def leaf_op(values):
      return values[0]

    with self.assertRaisesRegex(ValueError, "List cannot be empty"):
      structured_array_ops._extend_op([], leaf_op)

  def testRandomShuffle2021(self):
    original = StructuredTensor.from_pyval([
        {"x0": 0, "y": {"z": [[3, 13]]}},
        {"x0": 1, "y": {"z": [[3], [4, 13]]}},
        {"x0": 2, "y": {"z": [[3, 5], [4]]}},
        {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
        {"x0": 4, "y": {"z": [[3], [4]]}}])  # pyformat: disable
    random_seed.set_seed(1066)
    result = random_ops.random_shuffle(original, seed=2021)
    expected = StructuredTensor.from_pyval([
        {"x0": 0, "y": {"z": [[3, 13]]}},
        {"x0": 1, "y": {"z": [[3], [4, 13]]}},
        {"x0": 4, "y": {"z": [[3], [4]]}},
        {"x0": 2, "y": {"z": [[3, 5], [4]]}},
        {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},])  # pyformat: disable
    self.assertAllEqual(result, expected)

  def testRandomShuffle2022Eager(self):
    original = StructuredTensor.from_pyval([
        {"x0": 0, "y": {"z": [[3, 13]]}},
        {"x0": 1, "y": {"z": [[3], [4, 13]]}},
        {"x0": 2, "y": {"z": [[3, 5], [4]]}},
        {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
        {"x0": 4, "y": {"z": [[3], [4]]}}])  # pyformat: disable
    expected = StructuredTensor.from_pyval([
        {"x0": 1, "y": {"z": [[3], [4, 13]]}},
        {"x0": 0, "y": {"z": [[3, 13]]}},
        {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
        {"x0": 4, "y": {"z": [[3], [4]]}},
        {"x0": 2, "y": {"z": [[3, 5], [4]]}}])  # pyformat: disable
    random_seed.set_seed(1066)
    result = structured_array_ops.random_shuffle(original, seed=2022)
    self.assertAllEqual(result, expected)

  def testRandomShuffleScalarError(self):
    original = StructuredTensor.from_pyval(
        {"x0": 2, "y": {"z": [[3, 5], [4]]}})  # pyformat: disable

    with self.assertRaisesRegex(ValueError, "scalar"):
      random_ops.random_shuffle(original)

  def testStructuredTensorArrayLikeNoRank(self):
    """Test when the rank is unknown."""

    @def_function.function
    def my_fun(foo):
      bar_shape = math_ops.range(foo)
      bar = array_ops.zeros(shape=bar_shape)
      structured_array_ops._structured_tensor_like(bar)

    with self.assertRaisesRegex(ValueError,
                                "Can't build StructuredTensor w/ unknown rank"):
      my_fun(array_ops.constant(3))

  def testStructuredTensorArrayRankOneKnownShape(self):
    """Fully test structured_tensor_array_like."""
    foo = array_ops.zeros(shape=[4])
    result = structured_array_ops._structured_tensor_like(foo)
    self.assertAllEqual([{}, {}, {}, {}], result)

  # Note that we have to be careful about whether the indices are int32
  # or int64.
  def testStructuredTensorArrayRankOneUnknownShape(self):
    """Fully test structured_tensor_array_like."""

    @def_function.function
    def my_fun(my_shape):
      my_zeros = array_ops.zeros(my_shape)
      return structured_array_ops._structured_tensor_like(my_zeros)

    result = my_fun(array_ops.constant(4))
    shape = DynamicRaggedShape._from_inner_shape([4], dtype=dtypes.int32)
    expected = StructuredTensor.from_shape(shape)
    self.assertAllEqual(expected, result)

  def testStructuredTensorArrayRankTwoUnknownShape(self):
    """Fully test structured_tensor_array_like."""

    @def_function.function
    def my_fun(my_shape):
      my_zeros = array_ops.zeros(my_shape)
      return structured_array_ops._structured_tensor_like(my_zeros)

    result = my_fun(array_ops.constant([2, 2]))
    self.assertAllEqual([[{}, {}], [{}, {}]], result)

  def testStructuredTensorArrayRankZero(self):
    """Fully test structured_tensor_array_like."""
    foo = array_ops.zeros(shape=[])
    result = structured_array_ops._structured_tensor_like(foo)
    self.assertAllEqual({}, result)

  def testStructuredTensorLikeStructuredTensor(self):
    """Fully test structured_tensor_array_like."""
    foo = structured_tensor.StructuredTensor.from_pyval([{"a": 3}, {"a": 7}])
    result = structured_array_ops._structured_tensor_like(foo)
    self.assertAllEqual([{}, {}], result)

  def testStructuredTensorArrayLike(self):
    """There was a bug in a case in a private function.

    This was difficult to reach externally, so I wrote a test
    to check it directly.
    """
    rt = ragged_tensor.RaggedTensor.from_row_splits(
        array_ops.zeros(shape=[5, 3]), [0, 3, 5])
    result = structured_array_ops._structured_tensor_like(rt)
    self.assertEqual(3, result.rank)

  @parameterized.named_parameters([
      dict(
          testcase_name="list_empty",
          params=[{}, {}, {}],
          indices=[0, 2],
          axis=0,
          batch_dims=0,
          expected=[{}, {}]),
      dict(
          testcase_name="list_of_lists_empty",
          params=[[{}, {}], [{}], [{}, {}, {}]],
          indices=[2, 0],
          axis=0,
          batch_dims=0,
          expected=[[{}, {}, {}], [{}, {}]]),
      dict(
          testcase_name="list_with_fields",
          params=[{"a": 4, "b": [3, 4]}, {"a": 5, "b": [5, 6]},
                  {"a": 7, "b": [9, 10]}],
          indices=[2, 0, 0],
          axis=0,
          batch_dims=0,
          expected=[{"a": 7, "b": [9, 10]}, {"a": 4, "b": [3, 4]},
                    {"a": 4, "b": [3, 4]}]),
      dict(
          testcase_name="list_with_submessages",
          params=[{"a": {"foo": 3}, "b": [3, 4]},
                  {"a": {"foo": 4}, "b": [5, 6]},
                  {"a": {"foo": 7}, "b": [9, 10]}],
          indices=[2, 0],
          axis=0,
          batch_dims=0,
          expected=[{"a": {"foo": 7}, "b": [9, 10]},
                    {"a": {"foo": 3}, "b": [3, 4]}]),
      dict(
          testcase_name="list_with_empty_submessages",
          params=[{"a": {}, "b": [3, 4]},
                  {"a": {}, "b": [5, 6]},
                  {"a": {}, "b": [9, 10]}],
          indices=[2, 0],
          axis=0,
          batch_dims=0,
          expected=[{"a": {}, "b": [9, 10]},
                    {"a": {}, "b": [3, 4]}]),
      dict(
          testcase_name="lists_of_lists",
          params=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                  [{"a": {}, "b": [7, 8, 9]}],
                  [{"a": {}, "b": []}]],
          indices=[2, 0, 0],
          axis=0,
          batch_dims=0,
          expected=[[{"a": {}, "b": []}],
                    [{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                    [{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}]]),
      dict(
          testcase_name="lists_of_lists_axis_1",
          params=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                  [{"a": {}, "b": [7, 8, 9]}, {"a": {}, "b": [2, 8, 2]}],
                  [{"a": {}, "b": []}, {"a": {}, "b": [4]}]],
          indices=[1, 0],
          axis=1,
          batch_dims=0,
          expected=[[{"a": {}, "b": [5]}, {"a": {}, "b": [3, 4]}],
                    [{"a": {}, "b": [2, 8, 2]}, {"a": {}, "b": [7, 8, 9]}],
                    [{"a": {}, "b": [4]}, {"a": {}, "b": []}]]),
      dict(
          testcase_name="lists_of_lists_axis_minus_2",
          params=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                  [{"a": {}, "b": [7, 8, 9]}],
                  [{"a": {}, "b": []}]],
          indices=[2, 0, 0],
          axis=-2,  # same as 0
          batch_dims=0,
          expected=[[{"a": {}, "b": []}],
                    [{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                    [{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}]]),
      dict(
          testcase_name="lists_of_lists_axis_minus_1",
          params=[[{"a": {}, "b": [3, 4]}, {"a": {}, "b": [5]}],
                  [{"a": {}, "b": [7, 8, 9]}, {"a": {}, "b": [2, 8, 2]}],
                  [{"a": {}, "b": []}, {"a": {}, "b": [4]}]],
          indices=[1, 0],
          axis=-1,  # same as 1
          batch_dims=0,
          expected=[[{"a": {}, "b": [5]}, {"a": {}, "b": [3, 4]}],
                    [{"a": {}, "b": [2, 8, 2]}, {"a": {}, "b": [7, 8, 9]}],
                    [{"a": {}, "b": [4]}, {"a": {}, "b": []}]]),
      dict(
          testcase_name="from_structured_tensor_util_test",
          params=[{"x0": 0, "y": {"z": [[3, 13]]}},
                  {"x0": 1, "y": {"z": [[3], [4, 13]]}},
                  {"x0": 2, "y": {"z": [[3, 5], [4]]}},
                  {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                  {"x0": 4, "y": {"z": [[3], [4]]}}],
          indices=[1, 0, 4, 3, 2],
          axis=0,
          batch_dims=0,
          expected=[{"x0": 1, "y": {"z": [[3], [4, 13]]}},
                    {"x0": 0, "y": {"z": [[3, 13]]}},
                    {"x0": 4, "y": {"z": [[3], [4]]}},
                    {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                    {"x0": 2, "y": {"z": [[3, 5], [4]]}}]),
      dict(
          testcase_name="scalar_index_axis_0",
          params=[{"x0": 0, "y": {"z": [[3, 13]]}},
                  {"x0": 1, "y": {"z": [[3], [4, 13]]}},
                  {"x0": 2, "y": {"z": [[3, 5], [4]]}},
                  {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                  {"x0": 4, "y": {"z": [[3], [4]]}}],
          indices=3,
          axis=0,
          batch_dims=0,
          expected={"x0": 3, "y": {"z": [[3, 7, 1], [4]]}}),
      dict(
          testcase_name="params_2D_vector_index_axis_1_batch_dims_1",
          params=[[{"x0": 0, "y": {"z": [[3, 13]]}},
                   {"x0": 1, "y": {"z": [[3], [4, 13]]}}],
                  [{"x0": 2, "y": {"z": [[3, 5], [4]]}},
                   {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                   {"x0": 4, "y": {"z": [[3], [4]]}}]],
          indices=[1, 0],
          axis=1,
          batch_dims=1,
          expected=[{"x0": 1, "y": {"z": [[3], [4, 13]]}},
                    {"x0": 2, "y": {"z": [[3, 5], [4]]}}]),
  ])  # pyformat: disable
  def testGather(self, params, indices, axis, batch_dims, expected):
    params = StructuredTensor.from_pyval(params)
    # validate_indices isn't actually used, and we aren't testing names
    actual = array_ops.gather(
        params,
        indices,
        validate_indices=True,
        axis=axis,
        name=None,
        batch_dims=batch_dims)
    self.assertAllEqual(actual, expected)

  @parameterized.named_parameters([
      dict(
          testcase_name="params_2D_index_2D_axis_1_batch_dims_1",
          params=[[{"x0": 0, "y": {"z": [[3, 13]]}},
                   {"x0": 1, "y": {"z": [[3], [4, 13]]}}],
                  [{"x0": 2, "y": {"z": [[3, 5], [4]]}},
                   {"x0": 3, "y": {"z": [[3, 7, 1], [4]]}},
                   {"x0": 4, "y": {"z": [[3], [4]]}}]],
          indices=[[1, 0], [0, 2]],
          axis=1,
          batch_dims=1,
          expected=[[{"x0": 1, "y": {"z": [[3], [4, 13]]}},
                     {"x0": 0, "y": {"z": [[3, 13]]}}],
                    [{"x0": 2, "y": {"z": [[3, 5], [4]]}},
                     {"x0": 4, "y": {"z": [[3], [4]]}}]]),
      dict(
          testcase_name="params_1D_index_2D_axis_0_batch_dims_0",
          params=[{"x0": 0, "y": {"z": [[3, 13]]}}],
          indices=[[0], [0, 0]],
          axis=0,
          batch_dims=0,
          expected=[[{"x0": 0, "y": {"z": [[3, 13]]}}],
                    [{"x0": 0, "y": {"z": [[3, 13]]}},
                     {"x0": 0, "y": {"z": [[3, 13]]}}]]),
  ])  # pyformat: disable
  def testGatherRagged(self, params, indices, axis, batch_dims, expected):
    params = StructuredTensor.from_pyval(params)
    # Shouldn't need to do this, but see cl/366396997
    indices = ragged_factory_ops.constant(indices)
    # validate_indices isn't actually used, and we aren't testing names
    actual = array_ops.gather(
        params,
        indices,
        validate_indices=True,
        axis=axis,
        name=None,
        batch_dims=batch_dims)
    self.assertAllEqual(actual, expected)

  @parameterized.named_parameters([
      dict(testcase_name="params_scalar",
           params={"a": [3]},
           indices=0,
           axis=0,
           batch_dims=0,
           error_type=ValueError,
           error_regex="axis=0 out of bounds",
          ),
      dict(testcase_name="axis_large",
           params=[{"a": [3]}],
           indices=0,
           axis=1,
           batch_dims=0,
           error_type=ValueError,
           error_regex="axis=1 out of bounds",
          ),
      dict(testcase_name="axis_large_neg",
           params=[{"a": [3]}],
           indices=0,
           axis=-2,
           batch_dims=0,
           error_type=ValueError,
           error_regex="axis=-2 out of bounds",
          ),
      dict(testcase_name="batch_large",
           params=[[{"a": [3]}]],
           indices=0,
           axis=0,
           batch_dims=1,
           error_type=ValueError,
           error_regex="batch_dims=1 out of bounds",
          ),
  ])  # pyformat: disable
  def testGatherError(self, params, indices, axis, batch_dims, error_type,
                      error_regex):
    params = StructuredTensor.from_pyval(params)
    with self.assertRaisesRegex(error_type, error_regex):
      structured_array_ops.gather(
          params,
          indices,
          validate_indices=True,
          axis=axis,
          name=None,
          batch_dims=batch_dims)


if __name__ == "__main__":
  googletest.main()
