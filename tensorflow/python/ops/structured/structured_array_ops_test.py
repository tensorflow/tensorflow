# Lint as python3
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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import row_partition
from tensorflow.python.ops.structured import structured_array_ops
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.ops.structured.structured_tensor import StructuredTensor
from tensorflow.python.platform import googletest


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
    actual = structured_array_ops.zeros_like_object(st, dtype)
    self.assertAllEqual(actual, expected)

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
  ])  # pyformat: disable
  def testZerosLikeObjectAlt(self, values, dtype, expected):
    st = StructuredTensor.from_pyval(values)
    # NOTE: zeros_like is very robust. There aren't arguments that
    # should cause this operation to fail.
    actual = structured_array_ops.zeros_like_object(st, dtype)
    self.assertAllEqual(actual, expected)

if __name__ == "__main__":
  googletest.main()
