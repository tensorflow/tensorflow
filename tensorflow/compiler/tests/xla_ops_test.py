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
"""Tests for XLA op wrappers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class XlaOpsNumericalTest(xla_test.XLATestCase, parameterized.TestCase):

  def _assertOpOutputMatchesExpected(self, op, args, expected,
                                     equality_fn=None):
    with self.session() as session:
      with self.test_scope():
        placeholders = [
            array_ops.placeholder(dtypes.as_dtype(arg.dtype), arg.shape)
            for arg in args
        ]
        feeds = {placeholders[i]: args[i] for i in range(0, len(args))}
        output = op(*placeholders)
      result = session.run(output, feeds)
      if not equality_fn:
        equality_fn = self.assertAllClose
      equality_fn(result, expected, rtol=1e-3)

  @test_util.disable_mlir_bridge('Not supported yet')
  def testAdd(self):
    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          xla.add,
          args=(np.array([1, 2, 3], dtype=dtype),
                np.array([4, 5, 6], dtype=dtype)),
          expected=np.array([5, 7, 9], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          lambda x, y: xla.add(x, y, broadcast_dims=(0,)),
          args=(np.array([[1, 2], [3, 4]], dtype=dtype),
                np.array([7, 11], dtype=dtype)),
          expected=np.array([[8, 9], [14, 15]], dtype=dtype))

      self._assertOpOutputMatchesExpected(
          lambda x, y: xla.add(x, y, broadcast_dims=(1,)),
          args=(np.array([[1, 2], [3, 4]], dtype=dtype),
                np.array([7, 11], dtype=dtype)),
          expected=np.array([[8, 13], [10, 15]], dtype=dtype))

  @test_util.disable_mlir_bridge('Not supported yet')
  def testBroadcast(self):
    for dtype in self.numeric_types:
      v = np.arange(4, dtype=np.int32).astype(dtype).reshape([2, 2])
      self._assertOpOutputMatchesExpected(
          lambda x: xla.broadcast(x, (7, 42)),
          args=(v,),
          expected=np.tile(v, (7, 42, 1, 1)))

  @test_util.disable_mlir_bridge('Unsigned ints are not supported yet')
  def testShiftRightLogical(self):
    self._assertOpOutputMatchesExpected(
        xla.shift_right_logical,
        args=(np.array([-1, 16], dtype=np.int32), np.int32(4)),
        expected=np.array([0x0FFFFFFF, 1], dtype=np.int32))

    self._assertOpOutputMatchesExpected(
        xla.shift_right_logical,
        args=(np.array([0xFFFFFFFF, 16], dtype=np.uint32), np.uint32(4)),
        expected=np.array([0x0FFFFFFF, 1], dtype=np.uint32))

  @test_util.disable_mlir_bridge('Unsigned ints are not supported yet')
  def testShiftRightArithmetic(self):
    self._assertOpOutputMatchesExpected(
        xla.shift_right_arithmetic,
        args=(np.array([-1, 16], dtype=np.int32), np.int32(4)),
        expected=np.array([-1, 1], dtype=np.int32))

    self._assertOpOutputMatchesExpected(
        xla.shift_right_arithmetic,
        args=(np.array([0xFFFFFFFF, 16], dtype=np.uint32), np.uint32(4)),
        expected=np.array([0xFFFFFFFF, 1], dtype=np.uint32))

  PRECISION_VALUES = (None, xla_data_pb2.PrecisionConfig.DEFAULT,
                      xla_data_pb2.PrecisionConfig.HIGH,
                      xla_data_pb2.PrecisionConfig.HIGHEST)

  @parameterized.parameters(*PRECISION_VALUES)
  @test_util.disable_mlir_bridge('Not supported yet')
  def testConv(self, precision):
    for dtype in set(self.float_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])):

      def conv_1d_fn(lhs, rhs):
        dnums = xla_data_pb2.ConvolutionDimensionNumbers()
        num_spatial_dims = 1
        dnums.input_batch_dimension = 0
        dnums.input_feature_dimension = 1
        dnums.output_batch_dimension = 0
        dnums.output_feature_dimension = 1
        dnums.kernel_output_feature_dimension = 0
        dnums.kernel_input_feature_dimension = 1
        dnums.input_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
        dnums.kernel_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
        dnums.output_spatial_dimensions.extend(range(2, 2 + num_spatial_dims))
        precision_config = None
        if precision:
          precision_config = xla_data_pb2.PrecisionConfig()
          precision_config.operand_precision.extend([precision, precision])
        return xla.conv(
            lhs,
            rhs,
            window_strides=(1,),
            padding=((2, 1),),
            lhs_dilation=(1,),
            rhs_dilation=(2,),
            dimension_numbers=dnums)

      self._assertOpOutputMatchesExpected(
          conv_1d_fn,
          args=(
              np.array([[[3, 4, 5, 6]]], dtype=dtype),
              np.array([[[-2, -3]]], dtype=dtype),
          ),
          expected=np.array([[[-9, -12, -21, -26, -10]]], dtype=dtype))

  @parameterized.parameters(*PRECISION_VALUES)
  def testDotGeneral(self, precision):
    for dtype in self.float_types:

      def dot_fn(lhs, rhs):
        dnums = xla_data_pb2.DotDimensionNumbers()
        dnums.lhs_contracting_dimensions.append(2)
        dnums.rhs_contracting_dimensions.append(1)
        dnums.lhs_batch_dimensions.append(0)
        dnums.rhs_batch_dimensions.append(0)
        precision_config = None
        if precision:
          precision_config = xla_data_pb2.PrecisionConfig()
          precision_config.operand_precision.extend([precision, precision])
        return xla.dot_general(
            lhs,
            rhs,
            dimension_numbers=dnums,
            precision_config=precision_config)

      lhs = np.array(
          [
              [[1, 2], [3, 4]],
              [[5, 6], [7, 8]],
          ], dtype=dtype)
      rhs = np.array(
          [
              [[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]],
          ], dtype=dtype)
      self._assertOpOutputMatchesExpected(
          dot_fn,
          args=(lhs, rhs),
          expected=np.array(
              [
                  [[9, 12, 15], [19, 26, 33]],
                  [[95, 106, 117], [129, 144, 159]],
              ],
              dtype=dtype))

  def testNeg(self):
    for dtype in self.numeric_types - {np.uint8, np.int8}:
      self._assertOpOutputMatchesExpected(
          xla.neg,
          args=(np.array([1, 2, 3], dtype=dtype),),
          expected=np.array([-1, -2, -3], dtype=dtype))

  @test_util.disable_mlir_bridge('Not supported yet')
  def testPad(self):
    for dtype in self.numeric_types:

      def pad_fn(x):
        return xla.pad(
            x,
            padding_value=7,
            padding_low=[2, 1],
            padding_high=[1, 2],
            padding_interior=[1, 0])

      self._assertOpOutputMatchesExpected(
          pad_fn,
          args=(np.arange(4, dtype=np.int32).astype(dtype).reshape([2, 2]),),
          expected=np.array(
              [[7, 7, 7, 7, 7], [7, 7, 7, 7, 7], [7, 0, 1, 7, 7],
               [7, 7, 7, 7, 7], [7, 2, 3, 7, 7], [7, 7, 7, 7, 7]],
              dtype=dtype))

  @test_util.disable_mlir_bridge('Not supported yet')
  def testReduce(self):
    for dtype in set(self.numeric_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])):

      @function.Defun(dtype, dtype)
      def sum_reducer(x, y):
        return x + y

      def sum_reduction(dims):

        def fn(x):
          return xla.reduce(
              x, init_value=0, dimensions_to_reduce=dims, reducer=sum_reducer)

        return fn

      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]))
      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[0]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.array([12, 15, 18, 21], dtype=dtype))
      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[1]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.array([6, 22, 38], dtype=dtype))
      self._assertOpOutputMatchesExpected(
          sum_reduction(dims=[0, 1]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=dtype(66))

      @function.Defun(dtype, dtype)
      def mul_reducer(x, y):
        return x * y

      def mul_reduction(dims):

        def fn(x):
          return xla.reduce(
              x, init_value=1, dimensions_to_reduce=dims, reducer=mul_reducer)

        return fn

      self._assertOpOutputMatchesExpected(
          mul_reduction(dims=[0]),
          args=(np.arange(12, dtype=np.int32).astype(dtype).reshape([3, 4]),),
          expected=np.array([0, 45, 120, 231], dtype=dtype))

  @test_util.disable_mlir_bridge('Not supported yet')
  def testSelectAndScatter(self):
    for dtype in set(self.numeric_types).intersection(
        set([dtypes.bfloat16.as_numpy_dtype, np.float32])):

      @function.Defun(dtype, dtype)
      def add_scatter(x, y):
        return x + y

      @function.Defun(dtype, dtype)
      def ge_select(x, y):
        return x >= y

      def test_fn(operand, source):
        return xla.select_and_scatter(
            operand,
            window_dimensions=[2, 3, 1, 1],
            window_strides=[2, 2, 1, 1],
            padding=[[0, 0]] * 4,
            source=source,
            init_value=0,
            select=ge_select,
            scatter=add_scatter)

      self._assertOpOutputMatchesExpected(
          test_fn,
          args=(np.array(
              [[7, 2, 5, 3, 8], [3, 8, 9, 3, 4], [1, 5, 7, 5, 6],
               [0, 6, 2, 10, 2]],
              dtype=dtype).reshape((4, 5, 1, 1)),
                np.array([[2, 6], [3, 1]], dtype=dtype).reshape((2, 2, 1, 1))),
          expected=np.array(
              [[0, 0, 0, 0, 0], [0, 0, 8, 0, 0], [0, 0, 3, 0, 0],
               [0, 0, 0, 1, 0]],
              dtype=dtype).reshape((4, 5, 1, 1)))

  def testTranspose(self):
    for dtype in self.numeric_types:
      v = np.arange(4, dtype=np.int32).astype(dtype).reshape([2, 2])
      self._assertOpOutputMatchesExpected(
          lambda x: xla.transpose(x, [1, 0]), args=(v,), expected=v.T)

  def testDynamicSlice(self):
    for dtype in self.numeric_types:
      self._assertOpOutputMatchesExpected(
          xla.dynamic_slice,
          args=(np.arange(1000,
                          dtype=np.int32).astype(dtype).reshape([10, 10, 10]),
                np.array([5, 7, 3]), np.array([2, 3, 2])),
          expected=np.array(
              np.array([[[573, 574], [583, 584], [593, 594]],
                        [[673, 674], [683, 684], [693, 694]]]),
              dtype=dtype))

  def testDynamicSliceWithIncorrectStartIndicesShape(self):
    with self.session() as session:
      with self.test_scope():
        output = xla.dynamic_slice(
            np.arange(1000, dtype=np.int32).reshape([10, 10, 10]),
            np.array([5, 7]), np.array([2, 3, 4]))
      with self.assertRaises(errors.InvalidArgumentError) as invalid_arg_error:
        session.run(output)
      self.assertRegexpMatches(
          invalid_arg_error.exception.message,
          (r'start_indices must be a vector with length equal to input rank, '
           r'but input rank is 3 and start_indices has shape \[2\].*'))

  def testDynamicSliceWithIncorrectSizeIndicesShape(self):
    with self.session() as session:
      with self.test_scope():
        output = xla.dynamic_slice(
            np.arange(1000, dtype=np.int32).reshape([10, 10, 10]),
            np.array([5, 7, 3]), np.array([2, 3]))
      with self.assertRaises(errors.InvalidArgumentError) as invalid_arg_error:
        session.run(output)
      self.assertRegexpMatches(
          invalid_arg_error.exception.message,
          (r'size_indices must be a vector with length equal to input rank, '
           r'but input rank is 3 and size_indices has shape \[2\].*'))


class XlaOpsShapeInferenceTest(xla_test.XLATestCase, parameterized.TestCase):

  def testDotDifferentNumberOfContractingDimensions(self):
    a = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(3)

    with self.assertRaisesRegex(ValueError,
                                'Must specify the same number of contracting '
                                'dimensions for lhs and rhs. Got: 1 and 2'):
      xla.dot_general(a, b, dim_nums)

  def testDotDifferentContractingDimensionsSizes(self):
    a = array_ops.placeholder(np.float32, shape=(2, 2, 2, 2))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(3)

    with self.assertRaisesRegex(ValueError,
                                'Contracting dimension sizes do not match. '
                                'Got: 2 and 4'):
      xla.dot_general(a, b, dim_nums)

  def testDotDifferentNumberOfBatchDimensions(self):
    a = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 4))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_batch_dimensions.append(2)
    dim_nums.rhs_batch_dimensions.append(2)
    dim_nums.rhs_batch_dimensions.append(3)

    with self.assertRaisesRegex(ValueError,
                                'Must specify the same number of batch '
                                'dimensions for lhs and rhs. Got: 1 and 2'):
      xla.dot_general(a, b, dim_nums)

  def testDotDifferentBatchDimensionsSizes(self):
    a = array_ops.placeholder(np.float32, shape=(2, 2, 2, 2))
    b = array_ops.placeholder(np.float32, shape=(4, 4, 4, 2))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(2)
    dim_nums.rhs_contracting_dimensions.append(3)
    dim_nums.lhs_batch_dimensions.append(0)
    dim_nums.rhs_batch_dimensions.append(0)

    with self.assertRaisesRegex(ValueError,
                                'Batch dimension sizes do not match. '
                                'Got: 2 and 4'):
      xla.dot_general(a, b, dim_nums)

  def testDotShapeInference(self):
    a = array_ops.placeholder(np.float32, shape=(1, 2, 3, 4))
    b = array_ops.placeholder(np.float32, shape=(4, 3, 2, 1))

    dim_nums = xla_data_pb2.DotDimensionNumbers()
    dim_nums.lhs_contracting_dimensions.append(1)
    dim_nums.rhs_contracting_dimensions.append(2)
    dim_nums.lhs_batch_dimensions.append(3)
    dim_nums.rhs_batch_dimensions.append(0)

    c = xla.dot_general(a, b, dim_nums)
    self.assertEqual(c.shape, tensor_shape.TensorShape([4, 1, 3, 3, 1]))


if __name__ == '__main__':
  # This test is using Tensorflow sessions which are not compatible with eager
  # mode.
  ops.disable_eager_execution()
  googletest.main()
