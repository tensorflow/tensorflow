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
"""Tests for Conv2D via the XLA JIT.

The canned results in these tests are created by running each test using the
Tensorflow CPU device and saving the output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


class Conv2DTest(XLATestCase):

  def _VerifyValues(self,
                    input_sizes=None,
                    filter_sizes=None,
                    strides=None,
                    dilations=None,
                    padding=None,
                    expected=None):
    """Tests that tf.nn.conv2d produces the expected value.

    Args:
      input_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      strides: Strides.
      dilations: RHS dilations.
      padding: Padding type.
      expected: Expected output.
    """
    total_size_1 = np.prod(input_sizes)
    total_size_2 = np.prod(filter_sizes)
    x1 = np.arange(1, total_size_1 + 1, dtype=np.float32).reshape(input_sizes)
    x2 = np.arange(1, total_size_2 + 1, dtype=np.float32).reshape(filter_sizes)
    strides = [1] + strides + [1]
    if dilations is None:
      dilations = [1, 1]
    dilations = [1] + dilations + [1]

    with self.test_session() as sess:
      t1 = array_ops.placeholder(dtypes.float32, shape=input_sizes)
      t2 = array_ops.placeholder(dtypes.float32, shape=filter_sizes)
      with self.test_scope():
        out = nn_ops.conv2d(
            t1,
            t2,
            strides=strides,
            padding=padding,
            data_format="NHWC",
            dilations=dilations)
      value = sess.run(out, {t1: x1, t2: x2})
      self.assertAllClose(expected, value, 1e-3)

  def testConv2D1x1Filter(self):
    expected_output = np.reshape([
        30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
        204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
    ], [1, 2, 3, 3])
    self._VerifyValues(
        input_sizes=[1, 2, 3, 3],
        filter_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2Filter(self):
    expected_output = np.reshape(
        [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0], [1, 1, 2, 3])
    self._VerifyValues(
        input_sizes=[1, 2, 3, 3],
        filter_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2Filter2x1Dilation(self):
    expected_output = np.array([[[[72], [82], [92]], [[112], [122], [132]]]])
    self._VerifyValues(
        input_sizes=[1, 4, 4, 1],
        filter_sizes=[2, 2, 1, 1],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2Filter(self):
    expected_output = np.reshape([
        231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
        936.0, 1029.0
    ], [1, 2, 2, 3])
    self._VerifyValues(
        input_sizes=[1, 2, 3, 3],
        filter_sizes=[1, 2, 3, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterStride2(self):
    expected_output = np.reshape([2271.0, 2367.0, 2463.0], [1, 1, 1, 3])
    self._VerifyValues(
        input_sizes=[1, 2, 3, 3],
        filter_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterStride2Same(self):
    expected_output = np.reshape(
        [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0], [1, 1, 2, 3])
    self._VerifyValues(
        input_sizes=[1, 2, 3, 3],
        filter_sizes=[2, 2, 3, 3],
        strides=[2, 2],
        padding="SAME",
        expected=expected_output)

  def testConv2DEmptyDilation(self):
    self._VerifyValues(
        input_sizes=[0, 2, 3, 3],
        filter_sizes=[1, 1, 3, 3],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID",
        expected=np.zeros([0, 2, 3, 3]))

  def testConv2D2x2FilterDilation(self):
    self._VerifyValues(
        input_sizes=[1, 2, 3, 3],
        filter_sizes=[2, 2, 3, 3],
        strides=[1, 1],
        dilations=[1, 2],
        padding="VALID",
        expected=np.reshape([2667, 2781, 2895], [1, 1, 1, 3]))

  def testConv2D1x2FilterDilation(self):
    self._VerifyValues(
        input_sizes=[1, 2, 3, 3],
        filter_sizes=[1, 2, 3, 3],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID",
        expected=np.array([[[[231, 252, 273], [384, 423, 462]],
                            [[690, 765, 840], [843, 936, 1029]]]]))

  def testConv2DKernelSizeMatchesInputSizeDilation(self):
    self._VerifyValues(
        input_sizes=[1, 3, 3, 1],
        filter_sizes=[2, 2, 1, 2],
        strides=[1, 1],
        dilations=[2, 2],
        padding="VALID",
        expected=np.reshape([108, 128], [1, 1, 1, 2]))


class Conv2DBackpropInputTest(XLATestCase):

  def _VerifyValues(self,
                    input_sizes=None,
                    filter_sizes=None,
                    out_backprop_sizes=None,
                    strides=None,
                    dilations=None,
                    padding=None,
                    expected=None):
    """Tests that gen_nn_ops.conv2d_backprop_input produces the expected output.

    Args:
      input_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      out_backprop_sizes: Output gradients tensor dimensions.
      strides: Strides.
      dilations: Dilations.
      padding: Padding type.
      expected: Expected output.
    """
    total_size_1 = np.prod(filter_sizes)
    total_size_2 = np.prod(out_backprop_sizes)
    x1 = np.arange(1, total_size_1 + 1, dtype=np.float32).reshape(filter_sizes)
    x2 = np.arange(
        1, total_size_2 + 1, dtype=np.float32).reshape(out_backprop_sizes)
    strides = [1] + strides + [1]
    if dilations is not None:
      dilations = [1] + dilations + [1]

    with self.test_session() as sess:
      t1 = array_ops.placeholder(dtypes.float32, shape=filter_sizes)
      t2 = array_ops.placeholder(dtypes.float32, shape=out_backprop_sizes)
      with self.test_scope():
        out = gen_nn_ops.conv2d_backprop_input(
            input_sizes=input_sizes,
            filter=t1,
            out_backprop=t2,
            strides=strides,
            dilations=dilations,
            padding=padding,
            data_format="NHWC")
      value = sess.run(out, {t1: x1, t2: x2})
      self.assertAllEqual(input_sizes, value.shape)
      self.assertAllClose(expected, np.ravel(value), 1e-3)

  def testConv2D1x1Filter(self):
    expected_output = [
        5, 11, 17, 11, 25, 39, 17, 39, 61, 23, 53, 83, 29, 67, 105, 35, 81, 127,
        41, 95, 149, 47, 109, 171, 53, 123, 193, 59, 137, 215, 65, 151, 237, 71,
        165, 259, 77, 179, 281, 83, 193, 303, 89, 207, 325, 95, 221, 347.
    ]
    self._VerifyValues(
        input_sizes=[1, 4, 4, 3],
        filter_sizes=[1, 1, 3, 2],
        out_backprop_sizes=[1, 4, 4, 2],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2FilterStride3Width5(self):
    expected_output = [1, 2, 0, 2, 4]
    self._VerifyValues(
        input_sizes=[1, 1, 5, 1],
        filter_sizes=[1, 2, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[3, 3],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2FilterStride3Width6(self):
    expected_output = [1, 2, 0, 2, 4, 0]
    self._VerifyValues(
        input_sizes=[1, 1, 6, 1],
        filter_sizes=[1, 2, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[3, 3],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2FilterStride3Width7(self):
    expected_output = [1, 2, 0, 2, 4, 0, 0]
    self._VerifyValues(
        input_sizes=[1, 1, 7, 1],
        filter_sizes=[1, 2, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[3, 3],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterC1Same(self):
    expected_output = [1, 4, 7, 7, 23, 33]
    self._VerifyValues(
        input_sizes=[1, 2, 3, 1],
        filter_sizes=[2, 2, 1, 1],
        out_backprop_sizes=[1, 2, 3, 1],
        strides=[1, 1],
        padding="SAME",
        expected=expected_output)

  def testConv2D2x2Filter(self):
    expected_output = [
        14, 32, 50, 100, 163, 226, 167, 212, 257, 122, 140, 158, 478, 541, 604,
        437, 482, 527
    ]
    self._VerifyValues(
        input_sizes=[1, 2, 3, 3],
        filter_sizes=[2, 2, 3, 3],
        out_backprop_sizes=[1, 1, 2, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterSame(self):
    expected_output = [
        14, 32, 50, 100, 163, 226, 217, 334, 451, 190, 307, 424, 929, 1217,
        1505, 1487, 1883, 2279
    ]
    self._VerifyValues(
        input_sizes=[1, 2, 3, 3],
        filter_sizes=[2, 2, 3, 3],
        out_backprop_sizes=[1, 2, 3, 3],
        strides=[1, 1],
        padding="SAME",
        expected=expected_output)

  def testConv2D1x2Filter(self):
    expected_output = [1, 4, 4, 3, 10, 8, 5, 16, 12]
    self._VerifyValues(
        input_sizes=[1, 3, 3, 1],
        filter_sizes=[1, 2, 1, 1],
        out_backprop_sizes=[1, 3, 2, 1],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2FilterSame(self):
    expected_output = [1, 4, 7, 4, 13, 16, 7, 22, 25]
    self._VerifyValues(
        input_sizes=[1, 3, 3, 1],
        filter_sizes=[1, 2, 1, 1],
        out_backprop_sizes=[1, 3, 3, 1],
        strides=[1, 1],
        padding="SAME",
        expected=expected_output)

  def testConv2D2x2FilterStride2(self):
    expected_output = [1, 2, 5, 4, 6, 0, 0, 0, 0, 0, 3, 6, 13, 8, 12]
    self._VerifyValues(
        input_sizes=[1, 3, 5, 1],
        filter_sizes=[1, 3, 1, 1],
        out_backprop_sizes=[1, 2, 2, 1],
        strides=[2, 2],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterStride2Same(self):
    expected_output = [1, 2, 2, 3, 4, 6]
    self._VerifyValues(
        input_sizes=[1, 2, 3, 1],
        filter_sizes=[2, 2, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[2, 2],
        padding="SAME",
        expected=expected_output)

  def testConv2D2x2Depth3ValidBackpropInputStride1x1Dilation2x1(self):
    self._VerifyValues(
        input_sizes=[1, 3, 6, 1],
        filter_sizes=[2, 2, 1, 1],
        out_backprop_sizes=[1, 1, 5, 1],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID",
        expected=[1, 4, 7, 10, 13, 10, 0, 0, 0, 0, 0, 0, 3, 10, 17, 24, 31, 20])

  def testConv2D2x2Depth1ValidBackpropInputDilation1x2(self):
    self._VerifyValues(
        input_sizes=[1, 2, 3, 1],
        filter_sizes=[2, 2, 1, 1],
        out_backprop_sizes=[1, 1, 1, 1],
        strides=[1, 1],
        dilations=[1, 2],
        padding="VALID",
        expected=[1, 0, 2, 3, 0, 4])

  def testConv2DEmptyBackpropInputDilation1x2(self):
    self._VerifyValues(
        input_sizes=[0, 2, 3, 1],
        filter_sizes=[2, 2, 1, 1],
        out_backprop_sizes=[0, 1, 1, 1],
        strides=[1, 1],
        dilations=[1, 2],
        padding="VALID",
        expected=np.zeros([0]))

  def testConv2D2x2Depth3ValidBackpropInputDilation2x1(self):
    # The GPU version of this test is not very stable. So adjusting the
    # error threshold to 1e-4.
    self._VerifyValues(
        input_sizes=[1, 3, 2, 3],
        filter_sizes=[2, 2, 3, 3],
        out_backprop_sizes=[1, 1, 1, 3],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID",
        expected=[
            14, 32, 50, 68, 86, 104, 0, 0, 0, 0, 0, 0, 122, 140, 158, 176, 194,
            212
        ])

  def testConv2DKernelSizeMatchesInputSizeBackpropInputDilation2x2(self):
    self._VerifyValues(
        input_sizes=[1, 3, 3, 1],
        filter_sizes=[2, 2, 1, 2],
        out_backprop_sizes=[1, 1, 1, 2],
        strides=[1, 1],
        dilations=[2, 2],
        padding="VALID",
        expected=[5, 0, 11, 0, 0, 0, 17, 0, 23])


class Conv2DBackpropFilterTest(XLATestCase):

  def _VerifyValues(self,
                    input_sizes=None,
                    filter_sizes=None,
                    out_backprop_sizes=None,
                    strides=None,
                    dilations=None,
                    padding=None,
                    expected=None):
    """Tests that gen_nn_ops.conv2d_backprop_filter produces the right output.

    Args:
      input_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      out_backprop_sizes: Output gradients tensor dimensions.
      strides: Stride.
      dilations: Dilations.
      padding: Padding type.
      expected: Expected output.
    """

    total_size_1 = np.prod(input_sizes)
    total_size_2 = np.prod(out_backprop_sizes)
    x1 = np.arange(1, total_size_1 + 1, dtype=np.float32).reshape(input_sizes)
    x2 = np.arange(
        1, total_size_2 + 1, dtype=np.float32).reshape(out_backprop_sizes)
    strides = [1] + strides + [1]
    if dilations is not None:
      dilations = [1] + dilations + [1]

    with self.test_session() as sess:
      t1 = array_ops.placeholder(dtypes.float32, shape=input_sizes)
      t2 = array_ops.placeholder(dtypes.float32, shape=out_backprop_sizes)
      with self.test_scope():
        tensor = gen_nn_ops.conv2d_backprop_filter(
            input=t1,
            filter_sizes=filter_sizes,
            out_backprop=t2,
            strides=strides,
            dilations=dilations,
            padding=padding,
            data_format="NHWC")

      value = sess.run(tensor, {t1: x1, t2: x2})
      self.assertAllEqual(filter_sizes, value.shape)
      self.assertAllClose(expected, np.ravel(value), 1e-3)

  def testConv2D1x1Filter(self):
    expected_output = [8056, 8432, 8312, 8704, 8568, 8976]
    self._VerifyValues(
        input_sizes=[1, 4, 4, 3],
        filter_sizes=[1, 1, 3, 2],
        out_backprop_sizes=[1, 4, 4, 2],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2Filter(self):
    expected_output = [120, 141]
    self._VerifyValues(
        input_sizes=[1, 3, 3, 1],
        filter_sizes=[1, 2, 1, 1],
        out_backprop_sizes=[1, 3, 2, 1],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterDepth1(self):
    expected_output = [5, 8, 14, 17]
    self._VerifyValues(
        input_sizes=[1, 2, 3, 1],
        filter_sizes=[2, 2, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2Filter(self):
    expected_output = [
        17, 22, 27, 22, 29, 36, 27, 36, 45, 32, 43, 54, 37, 50, 63, 42, 57, 72,
        62, 85, 108, 67, 92, 117, 72, 99, 126, 77, 106, 135, 82, 113, 144, 87,
        120, 153
    ]
    self._VerifyValues(
        input_sizes=[1, 2, 3, 3],
        filter_sizes=[2, 2, 3, 3],
        out_backprop_sizes=[1, 1, 2, 3],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2FilterStride3Width5(self):
    expected_output = [9, 12]
    self._VerifyValues(
        input_sizes=[1, 1, 5, 1],
        filter_sizes=[1, 2, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[3, 3],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2FilterStride3Width6(self):
    expected_output = [9, 12]
    self._VerifyValues(
        input_sizes=[1, 1, 6, 1],
        filter_sizes=[1, 2, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[3, 3],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2FilterStride3Width7(self):
    expected_output = [9, 12]
    self._VerifyValues(
        input_sizes=[1, 1, 7, 1],
        filter_sizes=[1, 2, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[3, 3],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x3Filter(self):
    expected_output = [5, 8, 11]
    self._VerifyValues(
        input_sizes=[1, 1, 4, 1],
        filter_sizes=[1, 3, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[1, 1],
        padding="VALID",
        expected=expected_output)

  def testConv2D1x3FilterSame(self):
    expected_output = [20, 30, 20]
    self._VerifyValues(
        input_sizes=[1, 1, 4, 1],
        filter_sizes=[1, 3, 1, 1],
        out_backprop_sizes=[1, 1, 4, 1],
        strides=[1, 1],
        padding="SAME",
        expected=expected_output)

  def testConv2D1x3FilterSameOutbackprop2(self):
    expected_output = [7, 10, 3]
    self._VerifyValues(
        input_sizes=[1, 1, 4, 1],
        filter_sizes=[1, 3, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[2, 2],
        padding="SAME",
        expected=expected_output)

  def testConv2D2x2FilterC1Same(self):
    expected_output = [91, 58, 32, 17]
    self._VerifyValues(
        input_sizes=[1, 2, 3, 1],
        filter_sizes=[2, 2, 1, 1],
        out_backprop_sizes=[1, 2, 3, 1],
        strides=[1, 1],
        padding="SAME",
        expected=expected_output)

  def testConv2D2x2FilterStride2(self):
    expected_output = [92, 102, 112]
    self._VerifyValues(
        input_sizes=[1, 3, 5, 1],
        filter_sizes=[1, 3, 1, 1],
        out_backprop_sizes=[1, 2, 2, 1],
        strides=[2, 2],
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterStride2Same(self):
    expected_output = [7, 2, 16, 5]
    self._VerifyValues(
        input_sizes=[1, 2, 3, 1],
        filter_sizes=[2, 2, 1, 1],
        out_backprop_sizes=[1, 1, 2, 1],
        strides=[2, 2],
        padding="SAME",
        expected=expected_output)

  def testConv2D2x2Depth3ValidBackpropFilterStride1x1Dilation2x1(self):
    self._VerifyValues(
        input_sizes=[1, 3, 6, 1],
        filter_sizes=[2, 2, 1, 1],
        out_backprop_sizes=[1, 1, 5, 1],
        strides=[1, 1],
        dilations=[2, 1],
        padding="VALID",
        expected=[55, 70, 235, 250])

  def testConv2D2x2Depth1ValidBackpropFilterDilation1x2(self):
    self._VerifyValues(
        input_sizes=[1, 2, 3, 1],
        filter_sizes=[2, 2, 1, 1],
        out_backprop_sizes=[1, 1, 1, 1],
        strides=[1, 1],
        dilations=[1, 2],
        padding="VALID",
        expected=[1, 3, 4, 6])

  def testConv2DEmptyBackpropFilterDilation1x2(self):
    self._VerifyValues(
        input_sizes=[1, 2, 3, 1],
        filter_sizes=[2, 2, 1, 0],
        out_backprop_sizes=[1, 1, 1, 0],
        strides=[1, 1],
        dilations=[1, 2],
        padding="VALID",
        expected=np.zeros([0]))

  def testConv2D2x2Depth3ValidBackpropFilterDilation2x2(self):
    self._VerifyValues(
        input_sizes=[1, 3, 4, 3],
        filter_sizes=[2, 2, 3, 3],
        out_backprop_sizes=[1, 1, 2, 3],
        strides=[1, 1],
        dilations=[2, 2],
        padding="VALID",
        expected=[
            17, 22, 27, 22, 29, 36, 27, 36, 45, 47, 64, 81, 52, 71, 90, 57, 78,
            99, 137, 190, 243, 142, 197, 252, 147, 204, 261, 167, 232, 297, 172,
            239, 306, 177, 246, 315
        ])

  def testConv2DKernelSizeMatchesInputSizeBackpropFilterDilation2x2(self):
    self._VerifyValues(
        input_sizes=[1, 3, 3, 1],
        filter_sizes=[2, 2, 1, 2],
        out_backprop_sizes=[1, 1, 1, 2],
        strides=[1, 1],
        dilations=[2, 2],
        padding="VALID",
        expected=[1, 2, 3, 6, 7, 14, 9, 18])


if __name__ == "__main__":
  googletest.main()
