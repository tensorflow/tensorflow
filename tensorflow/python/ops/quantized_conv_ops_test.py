# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for quantized convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class Conv2DTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(Conv2DTest, self).__init__(method_name)

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                    expected):
    """Verifies the output values of the convolution function.

    Args:
      tensor_in_sizes: Input tensor dimensions in
        [batch, input_rows, input_cols, input_depth].
      filter_in_sizes: Filter tensor dimensions in
        [kernel_rows, kernel_cols, input_depth, output_depth].
      stride: Stride.
      padding: Padding type.
      expected: An array containing the expected operation outputs.
    """
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = np.array([f for f in range(1, total_size_1 + 1)])
    x1 = x1.astype(np.uint8).reshape(tensor_in_sizes)
    x1_min = 0.0
    x1_max = 255.0
    x2 = np.array([f for f in range(1, total_size_2 + 1)]).astype(np.uint8)
    x2 = x2.astype(np.uint8).reshape(filter_in_sizes)
    x2_min = 0.0
    x2_max = 255.0
    with self.test_session(use_gpu=False) as sess:
      t1 = constant_op.constant(x1, shape=tensor_in_sizes, dtype=dtypes.quint8)
      t2 = constant_op.constant(x2, shape=filter_in_sizes, dtype=dtypes.quint8)
      conv = nn_ops.quantized_conv2d(
          t1,
          t2,
          out_type=dtypes.qint32,
          strides=[1, stride, stride, 1],
          padding=padding,
          min_input=x1_min,
          max_input=x1_max,
          min_filter=x2_min,
          max_filter=x2_max)
      value = sess.run(conv)
    quantized_output = value[0]
    output_min = value[1]
    output_max = value[2]
    float_output = self._QuantizedOutputToFloat(quantized_output, output_min,
                                                output_max)
    self.assertArrayNear(expected, float_output.flatten(), 1.0)
    self.assertEqual(value[0].shape, conv[0].get_shape())

  def _assertQuantizedArrayEquals(self, iarray1, iarray2):
    for i1, i2 in zip(iarray1, iarray2):
      self.assertTrue(i1 == i2)

  def _QuantizedOutputToFloat(self, quantized, quantized_min, quantized_max):
    number_of_bits = 32
    number_of_steps = 1 << number_of_bits
    range_adjust = (number_of_steps / (number_of_steps - 1.0))
    quantized_range = ((quantized_max - quantized_min) * range_adjust)
    range_scale = (quantized_range / number_of_steps)
    lowest_quantized = -(1 << (number_of_bits - 1))
    result = np.array([(quantized_min + ((float(x) - lowest_quantized) * range_scale))
                       for x in quantized.flatten()])
    return result

  def testConv2D1x1Filter(self):
    # Our generated input is [batch, rows, cols, depth], and looks like this:
    # (1,2,3)    (4,5,6)    (7,8,9)
    # (10,11,12) (13,14,15) (16,17,18)
    # The filter data is:
    # (1,4,7) (2,5,8) (3,6,9)
    # That means the calculations are:
    # 1*1+2*4+3*7=30
    # 1*2+2*5+3*8=36
    # 1*3+2*6+3*9=42
    # 4*1+5*4+6*7=66
    # 4*2+5*5+6*8=81
    # 4*3+5*6+6*9=96
    # 7*1+5*8+6*9=102
    # 7*2+8*5+9*8=126
    # 7*3+8*6+9*9=150
    # 10*1+11*4+12*7=138
    # 10*2+11*5+12*8=171
    # 10*3+11*6+12*9=204
    # 13*1+14*4+15*7=174
    # 13*2+14*5+15*8=216
    # 13*3+14*6+15*9=258, clamped to 255
    # 16*1+17*4+18*7=210
    # 16*2+17*5+18*8=261, clamped to 255
    # 16*3+17*6+18*9=312, clamped to 255
    # Because the output shift is zero, we call the non-optimized reference
    # path for the convolution.
    expected_output = [
        30, 36, 42, 66, 81, 96, 102, 126, 150, 138, 171, 204, 174, 216, 258,
        210, 261, 312
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2Filter(self):
    # Our generated input is [batch, rows, cols, depth], and looks like this:
    # (1,2,3)    (4,5,6)    (7,8,9)
    # (10,11,12) (13,14,15) (16,17,18)
    # The filter data is [filter_height, filter_width, depth, filter_count]:
    # ( 1, 4, 7) (10, 13, 16)
    # (19,22,25) (28, 31, 34)
    # -
    # ( 2, 5, 8) (11, 14, 17)
    # (20,23,26) (29, 32, 35)
    # -
    # ( 3, 6, 9) (12, 15, 18)
    # (21,24,27) (30, 33, 36)
    # The raw accumulated totals are:
    # 1*1+2*4+3*7+4*10+5*13+6*16+10*19+11*22+12*25+13*28+14*31+15*34=2271
    # 1*2+2*5+3*8+4*11+5*14+6*17+10*20+11*23+12*26+13*29+14*32+15*35=2367
    # 1*3+2*6+3*9+4*12+5*15+6*18+10*21+11*24+12*27+13*30+14*33+15*36=2463
    # 4*1+5*4+6*7+7*10+8*13+9*16+13*19+14*22+15*25+16*28+17*31+18*34=2901
    # 4*2+5*5+6*8+7*11+8*14+9*17+13*20+14*23+15*26+16*29+17*32+18*35=3033
    # 4*3+5*6+6*9+7*12+8*15+9*18+13*21+14*24+15*27+16*30+17*33+18*36=3165
    # The expected values are taken from the raw totals and rescaled to fit into
    # eight bits.
    expected_output = [2271.0, 2367.0, 2463.0, 2901.0, 3033.0, 3165.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)

  def testConv2D1x2Filter(self):
    # The outputs are computed using third_party/py/IPython/notebook.
    # With a shift of 21, we should execute the optimized path here.
    expected_output = [
        231.0, 252.0, 273.0, 384.0, 423.0, 462.0, 690.0, 765.0, 840.0, 843.0,
        936.0, 1029.0
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[1, 2, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterStride2(self):
    # With a shift of 21, we should execute the optimized path here.
    expected_output = [2271.0, 2367.0, 2463.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        stride=2,
        padding="VALID",
        expected=expected_output)

  def testConv2D2x2FilterStride2Same(self):
    # With a shift of 21, we should execute the optimized path here.
    expected_output = [2271.0, 2367.0, 2463.0, 1230.0, 1305.0, 1380.0]
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 3],
        filter_in_sizes=[2, 2, 3, 3],
        stride=2,
        padding="SAME",
        expected=expected_output)


if __name__ == "__main__":
  test.main()
