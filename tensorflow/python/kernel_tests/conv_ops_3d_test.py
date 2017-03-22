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
"""Functional tests for 3d convolutional operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


def GetTestConfigs():
  """Get all the valid tests configs to run.

  Returns:
    all the valid test configs as tuples of data_format and use_gpu.
  """
  test_configs = [("NDHWC", False), ("NDHWC", True)]
  if test.is_gpu_available(cuda_only=True):
    # "NCDHW" format is only supported on CUDA.
    test_configs += [("NCDHW", True)]
  return test_configs


class Conv3DTest(test.TestCase):

  def _SetupValuesForDevice(self, tensor_in_sizes, filter_in_sizes, stride,
                            padding, data_format, use_gpu):
    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
      total_size_1 *= s
    for s in filter_in_sizes:
      total_size_2 *= s

    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    x1 = [f * 1.0 for f in range(1, total_size_1 + 1)]
    x2 = [f * 1.0 for f in range(1, total_size_2 + 1)]
    with self.test_session(use_gpu=use_gpu):
      t1 = constant_op.constant(x1, shape=tensor_in_sizes)
      t2 = constant_op.constant(x2, shape=filter_in_sizes)

      if isinstance(stride, collections.Iterable):
        strides = [1] + list(stride) + [1]
      else:
        strides = [1, stride, stride, stride, 1]

      if data_format == "NCDHW":
        t1 = test_util.NHWCToNCHW(t1)
        strides = test_util.NHWCToNCHW(strides)
      conv = nn_ops.conv3d(t1, t2, strides, padding=padding,
                           data_format=data_format)
      if data_format == "NCDHW":
        conv = test_util.NCHWToNHWC(conv)

      return conv

  def _VerifyValues(self, tensor_in_sizes, filter_in_sizes, stride, padding,
                    expected):
    results = []
    for data_format, use_gpu in GetTestConfigs():
      result = self._SetupValuesForDevice(
          tensor_in_sizes,
          filter_in_sizes,
          stride,
          padding,
          data_format,
          use_gpu=use_gpu)
      results.append(result)
      with self.test_session() as sess:
        values = sess.run(results)
        for value in values:
          print("expected = ", expected)
          print("actual = ", value)
          self.assertArrayNear(expected, value.flatten(), 1e-5)

  def testConv3D1x1x1Filter(self):
    expected_output = [
        30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0,
        204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0
    ]

    # These are equivalent to the Conv2D1x1 case.
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 3, 1, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 1, 3, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)
    self._VerifyValues(
        tensor_in_sizes=[1, 1, 2, 3, 3],
        filter_in_sizes=[1, 1, 1, 3, 3],
        stride=1,
        padding="VALID",
        expected=expected_output)

  # Expected values computed using scipy's correlate function.
  def testConv3D2x2x2Filter(self):
    expected_output = [
        19554., 19962., 20370., 22110., 22590., 23070., 34890., 35730., 36570.,
        37446., 38358., 39270., 50226., 51498., 52770., 52782., 54126., 55470.
    ]
    # expected_shape = [1, 3, 1, 2, 5]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],  # b, z, y, x, fin
        filter_in_sizes=[2, 2, 2, 3, 3],  # z, y, x, fin, fout
        stride=1,
        padding="VALID",
        expected=expected_output)

  def testConv3DStrides(self):
    expected_output = [
        102.,
        151.,
        172.,
        193.,
        214.,
        235.,
        142.,
        438.,
        592.,
        613.,
        634.,
        655.,
        676.,
        394.,
        774.,
        1033.,
        1054.,
        1075.,
        1096.,
        1117.,
        646.,
        1894.,
        2503.,
        2524.,
        2545.,
        2566.,
        2587.,
        1486.,
        2230.,
        2944.,
        2965.,
        2986.,
        3007.,
        3028.,
        1738.,
        2566.,
        3385.,
        3406.,
        3427.,
        3448.,
        3469.,
        1990.,
        3686.,
        4855.,
        4876.,
        4897.,
        4918.,
        4939.,
        2830.,
        4022.,
        5296.,
        5317.,
        5338.,
        5359.,
        5380.,
        3082.,
        4358.,
        5737.,
        5758.,
        5779.,
        5800.,
        5821.,
        3334.,
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 5, 8, 7, 1],
        filter_in_sizes=[1, 2, 3, 1, 1],
        stride=[2, 3, 1],  # different stride for each spatial dimension
        padding="SAME",
        expected=expected_output)

  def testConv3D2x2x2FilterStride2(self):
    expected_output = [19554., 19962., 20370., 50226., 51498., 52770.]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],
        filter_in_sizes=[2, 2, 2, 3, 3],
        stride=2,
        padding="VALID",
        expected=expected_output)

  def testConv3DStride3(self):
    expected_output = [
        36564., 38022., 39480., 37824., 39354., 40884., 39084., 40686., 42288.,
        46644., 48678., 50712., 47904., 50010., 52116., 49164., 51342., 53520.,
        107124., 112614., 118104., 108384., 113946., 119508., 109644., 115278.,
        120912., 117204., 123270., 129336., 118464., 124602., 130740., 119724.,
        125934., 132144.
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 6, 7, 8, 2],
        filter_in_sizes=[3, 2, 1, 2, 3],
        stride=3,
        padding="VALID",
        expected=expected_output)

  def testConv3D2x2x2FilterStride2Same(self):
    expected_output = [
        19554., 19962., 20370., 10452., 10710., 10968., 50226., 51498., 52770.,
        23844., 24534., 25224.
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 4, 2, 3, 3],
        filter_in_sizes=[2, 2, 2, 3, 3],
        stride=2,
        padding="SAME",
        expected=expected_output)

  def testKernelSmallerThanStride(self):
    expected_output = [1., 3., 7., 9., 19., 21., 25., 27.]
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1, 1],
        stride=2,
        padding="SAME",
        expected=expected_output)
    self._VerifyValues(
        tensor_in_sizes=[1, 3, 3, 3, 1],
        filter_in_sizes=[1, 1, 1, 1, 1],
        stride=2,
        padding="VALID",
        expected=expected_output)

    expected_output = [
        1484., 1592., 770., 2240., 2348., 1106., 1149., 1191., 539., 6776.,
        6884., 3122., 7532., 7640., 3458., 3207., 3249., 1421., 3005., 3035.,
        1225., 3215., 3245., 1309., 1013., 1022., 343.
    ]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 7, 1],
        filter_in_sizes=[2, 2, 2, 1, 1],
        stride=3,
        padding="SAME",
        expected=expected_output)

    expected_output = [1484., 1592., 2240., 2348., 6776., 6884., 7532., 7640.]
    self._VerifyValues(
        tensor_in_sizes=[1, 7, 7, 7, 1],
        filter_in_sizes=[2, 2, 2, 1, 1],
        stride=3,
        padding="VALID",
        expected=expected_output)

  def testKernelSizeMatchesInputSize(self):
    self._VerifyValues(
        tensor_in_sizes=[1, 2, 1, 2, 1],
        filter_in_sizes=[2, 1, 2, 1, 2],
        stride=1,
        padding="VALID",
        expected=[50, 60])

  def _ConstructAndTestGradientForConfig(
      self, batch, input_shape, filter_shape, in_depth, out_depth, stride,
      padding, test_input, data_format, use_gpu):

    input_planes, input_rows, input_cols = input_shape
    filter_planes, filter_rows, filter_cols = filter_shape

    input_shape = [batch, input_planes, input_rows, input_cols, in_depth]
    filter_shape = [
        filter_planes, filter_rows, filter_cols, in_depth, out_depth
    ]

    if isinstance(stride, collections.Iterable):
      strides = [1] + list(stride) + [1]
    else:
      strides = [1, stride, stride, stride, 1]

    if padding == "VALID":
      output_planes = int(
          math.ceil((input_planes - filter_planes + 1.0) / strides[1]))
      output_rows = int(
          math.ceil((input_rows - filter_rows + 1.0) / strides[2]))
      output_cols = int(
          math.ceil((input_cols - filter_cols + 1.0) / strides[3]))
    else:
      output_planes = int(math.ceil(float(input_planes) / strides[1]))
      output_rows = int(math.ceil(float(input_rows) / strides[2]))
      output_cols = int(math.ceil(float(input_cols) / strides[3]))
    output_shape = [batch, output_planes, output_rows, output_cols, out_depth]
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]

    if test.is_gpu_available() and use_gpu:
      data_type = dtypes.float32
      if test.is_gpu_available():
        tolerance = 4e-3
      else:
        # As of Aug 2016, higher tolerance is needed for some CPU architectures.
        # Runs on a single machine can also generate slightly different errors
        # because of multithreading.
        tolerance = 8e-3
    else:
      data_type = dtypes.float64
      tolerance = 1e-8
    with self.test_session(use_gpu=use_gpu):
      orig_input_tensor = constant_op.constant(
          input_data, shape=input_shape, dtype=data_type, name="input")
      filter_tensor = constant_op.constant(
          filter_data, shape=filter_shape, dtype=data_type, name="filter")

      if data_format == "NCDHW":
        input_tensor = test_util.NHWCToNCHW(orig_input_tensor)
        strides = test_util.NHWCToNCHW(strides)
      else:
        input_tensor = orig_input_tensor

      conv = nn_ops.conv3d(
          input_tensor, filter_tensor, strides, padding,
          data_format=data_format, name="conv")

      if data_format == "NCDHW":
        conv = test_util.NCHWToNHWC(conv)

      if test_input:
        err = gradient_checker.compute_gradient_error(orig_input_tensor,
                                                      input_shape,
                                                      conv, output_shape)
      else:
        err = gradient_checker.compute_gradient_error(filter_tensor,
                                                      filter_shape, conv,
                                                      output_shape)
    print("conv3d gradient error = ", err)
    self.assertLess(err, tolerance)

  def ConstructAndTestGradient(self, **kwargs):
    for data_format, use_gpu in GetTestConfigs():
      self._ConstructAndTestGradientForConfig(data_format=data_format,
                                              use_gpu=use_gpu, **kwargs)

  def testInputGradientValidPaddingStrideOne(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 5, 4),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=True)

  def testFilterGradientValidPaddingStrideOne(self):
    self.ConstructAndTestGradient(
        batch=4,
        input_shape=(4, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=False)

  def testInputGradientValidPaddingStrideTwo(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(6, 3, 5),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="VALID",
        test_input=True)

  def testFilterGradientValidPaddingStrideTwo(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(7, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="VALID",
        test_input=False)

  def testInputGradientValidPaddingStrideThree(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 7, 6),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="VALID",
        test_input=True)

  def testFilterGradientValidPaddingStrideThree(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(4, 4, 7),
        filter_shape=(4, 4, 4),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="VALID",
        test_input=False)

  def testInputGradientSamePaddingStrideOne(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 2, 2),
        filter_shape=(3, 2, 1),
        in_depth=2,
        out_depth=1,
        stride=1,
        padding="SAME",
        test_input=True)

  def testFilterGradientSamePaddingStrideOne(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(3, 6, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="SAME",
        test_input=False)

  def testInputGradientSamePaddingStrideTwo(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(6, 3, 4),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="SAME",
        test_input=True)

  def testFilterGradientSamePaddingStrideTwo(self):
    self.ConstructAndTestGradient(
        batch=4,
        input_shape=(7, 3, 5),
        filter_shape=(2, 2, 2),
        in_depth=2,
        out_depth=3,
        stride=2,
        padding="SAME",
        test_input=False)

  def testInputGradientSamePaddingStrideThree(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(9, 3, 6),
        filter_shape=(3, 3, 3),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="SAME",
        test_input=True)

  def testFilterGradientSamePaddingStrideThree(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(9, 4, 7),
        filter_shape=(4, 4, 4),
        in_depth=2,
        out_depth=3,
        stride=3,
        padding="SAME",
        test_input=False)

  def testInputGradientSamePaddingDifferentStrides(self):
    self.ConstructAndTestGradient(
        batch=1,
        input_shape=(5, 8, 7),
        filter_shape=(1, 2, 3),
        in_depth=2,
        out_depth=3,
        stride=[2, 3, 1],
        padding="SAME",
        test_input=True)

  def testFilterGradientKernelSizeMatchesInputSize(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(5, 4, 3),
        filter_shape=(5, 4, 3),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=False)

  def testInputGradientKernelSizeMatchesInputSize(self):
    self.ConstructAndTestGradient(
        batch=2,
        input_shape=(5, 4, 3),
        filter_shape=(5, 4, 3),
        in_depth=2,
        out_depth=3,
        stride=1,
        padding="VALID",
        test_input=True)

  def disabledtestFilterGradientSamePaddingDifferentStrides(self):
    self.ConstructAndTestGradient(
        batch=1,
        input_shape=(5, 8, 7),
        filter_shape=(1, 2, 3),
        in_depth=2,
        out_depth=3,
        stride=[2, 3, 1],
        padding="SAME",
        test_input=False)


if __name__ == "__main__":
  test.main()
