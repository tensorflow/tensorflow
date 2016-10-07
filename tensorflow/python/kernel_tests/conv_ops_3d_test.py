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

import math
import tensorflow as tf


class Conv3DTest(tf.test.TestCase):

  def _VerifyValues(
      self, tensor_in_sizes, filter_in_sizes, stride, padding, expected):
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
    with self.test_session(use_gpu=True) as sess:
      t1 = tf.constant(x1, shape=tensor_in_sizes)
      t2 = tf.constant(x2, shape=filter_in_sizes)
      conv = tf.nn.conv3d(t1,
                          t2, [1, stride, stride, stride, 1],
                          padding=padding)
      value = sess.run(conv)
    print("expected = ", expected)
    print("actual = ", value)
    self.assertArrayNear(expected, value.flatten(), 1e-5)

  def testConv3D1x1x1Filter(self):
    expected_output = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0,
                       138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0,
                       312.0]

    # These are equivalent to the Conv2D1x1 case.
    self._VerifyValues(tensor_in_sizes=[1, 2, 3, 1, 3],
                       filter_in_sizes=[1, 1, 1, 3, 3],
                       stride=1,
                       padding="VALID",
                       expected=expected_output)
    self._VerifyValues(tensor_in_sizes=[1, 2, 1, 3, 3],
                       filter_in_sizes=[1, 1, 1, 3, 3],
                       stride=1,
                       padding="VALID",
                       expected=expected_output)
    self._VerifyValues(tensor_in_sizes=[1, 1, 2, 3, 3],
                       filter_in_sizes=[1, 1, 1, 3, 3],
                       stride=1,
                       padding="VALID",
                       expected=expected_output)

  # Expected values computed using scipy's correlate function.
  def testConv3D2x2x2Filter(self):
    expected_output = [19554., 19962., 20370., 22110., 22590., 23070., 34890.,
                       35730., 36570., 37446., 38358., 39270., 50226., 51498.,
                       52770., 52782., 54126., 55470.]
    # expected_shape = [1, 3, 1, 2, 5]
    self._VerifyValues(tensor_in_sizes=[1, 4, 2, 3, 3],  # b, z, y, x, fin
                       filter_in_sizes=[2, 2, 2, 3, 3],  # z, y, x, fin, fout
                       stride=1, padding="VALID",
                       expected=expected_output)

  def testConv3D2x2x2FilterStride2(self):
    expected_output = [19554., 19962., 20370., 50226., 51498., 52770.]
    self._VerifyValues(tensor_in_sizes=[1, 4, 2, 3, 3],
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
    self._VerifyValues(tensor_in_sizes=[1, 6, 7, 8, 2],
                       filter_in_sizes=[3, 2, 1, 2, 3],
                       stride=3,
                       padding="VALID",
                       expected=expected_output)

  def testConv3D2x2x2FilterStride2Same(self):
    expected_output = [
        19554., 19962., 20370., 10452., 10710., 10968., 50226., 51498., 52770.,
        23844., 24534., 25224.
    ]
    self._VerifyValues(tensor_in_sizes=[1, 4, 2, 3, 3],
                       filter_in_sizes=[2, 2, 2, 3, 3],
                       stride=2,
                       padding="SAME",
                       expected=expected_output)

  def testKernelSmallerThanStride(self):
    expected_output = [1., 3., 7., 9., 19., 21., 25., 27.]
    self._VerifyValues(tensor_in_sizes=[1, 3, 3, 3, 1],
                       filter_in_sizes=[1, 1, 1, 1, 1],
                       stride=2,
                       padding="SAME",
                       expected=expected_output)
    self._VerifyValues(tensor_in_sizes=[1, 3, 3, 3, 1],
                       filter_in_sizes=[1, 1, 1, 1, 1],
                       stride=2,
                       padding="VALID",
                       expected=expected_output)

    expected_output = [1484., 1592., 770.,
                       2240., 2348., 1106.,
                       1149., 1191., 539.,

                       6776., 6884., 3122.,
                       7532., 7640., 3458.,
                       3207., 3249., 1421.,

                       3005., 3035., 1225.,
                       3215., 3245., 1309.,
                       1013., 1022., 343.]
    self._VerifyValues(tensor_in_sizes=[1, 7, 7, 7, 1],
                       filter_in_sizes=[2, 2, 2, 1, 1],
                       stride=3,
                       padding="SAME",
                       expected=expected_output)

    expected_output = [1484., 1592.,
                       2240., 2348.,

                       6776., 6884.,
                       7532., 7640.]
    self._VerifyValues(tensor_in_sizes=[1, 7, 7, 7, 1],
                       filter_in_sizes=[2, 2, 2, 1, 1],
                       stride=3,
                       padding="VALID",
                       expected=expected_output)

  def ConstructAndTestGradient(self, batch, input_planes, input_rows,
                               input_cols, filter_planes, filter_rows,
                               filter_cols, in_depth, out_depth, stride,
                               padding, test_input):
    input_shape = [batch, input_planes, input_rows, input_cols, in_depth]
    filter_shape = [filter_planes, filter_rows, filter_cols, in_depth,
                    out_depth]
    if padding == "VALID":
      output_planes = int(math.ceil((input_planes - filter_planes + 1.0) /
                                    stride))
      output_rows = int(math.ceil((input_rows - filter_rows + 1.0) / stride))
      output_cols = int(math.ceil((input_cols - filter_cols + 1.0) / stride))
    else:
      output_planes = int(math.ceil(float(input_planes) / stride))
      output_rows = int(math.ceil(float(input_rows) / stride))
      output_cols = int(math.ceil(float(input_cols) / stride))
    output_shape = [batch, output_planes, output_rows, output_cols, out_depth]
    input_size = 1
    for x in input_shape:
      input_size *= x
    filter_size = 1
    for x in filter_shape:
      filter_size *= x
    input_data = [x * 1.0 / input_size for x in range(0, input_size)]
    filter_data = [x * 1.0 / filter_size for x in range(0, filter_size)]
    if tf.test.is_gpu_available():
      data_type = tf.float32
      if tf.test.is_gpu_available():
        tolerance = 4e-3
      else:
        # As of Aug 2016, higher tolerance is needed for some CPU architectures.
        # Runs on a single machine can also generate slightly different errors
        # because of multithreading.
        tolerance = 8e-3
    else:
      data_type = tf.float64
      tolerance = 1e-8
    with self.test_session(use_gpu=True):
      input_tensor = tf.constant(input_data,
                                 shape=input_shape,
                                 dtype=data_type,
                                 name="input")
      filter_tensor = tf.constant(filter_data,
                                  shape=filter_shape,
                                  dtype=data_type,
                                  name="filter")
      conv = tf.nn.conv3d(input_tensor,
                          filter_tensor, [1, stride, stride, stride, 1],
                          padding,
                          name="conv")

      if test_input:
        err = tf.test.compute_gradient_error(input_tensor, input_shape, conv,
                                             output_shape)
      else:
        err = tf.test.compute_gradient_error(filter_tensor, filter_shape, conv,
                                             output_shape)
    print("conv3d gradient error = ", err)
    self.assertLess(err, tolerance)

  def testInputGradientValidPaddingStrideOne(self):
    self.ConstructAndTestGradient(batch=2,
                                  input_planes=3,
                                  input_rows=5,
                                  input_cols=4,
                                  filter_planes=3,
                                  filter_rows=3,
                                  filter_cols=3,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=1,
                                  padding="VALID",
                                  test_input=True)

  def testFilterGradientValidPaddingStrideOne(self):
    self.ConstructAndTestGradient(batch=4,
                                  input_planes=4,
                                  input_rows=6,
                                  input_cols=5,
                                  filter_planes=2,
                                  filter_rows=2,
                                  filter_cols=2,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=1,
                                  padding="VALID",
                                  test_input=False)

  def testInputGradientValidPaddingStrideTwo(self):
    self.ConstructAndTestGradient(batch=2,
                                  input_planes=6,
                                  input_rows=3,
                                  input_cols=5,
                                  filter_planes=3,
                                  filter_rows=3,
                                  filter_cols=3,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=2,
                                  padding="VALID",
                                  test_input=True)

  def testFilterGradientValidPaddingStrideTwo(self):
    self.ConstructAndTestGradient(batch=2,
                                  input_planes=7,
                                  input_rows=6,
                                  input_cols=5,
                                  filter_planes=2,
                                  filter_rows=2,
                                  filter_cols=2,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=2,
                                  padding="VALID",
                                  test_input=False)

  def testInputGradientValidPaddingStrideThree(self):
    self.ConstructAndTestGradient(batch=2,
                                  input_planes=3,
                                  input_rows=7,
                                  input_cols=6,
                                  filter_planes=3,
                                  filter_rows=3,
                                  filter_cols=3,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=3,
                                  padding="VALID",
                                  test_input=True)

  def testFilterGradientValidPaddingStrideThree(self):
    self.ConstructAndTestGradient(batch=2,
                                  input_planes=4,
                                  input_rows=4,
                                  input_cols=7,
                                  filter_planes=4,
                                  filter_rows=4,
                                  filter_cols=4,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=3,
                                  padding="VALID",
                                  test_input=False)

  def testInputGradientSamePaddingStrideOne(self):
    self.ConstructAndTestGradient(batch=2,
                                  input_planes=3,
                                  input_rows=2,
                                  input_cols=2,
                                  filter_planes=3,
                                  filter_rows=2,
                                  filter_cols=1,
                                  in_depth=2,
                                  out_depth=1,
                                  stride=1,
                                  padding="SAME",
                                  test_input=True)

  def testFilterGradientSamePaddingStrideOne(self):
    self.ConstructAndTestGradient(batch=2,
                                  input_planes=3,
                                  input_rows=6,
                                  input_cols=5,
                                  filter_planes=2,
                                  filter_rows=2,
                                  filter_cols=2,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=1,
                                  padding="SAME",
                                  test_input=False)

  def testInputGradientSamePaddingStrideTwo(self):
    self.ConstructAndTestGradient(batch=2,
                                  input_planes=6,
                                  input_rows=3,
                                  input_cols=4,
                                  filter_planes=3,
                                  filter_rows=3,
                                  filter_cols=3,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=2,
                                  padding="SAME",
                                  test_input=True)

  def testFilterGradientSamePaddingStrideTwo(self):
    self.ConstructAndTestGradient(batch=4,
                                  input_planes=7,
                                  input_rows=3,
                                  input_cols=5,
                                  filter_planes=2,
                                  filter_rows=2,
                                  filter_cols=2,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=2,
                                  padding="SAME",
                                  test_input=False)

  def testInputGradientSamePaddingStrideThree(self):
    self.ConstructAndTestGradient(batch=2,
                                  input_planes=9,
                                  input_rows=3,
                                  input_cols=6,
                                  filter_planes=3,
                                  filter_rows=3,
                                  filter_cols=3,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=3,
                                  padding="SAME",
                                  test_input=True)

  def testFilterGradientSamePaddingStrideThree(self):
    self.ConstructAndTestGradient(batch=2,
                                  input_planes=9,
                                  input_rows=4,
                                  input_cols=7,
                                  filter_planes=4,
                                  filter_rows=4,
                                  filter_cols=4,
                                  in_depth=2,
                                  out_depth=3,
                                  stride=3,
                                  padding="SAME",
                                  test_input=False)


if __name__ == "__main__":
  tf.test.main()
