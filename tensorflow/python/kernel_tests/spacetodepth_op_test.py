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
"""Functional tests for SpacetoDepth op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class SpaceToDepthTest(test.TestCase):

  def _testOne(self, inputs, block_size, outputs):
    input_nhwc = math_ops.to_float(inputs)
    with self.test_session(use_gpu=False):
      # test NHWC (default) on CPU
      x_tf = array_ops.space_to_depth(input_nhwc, block_size)
      self.assertAllEqual(x_tf.eval(), outputs)
    if test.is_gpu_available():
      with self.test_session(use_gpu=True):
        # test NHWC (default) on GPU
        x_tf = array_ops.space_to_depth(input_nhwc, block_size)
        self.assertAllEqual(x_tf.eval(), outputs)
        # test NCHW on GPU
        input_nchw = test_util.NHWCToNCHW(input_nhwc)
        output_nchw = array_ops.space_to_depth(
            input_nchw, block_size, data_format="NCHW")
        output_nhwc = test_util.NCHWToNHWC(output_nchw)
        self.assertAllEqual(output_nhwc.eval(), outputs)

  def testBasic(self):
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 2
    x_out = [[[[1, 2, 3, 4]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input dimensions. To make sure elements are
  # correctly ordered spatially.
  def testLargerInput2x2(self):
    x_np = [[[[1], [2], [5], [6]], [[3], [4], [7], [8]],
             [[9], [10], [13], [14]], [[11], [12], [15], [16]]]]
    block_size = 2
    x_out = [[[[1, 2, 3, 4], [5, 6, 7, 8]], [[9, 10, 11, 12],
                                             [13, 14, 15, 16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input dimensions. To make sure elements are
  # correctly ordered in depth. Here, larger block size.
  def testLargerInput4x4(self):
    x_np = [[[[1], [2], [5], [6]], [[3], [4], [7], [8]],
             [[9], [10], [13], [14]], [[11], [12], [15], [16]]]]
    block_size = 4
    x_out = [[[[1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths.
  # To make sure elements are properly interleaved in depth.
  def testDepthInterleaved(self):
    x_np = [[[[1, 10], [2, 20]], [[3, 30], [4, 40]]]]
    block_size = 2
    x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths. Here an odd depth.
  # To make sure elements are properly interleaved in depth.
  def testDepthInterleavedDepth3(self):
    x_np = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
    block_size = 2
    x_out = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input dimensions AND for larger input depths.
  # To make sure elements are properly interleaved in depth and ordered
  # spatially.
  def testDepthInterleavedLarge(self):
    x_np = [[[[1, 10], [2, 20], [5, 50], [6, 60]],
             [[3, 30], [4, 40], [7, 70], [8, 80]],
             [[9, 90], [10, 100], [13, 130], [14, 140]],
             [[11, 110], [12, 120], [15, 150], [16, 160]]]]
    block_size = 2
    x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40], [5, 50, 6, 60, 7, 70, 8, 80]],
              [[9, 90, 10, 100, 11, 110, 12, 120],
               [13, 130, 14, 140, 15, 150, 16, 160]]]]
    self._testOne(x_np, block_size, x_out)

  def testBlockSize2Batch10(self):
    block_size = 2

    def batch_input_elt(i):
      return [[[1 * i], [2 * i], [5 * i], [6 * i]],
              [[3 * i], [4 * i], [7 * i], [8 * i]],
              [[9 * i], [10 * i], [13 * i], [14 * i]],
              [[11 * i], [12 * i], [15 * i], [16 * i]]]

    def batch_output_elt(i):
      return [[[1 * i, 2 * i, 3 * i, 4 * i], [5 * i, 6 * i, 7 * i, 8 * i]],
              [[9 * i, 10 * i, 11 * i, 12 * i],
               [13 * i, 14 * i, 15 * i, 16 * i]]]

    batch_size = 10
    x_np = [batch_input_elt(i) for i in range(batch_size)]
    x_out = [batch_output_elt(i) for i in range(batch_size)]
    self._testOne(x_np, block_size, x_out)

  # Tests for different width and height.
  def testNonSquare(self):
    x_np = [[[[1, 10], [2, 20]], [[3, 30], [4, 40]], [[5, 50], [6, 60]],
             [[7, 70], [8, 80]], [[9, 90], [10, 100]], [[11, 110], [12, 120]]]]
    block_size = 2
    x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40]], [[5, 50, 6, 60, 7, 70, 8, 80]],
              [[9, 90, 10, 100, 11, 110, 12, 120]]]]
    self._testOne(x_np, block_size, x_out)

  # Error handling:

  def testInputWrongDimMissingDepth(self):
    # The input is missing the last dimension ("depth")
    x_np = [[[1, 2], [3, 4]]]
    block_size = 2
    with self.assertRaises(ValueError):
      out_tf = array_ops.space_to_depth(x_np, block_size)
      out_tf.eval()

  def testInputWrongDimMissingBatch(self):
    # The input is missing the first dimension ("batch")
    x_np = [[[1], [2]], [[3], [4]]]
    block_size = 2
    with self.assertRaises(ValueError):
      _ = array_ops.space_to_depth(x_np, block_size)

  def testBlockSize0(self):
    # The block size is 0.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 0
    with self.assertRaises(ValueError):
      out_tf = array_ops.space_to_depth(x_np, block_size)
      out_tf.eval()

  def testBlockSizeOne(self):
    # The block size is 1. The block size needs to be > 1.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 1
    with self.assertRaises(ValueError):
      out_tf = array_ops.space_to_depth(x_np, block_size)
      out_tf.eval()

  def testBlockSizeLarger(self):
    # The block size is too large for this input.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 10
    with self.assertRaises(ValueError):
      out_tf = array_ops.space_to_depth(x_np, block_size)
      out_tf.eval()

  def testBlockSizeNotDivisibleWidth(self):
    # The block size divides width but not height.
    x_np = [[[[1], [2], [3]], [[3], [4], [7]]]]
    block_size = 3
    with self.assertRaises(ValueError):
      _ = array_ops.space_to_depth(x_np, block_size)

  def testBlockSizeNotDivisibleHeight(self):
    # The block size divides height but not width.
    x_np = [[[[1], [2]], [[3], [4]], [[5], [6]]]]
    block_size = 3
    with self.assertRaises(ValueError):
      _ = array_ops.space_to_depth(x_np, block_size)

  def testBlockSizeNotDivisibleBoth(self):
    # The block size does not divide neither width or height.
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 3
    with self.assertRaises(ValueError):
      _ = array_ops.space_to_depth(x_np, block_size)

  def testUnknownShape(self):
    t = array_ops.space_to_depth(
        array_ops.placeholder(dtypes.float32), block_size=4)
    self.assertEqual(4, t.get_shape().ndims)

  def spaceToDepthUsingTranspose(self, tensor, block_size, data_format):
    block_size_sq = block_size * block_size
    if data_format == "NHWC":
      b, ih, iw, ic = tensor.shape.as_list()
      assert ih % block_size == 0, (ih, block_size)
      assert iw % block_size == 0, (iw, block_size)
      ow, oh, oc = iw // block_size, ih // block_size, ic * block_size_sq
      tensor = array_ops.reshape(tensor,
                                 [b, oh, block_size, ow, block_size, ic])
      tensor = array_ops.transpose(tensor, [0, 1, 3, 2, 4, 5])
      tensor = array_ops.reshape(tensor, [b, oh, ow, oc])
    elif data_format == "NCHW":
      b, ic, ih, iw = tensor.shape.as_list()
      assert ih % block_size == 0, (ih, block_size)
      assert iw % block_size == 0, (iw, block_size)
      ow, oh, oc = iw // block_size, ih // block_size, ic * block_size_sq
      tensor = array_ops.reshape(tensor,
                                 [b, ic, oh, block_size, ow, block_size])
      tensor = array_ops.transpose(tensor, [0, 3, 5, 1, 2, 4])
      tensor = array_ops.reshape(tensor, [b, oc, oh, ow])
    return tensor

  def compareToTranspose(self, data_format, use_gpu):
    if use_gpu and not test.is_gpu_available():
      print("gpu not available")
      return

    dtype = dtypes.float32
    batch_size = 3
    height = 4
    width = 6
    channels = 4
    block_size = 2

    if data_format == "NHWC":
      input_shape = [batch_size, height, width, channels]
    elif data_format == "NCHW":
      input_shape = [batch_size, channels, height, width]
    else:
      print("unsupported format")

    # Initialize the input tensor with ascending whole numbers.
    total_size = 1
    for dim_size in input_shape:
      total_size *= dim_size
    x = [f for f in range(total_size)]
    inputs = constant_op.constant(x, shape=input_shape, dtype=dtype)

    expected = self.spaceToDepthUsingTranspose(inputs, block_size, data_format)
    actual = array_ops.space_to_depth(
        inputs, block_size, data_format=data_format)

    with self.test_session(use_gpu=use_gpu) as sess:
      actual_vals, expected_vals = sess.run([actual, expected])
      self.assertTrue(np.array_equal(actual_vals, expected_vals))

  def testAgainstTranspose(self):
    self.compareToTranspose("NHWC", False)
    self.compareToTranspose("NHWC", True)
    self.compareToTranspose("NCHW", True)


class SpaceToDepthGradientTest(test.TestCase):

  # Check the gradients.
  def _checkGrad(self, x, block_size):
    assert 4 == x.ndim
    with self.test_session(use_gpu=True):
      tf_x = ops.convert_to_tensor(x)
      tf_y = array_ops.space_to_depth(tf_x, block_size)
      epsilon = 1e-2
      ((x_jacob_t, x_jacob_n)) = gradient_checker.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a gradient for space_to_depth of x which is a four dimensional
  # tensor of shape [b, h * block_size, w * block_size, d].
  def _compare(self, b, h, w, d, block_size):
    block_size_sq = block_size * block_size
    x = np.random.normal(0, 1, b * h * w * d *
                         block_size_sq).astype(np.float32).reshape(
                             [b, h * block_size, w * block_size, d])

    self._checkGrad(x, block_size)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  def testSmall(self):
    block_size = 2
    self._compare(1, 2, 3, 5, block_size)

  def testSmall2(self):
    block_size = 2
    self._compare(2, 4, 3, 2, block_size)


if __name__ == "__main__":
  test.main()
