# Copyright 2015 Google Inc. All Rights Reserved.
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
import tensorflow as tf


class SpaceToDepthTest(tf.test.TestCase):

  def _testOne(self, inputs, block_size, outputs):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = tf.space_to_depth(tf.to_float(inputs), block_size)
        self.assertAllEqual(x_tf.eval(), outputs)

  def testBasic(self):
    x_np = [[[[1], [2]],
             [[3], [4]]]]
    block_size = 2
    x_out = [[[[1, 2, 3, 4]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input dimensions. To make sure elements are
  # correctly ordered spatially.
  def testLargerInput2x2(self):
    x_np = [[[[1], [2], [5], [6]],
             [[3], [4], [7], [8]],
             [[9], [10], [13], [14]],
             [[11], [12], [15], [16]]]]
    block_size = 2
    x_out = [[[[1, 2, 3, 4],
               [5, 6, 7, 8]],
              [[9, 10, 11, 12],
               [13, 14, 15, 16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input dimensions. To make sure elements are
  # correctly ordered in depth. Here, larger block size.
  def testLargerInput4x4(self):
    x_np = [[[[1], [2], [5], [6]],
             [[3], [4], [7], [8]],
             [[9], [10], [13], [14]],
             [[11], [12], [15], [16]]]]
    block_size = 4
    x_out = [[[[1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths.
  # To make sure elements are properly interleaved in depth.
  def testDepthInterleaved(self):
    x_np = [[[[1, 10], [2, 20]],
             [[3, 30], [4, 40]]]]
    block_size = 2
    x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths. Here an odd depth.
  # To make sure elements are properly interleaved in depth.
  def testDepthInterleavedDepth3(self):
    x_np = [[[[1, 2, 3], [4, 5, 6]],
             [[7, 8, 9], [10, 11, 12]]]]
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
    x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40],
               [5, 50, 6, 60, 7, 70, 8, 80]],
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
      return [[[1 * i, 2 * i, 3 * i, 4 * i],
               [5 * i, 6 * i, 7 * i, 8 * i]],
              [[9 * i, 10 * i, 11 * i, 12 * i],
               [13 * i, 14 * i, 15 * i, 16 * i]]]
    batch_size = 10
    x_np = [batch_input_elt(i) for i in range(batch_size)]
    x_out = [batch_output_elt(i) for i in range(batch_size)]
    self._testOne(x_np, block_size, x_out)

  # Tests for different width and height.
  def testNonSquare(self):
    x_np = [[[[1, 10], [2, 20]],
             [[3, 30], [4, 40]],
             [[5, 50], [6, 60]],
             [[7, 70], [8, 80]],
             [[9, 90], [10, 100]],
             [[11, 110], [12, 120]]]]
    block_size = 2
    x_out = [[[[1, 10, 2, 20, 3, 30, 4, 40]],
              [[5, 50, 6, 60, 7, 70, 8, 80]],
              [[9, 90, 10, 100, 11, 110, 12, 120]]]]
    self._testOne(x_np, block_size, x_out)

  # Error handling:

  def testInputWrongDimMissingDepth(self):
    # The input is missing the last dimension ("depth")
    x_np = [[[1, 2],
             [3, 4]]]
    block_size = 2
    with self.assertRaises(ValueError):
      out_tf = tf.space_to_depth(x_np, block_size)
      out_tf.eval()

  def testInputWrongDimMissingBatch(self):
    # The input is missing the first dimension ("batch")
    x_np = [[[1], [2]],
            [[3], [4]]]
    block_size = 2
    with self.assertRaises(ValueError):
      _ = tf.space_to_depth(x_np, block_size)

  def testBlockSize0(self):
    # The block size is 0.
    x_np = [[[[1], [2]],
             [[3], [4]]]]
    block_size = 0
    with self.assertRaises(ValueError):
      out_tf = tf.space_to_depth(x_np, block_size)
      out_tf.eval()

  def testBlockSizeOne(self):
    # The block size is 1. The block size needs to be > 1.
    x_np = [[[[1], [2]],
             [[3], [4]]]]
    block_size = 1
    with self.assertRaises(ValueError):
      out_tf = tf.space_to_depth(x_np, block_size)
      out_tf.eval()

  def testBlockSizeLarger(self):
    # The block size is too large for this input.
    x_np = [[[[1], [2]],
             [[3], [4]]]]
    block_size = 10
    with self.assertRaises(IndexError):
      out_tf = tf.space_to_depth(x_np, block_size)
      out_tf.eval()

  def testBlockSizeNotDivisibleWidth(self):
    # The block size divides width but not height.
    x_np = [[[[1], [2], [3]],
             [[3], [4], [7]]]]
    block_size = 3
    with self.assertRaises(IndexError):
      _ = tf.space_to_depth(x_np, block_size)

  def testBlockSizeNotDivisibleHeight(self):
    # The block size divides height but not width.
    x_np = [[[[1], [2]],
             [[3], [4]],
             [[5], [6]]]]
    block_size = 3
    with self.assertRaises(IndexError):
      _ = tf.space_to_depth(x_np, block_size)

  def testBlockSizeNotDivisibleBoth(self):
    # The block size does not divide neither width or height.
    x_np = [[[[1], [2]],
             [[3], [4]]]]
    block_size = 3
    with self.assertRaises(IndexError):
      _ = tf.space_to_depth(x_np, block_size)


class SpaceToDepthGradientTest(tf.test.TestCase):

  # Check the gradients.
  def _checkGrad(self, x, block_size):
    assert 4 == x.ndim
    with self.test_session():
      tf_x = tf.convert_to_tensor(x)
      tf_y = tf.space_to_depth(tf_x, block_size)
      epsilon = 1e-2
      ((x_jacob_t, x_jacob_n)) = tf.test.compute_gradient(
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
    x = np.random.normal(
        0, 1, b * h * w * d * block_size_sq).astype(np.float32).reshape(
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
  tf.test.main()
