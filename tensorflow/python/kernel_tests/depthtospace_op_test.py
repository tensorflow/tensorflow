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

"""Functional tests for DepthToSpace op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class DepthToSpaceTest(tf.test.TestCase):

  def _testOne(self, inputs, block_size, outputs):
    with self.test_session():
      x_tf = tf.depth_to_space(tf.to_float(inputs), block_size)
      self.assertAllEqual(x_tf.eval(), outputs)

  def testBasic(self):
    x_np = [[[[1, 2, 3, 4]]]]
    block_size = 2
    x_out = [[[[1], [2]], [[3], [4]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input dimensions. To make sure elements are
  # correctly ordered spatially.
  def testBlockSize2(self):
    x_np = [[[[1, 2, 3, 4],
              [5, 6, 7, 8]],
             [[9, 10, 11, 12],
              [13, 14, 15, 16]]]]
    block_size = 2
    x_out = [[[[1], [2], [5], [6]],
              [[3], [4], [7], [8]],
              [[9], [10], [13], [14]],
              [[11], [12], [15], [16]]]]
    self._testOne(x_np, block_size, x_out)

  def testBlockSize2Batch10(self):
    block_size = 2
    def batch_input_elt(i):
      return [[[1 * i, 2 * i, 3 * i, 4 * i],
               [5 * i, 6 * i, 7 * i, 8 * i]],
              [[9 * i, 10 * i, 11 * i, 12 * i],
               [13 * i, 14 * i, 15 * i, 16 * i]]]
    def batch_output_elt(i):
      return [[[1 * i], [2 * i], [5 * i], [6 * i]],
              [[3 * i], [4 * i], [7 * i], [8 * i]],
              [[9 * i], [10 * i], [13 * i], [14 * i]],
              [[11 * i], [12 * i], [15 * i], [16 * i]]]
    batch_size = 10
    x_np = [batch_input_elt(i) for i in range(batch_size)]
    x_out = [batch_output_elt(i) for i in range(batch_size)]
    self._testOne(x_np, block_size, x_out)

  # Tests for different width and height.
  def testNonSquare(self):
    x_np = [[[[1, 10, 2, 20, 3, 30, 4, 40]],
             [[5, 50, 6, 60, 7, 70, 8, 80]],
             [[9, 90, 10, 100, 11, 110, 12, 120]]]]
    block_size = 2
    x_out = [[[[1, 10], [2, 20]],
              [[3, 30], [4, 40]],
              [[5, 50], [6, 60]],
              [[7, 70], [8, 80]],
              [[9, 90], [10, 100]],
              [[11, 110], [12, 120]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input dimensions. To make sure elements are
  # correctly ordered spatially.
  def testBlockSize4FlatInput(self):
    x_np = [[[[1, 2, 5, 6, 3, 4, 7, 8, 9, 10, 13, 14, 11, 12, 15, 16]]]]
    block_size = 4
    x_out = [[[[1], [2], [5], [6]],
              [[3], [4], [7], [8]],
              [[9], [10], [13], [14]],
              [[11], [12], [15], [16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths.
  # To make sure elements are properly interleaved in depth.
  def testDepthInterleaved(self):
    x_np = [[[[1, 10, 2, 20, 3, 30, 4, 40]]]]
    block_size = 2
    x_out = [[[[1, 10], [2, 20]],
              [[3, 30], [4, 40]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths. Here an odd depth.
  # To make sure elements are properly interleaved in depth.
  def testDepthInterleavedDepth3(self):
    x_np = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
    block_size = 2
    x_out = [[[[1, 2, 3], [4, 5, 6]],
              [[7, 8, 9], [10, 11, 12]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input depths.
  # To make sure elements are properly interleaved in depth.
  def testDepthInterleavedLarger(self):
    x_np = [[[[1, 10, 2, 20, 3, 30, 4, 40],
              [5, 50, 6, 60, 7, 70, 8, 80]],
             [[9, 90, 10, 100, 11, 110, 12, 120],
              [13, 130, 14, 140, 15, 150, 16, 160]]]]
    block_size = 2
    x_out = [[[[1, 10], [2, 20], [5, 50], [6, 60]],
              [[3, 30], [4, 40], [7, 70], [8, 80]],
              [[9, 90], [10, 100], [13, 130], [14, 140]],
              [[11, 110], [12, 120], [15, 150], [16, 160]]]]
    self._testOne(x_np, block_size, x_out)

  # Error handling:

  # Tests for a block larger for the depth. In this case should raise an
  # exception.
  def testBlockSizeTooLarge(self):
    x_np = [[[[1, 2, 3, 4],
              [5, 6, 7, 8]],
             [[9, 10, 11, 12],
              [13, 14, 15, 16]]]]
    block_size = 4
    # Raise an exception, since th depth is only 4 and needs to be
    # divisible by 16.
    with self.assertRaises(IndexError):
      out_tf = tf.depth_to_space(x_np, block_size)
      out_tf.eval()

  # Test when the block size is 0.
  def testBlockSize0(self):
    x_np = [[[[1], [2]],
             [[3], [4]]]]
    block_size = 0
    with self.assertRaises(ValueError):
      out_tf = tf.depth_to_space(x_np, block_size)
      out_tf.eval()

  # Test when the block size is 1. The block size should be > 1.
  def testBlockSizeOne(self):
    x_np = [[[[1, 1, 1, 1],
              [2, 2, 2, 2]],
             [[3, 3, 3, 3],
              [4, 4, 4, 4]]]]
    block_size = 1
    with self.assertRaises(ValueError):
      out_tf = tf.depth_to_space(x_np, block_size)
      out_tf.eval()

  def testBlockSizeLargerThanInput(self):
    # The block size is too large for this input.
    x_np = [[[[1], [2]],
             [[3], [4]]]]
    block_size = 10
    with self.assertRaises(IndexError):
      out_tf = tf.space_to_depth(x_np, block_size)
      out_tf.eval()

  def testBlockSizeNotDivisibleDepth(self):
    # The depth is not divisible by the square of the block size.
    x_np = [[[[1, 1, 1, 1],
              [2, 2, 2, 2]],
             [[3, 3, 3, 3],
              [4, 4, 4, 4]]]]
    block_size = 3
    with self.assertRaises(IndexError):
      _ = tf.space_to_depth(x_np, block_size)

  def testUnknownShape(self):
    t = tf.depth_to_space(tf.placeholder(tf.float32), block_size=4)
    self.assertEqual(4, t.get_shape().ndims)


class DepthToSpaceGradientTest(tf.test.TestCase):

  # Check the gradients.
  def _checkGrad(self, x, block_size):
    assert 4 == x.ndim
    with self.test_session():
      tf_x = tf.convert_to_tensor(x)
      tf_y = tf.depth_to_space(tf_x, block_size)
      epsilon = 1e-2
      ((x_jacob_t, x_jacob_n)) = tf.test.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a gradient for depth_to_space of x which is a four dimensional
  # tensor of shape [b, h, w, d * block_size * block_size].
  def _compare(self, b, h, w, d, block_size):
    block_size_sq = block_size * block_size
    x = np.random.normal(
        0, 1, b * h * w * d * block_size_sq).astype(np.float32).reshape(
            [b, h, w, d * block_size_sq])

    self._checkGrad(x, block_size)

  # Don't use very large numbers as dimensions here, as the result is tensor
  # with cartesian product of the dimensions.
  def testSmall(self):
    block_size = 2
    self._compare(3, 2, 5, 3, block_size)

  def testSmall2(self):
    block_size = 3
    self._compare(1, 2, 3, 2, block_size)


if __name__ == "__main__":
  tf.test.main()
