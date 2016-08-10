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

"""Functional tests for SpaceToBatch and BatchToSpace ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class SpaceToBatchTest(tf.test.TestCase):
  """Tests input-output pairs for the SpaceToBatch and BatchToSpace ops."""

  def _testPad(self, inputs, paddings, block_size, outputs):
    with self.test_session():
      # outputs = space_to_batch(inputs)
      x_tf = tf.space_to_batch(
          tf.to_float(inputs), paddings, block_size=block_size)
      self.assertAllEqual(x_tf.eval(), outputs)
      # inputs = batch_to_space(outputs)
      x_tf = tf.batch_to_space(
          tf.to_float(outputs), paddings, block_size=block_size)
      self.assertAllEqual(x_tf.eval(), inputs)

  def _testOne(self, inputs, block_size, outputs):
    paddings = np.zeros((2, 2), dtype=np.int32)
    self._testPad(inputs, paddings, block_size, outputs)

  # [1, 2, 2, 1] <-> [4, 1, 1, 1]
  def testSmallInput2x2(self):
    x_np = [[[[1], [2]], [[3], [4]]]]
    block_size = 2
    x_out = [[[[1]]], [[[2]]], [[[3]]], [[[4]]]]
    self._testOne(x_np, block_size, x_out)

  # [1, 2, 2, 1] <-> [1, 3, 3, 1] (padding) <-> [9, 1, 1, 1]
  def testSmallInput2x2Pad1x0(self):
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.array([[1, 0], [1, 0]], dtype=np.int32)
    block_size = 3
    x_out = [[[[0]]], [[[0]]], [[[0]]],
             [[[0]]], [[[1]]], [[[2]]],
             [[[0]]], [[[3]]], [[[4]]]]
    self._testPad(x_np, paddings, block_size, x_out)

  # Test with depth larger than 1.
  # [1, 2, 2, 3] <-> [4, 1, 1, 3]
  def testDepthInput2x2(self):
    x_np = [[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]]
    block_size = 2
    x_out = [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
    self._testOne(x_np, block_size, x_out)

  # Test for larger input dimensions.
  # [1, 4, 4, 1] <-> [4, 2, 2, 1]
  def testLargerInput2x2(self):
    x_np = [[[[1], [2], [3], [4]],
             [[5], [6], [7], [8]],
             [[9], [10], [11], [12]],
             [[13], [14], [15], [16]]]]
    block_size = 2
    x_out = [[[[1], [3]], [[9], [11]]],
             [[[2], [4]], [[10], [12]]],
             [[[5], [7]], [[13], [15]]],
             [[[6], [8]], [[14], [16]]]]
    self._testOne(x_np, block_size, x_out)

  # Test with batch larger than 1.
  # [2, 2, 4, 1] <-> [8, 1, 2, 1]
  def testBatchInput2x2(self):
    x_np = [[[[1], [2], [3], [4]],
             [[5], [6], [7], [8]]],
            [[[9], [10], [11], [12]],
             [[13], [14], [15], [16]]]]
    block_size = 2
    x_out = [[[[1], [3]]], [[[9], [11]]], [[[2], [4]]], [[[10], [12]]],
             [[[5], [7]]], [[[13], [15]]], [[[6], [8]]], [[[14], [16]]]]
    self._testOne(x_np, block_size, x_out)

  # Tests for larger input spatial dimensions AND batch larger than 1, to ensure
  # that elements are correctly laid out spatially and properly interleaved
  # along the batch dimension.
  # [2, 4, 4, 1] <-> [8, 2, 2, 1]
  def testLargerInputBatch2x2(self):
    x_np = [[[[1], [2], [3], [4]],
             [[5], [6], [7], [8]],
             [[9], [10], [11], [12]],
             [[13], [14], [15], [16]]],
            [[[17], [18], [19], [20]],
             [[21], [22], [23], [24]],
             [[25], [26], [27], [28]],
             [[29], [30], [31], [32]]]]
    x_out = [[[[1], [3]], [[9], [11]]],
             [[[17], [19]], [[25], [27]]],
             [[[2], [4]], [[10], [12]]],
             [[[18], [20]], [[26], [28]]],
             [[[5], [7]], [[13], [15]]],
             [[[21], [23]], [[29], [31]]],
             [[[6], [8]], [[14], [16]]],
             [[[22], [24]], [[30], [32]]]]
    block_size = 2
    self._testOne(x_np, block_size, x_out)


class SpaceToBatchSpaceToDepth(tf.test.TestCase):

  # Verifies that: space_to_batch(x) = transpose(space_to_depth(transpose(x)))
  def testSpaceToDepthTranspose(self):
    x = np.arange(5 * 10 * 16 * 7, dtype=np.float32).reshape([5, 10, 16, 7])
    block_size = 2
    paddings = np.zeros((2, 2), dtype=np.int32)
    y1 = tf.space_to_batch(x, paddings, block_size=block_size)
    y2 = tf.transpose(
        tf.space_to_depth(
            tf.transpose(x, [3, 1, 2, 0]), block_size=block_size),
        [3, 1, 2, 0])
    with self.test_session():
      self.assertAllEqual(y1.eval(), y2.eval())


class SpaceToBatchErrorHandlingTest(tf.test.TestCase):

  def testInputWrongDimMissingBatch(self):
    # The input is missing the first dimension ("batch")
    x_np = [[[1], [2]], [[3], [4]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 2
    with self.assertRaises(ValueError):
      _ = tf.space_to_batch(x_np, paddings, block_size)

  def testBlockSize0(self):
    # The block size is 0.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 0
    with self.assertRaises(ValueError):
      out_tf = tf.space_to_batch(x_np, paddings, block_size)
      out_tf.eval()

  def testBlockSizeOne(self):
    # The block size is 1. The block size needs to be > 1.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 1
    with self.assertRaises(ValueError):
      out_tf = tf.space_to_batch(x_np, paddings, block_size)
      out_tf.eval()

  def testBlockSizeLarger(self):
    # The block size is too large for this input.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 10
    with self.assertRaises(IndexError):
      out_tf = tf.space_to_batch(x_np, paddings, block_size)
      out_tf.eval()

  def testBlockSizeNotDivisibleWidth(self):
    # The block size divides width but not height.
    x_np = [[[[1], [2], [3]], [[3], [4], [7]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(IndexError):
      _ = tf.space_to_batch(x_np, paddings, block_size)

  def testBlockSizeNotDivisibleHeight(self):
    # The block size divides height but not width.
    x_np = [[[[1], [2]], [[3], [4]], [[5], [6]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(IndexError):
      _ = tf.space_to_batch(x_np, paddings, block_size)

  def testBlockSizeNotDivisibleBoth(self):
    # The block size does not divide neither width or height.
    x_np = [[[[1], [2]], [[3], [4]]]]
    paddings = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(IndexError):
      _ = tf.space_to_batch(x_np, paddings, block_size)

  def testUnknownShape(self):
    t = tf.space_to_batch(tf.placeholder(tf.float32), tf.placeholder(tf.int32),
                          block_size=4)
    self.assertEqual(4, t.get_shape().ndims)


class SpaceToBatchGradientTest(tf.test.TestCase):

  # Check the gradients.
  def _checkGrad(self, x, paddings, block_size):
    assert 4 == x.ndim
    with self.test_session():
      tf_x = tf.convert_to_tensor(x)
      tf_y = tf.space_to_batch(tf_x, paddings, block_size)
      epsilon = 1e-5
      ((x_jacob_t, x_jacob_n)) = tf.test.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a gradient for space_to_batch of x which is a four dimensional
  # tensor of shape [b, h * block_size, w * block_size, d].
  def _compare(self, b, h, w, d, block_size, pad_beg, pad_end):
    block_size_sq = block_size * block_size
    x = np.random.normal(
        0, 1, b * h * w * d * block_size_sq).astype(np.float32).reshape(
            [b, h * block_size, w * block_size, d])
    paddings = np.array([[pad_beg, pad_end], [pad_beg, pad_end]],
                        dtype=np.int32)

    self._checkGrad(x, paddings, block_size)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  def testSmall(self):
    block_size = 2
    pad_beg = 0
    pad_end = 0
    self._compare(1, 2, 3, 5, block_size, pad_beg, pad_end)

  def testSmall2(self):
    block_size = 2
    pad_beg = 0
    pad_end = 0
    self._compare(2, 4, 3, 2, block_size, pad_beg, pad_end)

  def testSmallPad1x1(self):
    block_size = 2
    pad_beg = 1
    pad_end = 1
    self._compare(1, 2, 3, 5, block_size, pad_beg, pad_end)


if __name__ == "__main__":
  tf.test.main()
