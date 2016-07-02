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

"""Functional tests for BatchToSpace op.

Additional tests are included in spacetobatch_op_test.py, where the BatchToSpace
op is tested in tandem with its reverse SpaceToBatch op.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class BatchToSpaceDepthToSpace(tf.test.TestCase):

  # Verifies that: batch_to_space(x) = transpose(depth_to_space(transpose(x)))
  def testDepthToSpaceTranspose(self):
    x = np.arange(20 * 5 * 8 * 7, dtype=np.float32).reshape([20, 5, 8, 7])
    block_size = 2
    crops = np.zeros((2, 2), dtype=np.int32)
    y1 = tf.batch_to_space(x, crops, block_size=block_size)
    y2 = tf.transpose(
        tf.depth_to_space(
            tf.transpose(x, [3, 1, 2, 0]), block_size=block_size),
        [3, 1, 2, 0])
    with self.test_session():
      self.assertAllEqual(y1.eval(), y2.eval())


class BatchToSpaceErrorHandlingTest(tf.test.TestCase):

  def testInputWrongDimMissingBatch(self):
    # The input is missing the first dimension ("batch")
    x_np = [[[1], [2]], [[3], [4]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 2
    with self.assertRaises(ValueError):
      _ = tf.batch_to_space(x_np, crops, block_size)

  def testBlockSize0(self):
    # The block size is 0.
    x_np = [[[[1], [2]], [[3], [4]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 0
    with self.assertRaises(ValueError):
      out_tf = tf.batch_to_space(x_np, crops, block_size)
      out_tf.eval()

  def testBlockSizeOne(self):
    # The block size is 1. The block size needs to be > 1.
    x_np = [[[[1], [2]], [[3], [4]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 1
    with self.assertRaises(ValueError):
      out_tf = tf.batch_to_space(x_np, crops, block_size)
      out_tf.eval()

  def testBlockSizeLarger(self):
    # The block size is too large for this input.
    x_np = [[[[1], [2]], [[3], [4]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 10
    with self.assertRaises(IndexError):
      out_tf = tf.batch_to_space(x_np, crops, block_size)
      out_tf.eval()

  def testBlockSizeSquaredNotDivisibleBatch(self):
    # The block size squared does not divide the batch.
    x_np = [[[[1], [2], [3]], [[3], [4], [7]]]]
    crops = np.zeros((2, 2), dtype=np.int32)
    block_size = 3
    with self.assertRaises(IndexError):
      _ = tf.batch_to_space(x_np, crops, block_size)

  def testUnknownShape(self):
    t = tf.batch_to_space(tf.placeholder(tf.float32), tf.placeholder(tf.int32),
                          block_size=4)
    self.assertEqual(4, t.get_shape().ndims)


class BatchToSpaceGradientTest(tf.test.TestCase):

  # Check the gradients.
  def _checkGrad(self, x, crops, block_size):
    assert 4 == x.ndim
    with self.test_session():
      tf_x = tf.convert_to_tensor(x)
      tf_y = tf.batch_to_space(tf_x, crops, block_size)
      epsilon = 1e-5
      ((x_jacob_t, x_jacob_n)) = tf.test.compute_gradient(
          tf_x,
          x.shape,
          tf_y,
          tf_y.get_shape().as_list(),
          x_init_value=x,
          delta=epsilon)

    self.assertAllClose(x_jacob_t, x_jacob_n, rtol=1e-2, atol=epsilon)

  # Tests a gradient for batch_to_space of x which is a four dimensional
  # tensor of shape [b * block_size * block_size, h, w, d].
  def _compare(self, b, h, w, d, block_size, crop_beg, crop_end):
    block_size_sq = block_size * block_size
    x = np.random.normal(
        0, 1, b * h * w * d * block_size_sq).astype(np.float32).reshape(
            [b * block_size * block_size, h, w, d])
    crops = np.array([[crop_beg, crop_end], [crop_beg, crop_end]],
                     dtype=np.int32)

    self._checkGrad(x, crops, block_size)

  # Don't use very large numbers as dimensions here as the result is tensor
  # with cartesian product of the dimensions.
  def testSmall(self):
    block_size = 2
    crop_beg = 0
    crop_end = 0
    self._compare(1, 2, 3, 5, block_size, crop_beg, crop_end)

  def testSmall2(self):
    block_size = 2
    crop_beg = 0
    crop_end = 0
    self._compare(2, 4, 3, 2, block_size, crop_beg, crop_end)

  def testSmallCrop1x1(self):
    block_size = 2
    crop_beg = 1
    crop_end = 1
    self._compare(1, 2, 3, 5, block_size, crop_beg, crop_end)


if __name__ == "__main__":
  tf.test.main()
