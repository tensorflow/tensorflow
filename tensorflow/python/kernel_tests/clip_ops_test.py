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

"""Tests for tensorflow.ops.clip_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ClipTest(tf.test.TestCase):

  # ClipByValue test
  def testClipByValue(self):
    with self.test_session():
      x = tf.constant([-5.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
      np_ans = [[-4.4, 2.0, 3.0],
                [4.0, 4.4, 4.4]]
      clip_value = 4.4
      ans = tf.clip_by_value(x, -clip_value, clip_value)
      tf_ans = ans.eval()

    self.assertAllClose(np_ans, tf_ans)

  def testClipByValueNonFinite(self):
    with self.test_session():
      x = tf.constant([float('NaN'), float('Inf'), -float('Inf')])
      np_ans = [float('NaN'), 4.0, -4.0]
      clip_value = 4.0
      ans = tf.clip_by_value(x, -clip_value, clip_value)
      tf_ans = ans.eval()

    self.assertAllClose(np_ans, tf_ans)

  # ClipByNorm tests
  def testClipByNormClipped(self):
    # Norm clipping when clip_norm < 5
    with self.test_session():
      x = tf.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Norm of x = sqrt(3^2 + 4^2) = 5
      np_ans = [[-2.4, 0.0, 0.0],
                [3.2, 0.0, 0.0]]
      clip_norm = 4.0
      ans = tf.clip_by_norm(x, clip_norm)
      tf_ans = ans.eval()

    self.assertAllClose(np_ans, tf_ans)

  def testClipByNormNotClipped(self):
    # No norm clipping when clip_norm >= 5
    with self.test_session():
      x = tf.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Norm of x = sqrt(3^2 + 4^2) = 5
      np_ans = [[-3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0]]
      clip_norm = 6.0
      ans = tf.clip_by_norm(x, clip_norm)
      tf_ans = ans.eval()

    self.assertAllClose(np_ans, tf_ans)

  def testClipByNormZero(self):
    # No norm clipping when norm = 0
    with self.test_session():
      x = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
      # Norm = 0, no changes
      np_ans = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]
      clip_norm = 6.0
      ans = tf.clip_by_norm(x, clip_norm)
      tf_ans = ans.eval()

    self.assertAllClose(np_ans, tf_ans)

  def testClipByGlobalNormClipped(self):
    # Norm clipping when clip_norm < 5
    with self.test_session():
      x0 = tf.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = tf.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = 4.0

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0],
                  [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = tf.clip_by_global_norm((x0, x1), clip_norm)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[1].eval()
      tf_norm = norm.eval()

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  def testClipByGlobalNormSupportsNone(self):
    # Norm clipping when clip_norm < 5
    with self.test_session():
      x0 = tf.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = tf.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = 4.0

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0],
                  [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = tf.clip_by_global_norm((x0, None, x1, None), clip_norm)
      self.assertTrue(ans[1] is None)
      self.assertTrue(ans[3] is None)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[2].eval()
      tf_norm = norm.eval()

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  # ClipByGlobalNorm tests
  def testClipByGlobalNormWithIndexedSlicesClipped(self):
    # Norm clipping when clip_norm < 5
    with self.test_session():
      x0 = tf.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = tf.IndexedSlices(tf.constant([1.0, -2.0]),
                            tf.constant([3, 4]))
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      clip_norm = 4.0

      # Answers are the original tensors scaled by 4.0/5.0
      np_ans_0 = [[-1.6, 0.0, 0.0],
                  [3.2, 0.0, 0.0]]
      np_ans_1 = [0.8, -1.6]

      ans, norm = tf.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[1].values.eval()
      tf_norm = norm.eval()

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  def testClipByGlobalNormPreservesDenseShape(self):
    dense_shape = (1,)
    slices = tf.IndexedSlices(
        tf.constant([1.0]),
        tf.constant([0]),
        dense_shape=dense_shape)
    ans, _ = tf.clip_by_global_norm([slices], 1.0)
    modified_slices = ans[0]
    self.assertEqual(dense_shape, slices.dense_shape)
    self.assertEqual(dense_shape, modified_slices.dense_shape)

  def testClipByGlobalNormNotClipped(self):
    # No norm clipping when clip_norm >= 5
    with self.test_session():
      x0 = tf.constant([-2.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      x1 = tf.constant([1.0, -2.0])
      # Global norm of x0 and x1 = sqrt(1 + 4^2 + 2^2 + 2^2) = 5
      np_ans_0 = [[-2.0, 0.0, 0.0],
                  [4.0, 0.0, 0.0]]
      np_ans_1 = [1.0, -2.0]
      clip_norm = 6.0

      ans, norm = tf.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[1].eval()
      tf_norm = norm.eval()

    self.assertAllClose(tf_norm, 5.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  def testClipByGlobalNormZero(self):
    # No norm clipping when norm = 0
    with self.test_session():
      x0 = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
      x1 = tf.constant([0.0, 0.0])
      # Norm = 0, no changes
      np_ans_0 = [[0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0]]
      np_ans_1 = [0.0, 0.0]
      clip_norm = 6.0

      ans, norm = tf.clip_by_global_norm([x0, x1], clip_norm)
      tf_ans_1 = ans[0].eval()
      tf_ans_2 = ans[1].eval()
      tf_norm = norm.eval()

    self.assertAllClose(tf_norm, 0.0)
    self.assertAllClose(np_ans_0, tf_ans_1)
    self.assertAllClose(np_ans_1, tf_ans_2)

  def testClipByAverageNormClipped(self):
    # Norm clipping when average clip_norm < 0.83333333
    with self.test_session():
      x = tf.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Average norm of x = sqrt(3^2 + 4^2) / 6 = 0.83333333
      np_ans = [[-2.88, 0.0, 0.0],
                [3.84, 0.0, 0.0]]
      clip_norm = 0.8
      ans = tf.clip_by_average_norm(x, clip_norm)
      tf_ans = ans.eval()

    self.assertAllClose(np_ans, tf_ans)

  def testClipByAverageNormNotClipped(self):
    # No norm clipping when average clip_norm >= 0.83333333
    with self.test_session():
      x = tf.constant([-3.0, 0.0, 0.0, 4.0, 0.0, 0.0], shape=[2, 3])
      # Average norm of x = sqrt(3^2 + 4^2) / 6 = 0.83333333
      np_ans = [[-3.0, 0.0, 0.0],
                [4.0, 0.0, 0.0]]
      clip_norm = 0.9
      ans = tf.clip_by_average_norm(x, clip_norm)
      tf_ans = ans.eval()

    self.assertAllClose(np_ans, tf_ans)

  def testClipByAverageNormZero(self):
    # No norm clipping when average clip_norm = 0
    with self.test_session():
      x = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], shape=[2, 3])
      # Average norm = 0, no changes
      np_ans = [[0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]]
      clip_norm = 0.9
      ans = tf.clip_by_average_norm(x, clip_norm)
      tf_ans = ans.eval()

    self.assertAllClose(np_ans, tf_ans)

if __name__ == "__main__":
  tf.test.main()
