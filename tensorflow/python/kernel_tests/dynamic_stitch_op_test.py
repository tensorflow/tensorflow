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

"""Tests for tensorflow.ops.data_flow_ops.dynamic_stitch."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class DynamicStitchTest(tf.test.TestCase):

  def testScalar(self):
    with self.test_session():
      indices = [tf.constant(0), tf.constant(1)]
      data = [tf.constant(40), tf.constant(60)]
      for step in -1, 1:
        stitched_t = tf.dynamic_stitch(indices[::step], data)
        stitched_val = stitched_t.eval()
        self.assertAllEqual([40, 60][::step], stitched_val)
        # Dimension 0 is determined by the max index in indices, so we
        # can only infer that the output is a vector of some unknown
        # length.
        self.assertEqual([None], stitched_t.get_shape().as_list())

  def testSimpleOneDimensional(self):
    with self.test_session():
      indices = [tf.constant([0, 4, 7]),
                 tf.constant([1, 6, 2, 3, 5])]
      data = [tf.constant([0, 40, 70]),
              tf.constant([10, 60, 20, 30, 50])]
      stitched_t = tf.dynamic_stitch(indices, data)
      stitched_val = stitched_t.eval()
      self.assertAllEqual([0, 10, 20, 30, 40, 50, 60, 70], stitched_val)
      # Dimension 0 is determined by the max index in indices, so we
      # can only infer that the output is a vector of some unknown
      # length.
      self.assertEqual([None], stitched_t.get_shape().as_list())

  def testSimpleTwoDimensional(self):
    with self.test_session():
      indices = [tf.constant([0, 4, 7]),
                 tf.constant([1, 6]),
                 tf.constant([2, 3, 5])]
      data = [tf.constant([[0, 1], [40, 41], [70, 71]]),
              tf.constant([[10, 11], [60, 61]]),
              tf.constant([[20, 21], [30, 31], [50, 51]])]
      stitched_t = tf.dynamic_stitch(indices, data)
      stitched_val = stitched_t.eval()
      self.assertAllEqual(
          [[0, 1], [10, 11], [20, 21], [30, 31],
           [40, 41], [50, 51], [60, 61], [70, 71]], stitched_val)
      # Dimension 0 is determined by the max index in indices, so we
      # can only infer that the output is a matrix with 2 columns and
      # some unknown number of rows.
      self.assertEqual([None, 2], stitched_t.get_shape().as_list())

  def testHigherRank(self):
    with self.test_session() as sess:
      indices = [tf.constant(6), tf.constant([4, 1]),
                 tf.constant([[5, 2], [0, 3]])]
      data = [tf.constant([61, 62]), tf.constant([[41, 42], [11, 12]]),
              tf.constant([[[51, 52], [21, 22]], [[1, 2], [31, 32]]])]
      stitched_t = tf.dynamic_stitch(indices, data)
      stitched_val = stitched_t.eval()
      correct = 10 * np.arange(7)[:, None] + [1, 2]
      self.assertAllEqual(correct, stitched_val)
      self.assertEqual([None, 2], stitched_t.get_shape().as_list())
      # Test gradients
      stitched_grad = 7 * stitched_val
      grads = tf.gradients(stitched_t, indices + data, stitched_grad)
      self.assertEqual(grads[:3], [None] * 3)  # Indices have no gradients
      for datum, grad in zip(data, sess.run(grads[3:])):
        self.assertAllEqual(7 * datum.eval(), grad)

  def testErrorIndicesMultiDimensional(self):
    indices = [tf.constant([0, 4, 7]),
               tf.constant([[1, 6, 2, 3, 5]])]
    data = [tf.constant([[0, 40, 70]]),
            tf.constant([10, 60, 20, 30, 50])]
    with self.assertRaises(ValueError):
      tf.dynamic_stitch(indices, data)

  def testErrorDataNumDimsMismatch(self):
    indices = [tf.constant([0, 4, 7]),
               tf.constant([1, 6, 2, 3, 5])]
    data = [tf.constant([0, 40, 70]),
            tf.constant([[10, 60, 20, 30, 50]])]
    with self.assertRaises(ValueError):
      tf.dynamic_stitch(indices, data)

  def testErrorDataDimSizeMismatch(self):
    indices = [tf.constant([0, 4, 5]),
               tf.constant([1, 6, 2, 3])]
    data = [tf.constant([[0], [40], [70]]),
            tf.constant([[10, 11], [60, 61], [20, 21], [30, 31]])]
    with self.assertRaises(ValueError):
      tf.dynamic_stitch(indices, data)

  def testErrorDataAndIndicesSizeMismatch(self):
    indices = [tf.constant([0, 4, 7]),
               tf.constant([1, 6, 2, 3, 5])]
    data = [tf.constant([0, 40, 70]),
            tf.constant([10, 60, 20, 30])]
    with self.assertRaises(ValueError):
      tf.dynamic_stitch(indices, data)


if __name__ == "__main__":
  tf.test.main()
