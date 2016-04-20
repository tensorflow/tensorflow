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

"""Tests for tensorflow.kernels.bcast_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


class FunctionalOpsTest(tf.test.TestCase):

  def testFoldl_Simple(self):
    with self.test_session():
      elems = tf.constant([1, 2, 3, 4, 5, 6], name="data")

      r = tf.foldl(lambda a, x: tf.mul(tf.add(a, x), 2), elems)
      self.assertAllEqual(208, r.eval())

      r = tf.foldl(
          lambda a, x: tf.mul(tf.add(a, x), 2), elems, initializer=10)
      self.assertAllEqual(880, r.eval())

  def testFoldr_Simple(self):
    with self.test_session():
      elems = tf.constant([1, 2, 3, 4, 5, 6], name="data")

      r = tf.foldr(lambda a, x: tf.mul(tf.add(a, x), 2), elems)
      self.assertAllEqual(450, r.eval())

      r = tf.foldr(
          lambda a, x: tf.mul(tf.add(a, x), 2), elems, initializer=10)
      self.assertAllEqual(1282, r.eval())

  def testFold_Grad(self):
    with self.test_session():
      elems = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
      v = tf.constant(2.0, name="v")

      r = tf.foldl(
          lambda a, x: tf.mul(a, x), elems, initializer=v)
      r = tf.gradients(r, v)[0]
      self.assertAllEqual(720.0, r.eval())

      r = tf.foldr(
          lambda a, x: tf.mul(a, x), elems, initializer=v)
      r = tf.gradients(r, v)[0]
      self.assertAllEqual(720.0, r.eval())

  def testMap_Simple(self):
    with self.test_session():
      nums = [1, 2, 3, 4, 5, 6]
      elems = tf.constant(nums, name="data")
      r = tf.map_fn(lambda x: tf.mul(tf.add(x, 3), 2), elems)
      self.assertAllEqual(np.array([(x + 3) * 2 for x in nums]), r.eval())

  def testScan_Simple(self):
    with self.test_session():
      elems = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
      v = tf.constant(2.0, name="v")

      r = tf.scan(lambda a, x: tf.mul(a, x), elems)
      self.assertAllEqual([1., 2., 6., 24., 120., 720.], r.eval())

      r = tf.scan(
          lambda a, x: tf.mul(a, x), elems, initializer=v)
      self.assertAllEqual([2., 4., 12., 48., 240., 1440.], r.eval())

  def testScan_Grad(self):
    with self.test_session():
      elems = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
      v = tf.constant(2.0, name="v")

      r = tf.scan(lambda a, x: tf.mul(a, x), elems, initializer=v)
      r = tf.gradients(r, v)[0]
      self.assertAllEqual(873.0, r.eval())

  def testFoldShape(self):
    with self.test_session():
      x = tf.constant([[1, 2, 3], [4, 5, 6]])
      def fn(_, current_input):
        return current_input
      initializer = tf.constant([0, 0, 0])
      y = tf.foldl(fn, x, initializer=initializer)
      self.assertAllEqual(y.get_shape(), y.eval().shape)

  def testMapShape(self):
    with self.test_session():
      x = tf.constant([[1, 2, 3], [4, 5, 6]])
      y = tf.map_fn(lambda e: e, x)
      self.assertAllEqual(y.get_shape(), y.eval().shape)

  def testScanShape(self):
    with self.test_session():
      x = tf.constant([[1, 2, 3], [4, 5, 6]])
      def fn(_, current_input):
        return current_input
      initializer = tf.constant([0, 0, 0])
      y = tf.scan(fn, x, initializer=initializer)
      self.assertAllEqual(y.get_shape(), y.eval().shape)

if __name__ == "__main__":
  tf.test.main()
