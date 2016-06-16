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

"""Tests for tensorflow.kernels.bcast_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def simple_scoped_fn(a, x):
  """Simple function: (a, x) -> 2(x+a), but with "2" as a variable in scope."""
  with tf.variable_scope("body"):
    # Dummy variable, just to check that scoping works as intended.
    two = tf.get_variable("two", [], dtype=tf.int32,
                          initializer=tf.constant_initializer(2))
    return tf.mul(tf.add(a, x), two)


class FunctionalOpsTest(tf.test.TestCase):

  def testFoldl_Simple(self):
    with self.test_session():
      elems = tf.constant([1, 2, 3, 4, 5, 6], name="data")

      r = tf.foldl(lambda a, x: tf.mul(tf.add(a, x), 2), elems)
      self.assertAllEqual(208, r.eval())

      r = tf.foldl(
          lambda a, x: tf.mul(tf.add(a, x), 2), elems, initializer=10)
      self.assertAllEqual(880, r.eval())

  def testFoldl_Scoped(self):
    with self.test_session() as sess:
      with tf.variable_scope("root") as varscope:
        elems = tf.constant([1, 2, 3, 4, 5, 6], name="data")

        r = tf.foldl(simple_scoped_fn, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(tf.trainable_variables()), 1)
        self.assertEqual(tf.trainable_variables()[0].name, "root/body/two:0")
        sess.run([tf.initialize_all_variables()])
        self.assertAllEqual(208, r.eval())

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = tf.foldl(simple_scoped_fn, elems, initializer=10)
        self.assertEqual(len(tf.trainable_variables()), 1)
        self.assertAllEqual(880, r.eval())

  def testFoldr_Simple(self):
    with self.test_session():
      elems = tf.constant([1, 2, 3, 4, 5, 6], name="data")

      r = tf.foldr(lambda a, x: tf.mul(tf.add(a, x), 2), elems)
      self.assertAllEqual(450, r.eval())

      r = tf.foldr(
          lambda a, x: tf.mul(tf.add(a, x), 2), elems, initializer=10)
      self.assertAllEqual(1282, r.eval())

  def testFoldr_Scoped(self):
    with self.test_session() as sess:
      with tf.variable_scope("root") as varscope:
        elems = tf.constant([1, 2, 3, 4, 5, 6], name="data")

        r = tf.foldr(simple_scoped_fn, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(tf.trainable_variables()), 1)
        self.assertEqual(tf.trainable_variables()[0].name, "root/body/two:0")
        sess.run([tf.initialize_all_variables()])
        self.assertAllEqual(450, r.eval())

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = tf.foldr(simple_scoped_fn, elems, initializer=10)
        self.assertEqual(len(tf.trainable_variables()), 1)
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

  def testMap_Scoped(self):
    with self.test_session() as sess:

      def double_scoped(x):
        """2x with a dummy 2 that is scoped."""
        with tf.variable_scope("body"):
          # Dummy variable, just to check that scoping works as intended.
          two = tf.get_variable("two", [], dtype=tf.int32,
                                initializer=tf.constant_initializer(2))
          return tf.mul(x, two)

      with tf.variable_scope("root") as varscope:
        elems = tf.constant([1, 2, 3, 4, 5, 6], name="data")
        doubles = np.array([2*x for x in [1, 2, 3, 4, 5, 6]])

        r = tf.map_fn(double_scoped, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(tf.trainable_variables()), 1)
        self.assertEqual(tf.trainable_variables()[0].name, "root/body/two:0")
        sess.run([tf.initialize_all_variables()])
        self.assertAllEqual(doubles, r.eval())

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = tf.map_fn(double_scoped, elems)
        self.assertEqual(len(tf.trainable_variables()), 1)
        self.assertAllEqual(doubles, r.eval())

  def testMap_SimpleNotTensor(self):
    with self.test_session():
      nums = [1, 2, 3, 4, 5, 6]
      r = tf.map_fn(lambda x: tf.mul(tf.add(x, 3), 2), nums)
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

  def testScan_Scoped(self):
    with self.test_session() as sess:
      with tf.variable_scope("root") as varscope:
        elems = tf.constant([1, 2, 3, 4, 5, 6], name="data")

        r = tf.scan(simple_scoped_fn, elems)
        # Check that we have the one variable we asked for here.
        self.assertEqual(len(tf.trainable_variables()), 1)
        self.assertEqual(tf.trainable_variables()[0].name, "root/body/two:0")
        sess.run([tf.initialize_all_variables()])
        results = np.array([1, 6, 18, 44, 98, 208])
        self.assertAllEqual(results, r.eval())

        # Now let's reuse our single variable.
        varscope.reuse_variables()
        r = tf.scan(simple_scoped_fn, elems, initializer=2)
        self.assertEqual(len(tf.trainable_variables()), 1)
        results = np.array([6, 16, 38, 84, 178, 368])
        self.assertAllEqual(results, r.eval())

  def testScanFoldl_Nested(self):
    with self.test_session():
      elems = tf.constant([1.0, 2.0, 3.0, 4.0], name="data")
      inner_elems = tf.constant([0.5, 0.5], name="data")

      def r_inner(a, x):
        return tf.foldl(lambda b, y: b * y * x, inner_elems, initializer=a)

      r = tf.scan(r_inner, elems)

      # t == 0 (returns 1)
      # t == 1, a == 1, x == 2 (returns 1)
      #   t_0 == 0, b == a == 1, y == 0.5, returns b * y * x = 1
      #   t_1 == 1, b == 1,      y == 0.5, returns b * y * x = 1
      # t == 2, a == 1, x == 3 (returns 1.5*1.5 == 2.25)
      #   t_0 == 0, b == a == 1, y == 0.5, returns b * y * x = 1.5
      #   t_1 == 1, b == 1.5,    y == 0.5, returns b * y * x = 1.5*1.5
      # t == 3, a == 2.25, x == 4 (returns 9)
      #   t_0 == 0, b == a == 2.25, y == 0.5, returns b * y * x = 4.5
      #   t_1 == 1, b == 4.5,       y == 0.5, returns b * y * x = 9
      self.assertAllClose([1., 1., 2.25, 9.], r.eval())

  def testScan_Control(self):
    with self.test_session() as sess:
      s = tf.placeholder(tf.float32, shape=[None])
      b = tf.placeholder(tf.bool)

      with tf.control_dependencies([b]):
        c = tf.scan(lambda a, x: x * a, s)
      self.assertAllClose(np.array([1.0, 3.0, 9.0]),
                          sess.run(c, {s: [1, 3, 3], b: True}))

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

  def testMapUnknownShape(self):
    x = tf.placeholder(tf.float32)
    y = tf.map_fn(lambda e: e, x)
    self.assertIs(None, y.get_shape().dims)

  def testScanShape(self):
    with self.test_session():
      x = tf.constant([[1, 2, 3], [4, 5, 6]])
      def fn(_, current_input):
        return current_input
      initializer = tf.constant([0, 0, 0])
      y = tf.scan(fn, x, initializer=initializer)
      self.assertAllEqual(y.get_shape(), y.eval().shape)

  def testScanUnknownShape(self):
    x = tf.placeholder(tf.float32)
    initializer = tf.placeholder(tf.float32)
    def fn(_, current_input):
      return current_input
    y = tf.scan(fn, x, initializer=initializer)
    self.assertIs(None, y.get_shape().dims)


if __name__ == "__main__":
  tf.test.main()
