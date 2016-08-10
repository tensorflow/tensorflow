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

  def testMap_Grad(self):
    with self.test_session():
      param = tf.constant(2.0)
      elems = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="elems")
      y = tf.map_fn(lambda x: tf.mul(tf.square(x), param), elems)
      r = tf.gradients(y, param)[0]
      self.assertAllEqual(91.0, r.eval())
      r = tf.gradients(y, elems)[0]
      self.assertAllEqual([4.0, 8.0, 12.0, 16.0, 20.0, 24.0], r.eval())

  def testMap_SimpleNotTensor(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      r = tf.map_fn(lambda x: tf.mul(tf.add(x, 3), 2), nums)
      self.assertAllEqual(np.array([(x + 3) * 2 for x in nums]), r.eval())

  def testMap_SingleInputMultiOutput(self):
    with self.test_session() as sess:
      nums = np.array([1, 2, 3, 4, 5, 6])
      r = tf.map_fn(lambda x: ((x + 3) * 2, -(x + 3) * 2), nums,
                    dtype=(tf.int64, tf.int64))
      self.assertEqual(2, len(r))
      self.assertEqual((6,), r[0].get_shape())
      self.assertEqual((6,), r[1].get_shape())
      received = sess.run(r)
      self.assertAllEqual((nums + 3) * 2, received[0])
      self.assertAllEqual(-(nums + 3) * 2, received[1])

  def testMap_MultiOutputMismatchedDtype(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      with self.assertRaisesRegexp(
          TypeError, r"two structures don't have the same sequence type."):
        # lambda emits tuple, but dtype is a list
        tf.map_fn(lambda x: ((x + 3) * 2, -(x + 3) * 2), nums,
                  dtype=[tf.int64, tf.int64])

  def testMap_MultiInputSingleOutput(self):
    with self.test_session():
      nums = np.array([1, 2, 3, 4, 5, 6])
      r = tf.map_fn(lambda x: x[0]*x[1][0] + x[1][1], (nums, (nums, -nums)),
                    dtype=tf.int64)
      self.assertEqual((6,), r.get_shape())
      received = r.eval()
      self.assertAllEqual(nums*nums + (-nums), received)

  def testMap_MultiInputSameStructureOutput(self):
    with self.test_session() as sess:
      nums = np.array([1, 2, 3, 4, 5, 6])
      r = tf.map_fn(
          lambda x: (x[1][0], (x[1][1], x[0])), (nums, (2*nums, -nums)))
      r = [r[0], r[1][0], r[1][1]]
      self.assertEqual((6,), r[0].get_shape())
      self.assertEqual((6,), r[1].get_shape())
      self.assertEqual((6,), r[2].get_shape())
      received = sess.run(r)
      self.assertAllEqual(2 * nums, received[0])
      self.assertAllEqual(-nums, received[1])
      self.assertAllEqual(nums, received[2])

  def testScan_Simple(self):
    with self.test_session():
      elems = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name="data")
      v = tf.constant(2.0, name="v")

      r = tf.scan(lambda a, x: tf.mul(a, x), elems)
      self.assertAllEqual([1., 2., 6., 24., 120., 720.], r.eval())

      r = tf.scan(
          lambda a, x: tf.mul(a, x), elems, initializer=v)
      self.assertAllEqual([2., 4., 12., 48., 240., 1440.], r.eval())

  def testScan_SingleInputMultiOutput(self):
    with self.test_session() as sess:
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = (np.array(1.0), np.array(-1.0))
      r = tf.scan(lambda a, x: (a[0] * x, -a[1] * x), elems, initializer)
      r_value = sess.run(r)

      self.assertAllEqual([1.0, 2.0, 6.0, 24.0, 120.0, 720.0], r_value[0])
      self.assertAllEqual([1.0, -2.0, 6.0, -24.0, 120.0, -720.0], r_value[1])

  def testScan_MultiInputSingleOutput(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = np.array(1.0)
      # Multiply a * 1 each time
      r = tf.scan(
          lambda a, x: a * (x[0] + x[1]), (elems + 1, -elems), initializer)
      self.assertAllEqual([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], r.eval())

  def testScan_MultiInputSameTypeOutput(self):
    with self.test_session() as sess:
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      r = tf.scan(
          lambda a, x: (a[0] + x[0], a[1] + x[1]), (elems, -elems))
      r_value = sess.run(r)
      self.assertAllEqual(np.cumsum(elems), r_value[0])
      self.assertAllEqual(np.cumsum(-elems), r_value[1])

  def testScan_MultiOutputMismatchedInitializer(self):
    with self.test_session():
      elems = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      initializer = np.array(1.0)
      # Multiply a * 1 each time
      with self.assertRaisesRegexp(
          ValueError, "two structures don't have the same number of elements"):
        tf.scan(lambda a, x: (a, -a), elems, initializer)

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

  def testMapEmptyScalar(self):
    with self.test_session():
      map_return = tf.map_fn(lambda x: 1, tf.constant([]))
      self.assertAllEqual([0], map_return.get_shape().dims)
      self.assertAllEqual([0], map_return.eval().shape)

  def testMapEmptyTensor(self):
    with self.test_session():
      map_return = tf.map_fn(lambda x: tf.zeros([3, 2]), tf.constant([]))
      self.assertAllEqual([0, 3, 2], map_return.get_shape().dims)
      self.assertAllEqual([0, 3, 2], map_return.eval().shape)

  def testScanShape(self):
    with self.test_session():
      x = tf.constant([[1, 2, 3], [4, 5, 6]])
      def fn(_, current_input):
        return current_input
      initializer = tf.constant([0, 0, 0])
      y = tf.scan(fn, x, initializer=initializer)
      self.assertAllEqual(y.get_shape(), y.eval().shape)

  def testScanEmptyTensor(self):
    with self.test_session():
      x = tf.scan(lambda x, _: x, tf.range(0), initializer=tf.ones([2, 4]))
      self.assertAllEqual([0, 2, 4], x.get_shape())
      self.assertAllEqual(x.get_shape(), x.eval().shape)

  def testScanUnknownShape(self):
    x = tf.placeholder(tf.float32)
    initializer = tf.placeholder(tf.float32)
    def fn(_, current_input):
      return current_input
    y = tf.scan(fn, x, initializer=initializer)
    self.assertIs(None, y.get_shape().dims)

  def testScanVaryingShape(self):
    with self.test_session() as sess:
      x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
      x_t = tf.transpose(x)
      # scan over dimension 0 (with shape None)
      result = tf.scan(lambda a, x: a + x, x)
      # scanned over transposed dimension 0 (with shape 2)
      result_t = tf.scan(lambda a, x: a + x, x_t, infer_shape=False)
      # ensure gradients can be calculated
      result_grad = tf.gradients(result, [x])[0]
      result_t_grad = tf.gradients(result_t, [x_t])[0]

      # smoke test to ensure they all evaluate
      sess.run([result, result_t, result_grad, result_t_grad],
               feed_dict={x: [[1.0, 2.0]]})


if __name__ == "__main__":
  tf.test.main()
