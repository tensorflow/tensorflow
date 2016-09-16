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

"""Functional tests for Pack Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def np_split_sqeeze(array, axis):
  axis_len = array.shape[axis]
  return [
      np.squeeze(arr, axis=(axis,))
      for arr in np.split(array, axis_len, axis=axis)
  ]


class PackOpTest(tf.test.TestCase):

  def testSimple(self):
    np.random.seed(7)
    with self.test_session(use_gpu=True):
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        data = np.random.randn(*shape)
        # Convert [data[0], data[1], ...] separately to tensorflow
        # TODO(irving): Remove list() once we handle maps correctly
        xs = list(map(tf.constant, data))
        # Pack back into a single tensorflow tensor
        c = tf.pack(xs)
        self.assertAllEqual(c.eval(), data)

  def testConst(self):
    np.random.seed(7)
    with self.test_session(use_gpu=True):
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        data = np.random.randn(*shape).astype(np.float32)
        # Pack back into a single tensorflow tensor directly using np array
        c = tf.pack(data)
        # This is implemented via a Const:
        self.assertEqual(c.op.type, "Const")
        self.assertAllEqual(c.eval(), data)

        # Python lists also work for 1-D case:
        if len(shape) == 1:
          data_list = list(data)
          cl = tf.pack(data_list)
          self.assertEqual(cl.op.type, "Const")
          self.assertAllEqual(cl.eval(), data)

      # Verify that shape induction works with shapes produced via const pack
      a = tf.constant([1, 2, 3, 4, 5, 6])
      b = tf.reshape(a, tf.pack([2, 3]))
      self.assertAllEqual(b.get_shape(), [2, 3])

  def testGradientsAxis0(self):
    np.random.seed(7)
    for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
      data = np.random.randn(*shape)
      shapes = [shape[1:]] * shape[0]
      with self.test_session(use_gpu=True):
        # TODO(irving): Remove list() once we handle maps correctly
        xs = list(map(tf.constant, data))
        c = tf.pack(xs)
        err = tf.test.compute_gradient_error(xs, shapes, c, shape)
        self.assertLess(err, 1e-6)

  def testGradientsAxis1(self):
    np.random.seed(7)
    for shape in (2, 3), (3, 2), (4, 3, 2):
      data = np.random.randn(*shape)
      shapes = [shape[1:]] * shape[0]
      out_shape = list(shape[1:])
      out_shape.insert(1, shape[0])
      with self.test_session(use_gpu=True):
        # TODO(irving): Remove list() once we handle maps correctly
        xs = list(map(tf.constant, data))
        c = tf.pack(xs, axis=1)
        err = tf.test.compute_gradient_error(xs, shapes, c, out_shape)
        self.assertLess(err, 1e-6)

  def testZeroSize(self):
    # Verify that pack doesn't crash for zero size inputs
    with self.test_session(use_gpu=True):
      for shape in (0,), (3,0), (0, 3):
        x = np.zeros((2,) + shape)
        p = tf.pack(list(x)).eval()
        self.assertAllEqual(p, x)

  def testAxis0Default(self):
    with self.test_session(use_gpu=True):
      t = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]

      packed = tf.pack(t).eval()

    self.assertAllEqual(packed, np.array([[1, 2, 3], [4, 5, 6]]))

  def testAgainstNumpy(self):
    # For 1 to 5 dimensions.
    for i in range(1, 6):
      expected = np.random.random(np.random.permutation(i) + 1)

      # For all the possible axis to split it, including negative indices.
      for j in range(-i, i):
        test_arrays = np_split_sqeeze(expected, j)

        with self.test_session(use_gpu=True):
          actual = tf.pack(test_arrays, axis=j)
          self.assertEqual(expected.shape, actual.get_shape())
          actual = actual.eval()

        self.assertNDArrayNear(expected, actual, 1e-6)

  def testDimOutOfRange(self):
    t = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]
    with self.assertRaisesRegexp(ValueError, r"axis = 2 not in \[-2, 2\)"):
      tf.unpack(t, axis=2)

  def testDimOutOfNegativeRange(self):
    t = [tf.constant([1, 2, 3]), tf.constant([4, 5, 6])]
    with self.assertRaisesRegexp(ValueError, r"axis = -3 not in \[-2, 2\)"):
      tf.unpack(t, axis=-3)


class AutomaticPackingTest(tf.test.TestCase):

  def testSimple(self):
    with self.test_session(use_gpu=True):
      self.assertAllEqual([1, 0, 2],
                          tf.convert_to_tensor([1, tf.constant(0), 2]).eval())
      self.assertAllEqual(
          [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
          tf.convert_to_tensor([[0, 0, 0],
                                [0, tf.constant(1), 0],
                                [0, 0, 0]]).eval())
      self.assertAllEqual(
          [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
          tf.convert_to_tensor([[0, 0, 0],
                                tf.constant([0, 1, 0]),
                                [0, 0, 0]]).eval())
      self.assertAllEqual(
          [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
          tf.convert_to_tensor([tf.constant([0, 0, 0]),
                                tf.constant([0, 1, 0]),
                                tf.constant([0, 0, 0])]).eval())

  def testWithNDArray(self):
    with self.test_session(use_gpu=True):
      result = tf.convert_to_tensor([[[0., 0.],
                                      tf.constant([1., 1.])],
                                     np.array([[2., 2.], [3., 3.]],
                                              dtype=np.float32)])
      self.assertAllEqual(
          [[[0., 0.], [1., 1.]], [[2., 2.], [3., 3.]]], result.eval())

  def testVariable(self):
    with self.test_session(use_gpu=True):
      v = tf.Variable(17)
      result = tf.convert_to_tensor([[0, 0, 0],
                                     [0, v, 0],
                                     [0, 0, 0]])
      v.initializer.run()
      self.assertAllEqual([[0, 0, 0], [0, 17, 0], [0, 0, 0]], result.eval())

      v.assign(38).op.run()
      self.assertAllEqual([[0, 0, 0], [0, 38, 0], [0, 0, 0]], result.eval())

  def testDtype(self):
    t_0 = tf.convert_to_tensor([[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.]])
    self.assertEqual(tf.float32, t_0.dtype)

    t_1 = tf.convert_to_tensor([[0., 0., 0.],
                                tf.constant([0., 0., 0.], dtype=tf.float64),
                                [0., 0., 0.]])
    self.assertEqual(tf.float64, t_1.dtype)

    t_2 = tf.convert_to_tensor([[0., 0., 0.],
                                [0., 0., 0.],
                                [0., 0., 0.]], dtype=tf.float64)
    self.assertEqual(tf.float64, t_2.dtype)

    with self.assertRaises(TypeError):
      tf.convert_to_tensor([tf.constant([0., 0., 0.], dtype=tf.float32),
                            tf.constant([0., 0., 0.], dtype=tf.float64),
                            [0., 0., 0.]])

    with self.assertRaises(TypeError):
      tf.convert_to_tensor([[0., 0., 0.],
                            tf.constant([0., 0., 0.], dtype=tf.float64),
                            [0., 0., 0.]], dtype=tf.float32)

    with self.assertRaises(TypeError):
      tf.convert_to_tensor([tf.constant([0., 0., 0.], dtype=tf.float64)],
                           dtype=tf.float32)

  def testPlaceholder(self):
    with self.test_session(use_gpu=True):
      # Test using placeholder with a defined shape.
      ph_0 = tf.placeholder(tf.int32, shape=[])
      result_0 = tf.convert_to_tensor([[0, 0, 0],
                                       [0, ph_0, 0],
                                       [0, 0, 0]])
      self.assertAllEqual([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]],
                          result_0.eval(feed_dict={ph_0: 1}))
      self.assertAllEqual([[0, 0, 0],
                           [0, 2, 0],
                           [0, 0, 0]],
                          result_0.eval(feed_dict={ph_0: 2}))

      # Test using placeholder with an undefined shape.
      ph_1 = tf.placeholder(tf.int32)
      result_1 = tf.convert_to_tensor([[0, 0, 0],
                                       [0, ph_1, 0],
                                       [0, 0, 0]])
      self.assertAllEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          result_1.eval(feed_dict={ph_1: 1}))
      self.assertAllEqual([[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                          result_1.eval(feed_dict={ph_1: 2}))

  def testShapeErrors(self):
    # Static shape error.
    ph_0 = tf.placeholder(tf.int32, shape=[1])
    with self.assertRaises(ValueError):
      tf.convert_to_tensor([[0, 0, 0], [0, ph_0, 0], [0, 0, 0]])

    # Dynamic shape error.
    ph_1 = tf.placeholder(tf.int32)
    result_1 = tf.convert_to_tensor([[0, 0, 0], [0, ph_1, 0], [0, 0, 0]])
    with self.test_session(use_gpu=True):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        result_1.eval(feed_dict={ph_1: [1]})


if __name__ == "__main__":
  tf.test.main()
