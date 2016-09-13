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

"""Functional tests for Unpack Op."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


def np_split_sqeeze(array, axis):
  axis_len = array.shape[axis]
  return [
      np.squeeze(arr, axis=(axis,))
      for arr in np.split(array, axis_len, axis=axis)
  ]


class UnpackOpTest(tf.test.TestCase):

  def testSimple(self):
    np.random.seed(7)
    with self.test_session(use_gpu=True):
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        data = np.random.randn(*shape)
        # Convert data to a single tensorflow tensor
        x = tf.constant(data)
        # Unpack into a list of tensors
        cs = tf.unpack(x, num=shape[0])
        self.assertEqual(type(cs), list)
        self.assertEqual(len(cs), shape[0])
        cs = [c.eval() for c in cs]
        self.assertAllEqual(cs, data)

  def testGradientsAxis0(self):
    for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
      data = np.random.randn(*shape)
      shapes = [shape[1:]] * shape[0]
      for i in xrange(shape[0]):
        with self.test_session(use_gpu=True):
          x = tf.constant(data)
          cs = tf.unpack(x, num=shape[0])
          err = tf.test.compute_gradient_error(x, shape, cs[i], shapes[i])
          self.assertLess(err, 1e-6)

  def testGradientsAxis1(self):
    for shape in (2, 3), (3, 2), (4, 3, 2):
      data = np.random.randn(*shape)
      out_shape = list(shape)
      del out_shape[1]
      for i in xrange(shape[1]):
        with self.test_session(use_gpu=True):
          x = tf.constant(data)
          cs = tf.unpack(x, num=shape[1], axis=1)
          err = tf.test.compute_gradient_error(x, shape, cs[i], out_shape)
          self.assertLess(err, 1e-6)

  def testInferNum(self):
    with self.test_session():
      for shape in (2,), (3,), (2, 3), (3, 2), (4, 3, 2):
        x = tf.placeholder(np.float32, shape=shape)
        cs = tf.unpack(x)
        self.assertEqual(type(cs), list)
        self.assertEqual(len(cs), shape[0])

  def testCannotInferNumFromUnknownShape(self):
    x = tf.placeholder(np.float32)
    with self.assertRaisesRegexp(
        ValueError, r'Cannot infer num from shape <unknown>'):
      tf.unpack(x)

  def testUnknownShapeOkWithNum(self):
    x = tf.placeholder(np.float32)
    tf.unpack(x, num=2)

  def testCannotInferNumFromNoneShape(self):
    x = tf.placeholder(np.float32, shape=(None,))
    with self.assertRaisesRegexp(ValueError,
                                 r'Cannot infer num from shape \(\?,\)'):
      tf.unpack(x)

  def testAgainstNumpy(self):
    # For 1 to 5 dimensions.
    for i in range(1, 6):
      a = np.random.random(np.random.permutation(i) + 1)

      # For all the possible axis to split it, including negative indices.
      for j in range(-i, i):
        expected = np_split_sqeeze(a, j)

        with self.test_session() as sess:
          actual = sess.run(tf.unpack(a, axis=j))

        self.assertAllEqual(expected, actual)

  def testAxis0Default(self):
    with self.test_session() as sess:
      a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')

      unpacked = sess.run(tf.unpack(a))

    self.assertEqual(len(unpacked), 2)
    self.assertAllEqual(unpacked[0], [1, 2, 3])
    self.assertAllEqual(unpacked[1], [4, 5, 6])

  def testAxisOutOfRange(self):
    a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
    with self.assertRaisesRegexp(ValueError, r'axis = 2 not in \[-2, 2\)'):
      tf.unpack(a, axis=2)

  def testAxisOutOfNegativeRange(self):
    a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
    with self.assertRaisesRegexp(ValueError, r'axis = -3 not in \[-2, 2\)'):
      tf.unpack(a, axis=-3)

  def testZeroLengthDim(self):
    with self.test_session():
      x = tf.zeros(shape=(0, 1, 2))
      y = tf.unpack(x, axis=1)[0].eval()
      self.assertEqual(y.shape, (0, 2))


if __name__ == '__main__':
  tf.test.main()
