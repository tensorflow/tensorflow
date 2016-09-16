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
"""Tests for ShapeUtil."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.distributions.python.ops.shape import _DistributionShape
from tensorflow.python.framework import tensor_util


_empty_shape = np.array([], dtype=np.int32)


def _eval(x):
  if hasattr(x, "__iter__"):
    return [x.eval() for x in x]
  return x.eval()


def _constant(x):
  if hasattr(x, "__iter__"):
    return [tensor_util.constant_value(x) for x in x]
  return tensor_util.constant_value(x)


class DistributionShapeTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _random_sample(self, sample_shape, dtype=tf.float64):
    return self._rng.random_sample(sample_shape).astype(dtype.as_numpy_dtype())

  def _assertNdArrayEqual(self, expected, actual):
    """Helper which properly compares two np.ndarray-like objects.

    This function checks for exact equality so is probably only suitable for
    integers or powers of 2.

    Args:
      expected: np.ndarray. Ground-truth value.
      actual: np.ndarray.  Observed value.
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    self.assertEqual(
        expected.shape, actual.shape,
        "Shape mismatch: expected %s, got %s." % (expected.shape, actual.shape))
    actual_item = actual.flat
    for expected_item in expected.flat:
      self.assertAllEqual(expected_item, next(actual_item))

  def testDistributionShapeGetNdimsStatic(self):
    with self.test_session():
      shaper = _DistributionShape(batch_ndims=0, event_ndims=0)
      x = 1
      self.assertEqual(0, shaper.get_sample_ndims(x).eval())
      self.assertEqual(0, shaper.batch_ndims.eval())
      self.assertEqual(0, shaper.event_ndims.eval())

      shaper = _DistributionShape(batch_ndims=1, event_ndims=1)
      x = self._random_sample((1, 2, 3))
      self.assertAllEqual(3, shaper.get_ndims(x).eval())
      self.assertEqual(1, shaper.get_sample_ndims(x).eval())
      self.assertEqual(1, shaper.batch_ndims.eval())
      self.assertEqual(1, shaper.event_ndims.eval())

      x += self._random_sample((1, 2, 3))
      self.assertAllEqual(3, shaper.get_ndims(x).eval())
      self.assertEqual(1, shaper.get_sample_ndims(x).eval())
      self.assertEqual(1, shaper.batch_ndims.eval())
      self.assertEqual(1, shaper.event_ndims.eval())

      # Test ndims functions work, even despite unfed Tensors.
      y = tf.placeholder(tf.float32, shape=(1024, None, 1024))
      self.assertEqual(3, shaper.get_ndims(y).eval())
      self.assertEqual(1, shaper.get_sample_ndims(y).eval())
      self.assertEqual(1, shaper.batch_ndims.eval())
      self.assertEqual(1, shaper.event_ndims.eval())

  def testDistributionShapeGetNdimsDynamic(self):
    with self.test_session() as sess:
      batch_ndims = tf.placeholder(tf.int32)
      event_ndims = tf.placeholder(tf.int32)
      shaper = _DistributionShape(batch_ndims=batch_ndims,
                                  event_ndims=event_ndims)
      y = tf.placeholder(tf.float32)
      y_value = np.ones((4, 2), dtype=y.dtype.as_numpy_dtype())
      feed_dict = {y: y_value, batch_ndims: 1, event_ndims: 1}
      self.assertEqual(2, sess.run(shaper.get_ndims(y),
                                   feed_dict=feed_dict))

  def testDistributionShapeGetDimsStatic(self):
    with self.test_session():
      shaper = _DistributionShape(batch_ndims=0, event_ndims=0)
      shaper = _DistributionShape(batch_ndims=0, event_ndims=0)
      x = 1
      self.assertAllEqual((_empty_shape, _empty_shape, _empty_shape),
                          _constant(shaper.get_dims(x)))
      shaper = _DistributionShape(batch_ndims=1, event_ndims=2)
      x += self._random_sample((1, 1, 2, 2))
      self._assertNdArrayEqual(
          ([0], [1], [2, 3]),
          _constant(shaper.get_dims(x)))
      x += x
      self._assertNdArrayEqual(
          ([0], [1], [2, 3]),
          _constant(shaper.get_dims(x)))

  def testDistributionShapeGetDimsDynamic(self):
    with self.test_session() as sess:
      # Works for static {batch,event}_ndims despite unfed input.
      shaper = _DistributionShape(batch_ndims=1, event_ndims=2)
      y = tf.placeholder(tf.float32, shape=(10, None, 5, 5))
      self._assertNdArrayEqual([[0], [1], [2, 3]], _eval(shaper.get_dims(y)))

      # Works for deferred {batch,event}_ndims.
      batch_ndims = tf.placeholder(tf.int32)
      event_ndims = tf.placeholder(tf.int32)
      shaper = _DistributionShape(batch_ndims=batch_ndims,
                                  event_ndims=event_ndims)
      y = tf.placeholder(tf.float32)
      y_value = self._random_sample((10, 3, 5, 5), dtype=y.dtype)
      feed_dict = {y: y_value, batch_ndims: 1, event_ndims: 2}
      self._assertNdArrayEqual(
          ([0], [1], [2, 3]),
          sess.run(shaper.get_dims(y), feed_dict=feed_dict))

  def testDistributionShapeGetShapeStatic(self):
    with self.test_session():
      shaper = _DistributionShape(batch_ndims=0, event_ndims=0)
      self.assertAllEqual((_empty_shape, _empty_shape, _empty_shape),
                          _constant(shaper.get_shape(1.)))
      self._assertNdArrayEqual(([1], _empty_shape, _empty_shape),
                               _constant(shaper.get_shape(np.ones(1))))
      self._assertNdArrayEqual(([2, 2], _empty_shape, _empty_shape),
                               _constant(shaper.get_shape(np.ones((2, 2)))))
      self._assertNdArrayEqual(([3, 2, 1], _empty_shape, _empty_shape),
                               _constant(shaper.get_shape(np.ones((3, 2, 1)))))

      shaper = _DistributionShape(batch_ndims=0, event_ndims=1)
      with self.assertRaisesRegexp(ValueError, "expected .* <= ndims"):
        shaper.get_shape(1.)
      self._assertNdArrayEqual((_empty_shape, _empty_shape, [1]),
                               _constant(shaper.get_shape(np.ones(1))))
      self._assertNdArrayEqual(([2], _empty_shape, [2]),
                               _constant(shaper.get_shape(np.ones((2, 2)))))
      self._assertNdArrayEqual(([3, 2], _empty_shape, [1]),
                               _constant(shaper.get_shape(np.ones((3, 2, 1)))))

      shaper = _DistributionShape(batch_ndims=1, event_ndims=0)
      with self.assertRaisesRegexp(ValueError, "expected .* <= ndims"):
        shaper.get_shape(1.)
      self._assertNdArrayEqual((_empty_shape, [1], _empty_shape),
                               _constant(shaper.get_shape(np.ones(1))))
      self._assertNdArrayEqual(([2], [2], _empty_shape),
                               _constant(shaper.get_shape(np.ones((2, 2)))))
      self._assertNdArrayEqual(([3, 2], [1], _empty_shape),
                               _constant(shaper.get_shape(np.ones((3, 2, 1)))))

      shaper = _DistributionShape(batch_ndims=1, event_ndims=1)
      with self.assertRaisesRegexp(ValueError, "expected .* <= ndims"):
        shaper.get_shape(1.)
      with self.assertRaisesRegexp(ValueError, "expected .* <= ndims"):
        shaper.get_shape(np.ones(1))
      self._assertNdArrayEqual((_empty_shape, [2], [2]),
                               _constant(shaper.get_shape(np.ones((2, 2)))))
      self._assertNdArrayEqual(([3], [2], [1]),
                               _constant(shaper.get_shape(np.ones((3, 2, 1)))))

  def testDistributionShapeGetShapeDynamic(self):
    with self.test_session() as sess:
      # Works for static ndims despite unknown static shape.
      shaper = _DistributionShape(batch_ndims=1, event_ndims=1)
      y = tf.placeholder(tf.int32, shape=(None, None, 2))
      y_value = np.ones((3, 4, 2), dtype=y.dtype.as_numpy_dtype())
      self._assertNdArrayEqual(
          ([3], [4], [2]),
          sess.run(shaper.get_shape(y), feed_dict={y: y_value}))

      shaper = _DistributionShape(batch_ndims=0, event_ndims=1)
      y = tf.placeholder(tf.int32, shape=(None, None))
      y_value = np.ones((3, 2), dtype=y.dtype.as_numpy_dtype())
      self._assertNdArrayEqual(
          ([3], _empty_shape, [2]),
          sess.run(shaper.get_shape(y), feed_dict={y: y_value}))

      # Works for deferred {batch,event}_ndims.
      batch_ndims = tf.placeholder(tf.int32)
      event_ndims = tf.placeholder(tf.int32)
      shaper = _DistributionShape(batch_ndims=batch_ndims,
                                  event_ndims=event_ndims)
      y = tf.placeholder(tf.float32)
      y_value = self._random_sample((3, 4, 2), dtype=y.dtype)
      feed_dict = {y: y_value, batch_ndims: 1, event_ndims: 1}
      self._assertNdArrayEqual(
          ([3], [4], [2]),
          sess.run(shaper.get_shape(y), feed_dict=feed_dict))

      y_value = self._random_sample((3, 2), dtype=y.dtype)
      feed_dict = {y: y_value, batch_ndims: 0, event_ndims: 1}
      self._assertNdArrayEqual(
          ([3], _empty_shape, [2]),
          sess.run(shaper.get_shape(y), feed_dict=feed_dict))

  def testDistributionShapeMakeBatchReadyStatic(self):
    with self.test_session() as sess:
      x = self._random_sample((1, 2, 3))
      shaper = _DistributionShape(batch_ndims=1, event_ndims=1)
      y, sample_shape = shaper.make_batch_of_event_sample_matrices(x)
      self.assertAllEqual(np.transpose(x, axes=(1, 2, 0)), y.eval())
      self.assertAllEqual((1,), sample_shape.eval())
      should_be_x_value = shaper.undo_make_batch_of_event_sample_matrices(
          y, sample_shape)
      self.assertAllEqual(x, should_be_x_value.eval())

      shaper = _DistributionShape(batch_ndims=1, event_ndims=1)
      x = tf.placeholder(tf.float32)
      x_value = self._random_sample((3, 4, 2), dtype=x.dtype)
      feed_dict = {x: x_value}
      y, sample_shape = shaper.make_batch_of_event_sample_matrices(x)
      self.assertAllEqual(
          (3,),
          sess.run(sample_shape, feed_dict=feed_dict))
      self.assertAllClose(
          np.transpose(np.reshape(x_value, (-1, 4, 2)), (1, 2, 0)),
          sess.run(y, feed_dict=feed_dict),
          rtol=1e-3)
      should_be_x_value = shaper.undo_make_batch_of_event_sample_matrices(
          y, sample_shape)
      self.assertAllEqual(x_value, sess.run(should_be_x_value,
                                            feed_dict=feed_dict))

      shaper = _DistributionShape(batch_ndims=0, event_ndims=0)
      x = tf.placeholder(tf.float32)
      x_value = np.ones((3,), dtype=x.dtype.as_numpy_dtype())
      feed_dict = {x: x_value}
      y, sample_shape = shaper.make_batch_of_event_sample_matrices(x)
      self.assertAllEqual(
          (3,),
          sess.run(sample_shape, feed_dict=feed_dict))
      # The following check shows we don't need to manually set_shape in the
      # ShapeUtil.
      self.assertAllEqual((1, 1, None),
                          y.get_shape().ndims and y.get_shape().as_list())
      self.assertAllEqual(
          np.ones((1, 1, 3), dtype=x.dtype.as_numpy_dtype()),
          sess.run(y, feed_dict=feed_dict))
      should_be_x_value = shaper.undo_make_batch_of_event_sample_matrices(
          y, sample_shape)
      self.assertAllEqual(x_value, sess.run(should_be_x_value,
                                            feed_dict=feed_dict))

  def testDistributionShapeMakeBatchReadyDynamic(self):
    with self.test_session() as sess:
      shaper = _DistributionShape(batch_ndims=1, event_ndims=1)
      x = tf.placeholder(tf.float32, shape=(1, 2, 3))
      x_value = self._random_sample(x.get_shape().as_list(), dtype=x.dtype)
      y, sample_shape = sess.run(
          shaper.make_batch_of_event_sample_matrices(x),
          feed_dict={x: x_value})
      self.assertAllEqual(np.transpose(x_value, (1, 2, 0)), y)
      self.assertAllEqual((1,), sample_shape)

      feed_dict = {x: x_value}
      y, sample_shape = shaper.make_batch_of_event_sample_matrices(x)
      self.assertAllEqual(
          (1,),
          sess.run(sample_shape, feed_dict=feed_dict))
      self.assertAllEqual(
          np.transpose(x_value, (1, 2, 0)),
          sess.run(y, feed_dict=feed_dict))
      should_be_x_value = shaper.undo_make_batch_of_event_sample_matrices(
          y, sample_shape)
      self.assertAllEqual(x_value, sess.run(should_be_x_value,
                                            feed_dict=feed_dict))

      batch_ndims = tf.placeholder(tf.int32)
      event_ndims = tf.placeholder(tf.int32)
      shaper = _DistributionShape(batch_ndims=batch_ndims,
                                  event_ndims=event_ndims)

      # batch_ndims = 1, event_ndims = 1.
      x = tf.placeholder(tf.float32)
      x_value = np.ones((3, 4, 2), dtype=x.dtype.as_numpy_dtype())
      feed_dict = {x: x_value, batch_ndims: 1, event_ndims: 1}
      y, sample_shape = shaper.make_batch_of_event_sample_matrices(x)
      self.assertAllEqual(
          (3,),
          sess.run(sample_shape, feed_dict=feed_dict))
      self.assertAllEqual(
          np.ones((4, 2, 3), dtype=x.dtype.as_numpy_dtype()),
          sess.run(y, feed_dict=feed_dict))
      should_be_x_value = shaper.undo_make_batch_of_event_sample_matrices(
          y, sample_shape)
      self.assertAllEqual(x_value, sess.run(should_be_x_value,
                                            feed_dict=feed_dict))

      # batch_ndims = 0, event_ndims = 0.
      x_value = np.ones((3,), dtype=x.dtype.as_numpy_dtype())
      feed_dict = {x: x_value, batch_ndims: 0, event_ndims: 0}
      y, sample_shape = shaper.make_batch_of_event_sample_matrices(x)
      self.assertAllEqual(
          (3,),
          sess.run(sample_shape, feed_dict=feed_dict))
      self.assertAllEqual(
          np.ones((1, 1, 3), dtype=x.dtype.as_numpy_dtype()),
          sess.run(y, feed_dict=feed_dict))
      should_be_x_value = shaper.undo_make_batch_of_event_sample_matrices(
          y, sample_shape)
      self.assertAllEqual(x_value, sess.run(should_be_x_value,
                                            feed_dict=feed_dict))

      # batch_ndims = 0, event_ndims = 1.
      x_value = np.ones((1, 2,), dtype=x.dtype.as_numpy_dtype())
      feed_dict = {x: x_value, batch_ndims: 0, event_ndims: 1}
      y, sample_shape = shaper.make_batch_of_event_sample_matrices(x)
      self.assertAllEqual(
          (1,),
          sess.run(sample_shape, feed_dict=feed_dict))
      self.assertAllEqual(
          np.ones((1, 2, 1), dtype=x.dtype.as_numpy_dtype()),
          sess.run(y, feed_dict=feed_dict))
      should_be_x_value = shaper.undo_make_batch_of_event_sample_matrices(
          y, sample_shape)
      self.assertAllEqual(x_value, sess.run(should_be_x_value,
                                            feed_dict=feed_dict))

      # batch_ndims = 1, event_ndims = 0.
      x_value = np.ones((1, 2), dtype=x.dtype.as_numpy_dtype())
      feed_dict = {x: x_value, batch_ndims: 1, event_ndims: 0}
      y, sample_shape = shaper.make_batch_of_event_sample_matrices(x)
      self.assertAllEqual(
          (1,),
          sess.run(sample_shape, feed_dict=feed_dict))
      self.assertAllEqual(
          np.ones((2, 1, 1), dtype=x.dtype.as_numpy_dtype()),
          sess.run(y, feed_dict=feed_dict))
      should_be_x_value = shaper.undo_make_batch_of_event_sample_matrices(
          y, sample_shape)
      self.assertAllEqual(x_value, sess.run(should_be_x_value,
                                            feed_dict=feed_dict))


if __name__ == "__main__":
  tf.test.main()
