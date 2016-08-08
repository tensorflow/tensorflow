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
from tensorflow.contrib.distributions.python.ops.shape import _ShapeUtil  # pylint: disable=line-too-long


class ShapeUtilTest(tf.test.TestCase):

  def testShapeUtilGetNdims(self):
    with self.test_session():
      shaper = _ShapeUtil(batch_ndims=0, event_ndims=0)
      x = 1
      self.assertEqual(shaper.get_sample_ndims(x), 0)
      self.assertEqual(shaper.batch_ndims, 0)
      self.assertEqual(shaper.event_ndims, 0)

      shaper = _ShapeUtil(batch_ndims=1, event_ndims=1)
      x = [[[0., 1, 2], [3, 4, 5]]]
      self.assertAllEqual(shaper.get_ndims(x), 3)
      self.assertEqual(shaper.get_sample_ndims(x), 1)
      self.assertEqual(shaper.batch_ndims, 1)
      self.assertEqual(shaper.event_ndims, 1)

      x += [[[6, 7, 8], [9, 10, 11]]]
      self.assertAllEqual(shaper.get_ndims(x), 3)
      self.assertEqual(shaper.get_sample_ndims(x), 1)
      self.assertEqual(shaper.batch_ndims, 1)
      self.assertEqual(shaper.event_ndims, 1)

      # Test ndims functions work, even despite unfed Tensors.
      y = tf.placeholder(tf.float32, shape=(1024, None, 1024))
      self.assertAllEqual(shaper.get_ndims(y), 3)
      self.assertEqual(shaper.get_sample_ndims(y), 1)
      self.assertEqual(shaper.batch_ndims, 1)
      self.assertEqual(shaper.event_ndims, 1)

      with self.assertRaises(ValueError):
        y = tf.placeholder(tf.float32)
        shaper.get_ndims(y)

  def testShapeUtilGetDims(self):
    with self.test_session():
      shaper = _ShapeUtil(batch_ndims=0, event_ndims=0)
      with self.assertRaises(ValueError):
        y = tf.placeholder(tf.float32)
        shaper.get_sample_dims(y)
      with self.assertRaises(ValueError):
        y = tf.placeholder(tf.float32)
        shaper.get_batch_dims(y)
      with self.assertRaises(ValueError):
        y = tf.placeholder(tf.float32)
        shaper.get_event_dims(y)

      shaper = _ShapeUtil(batch_ndims=0, event_ndims=0)
      x = 1
      self.assertAllEqual(shaper.get_sample_dims(x), [])
      self.assertAllEqual(shaper.get_batch_dims(x), [])
      self.assertAllEqual(shaper.get_event_dims(x), [])
      self.assertAllEqual(shaper.get_dims(x, sample=False), [])

      shaper = _ShapeUtil(batch_ndims=1, event_ndims=2)
      x = [[[[0., 1], [2, 4]]]]
      self.assertAllEqual(shaper.get_sample_dims(x), [0])
      self.assertAllEqual(shaper.get_batch_dims(x), [1])
      self.assertAllEqual(shaper.get_event_dims(x), [2, 3])
      self.assertAllEqual(shaper.get_dims(x, sample=False), [1, 2, 3])

      x += x
      self.assertAllEqual(shaper.get_sample_dims(x), [0])
      self.assertAllEqual(shaper.get_batch_dims(x), [1])
      self.assertAllEqual(shaper.get_event_dims(x), [2, 3])
      self.assertAllEqual(shaper.get_dims(x, sample=False), [1, 2, 3])

      # Test dims functions work, despite unfed Tensors.
      y = tf.placeholder(tf.float32, shape=(1024, None, 5, 5))
      self.assertAllEqual(shaper.get_sample_dims(y), [0])
      self.assertAllEqual(shaper.get_batch_dims(y), [1])
      self.assertAllEqual(shaper.get_event_dims(y), [2, 3])

  def testShapeUtilGetShape(self):
    with self.test_session() as sess:
      shaper = _ShapeUtil(batch_ndims=0, event_ndims=0)
      with self.assertRaises(ValueError):
        y = tf.placeholder(tf.float32)
        shaper.get_sample_shape(y)
      with self.assertRaises(ValueError):
        y = tf.placeholder(tf.float32)
        shaper.get_batch_shape(y)
      with self.assertRaises(ValueError):
        y = tf.placeholder(tf.float32)
        shaper.get_event_shape(y)

      shaper = _ShapeUtil(batch_ndims=0, event_ndims=0)
      x = 1
      self.assertAllEqual(shaper.get_sample_shape(x), [])
      self.assertAllEqual(shaper.get_batch_shape(x), [])
      self.assertAllEqual(shaper.get_event_shape(x), [])
      self.assertAllEqual(shaper.get_shape(x, batch=False), [])

      shaper = _ShapeUtil(batch_ndims=1, event_ndims=1)
      x = [[[0., 1, 2], [3, 4, 5]]]
      self.assertAllEqual(shaper.get_sample_shape(x), [1])
      self.assertAllEqual(shaper.get_batch_shape(x), [2])
      self.assertAllEqual(shaper.get_event_shape(x), [3])
      self.assertAllEqual(shaper.get_shape(x, batch=False), [1, 3])

      x += [[[6, 7, 8], [9, 10, 11]]]
      self.assertAllEqual(shaper.get_sample_shape(x), [2])
      self.assertAllEqual(shaper.get_batch_shape(x), [2])
      self.assertAllEqual(shaper.get_event_shape(x), [3])
      self.assertAllEqual(shaper.get_shape(x, batch=False), [2, 3])

      shaper = _ShapeUtil(batch_ndims=0, event_ndims=1)
      x = tf.ones((3, 2))
      self.assertAllEqual(shaper.get_shape(x, sample=False), (2,))

      def feed_eval(fun, build_shape=(None, None, 2), graph_shape=(3, 4, 2)):
        """Helper to use a deferred-shape tensor eval'ed at graph runtime."""
        y = tf.placeholder(tf.int32, shape=build_shape)
        y_value = np.ones(graph_shape, dtype=y.dtype.as_numpy_dtype())
        return sess.run(fun(y),
                        feed_dict={y: y_value})

      shaper = _ShapeUtil(batch_ndims=1, event_ndims=1)
      self.assertAllEqual(feed_eval(shaper.get_sample_shape), [3])
      self.assertAllEqual(feed_eval(shaper.get_batch_shape), [4])
      self.assertAllEqual(feed_eval(shaper.get_event_shape), [2])
      self.assertAllEqual(
          feed_eval(lambda y: shaper.get_shape(y, batch=False)),
          [3, 2])

      shaper = _ShapeUtil(batch_ndims=0, event_ndims=1)
      self.assertAllEqual(
          feed_eval(lambda y: shaper.get_shape(y, batch=False),
                    (None, None),
                    (3, 2)),
          [3, 2])
      self.assertAllEqual(
          feed_eval(lambda y: shaper.get_shape(y, sample=False),
                    (None, None),
                    (3, 2)),
          [2])


if __name__ == "__main__":
  tf.test.main()
