# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for slim.data.prefetch_queue."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim


class PrefetchQueueTest(tf.test.TestCase):

  def testOneThread(self):
    with self.test_session() as sess:
      batch_size = 10
      image_size = 32
      num_batches = 5

      zero64 = tf.constant(0, dtype=tf.int64)

      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      image = tf.random_normal([image_size, image_size, 3],
                               dtype=tf.float32,
                               name='images')
      label = tf.random_uniform([1], 0, 10, dtype=tf.int32, name='labels')

      batches = tf.train.batch([counter, image, label],
                               batch_size=batch_size,
                               num_threads=1)

      batches = slim.prefetch_queue.prefetch_queue(
          batches).dequeue()

      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      for i in range(num_batches):
        results = sess.run(batches)
        self.assertAllEqual(results[0], np.arange(i * batch_size,
                                                  (i + 1) * batch_size))
        self.assertEquals(results[1].shape,
                          (batch_size, image_size, image_size, 3))
        self.assertEquals(results[2].shape, (batch_size, 1))

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batches)
      for thread in threads:
        thread.join()

  def testMultiThread(self):
    with self.test_session() as sess:
      batch_size = 10
      image_size = 32
      num_batches = 5

      zero64 = tf.constant(0, dtype=tf.int64)

      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      image = tf.random_normal([image_size, image_size, 3],
                               dtype=tf.float32,
                               name='images')
      label = tf.random_uniform([1], 0, 10, dtype=tf.int32, name='labels')

      batches = tf.train.batch([counter, image, label],
                               batch_size=batch_size,
                               num_threads=4)

      batches = slim.prefetch_queue.prefetch_queue(
          batches).dequeue()

      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      value_counter = []
      for _ in range(num_batches):
        results = sess.run(batches)
        value_counter.append(results[0])
        self.assertEqual(results[1].shape,
                         (batch_size, image_size, image_size, 3))
        self.assertEqual(results[2].shape, (batch_size, 1))

      self.assertAllEqual(np.sort(np.concatenate(value_counter)),
                          np.arange(0, num_batches * batch_size))
      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batches)
      for thread in threads:
        thread.join()

  def testMultipleDequeue(self):
    with self.test_session() as sess:
      batch_size = 10
      image_size = 32
      num_batches = 4

      zero64 = tf.constant(0, dtype=tf.int64)

      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      image = tf.random_normal([image_size, image_size, 3],
                               dtype=tf.float32,
                               name='images')
      label = tf.random_uniform([1], 0, 10, dtype=tf.int32, name='labels')

      batches = tf.train.batch([counter, image, label],
                               batch_size=batch_size,
                               num_threads=4)

      batcher = slim.prefetch_queue.prefetch_queue(batches)
      batches_list = [batcher.dequeue() for _ in range(2)]

      tf.initialize_all_variables().run()
      threads = tf.train.start_queue_runners()

      value_counter = []
      for _ in range(int(num_batches/2)):
        for batches in batches_list:
          results = sess.run(batches)
          value_counter.append(results[0])
          self.assertEquals(results[1].shape,
                            (batch_size, image_size, image_size, 3))
          self.assertEquals(results[2].shape, (batch_size, 1))

      self.assertAllEqual(np.sort(np.concatenate(value_counter)),
                          np.arange(0, num_batches * batch_size))
      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batches)
      for thread in threads:
        thread.join()

  def testDictConstruction(self):
    with tf.Graph().as_default():
      batches = {'first': tf.constant([1]), 'second': tf.constant([2.0, 2.1])}
      prefetcher = slim.prefetch_queue.prefetch_queue(batches)
      dequeued = prefetcher.dequeue()
      self.assertTrue(isinstance(dequeued, dict))
      self.assertEqual(2, len(dequeued))
      self.assertEqual(tf.int32, dequeued['first'].dtype)
      self.assertEqual(tf.float32, dequeued['second'].dtype)


if __name__ == '__main__':
  tf.test.main()
