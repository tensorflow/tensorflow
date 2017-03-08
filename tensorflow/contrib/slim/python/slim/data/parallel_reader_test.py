# Copyright 2016 Google Inc. All Rights Reserved.
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
"""Tests for slim.data.parallel_reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.data import test_utils


class ParallelReaderTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def _verify_all_data_sources_read(self, shared_queue):
    with self.test_session():
      tfrecord_paths = test_utils.create_tfrecord_files(
          self.get_temp_dir(),
          num_files=3)

    num_readers = len(tfrecord_paths)
    p_reader = slim.parallel_reader.ParallelReader(
        tf.TFRecordReader,
        shared_queue,
        num_readers=num_readers)

    data_files = slim.parallel_reader.get_data_files(
        tfrecord_paths)
    filename_queue = tf.train.string_input_producer(data_files)
    key, value = p_reader.read(filename_queue)

    count0 = 0
    count1 = 0
    count2 = 0

    num_reads = 50

    sv = tf.train.Supervisor(logdir=self.get_temp_dir())
    with sv.prepare_or_wait_for_session() as sess:
      sv.start_queue_runners(sess)

      for _ in range(num_reads):
        current_key, _ = sess.run([key, value])
        if '0-of-3' in str(current_key):
          count0 += 1
        if '1-of-3' in str(current_key):
          count1 += 1
        if '2-of-3' in str(current_key):
          count2 += 1

    self.assertGreater(count0, 0)
    self.assertGreater(count1, 0)
    self.assertGreater(count2, 0)
    self.assertEquals(count0 + count1 + count2, num_reads)

  def testRandomShuffleQueue(self):
    shared_queue = tf.RandomShuffleQueue(capacity=256,
                                         min_after_dequeue=128,
                                         dtypes=[tf.string, tf.string])
    self._verify_all_data_sources_read(shared_queue)

  def testFIFOSharedQueue(self):
    shared_queue = tf.FIFOQueue(capacity=256, dtypes=[tf.string, tf.string])
    self._verify_all_data_sources_read(shared_queue)


class ParallelReadTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testTFRecordReader(self):
    with self.test_session():
      self._tfrecord_paths = test_utils.create_tfrecord_files(
          self.get_temp_dir(),
          num_files=3)

    key, value = slim.parallel_reader.parallel_read(
        self._tfrecord_paths,
        reader_class=tf.TFRecordReader,
        num_readers=3)

    sv = tf.train.Supervisor(logdir=self.get_temp_dir())
    with sv.prepare_or_wait_for_session() as sess:
      sv.start_queue_runners(sess)

      flowers = 0
      num_reads = 100
      for _ in range(num_reads):
        current_key, _ = sess.run([key, value])
        if 'flowers' in str(current_key):
          flowers += 1
      self.assertGreater(flowers, 0)
      self.assertEquals(flowers, num_reads)


class SinglePassReadTest(tf.test.TestCase):

  def setUp(self):
    tf.reset_default_graph()

  def testOutOfRangeError(self):
    with self.test_session():
      [tfrecord_path] = test_utils.create_tfrecord_files(
          self.get_temp_dir(),
          num_files=1)

    key, value = slim.parallel_reader.single_pass_read(
        tfrecord_path, reader_class=tf.TFRecordReader)
    init_op = tf.initialize_local_variables()

    with self.test_session() as sess:
      sess.run(init_op)
      with tf.contrib.slim.queues.QueueRunners(sess):
        num_reads = 11
        with self.assertRaises(tf.errors.OutOfRangeError):
          for _ in range(num_reads):
            sess.run([key, value])

  def testTFRecordReader(self):
    with self.test_session():
      [tfrecord_path] = test_utils.create_tfrecord_files(
          self.get_temp_dir(),
          num_files=1)

    key, value = slim.parallel_reader.single_pass_read(
        tfrecord_path, reader_class=tf.TFRecordReader)
    init_op = tf.initialize_local_variables()

    with self.test_session() as sess:
      sess.run(init_op)
      with tf.contrib.slim.queues.QueueRunners(sess):
        flowers = 0
        num_reads = 9
        for _ in range(num_reads):
          current_key, _ = sess.run([key, value])
          if 'flowers' in str(current_key):
            flowers += 1
        self.assertGreater(flowers, 0)
        self.assertEquals(flowers, num_reads)


if __name__ == '__main__':
  tf.test.main()
