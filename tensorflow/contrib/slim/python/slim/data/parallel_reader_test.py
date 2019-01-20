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
"""Tests for slim.data.parallel_reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.slim.python.slim import queues
from tensorflow.contrib.slim.python.slim.data import parallel_reader
from tensorflow.contrib.slim.python.slim.data import test_utils
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import input as input_lib
from tensorflow.python.training import supervisor


class ParallelReaderTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  def _verify_all_data_sources_read(self, shared_queue):
    with self.cached_session():
      tfrecord_paths = test_utils.create_tfrecord_files(
          self.get_temp_dir(), num_files=3)

    num_readers = len(tfrecord_paths)
    p_reader = parallel_reader.ParallelReader(
        io_ops.TFRecordReader, shared_queue, num_readers=num_readers)

    data_files = parallel_reader.get_data_files(tfrecord_paths)
    filename_queue = input_lib.string_input_producer(data_files)
    key, value = p_reader.read(filename_queue)

    count0 = 0
    count1 = 0
    count2 = 0

    num_reads = 50

    sv = supervisor.Supervisor(logdir=self.get_temp_dir())
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

  def _verify_read_up_to_out(self, shared_queue):
    with self.cached_session():
      num_files = 3
      num_records_per_file = 7
      tfrecord_paths = test_utils.create_tfrecord_files(
          self.get_temp_dir(),
          num_files=num_files,
          num_records_per_file=num_records_per_file)

    p_reader = parallel_reader.ParallelReader(
        io_ops.TFRecordReader, shared_queue, num_readers=5)

    data_files = parallel_reader.get_data_files(tfrecord_paths)
    filename_queue = input_lib.string_input_producer(data_files, num_epochs=1)
    key, value = p_reader.read_up_to(filename_queue, 4)

    count0 = 0
    count1 = 0
    count2 = 0
    all_keys_count = 0
    all_values_count = 0

    sv = supervisor.Supervisor(logdir=self.get_temp_dir())
    with sv.prepare_or_wait_for_session() as sess:
      sv.start_queue_runners(sess)
      while True:
        try:
          current_keys, current_values = sess.run([key, value])
          self.assertEquals(len(current_keys), len(current_values))
          all_keys_count += len(current_keys)
          all_values_count += len(current_values)
          for current_key in current_keys:
            if '0-of-3' in str(current_key):
              count0 += 1
            if '1-of-3' in str(current_key):
              count1 += 1
            if '2-of-3' in str(current_key):
              count2 += 1
        except errors_impl.OutOfRangeError:
          break

    self.assertEquals(count0, num_records_per_file)
    self.assertEquals(count1, num_records_per_file)
    self.assertEquals(count2, num_records_per_file)
    self.assertEquals(
        all_keys_count,
        num_files * num_records_per_file)
    self.assertEquals(all_values_count, all_keys_count)
    self.assertEquals(
        count0 + count1 + count2,
        all_keys_count)

  def testRandomShuffleQueue(self):
    shared_queue = data_flow_ops.RandomShuffleQueue(
        capacity=256,
        min_after_dequeue=128,
        dtypes=[dtypes_lib.string, dtypes_lib.string])
    self._verify_all_data_sources_read(shared_queue)

  def testFIFOSharedQueue(self):
    shared_queue = data_flow_ops.FIFOQueue(
        capacity=256, dtypes=[dtypes_lib.string, dtypes_lib.string])
    self._verify_all_data_sources_read(shared_queue)

  def testReadUpToFromRandomShuffleQueue(self):
    shared_queue = data_flow_ops.RandomShuffleQueue(
        capacity=55,
        min_after_dequeue=28,
        dtypes=[dtypes_lib.string, dtypes_lib.string],
        shapes=[tensor_shape.scalar(), tensor_shape.scalar()])
    self._verify_read_up_to_out(shared_queue)

  def testReadUpToFromFIFOQueue(self):
    shared_queue = data_flow_ops.FIFOQueue(
        capacity=99,
        dtypes=[dtypes_lib.string, dtypes_lib.string],
        shapes=[tensor_shape.scalar(), tensor_shape.scalar()])
    self._verify_read_up_to_out(shared_queue)


class ParallelReadTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  def testTFRecordReader(self):
    with self.cached_session():
      self._tfrecord_paths = test_utils.create_tfrecord_files(
          self.get_temp_dir(), num_files=3)

    key, value = parallel_reader.parallel_read(
        self._tfrecord_paths, reader_class=io_ops.TFRecordReader, num_readers=3)

    sv = supervisor.Supervisor(logdir=self.get_temp_dir())
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


class SinglePassReadTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

  def testOutOfRangeError(self):
    with self.cached_session():
      [tfrecord_path] = test_utils.create_tfrecord_files(
          self.get_temp_dir(), num_files=1)

    key, value = parallel_reader.single_pass_read(
        tfrecord_path, reader_class=io_ops.TFRecordReader)
    init_op = variables.local_variables_initializer()

    with self.cached_session() as sess:
      sess.run(init_op)
      with queues.QueueRunners(sess):
        num_reads = 11
        with self.assertRaises(errors_impl.OutOfRangeError):
          for _ in range(num_reads):
            sess.run([key, value])

  def testTFRecordReader(self):
    with self.cached_session():
      [tfrecord_path] = test_utils.create_tfrecord_files(
          self.get_temp_dir(), num_files=1)

    key, value = parallel_reader.single_pass_read(
        tfrecord_path, reader_class=io_ops.TFRecordReader)
    init_op = variables.local_variables_initializer()

    with self.cached_session() as sess:
      sess.run(init_op)
      with queues.QueueRunners(sess):
        flowers = 0
        num_reads = 9
        for _ in range(num_reads):
          current_key, _ = sess.run([key, value])
          if 'flowers' in str(current_key):
            flowers += 1
        self.assertGreater(flowers, 0)
        self.assertEquals(flowers, num_reads)


if __name__ == '__main__':
  test.main()
