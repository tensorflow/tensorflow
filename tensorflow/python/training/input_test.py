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

"""Tests for training.input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.training import input as inp


class MatchFilenamesOnceTest(tf.test.TestCase):

  def test(self):
    temp_dir = self.get_temp_dir()
    filenames = [os.path.join(temp_dir, n) for n in os.listdir(temp_dir)]
    additional = [os.path.join(self.get_temp_dir(), "match_filenames.%d" % i)
                  for i in range(3)]
    for name in additional:
      open(name, "w").write("Some contents")
    filenames = list(set(filenames + additional))
    with self.test_session():
      star = tf.train.match_filenames_once(
          os.path.join(self.get_temp_dir(), "*"))
      question = tf.train.match_filenames_once(
          os.path.join(self.get_temp_dir(), "match_filenames.?"))
      one = tf.train.match_filenames_once(additional[1])
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      self.assertItemsEqual(map(tf.compat.as_bytes, filenames), star.eval())
      self.assertItemsEqual(map(tf.compat.as_bytes, additional),
                            question.eval())
      self.assertItemsEqual([tf.compat.as_bytes(additional[1])], one.eval())


class LimitEpochsTest(tf.test.TestCase):

  def testNoLimit(self):
    with self.test_session():
      seven = tf.constant(7)
      seven_forever = tf.train.limit_epochs(seven)
      tf.local_variables_initializer().run()
      for _ in range(100):
        self.assertEqual(7, seven_forever.eval())

  def testLimit(self):
    with self.test_session():
      love_me = tf.constant("Love Me")
      love_me_two_times = tf.train.limit_epochs(love_me, num_epochs=2)
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      self.assertEqual(b"Love Me", love_me_two_times.eval())
      self.assertEqual(b"Love Me", love_me_two_times.eval())
      with self.assertRaises(tf.errors.OutOfRangeError):
        love_me_two_times.eval()


class InputProducerTest(tf.test.TestCase):

  def testNoShuffle(self):
    with self.test_session():
      input_tensor = [[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]]
      num_epochs = 2
      queue = tf.train.input_producer(
          input_tensor, num_epochs=num_epochs, shuffle=False)
      dequeue_many = queue.dequeue_many(len(input_tensor) * num_epochs)
      dequeue = queue.dequeue()
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      self.assertAllEqual(input_tensor * num_epochs, dequeue_many.eval())

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def testNoShapeInference(self):
    with self.test_session():
      # Disable shape inference for the input.
      input_value = [[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]]
      input_tensor = tf.placeholder_with_default(input_value, shape=None)
      num_epochs = 2
      queue = tf.train.input_producer(
          input_tensor, element_shape=[4], num_epochs=num_epochs, shuffle=False)
      dequeue_many = queue.dequeue_many(len(input_value) * num_epochs)
      dequeue = queue.dequeue()
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      self.assertAllEqual(input_value * num_epochs, dequeue_many.eval())

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def testShapeError(self):
    input_tensor = tf.placeholder(tf.float32, None)
    with self.assertRaisesRegexp(ValueError, "fully defined shape"):
      _ = tf.train.input_producer(input_tensor)


class StringInputProducerTest(tf.test.TestCase):

  def testNoShuffle(self):
    with self.test_session():
      strings = [b"to", b"be", b"or", b"not", b"to", b"be"]
      num_epochs = 3
      queue = tf.train.string_input_producer(
          strings, num_epochs=num_epochs, shuffle=False)
      dequeue_many = queue.dequeue_many(len(strings) * num_epochs)
      dequeue = queue.dequeue()
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      output = dequeue_many.eval()
      self.assertAllEqual(strings * num_epochs, output)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def testShuffle(self):
    with self.test_session():
      strings = [b"a", b"b", b"c"]
      num_epochs = 600
      queue = tf.train.string_input_producer(
          strings, num_epochs=num_epochs, shuffle=True, seed=271828)
      dequeue_many = queue.dequeue_many(len(strings))
      dequeue = queue.dequeue()
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # Validate that we only shuffle the strings within an epoch and
      # count how often each possible order appears.
      expected = [b"abc", b"acb", b"bac", b"bca", b"cab", b"cba"]
      frequency = {}
      for e in expected:
        frequency[e] = 0
      for _ in range(num_epochs):
        output = dequeue_many.eval()
        key = b"".join(output)
        self.assertIn(key, expected)
        frequency[key] += 1

      # Expect an approximately even distribution over all possible orders.
      expected_frequency = num_epochs / len(expected)
      margin = expected_frequency * 0.4
      tf.logging.info("Observed counts: %s", frequency)
      for key in expected:
        value = frequency[key]
        self.assertGreater(value, expected_frequency - margin)
        self.assertLess(value, expected_frequency + margin)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def testNullStringPython(self):
    # Graph-construction time check for empty string list:
    with self.test_session():
      with self.assertRaises(ValueError):
        _ = tf.train.string_input_producer([])

  def testNullString(self):
    # Runtime check for empty string list.  This is slightly oblique:
    # The queue runner should die with an assertion error on the null
    # input tensor, causing the dequeue to fail with an OutOfRangeError.
    with self.test_session():
      coord = tf.train.Coordinator()
      queue = tf.train.string_input_producer(tf.constant([], dtype=tf.string))
      dequeue = queue.dequeue()
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners(coord=coord)
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      coord.request_stop()
      for thread in threads:
        thread.join()

  def testSharedName(self):
    with self.test_session():
      strings = [b"to", b"be", b"or", b"not", b"to", b"be"]
      queue = tf.train.string_input_producer(
          strings, shared_name="SHARED_NAME_XYZ", name="Q")
      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          queue.queue_ref.op.node_def.attr["shared_name"])

  def testConstructionRace(self):
    with self.test_session() as sess:
      strings = [b"to", b"be", b"or", b"not", b"to", b"be"]
      queue = tf.train.string_input_producer(strings, shuffle=False)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      for _ in range(2):
        for string in strings:
          # NOTE(mrry): This is not the recommended way to write
          # dequeuing code (instead you should create a single dequeue
          # op before starting the queue runners, and run it
          # repeatedly), because it leads to concurrent reading and
          # writing of the `tf.Graph` object. However, many users
          # write code this way, so we include this test to ensure
          # that we can support it.
          self.assertEquals(string, sess.run(queue.dequeue()))
      coord.request_stop()
      coord.join(threads)


class RangeInputProducerTest(tf.test.TestCase):

  def testNoShuffle(self):
    with self.test_session():
      num_epochs = 3
      range_size = 5
      queue = tf.train.range_input_producer(
          range_size, num_epochs=num_epochs, shuffle=False)
      dequeue_many = queue.dequeue_many(range_size * num_epochs)
      dequeue = queue.dequeue()
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      output = dequeue_many.eval()
      self.assertAllEqual(list(xrange(range_size)) * num_epochs, output)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def testShuffle(self):
    with self.test_session():
      num_epochs = 200
      range_size = 2
      queue = tf.train.range_input_producer(
          range_size, num_epochs=num_epochs, shuffle=True, seed=314159)
      dequeue_many = queue.dequeue_many(range_size)
      dequeue = queue.dequeue()
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # Validate that we only shuffle the integers within an epoch and
      # count how often each possible order appears.
      expected = [12, 21]
      frequency = {}
      for e in expected:
        frequency[e] = 0
      for _ in range(num_epochs):
        output = dequeue_many.eval()
        key = 10 * (output[0] + 1) + (output[1] + 1)
        self.assertIn(key, expected)
        frequency[key] += 1

      # Expect an approximately even distribution over all possible orders.
      expected_frequency = num_epochs / len(expected)
      margin = expected_frequency * 0.4
      tf.logging.info("Observed counts: %s", frequency)
      for key in expected:
        value = frequency[key]
        self.assertGreater(value, expected_frequency - margin)
        self.assertLess(value, expected_frequency + margin)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        dequeue.eval()
      for thread in threads:
        thread.join()

  def testSharedName(self):
    with self.test_session():
      range_size = 5
      queue = tf.train.range_input_producer(
          range_size, shared_name="SHARED_NAME_XYZ", name="Q")
      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          queue.queue_ref.op.node_def.attr["shared_name"])


class SliceInputProducerTest(tf.test.TestCase):

  def testNoShuffle(self):
    with self.test_session() as sess:
      num_epochs = 3
      source_strings = [b"Alpha", b"Beta", b"Delta", b"Gamma"]
      source_ints = [2, 3, 5, 7]
      slices = tf.train.slice_input_producer(
          [source_strings, source_ints], num_epochs=num_epochs, shuffle=False)
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # No randomness, so just see repeated copies of the input.
      num_items = len(source_strings) * num_epochs
      output = [sess.run(slices) for _ in range(num_items)]
      out_strings, out_ints = zip(*output)
      self.assertAllEqual(source_strings * num_epochs, out_strings)
      self.assertAllEqual(source_ints * num_epochs, out_ints)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(slices)
      for thread in threads:
        thread.join()

  def testShuffle(self):
    with self.test_session() as sess:
      num_epochs = 1200
      source_strings = ["A", "B", "D", "G"]
      source_ints = [7, 3, 5, 2]
      slices = tf.train.slice_input_producer(
          [source_strings, source_ints], num_epochs=num_epochs, shuffle=True,
          seed=161803)
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # Validate that we only shuffle the integers within an epoch and
      # count how often each possible order appears.
      expected = [b",".join(x) for x in
                  itertools.permutations([b"A7", b"B3", b"D5", b"G2"])]
      frequency = {}
      for e in expected:
        frequency[e] = 0
      for _ in range(num_epochs):
        output = [sess.run(slices) for _ in range(len(source_strings))]
        key = b",".join([s + tf.compat.as_bytes(str(i)) for s, i in output])
        self.assertIn(key, expected)
        frequency[key] += 1

      # Expect an approximately even distribution over all possible orders.
      expected_frequency = num_epochs / len(expected)
      margin = expected_frequency * 0.4
      tf.logging.info("Observed counts: %s", frequency)
      for key in expected:
        value = frequency[key]
        self.assertGreater(value, expected_frequency - margin)
        self.assertLess(value, expected_frequency + margin)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(slices)
      for thread in threads:
        thread.join()

  def testSharedName(self):
    with self.test_session():
      source_strings = ["A", "B", "D", "G"]
      source_ints = [7, 3, 5, 2]
      slices = tf.train.slice_input_producer(
          [source_strings, source_ints], shared_name="SHARED_NAME_XYZ",
          name="sip")

      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          slices[0].op.inputs[1].op.inputs[0].op.node_def.attr["shared_name"])


class DictHelperTest(tf.test.TestCase):

  def testListInputs(self):
    l = [1, 2, 3, 11, 22, 33]
    l2 = inp._as_tensor_list(l)
    self.assertEquals(l, l2)
    l3 = inp._as_original_type(l, l2)
    self.assertEquals(l, l3)

  def testDictInputs(self):
    d = {"a": 1, "b": 2, "c": 3, "aa": 11, "bb": 22, "cc": 33}
    l = inp._as_tensor_list(d)
    self.assertEquals([1, 11, 2, 22, 3, 33], l)
    d2 = inp._as_original_type(d, l)
    self.assertEquals(d, d2)


class BatchTest(tf.test.TestCase):

  def _testOneThreadHelper(self, use_dict):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(tf.stack([zero64, zero64 + 1]), [2, 1]),
          values=tf.cast(tf.stack([counter, -counter]), tf.float32),
          dense_shape=[2])
      if use_dict:
        batched = tf.train.batch(
            {"c": counter, "s": sparse_counter, "S": "string"},
            batch_size=batch_size)
        batched_fetch = [batched["c"], batched["s"], batched["S"]]
      else:
        batched = tf.train.batch(
            [counter, sparse_counter, "string"], batch_size=batch_size)
        batched_fetch = batched
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      for i in range(num_batches):
        results = sess.run(batched_fetch)
        self.assertAllEqual(results[0], np.arange(i * batch_size,
                                                  (i + 1) * batch_size))
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(2 * batch_size) // 2,  # 0, 0, 1, 1, ...
                       [0, 1] * batch_size)).T)
        #  [x, -x, x+1, -(x+1), ...]
        expected = np.arange(2 * i * batch_size, 2 * (i + 1) * batch_size) // 2
        expected *= ([1, -1] * batch_size)  # mult by [1, -1, 1, -1, ...]
        self.assertAllEqual(results[1].values, expected)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 2])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched_fetch)
      for thread in threads:
        thread.join()

  def testOneThread(self):
    self._testOneThreadHelper(use_dict=False)

  def testOneThreadDict(self):
    self._testOneThreadHelper(use_dict=True)

  def testOneThreadDynamicPad(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      string = tf.tile(["string"], tf.to_int32(tf.stack([counter])))
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      batched = tf.train.batch(
          [counter, string], batch_size=batch_size, dynamic_pad=True)
      threads = tf.train.start_queue_runners()

      for i in range(num_batches):
        results = sess.run(batched)
        expected_results = np.arange(i * batch_size, (i + 1) * batch_size)
        max_len = expected_results[-1]
        self.assertAllEqual(results[0], expected_results)
        expected_strings = [
            [b"string"] * rep + [b""] * (max_len - rep)
            for rep in expected_results]
        self.assertAllEqual(results[1], expected_strings)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testOneThreadEnqueueMany(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      pre_batched = tf.train.batch(
          [counter, sparse_counter, "string"], batch_size=2)
      batched = tf.train.batch(pre_batched, enqueue_many=True,
                               batch_size=batch_size)
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      for i in range(num_batches):
        results = sess.run(batched)
        self.assertAllEqual(results[0], np.arange(i * batch_size,
                                                  (i + 1) * batch_size))
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(
            results[1].values, np.arange(i * batch_size, (i + 1) * batch_size))
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testManyThreads(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)

      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      batched = tf.train.batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size, num_threads=4)
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        all_counts.extend(results[0])
        self.assertAllEqual(results[2], [b"string"] * batch_size)
      self.assertItemsEqual(all_counts, range(num_batches * batch_size))

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testOneThreadSmallerBatch(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      extra_elements = 5
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size + extra_elements)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(tf.stack([zero64, zero64 + 1]), [2, 1]),
          values=tf.cast(tf.stack([counter, -counter]), tf.float32),
          dense_shape=[2])
      batched = tf.train.batch(
          [counter, sparse_counter, "string"], batch_size=batch_size,
          allow_smaller_final_batch=True)
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      for i in range(num_batches):
        results = sess.run(batched)
        self.assertAllEqual(results[0], np.arange(i * batch_size,
                                                  (i + 1) * batch_size))
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(2 * batch_size) // 2,  # 0, 0, 1, 1, ...
                       [0, 1] * batch_size)).T)
        #  [x, -x, x+1, -(x+1), ...]
        expected = np.arange(2 * i * batch_size, 2 * (i + 1) * batch_size) // 2
        expected *= ([1, -1] * batch_size)  # mult by [1, -1, 1, -1, ...]
        self.assertAllEqual(results[1].values, expected)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 2])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the final batch with extra_elements.
      results = sess.run(batched)
      self.assertAllEqual(results[0],
                          np.arange(num_batches * batch_size,
                                    num_batches * batch_size + extra_elements))
      self.assertAllEqual(
          results[1].indices,
          np.vstack((np.arange(2 * extra_elements) // 2,  # 0, 0, 1, 1, ...
                     [0, 1] * extra_elements)).T)
      self.assertAllEqual(results[1].dense_shape, [extra_elements, 2])
      self.assertAllEqual(results[2], [b"string"] * extra_elements)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testManyThreadsSmallerBatch(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      extra_elements = 5
      zero64 = tf.constant(0, dtype=tf.int64)

      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size + extra_elements)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      batched = tf.train.batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size, num_threads=4, allow_smaller_final_batch=True)
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        all_counts.extend(results[0])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the final batch with extra_elements.
      results = sess.run(batched)
      tf.logging.info("Last Batch: %s", results[0])
      self.assertEqual(len(results[0]), extra_elements)
      self.assertAllEqual(results[0], results[1].values)
      self.assertAllEqual(
          results[1].indices,
          np.vstack((np.arange(extra_elements), np.zeros(extra_elements))).T)
      self.assertAllEqual(results[1].dense_shape, [extra_elements, 1])
      all_counts.extend(results[0])
      self.assertAllEqual(results[2], [b"string"] * extra_elements)
      self.assertItemsEqual(all_counts,
                            range(num_batches * batch_size + extra_elements))

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testSharedName(self):
    with self.test_session():
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = tf.train.batch(
          [counter, "string"], batch_size=batch_size,
          shared_name="SHARED_NAME_XYZ", name="Q")

      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          batched[0].op.inputs[0].op.node_def.attr["shared_name"])

  def testCannotInferRankError(self):
    with self.test_session():
      x = tf.placeholder(dtype=tf.int64)
      with self.assertRaisesRegexp(ValueError, "Cannot infer Tensor's rank"):
        tf.train.batch([x], batch_size=2)

  def testBatchedSparseTensorInferredShape(self):
    sparse = tf.SparseTensor(indices=[[0]], values=[1.0], dense_shape=[1])
    self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
    batched = tf.train.batch([sparse], batch_size=2)
    self.assertAllEqual((2,), batched.dense_shape.get_shape().as_list())

  def testBatchedSparseTensorInferredShapeEnqueueMany(self):
    sparse = tf.SparseTensor(indices=[[0]], values=[1.0], dense_shape=[1])
    self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
    batched = tf.train.batch([sparse], batch_size=2, enqueue_many=True)
    self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testBatchedSparseTensorInferredShapeUnknownRank(self):
    sparse = tf.SparseTensor(
        indices=tf.placeholder(tf.int64),
        values=tf.placeholder(tf.float32),
        dense_shape=tf.placeholder(tf.int64))
    self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
    batched = tf.train.batch([sparse], batch_size=2)
    self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testBatchedSparseTensorInferredShapeUnknownRankEnqueueMany(self):
    sparse = tf.SparseTensor(
        indices=tf.placeholder(tf.int64),
        values=tf.placeholder(tf.float32),
        dense_shape=tf.placeholder(tf.int64))
    self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
    batched = tf.train.batch([sparse], batch_size=2, enqueue_many=True)
    self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testSingleElementDict(self):
    x = tf.train.batch({"c": [12, 12]}, batch_size=8)
    self.assertAllEqual((8, 2), x["c"].get_shape().as_list())

  def _testKeepInputHelper(self, num_threads, enqueue_many):
    with self.test_session() as sess:
      batch_size = 5
      num_batches = 4
      examples = tf.Variable(0)
      counter = examples.count_up_to(num_batches * batch_size * 2)
      sparse_counter = tf.SparseTensor(
          indices=tf.zeros([1, 1], dtype=tf.int64),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      to_batch = [counter, sparse_counter, "string"]
      if enqueue_many:
        to_batch = tf.train.batch(to_batch, 1)
      keep_input = tf.squeeze(tf.equal(0, tf.mod(to_batch[0], 2)))
      batched = tf.train.maybe_batch(
          to_batch, keep_input, batch_size, num_threads=num_threads,
          enqueue_many=enqueue_many)
      tf.initialize_all_variables().run()
      tf.initialize_local_variables().run()
      threads = tf.train.start_queue_runners()

      for _ in range(num_batches):
        results = sess.run(batched)
        self.assertAllEqual([0] * batch_size, np.mod(results[0], 2))
        self.assertAllEqual([0] * batch_size, np.mod(results[1].values, 2))
        self.assertAllEqual([b"string"] * batch_size, results[2])

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testSingleThreadKeepInput(self):
    self._testKeepInputHelper(1, False)

  def testSingleThreadKeepInputEnqueueMany(self):
    self._testKeepInputHelper(1, True)

  def testMultipleThreadKeepInput(self):
    self._testKeepInputHelper(5, False)

  def testMultipleThreadKeepInputEnqueueMany(self):
    self._testKeepInputHelper(5, True)

  def testMaybeBatchedSparseTensorInferredShape(self):
    sparse = tf.SparseTensor(indices=[[0]], values=[1.0], dense_shape=[1])
    self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
    batched = tf.train.maybe_batch([sparse], keep_input=True, batch_size=2)
    self.assertAllEqual((2,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueMany(self):
    sparse = tf.SparseTensor(indices=[[0]], values=[1.0], dense_shape=[1])
    self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
    batched = tf.train.maybe_batch(
        [sparse], keep_input=True, batch_size=2, enqueue_many=True)
    self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRank(self):
    sparse = tf.SparseTensor(
        indices=tf.placeholder(tf.int64),
        values=tf.placeholder(tf.float32),
        dense_shape=tf.placeholder(tf.int64))
    self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
    batched = tf.train.maybe_batch([sparse], keep_input=True, batch_size=2)
    self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankEnqueueMany(self):
    sparse = tf.SparseTensor(
        indices=tf.placeholder(tf.int64),
        values=tf.placeholder(tf.float32),
        dense_shape=tf.placeholder(tf.int64))
    self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
    batched = tf.train.maybe_batch(
        [sparse], keep_input=True, batch_size=2, enqueue_many=True)
    self.assertIs(None, batched.dense_shape.get_shape().num_elements())


class BatchJoinTest(tf.test.TestCase):

  def _testTwoThreadsHelper(self, use_dict):
    with self.test_session() as sess:
      # Two threads, the first generates (0..69, "a").
      num_a = 70
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_a)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])

      # The second generates (99, "b") 90 times and then stops.
      num_b = 90
      ninety_nine = tf.train.limit_epochs(
          tf.constant(99, dtype=tf.int64), num_b)
      sparse_ninety_nine = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(ninety_nine, tf.float32)]),
          dense_shape=[1])

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      if use_dict:
        batched = tf.train.batch_join(
            [{"c": counter, "s": sparse_counter, "S": "a"},
             {"c": ninety_nine, "s": sparse_ninety_nine, "S": "b"}],
            batch_size=batch_size)
        batched_fetch = [batched["c"], batched["s"], batched["S"]]
      else:
        batched = tf.train.batch_join(
            [[counter, sparse_counter, "a"],
             [ninety_nine, sparse_ninety_nine, "b"]],
            batch_size=batch_size)
        batched_fetch = batched

      # Shapes.
      self.assertEqual(3, len(batched_fetch))
      self.assertAllEqual((batch_size,), batched_fetch[0].get_shape().as_list())
      self.assertAllEqual(
          (None, 2), batched_fetch[1].indices.get_shape().as_list())
      self.assertAllEqual(
          (None,), batched_fetch[1].values.get_shape().as_list())
      self.assertAllEqual(
          (2,), batched_fetch[1].dense_shape.get_shape().as_list())
      self.assertAllEqual((batch_size,), batched_fetch[2].get_shape().as_list())

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = sess.run(batched_fetch)
        self.assertEqual(3, len(results))
        self.assertEqual(batch_size, len(results[0]))
        self.assertEqual(batch_size, len(results[2]))
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
        which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if len(which_a) > 0 and len(which_b) > 0: saw_both += 1
        all_a.extend([results[0][i] for i in which_a])
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Some minimum level of mixing of the results of both threads.
      self.assertGreater(saw_both, 1)

      # Verify the order of results from "a" were preserved.
      self.assertAllEqual(all_a, np.arange(num_a))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched_fetch)
      for thread in threads:
        thread.join()

  def testTwoThreads(self):
    self._testTwoThreadsHelper(use_dict=False)

  def testTwoThreadsDict(self):
    self._testTwoThreadsHelper(use_dict=True)

  def testMismatchedDictKeys(self):
    with self.assertRaisesRegexp(ValueError, "must have the same keys"):
      tf.train.batch_join(
          [{"c": 12, "s": 123, "S": "a"},
           {"cool": -12, "s": 99, "S": "b"}],
          batch_size=8)

  def testTwoThreadsDynamicPad(self):
    with self.test_session() as sess:
      # Two threads, the first generates (0..69, ["a"] * 1..70).
      num_a = 70
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_a)

      # The second generates (99, ["b"] * 99) 90 times and then stops.
      num_b = 90
      ninety_nine = tf.train.limit_epochs(
          tf.constant(99, dtype=tf.int64), num_b)

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      a = tf.tile(["a"], tf.to_int32(tf.stack([counter + 1])))
      b = tf.tile(["b"], tf.to_int32(tf.stack([ninety_nine])))
      batched = tf.train.batch_join(
          [[counter, a],
           [ninety_nine, b]],
          batch_size=batch_size, dynamic_pad=True)

      # Shapes.
      self.assertEqual(2, len(batched))
      self.assertAllEqual((batch_size,), batched[0].get_shape().as_list())
      self.assertAllEqual((batch_size, None), batched[1].get_shape().as_list())

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      count_string_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = sess.run(batched)
        self.assertEqual(2, len(results))
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[1]), batch_size)
        for s in results[1]:
          if s[0] == b"b":
            self.assertAllEqual(s, [b"b"] * 99)
          else:
            count_string_a.append(sum(x == b"a" for x in s))
        which_a = [i for i, s in enumerate(results[1]) if s[0] == b"a"]
        which_b = [i for i, s in enumerate(results[1]) if s[0] == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if len(which_a) > 0 and len(which_b) > 0: saw_both += 1
        all_a.extend([results[0][i] for i in which_a])
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Some minimum level of mixing of the results of both threads.
      self.assertGreater(saw_both, 1)

      # Verify the order of results from "a" were preserved.
      self.assertAllEqual(  # tiled "a" with counter + 1
          count_string_a, np.arange(num_a) + 1)
      self.assertAllEqual(all_a, np.arange(num_a))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testTwoThreadsSmallerBatch(self):
    with self.test_session() as sess:
      extra_elements = 2
      # Two threads, the first generates (0..69, "a").
      num_a = 70 + extra_elements
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_a)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])

      # The second generates (99, "b") 90 times and then stops.
      num_b = 90 + extra_elements
      ninety_nine = tf.train.limit_epochs(
          tf.constant(99, dtype=tf.int64), num_b)
      sparse_ninety_nine = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(ninety_nine, tf.float32)]),
          dense_shape=[1])

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      batched = tf.train.batch_join(
          [[counter, sparse_counter, "a"],
           [ninety_nine, sparse_ninety_nine, "b"]],
          batch_size=batch_size,
          allow_smaller_final_batch=True)

      # Shapes.
      self.assertEqual(3, len(batched))
      self.assertAllEqual((None,), batched[0].get_shape().as_list())
      self.assertAllEqual((None, 2), batched[1].indices.get_shape().as_list())
      self.assertAllEqual((None,), batched[1].values.get_shape().as_list())
      self.assertAllEqual((2,), batched[1].dense_shape.get_shape().as_list())
      self.assertAllEqual((None,), batched[2].get_shape().as_list())

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[2]), batch_size)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
        which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if which_a and which_b: saw_both += 1
        all_a.extend([results[0][i] for i in which_a])
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Reached the final batch with 2 * extra_elements.
      results = sess.run(batched)
      tf.logging.info("Last Batch: %s", results[0])
      self.assertEqual(len(results[0]), 2 * extra_elements)
      self.assertEqual(len(results[2]), 2 * extra_elements)
      self.assertAllEqual(results[0], results[1].values)
      self.assertAllEqual(
          results[1].indices,
          np.vstack((np.arange(2 * extra_elements),
                     np.zeros(2 * extra_elements))).T)
      self.assertAllEqual(results[1].dense_shape, [2 * extra_elements, 1])
      which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
      which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
      self.assertEqual(len(which_a) + len(which_b), 2 * extra_elements)
      if which_a and which_b: saw_both += 1
      all_a.extend([results[0][i] for i in which_a])
      seen_b += len(which_b)

      # Some minimum level of mixing of the results of both threads.
      self.assertGreater(saw_both, 1)

      # Verify the order of results from "a" were preserved.
      self.assertAllEqual(all_a, np.arange(num_a))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testTwoThreadsDynamicPadSmallerBatch(self):
    with self.test_session() as sess:
      extra_elements = 2
      # Two threads, the first generates (0..69, ["a"] * 1..70).
      num_a = 70 + extra_elements
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_a)

      # The second generates (99, ["b"] * 99) 90 times and then stops.
      num_b = 90 + extra_elements
      ninety_nine = tf.train.limit_epochs(
          tf.constant(99, dtype=tf.int64), num_b)

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      a = tf.tile(["a"], tf.to_int32(tf.stack([counter + 1])))
      b = tf.tile(["b"], tf.to_int32(tf.stack([ninety_nine])))
      batched = tf.train.batch_join(
          [[counter, a],
           [ninety_nine, b]],
          batch_size=batch_size,
          dynamic_pad=True,
          allow_smaller_final_batch=True)

      # Shapes.
      self.assertEqual(2, len(batched))
      self.assertAllEqual((None,), batched[0].get_shape().as_list())
      self.assertAllEqual((None, None), batched[1].get_shape().as_list())

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      count_string_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[1]), batch_size)
        for s in results[1]:
          if s[0] == b"b":
            self.assertAllEqual(s, [b"b"] * 99)
          else:
            count_string_a.append(sum(x == b"a" for x in s))
        which_a = [i for i, s in enumerate(results[1]) if s[0] == b"a"]
        which_b = [i for i, s in enumerate(results[1]) if s[0] == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if which_a and which_b: saw_both += 1
        all_a.extend([results[0][i] for i in which_a])
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Reached the final batch with 2 * extra_elements.
      results = sess.run(batched)
      tf.logging.info("Last Batch: %s", results[0])
      self.assertEqual(len(results[0]), 2 * extra_elements)
      self.assertEqual(len(results[1]), 2 * extra_elements)
      for s in results[1]:
        if s[0] == b"b":
          self.assertAllEqual(s, [b"b"] * 99)
        else:
          count_string_a.append(sum(x == b"a" for x in s))
      which_a = [i for i, s in enumerate(results[1]) if s[0] == b"a"]
      which_b = [i for i, s in enumerate(results[1]) if s[0] == b"b"]
      self.assertEqual(len(which_a) + len(which_b), 2 * extra_elements)
      if which_a and which_b: saw_both += 1
      all_a.extend([results[0][i] for i in which_a])
      seen_b += len(which_b)

      # Some minimum level of mixing of the results of both threads.
      self.assertGreater(saw_both, 1)

      # Verify the order of results from "a" were preserved.
      self.assertAllEqual(  # tiled "a" with counter + 1
          count_string_a, np.arange(num_a) + 1)
      self.assertAllEqual(all_a, np.arange(num_a))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testSharedName(self):
    with self.test_session():
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = tf.train.batch_join(
          [[counter, "string"]], batch_size=batch_size,
          shared_name="SHARED_NAME_XYZ", name="Q")

      # Shapes.
      self.assertEqual(2, len(batched))
      self.assertAllEqual((batch_size,), batched[0].get_shape().as_list())
      self.assertAllEqual((batch_size,), batched[1].get_shape().as_list())

      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          batched[0].op.inputs[0].op.node_def.attr["shared_name"])

  def testCannotInferRankError(self):
    with self.test_session():
      x = tf.placeholder(dtype=tf.int64)
      with self.assertRaisesRegexp(ValueError, "Cannot infer Tensor's rank"):
        tf.train.batch_join([[x]], batch_size=2)

  def testSingleElementDict(self):
    x = tf.train.batch_join([{"c": [12, 12]}], batch_size=8)
    self.assertAllEqual((8, 2), x["c"].get_shape().as_list())

  def _testKeepInputHelper(self, num_threads, enqueue_many):
    with self.test_session() as sess:
      batch_size = 5
      num_batches = 4
      examples = tf.Variable(0)
      counter = examples.count_up_to(num_batches * batch_size * 2)
      sparse_counter = tf.SparseTensor(
          indices=tf.zeros([1, 1], dtype=tf.int64),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      to_batch = [counter, sparse_counter, "string"]
      if enqueue_many:
        to_batch = tf.train.batch(to_batch, 1)
      keep_input = tf.squeeze(tf.equal(0, tf.mod(to_batch[0], 2)))
      batched = tf.train.maybe_batch_join(
          [to_batch] * num_threads, keep_input, batch_size,
          enqueue_many=enqueue_many)
      tf.initialize_all_variables().run()
      tf.initialize_local_variables().run()
      threads = tf.train.start_queue_runners()

      for _ in range(num_batches):
        results = sess.run(batched)
        self.assertAllEqual([0] * batch_size, np.mod(results[0], 2),)
        self.assertAllEqual([0] * batch_size, np.mod(results[1].values, 2),)
        self.assertAllEqual([b"string"] * batch_size, results[2])

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testSingleThreadKeepInput(self):
    self._testKeepInputHelper(1, False)

  def testSingleThreadKeepInputEnqueueMany(self):
    self._testKeepInputHelper(1, True)

  def testMultipleThreadKeepInput(self):
    self._testKeepInputHelper(5, False)

  def testMultipleThreadKeepInputEnqueueMany(self):
    self._testKeepInputHelper(5, True)

  def testMaybeBatchedSparseTensorInferredShape(self):
    sparse = tf.SparseTensor(indices=[[0]], values=[1.0], dense_shape=[1])
    self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
    batched = tf.train.maybe_batch_join(
        [[sparse]], keep_input=True, batch_size=2)
    self.assertAllEqual((2,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueMany(self):
    sparse = tf.SparseTensor(indices=[[0]], values=[1.0], dense_shape=[1])
    self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
    batched = tf.train.maybe_batch_join(
        [[sparse]], keep_input=True, batch_size=2, enqueue_many=True)
    self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRank(self):
    sparse = tf.SparseTensor(
        indices=tf.placeholder(tf.int64),
        values=tf.placeholder(tf.float32),
        dense_shape=tf.placeholder(tf.int64))
    self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
    batched = tf.train.maybe_batch_join(
        [[sparse]], keep_input=True, batch_size=2)
    self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankEnqueueMany(self):
    sparse = tf.SparseTensor(
        indices=tf.placeholder(tf.int64),
        values=tf.placeholder(tf.float32),
        dense_shape=tf.placeholder(tf.int64))
    self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
    batched = tf.train.maybe_batch_join(
        [[sparse]], keep_input=True, batch_size=2, enqueue_many=True)
    self.assertIs(None, batched.dense_shape.get_shape().num_elements())


class ShuffleBatchTest(tf.test.TestCase):

  def _testOneThreadHelper(self, use_dict):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      if use_dict:
        batched = tf.train.shuffle_batch(
            {"c": counter, "s": sparse_counter, "S": "string"},
            batch_size=batch_size, capacity=32,
            min_after_dequeue=16, seed=141421)
        batched_fetch = [batched["c"], batched["s"], batched["S"]]
      else:
        batched = tf.train.shuffle_batch(
            [counter, sparse_counter, "string"],
            batch_size=batch_size, capacity=32,
            min_after_dequeue=16, seed=141421)
        batched_fetch = batched
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = sess.run(batched_fetch)
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        self.assertAllEqual(results[2], [b"string"] * batch_size)
      # Results scrambled, but include all the expected numbers.
      deltas = [all_counts[i + 1] - all_counts[i]
                for i in range(len(all_counts) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertItemsEqual(all_counts, range(num_batches * batch_size))

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched_fetch)
      for thread in threads:
        thread.join()

  def testOneThread(self):
    self._testOneThreadHelper(use_dict=False)

  def testOneThreadDict(self):
    self._testOneThreadHelper(use_dict=True)

  def testOneThreadSmallerBatch(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      extra_elements = 5
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      total_elements = num_batches * batch_size + extra_elements
      counter = examples.count_up_to(total_elements)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      batched = tf.train.shuffle_batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size, capacity=32,
          min_after_dequeue=16, seed=141421,
          allow_smaller_final_batch=True)
      batched_fetch = batched
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      all_counts = []
      for _ in range(num_batches):
        results = sess.run(batched_fetch)
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the final batch with extra elements.
      results = sess.run(batched)
      self.assertAllEqual(results[1].dense_shape, [extra_elements, 1])
      self.assertAllEqual(results[2], [b"string"] * extra_elements)
      all_counts.extend(results[0])

      # Results scrambled, but include all the expected numbers.
      deltas = [all_counts[i + 1] - all_counts[i]
                for i in range(len(all_counts) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertItemsEqual(all_counts, range(total_elements))

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched_fetch)
      for thread in threads:
        thread.join()

  def testManyThreads(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      batched = tf.train.shuffle_batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size, capacity=32,
          min_after_dequeue=16, seed=173205, num_threads=4)
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        self.assertAllEqual(results[2], [b"string"] * batch_size)
      # Results scrambled, but include all the expected numbers.
      deltas = [all_counts[i + 1] - all_counts[i]
                for i in range(len(all_counts) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertItemsEqual(all_counts, range(num_batches * batch_size))

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testManyThreadsSmallerBatch(self):
    with self.test_session() as sess:
      batch_size = 10
      num_batches = 3
      extra_elements = 5
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      total_elements = num_batches * batch_size + extra_elements
      counter = examples.count_up_to(total_elements)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      batched = tf.train.shuffle_batch(
          [counter, sparse_counter, "string"],
          batch_size=batch_size, capacity=32,
          min_after_dequeue=16, seed=173205, num_threads=4,
          allow_smaller_final_batch=True)
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      all_counts = []
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        all_counts.extend(results[0])
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        self.assertAllEqual(results[2], [b"string"] * batch_size)

      # Reached the final batch with extra elements.
      results = sess.run(batched)
      self.assertAllEqual(results[0].shape, [extra_elements])
      self.assertAllEqual(results[1].dense_shape, [extra_elements, 1])
      self.assertAllEqual(results[2], [b"string"] * extra_elements)
      all_counts.extend(results[0])

      # Results scrambled, but include all the expected numbers.
      deltas = [all_counts[i + 1] - all_counts[i]
                for i in range(len(all_counts) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertItemsEqual(all_counts, range(total_elements))

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testSharedName(self):
    with self.test_session():
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = tf.train.shuffle_batch(
          [counter, "string"], batch_size=batch_size,
          capacity=32,
          min_after_dequeue=10,
          shared_name="SHARED_NAME_XYZ", name="Q")

      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          batched[0].op.inputs[0].op.node_def.attr["shared_name"])

  def _testKeepInputHelper(self, num_threads, enqueue_many):
    with self.test_session() as sess:
      batch_size = 5
      num_batches = 4
      examples = tf.Variable(0)
      counter = examples.count_up_to(num_batches * batch_size * 2)
      sparse_counter = tf.SparseTensor(
          indices=tf.zeros([1, 1], dtype=tf.int64),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      to_batch = [counter, sparse_counter, "string"]
      if enqueue_many:
        to_batch = tf.train.batch(to_batch, 1)
      keep_input = tf.squeeze(tf.equal(0, tf.mod(to_batch[0], 2)))
      batched = tf.train.maybe_shuffle_batch(
          to_batch, batch_size, 10, 1, keep_input, num_threads=num_threads,
          enqueue_many=enqueue_many)
      tf.initialize_all_variables().run()
      tf.initialize_local_variables().run()
      threads = tf.train.start_queue_runners()

      for _ in range(num_batches):
        results = sess.run(batched)
        self.assertAllEqual([0] * batch_size, np.mod(results[0], 2))
        self.assertAllEqual([0] * batch_size, np.mod(results[1].values, 2))
        self.assertAllEqual([b"string"] * batch_size, results[2])

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testSingleThreadKeepInput(self):
    self._testKeepInputHelper(1, False)

  def testSingleThreadKeepInputEnqueueMany(self):
    self._testKeepInputHelper(1, True)

  def testMultipleThreadKeepInput(self):
    self._testKeepInputHelper(5, False)

  def testMultipleThreadKeepInputEnqueueMany(self):
    self._testKeepInputHelper(5, True)

  def testMaybeBatchedSparseTensorInferredShape(self):
    sparse = tf.SparseTensor(indices=[[0]], values=[1.0], dense_shape=[1])
    self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
    batched = tf.train.maybe_shuffle_batch([sparse], 2, 10, 1, True)
    self.assertAllEqual((2,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueMany(self):
    sparse = tf.SparseTensor(indices=[[0]], values=[1.0], dense_shape=[1])
    self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
    batched = tf.train.maybe_shuffle_batch(
        [sparse], 2, 10, 1, True, enqueue_many=True)
    self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRank(self):
    sparse = tf.SparseTensor(
        indices=tf.placeholder(tf.int64),
        values=tf.placeholder(tf.float32),
        dense_shape=tf.placeholder(tf.int64))
    self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
    batched = tf.train.maybe_shuffle_batch([sparse], 2, 10, 1, True)
    self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankEnqueueMany(self):
    sparse = tf.SparseTensor(
        indices=tf.placeholder(tf.int64),
        values=tf.placeholder(tf.float32),
        dense_shape=tf.placeholder(tf.int64))
    self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
    batched = tf.train.maybe_shuffle_batch(
        [sparse], 2, 10, 1, True, enqueue_many=True)
    self.assertIs(None, batched.dense_shape.get_shape().num_elements())


class ShuffleBatchJoinTest(tf.test.TestCase):

  def _testTwoThreadsHelper(self, use_dict):
    with self.test_session() as sess:
      # Two threads, the first generates (0..24, "a").
      num_a = 25
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_a)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])

      # The second generates (99, "b") 35 times and then stops.
      num_b = 35
      ninety_nine = tf.train.limit_epochs(
          tf.constant(99, dtype=tf.int64), num_b)
      sparse_ninety_nine = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(ninety_nine, tf.float32)]),
          dense_shape=[1])

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      if use_dict:
        batched = tf.train.shuffle_batch_join(
            [{"c": counter, "s": sparse_counter, "S": "a"},
             {"c": ninety_nine, "s": sparse_ninety_nine, "S": "b"}],
            batch_size=batch_size, capacity=32,
            min_after_dequeue=16, seed=223607)
        batched_fetch = [batched["c"], batched["s"], batched["S"]]
      else:
        batched = tf.train.shuffle_batch_join(
            [[counter, sparse_counter, "a"],
             [ninety_nine, sparse_ninety_nine, "b"]],
            batch_size=batch_size, capacity=32,
            min_after_dequeue=16, seed=223607)
        batched_fetch = batched

      # Shapes.
      self.assertEqual(3, len(batched_fetch))
      self.assertAllEqual((batch_size,), batched_fetch[0].get_shape().as_list())
      self.assertAllEqual(
          (None, 2), batched_fetch[1].indices.get_shape().as_list())
      self.assertAllEqual(
          (None,), batched_fetch[1].values.get_shape().as_list())
      self.assertAllEqual(
          (2,), batched_fetch[1].dense_shape.get_shape().as_list())
      self.assertAllEqual((batch_size,), batched_fetch[2].get_shape().as_list())

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = sess.run(batched_fetch)
        self.assertEqual(3, len(results))
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[2]), batch_size)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
        which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if which_a and which_b: saw_both += 1
        all_a.extend([results[0][i] for i in which_a])
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Some minimum level of mixing of the results of both threads.
      self.assertGreater(saw_both, 1)

      # Saw all the items from "a", but scrambled.
      self.assertItemsEqual(all_a, range(num_a))
      deltas = [all_a[i + 1] - all_a[i]
                for i in range(len(all_a) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched_fetch)
      for thread in threads:
        thread.join()

  def testTwoThreads(self):
    self._testTwoThreadsHelper(use_dict=False)

  def testTwoThreadsDict(self):
    self._testTwoThreadsHelper(use_dict=True)

  def testTwoThreadsSmallerBatch(self):
    with self.test_session() as sess:
      # Two threads, the first generates (0..26, "a").
      extra_elements = 2
      num_a = 25 + extra_elements
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_a)
      sparse_counter = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])

      # The second generates (99, "b") 37 times and then stops.
      num_b = 35 + extra_elements
      ninety_nine = tf.train.limit_epochs(
          tf.constant(99, dtype=tf.int64), num_b)
      sparse_ninety_nine = tf.SparseTensor(
          indices=tf.reshape(zero64, [1, 1]),
          values=tf.stack([tf.cast(ninety_nine, tf.float32)]),
          dense_shape=[1])

      # These get joined together and grouped into batches of 5.
      batch_size = 5
      batched = tf.train.shuffle_batch_join(
          [[counter, sparse_counter, "a"],
           [ninety_nine, sparse_ninety_nine, "b"]],
          batch_size=batch_size, capacity=32,
          min_after_dequeue=16, seed=223607, allow_smaller_final_batch=True)

      # Shapes.
      self.assertEqual(3, len(batched))
      self.assertAllEqual((None,), batched[0].get_shape().as_list())
      self.assertAllEqual((None, 2), batched[1].indices.get_shape().as_list())
      self.assertAllEqual((None,), batched[1].values.get_shape().as_list())
      self.assertAllEqual((2,), batched[1].dense_shape.get_shape().as_list())
      self.assertAllEqual((None,), batched[2].get_shape().as_list())

      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      threads = tf.train.start_queue_runners()

      # Should see the "a" and "b" threads mixed together.
      all_a = []
      seen_b = 0
      saw_both = 0
      num_batches = (num_a + num_b) // batch_size
      for i in range(num_batches):
        results = sess.run(batched)
        tf.logging.info("Batch %d: %s", i, results[0])
        self.assertEqual(len(results[0]), batch_size)
        self.assertEqual(len(results[2]), batch_size)
        self.assertAllEqual(results[0], results[1].values)
        self.assertAllEqual(
            results[1].indices,
            np.vstack((np.arange(batch_size), np.zeros(batch_size))).T)
        self.assertAllEqual(results[1].dense_shape, [batch_size, 1])
        which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
        which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
        self.assertEqual(len(which_a) + len(which_b), batch_size)
        if which_a and which_b: saw_both += 1
        all_a.extend([results[0][i] for i in which_a])
        seen_b += len(which_b)
        self.assertAllEqual([99] * len(which_b),
                            [results[0][i] for i in which_b])

      # Reached end with 2 * extra_elements left
      results = sess.run(batched)
      self.assertEqual(len(results[0]), 2 * extra_elements)
      self.assertAllEqual(results[1].dense_shape, [2 * extra_elements, 1])
      self.assertEqual(len(results[2]), 2 * extra_elements)
      self.assertAllEqual(results[0], results[1].values)
      self.assertAllEqual(
          results[1].indices,
          np.vstack((np.arange(2 * extra_elements),
                     np.zeros(2 * extra_elements))).T)
      which_a = [i for i, s in enumerate(results[2]) if s == b"a"]
      which_b = [i for i, s in enumerate(results[2]) if s == b"b"]
      self.assertEqual(len(which_a) + len(which_b), 2 * extra_elements)
      if which_a and which_b: saw_both += 1
      all_a.extend([results[0][i] for i in which_a])
      seen_b += len(which_b)

      # Some minimum level of mixing of the results of both threads.
      self.assertGreater(saw_both, 1)

      # Saw all the items from "a", but scrambled, including extras.
      self.assertItemsEqual(all_a, range(num_a))
      deltas = [all_a[i + 1] - all_a[i]
                for i in range(len(all_a) - 1)]
      self.assertFalse(all(d == deltas[0] for d in deltas))
      self.assertEqual(seen_b, num_b)

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testMismatchedDictKeys(self):
    with self.assertRaisesRegexp(ValueError, "must have the same keys"):
      tf.train.shuffle_batch_join(
          [{"c": 12, "s": 123, "S": "a"},
           {"cool": -12, "s": 99, "S": "b"}],
          batch_size=8, capacity=32,
          min_after_dequeue=16, seed=223607)

  def testSharedName(self):
    with self.test_session():
      batch_size = 10
      num_batches = 3
      zero64 = tf.constant(0, dtype=tf.int64)
      examples = tf.Variable(zero64)
      counter = examples.count_up_to(num_batches * batch_size)
      batched = tf.train.shuffle_batch_join(
          [[counter, "string"]], batch_size=batch_size,
          capacity=32,
          min_after_dequeue=10,
          shared_name="SHARED_NAME_XYZ", name="Q")

      # Shapes.
      self.assertEqual(2, len(batched))
      self.assertAllEqual((batch_size,), batched[0].get_shape().as_list())
      self.assertAllEqual((batch_size,), batched[1].get_shape().as_list())

      self.assertProtoEquals(
          "s: 'SHARED_NAME_XYZ'",
          batched[0].op.inputs[0].op.node_def.attr["shared_name"])

  def _testKeepInputHelper(self, num_threads, enqueue_many):
    with self.test_session() as sess:
      batch_size = 5
      num_batches = 4
      examples = tf.Variable(0)
      counter = examples.count_up_to(num_batches * batch_size * 2)
      sparse_counter = tf.SparseTensor(
          indices=tf.zeros([1, 1], dtype=tf.int64),
          values=tf.stack([tf.cast(counter, tf.float32)]),
          dense_shape=[1])
      to_batch = [counter, sparse_counter, "string"]
      if enqueue_many:
        to_batch = tf.train.batch(to_batch, 1)
      keep_input = tf.squeeze(tf.equal(0, tf.mod(to_batch[0], 2)))
      batched = tf.train.maybe_shuffle_batch_join(
          [to_batch] * num_threads, batch_size, 10, 1, keep_input,
          enqueue_many=enqueue_many)
      tf.initialize_all_variables().run()
      tf.initialize_local_variables().run()
      threads = tf.train.start_queue_runners()

      for _ in range(num_batches):
        results = sess.run(batched)
        self.assertAllEqual([0] * batch_size, np.mod(results[0], 2))
        self.assertAllEqual([0] * batch_size, np.mod(results[1].values, 2))
        self.assertAllEqual([b"string"] * batch_size, results[2])

      # Reached the limit.
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(batched)
      for thread in threads:
        thread.join()

  def testSingleThreadKeepInput(self):
    self._testKeepInputHelper(1, False)

  def testSingleThreadKeepInputEnqueueMany(self):
    self._testKeepInputHelper(1, True)

  def testMultipleThreadKeepInput(self):
    self._testKeepInputHelper(5, False)

  def testMultipleThreadKeepInputEnqueueMany(self):
    self._testKeepInputHelper(5, True)

  def testMaybeBatchedSparseTensorInferredShape(self):
    sparse = tf.SparseTensor(indices=[[0]], values=[1.0], dense_shape=[1])
    self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
    batched = tf.train.maybe_shuffle_batch_join([[sparse]], 2, 10, 1, True)
    self.assertAllEqual((2,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeEnqueueMany(self):
    sparse = tf.SparseTensor(indices=[[0]], values=[1.0], dense_shape=[1])
    self.assertAllEqual((1,), sparse.dense_shape.get_shape().as_list())
    batched = tf.train.maybe_shuffle_batch_join(
        [[sparse]], 2, 10, 1, True, enqueue_many=True)
    self.assertAllEqual((1,), batched.dense_shape.get_shape().as_list())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRank(self):
    sparse = tf.SparseTensor(
        indices=tf.placeholder(tf.int64),
        values=tf.placeholder(tf.float32),
        dense_shape=tf.placeholder(tf.int64))
    self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
    batched = tf.train.maybe_shuffle_batch_join([[sparse]], 2, 10, 1, True)
    self.assertIs(None, batched.dense_shape.get_shape().num_elements())

  def testMaybeBatchedSparseTensorInferredShapeUnknownRankEnqueueMany(self):
    sparse = tf.SparseTensor(
        indices=tf.placeholder(tf.int64),
        values=tf.placeholder(tf.float32),
        dense_shape=tf.placeholder(tf.int64))
    self.assertIs(None, sparse.dense_shape.get_shape().num_elements())
    batched = tf.train.maybe_shuffle_batch_join(
        [[sparse]], 2, 10, 1, True, enqueue_many=True)
    self.assertIs(None, batched.dense_shape.get_shape().num_elements())


if __name__ == "__main__":
  tf.test.main()
