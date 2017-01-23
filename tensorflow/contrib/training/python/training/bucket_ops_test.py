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
"""Tests for tf.contrib.training.bucket."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from tensorflow.contrib.training.python.training import bucket_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner_impl


def _which_bucket(bucket_edges, v):
  """Identify which bucket v falls into.

  Args:
    bucket_edges: int array, bucket edges
    v: int scalar, index
  Returns:
    int scalar, the bucket.
    If v < bucket_edges[0], return 0.
    If bucket_edges[0] <= v < bucket_edges[1], return 1.
    ...
    If bucket_edges[-2] <= v < bucket_edges[-1], return len(bucket_edges).
    If v >= bucket_edges[-1], return len(bucket_edges) + 1
  """
  v = np.asarray(v)
  full = [0] + bucket_edges
  found = np.where(np.logical_and(v >= full[:-1], v < full[1:]))[0]
  if not found.size:
    return len(full)
  return found[0]


class BucketTest(test.TestCase):

  def setUp(self):
    ops.reset_default_graph()

    self.scalar_int_feed = array_ops.placeholder(dtypes_lib.int32, ())
    self.unk_int64_feed = array_ops.placeholder(dtypes_lib.int64, (None,))
    self.vec3_str_feed = array_ops.placeholder(dtypes_lib.string, (3,))

    self._coord = coordinator.Coordinator()
    # Make capacity very large so we can feed all the inputs in the
    # main thread without blocking
    input_queue = data_flow_ops.PaddingFIFOQueue(
        5000,
        dtypes=[dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.string],
        shapes=[(), (None,), (3,)])

    self._input_enqueue_op = input_queue.enqueue(
        (self.scalar_int_feed, self.unk_int64_feed, self.vec3_str_feed))
    self.scalar_int, self.unk_int64, self.vec3_str = input_queue.dequeue()
    self._threads = None
    self._close_op = input_queue.close()
    self._sess = None

  def enqueue_inputs(self, sess, feed_dict):
    sess.run(self._input_enqueue_op, feed_dict=feed_dict)

  def start_queue_runners(self, sess):
    # Store session to be able to close inputs later
    if self._sess is None:
      self._sess = sess
    self._threads = queue_runner_impl.start_queue_runners(coord=self._coord)

  def tearDown(self):
    if self._sess is not None:
      self._sess.run(self._close_op)
    self._coord.request_stop()
    self._coord.join(self._threads)

  def testSingleBucket(self):
    bucketed_dynamic = bucket_ops.bucket(
        tensors=[self.scalar_int, self.unk_int64, self.vec3_str],
        which_bucket=constant_op.constant(0),
        num_buckets=2,
        batch_size=32,
        num_threads=10,
        dynamic_pad=True)
    # Check shape inference on bucketing outputs
    self.assertAllEqual(
        [[32], [32, None], [32, 3]],
        [out.get_shape().as_list() for out in bucketed_dynamic[1]])
    with self.test_session() as sess:
      for v in range(32):
        self.enqueue_inputs(sess, {
            self.scalar_int_feed: v,
            self.unk_int64_feed: v * [v],
            self.vec3_str_feed: 3 * [str(v)]
        })
      self.start_queue_runners(sess)

      # Get a single minibatch
      bucketed_values = sess.run(bucketed_dynamic)

      # (which_bucket, bucket_tensors).
      self.assertEqual(2, len(bucketed_values))

      # Count number of bucket_tensors.
      self.assertEqual(3, len(bucketed_values[1]))

      # Ensure bucket 0 was used for all minibatch entries.
      self.assertAllEqual(0, bucketed_values[0])

      expected_scalar_int = np.arange(32)
      expected_unk_int64 = np.zeros((32, 31)).astype(np.int64)
      for i in range(32):
        expected_unk_int64[i, :i] = i
      expected_vec3_str = np.vstack(3 * [np.arange(32).astype(bytes)]).T

      # Must resort the output because num_threads > 1 leads to
      # sometimes-inconsistent insertion order.
      resort = np.argsort(bucketed_values[1][0])
      self.assertAllEqual(expected_scalar_int, bucketed_values[1][0][resort])
      self.assertAllEqual(expected_unk_int64, bucketed_values[1][1][resort])
      self.assertAllEqual(expected_vec3_str, bucketed_values[1][2][resort])

  def testBatchSizePerBucket(self):
    which_bucket = control_flow_ops.cond(self.scalar_int < 5,
                                         lambda: constant_op.constant(0),
                                         lambda: constant_op.constant(1))
    batch_sizes = [5, 10]
    bucketed_dynamic = bucket_ops.bucket(
        tensors=[self.scalar_int, self.unk_int64, self.vec3_str],
        which_bucket=which_bucket,
        num_buckets=2,
        batch_size=batch_sizes,
        num_threads=1,
        dynamic_pad=True)
    # Check shape inference on bucketing outputs
    self.assertAllEqual(
        [[None], [None, None], [None, 3]],
        [out.get_shape().as_list() for out in bucketed_dynamic[1]])
    with self.test_session() as sess:
      for v in range(15):
        self.enqueue_inputs(sess, {
            self.scalar_int_feed: v,
            self.unk_int64_feed: v * [v],
            self.vec3_str_feed: 3 * [str(v)]
        })
      self.start_queue_runners(sess)

      # Get two minibatches (one with small values, one with large).
      bucketed_values_0 = sess.run(bucketed_dynamic)
      bucketed_values_1 = sess.run(bucketed_dynamic)

      # Figure out which output has the small values
      if bucketed_values_0[0] < 5:
        bucketed_values_large, bucketed_values_small = (bucketed_values_1,
                                                        bucketed_values_0)
      else:
        bucketed_values_small, bucketed_values_large = (bucketed_values_0,
                                                        bucketed_values_1)

      # Ensure bucket 0 was used for all minibatch entries.
      self.assertAllEqual(0, bucketed_values_small[0])
      self.assertAllEqual(1, bucketed_values_large[0])

      # Check that the batch sizes differ per bucket
      self.assertEqual(5, len(bucketed_values_small[1][0]))
      self.assertEqual(10, len(bucketed_values_large[1][0]))

  def testEvenOddBuckets(self):
    which_bucket = (self.scalar_int % 2)
    bucketed_dynamic = bucket_ops.bucket(
        tensors=[self.scalar_int, self.unk_int64, self.vec3_str],
        which_bucket=which_bucket,
        num_buckets=2,
        batch_size=32,
        num_threads=10,
        dynamic_pad=True)
    # Check shape inference on bucketing outputs
    self.assertAllEqual(
        [[32], [32, None], [32, 3]],
        [out.get_shape().as_list() for out in bucketed_dynamic[1]])
    with self.test_session() as sess:
      for v in range(64):
        self.enqueue_inputs(sess, {
            self.scalar_int_feed: v,
            self.unk_int64_feed: v * [v],
            self.vec3_str_feed: 3 * [str(v)]
        })
      self.start_queue_runners(sess)

      # Get two minibatches (one containing even values, one containing odds)
      bucketed_values_0 = sess.run(bucketed_dynamic)
      bucketed_values_1 = sess.run(bucketed_dynamic)

      # (which_bucket, bucket_tensors).
      self.assertEqual(2, len(bucketed_values_0))
      self.assertEqual(2, len(bucketed_values_1))

      # Count number of bucket_tensors.
      self.assertEqual(3, len(bucketed_values_0[1]))
      self.assertEqual(3, len(bucketed_values_1[1]))

      # Figure out which output has the even values (there's
      # randomness due to the multithreaded nature of bucketing)
      if bucketed_values_0[0] % 2 == 1:
        bucketed_values_even, bucketed_values_odd = (bucketed_values_1,
                                                     bucketed_values_0)
      else:
        bucketed_values_even, bucketed_values_odd = (bucketed_values_0,
                                                     bucketed_values_1)

      # Ensure bucket 0 was used for all minibatch entries.
      self.assertAllEqual(0, bucketed_values_even[0])
      self.assertAllEqual(1, bucketed_values_odd[0])

      # Test the first bucket outputted, the events starting at 0
      expected_scalar_int = np.arange(0, 32 * 2, 2)
      expected_unk_int64 = np.zeros((32, 31 * 2)).astype(np.int64)
      for i in range(0, 32):
        expected_unk_int64[i, :2 * i] = 2 * i
      expected_vec3_str = np.vstack(3 *
                                    [np.arange(0, 32 * 2, 2).astype(bytes)]).T

      # Must resort the output because num_threads > 1 leads to
      # sometimes-inconsistent insertion order.
      resort = np.argsort(bucketed_values_even[1][0])
      self.assertAllEqual(expected_scalar_int,
                          bucketed_values_even[1][0][resort])
      self.assertAllEqual(expected_unk_int64,
                          bucketed_values_even[1][1][resort])
      self.assertAllEqual(expected_vec3_str, bucketed_values_even[1][2][resort])

      # Test the second bucket outputted, the odds starting at 1
      expected_scalar_int = np.arange(1, 32 * 2 + 1, 2)
      expected_unk_int64 = np.zeros((32, 31 * 2 + 1)).astype(np.int64)
      for i in range(0, 32):
        expected_unk_int64[i, :2 * i + 1] = 2 * i + 1
      expected_vec3_str = np.vstack(
          3 * [np.arange(1, 32 * 2 + 1, 2).astype(bytes)]).T

      # Must resort the output because num_threads > 1 leads to
      # sometimes-inconsistent insertion order.
      resort = np.argsort(bucketed_values_odd[1][0])
      self.assertAllEqual(expected_scalar_int,
                          bucketed_values_odd[1][0][resort])
      self.assertAllEqual(expected_unk_int64, bucketed_values_odd[1][1][resort])
      self.assertAllEqual(expected_vec3_str, bucketed_values_odd[1][2][resort])

  def testEvenOddBucketsFilterOutAllOdd(self):
    which_bucket = (self.scalar_int % 2)
    keep_input = math_ops.equal(which_bucket, 0)
    bucketed_dynamic = bucket_ops.bucket(
        tensors=[self.scalar_int, self.unk_int64, self.vec3_str],
        which_bucket=which_bucket,
        num_buckets=2,
        batch_size=32,
        num_threads=10,
        keep_input=keep_input,
        dynamic_pad=True)
    # Check shape inference on bucketing outputs
    self.assertAllEqual(
        [[32], [32, None], [32, 3]],
        [out.get_shape().as_list() for out in bucketed_dynamic[1]])
    with self.test_session() as sess:
      for v in range(128):
        self.enqueue_inputs(sess, {
            self.scalar_int_feed: v,
            self.unk_int64_feed: v * [v],
            self.vec3_str_feed: 3 * [str(v)]
        })
      self.start_queue_runners(sess)

      # Get two minibatches ([0, 2, ...] and [64, 66, ...])
      bucketed_values_even0 = sess.run(bucketed_dynamic)
      bucketed_values_even1 = sess.run(bucketed_dynamic)

      # Ensure that bucket 1 was completely filtered out
      self.assertAllEqual(0, bucketed_values_even0[0])
      self.assertAllEqual(0, bucketed_values_even1[0])

      # Merge their output for sorting and comparison
      bucketed_values_all_elem0 = np.concatenate((bucketed_values_even0[1][0],
                                                  bucketed_values_even1[1][0]))

      self.assertAllEqual(
          np.arange(0, 128, 2), sorted(bucketed_values_all_elem0))


class BucketBySequenceLengthTest(test.TestCase):

  def _testBucketBySequenceLength(self, allow_small_batch):
    ops.reset_default_graph()

    # All inputs must be identical lengths across tuple index.
    # The input reader will get input_length from the first tuple
    # entry.
    data_len = 4
    labels_len = 3
    input_pairs = [(length, ([np.int64(length)] * data_len,
                             [str(length).encode("ascii")] * labels_len))
                   for length in (1, 3, 4, 5, 6, 10)]

    lengths = array_ops.placeholder(dtypes_lib.int32, ())
    data = array_ops.placeholder(dtypes_lib.int64, (data_len,))
    labels = array_ops.placeholder(dtypes_lib.string, (labels_len,))

    batch_size = 8
    bucket_boundaries = [3, 4, 5, 10]

    # Make capacity very large so we can feed all the inputs in the
    # main thread without blocking
    input_queue = data_flow_ops.FIFOQueue(
        5000, (dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.string), (
            (), (data_len,), (labels_len,)))
    input_enqueue_op = input_queue.enqueue((lengths, data, labels))
    lengths_t, data_t, labels_t = input_queue.dequeue()
    close_input_op = input_queue.close()

    (out_lengths_t, data_and_labels_t) = (bucket_ops.bucket_by_sequence_length(
        input_length=lengths_t,
        tensors=[data_t, labels_t],
        batch_size=batch_size,
        bucket_boundaries=bucket_boundaries,
        allow_smaller_final_batch=allow_small_batch,
        num_threads=10))

    expected_batch_size = None if allow_small_batch else batch_size
    self.assertEqual(out_lengths_t.get_shape().as_list(), [expected_batch_size])
    self.assertEqual(data_and_labels_t[0].get_shape().as_list(),
                     [expected_batch_size, data_len])
    self.assertEqual(data_and_labels_t[1].get_shape().as_list(),
                     [expected_batch_size, labels_len])

    def _read_test(sess):
      for _ in range(50):
        (out_lengths, (data, labels)) = sess.run(
            (out_lengths_t, data_and_labels_t))
        if allow_small_batch:
          self.assertEqual(data_len, data.shape[1])
          self.assertEqual(labels_len, labels.shape[1])
          self.assertGreaterEqual(batch_size, out_lengths.shape[0])
          self.assertGreaterEqual(batch_size, data.shape[0])
          self.assertGreaterEqual(batch_size, labels.shape[0])
        else:
          self.assertEqual((batch_size, data_len), data.shape)
          self.assertEqual((batch_size, labels_len), labels.shape)
          self.assertEqual((batch_size,), out_lengths.shape)
        for (lr, dr, tr) in zip(out_lengths, data, labels):
          # Make sure length matches data (here it's the same value).
          self.assertEqual(dr[0], lr)
          # Make sure data & labels match.
          self.assertEqual(dr[0], int(tr[0].decode("ascii")))
          # Make sure for each row, data came from the same bucket.
          self.assertEqual(
              _which_bucket(bucket_boundaries, dr[0]),
              _which_bucket(bucket_boundaries, dr[1]))

    with self.test_session() as sess:
      coord = coordinator.Coordinator()

      # Feed the inputs, then close the input thread.
      for _ in range(50 * batch_size + 100):
        which = random.randint(0, len(input_pairs) - 1)
        length, pair = input_pairs[which]
        sess.run(input_enqueue_op,
                 feed_dict={lengths: length,
                            data: pair[0],
                            labels: pair[1]})
      sess.run(close_input_op)

      # Start the queue runners
      threads = queue_runner_impl.start_queue_runners(coord=coord)
      # Read off the top of the bucket and ensure correctness of output
      _read_test(sess)
      coord.request_stop()
      coord.join(threads)

  def testBucketBySequenceLength(self):
    self._testBucketBySequenceLength(allow_small_batch=False)

  def testBucketBySequenceLengthAllow(self):
    self._testBucketBySequenceLength(allow_small_batch=True)


if __name__ == "__main__":
  test.main()
