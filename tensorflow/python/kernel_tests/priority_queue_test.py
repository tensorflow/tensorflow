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
"""Tests for tensorflow.ops.data_flow_ops.PriorityQueue."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random
import threading

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


@test_util.run_v1_only("PriorityQueue removed from v2")
class PriorityQueueTest(test.TestCase):

  def testRoundTripInsertReadOnceSorts(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string, dtypes.string), (
          (), ()))
      elem = np.random.randint(-5, 5, size=100).astype(np.int64)
      side_value_0 = np.random.rand(100).astype(bytes)
      side_value_1 = np.random.rand(100).astype(bytes)
      enq_list = [
          q.enqueue((e, constant_op.constant(v0), constant_op.constant(v1)))
          for e, v0, v1 in zip(elem, side_value_0, side_value_1)
      ]
      for enq in enq_list:
        enq.run()

      deq = q.dequeue_many(100)
      deq_elem, deq_value_0, deq_value_1 = self.evaluate(deq)

      allowed = {}
      missed = set()
      for e, v0, v1 in zip(elem, side_value_0, side_value_1):
        if e not in allowed:
          allowed[e] = set()
        allowed[e].add((v0, v1))
        missed.add((v0, v1))

      self.assertAllEqual(deq_elem, sorted(elem))
      for e, dv0, dv1 in zip(deq_elem, deq_value_0, deq_value_1):
        self.assertTrue((dv0, dv1) in allowed[e])
        missed.remove((dv0, dv1))
      self.assertEqual(missed, set())

  def testRoundTripInsertMultiThreadedReadOnceSorts(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string, dtypes.string), (
          (), ()))
      elem = np.random.randint(-5, 5, size=100).astype(np.int64)
      side_value_0 = np.random.rand(100).astype(bytes)
      side_value_1 = np.random.rand(100).astype(bytes)

      enqueue_ops = [
          q.enqueue((e, constant_op.constant(v0), constant_op.constant(v1)))
          for e, v0, v1 in zip(elem, side_value_0, side_value_1)
      ]

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        self.evaluate(enqueue_op)

      dequeue_op = q.dequeue_many(100)

      enqueue_threads = [
          self.checkedThread(
              target=enqueue, args=(op,)) for op in enqueue_ops
      ]

      for t in enqueue_threads:
        t.start()

      deq_elem, deq_value_0, deq_value_1 = self.evaluate(dequeue_op)

      for t in enqueue_threads:
        t.join()

      allowed = {}
      missed = set()
      for e, v0, v1 in zip(elem, side_value_0, side_value_1):
        if e not in allowed:
          allowed[e] = set()
        allowed[e].add((v0, v1))
        missed.add((v0, v1))

      self.assertAllEqual(deq_elem, sorted(elem))
      for e, dv0, dv1 in zip(deq_elem, deq_value_0, deq_value_1):
        self.assertTrue((dv0, dv1) in allowed[e])
        missed.remove((dv0, dv1))
      self.assertEqual(missed, set())

  def testRoundTripFillsCapacityMultiThreadedEnqueueAndDequeue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(10, (dtypes.int64), (()))

      num_threads = 40
      enqueue_counts = np.random.randint(10, size=num_threads)
      enqueue_values = [
          np.random.randint(
              5, size=count) for count in enqueue_counts
      ]
      enqueue_ops = [
          q.enqueue_many((values, values)) for values in enqueue_values
      ]
      shuffled_counts = copy.deepcopy(enqueue_counts)
      random.shuffle(shuffled_counts)
      dequeue_ops = [q.dequeue_many(count) for count in shuffled_counts]
      all_enqueued_values = np.hstack(enqueue_values)

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        self.evaluate(enqueue_op)

      dequeued = []

      def dequeue(dequeue_op):
        (dequeue_indices, dequeue_values) = self.evaluate(dequeue_op)
        self.assertAllEqual(dequeue_indices, dequeue_values)
        dequeued.extend(dequeue_indices)

      enqueue_threads = [
          self.checkedThread(
              target=enqueue, args=(op,)) for op in enqueue_ops
      ]
      dequeue_threads = [
          self.checkedThread(
              target=dequeue, args=(op,)) for op in dequeue_ops
      ]

      # Dequeue and check
      for t in dequeue_threads:
        t.start()
      for t in enqueue_threads:
        t.start()
      for t in enqueue_threads:
        t.join()
      for t in dequeue_threads:
        t.join()

      self.assertAllEqual(sorted(dequeued), sorted(all_enqueued_values))

  def testRoundTripInsertManyMultiThreadedReadManyMultithreadedSorts(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(2000, (dtypes.int64), (()))

      num_threads = 40
      enqueue_counts = np.random.randint(10, size=num_threads)
      enqueue_values = [
          np.random.randint(
              5, size=count) for count in enqueue_counts
      ]
      enqueue_ops = [
          q.enqueue_many((values, values)) for values in enqueue_values
      ]
      shuffled_counts = copy.deepcopy(enqueue_counts)
      random.shuffle(shuffled_counts)
      dequeue_ops = [q.dequeue_many(count) for count in shuffled_counts]
      all_enqueued_values = np.hstack(enqueue_values)

      dequeue_wait = threading.Condition()

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        self.evaluate(enqueue_op)

      def dequeue(dequeue_op, dequeued):
        (dequeue_indices, dequeue_values) = self.evaluate(dequeue_op)
        self.assertAllEqual(dequeue_indices, dequeue_values)
        dequeue_wait.acquire()
        dequeued.extend(dequeue_indices)
        dequeue_wait.release()

      dequeued = []
      enqueue_threads = [
          self.checkedThread(
              target=enqueue, args=(op,)) for op in enqueue_ops
      ]
      dequeue_threads = [
          self.checkedThread(
              target=dequeue, args=(op, dequeued)) for op in dequeue_ops
      ]

      for t in enqueue_threads:
        t.start()
      for t in enqueue_threads:
        t.join()
      # Dequeue and check
      for t in dequeue_threads:
        t.start()
      for t in dequeue_threads:
        t.join()

      # We can't guarantee full sorting because we can't guarantee
      # that the dequeued.extend() call runs immediately after the
      # self.evaluate() call.  Here we're just happy everything came out.
      self.assertAllEqual(set(dequeued), set(all_enqueued_values))

  def testRoundTripInsertManyMultiThreadedReadOnceSorts(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string, dtypes.string), (
          (), ()))
      elem = np.random.randint(-5, 5, size=100).astype(np.int64)
      side_value_0 = np.random.rand(100).astype(bytes)
      side_value_1 = np.random.rand(100).astype(bytes)

      batch = 5
      enqueue_ops = [
          q.enqueue_many((elem[i * batch:(i + 1) * batch],
                          side_value_0[i * batch:(i + 1) * batch],
                          side_value_1[i * batch:(i + 1) * batch]))
          for i in range(20)
      ]

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        self.evaluate(enqueue_op)

      dequeue_op = q.dequeue_many(100)

      enqueue_threads = [
          self.checkedThread(
              target=enqueue, args=(op,)) for op in enqueue_ops
      ]

      for t in enqueue_threads:
        t.start()

      deq_elem, deq_value_0, deq_value_1 = self.evaluate(dequeue_op)

      for t in enqueue_threads:
        t.join()

      allowed = {}
      missed = set()
      for e, v0, v1 in zip(elem, side_value_0, side_value_1):
        if e not in allowed:
          allowed[e] = set()
        allowed[e].add((v0, v1))
        missed.add((v0, v1))

      self.assertAllEqual(deq_elem, sorted(elem))
      for e, dv0, dv1 in zip(deq_elem, deq_value_0, deq_value_1):
        self.assertTrue((dv0, dv1) in allowed[e])
        missed.remove((dv0, dv1))
      self.assertEqual(missed, set())

  def testRoundTripInsertOnceReadOnceSorts(self):
    with self.cached_session() as sess:
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string, dtypes.string), (
          (), ()))
      elem = np.random.randint(-100, 100, size=1000).astype(np.int64)
      side_value_0 = np.random.rand(1000).astype(bytes)
      side_value_1 = np.random.rand(1000).astype(bytes)
      q.enqueue_many((elem, side_value_0, side_value_1)).run()
      deq = q.dequeue_many(1000)
      deq_elem, deq_value_0, deq_value_1 = self.evaluate(deq)

      allowed = {}
      for e, v0, v1 in zip(elem, side_value_0, side_value_1):
        if e not in allowed:
          allowed[e] = set()
        allowed[e].add((v0, v1))

      self.assertAllEqual(deq_elem, sorted(elem))
      for e, dv0, dv1 in zip(deq_elem, deq_value_0, deq_value_1):
        self.assertTrue((dv0, dv1) in allowed[e])

  def testRoundTripInsertOnceReadManySorts(self):
    with self.cached_session():
      q = data_flow_ops.PriorityQueue(2000, (dtypes.int64), (()))
      elem = np.random.randint(-100, 100, size=1000).astype(np.int64)
      q.enqueue_many((elem, elem)).run()
      deq_values = np.hstack((q.dequeue_many(100)[0].eval() for _ in range(10)))
      self.assertAllEqual(deq_values, sorted(elem))

  def testRoundTripInsertOnceReadOnceLotsSorts(self):
    with self.cached_session():
      q = data_flow_ops.PriorityQueue(2000, (dtypes.int64), (()))
      elem = np.random.randint(-100, 100, size=1000).astype(np.int64)
      q.enqueue_many((elem, elem)).run()
      dequeue_op = q.dequeue()
      deq_values = np.hstack(dequeue_op[0].eval() for _ in range(1000))
      self.assertAllEqual(deq_values, sorted(elem))

  def testInsertingNonInt64Fails(self):
    with self.cached_session():
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string), (()))
      with self.assertRaises(TypeError):
        q.enqueue_many((["a", "b", "c"], ["a", "b", "c"])).run()

  def testInsertingNonScalarFails(self):
    with self.cached_session() as sess:
      input_priority = array_ops.placeholder(dtypes.int64)
      input_other = array_ops.placeholder(dtypes.string)
      q = data_flow_ops.PriorityQueue(2000, (dtypes.string,), (()))

      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          r"Shape mismatch in tuple component 0. Expected \[\], got \[2\]"):
        sess.run([q.enqueue((input_priority, input_other))],
                 feed_dict={
                     input_priority: np.array(
                         [0, 2], dtype=np.int64),
                     input_other: np.random.rand(3, 5).astype(bytes)
                 })

      with self.assertRaisesRegexp(
          errors_impl.InvalidArgumentError,
          r"Shape mismatch in tuple component 0. Expected \[2\], got \[2,2\]"):
        sess.run(
            [q.enqueue_many((input_priority, input_other))],
            feed_dict={
                input_priority: np.array(
                    [[0, 2], [3, 4]], dtype=np.int64),
                input_other: np.random.rand(2, 3).astype(bytes)
            })


if __name__ == "__main__":
  test.main()
