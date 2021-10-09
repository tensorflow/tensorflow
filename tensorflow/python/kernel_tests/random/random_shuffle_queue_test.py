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
"""Tests for tensorflow.ops.data_flow_ops.Queue."""

import random
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


@test_util.run_v1_only("RandomShuffleQueue removed from v2")
class RandomShuffleQueueTest(test.TestCase):

  def setUp(self):
    # Useful for debugging when a test times out.
    super(RandomShuffleQueueTest, self).setUp()
    tf_logging.error("Starting: %s", self._testMethodName)
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()

  def tearDown(self):
    super(RandomShuffleQueueTest, self).tearDown()
    tf_logging.error("Finished: %s", self._testMethodName)

  def testEnqueue(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 5, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      self.assertAllEqual(0, q.size())
      enqueue_op.run()
      self.assertAllEqual(1, q.size())

  def testEnqueueWithShape(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shapes=tensor_shape.TensorShape([3, 2]))
      enqueue_correct_op = q.enqueue(([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],))
      enqueue_correct_op.run()
      self.assertAllEqual(1, q.size())
      with self.assertRaises(ValueError):
        q.enqueue(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],))

  def testEnqueueManyWithShape(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(
          10, 5, [dtypes_lib.int32, dtypes_lib.int32], shapes=[(), (2,)])
      q.enqueue_many([[1, 2, 3, 4], [[1, 1], [2, 2], [3, 3], [4, 4]]]).run()
      self.assertAllEqual(4, q.size())

      q2 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.int32, shapes=tensor_shape.TensorShape([3]))
      q2.enqueue(([1, 2, 3],))
      q2.enqueue_many(([[1, 2, 3]],))

  def testScalarShapes(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(
          10, 0, [dtypes_lib.int32, dtypes_lib.int32], shapes=[(), (1,)])
      q.enqueue_many([[1, 2, 3, 4], [[5], [6], [7], [8]]]).run()
      q.enqueue([9, [10]]).run()
      dequeue_t = q.dequeue()
      results = []
      for _ in range(2):
        a, b = self.evaluate(dequeue_t)
        results.append((a, b))
      a, b = self.evaluate(q.dequeue_many(3))
      for i in range(3):
        results.append((a[i], b[i]))
      self.assertItemsEqual([(1, [5]), (2, [6]), (3, [7]), (4, [8]), (9, [10])],
                            results)

  def testParallelEnqueue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        self.evaluate(enqueue_op)

      threads = [
          self.checkedThread(
              target=enqueue, args=(e,)) for e in enqueue_ops
      ]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      # Dequeue every element using a single thread.
      results = []
      for _ in xrange(len(elems)):
        results.append(dequeued_t.eval())
      self.assertItemsEqual(elems, results)

  def testParallelDequeue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Enqueue every element using a single thread.
      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      # Run one consumer thread for each element in elems.
      results = []

      def dequeue():
        results.append(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in enqueue_ops]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, results)

  def testDequeue(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      vals = [dequeued_t.eval() for _ in xrange(len(elems))]
      self.assertItemsEqual(elems, vals)

  def testEnqueueAndBlockingDequeue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(3, 0, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      def enqueue():
        # The enqueue_ops should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        for enqueue_op in enqueue_ops:
          self.evaluate(enqueue_op)

      results = []

      def dequeue():
        for _ in xrange(len(elems)):
          results.append(self.evaluate(dequeued_t))

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      self.assertItemsEqual(elems, results)

  def testMultiEnqueueAndDequeue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(
          10, 0, (dtypes_lib.int32, dtypes_lib.float32))
      elems = [(5, 10.0), (10, 20.0), (15, 30.0)]
      enqueue_ops = [q.enqueue((x, y)) for x, y in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      results = []
      for _ in xrange(len(elems)):
        x, y = self.evaluate(dequeued_t)
        results.append((x, y))
      self.assertItemsEqual(elems, results)

  def testQueueSizeEmpty(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 5, dtypes_lib.float32)
      self.assertEqual(0, q.size().eval())

  def testQueueSizeAfterEnqueueAndDequeue(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue()
      size = q.size()
      self.assertEqual([], size.get_shape())

      enqueue_op.run()
      self.assertEqual([1], self.evaluate(size))
      dequeued_t.op.run()
      self.assertEqual([0], self.evaluate(size))

  def testEnqueueMany(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue()
      enqueue_op.run()
      enqueue_op.run()

      results = []
      for _ in range(8):
        results.append(dequeued_t.eval())
      self.assertItemsEqual(elems + elems, results)

  def testEmptyEnqueueMany(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 5, dtypes_lib.float32)
      empty_t = constant_op.constant(
          [], dtype=dtypes_lib.float32, shape=[0, 2, 3])
      enqueue_op = q.enqueue_many((empty_t,))
      size_t = q.size()

      self.assertEqual(0, self.evaluate(size_t))
      enqueue_op.run()
      self.assertEqual(0, self.evaluate(size_t))

  def testEmptyDequeueMany(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32, shapes=())
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue_many(0)

      self.assertEqual([], self.evaluate(dequeued_t).tolist())
      enqueue_op.run()
      self.assertEqual([], self.evaluate(dequeued_t).tolist())

  def testEmptyDequeueUpTo(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32, shapes=())
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue_up_to(0)

      self.assertEqual([], self.evaluate(dequeued_t).tolist())
      enqueue_op.run()
      self.assertEqual([], self.evaluate(dequeued_t).tolist())

  def testEmptyDequeueManyWithNoShape(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32)
      enqueue_op = q.enqueue((constant_op.constant(
          [10.0, 20.0], shape=(1, 2)),))
      dequeued_t = q.dequeue_many(0)

      # Expect the operation to fail due to the shape not being constrained.
      with self.assertRaisesOpError(
          "require the components to have specified shapes"):
        self.evaluate(dequeued_t)

      enqueue_op.run()

      # RandomShuffleQueue does not make any attempt to support DequeueMany
      # with unspecified shapes, even if a shape could be inferred from the
      # elements enqueued.
      with self.assertRaisesOpError(
          "require the components to have specified shapes"):
        self.evaluate(dequeued_t)

  def testEmptyDequeueUpToWithNoShape(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32)
      enqueue_op = q.enqueue((constant_op.constant(
          [10.0, 20.0], shape=(1, 2)),))
      dequeued_t = q.dequeue_up_to(0)

      # Expect the operation to fail due to the shape not being constrained.
      with self.assertRaisesOpError(
          "require the components to have specified shapes"):
        self.evaluate(dequeued_t)

      enqueue_op.run()

      # RandomShuffleQueue does not make any attempt to support DequeueUpTo
      # with unspecified shapes, even if a shape could be inferred from the
      # elements enqueued.
      with self.assertRaisesOpError(
          "require the components to have specified shapes"):
        self.evaluate(dequeued_t)

  def testMultiEnqueueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(
          10, 0, (dtypes_lib.float32, dtypes_lib.int32))
      float_elems = [10.0, 20.0, 30.0, 40.0]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue()

      enqueue_op.run()
      enqueue_op.run()

      results = []
      for _ in range(8):
        float_val, int_val = self.evaluate(dequeued_t)
        results.append((float_val, [int_val[0], int_val[1]]))
      expected = list(zip(float_elems, int_elems)) * 2
      self.assertItemsEqual(expected, results)

  def testDequeueMany(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(5)

      enqueue_op.run()

      results = self.evaluate(dequeued_t).tolist()
      results.extend(dequeued_t.eval())
      self.assertItemsEqual(elems, results)

  def testDequeueUpToNoBlocking(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_up_to(5)

      enqueue_op.run()

      results = self.evaluate(dequeued_t).tolist()
      results.extend(dequeued_t.eval())
      self.assertItemsEqual(elems, results)

  def testMultiDequeueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(
          10, 0, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (2,)))
      float_elems = [
          10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
      ]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                   [15, 16], [17, 18], [19, 20]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue_many(4)
      dequeued_single_t = q.dequeue()

      enqueue_op.run()

      results = []
      float_val, int_val = self.evaluate(dequeued_t)
      self.assertEqual(float_val.shape, dequeued_t[0].get_shape())
      self.assertEqual(int_val.shape, dequeued_t[1].get_shape())
      results.extend(zip(float_val, int_val.tolist()))

      float_val, int_val = self.evaluate(dequeued_t)
      results.extend(zip(float_val, int_val.tolist()))

      float_val, int_val = self.evaluate(dequeued_single_t)
      self.assertEqual(float_val.shape, dequeued_single_t[0].get_shape())
      self.assertEqual(int_val.shape, dequeued_single_t[1].get_shape())
      results.append((float_val, int_val.tolist()))

      float_val, int_val = self.evaluate(dequeued_single_t)
      results.append((float_val, int_val.tolist()))

      self.assertItemsEqual(zip(float_elems, int_elems), results)

  def testMultiDequeueUpToNoBlocking(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(
          10, 0, (dtypes_lib.float32, dtypes_lib.int32), shapes=((), (2,)))
      float_elems = [
          10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0
      ]
      int_elems = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
                   [15, 16], [17, 18], [19, 20]]
      enqueue_op = q.enqueue_many((float_elems, int_elems))
      dequeued_t = q.dequeue_up_to(4)
      dequeued_single_t = q.dequeue()

      enqueue_op.run()

      results = []
      float_val, int_val = self.evaluate(dequeued_t)
      # dequeue_up_to has undefined shape.
      self.assertEqual([None], dequeued_t[0].get_shape().as_list())
      self.assertEqual([None, 2], dequeued_t[1].get_shape().as_list())
      results.extend(zip(float_val, int_val.tolist()))

      float_val, int_val = self.evaluate(dequeued_t)
      results.extend(zip(float_val, int_val.tolist()))

      float_val, int_val = self.evaluate(dequeued_single_t)
      self.assertEqual(float_val.shape, dequeued_single_t[0].get_shape())
      self.assertEqual(int_val.shape, dequeued_single_t[1].get_shape())
      results.append((float_val, int_val.tolist()))

      float_val, int_val = self.evaluate(dequeued_single_t)
      results.append((float_val, int_val.tolist()))

      self.assertItemsEqual(zip(float_elems, int_elems), results)

  def testHighDimension(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.int32, (
          (4, 4, 4, 4)))
      elems = np.array([[[[[x] * 4] * 4] * 4] * 4 for x in range(10)], np.int32)
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(10)

      enqueue_op.run()
      self.assertItemsEqual(dequeued_t.eval().tolist(), elems.tolist())

  def testParallelEnqueueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(
          1000, 0, dtypes_lib.float32, shapes=())
      elems = [10.0 * x for x in range(100)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(1000)

      # Enqueue 100 items in parallel on 10 threads.
      def enqueue():
        self.evaluate(enqueue_op)

      threads = [self.checkedThread(target=enqueue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()

      self.assertItemsEqual(dequeued_t.eval(), elems * 10)

  def testParallelDequeueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(
          1000, 0, dtypes_lib.float32, shapes=())
      elems = [10.0 * x for x in range(1000)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(100)

      enqueue_op.run()

      # Dequeue 100 items in parallel on 10 threads.
      dequeued_elems = []

      def dequeue():
        dequeued_elems.extend(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testParallelDequeueUpTo(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(
          1000, 0, dtypes_lib.float32, shapes=())
      elems = [10.0 * x for x in range(1000)]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_up_to(100)

      enqueue_op.run()

      # Dequeue 100 items in parallel on 10 threads.
      dequeued_elems = []

      def dequeue():
        dequeued_elems.extend(self.evaluate(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in range(10)]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testParallelDequeueUpToRandomPartition(self):
    with self.cached_session() as sess:
      dequeue_sizes = [random.randint(50, 150) for _ in xrange(10)]
      total_elements = sum(dequeue_sizes)
      q = data_flow_ops.RandomShuffleQueue(
          total_elements, 0, dtypes_lib.float32, shapes=())

      elems = [10.0 * x for x in xrange(total_elements)]
      enqueue_op = q.enqueue_many((elems,))
      dequeue_ops = [q.dequeue_up_to(size) for size in dequeue_sizes]

      enqueue_op.run()

      # Dequeue random number of items in parallel on 10 threads.
      dequeued_elems = []

      def dequeue(dequeue_op):
        dequeued_elems.extend(self.evaluate(dequeue_op))

      threads = []
      for dequeue_op in dequeue_ops:
        threads.append(self.checkedThread(target=dequeue, args=(dequeue_op,)))
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, dequeued_elems)

  def testBlockingDequeueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_many(4)

      dequeued_elems = []

      def enqueue():
        # The enqueue_op should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.evaluate(enqueue_op)

      def dequeue():
        dequeued_elems.extend(self.evaluate(dequeued_t).tolist())

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      self.assertItemsEqual(elems, dequeued_elems)

  def testBlockingDequeueUpTo(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      dequeued_t = q.dequeue_up_to(4)

      dequeued_elems = []

      def enqueue():
        # The enqueue_op should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.evaluate(enqueue_op)

      def dequeue():
        dequeued_elems.extend(self.evaluate(dequeued_t).tolist())

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      self.assertItemsEqual(elems, dequeued_elems)

  def testDequeueManyWithTensorParameter(self):
    with self.cached_session():
      # Define a first queue that contains integer counts.
      dequeue_counts = [random.randint(1, 10) for _ in range(100)]
      count_q = data_flow_ops.RandomShuffleQueue(100, 0, dtypes_lib.int32)
      enqueue_counts_op = count_q.enqueue_many((dequeue_counts,))
      total_count = sum(dequeue_counts)

      # Define a second queue that contains total_count elements.
      elems = [random.randint(0, 100) for _ in range(total_count)]
      q = data_flow_ops.RandomShuffleQueue(total_count, 0, dtypes_lib.int32, (
          (),))
      enqueue_elems_op = q.enqueue_many((elems,))

      # Define a subgraph that first dequeues a count, then DequeuesMany
      # that number of elements.
      dequeued_t = q.dequeue_many(count_q.dequeue())

      enqueue_counts_op.run()
      enqueue_elems_op.run()

      dequeued_elems = []
      for _ in dequeue_counts:
        dequeued_elems.extend(dequeued_t.eval())
      self.assertItemsEqual(elems, dequeued_elems)

  def testDequeueUpToWithTensorParameter(self):
    with self.cached_session():
      # Define a first queue that contains integer counts.
      dequeue_counts = [random.randint(1, 10) for _ in range(100)]
      count_q = data_flow_ops.RandomShuffleQueue(100, 0, dtypes_lib.int32)
      enqueue_counts_op = count_q.enqueue_many((dequeue_counts,))
      total_count = sum(dequeue_counts)

      # Define a second queue that contains total_count elements.
      elems = [random.randint(0, 100) for _ in range(total_count)]
      q = data_flow_ops.RandomShuffleQueue(total_count, 0, dtypes_lib.int32, (
          (),))
      enqueue_elems_op = q.enqueue_many((elems,))

      # Define a subgraph that first dequeues a count, then DequeuesUpTo
      # that number of elements.
      dequeued_t = q.dequeue_up_to(count_q.dequeue())

      enqueue_counts_op.run()
      enqueue_elems_op.run()

      dequeued_elems = []
      for _ in dequeue_counts:
        dequeued_elems.extend(dequeued_t.eval())
      self.assertItemsEqual(elems, dequeued_elems)

  def testDequeueFromClosedQueue(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 2, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()
      close_op.run()
      results = [dequeued_t.eval() for _ in elems]
      expected = [[elem] for elem in elems]
      self.assertItemsEqual(expected, results)

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                  "is closed and has insufficient"):
        self.evaluate(dequeued_t)

  def testBlockingDequeueFromClosedQueue(self):
    with self.cached_session() as sess:
      min_size = 2
      q = data_flow_ops.RandomShuffleQueue(10, min_size, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue()

      enqueue_op.run()

      results = []

      # Manually dequeue until we hit min_size.
      results.append(self.evaluate(dequeued_t))
      results.append(self.evaluate(dequeued_t))

      def blocking_dequeue():
        results.append(self.evaluate(dequeued_t))
        results.append(self.evaluate(dequeued_t))

        self.assertItemsEqual(elems, results)
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=blocking_dequeue)
      dequeue_thread.start()
      time.sleep(0.1)

      # The dequeue thread blocked when it hit the min_size requirement.
      self.assertEqual(len(results), 2)
      close_op.run()
      dequeue_thread.join()
      # Once the queue is closed, the min_size requirement is lifted.
      self.assertEqual(len(results), 4)

  def testBlockingDequeueFromClosedEmptyQueue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32)
      close_op = q.close()
      dequeued_t = q.dequeue()

      finished = []  # Needs to be a mutable type

      def dequeue():
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)
        finished.append(True)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      self.assertEqual(len(finished), 0)
      close_op.run()
      dequeue_thread.join()
      self.assertEqual(len(finished), 1)

  def testBlockingDequeueManyFromClosedQueue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(4)

      enqueue_op.run()

      progress = []  # Must be mutable

      def dequeue():
        self.assertItemsEqual(elems, self.evaluate(dequeued_t))
        progress.append(1)
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)
        progress.append(2)

      self.assertEqual(len(progress), 0)
      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      for _ in range(100):
        time.sleep(0.01)
        if len(progress) == 1:
          break
      self.assertEqual(len(progress), 1)
      time.sleep(0.01)
      close_op.run()
      dequeue_thread.join()
      self.assertEqual(len(progress), 2)

  def testBlockingDequeueUpToFromClosedQueueReturnsRemainder(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(3)

      enqueue_op.run()

      results = []

      def dequeue():
        results.extend(self.evaluate(dequeued_t))
        self.assertEqual(3, len(results))
        results.extend(self.evaluate(dequeued_t))
        self.assertEqual(4, len(results))

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()
      self.assertItemsEqual(results, elems)

  def testBlockingDequeueUpToSmallerThanMinAfterDequeue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(
          capacity=10,
          min_after_dequeue=2,
          dtypes=dtypes_lib.float32,
          shapes=((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(3)

      enqueue_op.run()

      results = []

      def dequeue():
        results.extend(self.evaluate(dequeued_t))
        self.assertEqual(3, len(results))
        # min_after_dequeue is 2, we ask for 3 elements, and we end up only
        # getting the remaining 1.
        results.extend(self.evaluate(dequeued_t))
        self.assertEqual(4, len(results))

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()
      self.assertItemsEqual(results, elems)

  def testBlockingDequeueManyFromClosedQueueWithElementsRemaining(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()
      dequeued_t = q.dequeue_many(3)
      cleanup_dequeue_t = q.dequeue_many(q.size())

      enqueue_op.run()

      results = []

      def dequeue():
        results.extend(self.evaluate(dequeued_t))
        self.assertEqual(len(results), 3)
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)
        # While the last dequeue failed, we want to insure that it returns
        # any elements that it potentially reserved to dequeue. Thus the
        # next cleanup should return a single element.
        results.extend(self.evaluate(cleanup_dequeue_t))

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()
      self.assertEqual(len(results), 4)

  def testBlockingDequeueManyFromClosedEmptyQueue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(10, 5, dtypes_lib.float32, ((),))
      close_op = q.close()
      dequeued_t = q.dequeue_many(4)

      def dequeue():
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testBlockingDequeueUpToFromClosedEmptyQueue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(10, 5, dtypes_lib.float32, ((),))
      close_op = q.close()
      dequeued_t = q.dequeue_up_to(4)

      def dequeue():
        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.OutOfRangeError,
                                    "is closed and has insufficient"):
          self.evaluate(dequeued_t)

      dequeue_thread = self.checkedThread(target=dequeue)
      dequeue_thread.start()
      # The close_op should run after the dequeue_thread has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      close_op.run()
      dequeue_thread.join()

  def testEnqueueToClosedQueue(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 4, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      close_op = q.close()

      enqueue_op.run()
      close_op.run()

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegex(errors_impl.CancelledError, "is closed"):
        enqueue_op.run()

  def testEnqueueManyToClosedQueue(self):
    with self.cached_session():
      q = data_flow_ops.RandomShuffleQueue(10, 5, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      close_op = q.close()

      enqueue_op.run()
      close_op.run()

      # Expect the operation to fail due to the queue being closed.
      with self.assertRaisesRegex(errors_impl.CancelledError, "is closed"):
        enqueue_op.run()

  def testBlockingEnqueueToFullQueue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(4, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue((50.0,))
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        self.evaluate(blocking_enqueue_op)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()
      # The dequeue ops should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)
      results = []
      for _ in elems:
        results.append(dequeued_t.eval())
      results.append(dequeued_t.eval())
      self.assertItemsEqual(elems + [50.0], results)
      # There wasn't room for 50.0 in the queue when the first element was
      # dequeued.
      self.assertNotEqual(50.0, results[0])
      thread.join()

  def testBlockingEnqueueManyToFullQueue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(4, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue_many(([50.0, 60.0],))
      dequeued_t = q.dequeue()

      enqueue_op.run()

      def blocking_enqueue():
        self.evaluate(blocking_enqueue_op)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()
      # The dequeue ops should run after the blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      results = []
      for _ in elems:
        time.sleep(0.01)
        results.append(dequeued_t.eval())
      results.append(dequeued_t.eval())
      results.append(dequeued_t.eval())
      self.assertItemsEqual(elems + [50.0, 60.0], results)
      # There wasn't room for 50.0 or 60.0 in the queue when the first
      # element was dequeued.
      self.assertNotEqual(50.0, results[0])
      self.assertNotEqual(60.0, results[0])
      # Similarly for 60.0 and the second element.
      self.assertNotEqual(60.0, results[1])
      thread.join()

  def testBlockingEnqueueToClosedQueue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(4, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0, 40.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue((50.0,))
      dequeued_t = q.dequeue()
      close_op = q.close()

      enqueue_op.run()

      def blocking_enqueue():
        # Expect the operation to succeed since it will complete
        # before the queue is closed.
        self.evaluate(blocking_enqueue_op)

        # Expect the operation to fail due to the queue being closed.
        with self.assertRaisesRegex(errors_impl.CancelledError, "closed"):
          self.evaluate(blocking_enqueue_op)

      thread1 = self.checkedThread(target=blocking_enqueue)
      thread1.start()

      # The close_op should run after the first blocking_enqueue_op has blocked.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      def blocking_close():
        self.evaluate(close_op)

      thread2 = self.checkedThread(target=blocking_close)
      thread2.start()

      # Wait for the close op to block before unblocking the enqueue.
      # TODO(mrry): Figure out how to do this without sleeping.
      time.sleep(0.1)

      results = []
      # Dequeue to unblock the first blocking_enqueue_op, after which the
      # close will complete.
      results.append(dequeued_t.eval())
      self.assertTrue(results[0] in elems)
      thread2.join()
      thread1.join()

  def testBlockingEnqueueManyToClosedQueue(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(4, 0, dtypes_lib.float32, ((),))
      elems = [10.0, 20.0, 30.0]
      enqueue_op = q.enqueue_many((elems,))
      blocking_enqueue_op = q.enqueue_many(([50.0, 60.0],))
      close_op = q.close()
      size_t = q.size()

      enqueue_op.run()
      self.assertEqual(size_t.eval(), 3)

      def blocking_enqueue():
        # This will block until the dequeue after the close.
        self.evaluate(blocking_enqueue_op)

      thread1 = self.checkedThread(target=blocking_enqueue)
      thread1.start()

      # First blocking_enqueue_op of blocking_enqueue has enqueued 1 of 2
      # elements, and is blocked waiting for one more element to be dequeue.
      for i in range(50):
        queue_size = self.evaluate(size_t)
        if queue_size == 4:
          break
        elif i == 49:
          self.fail(
              "Blocking enqueue op did not execute within the expected time.")

        time.sleep(0.1)

      def blocking_close():
        self.evaluate(close_op)

      thread2 = self.checkedThread(target=blocking_close)
      thread2.start()

      # Unblock the first blocking_enqueue_op in blocking_enqueue.
      q.dequeue().eval()

      thread2.join()
      thread1.join()

      # At this point the close operation will complete, so the next enqueue
      # will fail.
      with self.assertRaisesRegex(errors_impl.CancelledError, "closed"):
        self.evaluate(blocking_enqueue_op)

  def testSharedQueueSameSession(self):
    with self.cached_session():
      q1 = data_flow_ops.RandomShuffleQueue(
          1, 0, dtypes_lib.float32, ((),), shared_name="shared_queue")
      q1.enqueue((10.0,)).run()

      # TensorFlow TestCase adds a default graph seed (=87654321). We check if
      # the seed computed from the default graph seed is reproduced.
      seed = 887634792
      q2 = data_flow_ops.RandomShuffleQueue(
          1,
          0,
          dtypes_lib.float32, ((),),
          shared_name="shared_queue",
          seed=seed)

      q1_size_t = q1.size()
      q2_size_t = q2.size()

      self.assertEqual(q1_size_t.eval(), 1)
      self.assertEqual(q2_size_t.eval(), 1)

      self.assertEqual(q2.dequeue().eval(), 10.0)

      self.assertEqual(q1_size_t.eval(), 0)
      self.assertEqual(q2_size_t.eval(), 0)

      q2.enqueue((20.0,)).run()

      self.assertEqual(q1_size_t.eval(), 1)
      self.assertEqual(q2_size_t.eval(), 1)

      self.assertEqual(q1.dequeue().eval(), 20.0)

      self.assertEqual(q1_size_t.eval(), 0)
      self.assertEqual(q2_size_t.eval(), 0)

  def testSharedQueueSameSessionGraphSeedNone(self):
    with self.cached_session():
      q1 = data_flow_ops.RandomShuffleQueue(
          1,
          0,
          dtypes_lib.float32, ((),),
          shared_name="shared_queue",
          seed=98765432)
      q1.enqueue((10.0,)).run()

      # If both graph and op seeds are not provided, the default value must be
      # used, and in case a shared queue is already created, the second queue op
      # must accept any previous seed value.
      random_seed.set_random_seed(None)
      q2 = data_flow_ops.RandomShuffleQueue(
          1, 0, dtypes_lib.float32, ((),), shared_name="shared_queue")

      q1_size_t = q1.size()
      q2_size_t = q2.size()

      self.assertEqual(q1_size_t.eval(), 1)
      self.assertEqual(q2_size_t.eval(), 1)

  def testIncompatibleSharedQueueErrors(self):
    with self.cached_session():
      q_a_1 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shared_name="q_a")
      q_a_2 = data_flow_ops.RandomShuffleQueue(
          15, 5, dtypes_lib.float32, shared_name="q_a")
      q_a_1.queue_ref.op.run()
      with self.assertRaisesOpError("capacity"):
        q_a_2.queue_ref.op.run()

      q_b_1 = data_flow_ops.RandomShuffleQueue(
          10, 0, dtypes_lib.float32, shared_name="q_b")
      q_b_2 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shared_name="q_b")
      q_b_1.queue_ref.op.run()
      with self.assertRaisesOpError("min_after_dequeue"):
        q_b_2.queue_ref.op.run()

      q_c_1 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shared_name="q_c")
      q_c_2 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.int32, shared_name="q_c")
      q_c_1.queue_ref.op.run()
      with self.assertRaisesOpError("component types"):
        q_c_2.queue_ref.op.run()

      q_d_1 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shared_name="q_d")
      q_d_2 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_d")
      q_d_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_d_2.queue_ref.op.run()

      q_e_1 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_e")
      q_e_2 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shared_name="q_e")
      q_e_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_e_2.queue_ref.op.run()

      q_f_1 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shapes=[(1, 1, 2, 3)], shared_name="q_f")
      q_f_2 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shapes=[(1, 1, 2, 4)], shared_name="q_f")
      q_f_1.queue_ref.op.run()
      with self.assertRaisesOpError("component shapes"):
        q_f_2.queue_ref.op.run()

      q_g_1 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, shared_name="q_g")
      q_g_2 = data_flow_ops.RandomShuffleQueue(
          10, 5, (dtypes_lib.float32, dtypes_lib.int32), shared_name="q_g")
      q_g_1.queue_ref.op.run()
      with self.assertRaisesOpError("component types"):
        q_g_2.queue_ref.op.run()

      q_h_1 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, seed=12, shared_name="q_h")
      q_h_2 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.float32, seed=21, shared_name="q_h")
      q_h_1.queue_ref.op.run()
      with self.assertRaisesOpError("random seeds"):
        q_h_2.queue_ref.op.run()

  def testSelectQueue(self):
    with self.cached_session():
      num_queues = 10
      qlist = []
      for _ in xrange(num_queues):
        qlist.append(
            data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32))
      # Enqueue/Dequeue into a dynamically selected queue
      for _ in xrange(20):
        index = np.random.randint(num_queues)
        q = data_flow_ops.RandomShuffleQueue.from_list(index, qlist)
        q.enqueue((10.,)).run()
        self.assertEqual(q.dequeue().eval(), 10.0)

  def testSelectQueueOutOfRange(self):
    with self.cached_session():
      q1 = data_flow_ops.RandomShuffleQueue(10, 0, dtypes_lib.float32)
      q2 = data_flow_ops.RandomShuffleQueue(15, 0, dtypes_lib.float32)
      enq_q = data_flow_ops.RandomShuffleQueue.from_list(3, [q1, q2])
      with self.assertRaisesOpError("is not in"):
        enq_q.dequeue().eval()

  def _blockingDequeue(self, sess, dequeue_op):
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(dequeue_op)

  def _blockingDequeueMany(self, sess, dequeue_many_op):
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(dequeue_many_op)

  def _blockingDequeueUpTo(self, sess, dequeue_up_to_op):
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(dequeue_up_to_op)

  def _blockingEnqueue(self, sess, enqueue_op):
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(enqueue_op)

  def _blockingEnqueueMany(self, sess, enqueue_many_op):
    with self.assertRaisesOpError("was cancelled"):
      self.evaluate(enqueue_many_op)

  def testResetOfBlockingOperation(self):
    # We need each thread to keep its own device stack or the device scopes
    # won't be properly nested.
    ops.get_default_graph().switch_to_thread_local()
    with self.cached_session() as sess:
      q_empty = data_flow_ops.RandomShuffleQueue(5, 0, dtypes_lib.float32, (
          (),))
      dequeue_op = q_empty.dequeue()
      dequeue_many_op = q_empty.dequeue_many(1)
      dequeue_up_to_op = q_empty.dequeue_up_to(1)

      q_full = data_flow_ops.RandomShuffleQueue(5, 0, dtypes_lib.float32, ((),))
      sess.run(q_full.enqueue_many(([1.0, 2.0, 3.0, 4.0, 5.0],)))
      enqueue_op = q_full.enqueue((6.0,))
      enqueue_many_op = q_full.enqueue_many(([6.0],))

      threads = [
          self.checkedThread(
              self._blockingDequeue, args=(sess, dequeue_op)),
          self.checkedThread(
              self._blockingDequeueMany, args=(sess, dequeue_many_op)),
          self.checkedThread(
              self._blockingDequeueUpTo, args=(sess, dequeue_up_to_op)),
          self.checkedThread(
              self._blockingEnqueue, args=(sess, enqueue_op)),
          self.checkedThread(
              self._blockingEnqueueMany, args=(sess, enqueue_many_op))
      ]
      for t in threads:
        t.start()
      time.sleep(0.1)
      sess.close()  # Will cancel the blocked operations.
      for t in threads:
        t.join()

  def testDequeueManyInDifferentOrders(self):
    with self.cached_session():
      # Specify seeds to make the test deterministic
      # (https://en.wikipedia.org/wiki/Taxicab_number).
      q1 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.int32, ((),), seed=1729)
      q2 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.int32, ((),), seed=87539319)
      enq1 = q1.enqueue_many(([1, 2, 3, 4, 5],))
      enq2 = q2.enqueue_many(([1, 2, 3, 4, 5],))
      deq1 = q1.dequeue_many(5)
      deq2 = q2.dequeue_many(5)

      enq1.run()
      enq1.run()
      enq2.run()
      enq2.run()

      results = [[], [], [], []]

      results[0].extend(deq1.eval())
      results[1].extend(deq2.eval())

      q1.close().run()
      q2.close().run()

      results[2].extend(deq1.eval())
      results[3].extend(deq2.eval())

      # No two should match
      for i in range(1, 4):
        for j in range(i):
          self.assertNotEqual(results[i], results[j])

  def testDequeueUpToInDifferentOrders(self):
    with self.cached_session():
      # Specify seeds to make the test deterministic
      # (https://en.wikipedia.org/wiki/Taxicab_number).
      q1 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.int32, ((),), seed=1729)
      q2 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.int32, ((),), seed=87539319)
      enq1 = q1.enqueue_many(([1, 2, 3, 4, 5],))
      enq2 = q2.enqueue_many(([1, 2, 3, 4, 5],))
      deq1 = q1.dequeue_up_to(5)
      deq2 = q2.dequeue_up_to(5)

      enq1.run()
      enq1.run()
      enq2.run()
      enq2.run()

      results = [[], [], [], []]

      results[0].extend(deq1.eval())
      results[1].extend(deq2.eval())

      q1.close().run()
      q2.close().run()

      results[2].extend(deq1.eval())
      results[3].extend(deq2.eval())

      # No two should match
      for i in range(1, 4):
        for j in range(i):
          self.assertNotEqual(results[i], results[j])

  def testDequeueInDifferentOrders(self):
    with self.cached_session():
      # Specify seeds to make the test deterministic
      # (https://en.wikipedia.org/wiki/Taxicab_number).
      q1 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.int32, ((),), seed=1729)
      q2 = data_flow_ops.RandomShuffleQueue(
          10, 5, dtypes_lib.int32, ((),), seed=87539319)
      enq1 = q1.enqueue_many(([1, 2, 3, 4, 5],))
      enq2 = q2.enqueue_many(([1, 2, 3, 4, 5],))
      deq1 = q1.dequeue()
      deq2 = q2.dequeue()

      enq1.run()
      enq1.run()
      enq2.run()
      enq2.run()

      results = [[], [], [], []]

      for _ in range(5):
        results[0].append(deq1.eval())
        results[1].append(deq2.eval())

      q1.close().run()
      q2.close().run()

      for _ in range(5):
        results[2].append(deq1.eval())
        results[3].append(deq2.eval())

      # No two should match
      for i in range(1, 4):
        for j in range(i):
          self.assertNotEqual(results[i], results[j])

  def testBigEnqueueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(5, 0, dtypes_lib.int32, ((),))
      elem = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      enq = q.enqueue_many((elem,))
      deq = q.dequeue()
      size_op = q.size()

      enq_done = []

      def blocking_enqueue():
        enq_done.append(False)
        # This will fill the queue and then block until enough dequeues happen.
        self.evaluate(enq)
        enq_done.append(True)

      thread = self.checkedThread(target=blocking_enqueue)
      thread.start()

      # The enqueue should start and then block.
      results = []
      results.append(deq.eval())  # Will only complete after the enqueue starts.
      self.assertEqual(len(enq_done), 1)
      self.assertEqual(self.evaluate(size_op), 5)

      for _ in range(3):
        results.append(deq.eval())

      time.sleep(0.1)
      self.assertEqual(len(enq_done), 1)
      self.assertEqual(self.evaluate(size_op), 5)

      # This dequeue will unblock the thread.
      results.append(deq.eval())
      time.sleep(0.1)
      self.assertEqual(len(enq_done), 2)
      thread.join()

      for i in range(5):
        self.assertEqual(size_op.eval(), 5 - i)
        results.append(deq.eval())
        self.assertEqual(size_op.eval(), 5 - i - 1)

      self.assertItemsEqual(elem, results)

  def testBigDequeueMany(self):
    with self.cached_session() as sess:
      q = data_flow_ops.RandomShuffleQueue(2, 0, dtypes_lib.int32, ((),))
      elem = np.arange(4, dtype=np.int32)
      enq_list = [q.enqueue((e,)) for e in elem]
      deq = q.dequeue_many(4)

      results = []

      def blocking_dequeue():
        # Will only complete after 4 enqueues complete.
        results.extend(self.evaluate(deq))

      thread = self.checkedThread(target=blocking_dequeue)
      thread.start()
      # The dequeue should start and then block.
      for enq in enq_list:
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        self.assertEqual(len(results), 0)
        self.evaluate(enq)

      # Enough enqueued to unblock the dequeue
      thread.join()
      self.assertItemsEqual(elem, results)


if __name__ == "__main__":
  test.main()
