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
"""Tests for tensorflow.ops.data_flow_ops.FIFOQueue."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


class FIFOQueueTest(xla_test.XLATestCase):

  def testEnqueue(self):
    with self.cached_session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      enqueue_op.run()

  def testEnqueueWithShape(self):
    with self.cached_session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shapes=(3, 2))
      enqueue_correct_op = q.enqueue(([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],))
      enqueue_correct_op.run()
      with self.assertRaises(ValueError):
        q.enqueue(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],))
      self.assertEqual(1, q.size().eval())

  def testMultipleDequeues(self):
    with self.cached_session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      self.evaluate(q.enqueue([1]))
      self.evaluate(q.enqueue([2]))
      self.evaluate(q.enqueue([3]))
      a, b, c = self.evaluate([q.dequeue(), q.dequeue(), q.dequeue()])
      self.assertAllEqual(set([1, 2, 3]), set([a, b, c]))

  def testQueuesDontShare(self):
    with self.cached_session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      self.evaluate(q.enqueue(1))
      q2 = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
      self.evaluate(q2.enqueue(2))
      self.assertAllEqual(self.evaluate(q2.dequeue()), 2)
      self.assertAllEqual(self.evaluate(q.dequeue()), 1)

  def testEnqueueDictWithoutNames(self):
    with self.cached_session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      with self.assertRaisesRegexp(ValueError, "must have names"):
        q.enqueue({"a": 12.0})

  def testParallelEnqueue(self):
    with self.cached_session() as sess, self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Run one producer thread for each element in elems.
      def enqueue(enqueue_op):
        sess.run(enqueue_op)

      threads = [
          self.checkedThread(target=enqueue, args=(e,)) for e in enqueue_ops
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
    with self.cached_session() as sess, self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      # Enqueue every element using a single thread.
      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      # Run one consumer thread for each element in elems.
      results = []

      def dequeue():
        results.append(sess.run(dequeued_t))

      threads = [self.checkedThread(target=dequeue) for _ in enqueue_ops]
      for thread in threads:
        thread.start()
      for thread in threads:
        thread.join()
      self.assertItemsEqual(elems, results)

  def testDequeue(self):
    with self.cached_session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      for i in xrange(len(elems)):
        vals = dequeued_t.eval()
        self.assertEqual([elems[i]], vals)

  def testEnqueueAndBlockingDequeue(self):
    with self.cached_session() as sess, self.test_scope():
      q = data_flow_ops.FIFOQueue(3, dtypes_lib.float32)
      elems = [10.0, 20.0, 30.0]
      enqueue_ops = [q.enqueue((x,)) for x in elems]
      dequeued_t = q.dequeue()

      def enqueue():
        # The enqueue_ops should run after the dequeue op has blocked.
        # TODO(mrry): Figure out how to do this without sleeping.
        time.sleep(0.1)
        for enqueue_op in enqueue_ops:
          sess.run(enqueue_op)

      results = []

      def dequeue():
        for _ in xrange(len(elems)):
          results.append(sess.run(dequeued_t))

      enqueue_thread = self.checkedThread(target=enqueue)
      dequeue_thread = self.checkedThread(target=dequeue)
      enqueue_thread.start()
      dequeue_thread.start()
      enqueue_thread.join()
      dequeue_thread.join()

      for elem, result in zip(elems, results):
        self.assertEqual([elem], result)

  def testMultiEnqueueAndDequeue(self):
    with self.cached_session() as sess, self.test_scope():
      q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.float32))
      elems = [(5, 10.0), (10, 20.0), (15, 30.0)]
      enqueue_ops = [q.enqueue((x, y)) for x, y in elems]
      dequeued_t = q.dequeue()

      for enqueue_op in enqueue_ops:
        enqueue_op.run()

      for i in xrange(len(elems)):
        x_val, y_val = sess.run(dequeued_t)
        x, y = elems[i]
        self.assertEqual([x], x_val)
        self.assertEqual([y], y_val)

  def testQueueSizeEmpty(self):
    with self.cached_session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      self.assertEqual([0], q.size().eval())

  def testQueueSizeAfterEnqueueAndDequeue(self):
    with self.cached_session(), self.test_scope():
      q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
      enqueue_op = q.enqueue((10.0,))
      dequeued_t = q.dequeue()
      size = q.size()
      self.assertEqual([], size.get_shape())

      enqueue_op.run()
      self.assertEqual(1, size.eval())
      dequeued_t.op.run()
      self.assertEqual(0, size.eval())


if __name__ == "__main__":
  test.main()
