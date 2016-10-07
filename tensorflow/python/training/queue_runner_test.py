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

"""Tests for QueueRunner."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf


class QueueRunnerTest(tf.test.TestCase):

  def _wait_for_thread_registration(self, coord, N):
    """Wait for N threads to register with the coordinator.

    This is necessary in some tests that launch threads and
    then want to join() them in the coordinator.

    Args:
      coord: A Coordinator object.
      N: Number of threads to wait for.
    """
    while len(coord._registered_threads) < N:
      time.sleep(0.001)

  def testBasic(self):
    with self.test_session() as sess:
      # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
      zero64 = tf.constant(0, dtype=tf.int64)
      var = tf.Variable(zero64)
      count_up_to = var.count_up_to(3)
      queue = tf.FIFOQueue(10, tf.float32)
      tf.initialize_all_variables().run()
      qr = tf.train.QueueRunner(queue, [count_up_to])
      threads = qr.create_threads(sess)
      for t in threads:
        t.start()
      for t in threads:
        t.join()
      self.assertEqual(0, len(qr.exceptions_raised))
      # The variable should be 3.
      self.assertEqual(3, var.eval())

  def testTwoOps(self):
    with self.test_session() as sess:
      # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
      zero64 = tf.constant(0, dtype=tf.int64)
      var0 = tf.Variable(zero64)
      count_up_to_3 = var0.count_up_to(3)
      var1 = tf.Variable(zero64)
      count_up_to_30 = var1.count_up_to(30)
      queue = tf.FIFOQueue(10, tf.float32)
      qr = tf.train.QueueRunner(queue, [count_up_to_3, count_up_to_30])
      threads = qr.create_threads(sess)
      tf.initialize_all_variables().run()
      for t in threads:
        t.start()
      for t in threads:
        t.join()
      self.assertEqual(0, len(qr.exceptions_raised))
      self.assertEqual(3, var0.eval())
      self.assertEqual(30, var1.eval())

  def testExceptionsCaptured(self):
    with self.test_session() as sess:
      queue = tf.FIFOQueue(10, tf.float32)
      qr = tf.train.QueueRunner(queue, ["i fail", "so fail"])
      threads = qr.create_threads(sess)
      tf.initialize_all_variables().run()
      for t in threads:
        t.start()
      for t in threads:
        t.join()
      exceptions = qr.exceptions_raised
      self.assertEqual(2, len(exceptions))
      self.assertTrue("Operation not in the graph" in str(exceptions[0]))
      self.assertTrue("Operation not in the graph" in str(exceptions[1]))

  def testRealDequeueEnqueue(self):
    with self.test_session() as sess:
      q0 = tf.FIFOQueue(3, tf.float32)
      enqueue0 = q0.enqueue((10.0,))
      close0 = q0.close()
      q1 = tf.FIFOQueue(30, tf.float32)
      enqueue1 = q1.enqueue((q0.dequeue(),))
      dequeue1 = q1.dequeue()
      qr = tf.train.QueueRunner(q1, [enqueue1])
      threads = qr.create_threads(sess)
      for t in threads:
        t.start()
      # Enqueue 2 values, then close queue0.
      enqueue0.run()
      enqueue0.run()
      close0.run()
      # Wait for the queue runner to terminate.
      for t in threads:
        t.join()
      # It should have terminated cleanly.
      self.assertEqual(0, len(qr.exceptions_raised))
      # The 2 values should be in queue1.
      self.assertEqual(10.0, dequeue1.eval())
      self.assertEqual(10.0, dequeue1.eval())
      # And queue1 should now be closed.
      with self.assertRaisesRegexp(tf.errors.OutOfRangeError, "is closed"):
        dequeue1.eval()

  def testRespectCoordShouldStop(self):
    with self.test_session() as sess:
      # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
      zero64 = tf.constant(0, dtype=tf.int64)
      var = tf.Variable(zero64)
      count_up_to = var.count_up_to(3)
      queue = tf.FIFOQueue(10, tf.float32)
      tf.initialize_all_variables().run()
      qr = tf.train.QueueRunner(queue, [count_up_to])
      # As the coordinator to stop.  The queue runner should
      # finish immediately.
      coord = tf.train.Coordinator()
      coord.request_stop()
      threads = qr.create_threads(sess, coord)
      for t in threads:
        t.start()
      self._wait_for_thread_registration(coord, len(threads))
      coord.join()
      self.assertEqual(0, len(qr.exceptions_raised))
      # The variable should be 0.
      self.assertEqual(0, var.eval())

  def testRequestStopOnException(self):
    with self.test_session() as sess:
      queue = tf.FIFOQueue(10, tf.float32)
      qr = tf.train.QueueRunner(queue, ["not an op"])
      coord = tf.train.Coordinator()
      threads = qr.create_threads(sess, coord)
      for t in threads:
        t.start()
      self._wait_for_thread_registration(coord, len(threads))
      # The exception should be re-raised when joining.
      with self.assertRaisesRegexp(ValueError, "Operation not in the graph"):
        coord.join()

  def testGracePeriod(self):
    with self.test_session() as sess:
      # The enqueue will quickly block.
      queue = tf.FIFOQueue(2, tf.float32)
      enqueue = queue.enqueue((10.0,))
      dequeue = queue.dequeue()
      qr = tf.train.QueueRunner(queue, [enqueue])
      coord = tf.train.Coordinator()
      threads = qr.create_threads(sess, coord, start=True)
      # Wait for the threads to have registered with the coordinator.
      self._wait_for_thread_registration(coord, len(threads))
      # Dequeue one element and then request stop.
      dequeue.op.run()
      time.sleep(0.02)
      coord.request_stop()
      # We should be able to join because the RequestStop() will cause
      # the queue to be closed and the enqueue to terminate.
      coord.join(stop_grace_period_secs=0.05)

  def testIgnoreMultiStarts(self):
    with self.test_session() as sess:
      # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
      zero64 = tf.constant(0, dtype=tf.int64)
      var = tf.Variable(zero64)
      count_up_to = var.count_up_to(3)
      queue = tf.FIFOQueue(10, tf.float32)
      tf.initialize_all_variables().run()
      coord = tf.train.Coordinator()
      qr = tf.train.QueueRunner(queue, [count_up_to])
      threads = []
      # NOTE that this test does not actually start the threads.
      threads.extend(qr.create_threads(sess, coord=coord))
      new_threads = qr.create_threads(sess, coord=coord)
      self.assertEqual([], new_threads)

  def testThreads(self):
    with self.test_session() as sess:
      # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
      zero64 = tf.constant(0, dtype=tf.int64)
      var = tf.Variable(zero64)
      count_up_to = var.count_up_to(3)
      queue = tf.FIFOQueue(10, tf.float32)
      tf.initialize_all_variables().run()
      qr = tf.train.QueueRunner(queue, [count_up_to, "bad op"])
      threads = qr.create_threads(sess, start=True)
      for t in threads:
        t.join()
      exceptions = qr.exceptions_raised
      self.assertEqual(1, len(exceptions))
      self.assertTrue("Operation not in the graph" in str(exceptions[0]))

      threads = qr.create_threads(sess, start=True)
      for t in threads:
        t.join()
      exceptions = qr.exceptions_raised
      self.assertEqual(1, len(exceptions))
      self.assertTrue("Operation not in the graph" in str(exceptions[0]))

  def testName(self):
    with tf.name_scope("scope"):
      queue = tf.FIFOQueue(10, tf.float32, name="queue")
    qr = tf.train.QueueRunner(queue, [tf.no_op()])
    self.assertEqual("scope/queue", qr.name)
    tf.train.add_queue_runner(qr)
    self.assertEqual(1, len(tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS,
                                              "scope")))

  def testStartQueueRunners(self):
    # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
    zero64 = tf.constant(0, dtype=tf.int64)
    var = tf.Variable(zero64)
    count_up_to = var.count_up_to(3)
    queue = tf.FIFOQueue(10, tf.float32)
    init_op = tf.initialize_all_variables()
    qr = tf.train.QueueRunner(queue, [count_up_to])
    tf.train.add_queue_runner(qr)
    with self.test_session() as sess:
      init_op.run()
      threads = tf.train.start_queue_runners(sess)
      for t in threads:
        t.join()
      self.assertEqual(0, len(qr.exceptions_raised))
      # The variable should be 3.
      self.assertEqual(3, var.eval())

  def testStartQueueRunnersNonDefaultGraph(self):
    # CountUpTo will raise OUT_OF_RANGE when it reaches the count.
    graph = tf.Graph()
    with graph.as_default():
      zero64 = tf.constant(0, dtype=tf.int64)
      var = tf.Variable(zero64)
      count_up_to = var.count_up_to(3)
      queue = tf.FIFOQueue(10, tf.float32)
      init_op = tf.initialize_all_variables()
      qr = tf.train.QueueRunner(queue, [count_up_to])
      tf.train.add_queue_runner(qr)
    with self.test_session(graph=graph) as sess:
      init_op.run()
      threads = tf.train.start_queue_runners(sess)
      for t in threads:
        t.join()
      self.assertEqual(0, len(qr.exceptions_raised))
      # The variable should be 3.
      self.assertEqual(3, var.eval())

  def testQueueRunnerSerializationRoundTrip(self):
    graph = tf.Graph()
    with graph.as_default():
      queue = tf.FIFOQueue(10, tf.float32, name="queue")
      enqueue_op = tf.no_op(name="enqueue")
      close_op = tf.no_op(name="close")
      cancel_op = tf.no_op(name="cancel")
      qr0 = tf.train.QueueRunner(
          queue, [enqueue_op], close_op, cancel_op,
          queue_closed_exception_types=(
              tf.errors.OutOfRangeError, tf.errors.CancelledError))
      qr0_proto = tf.train.QueueRunner.to_proto(qr0)
      qr0_recon = tf.train.QueueRunner.from_proto(qr0_proto)
      self.assertEqual("queue", qr0_recon.queue.name)
      self.assertEqual(1, len(qr0_recon.enqueue_ops))
      self.assertEqual(enqueue_op, qr0_recon.enqueue_ops[0])
      self.assertEqual(close_op, qr0_recon.close_op)
      self.assertEqual(cancel_op, qr0_recon.cancel_op)
      self.assertEqual(
          (tf.errors.OutOfRangeError, tf.errors.CancelledError),
          qr0_recon.queue_closed_exception_types)

      # Assert we reconstruct an OutOfRangeError for QueueRunners
      # created before QueueRunnerDef had a queue_closed_exception_types field.
      del qr0_proto.queue_closed_exception_types[:]
      qr0_legacy_recon = tf.train.QueueRunner.from_proto(qr0_proto)
      self.assertEqual("queue", qr0_legacy_recon.queue.name)
      self.assertEqual(1, len(qr0_legacy_recon.enqueue_ops))
      self.assertEqual(enqueue_op, qr0_legacy_recon.enqueue_ops[0])
      self.assertEqual(close_op, qr0_legacy_recon.close_op)
      self.assertEqual(cancel_op, qr0_legacy_recon.cancel_op)
      self.assertEqual(
          (tf.errors.OutOfRangeError,),
          qr0_legacy_recon.queue_closed_exception_types)


if __name__ == "__main__":
  tf.test.main()
