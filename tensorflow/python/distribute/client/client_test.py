# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for client.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import threading
import time
from absl import logging

from tensorflow.python.distribute.client import client
from tensorflow.python.eager import def_function
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.util import nest


class CoordinatedClosureQueueTest(test.TestCase):

  def testBasic(self):
    queue = client._CoordinatedClosureQueue()
    closure1 = self._create_closure()
    queue.put(closure1)
    self.assertIs(closure1, queue.get())
    self.assertFalse(queue.done())
    queue.put_back(closure1)
    self.assertEqual(closure1, queue.get())
    queue.mark_finished(closure1)
    self.assertTrue(queue.done())
    queue.wait()

  def testProcessAtLeaseOnce(self):
    closure_queue = client._CoordinatedClosureQueue()
    labels = ['A', 'B', 'C', 'D', 'E']
    processed_count = collections.defaultdict(int)

    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def process_queue():
      with coord.stop_on_exception():
        has_been_put_back = False
        while True:
          closure = closure_queue.get(timeout=30)
          if closure is None:
            break
          if not has_been_put_back:
            has_been_put_back = True
            closure_queue.put_back(closure)
            continue
          closure._function()
          closure_queue.mark_finished(closure)

    def get_func(label):

      def func():
        logging.info('Label: %s, before waiting 3 sec', label)
        time.sleep(3)
        processed_count[label] += 1
        logging.info('Label: %s, after waiting 3 sec', label)

      return func

    for label in labels:
      closure_queue.put(client.Closure(get_func(label)))
    t1 = threading.Thread(target=process_queue, daemon=True)
    t1.start()
    t2 = threading.Thread(target=process_queue, daemon=True)
    t2.start()

    # Make sure multiple wait() calls are fine.
    closure_queue.wait()
    closure_queue.wait()
    closure_queue.wait()
    closure_queue.wait()

    self.assertEqual(processed_count, collections.Counter(labels))

    coord.join([t1, t2])

  def testNotifyBeforeWait(self):
    closure_queue = client._CoordinatedClosureQueue()

    def func():
      logging.info('func running')

    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def process_queue():
      with coord.stop_on_exception():
        closure = closure_queue.get()
        closure_queue.mark_finished(closure)

    closure_queue.put(client.Closure(func))
    t = threading.Thread(target=process_queue)
    t.start()
    coord.join([t])

    # This test asserts that waiting at the time the function has been processed
    # doesn't time out.
    closure_queue.wait()

  def testWaitRaiseErrorAfterMarkFailure(self):
    closure_queue = client._CoordinatedClosureQueue()
    closure_queue.put(self._create_closure())
    closure = closure_queue.get()

    wait_finish_event = threading.Event()
    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    # Using a thread to verify that closure_queue.wait() will not return until
    # all inflight closures are finished.

    def mark_finished_fn():
      with coord.stop_on_exception():
        self.assertFalse(wait_finish_event.is_set())
        try:
          raise ValueError('Some error.')
        except ValueError as e:
          closure_queue.mark_failed(e, closure)
        wait_finish_event.wait()

    t = threading.Thread(target=mark_finished_fn)
    t.start()

    with self.assertRaises(ValueError):
      closure_queue.wait()
    wait_finish_event.set()

    coord.join([t])
    self.assertTrue(closure_queue.done())

  def _create_closure(self):

    @def_function.function()
    def some_function():
      return 1.0

    return client.Closure(some_function)

  def _put_two_closures_and_get_one(self):
    closure_queue = client._CoordinatedClosureQueue()
    closure1 = self._create_closure()
    closure_queue.put(closure1)

    closure2 = self._create_closure()
    closure_queue.put(closure2)

    closure_got = closure_queue.get()  # returns closure1
    self.assertIs(closure_got, closure1)
    self.assertIsNot(closure_got, closure2)
    return closure_queue, closure1, closure2

  def testPutRaiseError(self):
    closure_queue, closure1, closure2 = self._put_two_closures_and_get_one()

    closure_queue.mark_failed(ValueError(), closure1)

    with self.assertRaises(ValueError):
      closure_queue.put(self._create_closure())

    self.assertTrue(closure_queue.done())

    with self.assertRaisesRegex(
        client.FunctionRetryableError,
        'The corresponding function is cancelled. Please reschedule the '
        'function.'):
      closure2._fetch_output_remote_values()

    # The error is cleared.
    closure_queue.put(self._create_closure())

  def testWaitRaiseError(self):
    closure_queue, closure1, closure2 = self._put_two_closures_and_get_one()

    closure_queue.mark_failed(ValueError(), closure1)

    with self.assertRaises(ValueError):
      closure_queue.wait()
    self.assertTrue(closure_queue.done())

    with self.assertRaisesRegex(
        client.FunctionRetryableError,
        'The corresponding function is cancelled. Please reschedule the '
        'function.'):
      closure2._fetch_output_remote_values()

    # The error is cleared.
    closure_queue.wait()

  def testDoneRaiseError(self):
    closure_queue, closure1, _ = self._put_two_closures_and_get_one()
    closure_queue.get()

    self.assertFalse(closure_queue.done())
    closure_queue.mark_failed(ValueError(), closure1)
    with self.assertRaises(ValueError):
      closure_queue.done()

  def _test_error_reporting_and_cancel_flow(self, call_wait):
    closure_queue, closure1, closure2 = self._put_two_closures_and_get_one()
    closure_queue.put(self._create_closure())
    closure_queue.get()
    # At this moment, there are two inflight, one in queue.
    self.assertEqual(closure_queue._inflight_closure_count, 2)

    # Simulating closure1 fails.
    try:
      raise ValueError('Some error.')
    except ValueError as e:
      nest.map_structure(lambda x: x._set_error(e),
                         closure1._output_remote_values)
      self.assertEqual(closure_queue._error_generation, 0)  # pylint: disable=g-assert-in-except
      closure_queue.mark_failed(e, closure1)
      self.assertEqual(closure_queue._error_generation, 1)
    # At this moment, there are one inflight, nothing
    # in queue (because the ones in queue should have been removed and
    # cancelled).
    self.assertTrue(closure_queue._queue.empty())
    # Doesn't include out of generation closures.
    self.assertEqual(closure_queue._inflight_closure_count, 1)

    coord = coordinator.Coordinator(clean_stop_exception_types=[])
    closure3 = self._create_closure()

    with self.assertRaises(ValueError):
      # Verifying `wait()` or `put()` raises even if one closure is in
      # flight.
      if call_wait:
        closure_queue.wait()
      else:
        closure_queue.put(closure3)
    # At this moment, there is one inflight, nothing in queue.
    self.assertTrue(closure_queue._queue.empty())
    self.assertEqual(closure_queue._inflight_closure_count, 1)

    # This asserts that closure1 has errored.
    with self.assertRaisesRegex(ValueError, 'Some error.'):
      closure1._fetch_output_remote_values()

    # The following asserts that closure3 should have been cancelled.
    if not call_wait:
      with self.assertRaisesRegex(
          client.FunctionRetryableError,
          'The corresponding function is cancelled. Please reschedule the '
          'function.'):
        closure3._fetch_output_remote_values()

    # Closure2 is inflight, so it shouldn't be ready.
    self.assertEqual(closure2._output_remote_values._status,
                     client._RemoteValueStatus.NOT_READY)

    # And `wait` should block because closure2 is not back yet.
    self.assertFalse(closure_queue.wait(timeout=20))

    # Now let's assume that closure2 isn't successful due to worker preemption,
    # and now it's attempted to be put back, but ends up getting cancelled.
    self.assertEqual(closure2._error_generation, 0)
    self.assertEqual(closure_queue._error_generation, 1)
    closure_queue.put_back(closure2)

    with self.assertRaisesRegex(
        client.FunctionRetryableError,
        'The corresponding function is cancelled. Please reschedule the '
        'function.'):
      closure2._fetch_output_remote_values()

    # At this moment, there is nothing inflight, and the queue is also empty
    # (because closure2 should not be added back to the queue).
    self.assertTrue(closure_queue._queue.empty())
    self.assertEqual(closure_queue._inflight_closure_count, 0)

    closure4 = self._create_closure()

    e = threading.Event()

    def get_fn():
      with coord.stop_on_exception():
        # This should end up getting closure4, not closure2, because closure2
        # has been cancelled and should not be got.
        closure_got = closure_queue.get()
        e.set()
        self.assertEqual(closure_got._error_generation, 1)
        self.assertEqual(closure_queue._error_generation, 1)
        self.assertIs(closure4, closure_got)
        self.assertIsNot(closure2, closure_got)

    t = threading.Thread(target=get_fn)
    t.start()

    time.sleep(10)

    # Make sure `closure_got = closure_queue.get()` is unblocked as a result of
    # `closure_queue.put(closure4)`.
    self.assertFalse(e.is_set())
    closure_queue.put(closure4)
    self.assertTrue(e.wait())
    coord.join([t])

    self.assertEqual(closure_queue._inflight_closure_count, 1)
    closure_queue.mark_finished(closure4)
    # The queue is now cleared and nothing inflight.
    self.assertEqual(closure_queue._inflight_closure_count, 0)
    closure_queue.wait()

  def testWaitRaiseErrorAfterAnErrorIsReported(self):
    self._test_error_reporting_and_cancel_flow(call_wait=True)

  def testPutRaiseErrorAfterAnErrorIsReported(self):
    self._test_error_reporting_and_cancel_flow(call_wait=False)

  def testStateIsRestoredAfterJoinIsCalled(self):
    closure_queue, closure1, closure2 = self._put_two_closures_and_get_one()
    closure_queue.get()
    self.assertEqual(closure_queue._inflight_closure_count, 2)
    closure_queue.mark_failed(ValueError('test error'), closure1)
    with self.assertRaises(ValueError):
      closure_queue.put(self._create_closure())
    closure_queue.mark_failed(ValueError('test error'), closure2)

    # closure2's error is previous generation so should not raise at this
    # following put, and _error should have been cleared.
    self.assertIsNone(closure_queue._error)
    closure_queue.put(self._create_closure())
    self.assertIsNone(closure_queue._error)

  def testStateIsRestoredAfterJoinIsCalled_WaitShouldReturn(self):
    closure_queue, closure1, closure2 = self._put_two_closures_and_get_one()
    closure_queue.put(self._create_closure())
    closure_queue.get()  # got closure2
    self.assertFalse(closure_queue._queue.empty())  # still has closure3
    self.assertEqual(closure_queue._inflight_closure_count, 2)  # closure1,2
    closure_queue.mark_failed(ValueError('test error'), closure1)
    self.assertTrue(closure_queue._queue.empty())  # closure3 cancelled
    self.assertEqual(closure_queue._inflight_closure_count, 1)
    with self.assertRaises(ValueError):
      closure_queue.wait()  # reports error from closure1

    # `wait` should block because closure2 is not back yet, even if closure2
    # was sent inflight before the error.
    self.assertFalse(closure_queue.wait(timeout=20))
    self.assertEqual(closure_queue._inflight_closure_count, 1)
    closure_queue.mark_finished(closure2)
    closure_queue.wait()  # wait should pass immediately
    self.assertEqual(closure_queue._inflight_closure_count, 0)

  def testThreadSafey(self):
    thread_count = 10
    queue = client._CoordinatedClosureQueue()

    # Each thread performs 20 queue actions: 10 are `put_back` and 10 are
    # `mark_finished`.
    action_count = 20

    def func():
      for i in range(action_count):
        closure = queue.get()
        if i % 2 == 0:
          queue.put_back(closure)
        else:
          queue.mark_finished(closure)

    threads = [threading.Thread(target=func) for i in range(thread_count)]
    for t in threads:
      t.start()

    for _ in range(thread_count * action_count // 2):
      queue.put(self._create_closure())
    queue.wait()
    self.assertTrue(queue.done())


if __name__ == '__main__':
  test.main()
