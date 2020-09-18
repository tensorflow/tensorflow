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
import platform
import sys
import threading
import time
from absl import logging

from tensorflow.python.distribute.client import client
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import def_function
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.util import nest


class CoordinatedClosureQueueTest(test.TestCase):

  def testBasic(self):
    queue = client._CoordinatedClosureQueue()
    closure1 = self._create_closure(queue._cancellation_mgr)
    queue.put(closure1)
    self.assertIs(closure1, queue.get())
    self.assertFalse(queue.done())
    queue.put_back(closure1)
    self.assertEqual(closure1, queue.get())
    queue.mark_finished()
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
          closure_queue.mark_finished()

    def get_func(label):

      def func():
        time.sleep(3)
        processed_count[label] += 1

      return func

    cm = cancellation.CancellationManager()
    for label in labels:
      closure_queue.put(client.Closure(get_func(label), cm))
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
        closure_queue.get()
        closure_queue.mark_finished()

    closure_queue.put(client.Closure(func, closure_queue._cancellation_mgr))
    t = threading.Thread(target=process_queue)
    t.start()
    coord.join([t])

    # This test asserts that waiting at the time the function has been processed
    # doesn't time out.
    closure_queue.wait()

  def _assert_one_unblock_the_other(self, first_fn, second_fn):
    """Asserts `second_fn` wouldn't return before `first_fn` is finished."""
    first_fn_done = threading.Event()
    second_fn_done = threading.Event()
    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    def wrapped_first_fn():
      with coord.stop_on_exception():
        self.assertFalse(second_fn_done.is_set())
        first_fn()
        first_fn_done.set()

    self.assertFalse(first_fn_done.is_set())
    t = threading.Thread(target=wrapped_first_fn)
    t.start()

    second_fn()
    self.assertTrue(first_fn_done.is_set())
    second_fn_done.set()

    coord.join([t])

  def testWaitRaiseErrorAfterMarkFailure(self):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue = client._CoordinatedClosureQueue()
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))
    closure = closure_queue.get()

    wait_finish_event = threading.Event()
    coord = coordinator.Coordinator(clean_stop_exception_types=[])

    # Using a thread to verify that closure_queue.wait() will not return until
    # all inflight closures are finished.

    def mark_finished_fn():
      try:
        raise ValueError('Some error.')
      except ValueError as e:
        closure_queue.mark_failed(e)

    def wait_fn():
      with self.assertRaises(ValueError):
        closure_queue.wait()

    self._assert_one_unblock_the_other(mark_finished_fn, wait_fn)

    self.assertTrue(closure_queue.done())

  def _create_closure(self, cancellation_mgr):

    @def_function.function()
    def some_function():
      return 1.0

    return client.Closure(some_function, cancellation_mgr)

  def _put_two_closures_and_get_one(self):
    closure_queue = client._CoordinatedClosureQueue()
    closure1 = self._create_closure(closure_queue._cancellation_mgr)
    closure_queue.put(closure1)

    closure2 = self._create_closure(closure_queue._cancellation_mgr)
    closure_queue.put(closure2)

    closure_got = closure_queue.get()  # returns closure1
    self.assertIs(closure_got, closure1)
    self.assertIsNot(closure_got, closure2)
    return closure_queue, closure1, closure2

  def testPutRaiseError(self):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, closure2 = self._put_two_closures_and_get_one()

    closure_queue.mark_failed(ValueError())

    with self.assertRaises(ValueError):
      closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))

    self.assertTrue(closure_queue.done())

    with self.assertRaisesRegex(
        client.FunctionRetryableError,
        'The corresponding function is cancelled. Please reschedule the '
        'function.'):
      closure2._fetch_output_remote_values()

    # The error is cleared.
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))

  def testWaitRaiseError(self):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, closure2 = self._put_two_closures_and_get_one()

    closure_queue.mark_failed(ValueError())

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
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, _ = self._put_two_closures_and_get_one()

    self.assertFalse(closure_queue.done())
    closure_queue.mark_failed(ValueError())
    with self.assertRaises(ValueError):
      closure_queue.done()

  def _set_error(self, closure_queue, closure, error):
    try:
      raise error
    except Exception as e:  # pylint: disable=broad-except
      nest.map_structure(lambda x: x._set_error(e),
                         closure._output_remote_values)
      closure_queue.mark_failed(e)

  def _test_cancel_closure_when_error(self, call_wait):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, closure1, closure2 = self._put_two_closures_and_get_one()
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))
    closure_queue.get()
    # At this moment, there are two inflight, one in queue.
    self.assertEqual(closure_queue._inflight_closure_count, 2)

    # Hold a copy of the queue's cancellation manager at this point
    initial_cm = closure_queue._cancellation_mgr

    # Simulating closure1 fails.
    self._set_error(closure_queue, closure1, ValueError('Some error.'))

    # At this moment, there are one inflight, one in queue.
    self.assertEqual(closure_queue._queue.qsize(), 1)
    self.assertEqual(closure_queue._inflight_closure_count, 1)

    closure3 = self._create_closure(closure_queue._cancellation_mgr)

    def fake_cancellation():
      self._set_error(closure_queue, closure2,
                      ValueError('Fake cancellation error.'))

    def report_error():
      # It should not report the fake cancellation error.
      with self.assertRaisesRegex(ValueError, 'Some error.'):
        # Verifying `wait()` or `put()` raises even if one closure is in
        # flight.
        if call_wait:
          closure_queue.wait()
        else:
          closure_queue.put(closure3)

    self._assert_one_unblock_the_other(fake_cancellation, report_error)

    # The original cancellation manager of the queue has been cancelled.
    self.assertTrue(initial_cm.is_cancelled)

    # At this moment, there is zero inflight, nothing in queue.
    self.assertTrue(closure_queue._queue.empty())
    self.assertEqual(closure_queue._inflight_closure_count, 0)
    self.assertIsNone(closure_queue._error)

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

    # Closure2 was an inflight closure when it got cancelled.
    self.assertEqual(closure2._output_remote_values._status,
                     client._RemoteValueStatus.READY)
    with self.assertRaisesRegex(ValueError, 'Fake cancellation error.'):
      closure2._fetch_output_remote_values()

    # This asserts that the queue has a clear state.
    self.testBasic()

  def testWaitRaiseErrorAfterCancelClosure(self):
    self._test_cancel_closure_when_error(call_wait=True)

  def testPutRaiseErrorAfterCancelClosure(self):
    self._test_cancel_closure_when_error(call_wait=False)

  def testStateIsRestoredAfterJoinIsCalled(self):
    if sys.version_info >= (3, 8) and platform.system() == 'Windows':
      # TODO(b/165013260): Fix this
      self.skipTest('Test is currently broken on Windows with Python 3.8')

    closure_queue, _, _ = self._put_two_closures_and_get_one()
    self.assertEqual(closure_queue._inflight_closure_count, 1)
    closure_queue.mark_failed(ValueError('test error'))
    with self.assertRaises(ValueError):
      closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))

    # Its error should have been cleared.
    self.assertIsNone(closure_queue._error)
    closure_queue.put(self._create_closure(closure_queue._cancellation_mgr))
    self.assertIsNone(closure_queue._error)

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
          queue.mark_finished()

    threads = [threading.Thread(target=func) for i in range(thread_count)]
    for t in threads:
      t.start()

    for _ in range(thread_count * action_count // 2):
      queue.put(self._create_closure(queue._cancellation_mgr))
    queue.wait()
    self.assertTrue(queue.done())


if __name__ == '__main__':
  test.main()
