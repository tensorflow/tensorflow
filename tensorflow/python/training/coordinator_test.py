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

"""Tests for Coordinator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import threading
import time

import tensorflow as tf


def StopOnEvent(coord, wait_for_stop, set_when_stopped):
  wait_for_stop.wait()
  coord.request_stop()
  set_when_stopped.set()


def RaiseInN(coord, n_secs, ex, report_exception):
  try:
    time.sleep(n_secs)
    raise ex
  except RuntimeError as e:
    if report_exception:
      coord.request_stop(e)
    else:
      coord.request_stop(sys.exc_info())


def RaiseInNUsingContextHandler(coord, n_secs, ex):
  with coord.stop_on_exception():
    time.sleep(n_secs)
    raise ex


def SleepABit(n_secs, coord=None):
  if coord:
    coord.register_thread(threading.current_thread())
  time.sleep(n_secs)


def WaitForThreadsToRegister(coord, num_threads):
  while True:
    with coord._lock:
      if len(coord._registered_threads) == num_threads:
        break
    time.sleep(0.001)


class CoordinatorTest(tf.test.TestCase):

  def testStopAPI(self):
    coord = tf.train.Coordinator()
    self.assertFalse(coord.should_stop())
    self.assertFalse(coord.wait_for_stop(0.01))
    coord.request_stop()
    self.assertTrue(coord.should_stop())
    self.assertTrue(coord.wait_for_stop(0.01))

  def testStopAsync(self):
    coord = tf.train.Coordinator()
    self.assertFalse(coord.should_stop())
    self.assertFalse(coord.wait_for_stop(0.1))
    wait_for_stop_ev = threading.Event()
    has_stopped_ev = threading.Event()
    t = threading.Thread(target=StopOnEvent,
                         args=(coord, wait_for_stop_ev, has_stopped_ev))
    t.start()
    self.assertFalse(coord.should_stop())
    self.assertFalse(coord.wait_for_stop(0.01))
    wait_for_stop_ev.set()
    has_stopped_ev.wait()
    self.assertTrue(coord.wait_for_stop(0.05))
    self.assertTrue(coord.should_stop())

  def testJoin(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=SleepABit, args=(0.01,)),
        threading.Thread(target=SleepABit, args=(0.02,)),
        threading.Thread(target=SleepABit, args=(0.01,))]
    for t in threads:
      t.start()
    coord.join(threads)
    for t in threads:
      self.assertFalse(t.is_alive())

  def testJoinAllRegistered(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=SleepABit, args=(0.01, coord)),
        threading.Thread(target=SleepABit, args=(0.02, coord)),
        threading.Thread(target=SleepABit, args=(0.01, coord))]
    for t in threads:
      t.start()
    WaitForThreadsToRegister(coord, 3)
    coord.join()
    for t in threads:
      self.assertFalse(t.is_alive())

  def testJoinSomeRegistered(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=SleepABit, args=(0.01, coord)),
        threading.Thread(target=SleepABit, args=(0.02,)),
        threading.Thread(target=SleepABit, args=(0.01, coord))]
    for t in threads:
      t.start()
    WaitForThreadsToRegister(coord, 2)
    # threads[1] is not registered we must pass it in.
    coord.join(threads[1:1])
    for t in threads:
      self.assertFalse(t.is_alive())

  def testJoinGraceExpires(self):
    def TestWithGracePeriod(stop_grace_period):
      coord = tf.train.Coordinator()
      wait_for_stop_ev = threading.Event()
      has_stopped_ev = threading.Event()
      threads = [
          threading.Thread(target=StopOnEvent,
                           args=(coord, wait_for_stop_ev, has_stopped_ev)),
          threading.Thread(target=SleepABit, args=(10.0,))]
      for t in threads:
        t.daemon = True
        t.start()
      wait_for_stop_ev.set()
      has_stopped_ev.wait()
      with self.assertRaisesRegexp(RuntimeError, "threads still running"):
        coord.join(threads, stop_grace_period_secs=stop_grace_period)
    TestWithGracePeriod(1e-10)
    TestWithGracePeriod(0.002)
    TestWithGracePeriod(1.0)

  def testJoinRaiseReportExcInfo(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=RaiseInN,
                         args=(coord, 0.01, RuntimeError("First"), False)),
        threading.Thread(target=RaiseInN,
                         args=(coord, 0.05, RuntimeError("Too late"), False))]
    for t in threads:
      t.start()
    with self.assertRaisesRegexp(RuntimeError, "First"):
      coord.join(threads)

  def testJoinRaiseReportException(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=RaiseInN,
                         args=(coord, 0.01, RuntimeError("First"), True)),
        threading.Thread(target=RaiseInN,
                         args=(coord, 0.05, RuntimeError("Too late"), True))]
    for t in threads:
      t.start()
    with self.assertRaisesRegexp(RuntimeError, "First"):
      coord.join(threads)

  def testJoinIgnoresOutOfRange(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=RaiseInN,
                         args=(coord, 0.01,
                               tf.errors.OutOfRangeError(None, None, "First"),
                               True))
        ]
    for t in threads:
      t.start()
    coord.join(threads)

  def testJoinIgnoresMyExceptionType(self):
    coord = tf.train.Coordinator(clean_stop_exception_types=(ValueError,))
    threads = [
        threading.Thread(target=RaiseInN,
                         args=(coord, 0.01, ValueError("Clean stop"), True))
        ]
    for t in threads:
      t.start()
    coord.join(threads)

  def testJoinRaiseReportExceptionUsingHandler(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=RaiseInNUsingContextHandler,
                         args=(coord, 0.01, RuntimeError("First"))),
        threading.Thread(target=RaiseInNUsingContextHandler,
                         args=(coord, 0.05, RuntimeError("Too late")))]
    for t in threads:
      t.start()
    with self.assertRaisesRegexp(RuntimeError, "First"):
      coord.join(threads)

  def testClearStopClearsExceptionToo(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=RaiseInN,
                         args=(coord, 0.01, RuntimeError("First"), True)),
        ]
    for t in threads:
      t.start()
    with self.assertRaisesRegexp(RuntimeError, "First"):
      coord.join(threads)
    coord.clear_stop()
    threads = [
        threading.Thread(target=RaiseInN,
                         args=(coord, 0.01, RuntimeError("Second"), True)),
        ]
    for t in threads:
      t.start()
    with self.assertRaisesRegexp(RuntimeError, "Second"):
      coord.join(threads)

  def testRequestStopRaisesIfJoined(self):
    coord = tf.train.Coordinator()
    # Join the coordinator right away.
    coord.join([])
    reported = False
    with self.assertRaisesRegexp(RuntimeError, "Too late"):
      try:
        raise RuntimeError("Too late")
      except RuntimeError as e:
        reported = True
        coord.request_stop(e)
    self.assertTrue(reported)
    # If we clear_stop the exceptions are handled normally.
    coord.clear_stop()
    try:
      raise RuntimeError("After clear")
    except RuntimeError as e:
      coord.request_stop(e)
    with self.assertRaisesRegexp(RuntimeError, "After clear"):
      coord.join([])

  def testRequestStopRaisesIfJoined_ExcInfo(self):
    # Same as testRequestStopRaisesIfJoined but using syc.exc_info().
    coord = tf.train.Coordinator()
    # Join the coordinator right away.
    coord.join([])
    reported = False
    with self.assertRaisesRegexp(RuntimeError, "Too late"):
      try:
        raise RuntimeError("Too late")
      except RuntimeError:
        reported = True
        coord.request_stop(sys.exc_info())
    self.assertTrue(reported)
    # If we clear_stop the exceptions are handled normally.
    coord.clear_stop()
    try:
      raise RuntimeError("After clear")
    except RuntimeError:
      coord.request_stop(sys.exc_info())
    with self.assertRaisesRegexp(RuntimeError, "After clear"):
      coord.join([])


def _StopAt0(coord, n):
  if n[0] == 0:
    coord.request_stop()
  else:
    n[0] -= 1


class LooperTest(tf.test.TestCase):

  def testTargetArgs(self):
    n = [3]
    coord = tf.train.Coordinator()
    thread = tf.train.LooperThread.loop(coord, 0, target=_StopAt0,
                                        args=(coord, n))
    coord.join([thread])
    self.assertEqual(0, n[0])

  def testTargetKwargs(self):
    n = [3]
    coord = tf.train.Coordinator()
    thread = tf.train.LooperThread.loop(coord, 0, target=_StopAt0,
                                        kwargs={"coord": coord, "n": n})
    coord.join([thread])
    self.assertEqual(0, n[0])

  def testTargetMixedArgs(self):
    n = [3]
    coord = tf.train.Coordinator()
    thread = tf.train.LooperThread.loop(coord, 0, target=_StopAt0,
                                        args=(coord,), kwargs={"n": n})
    coord.join([thread])
    self.assertEqual(0, n[0])


if __name__ == "__main__":
  tf.test.main()
