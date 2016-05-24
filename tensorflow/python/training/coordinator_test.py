# Copyright 2015 Google Inc. All Rights Reserved.
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


def StopInN(coord, n_secs):
  time.sleep(n_secs)
  coord.request_stop()


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


def SleepABit(n_secs):
  time.sleep(n_secs)


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
    threading.Thread(target=StopInN, args=(coord, 0.02)).start()
    self.assertFalse(coord.should_stop())
    self.assertFalse(coord.wait_for_stop(0.01))
    self.assertTrue(coord.wait_for_stop(0.03))
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

  def testJoinGraceExpires(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=StopInN, args=(coord, 0.01)),
        threading.Thread(target=SleepABit, args=(10.0,))]
    for t in threads:
      t.daemon = True
      t.start()
    with self.assertRaisesRegexp(RuntimeError, "threads still running"):
      coord.join(threads, stop_grace_period_secs=0.02)

  def testJoinRaiseReportExcInfo(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=RaiseInN,
                         args=(coord, 0.01, RuntimeError("First"), False)),
        threading.Thread(target=RaiseInN,
                         args=(coord, 0.02, RuntimeError("Too late"), False))]
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
                         args=(coord, 0.02, RuntimeError("Too late"), True))]
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

  def testJoinRaiseReportExceptionUsingHandler(self):
    coord = tf.train.Coordinator()
    threads = [
        threading.Thread(target=RaiseInNUsingContextHandler,
                         args=(coord, 0.01, RuntimeError("First"))),
        threading.Thread(target=RaiseInNUsingContextHandler,
                         args=(coord, 0.02, RuntimeError("Too late")))]
    for t in threads:
      t.start()
    with self.assertRaisesRegexp(RuntimeError, "First"):
      coord.join(threads)


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
