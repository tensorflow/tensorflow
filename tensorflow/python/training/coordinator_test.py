"""Tests for Coordinator."""
import sys
import threading
import time

import tensorflow.python.platform

import tensorflow as tf


def StopInN(coord, n_secs):
  time.sleep(n_secs)
  coord.request_stop()


def RaiseInN(coord, n_secs, ex, report_exception):
  try:
    time.sleep(n_secs)
    raise ex
  except RuntimeError, e:
    if report_exception:
      coord.request_stop(e)
    else:
      coord.request_stop(sys.exc_info())


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


if __name__ == "__main__":
  tf.test.main()
