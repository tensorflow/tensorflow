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
"""Tests for tf.contrib.training.failure_tolerator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.contrib.training.python.training import failure_tolerator
from tensorflow.python.platform import test


class ForgiveMe(Exception):
  pass


class Unforgivable(Exception):
  pass


class FailureToleratorTest(test.TestCase):
  # Tests for the FailureTolerator helper

  def testHandledExceptions(self):
    tolerator = failure_tolerator.FailureTolerator(
        init_delay=0.0, handled_exceptions=[ForgiveMe])

    with tolerator.forgive():
      raise ForgiveMe()

    with self.assertRaises(Unforgivable):
      with tolerator.forgive():
        raise Unforgivable()

  def testLimit(self):
    tolerator = failure_tolerator.FailureTolerator(
        init_delay=0.0, limit=3, handled_exceptions=[ForgiveMe])

    with tolerator.forgive():
      raise ForgiveMe()
    with tolerator.forgive():
      raise ForgiveMe()

    with self.assertRaises(ForgiveMe):
      with tolerator.forgive():
        raise ForgiveMe()

  def testDelaysExponentially(self):
    # Tests that delays are appropriate, with exponential backoff.
    tolerator = failure_tolerator.FailureTolerator(
        init_delay=1.0, backoff_factor=1.5, handled_exceptions=[ForgiveMe])

    with test.mock.patch.object(time, 'sleep') as mock_sleep:
      with tolerator.forgive():
        raise ForgiveMe()

      with tolerator.forgive():
        raise ForgiveMe()

      with tolerator.forgive():
        raise ForgiveMe()

      with tolerator.forgive():
        raise ForgiveMe()

      mock_sleep.assert_has_calls(
          [test.mock.call(1.0), test.mock.call(1.5), test.mock.call(2.25)],
          any_order=False)
      self.assertEquals(3, mock_sleep.call_count)

  def testForgivesSuccessfully(self):
    # Tests that exceptions are forgiven after forgive_after_seconds
    tolerator = failure_tolerator.FailureTolerator(
        limit=3,
        init_delay=0.0,
        backoff_factor=1.0,  # no exponential backoff
        forgive_after_seconds=10.0,
        handled_exceptions=[ForgiveMe])

    cur_time = 10.0

    with test.mock.patch.object(time, 'time') as mock_time:
      mock_time.side_effect = lambda: cur_time

      with tolerator.forgive():
        raise ForgiveMe()
      cur_time = 15.0
      with tolerator.forgive():
        raise ForgiveMe()

      cur_time = 20.1  # advance more than forgive_after_seconds

      with tolerator.forgive():
        raise ForgiveMe()  # should not be raised

      cur_time = 24.0

      with self.assertRaises(ForgiveMe):
        with tolerator.forgive():
          raise ForgiveMe()  # third exception in < 10secs (t=15, 20.1, 24)

  def testForgivesDoesNotCountDelays(self):
    tolerator = failure_tolerator.FailureTolerator(
        limit=3,
        init_delay=1.0,
        backoff_factor=1.0,  # no exponential backoff
        forgive_after_seconds=10.0,
        handled_exceptions=[ForgiveMe])

    cur_time = [10.0]

    def _sleep(x):
      cur_time[0] += x

    with test.mock.patch.object(time, 'sleep') as mock_sleep:
      with test.mock.patch.object(time, 'time') as mock_time:
        mock_time.side_effect = lambda: cur_time[0]
        mock_sleep.side_effect = _sleep

        with tolerator.forgive():
          raise ForgiveMe()

        cur_time[0] += 1.0

        with tolerator.forgive():
          raise ForgiveMe()

        self.assertEquals(12.0, time.time())  # ensure there was a sleep

        cur_time[0] = 20.1  # 10.1 seconds after the first failure!

        with self.assertRaises(ForgiveMe):
          with tolerator.forgive():
            raise ForgiveMe()


if __name__ == '__main__':
  test.main()
