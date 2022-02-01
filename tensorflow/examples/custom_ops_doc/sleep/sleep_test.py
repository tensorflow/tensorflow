# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sleep."""

import time

import tensorflow as tf

from tensorflow.examples.custom_ops_doc.sleep import sleep_op
# This pylint disable is only needed for internal google users
from tensorflow.python.framework import errors_impl  # pylint: disable=g-direct-tensorflow-import


class SleepTest(tf.test.TestCase):

  def _check_sleep(self, op):
    """Check that one sleep op works in isolation.

    See sleep_bin.py for an example of how the synchronous and asynchronous
    sleep ops differ in behavior.

    Args:
      op: The sleep op, either sleep_op.SyncSleep or sleep_op.AsyncSleep.
    """
    delay = 0.3  # delay in seconds
    start_t = time.time()
    func = tf.function(lambda: op(delay))
    results = self.evaluate(func())
    end_t = time.time()
    delta_t = end_t - start_t
    self.assertEqual(results.shape, tuple())
    self.assertGreater(delta_t, 0.9 * delay)

  def test_sync_sleep(self):
    self._check_sleep(sleep_op.SyncSleep)

  def test_async_sleep(self):
    self._check_sleep(sleep_op.AsyncSleep)

  def test_async_sleep_error(self):
    # It is import that ComputeAsync() calls its done() callback if it returns
    # early due to an error.
    func = tf.function(lambda: sleep_op.AsyncSleep(-1.0))
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                'Input `delay` must be non-negative.'):
      self.evaluate(func())


if __name__ == '__main__':
  tf.test.main()
