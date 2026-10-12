# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.dtensor.python.heartbeat."""

from unittest import mock

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import heartbeat
from tensorflow.python.platform import test as tf_test
# pylint: enable=g-direct-tensorflow-import


class HeartbeatFatalTest(tf_test.TestCase):

  def test_fatal_crashes_the_process(self):
    with mock.patch.object(heartbeat.os, '_exit') as mock_exit:
      with mock.patch.object(heartbeat.logging, 'fatal') as mock_log_fatal:
        heartbeat._fatal('Heartbeat failure %d, limit of %d reached: %s',
                         1, 3, 'boom')

    mock_log_fatal.assert_called_once_with(
        'Heartbeat failure %d, limit of %d reached: %s', 1, 3, 'boom')
    mock_exit.assert_called_once_with(1)

  def test_fatal_still_crashes_if_logging_raises(self):
    # _fatal must guarantee a process crash even if the logging call itself
    # fails, since this is the last-resort path for a desynced worker.
    with mock.patch.object(heartbeat.os, '_exit') as mock_exit:
      with mock.patch.object(
          heartbeat.logging, 'fatal', side_effect=ValueError('bad format')):
        with self.assertRaises(ValueError):
          heartbeat._fatal('Unexpected heartbeat signal received: %s', [1, 0])

    mock_exit.assert_not_called()

  def test_fatal_forwards_kwargs(self):
    with mock.patch.object(heartbeat.os, '_exit'):
      with mock.patch.object(heartbeat.logging, 'fatal') as mock_log_fatal:
        heartbeat._fatal('msg', exc_info=True)

    mock_log_fatal.assert_called_once_with('msg', exc_info=True)


if __name__ == '__main__':
  tf_test.main()
