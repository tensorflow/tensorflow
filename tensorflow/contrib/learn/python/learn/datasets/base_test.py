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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.platform import test

mock = test.mock

_TIMEOUT = IOError(110, "timeout")


class BaseTest(test.TestCase):
  """Test load csv functions."""

  def testUrlretrieveRetriesOnIOError(self):
    with mock.patch.object(base, "time") as mock_time:
      with mock.patch.object(base, "urllib") as mock_urllib:
        mock_urllib.request.urlretrieve.side_effect = [
            _TIMEOUT, _TIMEOUT, _TIMEOUT, _TIMEOUT, _TIMEOUT, None
        ]
        base.urlretrieve_with_retry("http://dummy.com", "/tmp/dummy")

    # Assert full backoff was tried
    actual_list = [arg[0][0] for arg in mock_time.sleep.call_args_list]
    expected_list = [1, 2, 4, 8, 16]
    for actual, expected in zip(actual_list, expected_list):
      self.assertLessEqual(abs(actual - expected), 0.25 * expected)
    self.assertEquals(len(actual_list), len(expected_list))

  def testUrlretrieveRaisesAfterRetriesAreExhausted(self):
    with mock.patch.object(base, "time") as mock_time:
      with mock.patch.object(base, "urllib") as mock_urllib:
        mock_urllib.request.urlretrieve.side_effect = [
            _TIMEOUT,
            _TIMEOUT,
            _TIMEOUT,
            _TIMEOUT,
            _TIMEOUT,
            _TIMEOUT,
        ]
        with self.assertRaises(IOError):
          base.urlretrieve_with_retry("http://dummy.com", "/tmp/dummy")

    # Assert full backoff was tried
    actual_list = [arg[0][0] for arg in mock_time.sleep.call_args_list]
    expected_list = [1, 2, 4, 8, 16]
    for actual, expected in zip(actual_list, expected_list):
      self.assertLessEqual(abs(actual - expected), 0.25 * expected)
    self.assertEquals(len(actual_list), len(expected_list))

  def testUrlretrieveRaisesOnNonRetriableErrorWithoutRetry(self):
    with mock.patch.object(base, "time") as mock_time:
      with mock.patch.object(base, "urllib") as mock_urllib:
        mock_urllib.request.urlretrieve.side_effect = [
            IOError(2, "No such file or directory"),
        ]
        with self.assertRaises(IOError):
          base.urlretrieve_with_retry("http://dummy.com", "/tmp/dummy")

    # Assert no retries
    self.assertFalse(mock_time.called)


if __name__ == "__main__":
  test.main()
