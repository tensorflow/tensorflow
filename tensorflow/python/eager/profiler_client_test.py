# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for profiler_client."""

from tensorflow.python.eager import profiler_client
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util


class ProfilerClientTest(test_util.TensorFlowTestCase):

  def testStartTracing_ProcessInvalidAddress(self):
    with self.assertRaises(RuntimeError):
      profiler_client.start_tracing('localhost:6006', '/tmp/', 2000)

  def testMonitor_ProcessInvalidAddress(self):
    with self.assertRaises(RuntimeError):
      profiler_client.monitor('localhost:6006', 2000)


if __name__ == '__main__':
  test.main()
