# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from tensorflow.python.eager import test
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.profiler import profiler_client


class ProfilerClientTest(test_util.TensorFlowTestCase):

  def testStartTracing_ProcessInvalidAddress(self):
    with self.assertRaises(errors.UnavailableError):
      profiler_client.trace('localhost:6006', tempfile.mkdtemp(), 2000)

  def testMonitor_ProcessInvalidAddress(self):
    with self.assertRaises(errors.UnavailableError):
      profiler_client.monitor('localhost:6006', 2000)


if __name__ == '__main__':
  test.main()
