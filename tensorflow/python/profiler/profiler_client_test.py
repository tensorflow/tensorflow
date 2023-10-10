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

import portpicker

from tensorflow.python.eager import test
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.profiler import profiler_client
from tensorflow.python.profiler import profiler_v2 as profiler


class ProfilerClientTest(test_util.TensorFlowTestCase):

  def testTrace_ProfileIdleServer(self):
    test_port = portpicker.pick_unused_port()
    profiler.start_server(test_port)
    # Test the profilers are successfully started and connected to profiler
    # service on the worker. Since there is no op running, it is expected to
    # return UnavailableError with no trace events collected string.
    with self.assertRaises(errors.UnavailableError) as error:
      profiler_client.trace(
          'localhost:' + str(test_port), self.get_temp_dir(), duration_ms=10)
    self.assertStartsWith(str(error.exception), 'No trace event was collected')

  def testTrace_ProfileIdleServerWithOptions(self):
    test_port = portpicker.pick_unused_port()
    profiler.start_server(test_port)
    # Test the profilers are successfully started and connected to profiler
    # service on the worker. Since there is no op running, it is expected to
    # return UnavailableError with no trace events collected string.
    with self.assertRaises(errors.UnavailableError) as error:
      options = profiler.ProfilerOptions(
          host_tracer_level=3, device_tracer_level=0)
      profiler_client.trace(
          'localhost:' + str(test_port),
          self.get_temp_dir(),
          duration_ms=10,
          options=options)
    self.assertStartsWith(str(error.exception), 'No trace event was collected')

  def testMonitor_ProcessInvalidAddress(self):
    # Monitor is only supported in cloud TPU. Test invalid address instead.
    with self.assertRaises(errors.UnavailableError):
      profiler_client.monitor('localhost:6006', 2000)


if __name__ == '__main__':
  test.main()
