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
"""Profiler client APIs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.profiler.internal import _pywrap_profiler


def trace(service_addr,
          logdir,
          duration_ms,
          worker_list='',
          num_tracing_attempts=3):
  """Sends grpc requests to profiler server to perform on-demand profiling.

  This method will block caller thread until receives tracing result.

  Args:
    service_addr: Address of profiler service e.g. localhost:6009.
    logdir: Path of TensorBoard log directory e.g. /tmp/tb_log.
    duration_ms: Duration of tracing or monitoring in ms.
    worker_list: Optional. The list of workers that we are about to profile in
      the current session (TPU only).
    num_tracing_attempts: Optional. Automatically retry N times when no trace
      event is collected (default 3).

  Raises:
    UnavailableError: If no trace event is collected.
  """
  _pywrap_profiler.trace(service_addr, logdir, worker_list, True, duration_ms,
                         num_tracing_attempts)


def monitor(service_addr, duration_ms, level=1):
  """Sends grpc requests to profiler server to perform on-demand monitoring.

  This method will block caller thread until receives monitoring result.

  Args:
    service_addr: Address of profiler service e.g. localhost:6009.
    duration_ms: Duration of monitoring in ms.
    level: Choose a monitoring level between 1 and 2 to monitor your job. Level
      2 is more verbose than level 1 and shows more metrics.

  Returns:
    A string of monitoring output.
  """
  return _pywrap_profiler.monitor(service_addr, duration_ms, level, True)
