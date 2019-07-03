# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import errors


def start_tracing(service_addr,
                  logdir,
                  duration_ms,
                  worker_list='',
                  include_dataset_ops=True,
                  num_tracing_attempts=3):
  """Sends grpc requests to profiler server to perform on-demand profiling.

  This method will block caller thread until receives tracing result.

  Args:
    service_addr: Address of profiler service e.g. localhost:6009.
    logdir: Path of TensorBoard log directory e.g. /tmp/tb_log.
    duration_ms: Duration of tracing or monitoring in ms.
    worker_list: The list of worker TPUs that we are about to profile in the
      current session. (TPU only)
    include_dataset_ops: Set to false to profile longer traces.
    num_tracing_attempts: Automatically retry N times when no trace event is
      collected.

  Raises:
    UnavailableError: If no trace event is collected.
  """
  if not pywrap_tensorflow.TFE_ProfilerClientStartTracing(
      service_addr, logdir, worker_list, include_dataset_ops, duration_ms,
      num_tracing_attempts):
    raise errors.UnavailableError(None, None, 'No trace event is collected.')


def monitor(service_addr,
            duration_ms,
            monitoring_level=1,
            display_timestamp=False):
  """Sends grpc requests to profiler server to perform on-demand monitoring.

  This method will block caller thread until receives monitoring result.

  Args:
    service_addr: Address of profiler service e.g. localhost:6009.
    duration_ms: Duration of tracing or monitoring in ms.
    monitoring_level: Choose a monitoring level between 1 and 2 to monitor your
      job. Level 2 is more verbose than level 1 and shows more metrics.
    display_timestamp: Set to true to display timestamp in monitoring result.

  Returns:
    A string of monitoring output.
  """
  with c_api_util.tf_buffer() as buffer_:
    pywrap_tensorflow.TFE_ProfilerClientMonitor(service_addr, duration_ms,
                                                monitoring_level,
                                                display_timestamp, buffer_)
    return pywrap_tensorflow.TF_GetBuffer(buffer_)
