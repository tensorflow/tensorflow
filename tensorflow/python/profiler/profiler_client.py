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

from tensorflow.python.util.tf_export import tf_export

_GRPC_PREFIX = 'grpc://'


@tf_export('profiler.experimental.client.trace', v1=[])
def trace(service_addr,
          logdir,
          duration_ms,
          worker_list='',
          num_tracing_attempts=3,
          options=None):
  """Sends grpc requests to profiler server to perform on-demand profiling.

  This method will block caller thread until it receives tracing result. This
  method supports CPU, GPU, and Cloud TPU. This method supports profiling a
  single host for CPU, GPU, TPU, as well as multiple TPU workers.
  The profiled results will be saved to your specified TensorBoard log
  directory (e.g. the directory you save your model checkpoints). Use the
  TensorBoard profile plugin to view the visualization and analysis results.

  Args:
    service_addr: gRPC address of profiler service e.g. grpc://localhost:6009.
    logdir: Path of TensorBoard log directory e.g. /tmp/tb_log.
    duration_ms: Duration of tracing or monitoring in ms.
    worker_list: Optional. The list of workers that we are about to profile in
      the current session (TPU only).
    num_tracing_attempts: Optional. Automatically retry N times when no trace
      event is collected (default 3).
    options: profiler.experimental.ProfilerOptions namedtuple for miscellaneous
      profiler options.

  Raises:
    UnavailableError: If no trace event is collected.

  Example usage (CPU/GPU):
  # Start a profiler server before your model runs.
  ```python
  tf.profiler.experimental.server.start(6009)
  # your model code.
  # Send gRPC request to the profiler server to collect a trace of your model.
  ```python
  tf.profiler.experimental.client.trace('grpc://localhost:6009',
                                        '/tmp/tb_log', 2000)

  Example usage (TPU):
  # Send gRPC request to a TPU worker to collect a trace of your model. A
  # profiler service has been started in the TPU worker at port 8466.
  ```python
  # E.g. your TPU IP address is 10.0.0.2 and you want to profile for 2 seconds.
  tf.profiler.experimental.client.trace('grpc://10.0.0.2:8466',
                                        'gs://your_tb_dir', 2000)

  Example usage (Multiple TPUs):
  # Send gRPC request to a TPU pod to collect a trace of your model on multiple
  # TPUs. A profiler service has been started in all the TPU workers at the
  # port 8466.
  ```python
  # E.g. your TPU IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you want to
  # profile for 2 seconds.
  tf.profiler.experimental.client.trace('grpc://10.0.0.2:8466',
                                        'gs://your_tb_dir',
                                        2000, '10.0.0.3,10.0.0.4')

  Launch TensorBoard and point it to the same logdir you provided to this API.
  $ tensorboard --logdir=/tmp/tb_log (or gs://your_tb_dir in the above examples)
  Open your browser and go to localhost:6006/#profile to view profiling results.

  """
  opts = dict(options._asdict()) if options is not None else {}
  _pywrap_profiler.trace(
      _strip_prefix(service_addr, _GRPC_PREFIX), logdir, worker_list, True,
      duration_ms, num_tracing_attempts, opts)


@tf_export('profiler.experimental.client.monitor', v1=[])
def monitor(service_addr, duration_ms, level=1):
  """Sends grpc requests to profiler server to perform on-demand monitoring.

  The monitoring result is a light weight performance summary of your model
  execution. This method will block the caller thread until it receives the
  monitoring result. This method currently supports Cloud TPU only.

  Args:
    service_addr: gRPC address of profiler service e.g. grpc://10.0.0.2:8466.
    duration_ms: Duration of monitoring in ms.
    level: Choose a monitoring level between 1 and 2 to monitor your job. Level
      2 is more verbose than level 1 and shows more metrics.

  Returns:
    A string of monitoring output.

  Example usage:
  # Continuously send gRPC requests to the Cloud TPU to monitor the model
  # execution.
  ```python
  for query in range(0, 100):
    print(tf.profiler.experimental.client.monitor('grpc://10.0.0.2:8466', 1000))


  """
  return _pywrap_profiler.monitor(
      _strip_prefix(service_addr, _GRPC_PREFIX), duration_ms, level, True)


def _strip_prefix(s, prefix):
  return s[len(prefix):] if s.startswith(prefix) else s
