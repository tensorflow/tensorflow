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

from tensorflow.python.framework import errors
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
  """Sends gRPC requests to one or more profiler servers to perform on-demand profiling.

  This method will block the calling thread until it receives responses from all
  servers or until deadline expiration. Both single host and multiple host
  profiling are supported on CPU, GPU, and TPU.
  The profiled results will be saved by each server to the specified TensorBoard
  log directory (i.e. the directory you save your model checkpoints). Use the
  TensorBoard profile plugin to view the visualization and analysis results.

  Args:
    service_addr: A comma delimited string of gRPC addresses of the workers to
      profile.
      e.g. service_addr='grpc://localhost:6009'
           service_addr='grpc://10.0.0.2:8466,grpc://10.0.0.3:8466'
           service_addr='grpc://localhost:12345,grpc://localhost:23456'
    logdir: Path to save profile data to, typically a TensorBoard log directory.
      This path must be accessible to both the client and server.
      e.g. logdir='gs://your_tb_dir'
    duration_ms: Duration of tracing or monitoring in milliseconds. Must be
      greater than zero.
    worker_list: An optional TPU only configuration. The list of workers to
      profile in the current session.
    num_tracing_attempts: Optional. Automatically retry N times when no trace
      event is collected (default 3).
    options: profiler.experimental.ProfilerOptions namedtuple for miscellaneous
      profiler options.

  Raises:
    InvalidArgumentError: For when arguments fail validation checks.
    UnavailableError: If no trace event was collected.

  Example usage (CPU/GPU):

  ```python
    # Start a profiler server before your model runs.
    tf.profiler.experimental.server.start(6009)
    # (Model code goes here).
    # Send gRPC request to the profiler server to collect a trace of your model.
    tf.profiler.experimental.client.trace('grpc://localhost:6009',
                                          '/nfs/tb_log', 2000)
  ```

  Example usage (Multiple GPUs):

  ```python
    # E.g. your worker IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you
    # would like to schedule start of profiling 1 second from now, for a
    # duration of 2 seconds.
    options['delay_ms'] = 1000
    tf.profiler.experimental.client.trace(
        'grpc://10.0.0.2:8466,grpc://10.0.0.3:8466,grpc://10.0.0.4:8466',
        'gs://your_tb_dir',
        2000,
        options=options)
  ```

  Example usage (TPU):

  ```python
    # Send gRPC request to a TPU worker to collect a trace of your model. A
    # profiler service has been started in the TPU worker at port 8466.
    # E.g. your TPU IP address is 10.0.0.2 and you want to profile for 2 seconds
    # .
    tf.profiler.experimental.client.trace('grpc://10.0.0.2:8466',
                                          'gs://your_tb_dir', 2000)
  ```

  Example usage (Multiple TPUs):

  ```python
    # Send gRPC request to a TPU pod to collect a trace of your model on
    # multipleTPUs. A profiler service has been started in all the TPU workers
    # at theport 8466.
    # E.g. your TPU IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you want
    # to profile for 2 seconds.
    tf.profiler.experimental.client.trace('grpc://10.0.0.2:8466',
                                          'gs://your_tb_dir',
                                          2000, '10.0.0.2,10.0.0.3,10.0.0.4')
  ```

  Launch TensorBoard and point it to the same logdir you provided to this API.

  ```shell
    # logdir can be gs://your_tb_dir as in the above examples.
    $ tensorboard --logdir=/tmp/tb_log
  ```

  Open your browser and go to localhost:6006/#profile to view profiling results.

  """
  if duration_ms <= 0:
    raise errors.InvalidArgumentError(None, None,
                                      'duration_ms must be greater than zero.')

  opts = dict(options._asdict()) if options is not None else {}
  _pywrap_profiler.trace(
      _strip_addresses(service_addr, _GRPC_PREFIX), logdir, worker_list, True,
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

  ```python
    # Continuously send gRPC requests to the Cloud TPU to monitor the model
    # execution.

    for query in range(0, 100):
      print(
        tf.profiler.experimental.client.monitor('grpc://10.0.0.2:8466', 1000))
  ```

  """
  return _pywrap_profiler.monitor(
      _strip_prefix(service_addr, _GRPC_PREFIX), duration_ms, level, True)


def _strip_prefix(s, prefix):
  return s[len(prefix):] if s.startswith(prefix) else s


def _strip_addresses(addresses, prefix):
  return ','.join([_strip_prefix(s, prefix) for s in addresses.split(',')])
