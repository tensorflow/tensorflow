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
"""TensorFlow 2.0 Profiler for both Eager Mode and Graph Mode.

The profiler has two mode:
- Programmatic Mode: start(), stop() and Profiler class. It will perform
                    when calling start() or create Profiler class and will stop
                    when calling stop() or destroying Profiler class.
- On-demand Mode: start_profiler_server(). It will perform profiling when
                  receive profiling request.

NOTE: Only one active profiler session is allowed. Use of simultaneous
Programmatic Mode and On-demand Mode is undefined and will likely fail.

NOTE: The Keras TensorBoard callback will automatically perform sampled
profiling. Before enabling customized profiling, set the callback flag
"profile_batches=[]" to disable automatic sampled profiling.
customized profiling.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import os
import threading

from tensorflow.python import _pywrap_events_writer
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.eager import eager_util as c_api_util
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat

_profiler = None
_profiler_lock = threading.Lock()
_run_num = 0
# This suffix should be kept in sync with kProfileEmptySuffix in
# tensorflow/core/profiler/rpc/client/capture_profile.cc.
_EVENT_FILE_SUFFIX = '.profile-empty'


class ProfilerAlreadyRunningError(Exception):
  pass


class ProfilerNotRunningError(Exception):
  pass


def start():
  """Start profiling.

  Raises:
    ProfilerAlreadyRunningError: If another profiling session is running.
  """
  global _profiler
  with _profiler_lock:
    if _profiler is not None:
      raise ProfilerAlreadyRunningError('Another profiler is running.')
    if context.default_execution_mode == context.EAGER_MODE:
      context.ensure_initialized()
    _profiler = pywrap_tfe.TFE_NewProfiler()
    if not pywrap_tfe.TFE_ProfilerIsOk(_profiler):
      logging.warning('Another profiler session is running which is probably '
                      'created by profiler server. Please avoid using profiler '
                      'server and profiler APIs at the same time.')


def stop():
  """Stop current profiling session and return its result.

  Returns:
    A binary string of tensorflow.tpu.Trace. User can write the string
    to file for offline analysis by tensorboard.

  Raises:
    ProfilerNotRunningError: If there is no active profiling session.
  """
  global _profiler
  global _run_num
  with _profiler_lock:
    if _profiler is None:
      raise ProfilerNotRunningError(
          'Cannot stop profiling. No profiler is running.')
    if context.default_execution_mode == context.EAGER_MODE:
      context.context().executor.wait()
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_ProfilerSerializeToString(_profiler, buffer_)
      result = pywrap_tfe.TF_GetBuffer(buffer_)
    pywrap_tfe.TFE_DeleteProfiler(_profiler)
    _profiler = None
    _run_num += 1
  return result


def maybe_create_event_file(logdir):
  """Create an empty event file if not already exists.

  This event file indicates that we have a plugins/profile/ directory in the
  current logdir.

  Args:
    logdir: log directory.
  """
  for file_name in gfile.ListDirectory(logdir):
    if file_name.endswith(_EVENT_FILE_SUFFIX):
      return
  # TODO(b/127330388): Use summary_ops_v2.create_file_writer instead.
  event_writer = _pywrap_events_writer.EventsWriter(
      compat.as_bytes(os.path.join(logdir, 'events')))
  event_writer.InitWithSuffix(compat.as_bytes(_EVENT_FILE_SUFFIX))


def save(logdir, result):
  """Save profile result to TensorBoard logdir.

  Args:
    logdir: log directory read by TensorBoard.
    result: profiling result returned by stop().
  """
  plugin_dir = os.path.join(
      logdir, 'plugins', 'profile',
      datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
  gfile.MakeDirs(plugin_dir)
  maybe_create_event_file(logdir)
  with gfile.Open(os.path.join(plugin_dir, 'local.trace'), 'wb') as f:
    f.write(result)


def start_profiler_server(port):
  """Start a profiler grpc server that listens to given port.

  The profiler server will keep the program running even the training finishes.
  Please shutdown the server with CTRL-C. It can be used in both eager mode and
  graph mode. The service defined in
  tensorflow/core/profiler/profiler_service.proto. Please use
  tensorflow/contrib/tpu/profiler/capture_tpu_profile to capture tracable
  file following https://cloud.google.com/tpu/docs/cloud-tpu-tools#capture_trace

  Args:
    port: port profiler server listens to.
  """
  if context.default_execution_mode == context.EAGER_MODE:
    context.ensure_initialized()
  pywrap_tfe.TFE_StartProfilerServer(port)


class Profiler(object):
  """Context-manager eager profiler api.

  Example usage:
  ```python
  with Profiler("/path/to/logdir"):
    # do some work
  ```
  """

  def __init__(self, logdir):
    self._logdir = logdir

  def __enter__(self):
    start()

  def __exit__(self, typ, value, tb):
    result = stop()
    save(self._logdir, result)
