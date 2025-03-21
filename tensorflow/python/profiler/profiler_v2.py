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
"""TensorFlow 2.x Profiler.

The profiler has two modes:
- Programmatic Mode: start(logdir), stop(), and Profiler class. Profiling starts
                     when calling start(logdir) or create a Profiler class.
                     Profiling stops when calling stop() to save to
                     TensorBoard logdir or destroying the Profiler class.
- Sampling Mode: start_server(). It will perform profiling after receiving a
                 profiling request.

NOTE: Only one active profiler session is allowed. Use of simultaneous
Programmatic Mode and Sampling Mode is undefined and will likely fail.

NOTE: The Keras TensorBoard callback will automatically perform sampled
profiling. Before enabling customized profiling, set the callback flag
"profile_batches=[]" to disable automatic sampled profiling.
"""

import collections
import threading

from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util.tf_export import tf_export

_profiler = None
_profiler_lock = threading.Lock()


@tf_export('profiler.experimental.ProfilerOptions', v1=[])
class ProfilerOptions(
    collections.namedtuple('ProfilerOptions', [
        'host_tracer_level', 'python_tracer_level', 'device_tracer_level',
        'delay_ms', 'ignore_start_error'
    ])):
  """Options for finer control over the profiler.

  Use `tf.profiler.experimental.ProfilerOptions` to control `tf.profiler`
  behavior.

  Fields:
    host_tracer_level: Adjust CPU tracing level. Values are: `1` - critical info
      only, `2` - info, `3` - verbose. [default value is `2`]
    python_tracer_level: Toggle tracing of Python function calls. Values are:
      `1` - enabled, `0` - disabled [default value is `0`]
    device_tracer_level: Adjust device (TPU/GPU) tracing level. Values are:
      `1` - enabled, `0` - disabled [default value is `1`]
    delay_ms: Requests for all hosts to start profiling at a timestamp that is
      `delay_ms` away from the current time. `delay_ms` is in milliseconds. If
      zero, each host will start profiling immediately upon receiving the
      request. Default value is `None`, allowing the profiler guess the best
      value.
    ignore_start_error: If true, the profiler will ignore errors when starting
      profiling.
  """

  def __new__(cls,
              host_tracer_level=2,
              python_tracer_level=0,
              device_tracer_level=1,
              delay_ms=None,
              ignore_start_error=False):
    return super(ProfilerOptions,
                 cls).__new__(cls, host_tracer_level, python_tracer_level,
                              device_tracer_level, delay_ms,
                              ignore_start_error)


@tf_export('profiler.experimental.start', v1=[])
def start(logdir, options=None):
  """Start profiling TensorFlow performance.

  Args:
    logdir: Profiling results log directory.
    options: `ProfilerOptions` namedtuple to specify miscellaneous profiler
      options. See example usage below.

  Raises:
    AlreadyExistsError: If a profiling session is already running.

  Example usage:
  ```python
  options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,
                                                     python_tracer_level = 1,
                                                     device_tracer_level = 1)
  tf.profiler.experimental.start('logdir_path', options = options)
  # Training code here
  tf.profiler.experimental.stop()
  ```

  To view the profiling results, launch TensorBoard and point it to `logdir`.
  Open your browser and go to `localhost:6006/#profile` to view profiling
  results.

  """
  global _profiler
  with _profiler_lock:
    if _profiler is not None:
      raise errors.AlreadyExistsError(None, None,
                                      'Another profiler is running.')
    _profiler = _pywrap_profiler.ProfilerSession()
    try:
      # support for namedtuple in pybind11 is missing, we change it to
      # dict type first.
      opts = dict(options._asdict()) if options is not None else {}
      _profiler.start(logdir, opts)
    except errors.AlreadyExistsError:
      logging.warning('Another profiler session is running which is probably '
                      'created by profiler server. Please avoid using profiler '
                      'server and profiler APIs at the same time.')
      raise errors.AlreadyExistsError(None, None,
                                      'Another profiler is running.')
    except Exception:
      _profiler = None
      raise


@tf_export('profiler.experimental.stop', v1=[])
def stop(save=True):
  """Stops the current profiling session.

  The profiler session will be stopped and profile results can be saved.

  Args:
    save: An optional variable to save the results to TensorBoard. Default True.

  Raises:
    UnavailableError: If there is no active profiling session.
  """
  global _profiler
  with _profiler_lock:
    if _profiler is None:
      raise errors.UnavailableError(
          None, None,
          'Cannot export profiling results. No profiler is running.')
    if save:
      try:
        _profiler.export_to_tb()
      except Exception:
        _profiler = None
        raise
    _profiler = None


def warmup():
  """Warm-up the profiler session.

  The profiler session will set up profiling context, including loading CUPTI
  library for GPU profiling. This is used for improving the accuracy of
  the profiling results.

  """
  start('')
  stop(save=False)


@tf_export('profiler.experimental.server.start', v1=[])
def start_server(port):
  """Start a profiler grpc server that listens to given port.

  The profiler server will exit when the process finishes. The service is
  defined in tensorflow/core/profiler/profiler_service.proto.

  Args:
    port: port profiler server listens to.
  Example usage: ```python tf.profiler.experimental.server.start(6009) # do
    your training here.
  """
  _pywrap_profiler.start_server(port)


@tf_export('profiler.experimental.Profile', v1=[])
class Profile(object):
  """Context-manager profile API.

  Profiling will start when entering the scope, and stop and save the results to
  the logdir when exits the scope. Open TensorBoard profile tab to view results.

  Example usage:
  ```python
  with tf.profiler.experimental.Profile("/path/to/logdir"):
    # do some work
  ```
  """

  def __init__(self, logdir, options=None):
    """Creates a context manager object for profiler API.

    Args:
      logdir: profile data will save to this directory.
      options: An optional `tf.profiler.experimental.ProfilerOptions` can be
        provided to fine tune the profiler's behavior.
    """
    self._logdir = logdir
    self._options = options

  def __enter__(self):
    start(self._logdir, self._options)

  def __exit__(self, typ, value, tb):
    stop()
