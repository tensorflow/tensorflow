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
"""Profiler for eager mode."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.platform import gfile

LOGDIR_PLUGIN = 'plugins/profile'

_profiler = None
_profiler_lock = threading.Lock()
_run_num = 0


def start():
  """Start profiling.

  Only one active profiling session is allowed.

  Raises:
    AssertionError: If another profiling session is running.
  """
  global _profiler
  if _profiler is not None:
    raise AssertionError('Another profiler is running.')
  with _profiler_lock:
    _profiler = pywrap_tensorflow.TFE_NewProfiler(context.context()._handle)  # pylint: disable=protected-access


def stop():
  """Stop current profiling session and return its result.

  Returns:
    A binary string of tensorflow.tpu.Trace. User can write the string
    to file for offline analysis by tensorboard.

  Raises:
    AssertionError: If there is no active profiling session.
  """
  global _profiler
  global _run_num
  if _profiler is None:
    raise AssertionError('Cannot stop profiling. No profiler is running.')
  with c_api_util.tf_buffer() as buffer_:
    pywrap_tensorflow.TFE_ProfilerSerializeToString(
        context.context()._handle,  # pylint: disable=protected-access
        _profiler,
        buffer_)
    result = pywrap_tensorflow.TF_GetBuffer(buffer_)
  with _profiler_lock:
    pywrap_tensorflow.TFE_DeleteProfiler(_profiler)
    _profiler = None
    _run_num += 1
  return result


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
    plugin_dir = os.path.join(self._logdir, LOGDIR_PLUGIN,
                              'run{}'.format(_run_num))
    gfile.MakeDirs(plugin_dir)
    with gfile.Open(os.path.join(plugin_dir, 'local.trace'), 'wb') as f:
      f.write(result)
