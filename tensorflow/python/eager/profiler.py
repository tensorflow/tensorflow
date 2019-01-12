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

import threading

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.framework import c_api_util
from tensorflow.python.platform import gfile

_profiler = None
_profiler_lock = threading.Lock()


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
    A binary string of tensorflow.tfprof.ProfileProto. User can write the string
    to file for offline analysis by tfprof command-line tools or graphical user
    interface.

  Raises:
    AssertionError: If there is no active profiling session.
  """
  global _profiler
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
  return result


class Profiler(object):
  """Context-manager eager profiler api.

  Example usage:
  ```python
  with Profiler("/path/to/save/result"):
    # do some work
  ```
  """

  def __init__(self, filename):
    self._filename = filename

  def __enter__(self):
    start()

  def __exit__(self, typ, value, tb):
    result = stop()
    with gfile.Open(self._filename, 'wb') as f:
      f.write(result)

