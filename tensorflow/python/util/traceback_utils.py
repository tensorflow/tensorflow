# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities related to TensorFlow exception stack trace prettifying."""

import os
import sys
import threading
import traceback
import types
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export


_ENABLE_TRACEBACK_FILTERING = threading.local()
_EXCLUDED_PATHS = (
    os.path.abspath(os.path.join(__file__, '..', '..')),
)


@tf_export('debugging.is_traceback_filtering_enabled')
def is_traceback_filtering_enabled():
  """Check whether traceback filtering is currently enabled.

  See also `tf.debugging.enable_traceback_filtering()` and
  `tf.debugging.disable_traceback_filtering()`. Note that filtering out
  internal frames from the tracebacks of exceptions raised by TensorFlow code
  is the default behavior.

  Returns:
    True if traceback filtering is enabled
    (e.g. if `tf.debugging.enable_traceback_filtering()` was called),
    and False otherwise (e.g. if `tf.debugging.disable_traceback_filtering()`
    was called).
  """
  value = getattr(_ENABLE_TRACEBACK_FILTERING, 'value', True)
  return value


@tf_export('debugging.enable_traceback_filtering')
def enable_traceback_filtering():
  """Enable filtering out TensorFlow-internal frames in exception stack traces.

  Raw TensorFlow stack traces involve many internal frames, which can be
  challenging to read through, while not being actionable for end users.
  By default, TensorFlow filters internal frames in most exceptions that it
  raises, to keep stack traces short, readable, and focused on what's
  actionable for end users (their own code).

  If you have previously disabled traceback filtering via
  `tf.debugging.disable_traceback_filtering()`, you can re-enable it via
  `tf.debugging.enable_traceback_filtering()`.

  Raises:
    RuntimeError: If Python version is not at least 3.7.
  """
  if sys.version_info.major != 3 or sys.version_info.minor < 7:
    raise RuntimeError(
        f'Traceback filtering is only available with Python 3.7 or higher. '
        f'This Python version: {sys.version}')
  global _ENABLE_TRACEBACK_FILTERING
  _ENABLE_TRACEBACK_FILTERING.value = True


@tf_export('debugging.disable_traceback_filtering')
def disable_traceback_filtering():
  """Disable filtering out TensorFlow-internal frames in exception stack traces.

  Raw TensorFlow stack traces involve many internal frames, which can be
  challenging to read through, while not being actionable for end users.
  By default, TensorFlow filters internal frames in most exceptions that it
  raises, to keep stack traces short, readable, and focused on what's
  actionable for end users (their own code).

  Calling `tf.debugging.disable_traceback_filtering` disables this filtering
  mechanism, meaning that TensorFlow exceptions stack traces will include
  all frames, in particular TensorFlow-internal ones.

  **If you are debugging a TensorFlow-internal issue, you need to call
  `tf.debugging.disable_traceback_filtering`**.
  To re-enable traceback filtering afterwards, you can call
  `tf.debugging.enable_traceback_filtering()`.
  """
  global _ENABLE_TRACEBACK_FILTERING
  _ENABLE_TRACEBACK_FILTERING.value = False


def include_frame(fname):
  for exclusion in _EXCLUDED_PATHS:
    if exclusion in fname:
      return False
  return True


def _process_traceback_frames(tb):
  new_tb = None
  tb_list = list(traceback.walk_tb(tb))
  for f, line_no in reversed(tb_list):
    if include_frame(f.f_code.co_filename):
      new_tb = types.TracebackType(new_tb, f, f.f_lasti, line_no)
  if new_tb is None and tb_list:
    f, line_no = tb_list[-1]
    new_tb = types.TracebackType(new_tb, f, f.f_lasti, line_no)
  return new_tb


def filter_traceback(fn):
  """Decorator to filter out TF-internal stack trace frames in exceptions.

  Raw TensorFlow stack traces involve many internal frames, which can be
  challenging to read through, while not being actionable for end users.
  By default, TensorFlow filters internal frames in most exceptions that it
  raises, to keep stack traces short, readable, and focused on what's
  actionable for end users (their own code).

  Arguments:
    fn: The function or method to decorate. Any exception raised within the
      function will be reraised with its internal stack trace frames filtered
      out.

  Returns:
    Decorated function or method.
  """
  if sys.version_info.major != 3 or sys.version_info.minor < 7:
    return fn

  def error_handler(*args, **kwargs):
    try:
      if not is_traceback_filtering_enabled():
        return fn(*args, **kwargs)
    except NameError:
      # In some very rare cases,
      # `is_traceback_filtering_enabled` (from the outer scope) may not be
      # accessible from inside this function
      return fn(*args, **kwargs)

    filtered_tb = None
    try:
      return fn(*args, **kwargs)
    except Exception as e:
      filtered_tb = _process_traceback_frames(e.__traceback__)
      raise e.with_traceback(filtered_tb) from None
    finally:
      del filtered_tb

  return tf_decorator.make_decorator(fn, error_handler)
