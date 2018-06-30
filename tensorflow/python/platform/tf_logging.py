# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Logging utilities."""
# pylint: disable=unused-import
# pylint: disable=g-bad-import-order
# pylint: disable=invalid-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as _logging
import os as _os
import sys as _sys
import time as _time
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN
import threading

import six

from tensorflow.python.util.tf_export import tf_export


# Don't use this directly. Use _get_logger() instead.
_logger = None
_logger_lock = threading.Lock()


def _get_logger():
  global _logger

  # Use double-checked locking to avoid taking lock unnecessarily.
  if _logger:
    return _logger

  _logger_lock.acquire()

  try:
    if _logger:
      return _logger

    # Scope the TensorFlow logger to not conflict with users' loggers.
    logger = _logging.getLogger('tensorflow')

    # Don't further configure the TensorFlow logger if the root logger is
    # already configured. This prevents double logging in those cases.
    if not _logging.getLogger().handlers:
      # Determine whether we are in an interactive environment
      _interactive = False
      try:
        # This is only defined in interactive shells.
        if _sys.ps1: _interactive = True
      except AttributeError:
        # Even now, we may be in an interactive shell with `python -i`.
        _interactive = _sys.flags.interactive

      # If we are in an interactive environment (like Jupyter), set loglevel
      # to INFO and pipe the output to stdout.
      if _interactive:
        logger.setLevel(INFO)
        _logging_target = _sys.stdout
      else:
        _logging_target = _sys.stderr

      # Add the output handler.
      _handler = _logging.StreamHandler(_logging_target)
      _handler.setFormatter(_logging.Formatter(_logging.BASIC_FORMAT, None))
      logger.addHandler(_handler)

    _logger = logger
    return _logger

  finally:
    _logger_lock.release()


@tf_export('logging.log')
def log(level, msg, *args, **kwargs):
  _get_logger().log(level, msg, *args, **kwargs)


@tf_export('logging.debug')
def debug(msg, *args, **kwargs):
  _get_logger().debug(msg, *args, **kwargs)


@tf_export('logging.error')
def error(msg, *args, **kwargs):
  _get_logger().error(msg, *args, **kwargs)


@tf_export('logging.fatal')
def fatal(msg, *args, **kwargs):
  _get_logger().fatal(msg, *args, **kwargs)


@tf_export('logging.info')
def info(msg, *args, **kwargs):
  _get_logger().info(msg, *args, **kwargs)


@tf_export('logging.warn')
def warn(msg, *args, **kwargs):
  _get_logger().warn(msg, *args, **kwargs)


@tf_export('logging.warning')
def warning(msg, *args, **kwargs):
  _get_logger().warning(msg, *args, **kwargs)


_level_names = {
    FATAL: 'FATAL',
    ERROR: 'ERROR',
    WARN: 'WARN',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
}

# Mask to convert integer thread ids to unsigned quantities for logging
# purposes
_THREAD_ID_MASK = 2 * _sys.maxsize + 1

_log_prefix = None  # later set to google2_log_prefix

# Counter to keep track of number of log entries per token.
_log_counter_per_token = {}


@tf_export('logging.TaskLevelStatusMessage')
def TaskLevelStatusMessage(msg):
  error(msg)


@tf_export('logging.flush')
def flush():
  raise NotImplementedError()


# Code below is taken from pyglib/logging
@tf_export('logging.vlog')
def vlog(level, msg, *args, **kwargs):
  _get_logger().log(level, msg, *args, **kwargs)


def _GetNextLogCountPerToken(token):
  """Wrapper for _log_counter_per_token.

  Args:
    token: The token for which to look up the count.

  Returns:
    The number of times this function has been called with
    *token* as an argument (starting at 0)
  """
  global _log_counter_per_token  # pylint: disable=global-variable-not-assigned
  _log_counter_per_token[token] = 1 + _log_counter_per_token.get(token, -1)
  return _log_counter_per_token[token]


@tf_export('logging.log_every_n')
def log_every_n(level, msg, n, *args):
  """Log 'msg % args' at level 'level' once per 'n' times.

  Logs the 1st call, (N+1)st call, (2N+1)st call,  etc.
  Not threadsafe.

  Args:
    level: The level at which to log.
    msg: The message to be logged.
    n: The number of times this should be called before it is logged.
    *args: The args to be substituted into the msg.
  """
  count = _GetNextLogCountPerToken(_GetFileAndLine())
  log_if(level, msg, not (count % n), *args)


@tf_export('logging.log_first_n')
def log_first_n(level, msg, n, *args):  # pylint: disable=g-bad-name
  """Log 'msg % args' at level 'level' only first 'n' times.

  Not threadsafe.

  Args:
    level: The level at which to log.
    msg: The message to be logged.
    n: The number of times this should be called before it is logged.
    *args: The args to be substituted into the msg.
  """
  count = _GetNextLogCountPerToken(_GetFileAndLine())
  log_if(level, msg, count < n, *args)


@tf_export('logging.log_if')
def log_if(level, msg, condition, *args):
  """Log 'msg % args' at level 'level' only if condition is fulfilled."""
  if condition:
    vlog(level, msg, *args)


def _GetFileAndLine():
  """Returns (filename, linenumber) for the stack frame."""
  # Use sys._getframe().  This avoids creating a traceback object.
  # pylint: disable=protected-access
  f = _sys._getframe()
  # pylint: enable=protected-access
  our_file = f.f_code.co_filename
  f = f.f_back
  while f:
    code = f.f_code
    if code.co_filename != our_file:
      return (code.co_filename, f.f_lineno)
    f = f.f_back
  return ('<unknown>', 0)


def google2_log_prefix(level, timestamp=None, file_and_line=None):
  """Assemble a logline prefix using the google2 format."""
  # pylint: disable=global-variable-not-assigned
  global _level_names
  # pylint: enable=global-variable-not-assigned

  # Record current time
  now = timestamp or _time.time()
  now_tuple = _time.localtime(now)
  now_microsecond = int(1e6 * (now % 1.0))

  (filename, line) = file_and_line or _GetFileAndLine()
  basename = _os.path.basename(filename)

  # Severity string
  severity = 'I'
  if level in _level_names:
    severity = _level_names[level][0]

  s = '%c%02d%02d %02d:%02d:%02d.%06d %5d %s:%d] ' % (
      severity,
      now_tuple[1],  # month
      now_tuple[2],  # day
      now_tuple[3],  # hour
      now_tuple[4],  # min
      now_tuple[5],  # sec
      now_microsecond,
      _get_thread_id(),
      basename,
      line)

  return s


@tf_export('logging.get_verbosity')
def get_verbosity():
  """Return how much logging output will be produced."""
  return _get_logger().getEffectiveLevel()


@tf_export('logging.set_verbosity')
def set_verbosity(v):
  """Sets the threshold for what messages will be logged."""
  _get_logger().setLevel(v)


def _get_thread_id():
  """Get id of current thread, suitable for logging as an unsigned quantity."""
  # pylint: disable=protected-access
  thread_id = six.moves._thread.get_ident()
  # pylint:enable=protected-access
  return thread_id & _THREAD_ID_MASK


_log_prefix = google2_log_prefix

tf_export('logging.DEBUG').export_constant(__name__, 'DEBUG')
tf_export('logging.ERROR').export_constant(__name__, 'ERROR')
tf_export('logging.FATAL').export_constant(__name__, 'FATAL')
tf_export('logging.INFO').export_constant(__name__, 'INFO')
tf_export('logging.WARN').export_constant(__name__, 'WARN')
