# Copyright 2015 Google Inc. All Rights Reserved.
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

import logging
import os
import sys
import time
from logging import DEBUG
from logging import ERROR
from logging import FATAL
from logging import INFO
from logging import WARN

import six


# Controls which methods from pyglib.logging are available within the project
# Do not add methods here without also adding to platform/google/_logging.py
__all__ = ['log', 'debug', 'error', 'fatal', 'info', 'warn', 'warning',
           'DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN',
           'flush', 'log_every_n', 'log_first_n', 'vlog',
           'TaskLevelStatusMessage', 'get_verbosity', 'set_verbosity']

# Scope the tensorflow logger to not conflict with users' loggers
_logger = logging.getLogger('tensorflow')
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
_logger.addHandler(_handler)

log = _logger.log
debug = _logger.debug
error = _logger.error
fatal = _logger.fatal
info = _logger.info
warn = _logger.warn
warning = _logger.warn

_level_names = {
    FATAL: 'FATAL',
    ERROR: 'ERROR',
    WARN: 'WARN',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
}

# Mask to convert integer thread ids to unsigned quantities for logging
# purposes
_THREAD_ID_MASK = 2 * sys.maxsize + 1

_log_prefix = None  # later set to google2_log_prefix

# Counter to keep track of number of log entries per token.
_log_counter_per_token = {}


def TaskLevelStatusMessage(msg):
  error(msg)


def flush():
  raise NotImplementedError()


# Code below is taken from pyglib/logging
def vlog(level, msg, *args, **kwargs):
  _logger.log(level, msg, *args, **kwargs)


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


def log_if(level, msg, condition, *args):
  """Log 'msg % args' at level 'level' only if condition is fulfilled."""
  if condition:
    vlog(level, msg, *args)


def _GetFileAndLine():
  """Returns (filename, linenumber) for the stack frame."""
  # Use sys._getframe().  This avoids creating a traceback object.
  # pylint: disable=protected-access
  f = sys._getframe()
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
  global _logfile_map, _logfile_map_mutex
  # pylint: enable=global-variable-not-assigned

  # Record current time
  now = timestamp or time.time()
  now_tuple = time.localtime(now)
  now_microsecond = int(1e6 * (now % 1.0))

  (filename, line) = file_and_line or _GetFileAndLine()
  basename = os.path.basename(filename)

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


def get_verbosity():
  """Return how much logging output will be produced."""
  return _logger.getEffectiveLevel()


def set_verbosity(verbosity):
  """Sets the threshold for what messages will be logged."""
  _logger.setLevel(verbosity)


def _get_thread_id():
  """Get id of current thread, suitable for logging as an unsigned quantity."""
  # pylint: disable=protected-access
  thread_id = six.moves._thread.get_ident()
  # pylint:enable=protected-access
  return thread_id & _THREAD_ID_MASK


_log_prefix = google2_log_prefix
