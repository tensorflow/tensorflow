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
"""tensorboard_logging provides logging that is also written to the events file.

Any messages logged via this module will be logged both via the platform logging
mechanism and to the SummaryWriter set via `set_summary_writer`. This is useful
for logging messages that you might want to be visible from inside TensorBoard
or that should be permanently associated with the training session.

You can use this just like the logging module:

>>> tensorboard_logging.set_summary_writer(summary_writer)
>>> tensorboard_logging.info("my %s", "message")
>>> tensorboard_logging.log(tensorboard_logging.WARN, "something")
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.core.util import event_pb2
from tensorflow.python.platform import logging

DEBUG = 'DEBUG'
INFO = 'INFO'
WARN = 'WARN'
ERROR = 'ERROR'
FATAL = 'FATAL'

# Messages with levels below this verbosity will not be logged.
_verbosity = WARN

# A value meaning 'not set yet' so we can use None to mean 'user actively told
# us they don't want a SummaryWriter'.
_sentinel_summary_writer = object()

# The SummaryWriter instance to use when logging, or None to not log, or
# _sentinel_summary_writer to indicate that the user hasn't called
# set_summary_writer yet.
_summary_writer = _sentinel_summary_writer

# Map from the tensorboard_logging logging enum values to the proto's enum
# values.
_LEVEL_PROTO_MAP = {
    DEBUG: event_pb2.LogMessage.DEBUG,
    INFO: event_pb2.LogMessage.INFO,
    WARN: event_pb2.LogMessage.WARN,
    ERROR: event_pb2.LogMessage.ERROR,
    FATAL: event_pb2.LogMessage.FATAL,
}

# Map from the tensorboard_logging module levels to the logging module levels.
_PLATFORM_LOGGING_LEVEL_MAP = {
    DEBUG: logging.DEBUG,
    INFO: logging.INFO,
    WARN: logging.WARN,
    ERROR: logging.ERROR,
    FATAL: logging.FATAL
}


def get_verbosity():
  return _verbosity


def set_verbosity(verbosity):
  _check_verbosity(verbosity)
  global _verbosity
  _verbosity = verbosity


def _check_verbosity(verbosity):
  if verbosity not in _LEVEL_PROTO_MAP:
    raise ValueError('Level %s is not a valid tensorboard_logging level' %
                     verbosity)


def set_summary_writer(summary_writer):
  """Sets the summary writer that events will be logged to.

  Calling any logging methods inside this module without calling this method
  will fail. If you don't want to log, call `set_summary_writer(None)`.

  Args:
    summary_writer: Either a SummaryWriter or None. None will cause messages not
    to be logged to any SummaryWriter, but they will still be passed to the
    platform logging module.
  """
  global _summary_writer
  _summary_writer = summary_writer


def _clear_summary_writer():
  """Makes all subsequent log invocations error.

  This is only used for testing. If you want to disable TensorBoard logging,
  call `set_summary_writer(None)` instead.
  """
  global _summary_writer
  _summary_writer = _sentinel_summary_writer


def log(level, message, *args):
  """Conditionally logs `message % args` at the level `level`.

  Note that tensorboard_logging verbosity and logging verbosity are separate;
  the message will always be passed through to the logging module regardless of
  whether it passes the tensorboard_logging verbosity check.

  Args:
    level: The verbosity level to use. Must be one of
      tensorboard_logging.{DEBUG, INFO, WARN, ERROR, FATAL}.
    message: The message template to use.
    *args: Arguments to interpolate to the message template, if any.

  Raises:
    ValueError: If `level` is not a valid logging level.
    RuntimeError: If the `SummaryWriter` to use has not been set.
  """
  if _summary_writer is _sentinel_summary_writer:
    raise RuntimeError('Must call set_summary_writer before doing any '
                       'logging from tensorboard_logging')
  _check_verbosity(level)
  proto_level = _LEVEL_PROTO_MAP[level]
  if proto_level >= _LEVEL_PROTO_MAP[_verbosity]:
    log_message = event_pb2.LogMessage(level=proto_level,
                                       message=message % args)
    event = event_pb2.Event(wall_time=time.time(), log_message=log_message)

    if _summary_writer:
      _summary_writer.add_event(event)

  logging.log(_PLATFORM_LOGGING_LEVEL_MAP[level], message, *args)


def debug(message, *args):
  log(DEBUG, message, *args)


def info(message, *args):
  log(INFO, message, *args)


def warn(message, *args):
  log(WARN, message, *args)


def error(message, *args):
  log(ERROR, message, *args)


def fatal(message, *args):
  log(FATAL, message, *args)
