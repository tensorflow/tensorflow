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
"""Logging and debugging utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

# TODO(mdan): Use a custom logger class.
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

VERBOSITY_VAR_NAME = 'AUTOGRAPH_VERBOSITY'
DEFAULT_VERBOSITY = 0

verbosity_level = None  # vlog-like. Takes precedence over the env variable.
echo_log_to_stdout = False

# In interactive Python, logging echo is enabled by default.
if hasattr(sys, 'ps1') or hasattr(sys, 'ps2'):
  echo_log_to_stdout = True


@tf_export('autograph.set_verbosity')
def set_verbosity(level, alsologtostdout=False):
  """Sets the AutoGraph verbosity level.

  # Debug logging in AutoGraph

  More verbose logging is useful to enable when filing bug reports or debugging
  things.

  There are two controls that control the logging verbosity:
   * The `set_verbosity` function
   * The `AUTOGRAPH_VERBOSITY` environment variable
  `set_verbosity` takes precedence over the environment variable.

  By default, logs are output to absl's default logger, with INFO level. They
  can be mirrored to stdout by using the `alsologtostdout` argument. Mirroring
  is enabled by default when Python runs in interactive mode.

  Args:
    level: int, the verbosity level; uses the same scale as vlog
    alsologtostdout: bool, whether to ech log messages to stdout
  """
  global verbosity_level
  global echo_log_to_stdout
  verbosity_level = level
  echo_log_to_stdout = alsologtostdout


@tf_export('autograph.trace')
def trace(*args):
  """Traces argument information at compilation time."""
  print(*args)


def get_verbosity():
  global verbosity_level
  if verbosity_level is not None:
    return verbosity_level
  return os.getenv(VERBOSITY_VAR_NAME, DEFAULT_VERBOSITY)


def has_verbosity(level):
  return get_verbosity() >= level


def log(level, msg, *args, **kwargs):
  if has_verbosity(level):
    logging.info(msg, *args, **kwargs)
    if echo_log_to_stdout:
      print(msg % args)


def warn_first_n(msg, *args, **kwargs):
  logging.log_first_n(logging.WARNING, msg, *args, **kwargs)
