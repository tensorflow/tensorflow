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
import traceback

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

  _Debug logging in AutoGraph_

  More verbose logging is useful to enable when filing bug reports or doing
  more in-depth debugging.

  There are two means to control the logging verbosity:

   * The `set_verbosity` function

   * The `AUTOGRAPH_VERBOSITY` environment variable

  `set_verbosity` takes precedence over the environment variable.

  For example:

  ```python
  import os
  import tensorflow as tf

  os.environ['AUTOGRAPH_VERBOSITY'] = 5
  # Verbosity is now 5

  tf.autograph.set_verbosity(0)
  # Verbosity is now 0

  os.environ['AUTOGRAPH_VERBOSITY'] = 1
  # No effect, because set_verbosity was already called.
  ```

  Logs entries are output to [absl](https://abseil.io)'s 
  [default output](https://abseil.io/docs/python/guides/logging),
  with `INFO` level.
  Logs can be mirrored to stdout by using the `alsologtostdout` argument.
  Mirroring is enabled by default when Python runs in interactive mode.

  Args:
    level: int, the verbosity level; larger values specify increased verbosity;
      0 means no logging. When reporting bugs, it is recommended to set this
      value to a larger number, like 10.
    alsologtostdout: bool, whether to also output log messages to `sys.stdout`.
  """
  global verbosity_level
  global echo_log_to_stdout
  verbosity_level = level
  echo_log_to_stdout = alsologtostdout


@tf_export('autograph.trace')
def trace(*args):
  """Traces argument information at compilation time.

  `trace` is useful when debugging, and it always executes during the tracing
  phase, that is, when the TF graph is constructed.

  _Example usage_

  ```python
  import tensorflow as tf

  for i in tf.range(10):
    tf.autograph.trace(i)
  # Output: <Tensor ...>
  ```

  Args:
    *args: Arguments to print to `sys.stdout`.
  """
  print(*args)


def get_verbosity():
  global verbosity_level
  if verbosity_level is not None:
    return verbosity_level
  return int(os.getenv(VERBOSITY_VAR_NAME, DEFAULT_VERBOSITY))


def has_verbosity(level):
  return get_verbosity() >= level


def _output_to_stdout(msg, *args, **kwargs):
  print(msg % args)
  if kwargs.get('exc_info', False):
    traceback.print_exc()


def error(level, msg, *args, **kwargs):
  if has_verbosity(level):
    logging.error(msg, *args, **kwargs)
    if echo_log_to_stdout:
      _output_to_stdout('ERROR: ' + msg, *args, **kwargs)


def log(level, msg, *args, **kwargs):
  if has_verbosity(level):
    logging.info(msg, *args, **kwargs)
    if echo_log_to_stdout:
      _output_to_stdout(msg, *args, **kwargs)


def warn(msg, *args, **kwargs):
  logging.warn(msg, *args, **kwargs)
  if echo_log_to_stdout:
    _output_to_stdout('WARNING: ' + msg, *args, **kwargs)
