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
"""Code transformation exceptions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import traceback

from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.utils import ag_logging


class AutoGraphError(Exception):
  pass


class ExecutionError(AutoGraphError):
  """Raised by AutoGraph during various execution stages."""

  def __init__(self, stage, message):
    super(ExecutionError, self).__init__()
    self.stage = stage
    self.message = message

  def __str__(self):
    return 'Runtime error during {} stage: {}'.format(self.stage, self.message)


class InternalError(AutoGraphError):
  """Raised when AutoGraph finds an unexpected error."""

  def __init__(self, message, original_exc):
    super(InternalError, self).__init__()
    self.message = message
    self.original_exc = original_exc

  def __str__(self):
    return '{} during {}: {}'.format(
        type(self.original_exc).__name__, self.message, self.original_exc)


# TODO(znado): merge with ExecutionError.
class StagingError(AutoGraphError):
  """Raised when AutoGraph has an error while executing a converted function."""

  def __init__(self, user_trace, original_error):
    """Constructs a StagingError.

    Args:
      user_trace: Tuple[OriginInfo], the converted call traceback frames.
      original_error: Exception, the original error thrown.
    """
    super(StagingError, self).__init__()
    self.user_trace = user_trace
    self.original_error = original_error

  def __str__(self):
    indent_str = '    '
    new_stacktrace_lines = []
    for origin in self.user_trace:
      if not origin:
        continue
      frame_str = indent_str + '{}:{} ({})\n{}    {}'.format(
          origin.loc.filename, origin.loc.lineno, origin.function_name,
          indent_str, origin.source_code_line.strip())
      new_stacktrace_lines.append(frame_str)
    new_stacktrace_str = '\n'.join(new_stacktrace_lines)
    original_type = self.original_error.__class__.__name__
    original_message = str(self.original_error)
    new_message = original_type + ': ' + original_message
    return ('\nAn error occurred while executing AutoGraph transformed code. '
            'For details, set the verbosity to 10 (on Linux, '
            '`export AUTOGRAPH_VERBOSITY=10`). Corresponding code:\n' +
            new_stacktrace_str + '\n\n' + indent_str + new_message + '\n\n')


def report_internal_error(entity, exception):
  ag_logging.log(1, 'Error transforming %s', entity, exc_info=True)
  # TODO(znado): Add external bug reporting instructions.
  raise AutoGraphError(
      'Unexpected error transforming %s. If you believe this is due to a bug, '
      'please set the verbosity to 10 (on Linux, `export '
      'AUTOGRAPH_VERBOSITY=10`) and attach the full output when filing the bug '
      'report. Caused by: %s' % (entity, exception))


def extract_origin_info(converted_f):
  """Attempts to use converted_f's source map to get error origin info."""
  source_map = converted_f.ag_source_map
  original_traceback = traceback.extract_tb(sys.exc_info()[2])
  # Can go through all frames and check which ones have origin info in order to
  # filter for only the locations relevant to converted_f.
  #
  # Return the first occurrence of the reversed traceback in the source map in
  # order to return the innermost frame for this function. We want to do this
  # because when have a tf.cond we will have multiple matches and we want to
  # return the last one in this function, because any after that will be in
  # the next function/frame in the stacktrace.
  for frame in reversed(original_traceback):
    converted_loc = origin_info.LineLocation(
        filename=frame[0], lineno=frame[1])
    if converted_loc in source_map:
      return source_map[converted_loc]
  return None
