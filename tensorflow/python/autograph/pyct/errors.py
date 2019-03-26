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


def report_internal_error(entity, exception):
  ag_logging.log(1, 'Error transforming %s', entity, exc_info=True)
  # TODO(znado): Add external bug reporting instructions.
  raise AutoGraphError(
      'Unexpected error transforming %s. If you believe this is due to a bug,'
      ' please set the verbosity to 10 (on Linux, `export '
      'AUTOGRAPH_VERBOSITY=10`) and attach the full output when filing the bug '
      'report. Caused by: %s' % (entity, exception))
