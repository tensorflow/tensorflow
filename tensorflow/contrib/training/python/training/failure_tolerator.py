# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""A retry helper for tolerating transient failures in distributed training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import time

from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging


class FailureTolerator(object):
  """Helper for tolerating certain exceptions.

  When encountering a handled exception inside tolerator.forgive(), it
  is suppressed (but logged). A subsequent call to tolerator.forgive()
  will sleep for a period of time before continuing, with exponential
  backoff on multiple exceptions. (The delay avoids retrying too
  quickly -- a subsequent attempt will often only succeed after a
  transient failure has resolved itself.)

  If more than `limit` exceptions have been encountered,
  the error will not be suppressed.

  Exceptions occurring more than `forgive_after_seconds` ago
  (excluding time spent waiting between retries) are forgiven and no
  longer count towards the limit.

  An example loop using FailureTolerator to retry until a successful
  `session.run(...)` would look like:
  ```
  failure_tolerator = FailureTolerator()
  while True:
    with failure_tolerator.forgive():
      session = make_session_somehow()
      while not should_stop():
        session.run(...)
      break  # session.run was successful
  ```

  By using FailureTolerator, failures are logged, there are delays
  between retries, and there's a ceiling on the maximum number of
  retries available. (In the case of persistent errors, the task won't
  just loop forever!)
  """

  def __init__(self, limit=5, init_delay=5.0, backoff_factor=2.0,
               forgive_after_seconds=6000, handled_exceptions=None):
    """Creates a FailureTolerator.

    The result will pause for `init_delay *
    (backoff_factor^(failure_count-1))` when re-entering `forgive()`
    after a failure.

    Args:
      limit: The maximum number of suppressed, unforgiven, failures.
      init_delay: How long to pause once the first failure is
        encountered. Defaults to five seconds.
      backoff_factor: Each subsequent failure grows the pause by this factor.
      forgive_after_seconds: Failures older than this are forgiven.
      handled_exceptions: The exceptions to forgive. Defaults to
          `(errors.AbortedError,)`.

    """
    self.limit = limit
    self.backoff = backoff_factor
    self.delay = init_delay
    self.forgive_after = forgive_after_seconds
    self.exceptions = []
    self.time_in_delay = 0.0
    if handled_exceptions is None:
      self.handled = (errors.AbortedError,)
    else:
      self.handled = tuple(handled_exceptions)

  def _adjusted_now(self):
    """Returns what the current time would be if no delays had occurred."""
    return time.time() - self.time_in_delay

  def _forgive_old(self):
    adjusted_now = self._adjusted_now()
    self.exceptions = [t for t in self.exceptions
                       if (adjusted_now - t) < self.forgive_after]

  def _handle_error(self, e):
    if not isinstance(e, self.handled):
      return True

    self._forgive_old()
    self.exceptions.append(self._adjusted_now())

    return len(self.exceptions) >= self.limit

  # pylint: disable=broad-except
  @contextlib.contextmanager
  def forgive(self):
    self._forgive_old()
    if self.exceptions:
      delay = self.delay * (self.backoff ** (len(self.exceptions) - 1))
      logging.warning('Sleeping for %f seconds before resuming' % delay)
      time.sleep(delay)
      self.time_in_delay += delay
    try:
      yield
    except Exception as e:
      if self._handle_error(e):
        raise
      else:
        logging.warning('Forgiving an exception', exc_info=True)
