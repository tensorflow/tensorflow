# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ===================================================================
"""ErrorRendezvous handler for collecting errors from multiple threads."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import sys
import threading
import time

import six

from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging

_UNINTERESTING_ERRORS = (errors.CancelledError,)


class ErrorRendezvous(object):
  """Resolve errors from multiple threads during TPU execution.

  TPU errors can occur on the infeed or outfeed threads as well as the main
  training thread.

  Depending on which thread "wins" and receives the session error first, we may
  end up showing users a confusing and non-actionable error message (session
  cancelled) instead of a root cause (e.g. a bad filename).

  The rendezvous object provides a location to capture these errors until all
  threads terminate.  At that point we can choose the most informative error
  to report.
  """

  def __init__(self, num_sources):
    # string -> (message, traceback)
    self._errors = {}
    self._num_sources = num_sources
    self._session_cancel_timer = None

  def record_error(self, source, exc_info, session=None):
    """Report an exception from the given source.

    If a session is passed, a timer will be registered to close it after a few
    seconds.  This is necessary to ensure the main training loop does not hang
    if an infeed/oufeed error occurs.  We sleep a few seconds to allow a more
    interesting error from another thread to propagate.

    Args:
      source: string, source of the error
      exc_info: Output from `sys.exc_info` (type, value, traceback)
      session: Session to close after delay.
    """
    _, value, _ = exc_info
    self._errors[source] = exc_info
    logging.info('Error recorded from %s: %s', source, value)

    if session is not None and self._session_cancel_timer is None:

      def _cancel_session():
        time.sleep(5)
        try:
          session.close()
        except:  # pylint: disable=bare-except
          pass

      self._session_cancel_timer = threading.Thread(target=_cancel_session,)
      self._session_cancel_timer.daemon = True
      self._session_cancel_timer.start()

  def record_done(self, source):
    """Mark execution source `source` as done.

    If an error was originally reported from `source` it is left intact.

    Args:
      source: `str`, source being recorded
    """
    logging.info('%s marked as finished', source)
    if source not in self._errors:
      self._errors[source] = None

  @contextlib.contextmanager
  def catch_errors(self, source, session=None):
    """Context manager to report any errors within a block."""
    try:
      yield
    except Exception:  # pylint: disable=broad-except
      self.record_error(source, sys.exc_info(), session)

  def raise_errors(self, timeout_sec=0):
    """Wait for up to `timeout` seconds for all error sources to finish.

    Preferentially raise "interesting" errors (errors not in the
    _UNINTERESTING_ERRORS) set.

    Args:
      timeout_sec: Seconds to wait for other error sources.
    """
    for _ in range(timeout_sec):
      if len(self._errors) == self._num_sources:
        break
      time.sleep(1)

    kept_errors = [(k, v) for (k, v) in self._errors.items() if v is not None]

    # First check for any interesting errors, then fall back on the session
    # cancelled errors etc.
    for k, (typ, value, traceback) in kept_errors:
      if isinstance(value, _UNINTERESTING_ERRORS):
        continue
      else:
        logging.warn('Reraising captured error')
        six.reraise(typ, value, traceback)

    for k, (typ, value, traceback) in kept_errors:
      logging.warn('Reraising captured error')
      six.reraise(typ, value, traceback)
