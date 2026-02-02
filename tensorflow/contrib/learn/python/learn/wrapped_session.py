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
"""Wrapper for a Session object that handles threads and recovery."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class WrappedSession(object):
  """Wrapper around a `tf.Session`.

  This wrapper is used as a base class for various session wrappers
  that provide additional functionality such as monitoring, coordination,
  and recovery.

  In addition to the methods exported by `SessionInterface` the wrapper
  provides a method to check for stop and never raises exceptions from
  calls to `close()`.
  """

  def __init__(self, sess):
    """Creates a `WrappedSession`.

    Args:
      sess: A `tf.Session` or `WrappedSession` object.  The wrapped session.
    """
    self._sess = sess
    self._wrapped_is_stoppable = isinstance(self._sess, WrappedSession)

  @property
  def graph(self):
    return self._sess.graph

  @property
  def sess_str(self):
    return self._sess.sess_str

  def should_stop(self):
    """Return true if this session should not be used anymore.

    Always return True if the session was closed.

    Returns:
      True if the session should stop, False otherwise.
    """
    if self._check_stop():
      return True
    if self._sess:
      return self._wrapped_is_stoppable and self._sess.should_stop()
    return True

  def _check_stop(self):
    """Hook for subclasses to provide their own stop condition.

    Returns:
      True if the session should stop, False otherwise.
    """
    return False

  def close(self):
    if self._sess:
      try:
        self._sess.close()
      except Exception:  # pylint: disable=broad-except
        pass
      finally:
        self._sess = None

  def run(self, *args, **kwargs):
    return self._sess.run(*args, **kwargs)
