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

from tensorflow.contrib.learn.python.learn.wrapped_session import WrappedSession


class CoordinatedSession(WrappedSession):
  """A wrapped session that works with a `tf.Coordinator`.

  Calls to `run()` are delegated to the wrapped session.  If a call
  raises an exception, the exception is reported to the coordinator.

  In addition, after each call to `run()` this session ask the coordinator if
  the session should stop.  In that case it will will join all the coordinated
  threads passed to the constructor before returning.

  If the coordinator was requested to stop with an exception, that exception
  will be re-raised from the call to `run()`.
  """

  def __init__(self, sess, coord, coordinated_threads_to_join):
    """Create a new `CoordinatedSession`.

    Args:
      sess: A `tf.Session` object.  The wrapped session.
      coord: A `tf.train.Coordinator` object.
      coordinated_threads_to_join: A list of threads.
    """
    WrappedSession.__init__(self, sess)
    self._coord = coord
    self._coordinated_threads_to_join = coordinated_threads_to_join

  def _check_stop(self):
    # Check with the coordinator if we should stop.
    return self._coord.should_stop()

  def close(self):
    try:
      if not self._coord.should_stop():
        self._coord.request_stop()
        self._coord.join(self._coordinated_threads_to_join)
    except Exception:  # pylint: disable=broad-except
      # Don't raise exception at close
      pass
    finally:
      WrappedSession.close(self)

  def run(self, *args, **kwargs):
    try:
      return self._sess.run(*args, **kwargs)
    except Exception as e:  # pylint: disable=broad-except
      self._coord.request_stop(e)
    if self._coord.should_stop():
      self._coord.join(self._coordinated_threads_to_join)
