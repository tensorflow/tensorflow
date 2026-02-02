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
from tensorflow.python.framework import errors


class RecoverableSession(WrappedSession):
  """A wrapped session that recreates a session on `tf.errors.AbortedError`.

  The constructor is passed a session _factory_, not a session.  The factory is
  a no-argument function that must return a `Session`.

  Calls to `run()` are delegated to the wrapped session.  If a call raises the
  exception `tf.errors.AbortedError`, the wrapped session is closed, and a new
  one is created by calling the factory again.
  """

  def __init__(self, sess_factory):
    """Create a new `RecoverableSession`.

    The value returned by calling `sess_factory()` will be the
    session wrapped by this recoverable session.

    Args:
      sess_factory: A callable with no arguments that returns a
        `tf.Session` when called.
    """
    self._factory = sess_factory
    WrappedSession.__init__(self, sess_factory())

  def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
    while True:
      try:
        if not self._sess:
          self._sess = self._factory()
        return self._sess.run(fetches, feed_dict=feed_dict, options=options,
                              run_metadata=run_metadata)
      except errors.AbortedError:
        self.close()
        self._sess = None
