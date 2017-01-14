# pylint: disable=g-bad-file-header
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
"""Tests for RecoverableSession."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import recoverable_session


class AbortAtNSession(object):
  """A mock sessionthat aborts at the N-th run call."""

  def __init__(self, sess, n):
    self._sess = sess
    self._count = n

  def run(self, *args, **kwargs):
    if self._count == 0:
      raise tf.errors.AbortedError('Aborted at N', None, None)
    self._count -= 1
    return self._sess.run(*args, **kwargs)


class RecoverableSessionTest(tf.test.TestCase):
  """RecoverableSession tests."""

  def test_properties(self):
    with self.test_session() as sess:
      tf.constant(0.0)
      recoverable_sess = recoverable_session.RecoverableSession(lambda: sess)
      self.assertEquals(sess.graph, recoverable_sess.graph)
      self.assertEquals(sess.sess_str, recoverable_sess.sess_str)

  def test_run(self):
    with self.test_session() as sess:
      c = tf.constant(0)
      v = tf.identity(c)
      recoverable_sess = recoverable_session.RecoverableSession(lambda: sess)
      self.assertEqual(51, recoverable_sess.run(v, feed_dict={c: 51}))

  def test_recovery(self):
    with self.test_session() as sess:
      c = tf.constant(0)
      v = tf.identity(c)
      # List of 3 sessions to use for recovery.  The first one aborts
      # after 1 run() call, the second after 2 run calls, the third
      # after 3 run calls.
      sessions_to_use = [AbortAtNSession(sess, x + 1)
                         for x in range(3)]
      self.assertEqual(3, len(sessions_to_use))
      # Make the recoverable session uses these 3 sessions in sequence by
      # passing a factory that pops from the session_to_use list.
      recoverable_sess = recoverable_session.RecoverableSession(
          lambda: sessions_to_use.pop(0))
      self.assertEqual(2, len(sessions_to_use))  # One session popped.
      # Using first session.
      self.assertEqual(51, recoverable_sess.run(v, feed_dict={c: 51}))
      self.assertEqual(2, len(sessions_to_use))  # Still 2 sessions available
      # This will fail and recover by picking up the second session.
      self.assertEqual(42, recoverable_sess.run(v, feed_dict={c: 42}))
      self.assertEqual(1, len(sessions_to_use))  # Still 1 session available
      self.assertEqual(33, recoverable_sess.run(v, feed_dict={c: 33}))
      self.assertEqual(1, len(sessions_to_use))  # Still 1 session available
      # This will fail and recover by picking up the last session.
      self.assertEqual(24, recoverable_sess.run(v, feed_dict={c: 24}))
      self.assertEqual(0, len(sessions_to_use))  # All sessions used.
      self.assertEqual(11, recoverable_sess.run(v, feed_dict={c: 11}))
      self.assertEqual(0, recoverable_sess.run(v, feed_dict={c: 0}))
      # This will fail and throw a real error as the pop() will fail.
      with self.assertRaisesRegexp(IndexError, 'pop from empty list'):
        recoverable_sess.run(v, feed_dict={c: -12})


if __name__ == '__main__':
  tf.test.main()
