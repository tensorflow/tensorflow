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
"""Tests for WrappedSession."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import wrapped_session


class StopAtNSession(wrapped_session.WrappedSession):
  """A wrapped session that stops at the N-th call to _check_stop."""

  def __init__(self, sess, n):
    super(StopAtNSession, self).__init__(sess)
    self._count = n

  def _check_stop(self):
    if self._count == 0:
      return True
    self._count -= 1
    return False


class WrappedSessionTest(tf.test.TestCase):
  """WrappedSession tests."""

  def test_properties(self):
    with self.test_session() as sess:
      tf.constant(0.0)
      wrapped_sess = wrapped_session.WrappedSession(sess)
      self.assertEquals(sess.graph, wrapped_sess.graph)
      self.assertEquals(sess.sess_str, wrapped_sess.sess_str)

  def test_should_stop_on_close(self):
    with self.test_session() as sess:
      wrapped_sess = wrapped_session.WrappedSession(sess)
      self.assertFalse(wrapped_sess.should_stop())
      wrapped_sess.close()
      self.assertTrue(wrapped_sess.should_stop())

  def test_should_stop_uses_check_stop(self):
    with self.test_session() as sess:
      wrapped_sess = StopAtNSession(sess, 3)
      self.assertFalse(wrapped_sess.should_stop())
      self.assertFalse(wrapped_sess.should_stop())
      self.assertFalse(wrapped_sess.should_stop())
      self.assertTrue(wrapped_sess.should_stop())

  def test_should_stop_delegates_to_wrapped_session(self):
    with self.test_session() as sess:
      wrapped_sess0 = StopAtNSession(sess, 4)
      wrapped_sess1 = wrapped_session.WrappedSession(wrapped_sess0)
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertTrue(wrapped_sess1.should_stop())

  def test_close_twice(self):
    with self.test_session() as sess:
      wrapped_sess = wrapped_session.WrappedSession(sess)
      wrapped_sess.close()
      self.assertTrue(wrapped_sess.should_stop())
      wrapped_sess.close()
      self.assertTrue(wrapped_sess.should_stop())

  def test_run(self):
    with self.test_session() as sess:
      c = tf.constant(0)
      v = tf.identity(c)
      self.assertEqual(42, sess.run(v, feed_dict={c: 42}))
      wrapped_sess = wrapped_session.WrappedSession(sess)
      self.assertEqual(51, wrapped_sess.run(v, feed_dict={c: 51}))


if __name__ == '__main__':
  tf.test.main()
