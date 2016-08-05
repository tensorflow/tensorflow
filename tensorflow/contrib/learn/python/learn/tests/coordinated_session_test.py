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
"""Tests for CoordinatedSession."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import time

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import coordinated_session


def BusyWaitForCoordStop(coord):
  while not coord.should_stop():
    time.sleep(0.001)


class CoordinatedSessionTest(tf.test.TestCase):
  """CoordinatedSession tests."""

  def test_properties(self):
    with self.test_session() as sess:
      tf.constant(0.0)
      coord = tf.train.Coordinator()
      coord_sess = coordinated_session.CoordinatedSession(sess, coord, [])
      self.assertEquals(sess.graph, coord_sess.graph)
      self.assertEquals(sess.sess_str, coord_sess.sess_str)

  def test_run(self):
    with self.test_session() as sess:
      c = tf.constant(0)
      v = tf.identity(c)
      coord = tf.train.Coordinator()
      coord_sess = coordinated_session.CoordinatedSession(sess, coord, [])
      self.assertEqual(42, coord_sess.run(v, feed_dict={c: 42}))

  def test_should_stop_on_close(self):
    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      coord_sess = coordinated_session.CoordinatedSession(sess, coord, [])
      self.assertFalse(coord_sess.should_stop())
      coord_sess.close()
      self.assertTrue(coord_sess.should_stop())

  def test_should_stop_on_coord_stop(self):
    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      coord_sess = coordinated_session.CoordinatedSession(sess, coord, [])
      self.assertFalse(coord_sess.should_stop())
      coord.request_stop()
      self.assertTrue(coord_sess.should_stop())

  def test_request_stop_on_exception(self):
    with self.test_session() as sess:
      c = tf.constant(0)
      v = tf.identity(c)
      coord = tf.train.Coordinator()
      coord_sess = coordinated_session.CoordinatedSession(sess, coord, [])
      self.assertFalse(coord_sess.should_stop())
      self.assertEqual(0, coord_sess.run(c))
      self.assertEqual(1, coord_sess.run(v, feed_dict={c: 1}))
      with self.assertRaisesRegexp(TypeError, 'None has invalid type'):
        coord_sess.run([None], feed_dict={c: 2})
      self.assertTrue(coord.should_stop())
      self.assertTrue(coord_sess.should_stop())

  def test_stop_threads_on_exception(self):
    with self.test_session() as sess:
      c = tf.constant(0)
      v = tf.identity(c)
      coord = tf.train.Coordinator()
      threads = [threading.Thread(target=BusyWaitForCoordStop, args=(coord,))
                 for _ in range(3)]
      for t in threads:
        t.start()
      coord_sess = coordinated_session.CoordinatedSession(sess, coord, threads)
      self.assertFalse(coord_sess.should_stop())
      for t in threads:
        self.assertTrue(t.is_alive())
      self.assertEqual(0, coord_sess.run(c))
      for t in threads:
        self.assertTrue(t.is_alive())
      self.assertEqual(1, coord_sess.run(v, feed_dict={c: 1}))
      for t in threads:
        self.assertTrue(t.is_alive())
      with self.assertRaisesRegexp(TypeError, 'None has invalid type'):
        coord_sess.run([None], feed_dict={c: 2})
      for t in threads:
        self.assertFalse(t.is_alive())
      self.assertTrue(coord.should_stop())
      self.assertTrue(coord_sess.should_stop())

  def test_stop_threads_on_close(self):
    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      threads = [threading.Thread(target=BusyWaitForCoordStop,
                                  args=(coord,)) for _ in range(3)]
      for t in threads:
        t.start()
      coord_sess = coordinated_session.CoordinatedSession(sess, coord, threads)
      coord_sess.close()
      for t in threads:
        self.assertFalse(t.is_alive())
      self.assertTrue(coord.should_stop())
      self.assertTrue(coord_sess.should_stop())


if __name__ == '__main__':
  tf.test.main()
