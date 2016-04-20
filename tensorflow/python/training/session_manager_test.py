# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Tests for SessionManager."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow.python.framework import errors
from tensorflow.python.platform import gfile


class SessionManagerTest(tf.test.TestCase):

  def testPrepareSessionSucceeds(self):
    with tf.Graph().as_default():
      v = tf.Variable([1.0, 2.0, 3.0], name="v")
      sm = tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
      sess = sm.prepare_session("", init_op=tf.initialize_all_variables())
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFeedDict(self):
    with tf.Graph().as_default():
      p = tf.placeholder(tf.float32, shape=(3,))
      v = tf.Variable(p, name="v")
      sm = tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
      sess = sm.prepare_session("",
                                init_op=tf.initialize_all_variables(),
                                init_feed_dict={p: [1.0, 2.0, 3.0]})
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFn(self):
    with tf.Graph().as_default():
      v = tf.Variable([125], name="v")
      sm = tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
      sess = sm.prepare_session("",
                                init_fn=lambda sess: sess.run(v.initializer))
      self.assertAllClose([125], sess.run(v))

  def testPrepareSessionFails(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), "prepare_session")
    checkpoint_dir2 = os.path.join(self.get_temp_dir(), "prepare_session2")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
      gfile.DeleteRecursively(checkpoint_dir2)
    except OSError:
      pass                      # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with tf.Graph().as_default():
      v = tf.Variable([1.0, 2.0, 3.0], name="v")
      sm = tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
      saver = tf.train.Saver({"v": v})
      sess = sm.prepare_session("", init_op=tf.initialize_all_variables(),
                                saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))
      checkpoint_filename = os.path.join(checkpoint_dir,
                                         "prepare_session_checkpoint")
      saver.save(sess, checkpoint_filename)
    # Create a new Graph and SessionManager and recover.
    with tf.Graph().as_default():
      # Renames the checkpoint directory.
      os.rename(checkpoint_dir, checkpoint_dir2)
      gfile.MakeDirs(checkpoint_dir)
      v = tf.Variable([6.0, 7.0, 8.0], name="v")
      with self.test_session():
        self.assertEqual(False, tf.is_variable_initialized(v).eval())
      tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
      saver = tf.train.Saver({"v": v})
      # This should fail as there's no checkpoint within 2 seconds.
      with self.assertRaisesRegexp(RuntimeError,
                                   "no init_op or init_fn was given"):
        sess = sm.prepare_session("", init_op=None, saver=saver,
                                  checkpoint_dir=checkpoint_dir,
                                  wait_for_checkpoint=True, max_wait_secs=2)
      # Rename the checkpoint directory back.
      gfile.DeleteRecursively(checkpoint_dir)
      os.rename(checkpoint_dir2, checkpoint_dir)
      # This should succeed as there's checkpoint.
      sess = sm.prepare_session("", init_op=None, saver=saver,
                                checkpoint_dir=checkpoint_dir,
                                wait_for_checkpoint=True, max_wait_secs=2)
      self.assertEqual(
          True, tf.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))

  def testRecoverSession(self):
    # Create a checkpoint.
    checkpoint_dir = os.path.join(self.get_temp_dir(), "recover_session")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except OSError:
      pass                      # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with tf.Graph().as_default():
      v = tf.Variable(1, name="v")
      sm = tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
      saver = tf.train.Saver({"v": v})
      sess, initialized = sm.recover_session("", saver=saver,
                                             checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEquals(1, sess.run(v))
      saver.save(sess, os.path.join(checkpoint_dir,
                                    "recover_session_checkpoint"))
    # Create a new Graph and SessionManager and recover.
    with tf.Graph().as_default():
      v = tf.Variable(2, name="v")
      with self.test_session():
        self.assertEqual(False, tf.is_variable_initialized(v).eval())
      sm2 = tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
      saver = tf.train.Saver({"v": v})
      sess, initialized = sm2.recover_session("", saver=saver,
                                              checkpoint_dir=checkpoint_dir)
      self.assertTrue(initialized)
      self.assertEqual(
          True, tf.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEquals(1, sess.run(v))

  def testWaitForSessionReturnsNoneAfterTimeout(self):
    with tf.Graph().as_default():
      tf.Variable(1, name="v")
      sm = tf.train.SessionManager(ready_op=tf.assert_variables_initialized(),
                                   recovery_wait_secs=1)

      # Set max_wait_secs to allow us to try a few times.
      with self.assertRaises(errors.DeadlineExceededError):
        sm.wait_for_session(master="", max_wait_secs=3)

if __name__ == "__main__":
  tf.test.main()
