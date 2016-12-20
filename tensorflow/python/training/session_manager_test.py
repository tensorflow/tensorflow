# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
      sm = tf.train.SessionManager(ready_op=tf.report_uninitialized_variables())
      sess = sm.prepare_session("", init_op=tf.global_variables_initializer())
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFeedDict(self):
    with tf.Graph().as_default():
      p = tf.placeholder(tf.float32, shape=(3,))
      v = tf.Variable(p, name="v")
      sm = tf.train.SessionManager(ready_op=tf.report_uninitialized_variables())
      sess = sm.prepare_session("",
                                init_op=tf.global_variables_initializer(),
                                init_feed_dict={p: [1.0, 2.0, 3.0]})
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFn(self):
    with tf.Graph().as_default():
      v = tf.Variable([125], name="v")
      sm = tf.train.SessionManager(ready_op=tf.report_uninitialized_variables())
      sess = sm.prepare_session("",
                                init_fn=lambda sess: sess.run(v.initializer))
      self.assertAllClose([125], sess.run(v))

  def testPrepareSessionFails(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), "prepare_session")
    checkpoint_dir2 = os.path.join(self.get_temp_dir(), "prepare_session2")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
      gfile.DeleteRecursively(checkpoint_dir2)
    except errors.OpError:
      pass                      # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with tf.Graph().as_default():
      v = tf.Variable([1.0, 2.0, 3.0], name="v")
      sm = tf.train.SessionManager(ready_op=tf.report_uninitialized_variables())
      saver = tf.train.Saver({"v": v})
      sess = sm.prepare_session("", init_op=tf.global_variables_initializer(),
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
      tf.train.SessionManager(ready_op=tf.report_uninitialized_variables())
      saver = tf.train.Saver({"v": v})
      # This should fail as there's no checkpoint within 2 seconds.
      with self.assertRaisesRegexp(
          RuntimeError, "no init_op or init_fn or local_init_op was given"):
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
    except errors.OpError:
      pass                      # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with tf.Graph().as_default():
      v = tf.Variable(1, name="v")
      sm = tf.train.SessionManager(ready_op=tf.report_uninitialized_variables())
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
      sm2 = tf.train.SessionManager(
          ready_op=tf.report_uninitialized_variables())
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
      sm = tf.train.SessionManager(ready_op=tf.report_uninitialized_variables(),
                                   recovery_wait_secs=1)

      # Set max_wait_secs to allow us to try a few times.
      with self.assertRaises(errors.DeadlineExceededError):
        sm.wait_for_session(master="", max_wait_secs=3)

  def testInitWithNoneLocalInitOpError(self):
    # Creating a SessionManager with a None local_init_op but
    # non-None ready_for_local_init_op raises ValueError
    with self.assertRaisesRegexp(ValueError,
                                 "If you pass a ready_for_local_init_op "
                                 "you must also pass a local_init_op "):
      tf.train.SessionManager(
          ready_for_local_init_op=tf.report_uninitialized_variables(
              tf.global_variables()),
          local_init_op=None)

  def testRecoverSessionWithReadyForLocalInitOp(self):
    # Create a checkpoint.
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  "recover_session_ready_for_local_init")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with tf.Graph().as_default():
      v = tf.Variable(1, name="v")
      sm = tf.train.SessionManager(ready_op=tf.report_uninitialized_variables())
      saver = tf.train.Saver({"v": v})
      sess, initialized = sm.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEquals(1, sess.run(v))
      saver.save(sess, os.path.join(checkpoint_dir,
                                    "recover_session_checkpoint"))
    # Create a new Graph and SessionManager and recover.
    with tf.Graph().as_default():
      v = tf.Variable(2, name="v")
      w = tf.Variable(
          v,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, tf.is_variable_initialized(v).eval())
        self.assertEqual(False, tf.is_variable_initialized(w).eval())
      sm2 = tf.train.SessionManager(
          ready_op=tf.report_uninitialized_variables(),
          ready_for_local_init_op=tf.report_uninitialized_variables(
              tf.global_variables()),
          local_init_op=w.initializer)
      saver = tf.train.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertTrue(initialized)
      self.assertEqual(
          True,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("v:0")).eval(
              session=sess))
      self.assertEqual(
          True,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("w:0")).eval(
              session=sess))
      self.assertEquals(1, sess.run(v))
      self.assertEquals(1, sess.run(w))

  def testRecoverSessionWithReadyForLocalInitOpFailsToReadyLocal(self):
    # We use ready_for_local_init_op=tf.report_uninitialized_variables(),
    # which causes recover_session to not run local_init_op, and to return
    # initialized=False

    # Create a checkpoint.
    checkpoint_dir = os.path.join(
        self.get_temp_dir(),
        "recover_session_ready_for_local_init_fails_to_ready_local")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with tf.Graph().as_default():
      v = tf.Variable(1, name="v")
      sm = tf.train.SessionManager(ready_op=tf.report_uninitialized_variables())
      saver = tf.train.Saver({"v": v})
      sess, initialized = sm.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEquals(1, sess.run(v))
      saver.save(sess, os.path.join(checkpoint_dir,
                                    "recover_session_checkpoint"))
    # Create a new Graph and SessionManager and recover.
    with tf.Graph().as_default():
      v = tf.Variable(2, name="v")
      w = tf.Variable(
          v,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, tf.is_variable_initialized(v).eval())
        self.assertEqual(False, tf.is_variable_initialized(w).eval())
      sm2 = tf.train.SessionManager(
          ready_op=tf.report_uninitialized_variables(),
          ready_for_local_init_op=tf.report_uninitialized_variables(),
          local_init_op=w.initializer)
      saver = tf.train.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      self.assertEqual(
          True,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("v:0")).eval(
              session=sess))
      self.assertEqual(
          False,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("w:0")).eval(
              session=sess))
      self.assertEquals(1, sess.run(v))

  def testRecoverSessionNoChkptStillRunsLocalInitOp(self):
    # This test checks for backwards compatibility.
    # In particular, we continue to ensure that recover_session will execute
    # local_init_op exactly once, regardless of whether the session was
    # successfully recovered.
    with tf.Graph().as_default():
      w = tf.Variable(
          1,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, tf.is_variable_initialized(w).eval())
      sm2 = tf.train.SessionManager(
          ready_op=tf.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
      # Try to recover session from None
      sess, initialized = sm2.recover_session(
          "", saver=None, checkpoint_dir=None)
      # Succeeds because recover_session still run local_init_op
      self.assertFalse(initialized)
      self.assertEqual(
          True,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("w:0")).eval(
              session=sess))
      self.assertEquals(1, sess.run(w))

  def testRecoverSessionFailsStillRunsLocalInitOp(self):
    # Create a checkpoint.
    checkpoint_dir = os.path.join(
        self.get_temp_dir(),
        "recover_session_ready_for_local_init_fails_stil_run")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    # Create a new Graph and SessionManager and recover.
    with tf.Graph().as_default():
      v = tf.Variable(2, name="v")
      w = tf.Variable(
          1,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, tf.is_variable_initialized(v).eval())
        self.assertEqual(False, tf.is_variable_initialized(w).eval())
      sm2 = tf.train.SessionManager(
          ready_op=tf.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
      saver = tf.train.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "",
          saver=saver,
          checkpoint_dir=checkpoint_dir,
          wait_for_checkpoint=False)
      self.assertFalse(initialized)
      self.assertEqual(
          False,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("v:0")).eval(
              session=sess))
      self.assertEqual(
          True,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("w:0")).eval(
              session=sess))
      self.assertEquals(1, sess.run(w))

  def testWaitForSessionLocalInit(self):
    server = tf.train.Server.create_local_server()
    with tf.Graph().as_default() as graph:
      v = tf.Variable(1, name="v")
      w = tf.Variable(
          v,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      sm = tf.train.SessionManager(
          graph=graph,
          ready_op=tf.report_uninitialized_variables(),
          ready_for_local_init_op=tf.report_uninitialized_variables(
              tf.global_variables()),
          local_init_op=w.initializer)

      # Initialize v but not w
      s = tf.Session(server.target, graph=graph)
      s.run(v.initializer)

      sess = sm.wait_for_session(server.target, max_wait_secs=3)
      self.assertEqual(
          True,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("v:0")).eval(
              session=sess))
      self.assertEqual(
          True,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("w:0")).eval(
              session=sess))
      self.assertEquals(1, sess.run(v))
      self.assertEquals(1, sess.run(w))

  def testWaitForSessionWithReadyForLocalInitOpFailsToReadyLocal(self):
    with tf.Graph().as_default() as graph:
      v = tf.Variable(1, name="v")
      w = tf.Variable(
          v,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      sm = tf.train.SessionManager(
          graph=graph,
          ready_op=tf.report_uninitialized_variables(),
          ready_for_local_init_op=tf.report_uninitialized_variables(),
          local_init_op=w.initializer)

      with self.assertRaises(tf.errors.DeadlineExceededError):
        # Time-out because w fails to be initialized,
        # because of overly restrictive ready_for_local_init_op
        sm.wait_for_session("", max_wait_secs=3)

  def testWaitForSessionInsufficientReadyForLocalInitCheck(self):
    with tf.Graph().as_default() as graph:
      v = tf.Variable(1, name="v")
      w = tf.Variable(
          v,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      sm = tf.train.SessionManager(
          graph=graph,
          ready_op=tf.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
    with self.assertRaisesRegexp(tf.errors.FailedPreconditionError,
                                 "Attempting to use uninitialized value v"):
      sm.wait_for_session("", max_wait_secs=3)

  def testPrepareSessionWithReadyForLocalInitOp(self):
    with tf.Graph().as_default():
      v = tf.Variable(1, name="v")
      w = tf.Variable(
          v,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, tf.is_variable_initialized(v).eval())
        self.assertEqual(False, tf.is_variable_initialized(w).eval())
      sm2 = tf.train.SessionManager(
          ready_op=tf.report_uninitialized_variables(),
          ready_for_local_init_op=tf.report_uninitialized_variables(
              tf.global_variables()),
          local_init_op=w.initializer)
      sess = sm2.prepare_session("", init_op=v.initializer)
      self.assertEqual(
          True,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("v:0")).eval(
              session=sess))
      self.assertEqual(
          True,
          tf.is_variable_initialized(sess.graph.get_tensor_by_name("w:0")).eval(
              session=sess))
      self.assertEquals(1, sess.run(v))
      self.assertEquals(1, sess.run(w))

  def testPrepareSessionDidNotInitLocalVariable(self):
    with tf.Graph().as_default():
      v = tf.Variable(1, name="v")
      w = tf.Variable(
          v,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, tf.is_variable_initialized(v).eval())
        self.assertEqual(False, tf.is_variable_initialized(w).eval())
      sm2 = tf.train.SessionManager(
          ready_op=tf.report_uninitialized_variables())
      with self.assertRaisesRegexp(RuntimeError,
                                   "Init operations did not make model ready"):
        sm2.prepare_session("", init_op=v.initializer)

  def testPrepareSessionWithReadyNotReadyForLocal(self):
    with tf.Graph().as_default():
      v = tf.Variable(1, name="v")
      w = tf.Variable(
          v,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, tf.is_variable_initialized(v).eval())
        self.assertEqual(False, tf.is_variable_initialized(w).eval())
      sm2 = tf.train.SessionManager(
          ready_op=tf.report_uninitialized_variables(),
          ready_for_local_init_op=tf.report_uninitialized_variables(
              tf.global_variables()),
          local_init_op=w.initializer)
      with self.assertRaisesRegexp(
          RuntimeError,
          "Init operations did not make model ready for local_init"):
        sm2.prepare_session("", init_op=None)

  def testPrepareSessionWithInsufficientReadyForLocalInitCheck(self):
    with tf.Graph().as_default():
      v = tf.Variable(1, name="v")
      w = tf.Variable(
          v,
          trainable=False,
          collections=[tf.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, tf.is_variable_initialized(v).eval())
        self.assertEqual(False, tf.is_variable_initialized(w).eval())
      sm2 = tf.train.SessionManager(
          ready_op=tf.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
    with self.assertRaisesRegexp(tf.errors.FailedPreconditionError,
                                 "Attempting to use uninitialized value v"):
      sm2.prepare_session("", init_op=None)


class ObsoleteSessionManagerTest(tf.test.TestCase):

  def testPrepareSessionSucceeds(self):
    with tf.Graph().as_default():
      v = tf.Variable([1.0, 2.0, 3.0], name="v")
      sm = tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
      sess = sm.prepare_session("", init_op=tf.global_variables_initializer())
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFeedDict(self):
    with tf.Graph().as_default():
      p = tf.placeholder(tf.float32, shape=(3,))
      v = tf.Variable(p, name="v")
      sm = tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
      sess = sm.prepare_session("",
                                init_op=tf.global_variables_initializer(),
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
    except errors.OpError:
      pass                      # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with tf.Graph().as_default():
      v = tf.Variable([1.0, 2.0, 3.0], name="v")
      sm = tf.train.SessionManager(ready_op=tf.assert_variables_initialized())
      saver = tf.train.Saver({"v": v})
      sess = sm.prepare_session("", init_op=tf.global_variables_initializer(),
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
      with self.assertRaisesRegexp(
          RuntimeError, "no init_op or init_fn or local_init_op was given"):
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
    except errors.OpError:
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
