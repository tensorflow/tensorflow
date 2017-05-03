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

from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_manager


class SessionManagerTest(test.TestCase):

  def testPrepareSessionSucceeds(self):
    with ops.Graph().as_default():
      v = variables.Variable([1.0, 2.0, 3.0], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      sess = sm.prepare_session(
          "", init_op=variables.global_variables_initializer())
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFeedDict(self):
    with ops.Graph().as_default():
      p = array_ops.placeholder(dtypes.float32, shape=(3,))
      v = variables.Variable(p, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      sess = sm.prepare_session(
          "",
          init_op=variables.global_variables_initializer(),
          init_feed_dict={p: [1.0, 2.0, 3.0]})
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFn(self):
    with ops.Graph().as_default():
      v = variables.Variable([125], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      sess = sm.prepare_session(
          "", init_fn=lambda sess: sess.run(v.initializer))
      self.assertAllClose([125], sess.run(v))

  def testPrepareSessionFails(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), "prepare_session")
    checkpoint_dir2 = os.path.join(self.get_temp_dir(), "prepare_session2")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
      gfile.DeleteRecursively(checkpoint_dir2)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with ops.Graph().as_default():
      v = variables.Variable([1.0, 2.0, 3.0], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      sess = sm.prepare_session(
          "",
          init_op=variables.global_variables_initializer(),
          saver=saver,
          checkpoint_dir=checkpoint_dir)
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))
      checkpoint_filename = os.path.join(checkpoint_dir,
                                         "prepare_session_checkpoint")
      saver.save(sess, checkpoint_filename)
    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      # Renames the checkpoint directory.
      os.rename(checkpoint_dir, checkpoint_dir2)
      gfile.MakeDirs(checkpoint_dir)
      v = variables.Variable([6.0, 7.0, 8.0], name="v")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
      session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      # This should fail as there's no checkpoint within 2 seconds.
      with self.assertRaisesRegexp(
          RuntimeError, "no init_op or init_fn or local_init_op was given"):
        sess = sm.prepare_session(
            "",
            init_op=None,
            saver=saver,
            checkpoint_dir=checkpoint_dir,
            wait_for_checkpoint=True,
            max_wait_secs=2)
      # Rename the checkpoint directory back.
      gfile.DeleteRecursively(checkpoint_dir)
      os.rename(checkpoint_dir2, checkpoint_dir)
      # This should succeed as there's checkpoint.
      sess = sm.prepare_session(
          "",
          init_op=None,
          saver=saver,
          checkpoint_dir=checkpoint_dir,
          wait_for_checkpoint=True,
          max_wait_secs=2)
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))

  def _test_recovered_variable(self,
                               checkpoint_dir=None,
                               checkpoint_filename_with_path=None):
    # Create a new Graph and SessionManager and recover from a checkpoint.
    with ops.Graph().as_default():
      v = variables.Variable(2, name="v")
      with session_lib.Session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "",
          saver=saver,
          checkpoint_dir=checkpoint_dir,
          checkpoint_filename_with_path=checkpoint_filename_with_path)
      self.assertTrue(initialized)
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEquals(1, sess.run(v))

  def testRecoverSession(self):
    # Create a checkpoint.
    checkpoint_dir = os.path.join(self.get_temp_dir(), "recover_session")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with ops.Graph().as_default():
      v = variables.Variable(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEquals(1, sess.run(v))
      saver.save(sess,
                 os.path.join(checkpoint_dir, "recover_session_checkpoint"))
    self._test_recovered_variable(checkpoint_dir=checkpoint_dir)
    self._test_recovered_variable(
        checkpoint_filename_with_path=saver_lib.latest_checkpoint(
            checkpoint_dir))
    # Cannot set both checkpoint_dir and checkpoint_filename_with_path.
    with self.assertRaises(ValueError):
      self._test_recovered_variable(
          checkpoint_dir=checkpoint_dir,
          checkpoint_filename_with_path=saver_lib.latest_checkpoint(
              checkpoint_dir))

  def testWaitForSessionReturnsNoneAfterTimeout(self):
    with ops.Graph().as_default():
      variables.Variable(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
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
      session_manager.SessionManager(
          ready_for_local_init_op=variables.report_uninitialized_variables(
              variables.global_variables()),
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

    with ops.Graph().as_default():
      v = variables.Variable(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEquals(1, sess.run(v))
      saver.save(sess,
                 os.path.join(checkpoint_dir, "recover_session_checkpoint"))
    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      v = variables.Variable(2, name="v")
      w = variables.Variable(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
        self.assertEqual(False, variables.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(
              variables.global_variables()),
          local_init_op=w.initializer)
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertTrue(initialized)
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
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

    with ops.Graph().as_default():
      v = variables.Variable(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEquals(1, sess.run(v))
      saver.save(sess,
                 os.path.join(checkpoint_dir, "recover_session_checkpoint"))
    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      v = variables.Variable(2, name="v")
      w = variables.Variable(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
        self.assertEqual(False, variables.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(),
          local_init_op=w.initializer)
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          False,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEquals(1, sess.run(v))

  def testRecoverSessionNoChkptStillRunsLocalInitOp(self):
    # This test checks for backwards compatibility.
    # In particular, we continue to ensure that recover_session will execute
    # local_init_op exactly once, regardless of whether the session was
    # successfully recovered.
    with ops.Graph().as_default():
      w = variables.Variable(
          1,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
      # Try to recover session from None
      sess, initialized = sm2.recover_session(
          "", saver=None, checkpoint_dir=None)
      # Succeeds because recover_session still run local_init_op
      self.assertFalse(initialized)
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
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
    with ops.Graph().as_default():
      v = variables.Variable(2, name="v")
      w = variables.Variable(
          1,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
        self.assertEqual(False, variables.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "",
          saver=saver,
          checkpoint_dir=checkpoint_dir,
          wait_for_checkpoint=False)
      self.assertFalse(initialized)
      self.assertEqual(
          False,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEquals(1, sess.run(w))

  def testWaitForSessionLocalInit(self):
    server = server_lib.Server.create_local_server()
    with ops.Graph().as_default() as graph:
      v = variables.Variable(1, name="v")
      w = variables.Variable(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      sm = session_manager.SessionManager(
          graph=graph,
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(
              variables.global_variables()),
          local_init_op=w.initializer)

      # Initialize v but not w
      s = session_lib.Session(server.target, graph=graph)
      s.run(v.initializer)

      sess = sm.wait_for_session(server.target, max_wait_secs=3)
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEquals(1, sess.run(v))
      self.assertEquals(1, sess.run(w))

  def testWaitForSessionWithReadyForLocalInitOpFailsToReadyLocal(self):
    with ops.Graph().as_default() as graph:
      v = variables.Variable(1, name="v")
      w = variables.Variable(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      sm = session_manager.SessionManager(
          graph=graph,
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(),
          local_init_op=w.initializer)

      with self.assertRaises(errors_impl.DeadlineExceededError):
        # Time-out because w fails to be initialized,
        # because of overly restrictive ready_for_local_init_op
        sm.wait_for_session("", max_wait_secs=3)

  def testWaitForSessionInsufficientReadyForLocalInitCheck(self):
    with ops.Graph().as_default() as graph:
      v = variables.Variable(1, name="v")
      w = variables.Variable(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      sm = session_manager.SessionManager(
          graph=graph,
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
    with self.assertRaisesRegexp(errors_impl.FailedPreconditionError,
                                 "Attempting to use uninitialized value v"):
      sm.wait_for_session("", max_wait_secs=3)

  def testPrepareSessionWithReadyForLocalInitOp(self):
    with ops.Graph().as_default():
      v = variables.Variable(1, name="v")
      w = variables.Variable(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
        self.assertEqual(False, variables.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(
              variables.global_variables()),
          local_init_op=w.initializer)
      sess = sm2.prepare_session("", init_op=v.initializer)
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEquals(1, sess.run(v))
      self.assertEquals(1, sess.run(w))

  def testPrepareSessionDidNotInitLocalVariable(self):
    with ops.Graph().as_default():
      v = variables.Variable(1, name="v")
      w = variables.Variable(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
        self.assertEqual(False, variables.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      with self.assertRaisesRegexp(RuntimeError,
                                   "Init operations did not make model ready"):
        sm2.prepare_session("", init_op=v.initializer)

  def testPrepareSessionDidNotInitLocalVariableList(self):
    with ops.Graph().as_default():
      v = variables.Variable(1, name="v")
      w = variables.Variable(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
        self.assertEqual(False, variables.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      with self.assertRaisesRegexp(RuntimeError,
                                   "Init operations did not make model ready"):
        sm2.prepare_session("", init_op=[v.initializer])

  def testPrepareSessionWithReadyNotReadyForLocal(self):
    with ops.Graph().as_default():
      v = variables.Variable(1, name="v")
      w = variables.Variable(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
        self.assertEqual(False, variables.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(
              variables.global_variables()),
          local_init_op=w.initializer)
      with self.assertRaisesRegexp(
          RuntimeError,
          "Init operations did not make model ready for local_init"):
        sm2.prepare_session("", init_op=None)

  def testPrepareSessionWithInsufficientReadyForLocalInitCheck(self):
    with ops.Graph().as_default():
      v = variables.Variable(1, name="v")
      w = variables.Variable(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
        self.assertEqual(False, variables.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
    with self.assertRaisesRegexp(errors_impl.FailedPreconditionError,
                                 "Attempting to use uninitialized value v"):
      sm2.prepare_session("", init_op=None)


class ObsoleteSessionManagerTest(test.TestCase):

  def testPrepareSessionSucceeds(self):
    with ops.Graph().as_default():
      v = variables.Variable([1.0, 2.0, 3.0], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      sess = sm.prepare_session(
          "", init_op=variables.global_variables_initializer())
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFeedDict(self):
    with ops.Graph().as_default():
      p = array_ops.placeholder(dtypes.float32, shape=(3,))
      v = variables.Variable(p, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      sess = sm.prepare_session(
          "",
          init_op=variables.global_variables_initializer(),
          init_feed_dict={p: [1.0, 2.0, 3.0]})
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFn(self):
    with ops.Graph().as_default():
      v = variables.Variable([125], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      sess = sm.prepare_session(
          "", init_fn=lambda sess: sess.run(v.initializer))
      self.assertAllClose([125], sess.run(v))

  def testPrepareSessionFails(self):
    checkpoint_dir = os.path.join(self.get_temp_dir(), "prepare_session")
    checkpoint_dir2 = os.path.join(self.get_temp_dir(), "prepare_session2")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
      gfile.DeleteRecursively(checkpoint_dir2)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with ops.Graph().as_default():
      v = variables.Variable([1.0, 2.0, 3.0], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      saver = saver_lib.Saver({"v": v})
      sess = sm.prepare_session(
          "",
          init_op=variables.global_variables_initializer(),
          saver=saver,
          checkpoint_dir=checkpoint_dir)
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))
      checkpoint_filename = os.path.join(checkpoint_dir,
                                         "prepare_session_checkpoint")
      saver.save(sess, checkpoint_filename)
    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      # Renames the checkpoint directory.
      os.rename(checkpoint_dir, checkpoint_dir2)
      gfile.MakeDirs(checkpoint_dir)
      v = variables.Variable([6.0, 7.0, 8.0], name="v")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
      session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      saver = saver_lib.Saver({"v": v})
      # This should fail as there's no checkpoint within 2 seconds.
      with self.assertRaisesRegexp(
          RuntimeError, "no init_op or init_fn or local_init_op was given"):
        sess = sm.prepare_session(
            "",
            init_op=None,
            saver=saver,
            checkpoint_dir=checkpoint_dir,
            wait_for_checkpoint=True,
            max_wait_secs=2)
      # Rename the checkpoint directory back.
      gfile.DeleteRecursively(checkpoint_dir)
      os.rename(checkpoint_dir2, checkpoint_dir)
      # This should succeed as there's checkpoint.
      sess = sm.prepare_session(
          "",
          init_op=None,
          saver=saver,
          checkpoint_dir=checkpoint_dir,
          wait_for_checkpoint=True,
          max_wait_secs=2)
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))

  def testRecoverSession(self):
    # Create a checkpoint.
    checkpoint_dir = os.path.join(self.get_temp_dir(), "recover_session")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with ops.Graph().as_default():
      v = variables.Variable(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEquals(1, sess.run(v))
      saver.save(sess,
                 os.path.join(checkpoint_dir, "recover_session_checkpoint"))
    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      v = variables.Variable(2, name="v")
      with self.test_session():
        self.assertEqual(False, variables.is_variable_initialized(v).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertTrue(initialized)
      self.assertEqual(
          True,
          variables.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEquals(1, sess.run(v))

  def testWaitForSessionReturnsNoneAfterTimeout(self):
    with ops.Graph().as_default():
      variables.Variable(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized(),
          recovery_wait_secs=1)

      # Set max_wait_secs to allow us to try a few times.
      with self.assertRaises(errors.DeadlineExceededError):
        sm.wait_for_session(master="", max_wait_secs=3)


if __name__ == "__main__":
  test.main()
