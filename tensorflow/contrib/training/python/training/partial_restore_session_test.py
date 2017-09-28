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
"""Tests for basic_session_run_hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import time

from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.testing.python.framework import util_test
from tensorflow.contrib.training.python.training import partial_restore_session
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.training import saver
from tensorflow.python.training.monitored_session_test import FakeHook


class CheckpointRestorerHookTest(test.TestCase):

  def setUp(self):
    self.model_dir = tempfile.mkdtemp()
    self.checkpoint_path = os.path.join(self.model_dir, "model.ckpt")
    self.graph = ops.Graph()
    with self.graph.as_default():
      self.global_step = variables.get_or_create_global_step()
      self.train_op = state_ops.assign_add(self.global_step, 1)
      self.saver = saver.Saver()
      with session_lib.Session() as sess:
        sess.run(variables_lib.global_variables_initializer())
        sess.run(self.train_op)  # global_step == 1
        result = self.saver.save(sess, self.checkpoint_path)

  def tearDown(self):
    shutil.rmtree(self.model_dir, ignore_errors=True)
    pass

  def test_restore_all_variables(self):
    with ops.Graph().as_default():
      global_step = variables.get_or_create_global_step()
      hook = partial_restore_session.CheckpointRestorerHook(self.model_dir)
      with session_lib.Session() as sess:
        sess.run(variables_lib.global_variables_initializer())
        hook.after_create_session(sess, None)
        self.assertEqual(1, sess.run(global_step))

  def test_raise_missing_variables_when_restore_all(self):
    with ops.Graph().as_default():
      global_step = variables.get_or_create_global_step()
      new_var = variables_lib.Variable(0.0, name="new_var")
      hook = partial_restore_session.CheckpointRestorerHook(self.model_dir)
      with session_lib.Session() as sess:
        sess.run(variables_lib.global_variables_initializer())
        with self.assertRaises(errors.NotFoundError):
          hook.after_create_session(sess, None)

  def test_partial_restore_from_dir(self):
    with ops.Graph().as_default():
      global_step = variables.get_or_create_global_step()
      new_var = variables_lib.Variable(0.0, name="new_var")
      hook = partial_restore_session.CheckpointRestorerHook(
        checkpoint_dir=self.model_dir, var_list=[global_step])
      with session_lib.Session() as sess:
        sess.run(variables_lib.global_variables_initializer())
        hook.after_create_session(sess, None)
        self.assertEqual(1, sess.run(global_step))

  def test_partial_restore_from_file(self):
    with ops.Graph().as_default():
      global_step = variables.get_or_create_global_step()
      new_var = variables_lib.Variable(0.0, name="new_var")
      hook = partial_restore_session.CheckpointRestorerHook(
        checkpoint_file=self.checkpoint_path, var_list=[global_step])
      with session_lib.Session() as sess:
        sess.run(variables_lib.global_variables_initializer())
        hook.after_create_session(sess, None)
        self.assertEqual(1, sess.run(global_step))


class MonitoredTrainingSessionTest(test.TestCase):
  """Tests MonitoredTrainingSession."""

  def setUp(self):
    self.model_dir = tempfile.mkdtemp()
    self.checkpoint_path = os.path.join(self.model_dir, "model.ckpt")

  def tearDown(self):
    shutil.rmtree(self.model_dir, ignore_errors=True)

  def test_saving_restoring_checkpoint(self):
    logdir = self.model_dir
    # logdir = _test_dir(self.get_temp_dir(), 'test_saving_restoring_checkpoint')
    with ops.Graph().as_default():
      gstep = variables.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      with partial_restore_session.PartialRestoreSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
      # A restart will find the checkpoint and recover automatically.
      with partial_restore_session.PartialRestoreSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(2, session.run(gstep))

  def test_raise_saving_restoring_partial_checkpoint(self):
    logdir = self.model_dir
    # logdir = _test_dir(self.get_temp_dir(), 'test_saving_restoring_checkpoint')
    with ops.Graph().as_default():
      gstep = variables.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      with partial_restore_session.PartialRestoreSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
    with ops.Graph().as_default():
      gstep = variables.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      new_var = variables_lib.Variable(0, name="new_var")
      # A restart will find the checkpoint and attempt to recover all variables,
      # and raise error because of missing `new_var`.
      with self.assertRaises(errors.NotFoundError):
        with partial_restore_session.PartialRestoreSession(
            is_chief=True, checkpoint_dir=logdir) as session:
          self.assertEqual(2, session.run(gstep))

  def test_saving_restoring_partial_checkpoint(self):
    logdir = self.model_dir
    # logdir = _test_dir(self.get_temp_dir(), 'test_saving_restoring_checkpoint')
    with ops.Graph().as_default():
      gstep = variables.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      with partial_restore_session.PartialRestoreSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
    with ops.Graph().as_default():
      gstep = variables.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      new_var = variables_lib.Variable(0, name="new_var")
      # A restart will find the checkpoint and recover automatically.
      with partial_restore_session.PartialRestoreSession(
          is_chief=True, checkpoint_dir=logdir, restore_var_list=[gstep]) as session:
        self.assertEqual(2, session.run(gstep))

  def test_summaries_steps(self):
    logdir = self.model_dir
    # logdir = _test_dir(self.get_temp_dir(), 'test_summaries_steps')
    with ops.Graph().as_default():
      gstep = variables.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      summary_lib.scalar('my_summary_tag', new_gstep * 2)
      with partial_restore_session.PartialRestoreSession(
          is_chief=True,
          checkpoint_dir=logdir,
          save_summaries_steps=100,
          log_step_count_steps=10) as session:
        for _ in range(101):
          session.run(new_gstep)
    summaries = util_test.latest_summaries(logdir)
    tags = [s.summary.value[0].tag for s in summaries]
    self.assertIn('my_summary_tag', tags)
    self.assertIn('global_step/sec', tags)

  def test_summaries_secs(self):
    logdir = self.model_dir
    # logdir = _test_dir(self.get_temp_dir(), 'test_summaries_secs')
    with ops.Graph().as_default():
      gstep = variables.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      summary_lib.scalar('my_summary_tag', new_gstep * 2)
      with partial_restore_session.PartialRestoreSession(
          is_chief=True,
          checkpoint_dir=logdir,
          save_summaries_steps=None,
          save_summaries_secs=0.1,
          log_step_count_steps=10) as session:
        session.run(new_gstep)
        time.sleep(0.2)
        for _ in range(101):
          session.run(new_gstep)
    summaries = util_test.latest_summaries(logdir)
    tags = [s.summary.value[0].tag for s in summaries]
    self.assertIn('my_summary_tag', tags)
    self.assertIn('global_step/sec', tags)

  def test_custom_saving(self):
    logdir = self.model_dir
    # logdir = _test_dir(self.get_temp_dir(), 'test_saving_restoring_checkpoint')
    fake_hook = FakeHook()
    with ops.Graph().as_default():
      gstep = variables.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      with partial_restore_session.PartialRestoreSession(
          is_chief=True,
          checkpoint_dir=logdir,
          chief_only_hooks=[fake_hook],
          save_checkpoint_secs=0) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))

      # Check whether custom hook called or not
      self.assertEqual(1, fake_hook.call_counter['begin'])
      # A restart will not find the checkpoint, since we didn't save.
      with partial_restore_session.PartialRestoreSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(0, session.run(gstep))


if __name__ == '__main__':
  test.main()
