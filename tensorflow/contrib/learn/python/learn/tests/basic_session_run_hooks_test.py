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


import shutil
import tempfile
import time

import tensorflow as tf

from tensorflow.contrib import testing
from tensorflow.contrib.learn.python.learn import basic_session_run_hooks
from tensorflow.contrib.learn.python.learn import monitored_session


class StopAtStepTest(tf.test.TestCase):

  def test_raise_in_both_last_step_and_num_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.StopAtStepHook(num_steps=10, last_step=20)

  def test_stop_based_on_last_step(self):
    h = basic_session_run_hooks.StopAtStepHook(last_step=10)
    with tf.Graph().as_default():
      global_step = tf.contrib.framework.get_or_create_global_step()
      no_op = tf.no_op()
      h.begin()
      with tf.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [h])
        sess.run(tf.assign(global_step, 5))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(tf.assign(global_step, 9))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(tf.assign(global_step, 10))
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())
        sess.run(tf.assign(global_step, 11))
        mon_sess._should_stop = False
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())

  def test_stop_based_on_num_step(self):
    h = basic_session_run_hooks.StopAtStepHook(num_steps=10)

    with tf.Graph().as_default():
      global_step = tf.contrib.framework.get_or_create_global_step()
      no_op = tf.no_op()
      h.begin()
      with tf.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [h])
        sess.run(tf.assign(global_step, 5))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(tf.assign(global_step, 13))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(tf.assign(global_step, 14))
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())
        sess.run(tf.assign(global_step, 15))
        mon_sess._should_stop = False
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())


class LoggingTensorHookTest(tf.test.TestCase):

  def setUp(self):
    # Mock out logging calls so we can verify whether correct tensors are being
    # monitored.
    self._actual_log = tf.logging.info
    self.logged_message = None

    def mock_log(*args, **kwargs):
      self.logged_message = args
      self._actual_log(*args, **kwargs)

    tf.logging.info = mock_log

  def tearDown(self):
    tf.logging.info = self._actual_log

  def test_illegal_args(self):
    with self.assertRaisesRegexp(ValueError, 'nvalid every_n_iter'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'], every_n_iter=0)
    with self.assertRaisesRegexp(ValueError, 'nvalid every_n_iter'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'], every_n_iter=-10)

  def test_print(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      t = tf.constant(42.0, name='foo')
      train_op = tf.constant(3)
      hook = basic_session_run_hooks.LoggingTensorHook(tensors=[t.name],
                                                       every_n_iter=10)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      sess.run(tf.initialize_all_variables())
      mon_sess.run(train_op)
      self.assertRegexpMatches(str(self.logged_message), t.name)
      for j in range(3):
        _ = j
        self.logged_message = ''
        for i in range(9):
          _ = i
          mon_sess.run(train_op)
          # assertNotRegexpMatches is not supported by python 3.1 and later
          self.assertEqual(str(self.logged_message).find(t.name), -1)
        mon_sess.run(train_op)
        self.assertRegexpMatches(str(self.logged_message), t.name)


class CheckpointSaverHookTest(tf.test.TestCase):

  def setUp(self):
    self.model_dir = tempfile.mkdtemp()
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.scaffold = monitored_session.Scaffold()
      self.global_step = tf.contrib.framework.get_or_create_global_step()
      self.train_op = tf.assign_add(self.global_step, 1)

  def tearDown(self):
    shutil.rmtree(self.model_dir, ignore_errors=True)

  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=10, save_steps=20)

  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(self.model_dir)

  def test_save_secs_saves_in_first_step(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with tf.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(1, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))

  def test_save_secs_saves_periodically(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with tf.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(1, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        time.sleep(2.5)
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(3, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(3, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        time.sleep(2.5)
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(6, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))

  def test_save_steps_saves_in_first_step(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with tf.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(1, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))

  def test_save_steps_saves_periodically(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with tf.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(1, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(3, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(3, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(5, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))

  def test_save_saves_at_end(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with tf.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        hook.end(sess)
        self.assertEqual(2, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))


class StepCounterHookTest(tf.test.TestCase):

  def setUp(self):
    self.log_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.log_dir, ignore_errors=True)

  def test_step_counter(self):
    with tf.Graph().as_default() as g, tf.Session() as sess:
      global_step = tf.contrib.framework.get_or_create_global_step()
      train_op = tf.assign_add(global_step, 1)
      summary_writer = testing.FakeSummaryWriter(self.log_dir, g)
      hook = basic_session_run_hooks.StepCounterHook(
          summary_writer=summary_writer, every_n_steps=10)
      hook.begin()
      sess.run(tf.initialize_all_variables())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(30):
        time.sleep(0.01)
        mon_sess.run(train_op)
      hook.end(sess)
      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      for step in [11, 21]:
        summary_value = summary_writer.summaries[step][0].value[0]
        self.assertTrue(summary_value.tag, 'global_step/sec')
        # check at least 10 steps per sec is recorded.
        self.assertGreater(summary_value.simple_value, 10)


class SummarySaverHookTest(tf.test.TestCase):

  def test_summary_saver(self):
    with tf.Graph().as_default() as g, tf.Session() as sess:
      log_dir = 'log/dir'
      summary_writer = testing.FakeSummaryWriter(log_dir, g)
      var = tf.Variable(0.0)
      tensor = tf.assign_add(var, 1.0)
      summary_op = tf.scalar_summary('my_summary', tensor)
      global_step = tf.contrib.framework.get_or_create_global_step()
      train_op = tf.assign_add(global_step, 1)
      hook = basic_session_run_hooks.SummarySaverHook(
          summary_op=summary_op, save_steps=8, summary_writer=summary_writer)
      hook.begin()
      sess.run(tf.initialize_all_variables())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for i in range(30):
        _ = i
        mon_sess.run(train_op)
      hook.end(sess)
      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=log_dir,
          expected_graph=g,
          expected_summaries={
              1: {'my_summary': 1.0},
              9: {'my_summary': 2.0},
              17: {'my_summary': 3.0},
              25: {'my_summary': 4.0},
          })


if __name__ == '__main__':
  tf.test.main()
