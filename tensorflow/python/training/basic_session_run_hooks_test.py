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

import os.path
import shutil
import tempfile
import time

from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.summary.writer import fake_summary_writer
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util


# Provide a realistic start time for unit tests where we need to mock out
# calls to time.time().
MOCK_START_TIME = 1484695987.209386


class MockCheckpointSaverListener(
    basic_session_run_hooks.CheckpointSaverListener):

  def __init__(self):
    self.begin_count = 0
    self.before_save_count = 0
    self.after_save_count = 0
    self.end_count = 0
    self.ask_for_stop = False

  def begin(self):
    self.begin_count += 1

  def before_save(self, session, global_step):
    self.before_save_count += 1

  def after_save(self, session, global_step):
    self.after_save_count += 1
    if self.ask_for_stop:
      return True

  def end(self, session, global_step):
    self.end_count += 1

  def get_counts(self):
    return {
        'begin': self.begin_count,
        'before_save': self.before_save_count,
        'after_save': self.after_save_count,
        'end': self.end_count
    }


class SecondOrStepTimerTest(test.TestCase):

  @test_util.run_deprecated_v1
  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SecondOrStepTimer(every_secs=2.0, every_steps=10)

  @test_util.run_deprecated_v1
  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SecondOrStepTimer()

  @test.mock.patch.object(time, 'time')
  def test_every_secs(self, mock_time):
    mock_time.return_value = MOCK_START_TIME
    timer = basic_session_run_hooks.SecondOrStepTimer(every_secs=1.0)
    self.assertTrue(timer.should_trigger_for_step(1))

    timer.update_last_triggered_step(1)
    self.assertFalse(timer.should_trigger_for_step(1))
    self.assertFalse(timer.should_trigger_for_step(2))

    mock_time.return_value += 1.0
    self.assertFalse(timer.should_trigger_for_step(1))
    self.assertTrue(timer.should_trigger_for_step(2))

  def test_every_steps(self):
    timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=3)
    self.assertTrue(timer.should_trigger_for_step(1))

    timer.update_last_triggered_step(1)
    self.assertFalse(timer.should_trigger_for_step(1))
    self.assertFalse(timer.should_trigger_for_step(2))
    self.assertFalse(timer.should_trigger_for_step(3))
    self.assertTrue(timer.should_trigger_for_step(4))

  def test_update_last_triggered_step(self):
    timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=1)

    elapsed_secs, elapsed_steps = timer.update_last_triggered_step(1)
    self.assertEqual(None, elapsed_secs)
    self.assertEqual(None, elapsed_steps)

    elapsed_secs, elapsed_steps = timer.update_last_triggered_step(5)
    self.assertLess(0, elapsed_secs)
    self.assertEqual(4, elapsed_steps)

    elapsed_secs, elapsed_steps = timer.update_last_triggered_step(7)
    self.assertLess(0, elapsed_secs)
    self.assertEqual(2, elapsed_steps)


class StopAtStepTest(test.TestCase):

  def test_raise_in_both_last_step_and_num_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.StopAtStepHook(num_steps=10, last_step=20)

  def test_stop_based_on_last_step(self):
    h = basic_session_run_hooks.StopAtStepHook(last_step=10)
    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      no_op = control_flow_ops.no_op()
      h.begin()
      with session_lib.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [h])
        sess.run(state_ops.assign(global_step, 5))
        h.after_create_session(sess, None)
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 9))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 10))
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 11))
        mon_sess._should_stop = False
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())

  def test_stop_based_on_num_step(self):
    h = basic_session_run_hooks.StopAtStepHook(num_steps=10)

    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      no_op = control_flow_ops.no_op()
      h.begin()
      with session_lib.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [h])
        sess.run(state_ops.assign(global_step, 5))
        h.after_create_session(sess, None)
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 13))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 14))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 15))
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 16))
        mon_sess._should_stop = False
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())

  def test_stop_based_with_multiple_steps(self):
    h = basic_session_run_hooks.StopAtStepHook(num_steps=10)

    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      no_op = control_flow_ops.no_op()
      h.begin()
      with session_lib.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [h])
        sess.run(state_ops.assign(global_step, 5))
        h.after_create_session(sess, None)
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 15))
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())


class LoggingTensorHookTest(test.TestCase):

  def setUp(self):
    # Mock out logging calls so we can verify whether correct tensors are being
    # monitored.
    self._actual_log = tf_logging.info
    self.logged_message = None

    def mock_log(*args, **kwargs):
      self.logged_message = args
      self._actual_log(*args, **kwargs)

    tf_logging.info = mock_log

  def tearDown(self):
    tf_logging.info = self._actual_log

  def test_illegal_args(self):
    with self.assertRaisesRegex(ValueError, 'nvalid every_n_iter'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'], every_n_iter=0)
    with self.assertRaisesRegex(ValueError, 'nvalid every_n_iter'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'], every_n_iter=-10)
    with self.assertRaisesRegex(ValueError, 'xactly one of'):
      basic_session_run_hooks.LoggingTensorHook(
          tensors=['t'], every_n_iter=5, every_n_secs=5)
    with self.assertRaisesRegex(ValueError, 'xactly one of'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'])

  def test_print_at_end_only(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)
      hook = basic_session_run_hooks.LoggingTensorHook(
          tensors=[t.name], at_end=True)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.evaluate(variables_lib.global_variables_initializer())
      self.logged_message = ''
      for _ in range(3):
        mon_sess.run(train_op)
        # assertNotRegexpMatches is not supported by python 3.1 and later
        self.assertEqual(str(self.logged_message).find(t.name), -1)

      hook.end(sess)
      self.assertRegex(str(self.logged_message), t.name)

  def _validate_print_every_n_steps(self, sess, at_end):
    t = constant_op.constant(42.0, name='foo')

    train_op = constant_op.constant(3)
    hook = basic_session_run_hooks.LoggingTensorHook(
        tensors=[t.name], every_n_iter=10, at_end=at_end)
    hook.begin()
    mon_sess = monitored_session._HookedSession(sess, [hook])
    self.evaluate(variables_lib.global_variables_initializer())
    mon_sess.run(train_op)
    self.assertRegex(str(self.logged_message), t.name)
    for _ in range(3):
      self.logged_message = ''
      for _ in range(9):
        mon_sess.run(train_op)
        # assertNotRegexpMatches is not supported by python 3.1 and later
        self.assertEqual(str(self.logged_message).find(t.name), -1)
      mon_sess.run(train_op)
      self.assertRegex(str(self.logged_message), t.name)

    # Add additional run to verify proper reset when called multiple times.
    self.logged_message = ''
    mon_sess.run(train_op)
    # assertNotRegexpMatches is not supported by python 3.1 and later
    self.assertEqual(str(self.logged_message).find(t.name), -1)

    self.logged_message = ''
    hook.end(sess)
    if at_end:
      self.assertRegex(str(self.logged_message), t.name)
    else:
      # assertNotRegexpMatches is not supported by python 3.1 and later
      self.assertEqual(str(self.logged_message).find(t.name), -1)

  def test_print_every_n_steps(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      self._validate_print_every_n_steps(sess, at_end=False)
      # Verify proper reset.
      self._validate_print_every_n_steps(sess, at_end=False)

  def test_print_every_n_steps_and_end(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      self._validate_print_every_n_steps(sess, at_end=True)
      # Verify proper reset.
      self._validate_print_every_n_steps(sess, at_end=True)

  def test_print_first_step(self):
    # if it runs every iteration, first iteration has None duration.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)
      hook = basic_session_run_hooks.LoggingTensorHook(
          tensors={'foo': t}, every_n_iter=1)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess.run(train_op)
      self.assertRegex(str(self.logged_message), 'foo')
      # in first run, elapsed time is None.
      self.assertEqual(str(self.logged_message).find('sec'), -1)

  def _validate_print_every_n_secs(self, sess, at_end, mock_time):
    t = constant_op.constant(42.0, name='foo')
    train_op = constant_op.constant(3)

    hook = basic_session_run_hooks.LoggingTensorHook(
        tensors=[t.name], every_n_secs=1.0, at_end=at_end)
    hook.begin()
    mon_sess = monitored_session._HookedSession(sess, [hook])
    self.evaluate(variables_lib.global_variables_initializer())

    mon_sess.run(train_op)
    self.assertRegex(str(self.logged_message), t.name)

    # assertNotRegexpMatches is not supported by python 3.1 and later
    self.logged_message = ''
    mon_sess.run(train_op)
    self.assertEqual(str(self.logged_message).find(t.name), -1)
    mock_time.return_value += 1.0

    self.logged_message = ''
    mon_sess.run(train_op)
    self.assertRegex(str(self.logged_message), t.name)

    self.logged_message = ''
    hook.end(sess)
    if at_end:
      self.assertRegex(str(self.logged_message), t.name)
    else:
      # assertNotRegexpMatches is not supported by python 3.1 and later
      self.assertEqual(str(self.logged_message).find(t.name), -1)

  @test.mock.patch.object(time, 'time')
  def test_print_every_n_secs(self, mock_time):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_time.return_value = MOCK_START_TIME
      self._validate_print_every_n_secs(sess, at_end=False, mock_time=mock_time)
      # Verify proper reset.
      self._validate_print_every_n_secs(sess, at_end=False, mock_time=mock_time)

  @test.mock.patch.object(time, 'time')
  def test_print_every_n_secs_and_end(self, mock_time):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_time.return_value = MOCK_START_TIME
      self._validate_print_every_n_secs(sess, at_end=True, mock_time=mock_time)
      # Verify proper reset.
      self._validate_print_every_n_secs(sess, at_end=True, mock_time=mock_time)

  def test_print_formatter(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)
      hook = basic_session_run_hooks.LoggingTensorHook(
          tensors=[t.name], every_n_iter=10,
          formatter=lambda items: 'qqq=%s' % items[t.name])
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess.run(train_op)
      self.assertEqual(self.logged_message[0], 'qqq=42.0')


class CheckpointSaverHookTest(test.TestCase):

  def setUp(self):
    self.model_dir = tempfile.mkdtemp()
    self.graph = ops.Graph()
    with self.graph.as_default():
      self.scaffold = monitored_session.Scaffold()
      self.global_step = training_util.get_or_create_global_step()
      self.train_op = training_util._increment_global_step(1)

  def tearDown(self):
    shutil.rmtree(self.model_dir, ignore_errors=True)

  def test_saves_when_saver_and_scaffold_both_missing(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=1)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_raise_when_saver_and_scaffold_both_present(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, saver=self.scaffold.saver, scaffold=self.scaffold)

  @test_util.run_deprecated_v1
  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=10, save_steps=20)

  @test_util.run_deprecated_v1
  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(self.model_dir)

  def test_save_secs_saves_in_first_step(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_secs_calls_listeners_at_begin_and_end(self):
    with self.graph.as_default():
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_secs=2,
          scaffold=self.scaffold,
          listeners=[listener])
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)  # hook runs here
        mon_sess.run(self.train_op)  # hook won't run here, so it does at end
        hook.end(sess)  # hook runs here
      self.assertEqual({
          'begin': 1,
          'before_save': 2,
          'after_save': 2,
          'end': 1
      }, listener.get_counts())

  def test_listener_with_monitored_session(self):
    with ops.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      global_step = training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=1,
          scaffold=scaffold,
          listeners=[listener])
      with monitored_session.SingularMonitoredSession(
          hooks=[hook],
          scaffold=scaffold,
          checkpoint_dir=self.model_dir) as sess:
        sess.run(train_op)
        sess.run(train_op)
        global_step_val = sess.raw_session().run(global_step)
      listener_counts = listener.get_counts()
    self.assertEqual(2, global_step_val)
    self.assertEqual({
        'begin': 1,
        'before_save': 3,
        'after_save': 3,
        'end': 1
    }, listener_counts)

  def test_listener_stops_training_in_after_save(self):
    with ops.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=1, scaffold=scaffold, listeners=[listener])
      with monitored_session.SingularMonitoredSession(
          hooks=[hook], scaffold=scaffold,
          checkpoint_dir=self.model_dir) as sess:
        sess.run(train_op)
        self.assertFalse(sess.should_stop())
        sess.run(train_op)
        self.assertFalse(sess.should_stop())
        listener.ask_for_stop = True
        sess.run(train_op)
        self.assertTrue(sess.should_stop())

  def test_listener_with_default_saver(self):
    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=1,
          listeners=[listener])
      with monitored_session.SingularMonitoredSession(
          hooks=[hook],
          checkpoint_dir=self.model_dir) as sess:
        sess.run(train_op)
        sess.run(train_op)
        global_step_val = sess.raw_session().run(global_step)
      listener_counts = listener.get_counts()
    self.assertEqual(2, global_step_val)
    self.assertEqual({
        'begin': 1,
        'before_save': 3,
        'after_save': 3,
        'end': 1
    }, listener_counts)

    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      with monitored_session.SingularMonitoredSession(
          checkpoint_dir=self.model_dir) as sess2:
        global_step_saved_val = sess2.run(global_step)
    self.assertEqual(2, global_step_saved_val)

  def test_two_listeners_with_default_saver(self):
    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      listener1 = MockCheckpointSaverListener()
      listener2 = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=1,
          listeners=[listener1, listener2])
      with monitored_session.SingularMonitoredSession(
          hooks=[hook],
          checkpoint_dir=self.model_dir) as sess:
        sess.run(train_op)
        sess.run(train_op)
        global_step_val = sess.raw_session().run(global_step)
      listener1_counts = listener1.get_counts()
      listener2_counts = listener2.get_counts()
    self.assertEqual(2, global_step_val)
    self.assertEqual({
        'begin': 1,
        'before_save': 3,
        'after_save': 3,
        'end': 1
    }, listener1_counts)
    self.assertEqual(listener1_counts, listener2_counts)

    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      with monitored_session.SingularMonitoredSession(
          checkpoint_dir=self.model_dir) as sess2:
        global_step_saved_val = sess2.run(global_step)
    self.assertEqual(2, global_step_saved_val)

  @test.mock.patch.object(time, 'time')
  def test_save_secs_saves_periodically(self, mock_time):
    with self.graph.as_default():
      mock_time.return_value = MOCK_START_TIME
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()

      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])

        mock_time.return_value = MOCK_START_TIME
        mon_sess.run(self.train_op)  # Saved.

        mock_time.return_value = MOCK_START_TIME + 0.5
        mon_sess.run(self.train_op)  # Not saved.

        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        # Simulate 2.5 seconds of sleep.
        mock_time.return_value = MOCK_START_TIME + 2.5
        mon_sess.run(self.train_op)  # Saved.

        mock_time.return_value = MOCK_START_TIME + 2.6
        mon_sess.run(self.train_op)  # Not saved.

        mock_time.return_value = MOCK_START_TIME + 2.7
        mon_sess.run(self.train_op)  # Not saved.

        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        # Simulate 7.5 more seconds of sleep (10 seconds from start.
        mock_time.return_value = MOCK_START_TIME + 10
        mon_sess.run(self.train_op)  # Saved.
        self.assertEqual(6,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  @test.mock.patch.object(time, 'time')
  def test_save_secs_calls_listeners_periodically(self, mock_time):
    with self.graph.as_default():
      mock_time.return_value = MOCK_START_TIME
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_secs=2,
          scaffold=self.scaffold,
          listeners=[listener])
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])

        mock_time.return_value = MOCK_START_TIME + 0.5
        mon_sess.run(self.train_op)  # hook runs here

        mock_time.return_value = MOCK_START_TIME + 0.5
        mon_sess.run(self.train_op)

        mock_time.return_value = MOCK_START_TIME + 3.0
        mon_sess.run(self.train_op)  # hook runs here

        mock_time.return_value = MOCK_START_TIME + 3.5
        mon_sess.run(self.train_op)

        mock_time.return_value = MOCK_START_TIME + 4.0
        mon_sess.run(self.train_op)

        mock_time.return_value = MOCK_START_TIME + 6.5
        mon_sess.run(self.train_op)  # hook runs here

        mock_time.return_value = MOCK_START_TIME + 7.0
        mon_sess.run(self.train_op)  # hook won't run here, so it does at end

        mock_time.return_value = MOCK_START_TIME + 7.5
        hook.end(sess)  # hook runs here
      self.assertEqual({
          'begin': 1,
          'before_save': 4,
          'after_save': 4,
          'end': 1
      }, listener.get_counts())

  def test_save_steps_saves_in_first_step(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_steps_saves_periodically(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_saves_at_end(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        hook.end(sess)
        self.assertEqual(2,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_summary_writer_defs(self):
    fake_summary_writer.FakeSummaryWriter.install()
    writer_cache.FileWriterCache.clear()
    summary_writer = writer_cache.FileWriterCache.get(self.model_dir)

    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        hook.after_create_session(sess, None)
        mon_sess.run(self.train_op)
      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.model_dir,
          expected_added_meta_graphs=[
              meta_graph.create_meta_graph_def(
                  graph_def=self.graph.as_graph_def(add_shapes=True),
                  saver_def=self.scaffold.saver.saver_def)
          ])

    fake_summary_writer.FakeSummaryWriter.uninstall()

  def test_save_checkpoint_before_first_train_step(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [hook])
        sess.run(self.scaffold.init_op)
        hook.after_create_session(sess, None)
        # Verifies that checkpoint is saved at step 0.
        self.assertEqual(0,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        # Verifies that no checkpoint is saved after one training step.
        mon_sess.run(self.train_op)
        self.assertEqual(0,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        # Verifies that checkpoint is saved after save_steps.
        mon_sess.run(self.train_op)
        self.assertEqual(2,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_graph_def(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=1, scaffold=self.scaffold,
          save_graph_def=True)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        sess.run(self.scaffold.init_op)
        hook.after_create_session(sess, None)

        self.assertIn('graph.pbtxt', os.listdir(self.model_dir))
        # Should have a single .meta file for step 0
        self.assertLen(gfile.Glob(os.path.join(self.model_dir, '*.meta')), 1)

        mon_sess.run(self.train_op)
        self.assertLen(gfile.Glob(os.path.join(self.model_dir, '*.meta')), 2)

  def test_save_graph_def_false(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=1, scaffold=self.scaffold,
          save_graph_def=False)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        sess.run(self.scaffold.init_op)
        hook.after_create_session(sess, None)

        self.assertNotIn('graph.pbtxt', os.listdir(self.model_dir))
        # Should have a single .meta file for step 0
        self.assertEmpty(gfile.Glob(os.path.join(self.model_dir, '*.meta')))

        mon_sess.run(self.train_op)
        self.assertEmpty(gfile.Glob(os.path.join(self.model_dir, '*.meta')))




class CheckpointSaverHookMultiStepTest(test.TestCase):

  def setUp(self):
    self.model_dir = tempfile.mkdtemp()
    self.graph = ops.Graph()
    self.steps_per_run = 5
    with self.graph.as_default():
      self.scaffold = monitored_session.Scaffold()
      self.global_step = training_util.get_or_create_global_step()
      self.train_op = training_util._increment_global_step(self.steps_per_run)

  def tearDown(self):
    shutil.rmtree(self.model_dir, ignore_errors=True)

  def test_save_steps_saves_in_first_step(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=2*self.steps_per_run,
          scaffold=self.scaffold)
      hook._set_steps_per_run(self.steps_per_run)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_steps_saves_periodically(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=2*self.steps_per_run,
          scaffold=self.scaffold)
      hook._set_steps_per_run(self.steps_per_run)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        # Saved (step=5)
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        mon_sess.run(self.train_op)
        # Not saved (step=10)
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        mon_sess.run(self.train_op)
        # Saved (step=15)
        self.assertEqual(15,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        mon_sess.run(self.train_op)
        # Not saved (step=20)
        self.assertEqual(15,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        mon_sess.run(self.train_op)
        # Saved (step=25)
        self.assertEqual(25,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_steps_saves_at_end(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=2*self.steps_per_run,
          scaffold=self.scaffold)
      hook._set_steps_per_run(self.steps_per_run)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        hook.end(sess)
        self.assertEqual(10,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))


class ResourceCheckpointSaverHookTest(test.TestCase):

  def setUp(self):
    self.model_dir = tempfile.mkdtemp()
    self.graph = ops.Graph()
    with self.graph.as_default():
      self.scaffold = monitored_session.Scaffold()
      with variable_scope.variable_scope('foo', use_resource=True):
        self.global_step = training_util.get_or_create_global_step()
      self.train_op = training_util._increment_global_step(1)

  def test_save_steps_saves_periodically(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))


class StepCounterHookTest(test.TestCase):

  def setUp(self):
    self.log_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.log_dir, ignore_errors=True)

  @test.mock.patch.object(time, 'time')
  def test_step_counter_every_n_steps(self, mock_time):
    mock_time.return_value = MOCK_START_TIME
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir, g)
      hook = basic_session_run_hooks.StepCounterHook(
          summary_writer=summary_writer, every_n_steps=10)
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      with test.mock.patch.object(tf_logging, 'warning') as mock_log:
        for _ in range(30):
          mock_time.return_value += 0.01
          mon_sess.run(train_op)
        # logging.warning should not be called.
        self.assertIsNone(mock_log.call_args)
      hook.end(sess)
      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertItemsEqual([11, 21], summary_writer.summaries.keys())
      for step in [11, 21]:
        summary_value = summary_writer.summaries[step][0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)

  @test.mock.patch.object(time, 'time')
  def test_step_counter_every_n_secs(self, mock_time):
    mock_time.return_value = MOCK_START_TIME
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir, g)
      hook = basic_session_run_hooks.StepCounterHook(
          summary_writer=summary_writer, every_n_steps=None, every_n_secs=0.1)

      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      mon_sess.run(train_op)
      mock_time.return_value += 0.2
      mon_sess.run(train_op)
      mock_time.return_value += 0.2
      mon_sess.run(train_op)
      hook.end(sess)

      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertTrue(summary_writer.summaries, 'No summaries were created.')
      self.assertItemsEqual([2, 3], summary_writer.summaries.keys())
      for summary in summary_writer.summaries.values():
        summary_value = summary[0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)

  def test_global_step_name(self):
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      with variable_scope.variable_scope('bar'):
        variable_scope.get_variable(
            'foo',
            initializer=0,
            trainable=False,
            collections=[
                ops.GraphKeys.GLOBAL_STEP, ops.GraphKeys.GLOBAL_VARIABLES
            ])
      train_op = training_util._increment_global_step(1)
      summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir, g)
      hook = basic_session_run_hooks.StepCounterHook(
          summary_writer=summary_writer, every_n_steps=1, every_n_secs=None)

      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      mon_sess.run(train_op)
      mon_sess.run(train_op)
      hook.end(sess)

      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertTrue(summary_writer.summaries, 'No summaries were created.')
      self.assertItemsEqual([2], summary_writer.summaries.keys())
      summary_value = summary_writer.summaries[2][0].value[0]
      self.assertEqual('bar/foo/sec', summary_value.tag)

  def test_log_warning_if_global_step_not_increased(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(0)  # keep same.
      self.evaluate(variables_lib.global_variables_initializer())
      hook = basic_session_run_hooks.StepCounterHook(
          every_n_steps=1, every_n_secs=None)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      mon_sess.run(train_op)  # Run one step to record global step.
      with test.mock.patch.object(tf_logging, 'log_first_n') as mock_log:
        for _ in range(30):
          mon_sess.run(train_op)
        self.assertRegex(
            str(mock_log.call_args), 'global step.*has not been increased')
      hook.end(sess)

  def _setup_steps_per_run_test(self,
                                every_n_steps,
                                steps_per_run,
                                graph,
                                sess):
    training_util.get_or_create_global_step()
    self.train_op = training_util._increment_global_step(steps_per_run)
    self.summary_writer = fake_summary_writer.FakeSummaryWriter(
        self.log_dir, graph)
    self.hook = basic_session_run_hooks.StepCounterHook(
        summary_writer=self.summary_writer, every_n_steps=every_n_steps)
    self.hook._set_steps_per_run(steps_per_run)
    self.hook.begin()
    self.evaluate(variables_lib.global_variables_initializer())
    self.mon_sess = monitored_session._HookedSession(sess, [self.hook])

  @test.mock.patch.object(time, 'time')
  def test_steps_per_run_less_than_every_n_steps(self, mock_time):
    mock_time.return_value = MOCK_START_TIME
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      self._setup_steps_per_run_test(10, 5, g, sess)

      # Logs at 15, 25
      for _ in range(5):
        mock_time.return_value += 0.01
        self.mon_sess.run(self.train_op)

      self.hook.end(sess)
      self.summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertItemsEqual([15, 25], self.summary_writer.summaries.keys())
      for step in [15, 25]:
        summary_value = self.summary_writer.summaries[step][0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)

  @test.mock.patch.object(time, 'time')
  def test_steps_per_run_equal_every_n_steps(self, mock_time):
    mock_time.return_value = MOCK_START_TIME
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      self._setup_steps_per_run_test(5, 5, g, sess)

      # Logs at 10, 15, 20, 25
      for _ in range(5):
        mock_time.return_value += 0.01
        self.mon_sess.run(self.train_op)

      self.hook.end(sess)
      self.summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertItemsEqual([10, 15, 20, 25],
                            self.summary_writer.summaries.keys())
      for step in [10, 15, 20, 25]:
        summary_value = self.summary_writer.summaries[step][0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)

  @test.mock.patch.object(time, 'time')
  def test_steps_per_run_greater_than_every_n_steps(self, mock_time):
    mock_time.return_value = MOCK_START_TIME
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      self._setup_steps_per_run_test(5, 10, g, sess)

      # Logs at 20, 30, 40, 50
      for _ in range(5):
        mock_time.return_value += 0.01
        self.mon_sess.run(self.train_op)

      self.hook.end(sess)
      self.summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertItemsEqual([20, 30, 40, 50],
                            self.summary_writer.summaries.keys())
      for step in [20, 30, 40, 50]:
        summary_value = self.summary_writer.summaries[step][0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)


@test_util.run_deprecated_v1
class SummarySaverHookTest(test.TestCase):

  def setUp(self):
    test.TestCase.setUp(self)

    self.log_dir = 'log/dir'
    self.summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir)

    var = variables_lib.Variable(0.0)
    tensor = state_ops.assign_add(var, 1.0)
    tensor2 = tensor * 2
    self.summary_op = summary_lib.scalar('my_summary', tensor)
    self.summary_op2 = summary_lib.scalar('my_summary2', tensor2)

    training_util.get_or_create_global_step()
    self.train_op = training_util._increment_global_step(1)

  def test_raise_when_scaffold_and_summary_op_both_missing(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook()

  def test_raise_when_scaffold_and_summary_op_both_present(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook(
          scaffold=monitored_session.Scaffold(), summary_op=self.summary_op)

  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook(
          save_secs=10, save_steps=20, summary_writer=self.summary_writer)

  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook(
          save_secs=None, save_steps=None, summary_writer=self.summary_writer)

  def test_save_steps(self):
    hook = basic_session_run_hooks.SummarySaverHook(
        save_steps=8,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.cached_session() as sess:
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(30):
        mon_sess.run(self.train_op)
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            9: {
                'my_summary': 2.0
            },
            17: {
                'my_summary': 3.0
            },
            25: {
                'my_summary': 4.0
            },
        })

  def test_multiple_summaries(self):
    hook = basic_session_run_hooks.SummarySaverHook(
        save_steps=8,
        summary_writer=self.summary_writer,
        summary_op=[self.summary_op, self.summary_op2])

    with self.cached_session() as sess:
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(10):
        mon_sess.run(self.train_op)
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0,
                'my_summary2': 2.0
            },
            9: {
                'my_summary': 2.0,
                'my_summary2': 4.0
            },
        })

  @test.mock.patch.object(time, 'time')
  def test_save_secs_saving_once_every_step(self, mock_time):
    mock_time.return_value = MOCK_START_TIME
    hook = basic_session_run_hooks.SummarySaverHook(
        save_secs=0.5,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.cached_session() as sess:
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(4):
        mon_sess.run(self.train_op)
        mock_time.return_value += 0.5
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            2: {
                'my_summary': 2.0
            },
            3: {
                'my_summary': 3.0
            },
            4: {
                'my_summary': 4.0
            },
        })

  @test.mock.patch.object(time, 'time')
  def test_save_secs_saving_once_every_three_steps(self, mock_time):
    mock_time.return_value = 1484695987.209386
    hook = basic_session_run_hooks.SummarySaverHook(
        save_secs=9.,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.cached_session() as sess:
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(8):
        mon_sess.run(self.train_op)
        mock_time.return_value += 3.1
      hook.end(sess)

    # 24.8 seconds passed (3.1*8), it saves every 9 seconds starting from first:
    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            4: {
                'my_summary': 2.0
            },
            7: {
                'my_summary': 3.0
            },
        })


class GlobalStepWaiterHookTest(test.TestCase):

  def test_not_wait_for_step_zero(self):
    with ops.Graph().as_default():
      training_util.get_or_create_global_step()
      hook = basic_session_run_hooks.GlobalStepWaiterHook(wait_until_step=0)
      hook.begin()
      with session_lib.Session() as sess:
        # Before run should return without waiting gstep increment.
        hook.before_run(
            session_run_hook.SessionRunContext(
                original_args=None, session=sess))

  @test.mock.patch.object(time, 'sleep')
  def test_wait_for_step(self, mock_sleep):
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      hook = basic_session_run_hooks.GlobalStepWaiterHook(wait_until_step=1000)
      hook.begin()

      with session_lib.Session() as sess:
        # Mock out calls to time.sleep() to update the global step.

        class Context(object):
          counter = 0

        def mock_sleep_side_effect(seconds):
          del seconds  # argument is ignored
          Context.counter += 1
          if Context.counter == 1:
            # The first time sleep() is called, we update the global_step from
            # 0 to 500.
            sess.run(state_ops.assign(gstep, 500))
          elif Context.counter == 2:
            # The second time sleep() is called, we update the global_step from
            # 500 to 1100.
            sess.run(state_ops.assign(gstep, 1100))
          else:
            raise AssertionError(
                'Expected before_run() to terminate after the second call to '
                'time.sleep()')

        mock_sleep.side_effect = mock_sleep_side_effect

        # Run the mocked-out interaction with the hook.
        self.evaluate(variables_lib.global_variables_initializer())
        run_context = session_run_hook.SessionRunContext(
            original_args=None, session=sess)
        hook.before_run(run_context)
        self.assertEqual(Context.counter, 2)


class FinalOpsHookTest(test.TestCase):

  def test_final_ops_is_scalar_tensor(self):
    with ops.Graph().as_default():
      expected_value = 4
      final_ops = constant_op.constant(expected_value)

      hook = basic_session_run_hooks.FinalOpsHook(final_ops)
      hook.begin()

      with session_lib.Session() as session:
        hook.end(session)
        self.assertEqual(expected_value,
                         hook.final_ops_values)

  def test_final_ops_is_tensor(self):
    with ops.Graph().as_default():
      expected_values = [1, 6, 3, 5, 2, 4]
      final_ops = constant_op.constant(expected_values)

      hook = basic_session_run_hooks.FinalOpsHook(final_ops)
      hook.begin()

      with session_lib.Session() as session:
        hook.end(session)
        self.assertListEqual(expected_values,
                             hook.final_ops_values.tolist())

  def test_final_ops_triggers_out_of_range_error(self):
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.range(1)
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      read_ops = iterator.get_next()
      final_ops = read_ops

      hook = basic_session_run_hooks.FinalOpsHook(final_ops)
      hook.begin()

      with session_lib.Session() as session:
        session.run(read_ops)
        with test.mock.patch.object(tf_logging, 'warning') as mock_log:
          with self.assertRaisesRegex(errors.OutOfRangeError,
                                      'End of sequence'):
            hook.end(session)
          self.assertRegex(
              str(mock_log.call_args), 'dependency back to some input source')

  def test_final_ops_with_dictionary(self):
    with ops.Graph().as_default():
      expected_values = [4, -3]
      final_ops = array_ops.placeholder(dtype=dtypes.float32)
      final_ops_feed_dict = {final_ops: expected_values}

      hook = basic_session_run_hooks.FinalOpsHook(
          final_ops, final_ops_feed_dict)
      hook.begin()

      with session_lib.Session() as session:
        hook.end(session)
        self.assertListEqual(expected_values,
                             hook.final_ops_values.tolist())


@test_util.run_deprecated_v1
class ResourceSummarySaverHookTest(test.TestCase):

  def setUp(self):
    test.TestCase.setUp(self)

    self.log_dir = 'log/dir'
    self.summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir)

    var = variable_scope.get_variable('var', initializer=0.0, use_resource=True)
    tensor = state_ops.assign_add(var, 1.0)
    self.summary_op = summary_lib.scalar('my_summary', tensor)

    with variable_scope.variable_scope('foo', use_resource=True):
      training_util.create_global_step()
    self.train_op = training_util._increment_global_step(1)

  def test_save_steps(self):
    hook = basic_session_run_hooks.SummarySaverHook(
        save_steps=8,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.cached_session() as sess:
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(30):
        mon_sess.run(self.train_op)
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            9: {
                'my_summary': 2.0
            },
            17: {
                'my_summary': 3.0
            },
            25: {
                'my_summary': 4.0
            },
        })


class FeedFnHookTest(test.TestCase):

  def test_feeding_placeholder(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      x = array_ops.placeholder(dtype=dtypes.float32)
      y = x + 1
      hook = basic_session_run_hooks.FeedFnHook(
          feed_fn=lambda: {x: 1.0})
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.assertEqual(mon_sess.run(y), 2)


class ProfilerHookTest(test.TestCase):

  def setUp(self):
    super(ProfilerHookTest, self).setUp()
    self.output_dir = tempfile.mkdtemp()
    self.graph = ops.Graph()
    self.filepattern = os.path.join(self.output_dir, 'timeline-*.json')
    with self.graph.as_default():
      self.global_step = training_util.get_or_create_global_step()
      self.train_op = state_ops.assign_add(self.global_step, 1)

  def tearDown(self):
    super(ProfilerHookTest, self).tearDown()
    shutil.rmtree(self.output_dir, ignore_errors=True)

  def _count_timeline_files(self):
    return len(gfile.Glob(self.filepattern))

  @test_util.run_deprecated_v1
  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.ProfilerHook(save_secs=10, save_steps=20)

  @test_util.run_deprecated_v1
  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      basic_session_run_hooks.ProfilerHook(save_secs=None, save_steps=None)

  def test_save_secs_does_not_save_in_first_step(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.ProfilerHook(
          save_secs=2, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)
        self.assertEqual(0, self._count_timeline_files())

  @test.mock.patch.object(time, 'time')
  def test_save_secs_saves_periodically(self, mock_time):
    # Pick a fixed start time.
    with self.graph.as_default():
      mock_time.return_value = MOCK_START_TIME
      hook = basic_session_run_hooks.ProfilerHook(
          save_secs=2, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(0, self._count_timeline_files())
        # Simulate 2.5 seconds of sleep.
        mock_time.return_value = MOCK_START_TIME + 2.5
        sess.run(self.train_op)  # Saved.
        self.assertEqual(1, self._count_timeline_files())

        # Pretend some small amount of time has passed.
        mock_time.return_value = MOCK_START_TIME + 2.6
        sess.run(self.train_op)  # Not saved.
        # Edge test just before we should save the timeline.
        mock_time.return_value = MOCK_START_TIME + 4.4
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(1, self._count_timeline_files())

        mock_time.return_value = MOCK_START_TIME + 4.5
        sess.run(self.train_op)  # Saved.
        self.assertEqual(2, self._count_timeline_files())

  def test_save_steps_does_not_save_in_first_step(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.ProfilerHook(
          save_steps=1, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(0, self._count_timeline_files())

  def test_save_steps_saves_periodically(self):
    with self.graph.as_default():
      hook = basic_session_run_hooks.ProfilerHook(
          save_steps=2, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        self.assertEqual(0, self._count_timeline_files())
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(0, self._count_timeline_files())
        sess.run(self.train_op)  # Saved.
        self.assertEqual(1, self._count_timeline_files())
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(1, self._count_timeline_files())
        sess.run(self.train_op)  # Saved.
        self.assertEqual(2, self._count_timeline_files())
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(2, self._count_timeline_files())

  def test_run_metadata_saves(self):
    writer_cache.FileWriterCache.clear()
    fake_summary_writer.FakeSummaryWriter.install()
    fake_writer = writer_cache.FileWriterCache.get(self.output_dir)
    with self.graph.as_default():
      hook = basic_session_run_hooks.ProfilerHook(
          save_steps=1, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)  # Not saved.
        sess.run(self.train_op)  # Saved.
        self.assertEqual(
            list(fake_writer._added_run_metadata.keys()), ['step_2'])
    fake_summary_writer.FakeSummaryWriter.uninstall()


if __name__ == '__main__':
  test.main()
