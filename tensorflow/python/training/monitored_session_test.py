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
"""Tests for monitored_session."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import glob
import os
import threading
import time

from tensorflow.contrib.framework.python.ops import variables as variables_lib
from tensorflow.contrib.testing.python.framework import util_test
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import debug_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook


class ScaffoldTest(test.TestCase):
  """Scaffold tests."""

  def test_nothing_created_before_finalize(self):
    with ops.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      self.assertEqual(None, scaffold.init_op)
      self.assertEqual(None, scaffold.init_feed_dict)
      self.assertEqual(None, scaffold.init_fn)
      self.assertEqual(None, scaffold.ready_op)
      self.assertEqual(None, scaffold.ready_for_local_init_op)
      self.assertEqual(None, scaffold.local_init_op)
      self.assertEqual(None, scaffold.saver)

  def test_defaults_empty_graph(self):
    with ops.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      variables.Variable(1, name='my_var')
      variables.Variable(
          2, name='my_local_var', collections=[ops.GraphKeys.LOCAL_VARIABLES])
      scaffold.finalize()
      self.assertTrue(isinstance(scaffold.init_op, ops.Operation))
      self.assertEqual(None, scaffold.init_feed_dict)
      self.assertEqual(None, scaffold.init_fn)
      self.assertTrue(isinstance(scaffold.ready_op, ops.Tensor))
      self.assertTrue(isinstance(scaffold.ready_for_local_init_op, ops.Tensor))
      self.assertTrue(isinstance(scaffold.local_init_op, ops.Operation))
      self.assertTrue(isinstance(scaffold.saver, saver_lib.Saver))
      with self.test_session() as sess:
        self.assertItemsEqual([b'my_var', b'my_local_var'],
                              sess.run(scaffold.ready_op))
        self.assertItemsEqual([b'my_var'],
                              sess.run(scaffold.ready_for_local_init_op))
        sess.run(scaffold.init_op)
        self.assertEqual(0, len(sess.run(scaffold.ready_for_local_init_op)))
        sess.run(scaffold.local_init_op)
        self.assertEqual(0, len(sess.run(scaffold.ready_op)))

  def test_defaults_no_variables(self):
    with ops.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      constant_op.constant(1, name='my_const')
      scaffold.finalize()
      self.assertTrue(isinstance(scaffold.init_op, ops.Operation))
      self.assertEqual(None, scaffold.init_feed_dict)
      self.assertEqual(None, scaffold.init_fn)
      self.assertTrue(isinstance(scaffold.ready_op, ops.Tensor))
      self.assertTrue(isinstance(scaffold.ready_for_local_init_op, ops.Tensor))
      self.assertTrue(isinstance(scaffold.local_init_op, ops.Operation))
      self.assertTrue(isinstance(scaffold.saver, saver_lib.Saver))

  def test_caches_values(self):
    with ops.Graph().as_default():
      variables.Variable([1])
      scaffold1 = monitored_session.Scaffold()
      scaffold1.finalize()
      scaffold2 = monitored_session.Scaffold()
      scaffold2.finalize()
      self.assertEqual(scaffold1.init_op, scaffold2.init_op)
      self.assertEqual(scaffold1.ready_op, scaffold2.ready_op)
      self.assertEqual(scaffold1.ready_for_local_init_op,
                       scaffold2.ready_for_local_init_op)
      self.assertEqual(scaffold1.local_init_op, scaffold2.local_init_op)
      self.assertEqual(scaffold1.saver, scaffold2.saver)

  def test_raise_error_if_more_than_one_cached_item(self):
    with ops.Graph().as_default():
      variables.Variable([1])
      ops.add_to_collection(ops.GraphKeys.SAVERS, saver_lib.Saver())
      ops.add_to_collection(ops.GraphKeys.SAVERS, saver_lib.Saver())
      with self.assertRaisesRegexp(RuntimeError, 'More than one item'):
        monitored_session.Scaffold().finalize()

  def test_uses_passed_values(self):
    with ops.Graph().as_default():
      variables.Variable([1])
      saver = saver_lib.Saver()
      scaffold = monitored_session.Scaffold(
          init_op=2,
          init_feed_dict=3,
          init_fn=lambda scaffold, sess: 4,
          ready_op=5,
          ready_for_local_init_op=6,
          local_init_op=7,
          saver=saver)
      scaffold.finalize()
      self.assertEqual(2, scaffold.init_op)
      self.assertEqual(3, scaffold.init_feed_dict)
      self.assertTrue(callable(scaffold.init_fn))
      self.assertEqual(5, scaffold.ready_op)
      self.assertEqual(6, scaffold.ready_for_local_init_op)
      self.assertEqual(7, scaffold.local_init_op)
      self.assertEqual(saver, scaffold.saver)

  def test_graph_is_finalized(self):
    with ops.Graph().as_default():
      variables.Variable([1])
      monitored_session.Scaffold().finalize()
      with self.assertRaisesRegexp(RuntimeError,
                                   'Graph is finalized and cannot be modified'):
        constant_op.constant([0])


def _test_dir(temp_dir, test_name):
  """Create an empty dir to use for tests.

  Args:
    temp_dir: Tmp directory path.
    test_name: Name of the test.

  Returns:
    Absolute path to the test directory.
  """
  test_dir = os.path.join(temp_dir, test_name)
  if os.path.isdir(test_dir):
    for f in glob.glob('%s/*' % test_dir):
      os.remove(f)
  else:
    os.makedirs(test_dir)
  return test_dir


class FakeHook(session_run_hook.SessionRunHook):

  def __init__(self):
    self.should_stop = False
    self.request = None
    self.call_counter = collections.Counter()
    self.last_run_context = None
    self.last_run_values = None

  def begin(self):
    self.call_counter['begin'] += 1

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    self.call_counter['after_create_session'] += 1

  def before_run(self, run_context):
    self.call_counter['before_run'] += 1
    self.last_run_context = run_context
    return self.request

  def after_run(self, run_context, run_values):
    self.call_counter['after_run'] += 1
    self.last_run_values = run_values
    if self.should_stop:
      run_context.request_stop()

  def end(self, session):
    self.call_counter['end'] += 1


class MonitoredTrainingSessionTest(test.TestCase):
  """Tests MonitoredTrainingSession."""

  def test_saving_restoring_checkpoint(self):
    logdir = _test_dir(self.get_temp_dir(), 'test_saving_restoring_checkpoint')
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      with monitored_session.MonitoredTrainingSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
      # A restart will find the checkpoint and recover automatically.
      with monitored_session.MonitoredTrainingSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(2, session.run(gstep))

  def test_summaries_steps(self):
    logdir = _test_dir(self.get_temp_dir(), 'test_summaries_steps')
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      summary.scalar('my_summary_tag', new_gstep * 2)
      with monitored_session.MonitoredTrainingSession(
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
    logdir = _test_dir(self.get_temp_dir(), 'test_summaries_secs')
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      summary.scalar('my_summary_tag', new_gstep * 2)
      with monitored_session.MonitoredTrainingSession(
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
    logdir = _test_dir(self.get_temp_dir(), 'test_saving_restoring_checkpoint')
    fake_hook = FakeHook()
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      with monitored_session.MonitoredTrainingSession(
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
      with monitored_session.MonitoredTrainingSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(0, session.run(gstep))


class StopAtNSession(monitored_session._WrappedSession):
  """A wrapped session that stops at the N-th call to _check_stop."""

  def __init__(self, sess, n):
    super(StopAtNSession, self).__init__(sess)
    self._count = n

  def _check_stop(self):
    if self._count == 0:
      return True
    self._count -= 1
    return False


class WrappedSessionTest(test.TestCase):
  """_WrappedSession tests."""

  def test_properties(self):
    with self.test_session() as sess:
      constant_op.constant(0.0)
      wrapped_sess = monitored_session._WrappedSession(sess)
      self.assertEquals(sess.graph, wrapped_sess.graph)
      self.assertEquals(sess.sess_str, wrapped_sess.sess_str)

  def test_should_stop_on_close(self):
    with self.test_session() as sess:
      wrapped_sess = monitored_session._WrappedSession(sess)
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
      wrapped_sess1 = monitored_session._WrappedSession(wrapped_sess0)
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertTrue(wrapped_sess1.should_stop())

  def test_close_twice(self):
    with self.test_session() as sess:
      wrapped_sess = monitored_session._WrappedSession(sess)
      wrapped_sess.close()
      self.assertTrue(wrapped_sess.should_stop())
      wrapped_sess.close()
      self.assertTrue(wrapped_sess.should_stop())

  def test_run(self):
    with self.test_session() as sess:
      c = constant_op.constant(0)
      v = array_ops.identity(c)
      self.assertEqual(42, sess.run(v, feed_dict={c: 42}))
      wrapped_sess = monitored_session._WrappedSession(sess)
      self.assertEqual(51, wrapped_sess.run(v, feed_dict={c: 51}))


def busy_wait_for_coord_stop(coord):
  while not coord.should_stop():
    time.sleep(0.001)


class CoordinatedSessionTest(test.TestCase):
  """_CoordinatedSession tests."""

  def test_properties(self):
    with self.test_session() as sess:
      constant_op.constant(0.0)
      coord = coordinator.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertEquals(sess.graph, coord_sess.graph)
      self.assertEquals(sess.sess_str, coord_sess.sess_str)

  def test_run(self):
    with self.test_session() as sess:
      c = constant_op.constant(0)
      v = array_ops.identity(c)
      coord = coordinator.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertEqual(42, coord_sess.run(v, feed_dict={c: 42}))

  def test_should_stop_on_close(self):
    with self.test_session() as sess:
      coord = coordinator.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertFalse(coord_sess.should_stop())
      coord_sess.close()
      self.assertTrue(coord_sess.should_stop())

  def test_should_stop_on_coord_stop(self):
    with self.test_session() as sess:
      coord = coordinator.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertFalse(coord_sess.should_stop())
      coord.request_stop()
      self.assertTrue(coord_sess.should_stop())

  def test_dont_request_stop_on_exception_in_main_thread(self):
    with self.test_session() as sess:
      c = constant_op.constant(0)
      v = array_ops.identity(c)
      coord = coordinator.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertFalse(coord_sess.should_stop())
      self.assertEqual(0, coord_sess.run(c))
      self.assertEqual(1, coord_sess.run(v, feed_dict={c: 1}))
      with self.assertRaisesRegexp(TypeError, 'None has invalid type'):
        coord_sess.run([None], feed_dict={c: 2})
      self.assertFalse(coord.should_stop())
      self.assertFalse(coord_sess.should_stop())

  def test_stop_threads_on_close_after_exception(self):
    with self.test_session() as sess:
      c = constant_op.constant(0)
      v = array_ops.identity(c)
      coord = coordinator.Coordinator()
      threads = [
          threading.Thread(
              target=busy_wait_for_coord_stop, args=(coord,)) for _ in range(3)
      ]
      for t in threads:
        coord.register_thread(t)
        t.start()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
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
      coord_sess.close()
      for t in threads:
        self.assertFalse(t.is_alive())
      self.assertTrue(coord.should_stop())
      self.assertTrue(coord_sess.should_stop())

  def test_stop_threads_on_close(self):
    with self.test_session() as sess:
      coord = coordinator.Coordinator()
      threads = [
          threading.Thread(
              target=busy_wait_for_coord_stop, args=(coord,)) for _ in range(3)
      ]
      for t in threads:
        coord.register_thread(t)
        t.start()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      coord_sess.close()
      for t in threads:
        self.assertFalse(t.is_alive())
      self.assertTrue(coord.should_stop())
      self.assertTrue(coord_sess.should_stop())


class AbortAtNSession(object):
  """A mock sessionthat aborts at the N-th run call."""

  def __init__(self, sess, n):
    self._sess = sess
    self._count = n

  def close(self):
    pass

  def run(self, *args, **kwargs):
    if self._count == 0:
      raise errors_impl.AbortedError('Aborted at N', None, None)
    self._count -= 1
    return self._sess.run(*args, **kwargs)


class RecoverableSessionTest(test.TestCase):
  """_RecoverableSession tests."""

  class _SessionReturner(object):

    def __init__(self, sess):
      self._sess = sess

    def create_session(self):
      return self._sess

  def test_properties(self):
    with self.test_session() as sess:
      constant_op.constant(0.0)
      recoverable_sess = monitored_session._RecoverableSession(
          self._SessionReturner(sess))
      self.assertEquals(sess.graph, recoverable_sess.graph)
      self.assertEquals(sess.sess_str, recoverable_sess.sess_str)

  def test_run(self):
    with self.test_session() as sess:
      c = constant_op.constant(0)
      v = array_ops.identity(c)
      recoverable_sess = monitored_session._RecoverableSession(
          self._SessionReturner(sess))
      self.assertEqual(51, recoverable_sess.run(v, feed_dict={c: 51}))

  def test_recovery(self):
    with self.test_session() as sess:

      class StackSessionCreator(object):

        def __init__(self, sess):
          self.sessions_to_use = [
              AbortAtNSession(sess, x + 1) for x in range(3)
          ]

        def create_session(self):
          return self.sessions_to_use.pop(0)

      c = constant_op.constant(0)
      v = array_ops.identity(c)
      session_creator = StackSessionCreator(sess)
      # List of 3 sessions to use for recovery.  The first one aborts
      # after 1 run() call, the second after 2 run calls, the third
      # after 3 run calls.
      self.assertEqual(3, len(session_creator.sessions_to_use))
      # Make the recoverable session uses these 3 sessions in sequence by
      # passing a factory that pops from the session_to_use list.
      recoverable_sess = monitored_session._RecoverableSession(session_creator)
      self.assertEqual(
          2, len(session_creator.sessions_to_use))  # One session popped.
      # Using first session.
      self.assertEqual(51, recoverable_sess.run(v, feed_dict={c: 51}))
      self.assertEqual(
          2, len(session_creator.sessions_to_use))  # Still 2 sessions available
      # This will fail and recover by picking up the second session.
      self.assertEqual(42, recoverable_sess.run(v, feed_dict={c: 42}))
      self.assertEqual(
          1, len(session_creator.sessions_to_use))  # Still 1 session available
      self.assertEqual(33, recoverable_sess.run(v, feed_dict={c: 33}))
      self.assertEqual(
          1, len(session_creator.sessions_to_use))  # Still 1 session available
      # This will fail and recover by picking up the last session.
      self.assertEqual(24, recoverable_sess.run(v, feed_dict={c: 24}))
      self.assertEqual(
          0, len(session_creator.sessions_to_use))  # All sessions used.
      self.assertEqual(11, recoverable_sess.run(v, feed_dict={c: 11}))
      self.assertEqual(0, recoverable_sess.run(v, feed_dict={c: 0}))
      # This will fail and throw a real error as the pop() will fail.
      with self.assertRaisesRegexp(IndexError, 'pop from empty list'):
        recoverable_sess.run(v, feed_dict={c: -12})


class FakeSession(monitored_session._WrappedSession):

  def __init__(self, sess):
    monitored_session._WrappedSession.__init__(self, sess)
    self.args_called = {}

  def run(self, fetches, **kwargs):
    self.args_called = dict(kwargs)
    # Call run only with fetches since we directly pass other arguments.
    return monitored_session._WrappedSession.run(self, fetches)


class HookedSessionTest(test.TestCase):
  """Tests of _HookedSession."""

  def testRunPassesAllArguments(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_run = FakeSession(sess)
      mon_sess = monitored_session._HookedSession(sess=mock_run, hooks=[])
      a_tensor = constant_op.constant([0], name='a_tensor')
      sess.run(variables.global_variables_initializer())
      output = mon_sess.run(fetches=a_tensor,
                            feed_dict='a_feed',
                            options='an_option',
                            run_metadata='a_metadata')
      self.assertEqual(output, [0])
      self.assertEqual(mock_run.args_called, {
          'feed_dict': 'a_feed',
          'options': 'an_option',
          'run_metadata': 'a_metadata'
      })

  def testCallsHooksBeginEnd(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      sess.run(variables.global_variables_initializer())
      mon_sess.run(a_tensor)

      for hook in [mock_hook, mock_hook2]:
        self.assertEqual(
            hook.last_run_values,
            session_run_hook.SessionRunValues(
                results=None,
                options=config_pb2.RunOptions(),
                run_metadata=config_pb2.RunMetadata()))
        self.assertEqual(hook.last_run_context.original_args,
                         session_run_hook.SessionRunArgs(a_tensor))
        self.assertEqual(hook.last_run_context.session, sess)
        self.assertEqual(hook.call_counter['begin'], 0)
        self.assertEqual(hook.call_counter['after_create_session'], 0)
        self.assertEqual(hook.call_counter['before_run'], 1)
        self.assertEqual(hook.call_counter['after_run'], 1)

  def testShouldStop(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      constant_op.constant([0], name='a_tensor')
      sess.run(variables.global_variables_initializer())

      mon_sess.run(fetches='a_tensor')
      self.assertFalse(mon_sess.should_stop())

      mock_hook.should_stop = True
      mon_sess.run(fetches='a_tensor')
      self.assertTrue(mon_sess.should_stop())

  def testFetchesHookRequests(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      another_tensor = constant_op.constant([5], name='another_tensor')
      third_tensor = constant_op.constant([10], name='third_tensor')
      mock_hook.request = session_run_hook.SessionRunArgs([another_tensor])
      mock_hook2.request = session_run_hook.SessionRunArgs([third_tensor])
      sess.run(variables.global_variables_initializer())

      output = mon_sess.run(fetches=a_tensor)
      self.assertEqual(output, [0])
      self.assertEqual(mock_hook.last_run_values.results, [5])
      self.assertEqual(mock_hook2.last_run_values.results, [10])

  def testOnlyHooksHaveFeeds(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      sess.run(variables.global_variables_initializer())

      self.assertEqual(mon_sess.run(fetches=add_tensor), [15])

  def testBothHooksAndUserHaveFeeds(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      c_tensor = constant_op.constant([0], name='c_tensor')
      add_tensor = a_tensor + b_tensor + c_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      sess.run(variables.global_variables_initializer())

      feed_dict = {c_tensor: [20]}
      self.assertEqual(
          mon_sess.run(fetches=add_tensor, feed_dict=feed_dict), [35])
      # User feed_dict should not be changed
      self.assertEqual(len(feed_dict), 1)

  def testHooksFeedConflicts(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [10]})
      sess.run(variables.global_variables_initializer())

      with self.assertRaisesRegexp(RuntimeError, 'Same tensor is fed'):
        mon_sess.run(fetches=add_tensor)

  def testHooksAndUserFeedConflicts(self):
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      sess.run(variables.global_variables_initializer())

      with self.assertRaisesRegexp(RuntimeError, 'Same tensor is fed'):
        mon_sess.run(fetches=add_tensor, feed_dict={b_tensor: [10]})


class RaiseOnceAtCountN(session_run_hook.SessionRunHook):
  """Hook that raises an Exception at step N."""

  def __init__(self, n, ex):
    self.n = n
    self.ex = ex
    self.raised = False

  def before_run(self, run_context):
    # Raise the first time we reach step N.
    self.n -= 1
    if 0 == self.n and not self.raised:
      self.raised = True
      raise self.ex
    return None


class RunOptionsMetadataHook(session_run_hook.SessionRunHook):
  """A hook that observes & optionally modifies RunOptions and RunMetadata."""

  def __init__(self, trace_level, timeout_in_ms, output_partition_graphs,
               debug_tensor_watch):
    self._trace_level = trace_level
    self._timeout_in_ms = timeout_in_ms
    self._output_partition_graphs = output_partition_graphs
    self._debug_tensor_watch = debug_tensor_watch

    self.run_options_list = []
    self.run_metadata_list = []

  def before_run(self, run_context):
    options = config_pb2.RunOptions(
        trace_level=self._trace_level,
        timeout_in_ms=self._timeout_in_ms,
        output_partition_graphs=self._output_partition_graphs)
    options.debug_options.debug_tensor_watch_opts.extend(
        [self._debug_tensor_watch])
    return session_run_hook.SessionRunArgs(None, None, options=options)

  def after_run(self, run_context, run_values):
    self.run_options_list.append(run_values.options)
    self.run_metadata_list.append(run_values.run_metadata)


class MonitoredSessionTest(test.TestCase):
  """MonitoredSession tests."""

  def test_defaults(self):
    with ops.Graph().as_default():
      a_var = variables.Variable(0)
      with monitored_session.MonitoredSession() as session:
        self.assertEqual(0, session.run(a_var))

  def test_last_step(self):
    logdir = _test_dir(self.get_temp_dir(), 'test_last_step')
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      # Run till step 3 and save.
      hooks = [basic_session_run_hooks.StopAtStepHook(last_step=3)]
      scaffold = monitored_session.Scaffold().finalize()
      with monitored_session.MonitoredSession(hooks=hooks) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertFalse(session.should_stop())
        self.assertEqual(1, session.run(do_step))
        self.assertFalse(session.should_stop())
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())
        self.assertEqual(3, session.run(do_step))
        self.assertTrue(session.should_stop())
        save_path = scaffold.saver.save(session._coordinated_creator.tf_sess,
                                        os.path.join(logdir, 'step-3'))
      # Run till step 5 and save.
      def load_ckpt(scaffold, sess):
        scaffold.saver.restore(sess, save_path)

      session_creator = monitored_session.ChiefSessionCreator(
          monitored_session.Scaffold(init_fn=load_ckpt))
      hooks = [basic_session_run_hooks.StopAtStepHook(last_step=5)]
      with monitored_session.MonitoredSession(
          hooks=hooks, session_creator=session_creator) as session:
        self.assertEqual(3, session.run(gstep))
        self.assertFalse(session.should_stop())
        self.assertEqual(4, session.run(do_step))
        self.assertFalse(session.should_stop())
        self.assertEqual(5, session.run(do_step))
        self.assertTrue(session.should_stop())

  def test_num_steps(self):
    logdir = _test_dir(self.get_temp_dir(), 'test_num_steps')
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      # Do 3 steps and save.
      hooks = [basic_session_run_hooks.StopAtStepHook(num_steps=3)]
      scaffold = monitored_session.Scaffold().finalize()
      with monitored_session.MonitoredSession(hooks=hooks) as session:
        session.run(do_step)
        self.assertFalse(session.should_stop())
        session.run(do_step)
        self.assertFalse(session.should_stop())
        session.run(do_step)
        self.assertTrue(session.should_stop())
        save_path = scaffold.saver.save(session._coordinated_creator.tf_sess,
                                        os.path.join(logdir, 'step-3'))
      # Restore and do 4 steps.
      def load_ckpt(scaffold, sess):
        scaffold.saver.restore(sess, save_path)

      session_creator = monitored_session.ChiefSessionCreator(
          scaffold=monitored_session.Scaffold(init_fn=load_ckpt))
      hooks = [basic_session_run_hooks.StopAtStepHook(num_steps=4)]
      with monitored_session.MonitoredSession(
          hooks=hooks, session_creator=session_creator) as session:
        self.assertEqual(4, session.run(do_step))
        self.assertFalse(session.should_stop())
        session.run(do_step)
        self.assertFalse(session.should_stop())
        session.run(do_step)
        self.assertFalse(session.should_stop())
        session.run(do_step)
        self.assertTrue(session.should_stop())

  # This set of tests, verifies the supervised session behavior when exceptions
  # are raised next to the innermost session run() call.

  def test_recovery(self):
    logdir = _test_dir(self.get_temp_dir(), 'test_recovery')
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      scaffold = monitored_session.Scaffold()
      # Use a hook to save the model every 100 steps.  It also saves it at
      # the end.
      hooks = [
          basic_session_run_hooks.CheckpointSaverHook(
              logdir, save_steps=1, scaffold=scaffold)
      ]
      with monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              scaffold, checkpoint_dir=logdir),
          hooks=hooks) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
      # A restart will find the checkpoint and recover automatically.
      with monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              scaffold, checkpoint_dir=logdir)) as session:
        self.assertEqual(2, session.run(gstep))
      # A restart will find the checkpoint and recover automatically.
      with monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              scaffold,
              checkpoint_filename_with_path=saver_lib.latest_checkpoint(
                  logdir))) as session:
        self.assertEqual(2, session.run(gstep))

  def test_retry_initialization_on_aborted_error(self):
    # Tests that we silently retry on abort during initialization.
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      self.init_raised_aborted_error = False

      def _init_fn(scaffold, session):
        _, _ = scaffold, session
        if not self.init_raised_aborted_error:
          self.init_raised_aborted_error = True
          raise errors_impl.AbortedError(None, None, 'Abort')

      with monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              scaffold=monitored_session.Scaffold(
                  init_fn=_init_fn))) as session:
        self.assertFalse(session.should_stop())
        self.assertEqual(0, session.run(gstep))
      self.assertTrue(self.init_raised_aborted_error)

  def _retry_test(self, ex):
    # Tests that we silently retry on error.  Note that this does not test
    # recovery as we do not use a CheckpointSaver in this test.
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(4, ex)
      with monitored_session.MonitoredSession(hooks=[hook]) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())
        # Here at step 3, the hook triggers and raises AbortedError.  The
        # MonitoredSession automatically retries and restart from a freshly
        # initialized session, so the step is back to 0 and running do_step
        # moves it to 1.
        self.assertEqual(1, session.run(do_step))
        self.assertFalse(session.should_stop())
        self.assertTrue(hook.raised)
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())

  def test_retry_on_aborted_error(self):
    self._retry_test(errors_impl.AbortedError(None, None, 'Abort'))

  def test_retry_on_unavailable_error(self):
    self._retry_test(errors_impl.UnavailableError(None, None, 'Unavailable'))

  def test_recover_and_retry_on_aborted_error(self):
    # Tests that we silently retry and recover on abort.  This test uses
    # a CheckpointSaver to have something to recover from.
    logdir = _test_dir(self.get_temp_dir(),
                       'test_recover_and_retry_on_aborted_error')
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      scaffold = monitored_session.Scaffold()
      abort_hook = RaiseOnceAtCountN(
          4, errors_impl.AbortedError(None, None, 'Abort'))
      # Save after each step.
      ckpt_hook = basic_session_run_hooks.CheckpointSaverHook(
          logdir, save_steps=1, scaffold=scaffold)
      hooks = [abort_hook, ckpt_hook]
      with monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              scaffold, checkpoint_dir=logdir),
          hooks=hooks) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())
        # Here at step 3, the hook triggers and raises AbortedError.  The
        # MonitoredSession automatically restores and retries.
        self.assertEqual(3, session.run(do_step))
        self.assertTrue(abort_hook.raised)
        self.assertFalse(session.should_stop())
        self.assertEqual(4, session.run(do_step))
        self.assertFalse(session.should_stop())

  def test_exit_cleanly_on_out_of_range_exception(self):
    # Tests that we stop cleanly when OutOfRange is raised.
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(2, errors_impl.OutOfRangeError(None, None,
                                                              'EOI'))
      session = monitored_session.MonitoredSession(hooks=[hook])
      # session should cleanly exit from the context.
      with session:
        self.assertEqual(0, session.run(gstep))
        self.assertFalse(session.should_stop())
        # Here at step 1, the hook triggers and raises OutOfRange. The
        # session should go into should_stop() mode. It should raise the
        # exception. So next step should not be executed.
        session.run(do_step)
        self.assertTrue(False)
      self.assertTrue(session.should_stop())

  def test_exit_cleanly_on_stop_iteration_exception(self):
    # Tests that we stop cleanly when OutOfRange is raised.
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(2, StopIteration)
      session = monitored_session.MonitoredSession(hooks=[hook])
      # session should cleanly exit from the context.
      with session:
        self.assertEqual(0, session.run(gstep))
        self.assertFalse(session.should_stop())
        # Here at step 1, the hook triggers and raises StopIteration. The
        # session should go into should_stop() mode. It should raise the
        # exception. So next step should not be executed.
        session.run(do_step)
        self.assertTrue(False)
      self.assertTrue(session.should_stop())

  def test_regular_exception_pass_through_run(self):
    # Tests that regular exceptions just pass through a "with
    # MonitoredSession" block and set the session in stop mode.
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(4, RuntimeError('regular exception'))
      session = monitored_session.MonitoredSession(hooks=[hook])
      with self.assertRaisesRegexp(RuntimeError, 'regular exception'):
        with session:
          self.assertEqual(0, session.run(gstep))
          self.assertEqual(1, session.run(do_step))
          self.assertEqual(2, session.run(do_step))
          self.assertFalse(session.should_stop())
          # This triggers the hook and raises the exception
          session.run(do_step)
          # We should not hit this
          self.assertFalse(True)
      self.assertTrue(hook.raised)
      self.assertTrue(session.should_stop())

  def test_regular_exception_reported_to_coord_pass_through_run(self):
    # Tests that regular exceptions reported to the coordinator from a thread
    # passes through a "run()" call within a "with MonitoredSession" block and
    # set the session in stop mode.
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      session = monitored_session.MonitoredSession()
      run_performed_without_error = False
      with self.assertRaisesRegexp(RuntimeError, 'a thread wants to stop'):
        with session:
          self.assertEqual(0, session.run(gstep))
          # Report an exception through the coordinator.
          try:
            raise RuntimeError('a thread wants to stop')
          except RuntimeError as e:
            session._coordinated_creator.coord.request_stop(e)
          # Call run() which should perform normally.
          self.assertEqual(0, session.run(gstep))
          run_performed_without_error = True
      self.assertTrue(run_performed_without_error)

  def test_regular_exception_reported_to_coord_pass_through_return(self):
    # Tests that regular exceptions reported to the coordinator from a thread
    # passes through returning from a "with MonitoredSession" block and
    # set the session in stop mode.
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      session = monitored_session.MonitoredSession()
      with self.assertRaisesRegexp(RuntimeError, 'a thread wants to stop'):
        with session:
          self.assertEqual(0, session.run(gstep))
          # Report an exception through the coordinator.
          try:
            raise RuntimeError('a thread wants to stop')
          except RuntimeError as e:
            session._coordinated_creator.coord.request_stop(e)
          self.assertTrue(session.should_stop())

  # This set of tests, verifies the session behavior when exceptions are raised
  # from code inside a "with MonitoredSession:" context.

  def test_stop_cleanly_when_no_exception_in_with_body(self):
    # Tests that regular exceptions pass through
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      session = monitored_session.MonitoredSession()
      with session:
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())
      # Should have closed.
      self.assertTrue(session.should_stop())
      self.assertTrue(session._is_closed())

  def test_raises_regular_exceptions_in_with_body(self):
    # Tests that regular exceptions in "with body" are seen outside.
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      session = monitored_session.MonitoredSession()
      # We should see that exception.
      with self.assertRaisesRegexp(RuntimeError, 'regular exception'):
        with session:
          self.assertEqual(1, session.run(do_step))
          self.assertEqual(2, session.run(do_step))
          self.assertFalse(session.should_stop())
          # Will be visible outside the "with body".
          raise RuntimeError('regular exception')
      # Should have closed.
      self.assertTrue(session.should_stop())
      self.assertTrue(session._is_closed())

  def test_graph(self):
    with ops.Graph().as_default() as g:
      with monitored_session.MonitoredSession() as session:
        self.assertEqual(g, session.graph)

  def test_graph_finalized_during_run_unfinalized_after_exit(self):
    with ops.Graph().as_default() as g:
      a_var = variables.Variable(0)
      with monitored_session.MonitoredSession() as session:
        self.assertEqual(0, session.run(a_var))
        self.assertTrue(g.finalized)
      self.assertFalse(g.finalized)

  def test_keep_finalized_graph_as_finalized(self):
    with ops.Graph().as_default() as g:
      a_var = variables.Variable(0)
      monitored_session.Scaffold().finalize()
      with monitored_session.MonitoredSession() as session:
        self.assertEqual(0, session.run(a_var))
        self.assertTrue(g.finalized)
      self.assertTrue(g.finalized)

  def test_merge_run_options_from_hooks(self):
    """Test for rewriting RunOptions and observing RunMetadata with hooks."""

    with ops.Graph().as_default():
      my_const = constant_op.constant(42, name='my_const')
      _ = constant_op.constant(24, name='my_const_2')

      watch_a = debug_pb2.DebugTensorWatch(
          node_name='my_const',
          output_slot=0,
          debug_ops=['DebugIdentity'],
          debug_urls=[])
      hook_a = RunOptionsMetadataHook(2, 30000, False, watch_a)
      watch_b = debug_pb2.DebugTensorWatch(
          node_name='my_const_2',
          output_slot=0,
          debug_ops=['DebugIdentity'],
          debug_urls=[])
      hook_b = RunOptionsMetadataHook(3, 60000, True, watch_b)
      with monitored_session.MonitoredSession(
          hooks=[hook_a, hook_b]) as session:
        self.assertEqual(42, session.run(my_const))

        # trace_level=3 should have overridden trace_level=2;
        # timeout_in_ms=60000 should have overridden 30000;
        # output_partition_graphs=True should have overridden False.
        # The two debug tensor watches should have been merged.
        self.assertEqual(
            [
                config_pb2.RunOptions(
                    trace_level=3,
                    timeout_in_ms=60000,
                    output_partition_graphs=True,
                    debug_options=debug_pb2.DebugOptions(
                        debug_tensor_watch_opts=[watch_a, watch_b]))
            ],
            hook_b.run_options_list)
        self.assertEqual(1, len(hook_b.run_metadata_list))
        self.assertTrue(
            isinstance(hook_b.run_metadata_list[0], config_pb2.RunMetadata))
        self.assertGreater(len(hook_b.run_metadata_list[0].partition_graphs), 0)

  def test_merge_caller_and_hook_run_options(self):
    """Test that RunOptions from caller and hooks can be merged properly."""

    with ops.Graph().as_default():
      my_const = constant_op.constant(42, name='my_const')
      _ = constant_op.constant(24, name='my_const_2')

      hook_watch = debug_pb2.DebugTensorWatch(
          node_name='my_const_2',
          output_slot=0,
          debug_ops=['DebugIdentity'],
          debug_urls=[])
      hook = RunOptionsMetadataHook(2, 60000, False, hook_watch)
      with monitored_session.MonitoredSession(hooks=[hook]) as session:
        caller_watch = debug_pb2.DebugTensorWatch(
            node_name='my_const',
            output_slot=0,
            debug_ops=['DebugIdentity'],
            debug_urls=[])
        caller_options = config_pb2.RunOptions(
            trace_level=3, timeout_in_ms=30000, output_partition_graphs=True)
        caller_options.debug_options.debug_tensor_watch_opts.extend(
            [caller_watch])
        self.assertEqual(42, session.run(my_const, options=caller_options))

        # trace_level=3 from the caller should override 2 from the hook.
        # timeout_in_ms=60000 from the hook should override from the caller.
        # output_partition_graph=True from the caller should override False
        # from the hook.
        # The two debug watches from the caller and the hook should be merged,
        # in that order.
        self.assertEqual(
            [
                config_pb2.RunOptions(
                    trace_level=3,
                    timeout_in_ms=60000,
                    output_partition_graphs=True,
                    debug_options=debug_pb2.DebugOptions(
                        debug_tensor_watch_opts=[caller_watch, hook_watch]))
            ],
            hook.run_options_list)
        self.assertEqual(1, len(hook.run_metadata_list))
        self.assertTrue(
            isinstance(hook.run_metadata_list[0], config_pb2.RunMetadata))
        self.assertGreater(len(hook.run_metadata_list[0].partition_graphs), 0)


class SingularMonitoredSessionTest(test.TestCase):
  """Tests SingularMonitoredSession."""

  def test_handles_initialization(self):
    with ops.Graph().as_default():
      a_var = variables.Variable(0)
      with monitored_session.SingularMonitoredSession() as session:
        # If it's not initialized, following statement raises an error.
        self.assertEqual(0, session.run(a_var))

  def test_do_not_handle_aborted_error(self):
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()

      class _RaiseAbortedHook(session_run_hook.SessionRunHook):

        def before_run(self, run_context):
          raise errors_impl.AbortedError(None, None, 'Abort')

      with monitored_session.SingularMonitoredSession(
          hooks=[_RaiseAbortedHook()]) as session:
        with self.assertRaises(errors_impl.AbortedError):
          self.assertEqual(0, session.run(gstep))

      with self.assertRaises(errors_impl.AbortedError):
        with monitored_session.SingularMonitoredSession(
            hooks=[_RaiseAbortedHook()]) as session:
          self.assertEqual(0, session.run(gstep))

  def test_exit_cleanly_on_out_of_range_exception(self):
    # Tests that we stop cleanly when OutOfRange is raised.
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(2, errors_impl.OutOfRangeError(None, None,
                                                              'EOI'))
      session = monitored_session.SingularMonitoredSession(hooks=[hook])
      # session should cleanly exit from the context.
      with session:
        self.assertEqual(0, session.run(gstep))
        self.assertFalse(session.should_stop())
        # Here at step 1, the hook triggers and raises OutOfRange. The
        # session should go into should_stop() mode. It should raise the
        # exception. So next step should not be executed.
        session.run(do_step)
        self.assertTrue(False)
      self.assertTrue(session.should_stop())

  def test_regular_exception_reported_to_coord_pass_through_run(self):
    # Tests that regular exceptions reported to the coordinator from a thread
    # passes through a "run()" call within a "with MonitoredSession" block and
    # set the session in stop mode.
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      session = monitored_session.SingularMonitoredSession()
      run_performed_without_error = False
      with self.assertRaisesRegexp(RuntimeError, 'a thread wants to stop'):
        with session:
          self.assertEqual(0, session.run(gstep))
          # Report an exception through the coordinator.
          try:
            raise RuntimeError('a thread wants to stop')
          except RuntimeError as e:
            session._coordinated_creator.coord.request_stop(e)
          # Call run() which should perform normally.
          self.assertEqual(0, session.run(gstep))
          run_performed_without_error = True
      self.assertTrue(run_performed_without_error)

  def test_stop_cleanly_when_no_exception_in_with_body(self):
    # Tests that regular exceptions pass through
    with ops.Graph().as_default():
      gstep = variables_lib.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      session = monitored_session.SingularMonitoredSession()
      with session:
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())
      # Should have closed.
      self.assertTrue(session.should_stop())
      self.assertEqual(None, session.raw_session())

  def test_graph(self):
    with ops.Graph().as_default() as g:
      with monitored_session.SingularMonitoredSession() as session:
        self.assertEqual(g, session.graph)

  def test_raw_session(self):
    with ops.Graph().as_default():
      with monitored_session.SingularMonitoredSession() as session:
        self.assertTrue(isinstance(session.raw_session(), session_lib.Session))


if __name__ == '__main__':
  test.main()
