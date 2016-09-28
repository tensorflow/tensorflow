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

from collections import Counter
import glob
import os
import threading
import time

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import basic_session_run_hooks
from tensorflow.contrib.learn.python.learn import monitored_session
from tensorflow.contrib.learn.python.learn import session_run_hook


class ScaffoldTest(tf.test.TestCase):
  """Scaffold tests."""

  def test_nothing_created_before_finalize(self):
    with tf.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      self.assertEqual(None, scaffold.init_op)
      self.assertEqual(None, scaffold.init_feed_dict)
      self.assertEqual(None, scaffold.init_fn)
      self.assertEqual(None, scaffold.ready_op)
      self.assertEqual(None, scaffold.local_init_op)
      self.assertEqual(None, scaffold.saver)

  def test_defaults_empty_graph(self):
    with tf.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      tf.Variable(1, name='my_var')
      scaffold.finalize()
      self.assertTrue(isinstance(scaffold.init_op, tf.Operation))
      self.assertEqual(None, scaffold.init_feed_dict)
      self.assertEqual(None, scaffold.init_fn)
      self.assertTrue(isinstance(scaffold.ready_op, tf.Tensor))
      self.assertTrue(isinstance(scaffold.local_init_op, tf.Operation))
      self.assertTrue(isinstance(scaffold.saver, tf.train.Saver))
      with self.test_session() as sess:
        self.assertTrue(b'my_var' in sess.run(scaffold.ready_op))
        sess.run([scaffold.init_op, scaffold.local_init_op])
        self.assertEquals(0, len(sess.run(scaffold.ready_op)))

  def test_defaults_no_variables(self):
    with tf.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      tf.constant(1, name='my_const')
      scaffold.finalize()
      self.assertTrue(isinstance(scaffold.init_op, tf.Operation))
      self.assertEqual(None, scaffold.init_feed_dict)
      self.assertEqual(None, scaffold.init_fn)
      self.assertTrue(isinstance(scaffold.ready_op, tf.Tensor))
      self.assertTrue(isinstance(scaffold.local_init_op, tf.Operation))
      self.assertTrue(isinstance(scaffold.saver, tf.train.Saver))

  def test_caches_values(self):
    with tf.Graph().as_default():
      tf.Variable([1])
      scaffold1 = monitored_session.Scaffold()
      scaffold1.finalize()
      scaffold2 = monitored_session.Scaffold()
      scaffold2.finalize()
      self.assertEqual(scaffold1.init_op, scaffold2.init_op)
      self.assertEqual(scaffold1.ready_op, scaffold2.ready_op)
      self.assertEqual(scaffold1.local_init_op, scaffold2.local_init_op)
      self.assertEqual(scaffold1.saver, scaffold2.saver)

  def test_raise_error_if_more_than_one_cached_item(self):
    with tf.Graph().as_default():
      tf.Variable([1])
      tf.add_to_collection(tf.GraphKeys.SAVERS, tf.train.Saver())
      tf.add_to_collection(tf.GraphKeys.SAVERS, tf.train.Saver())
      with self.assertRaisesRegexp(RuntimeError, 'More than one item'):
        monitored_session.Scaffold().finalize()

  def test_uses_passed_values(self):
    with tf.Graph().as_default():
      tf.Variable([1])
      saver = tf.train.Saver()
      scaffold = monitored_session.Scaffold(
          init_op=2,
          init_feed_dict=3,
          init_fn=lambda scaffold, sess: 4,
          ready_op=5,
          local_init_op=6,
          saver=saver)
      scaffold.finalize()
      self.assertEqual(2, scaffold.init_op)
      self.assertEqual(3, scaffold.init_feed_dict)
      self.assertTrue(callable(scaffold.init_fn))
      self.assertEqual(5, scaffold.ready_op)
      self.assertEqual(6, scaffold.local_init_op)
      self.assertEqual(saver, scaffold.saver)

  def test_graph_is_finalized(self):
    with tf.Graph().as_default():
      tf.Variable([1])
      monitored_session.Scaffold().finalize()
      with self.assertRaisesRegexp(RuntimeError,
                                   'Graph is finalized and cannot be modified'):
        tf.constant([0])


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


class WrappedSessionTest(tf.test.TestCase):
  """_WrappedSession tests."""

  def test_properties(self):
    with self.test_session() as sess:
      tf.constant(0.0)
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
      c = tf.constant(0)
      v = tf.identity(c)
      self.assertEqual(42, sess.run(v, feed_dict={c: 42}))
      wrapped_sess = monitored_session._WrappedSession(sess)
      self.assertEqual(51, wrapped_sess.run(v, feed_dict={c: 51}))


def busy_wait_for_coord_stop(coord):
  while not coord.should_stop():
    time.sleep(0.001)


class CoordinatedSessionTest(tf.test.TestCase):
  """_CoordinatedSession tests."""

  def test_properties(self):
    with self.test_session() as sess:
      tf.constant(0.0)
      coord = tf.train.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertEquals(sess.graph, coord_sess.graph)
      self.assertEquals(sess.sess_str, coord_sess.sess_str)

  def test_run(self):
    with self.test_session() as sess:
      c = tf.constant(0)
      v = tf.identity(c)
      coord = tf.train.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertEqual(42, coord_sess.run(v, feed_dict={c: 42}))

  def test_should_stop_on_close(self):
    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertFalse(coord_sess.should_stop())
      coord_sess.close()
      self.assertTrue(coord_sess.should_stop())

  def test_should_stop_on_coord_stop(self):
    with self.test_session() as sess:
      coord = tf.train.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertFalse(coord_sess.should_stop())
      coord.request_stop()
      self.assertTrue(coord_sess.should_stop())

  def test_dont_request_stop_on_exception_in_main_thread(self):
    with self.test_session() as sess:
      c = tf.constant(0)
      v = tf.identity(c)
      coord = tf.train.Coordinator()
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
      c = tf.constant(0)
      v = tf.identity(c)
      coord = tf.train.Coordinator()
      threads = [threading.Thread(
          target=busy_wait_for_coord_stop, args=(coord,)) for _ in range(3)]
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
      coord = tf.train.Coordinator()
      threads = [threading.Thread(
          target=busy_wait_for_coord_stop, args=(coord,)) for _ in range(3)]
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
      raise tf.errors.AbortedError('Aborted at N', None, None)
    self._count -= 1
    return self._sess.run(*args, **kwargs)


class RecoverableSessionTest(tf.test.TestCase):
  """_RecoverableSession tests."""

  class _SessionReturner(object):

    def __init__(self, sess):
      self._sess = sess

    def create_session(self):
      return self._sess

  def test_properties(self):
    with self.test_session() as sess:
      tf.constant(0.0)
      recoverable_sess = monitored_session._RecoverableSession(
          self._SessionReturner(sess))
      self.assertEquals(sess.graph, recoverable_sess.graph)
      self.assertEquals(sess.sess_str, recoverable_sess.sess_str)

  def test_run(self):
    with self.test_session() as sess:
      c = tf.constant(0)
      v = tf.identity(c)
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

      c = tf.constant(0)
      v = tf.identity(c)
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


class FakeHook(session_run_hook.SessionRunHook):

  def __init__(self):
    self.should_stop = False
    self.request = None
    self.call_counter = Counter()
    self.last_run_context = None
    self.last_run_values = None

  def before_run(self, run_context):
    self.call_counter['before_run'] += 1
    self.last_run_context = run_context
    return self.request

  def after_run(self, run_context, run_values):
    self.call_counter['after_run'] += 1
    self.last_run_values = run_values
    if self.should_stop:
      run_context.request_stop()


class HookedSessionTest(tf.test.TestCase):

  def testRunPassesAllArguments(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      mock_run = FakeSession(sess)
      mon_sess = monitored_session._HookedSession(sess=mock_run, hooks=[])
      a_tensor = tf.constant([0], name='a_tensor')
      sess.run(tf.initialize_all_variables())
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
    with tf.Graph().as_default(), tf.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = tf.constant([0], name='a_tensor')
      sess.run(tf.initialize_all_variables())
      mon_sess.run(a_tensor)

      for hook in [mock_hook, mock_hook2]:
        self.assertEqual(
            hook.last_run_values,
            session_run_hook.SessionRunValues(results=None))
        self.assertEqual(hook.last_run_context.original_args,
                         session_run_hook.SessionRunArgs(a_tensor))
        self.assertEqual(hook.last_run_context.session, sess)
        self.assertEqual(hook.call_counter['before_run'], 1)
        self.assertEqual(hook.call_counter['after_run'], 1)

  def testShouldStop(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      tf.constant([0], name='a_tensor')
      sess.run(tf.initialize_all_variables())

      mon_sess.run(fetches='a_tensor')
      self.assertFalse(mon_sess.should_stop())

      mock_hook.should_stop = True
      mon_sess.run(fetches='a_tensor')
      self.assertTrue(mon_sess.should_stop())

  def testFetchesHookRequests(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = tf.constant([0], name='a_tensor')
      another_tensor = tf.constant([5], name='another_tensor')
      third_tensor = tf.constant([10], name='third_tensor')
      mock_hook.request = session_run_hook.SessionRunArgs([another_tensor])
      mock_hook2.request = session_run_hook.SessionRunArgs([third_tensor])
      sess.run(tf.initialize_all_variables())

      output = mon_sess.run(fetches=a_tensor)
      self.assertEqual(output, [0])
      self.assertEqual(mock_hook.last_run_values.results, [5])
      self.assertEqual(mock_hook2.last_run_values.results, [10])

  def testOnlyHooksHaveFeeds(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = tf.constant([0], name='a_tensor')
      b_tensor = tf.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      sess.run(tf.initialize_all_variables())

      self.assertEqual(mon_sess.run(fetches=add_tensor), [15])

  def testBothHooksAndUserHaveFeeds(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = tf.constant([0], name='a_tensor')
      b_tensor = tf.constant([0], name='b_tensor')
      c_tensor = tf.constant([0], name='c_tensor')
      add_tensor = a_tensor + b_tensor + c_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      sess.run(tf.initialize_all_variables())

      feed_dict = {c_tensor: [20]}
      self.assertEqual(
          mon_sess.run(fetches=add_tensor, feed_dict=feed_dict), [35])
      # User feed_dict should not be changed
      self.assertEqual(len(feed_dict), 1)

  def testHooksFeedConflicts(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = tf.constant([0], name='a_tensor')
      b_tensor = tf.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [10]})
      sess.run(tf.initialize_all_variables())

      with self.assertRaisesRegexp(RuntimeError, 'Same tensor is fed'):
        mon_sess.run(fetches=add_tensor)

  def testHooksAndUserFeedConflicts(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = tf.constant([0], name='a_tensor')
      b_tensor = tf.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      sess.run(tf.initialize_all_variables())

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


class MonitoredSessionTest(tf.test.TestCase):
  """MonitoredSession tests."""

  def _test_dir(self, test_name):
    """Create an empty dir to use for tests.

    Args:
      test_name: Name of the test.

    Returns:
      Absolute path to the test directory.
    """
    test_dir = os.path.join(self.get_temp_dir(), test_name)
    if os.path.isdir(test_dir):
      for f in glob.glob('%s/*' % test_dir):
        os.remove(f)
    else:
      os.makedirs(test_dir)
    return test_dir

  def test_defaults(self):
    with tf.Graph().as_default():
      a_var = tf.Variable(0)
      with monitored_session.MonitoredSession() as session:
        self.assertEqual(0, session.run(a_var))

  def test_last_step(self):
    logdir = self._test_dir('test_last_step')
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      do_step = tf.assign_add(gstep, 1)
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
    logdir = self._test_dir('test_num_steps')
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      do_step = tf.assign_add(gstep, 1)
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
    logdir = self._test_dir('test_recovery')
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      do_step = tf.assign_add(gstep, 1)
      scaffold = monitored_session.Scaffold()
      # Use a hook to save the model every 100 steps.  It also saves it at
      # the end.
      hooks = [basic_session_run_hooks.CheckpointSaverHook(
          logdir, save_steps=1, scaffold=scaffold)]
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

  def test_retry_on_aborted_error(self):
    # Tests that we silently retry on abort.  Note that this does not test
    # recovery as we do not use a CheckpointSaver in this test.
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      do_step = tf.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(4, tf.errors.AbortedError(None, None, 'Abort'))
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

  def test_recover_and_retry_on_aborted_error(self):
    # Tests that we silently retry and recover on abort.  This test uses
    # a CheckpointSaver to have something to recover from.
    logdir = self._test_dir('test_recover_and_retry_on_aborted_error')
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      do_step = tf.assign_add(gstep, 1)
      scaffold = monitored_session.Scaffold()
      abort_hook = RaiseOnceAtCountN(
          4, tf.errors.AbortedError(None, None, 'Abort'))
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
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      do_step = tf.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(2, tf.errors.OutOfRangeError(None, None, 'EOI'))
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
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      do_step = tf.assign_add(gstep, 1)
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
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      do_step = tf.assign_add(gstep, 1)
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
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
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
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
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
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      do_step = tf.assign_add(gstep, 1)
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
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      do_step = tf.assign_add(gstep, 1)
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
    g = tf.Graph()
    with g.as_default():
      session = monitored_session.MonitoredSession()
      self.assertEqual(g, session.graph)


if __name__ == '__main__':
  tf.test.main()
