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

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session
from tensorflow.contrib.learn.python.learn import monitors
from tensorflow.contrib.learn.python.learn import session_run_hook
from tensorflow.contrib.learn.python.learn.wrapped_session import WrappedSession


class FakeSession(WrappedSession):

  def __init__(self, sess):
    WrappedSession.__init__(self, sess)
    self.args_called = {}

  def run(self, fetches, **kwargs):
    self.args_called = dict(kwargs)
    # Call run only with fetches since we directly pass other arguments.
    return WrappedSession.run(self, fetches)


class FakeMonitor(monitors.BaseMonitor):

  def __init__(self):
    monitors.BaseMonitor.__init__(self)
    self.should_stop = False
    self.requested_tensors = []
    self.call_counter = Counter()
    self.last_begin_step = None
    self.last_end_step = None
    self.last_post_step = None

  def step_begin(self, step):
    self.call_counter['step_begin'] += 1
    self.last_begin_step = step
    return self.requested_tensors

  def step_end(self, step, output):
    self.call_counter['step_end'] += 1
    self.last_end_step = step
    self.output = output
    return self.should_stop

  def post_step(self, step, session):
    self.call_counter['post_step'] += 1
    self.last_post_step = step
    self.session = session


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


class MonitoredSessionTest(tf.test.TestCase):

  def testRunPassesAllArguments(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      mock_run = FakeSession(sess)
      mon_sess = monitored_session.MonitoredSession(sess=mock_run, hooks=[])
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
      mon_sess = monitored_session.MonitoredSession(
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
      mon_sess = monitored_session.MonitoredSession(
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
      mon_sess = monitored_session.MonitoredSession(
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
      mon_sess = monitored_session.MonitoredSession(
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
      mon_sess = monitored_session.MonitoredSession(
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
      mon_sess = monitored_session.MonitoredSession(
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
      mon_sess = monitored_session.MonitoredSession(
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


class MonitoredSessionWithMonitorsTest(tf.test.TestCase):

  def testCallsMonitorsBeginEndAndPost(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      global_step_tensor = tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess, hooks=[mock_mon, mock_mon2])
      a_tensor = tf.constant([0], name='a_tensor')
      sess.run(tf.initialize_all_variables())
      sess.run(global_step_tensor.assign(10))
      mon_sess.run(fetches=a_tensor)

      for mon in [mock_mon, mock_mon2]:
        self.assertEqual(mon.output, {})
        self.assertEqual(mon.last_begin_step, 11)
        self.assertEqual(mon.last_end_step, 11)
        self.assertEqual(mon.last_post_step, 11)
        self.assertEqual(mon.call_counter['step_end'], 1)
        self.assertEqual(mon.call_counter['step_begin'], 1)
        self.assertEqual(mon.call_counter['post_step'], 1)

  def testCallsMonitorsWithLastStep(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      global_step_tensor = tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess, hooks=[mock_mon, mock_mon2])
      inc_5 = tf.assign_add(global_step_tensor, 5)
      # Initialize global_step_tensor to '0':
      sess.run(tf.initialize_all_variables())

      mon_sess.run(fetches=[inc_5])
      for mon in [mock_mon, mock_mon2]:
        self.assertEqual(mon.last_begin_step, 1)
        self.assertEqual(mon.last_end_step, 1)
        self.assertEqual(mon.last_post_step, 1)

      mon_sess.run(fetches=[inc_5])
      for mon in [mock_mon, mock_mon2]:
        self.assertEqual(mon.last_begin_step, 6)
        self.assertEqual(mon.last_end_step, 6)
        self.assertEqual(mon.last_post_step, 6)

      mon_sess.run(fetches=[inc_5])
      for mon in [mock_mon, mock_mon2]:
        self.assertEqual(mon.last_begin_step, 11)
        self.assertEqual(mon.last_end_step, 11)
        self.assertEqual(mon.last_post_step, 11)

  def testShouldStop(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess, hooks=[mock_mon, mock_mon2])
      tf.constant([0], name='a_tensor')
      sess.run(tf.initialize_all_variables())

      mon_sess.run(fetches='a_tensor')
      self.assertFalse(mon_sess.should_stop())

      mock_mon.should_stop = True
      mon_sess.run(fetches='a_tensor')
      self.assertTrue(mon_sess.should_stop())

  def testFetchesMonitorRequests(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess, hooks=[mock_mon, mock_mon2])
      a_tensor = tf.constant([0], name='a_tensor')
      tf.constant([5], name='another_tensor')
      tf.constant([10], name='third_tensor')
      mock_mon.requested_tensors = ['another_tensor']
      mock_mon2.requested_tensors = ['third_tensor']
      sess.run(tf.initialize_all_variables())

      output = mon_sess.run(fetches=a_tensor)
      self.assertEqual(output, [0])
      self.assertEqual(mock_mon.output['another_tensor'], [5])
      self.assertEqual(mock_mon2.output['third_tensor'], [10])

  def testMonitorHasSameRequests(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess, hooks=[mock_mon, mock_mon2])
      a_tensor = tf.constant([0], name='a_tensor')
      tf.constant([5], name='another_tensor')
      mock_mon.requested_tensors = ['another_tensor']
      mock_mon2.requested_tensors = ['another_tensor']
      sess.run(tf.initialize_all_variables())

      output = mon_sess.run(fetches=a_tensor)
      self.assertEqual(output, [0])
      self.assertEqual(mock_mon.output['another_tensor'], [5])
      self.assertEqual(mock_mon2.output['another_tensor'], [5])

  def testMonitorHasSameRequestWithCaller(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess, hooks=[mock_mon, mock_mon2])
      a_tensor = tf.constant([0], name='a_tensor')
      tf.constant([10], name='third_tensor')
      mock_mon.requested_tensors = ['a_tensor']
      mock_mon2.requested_tensors = ['third_tensor']
      sess.run(tf.initialize_all_variables())

      output = mon_sess.run(fetches=a_tensor)
      self.assertEqual(output, [0])
      self.assertEqual(mock_mon.output['a_tensor'], [0])
      self.assertEqual(mock_mon2.output['third_tensor'], [10])

  def testMonitorRequestWithColonZero(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess, hooks=[mock_mon, mock_mon2])
      a_tensor = tf.constant([0], name='a_tensor')
      tf.constant([5], name='another_tensor')
      mock_mon.requested_tensors = ['another_tensor']
      mock_mon2.requested_tensors = ['another_tensor:0']
      sess.run(tf.initialize_all_variables())

      output = mon_sess.run(fetches=a_tensor)
      self.assertEqual(output, [0])
      self.assertEqual(mock_mon.output['another_tensor'], [5])
      self.assertEqual(mock_mon2.output['another_tensor:0'], [5])


if __name__ == '__main__':
  tf.test.main()
