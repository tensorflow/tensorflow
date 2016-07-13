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


class MonitoredSessionTest(tf.test.TestCase):

  def testRunPassesAllArguments(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      global_step_tensor = tf.contrib.framework.create_global_step()
      mock_run = FakeSession(sess)
      mon_sess = monitored_session.MonitoredSession(
          sess=mock_run,
          monitors=[],
          global_step_tensor=global_step_tensor)
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

  def testCallsMonitorsBeginEndAndPost(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      global_step_tensor = tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess,
          monitors=[mock_mon, mock_mon2],
          global_step_tensor=global_step_tensor)
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
          sess=sess,
          monitors=[mock_mon, mock_mon2],
          global_step_tensor=global_step_tensor)
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
      global_step_tensor = tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess,
          monitors=[mock_mon, mock_mon2],
          global_step_tensor=global_step_tensor)
      tf.constant([0], name='a_tensor')
      sess.run(tf.initialize_all_variables())

      mon_sess.run(fetches='a_tensor')
      self.assertFalse(mon_sess.should_stop())

      mock_mon.should_stop = True
      mon_sess.run(fetches='a_tensor')
      self.assertTrue(mon_sess.should_stop())

  def testFetchesMonitorRequests(self):
    with tf.Graph().as_default(), tf.Session() as sess:
      global_step_tensor = tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess,
          monitors=[mock_mon, mock_mon2],
          global_step_tensor=global_step_tensor)
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
      global_step_tensor = tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess,
          monitors=[mock_mon, mock_mon2],
          global_step_tensor=global_step_tensor)
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
      global_step_tensor = tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess,
          monitors=[mock_mon, mock_mon2],
          global_step_tensor=global_step_tensor)
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
      global_step_tensor = tf.contrib.framework.create_global_step()
      mock_mon = FakeMonitor()
      mock_mon2 = FakeMonitor()
      mon_sess = monitored_session.MonitoredSession(
          sess=sess,
          monitors=[mock_mon, mock_mon2],
          global_step_tensor=global_step_tensor)
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
