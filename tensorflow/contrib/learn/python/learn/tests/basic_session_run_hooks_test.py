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
from tensorflow.python.training import monitored_session


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
      hook = tf.train.StepCounterHook(
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
      hook = tf.train.SummarySaverHook(
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
