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

"""Monitors tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib import testing
from tensorflow.contrib.learn.python import learn
from tensorflow.contrib.learn.python.learn import supervised_session
from tensorflow.python.platform import tf_logging as logging


class _MyEveryN(learn.monitors.EveryN):

  def __init__(self, every_n_steps=100, first_n_steps=1):
    super(_MyEveryN, self).__init__(
        every_n_steps=every_n_steps, first_n_steps=first_n_steps)
    self._steps_begun = []
    self._steps_ended = []
    self._post_steps = []

  @property
  def steps_begun(self):
    return self._steps_begun

  @property
  def steps_ended(self):
    return self._steps_ended

  @property
  def post_steps(self):
    return self._post_steps

  def every_n_step_begin(self, step):
    super(_MyEveryN, self).every_n_step_begin(step)
    self._steps_begun.append(step)
    return []

  def every_n_step_end(self, step, outputs):
    super(_MyEveryN, self).every_n_step_end(step, outputs)
    self._steps_ended.append(step)
    return False

  def every_n_post_step(self, step, session):
    super(_MyEveryN, self).every_n_post_step(step, session)
    self._post_steps.append(step)
    return False


class MonitorsTest(tf.test.TestCase):
  """Monitors tests."""

  def setUp(self):
    # Mock out logging calls so we can verify whether correct tensors are being
    # monitored.
    self._actual_log = logging.info

    def mockLog(*args, **kwargs):
      self.logged_message = args
      self._actual_log(*args, **kwargs)

    logging.info = mockLog

  def tearDown(self):
    logging.info = self._actual_log

  def _run_monitor(self, monitor, num_epochs=3, num_steps_per_epoch=10,
                   pass_max_steps=True):
    if pass_max_steps:
      max_steps = num_epochs * num_steps_per_epoch - 1
    else:
      max_steps = None
    monitor.begin(max_steps=max_steps)
    for epoch in xrange(num_epochs):
      monitor.epoch_begin(epoch)
      should_stop = False
      step = epoch * num_steps_per_epoch
      next_epoch_step = step + num_steps_per_epoch
      while (not should_stop) and (step < next_epoch_step):
        tensors = monitor.step_begin(step)
        output = tf.get_default_session().run(tensors) if tensors else {}
        output = dict(zip(
            [t.name if isinstance(t, tf.Tensor) else t for t in tensors],
            output))
        should_stop = monitor.step_end(step=step, output=output)
        monitor.post_step(step=step, session=None)
        step += 1
      monitor.epoch_end(epoch)
    monitor.end()

  def test_base_monitor(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      self._run_monitor(learn.monitors.BaseMonitor())

  def test_every_n(self):
    monitor = _MyEveryN(every_n_steps=8, first_n_steps=2)
    with tf.Graph().as_default() as g, self.test_session(g):
      self._run_monitor(monitor, num_epochs=3, num_steps_per_epoch=10)
      expected_steps = [0, 1, 2, 10, 18, 26, 29]
      self.assertEqual(expected_steps, monitor.steps_begun)
      self.assertEqual(expected_steps, monitor.steps_ended)
      self.assertEqual(expected_steps, monitor.post_steps)

  def test_every_n_no_max_steps(self):
    monitor = _MyEveryN(every_n_steps=8, first_n_steps=2)
    with tf.Graph().as_default() as g, self.test_session(g):
      self._run_monitor(monitor, num_epochs=3, num_steps_per_epoch=10,
                        pass_max_steps=False)
      begin_end_steps = [0, 1, 2, 10, 18, 26]
      post_steps = [0, 1, 2, 10, 18, 26, 29]
      self.assertEqual(begin_end_steps, monitor.steps_begun)
      self.assertEqual(begin_end_steps, monitor.steps_ended)
      self.assertEqual(post_steps, monitor.post_steps)

  def test_every_n_recovered_after_step_begin(self):
    monitor = _MyEveryN(every_n_steps=8)
    with tf.Graph().as_default() as g, self.test_session(g):
      for step in [8, 16]:
        monitor.step_begin(step)
        monitor.step_begin(step)
        monitor.step_end(step, output=None)
        monitor.post_step(step, session=None)
      # It should call begin again since, end was not called
      self.assertEqual([8, 8, 16, 16], monitor.steps_begun)
      self.assertEqual([8, 16], monitor.steps_ended)
      self.assertEqual([8, 16], monitor.post_steps)

  def test_every_n_recovered_after_step_end(self):
    monitor = _MyEveryN(every_n_steps=8)
    with tf.Graph().as_default() as g, self.test_session(g):
      for step in [8, 16]:
        monitor.step_begin(step)
        monitor.step_end(step, output=None)
        monitor.post_step(step, session=None)
        monitor.step_begin(step)
        monitor.step_end(step, output=None)
        monitor.post_step(step, session=None)
      # It should not call begin twice since end was called
      self.assertEqual([8, 16], monitor.steps_begun)
      self.assertEqual([8, 16], monitor.steps_ended)
      self.assertEqual([8, 16], monitor.post_steps)

  def test_every_n_call_post_step_at_the_end(self):
    monitor = _MyEveryN(every_n_steps=8)
    with tf.Graph().as_default() as g, self.test_session(g):
      monitor.begin()
      for step in [8, 16]:
        monitor.step_begin(step)
        monitor.step_end(step, output=None)
        monitor.post_step(step, session=None)
      monitor.step_begin(19)
      monitor.step_end(19, output=None)
      monitor.post_step(19, session=None)
      monitor.end(session=None)
      # It should not call begin twice since end was called
      self.assertEqual([8, 16], monitor.steps_begun)
      self.assertEqual([8, 16], monitor.steps_ended)
      self.assertEqual([8, 16, 19], monitor.post_steps)

  def test_every_n_call_post_step_should_not_be_called_twice(self):
    monitor = _MyEveryN(every_n_steps=8)
    with tf.Graph().as_default() as g, self.test_session(g):
      monitor.begin()
      for step in [8, 16]:
        monitor.step_begin(step)
        monitor.step_end(step, output=None)
        monitor.post_step(step, session=None)
      monitor.step_begin(16)
      monitor.step_end(16, output=None)
      monitor.post_step(16, session=None)
      monitor.end(session=None)
      # It should not call begin twice since end was called
      self.assertEqual([8, 16], monitor.steps_begun)
      self.assertEqual([8, 16], monitor.steps_ended)
      self.assertEqual([8, 16], monitor.post_steps)

  def test_print(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      t = tf.constant(42.0, name='foo')
      self._run_monitor(learn.monitors.PrintTensor(tensor_names=[t.name]))
      self.assertRegexpMatches(str(self.logged_message), t.name)

  def test_logging_trainable(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      var = tf.Variable(tf.constant(42.0), name='foo')
      var.initializer.run()
      cof = tf.constant(1.0)
      loss = tf.sub(tf.mul(var, cof), tf.constant(1.0))
      train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
      tf.get_default_session().run(train_step)
      self._run_monitor(learn.monitors.LoggingTrainable('foo'))
      self.assertRegexpMatches(str(self.logged_message), var.name)

  def test_summary_saver(self):
    with tf.Graph().as_default() as g, self.test_session(g):
      log_dir = 'log/dir'
      summary_writer = testing.FakeSummaryWriter(log_dir, g)
      var = tf.Variable(0.0)
      var.initializer.run()
      tensor = tf.assign_add(var, 1.0)
      summary_op = tf.scalar_summary('my_summary', tensor)
      self._run_monitor(
          learn.monitors.SummarySaver(
              summary_op=summary_op, save_steps=8,
              summary_writer=summary_writer),
          num_epochs=3, num_steps_per_epoch=10)
      summary_writer.assert_summaries(
          test_case=self, expected_logdir=log_dir, expected_graph=g,
          expected_summaries={
              0: {'my_summary': 1.0},
              1: {'my_summary': 2.0},
              9: {'my_summary': 3.0},
              17: {'my_summary': 4.0},
              25: {'my_summary': 5.0},
              29: {'my_summary': 6.0},
          })

  # TODO(b/29293803): Add better tests with a mocked estimator.
  def test_validation_monitor(self):
    monitor = learn.monitors.ValidationMonitor(x=tf.constant(2.0))
    with tf.Graph().as_default() as g, self.test_session(g):
      with self.assertRaisesRegexp(ValueError, 'set_estimator'):
        self._run_monitor(monitor)

  def test_graph_dump(self):
    monitor0 = learn.monitors.GraphDump()
    monitor1 = learn.monitors.GraphDump()
    with tf.Graph().as_default() as g, self.test_session(g):
      const_var = tf.Variable(42.0, name='my_const')
      counter_var = tf.Variable(0.0, name='my_counter')
      assign_add = tf.assign_add(counter_var, 1.0, name='my_assign_add')
      tf.initialize_all_variables().run()

      self._run_monitor(monitor0, num_epochs=3, num_steps_per_epoch=10)
      self.assertEqual({
          step: {
              const_var.name: 42.0,
              counter_var.name: step + 1.0,
              assign_add.name: step + 1.0,
          } for step in xrange(30)
      }, monitor0.data)

      self._run_monitor(monitor1, num_epochs=3, num_steps_per_epoch=10)
      self.assertEqual({
          step: {
              const_var.name: 42.0,
              counter_var.name: step + 31.0,
              assign_add.name: step + 31.0,
          } for step in xrange(30)
      }, monitor1.data)

      for step in xrange(30):
        matched, non_matched = monitor1.compare(monitor0, step=step)
        self.assertEqual([const_var.name], matched)
        self.assertEqual({
            assign_add.name: (step + 31.0, step + 1.0),
            counter_var.name: (step + 31.0, step + 1.0),
        }, non_matched)
        matched, non_matched = monitor0.compare(monitor1, step=step)
        self.assertEqual([const_var.name], matched)
        self.assertEqual({
            assign_add.name: (step + 1.0, step + 31.0),
            counter_var.name: (step + 1.0, step + 31.0),
        }, non_matched)

  def test_capture_variable(self):
    monitor = learn.monitors.CaptureVariable(
        var_name='my_assign_add:0', every_n=8, first_n=2)
    with tf.Graph().as_default() as g, self.test_session(g):
      var = tf.Variable(0.0, name='my_var')
      var.initializer.run()
      tf.assign_add(var, 1.0, name='my_assign_add')
      self._run_monitor(monitor, num_epochs=3, num_steps_per_epoch=10)
      self.assertEqual({
          0: 1.0,
          1: 2.0,
          2: 3.0,
          10: 4.0,
          18: 5.0,
          26: 6.0,
          29: 7.0,
      }, monitor.values)


class StopAtStepTest(tf.test.TestCase):

  def test_raise_in_both_last_step_and_num_steps(self):
    with self.assertRaises(ValueError):
      learn.monitors.StopAtStep(num_steps=10, last_step=20)

  def test_stop_based_on_last_step(self):
    m = learn.monitors.StopAtStep(last_step=10)
    m.step_begin(5)
    self.assertFalse(m.step_end(5, None))
    m.step_begin(9)
    self.assertFalse(m.step_end(9, None))
    m.step_begin(10)
    self.assertTrue(m.step_end(10, None))
    m.step_begin(11)
    self.assertTrue(m.step_end(11, None))

  def test_stop_based_on_num_step(self):
    m = learn.monitors.StopAtStep(num_steps=10)
    m.step_begin(5)
    self.assertFalse(m.step_end(5, None))
    m.step_begin(13)
    self.assertFalse(m.step_end(13, None))
    m.step_begin(14)
    self.assertTrue(m.step_end(14, None))
    m.step_begin(15)
    self.assertTrue(m.step_end(15, None))


class CheckpointSaverTest(tf.test.TestCase):

  def setUp(self):
    self.model_dir = tempfile.mkdtemp()
    self.graph = tf.Graph()
    with self.graph.as_default():
      self.scaffold = supervised_session.Scaffold()
      self.global_step = tf.contrib.framework.get_or_create_global_step()
      self.train_op = tf.assign_add(self.global_step, 1)

  def tearDown(self):
    shutil.rmtree(self.model_dir, ignore_errors=True)

  def _run(self, monitor, step, train_op, sess):
    monitor.step_begin(step)
    sess.run(train_op)
    monitor.post_step(step, sess)

  def test_raise_in_both_secs_and_steps(self):
    with self.assertRaises(ValueError):
      learn.monitors.CheckpointSaver(
          self.model_dir, save_secs=10, save_steps=20)

  def test_raise_in_none_secs_and_steps(self):
    with self.assertRaises(ValueError):
      learn.monitors.CheckpointSaver(self.model_dir)

  def test_save_secs_saves_in_first_step(self):
    with self.graph.as_default():
      monitor = learn.monitors.CheckpointSaver(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      monitor.begin()
      self.scaffold.finalize()
      with tf.Session() as sess:
        sess.run(self.scaffold.init_op)
        self._run(monitor, 1, self.train_op, sess)
        self.assertEqual(1, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))

  def test_save_secs_saves_periodically(self):
    with self.graph.as_default():
      monitor = learn.monitors.CheckpointSaver(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      monitor.begin()
      self.scaffold.finalize()
      with tf.Session() as sess:
        sess.run(self.scaffold.init_op)
        self._run(monitor, 1, self.train_op, sess)
        self._run(monitor, 2, self.train_op, sess)
        # Not saved
        self.assertEqual(1, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        time.sleep(2.5)
        self._run(monitor, 3, self.train_op, sess)
        # saved
        self.assertEqual(3, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        self._run(monitor, 4, self.train_op, sess)
        self._run(monitor, 5, self.train_op, sess)
        # Not saved
        self.assertEqual(3, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        time.sleep(2.5)
        self._run(monitor, 6, self.train_op, sess)
        # saved
        self.assertEqual(6, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))

  def test_save_steps_saves_in_first_step(self):
    with self.graph.as_default():
      monitor = learn.monitors.CheckpointSaver(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      monitor.begin()
      self.scaffold.finalize()
      with tf.Session() as sess:
        sess.run(self.scaffold.init_op)
        self._run(monitor, 1, self.train_op, sess)
        self.assertEqual(1, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))

  def test_save_steps_saves_periodically(self):
    with self.graph.as_default():
      monitor = learn.monitors.CheckpointSaver(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      monitor.begin()
      self.scaffold.finalize()
      with tf.Session() as sess:
        sess.run(self.scaffold.init_op)
        self._run(monitor, 1, self.train_op, sess)
        self._run(monitor, 2, self.train_op, sess)
        # Not saved
        self.assertEqual(1, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        self._run(monitor, 3, self.train_op, sess)
        # saved
        self.assertEqual(3, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        self._run(monitor, 4, self.train_op, sess)
        # Not saved
        self.assertEqual(3, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))
        self._run(monitor, 5, self.train_op, sess)
        # saved
        self.assertEqual(5, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))

  def test_save_saves_at_end(self):
    with self.graph.as_default():
      monitor = learn.monitors.CheckpointSaver(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      monitor.begin()
      self.scaffold.finalize()
      with tf.Session() as sess:
        sess.run(self.scaffold.init_op)
        self._run(monitor, 1, self.train_op, sess)
        self._run(monitor, 2, self.train_op, sess)
        monitor.end(sess)
        self.assertEqual(2, tf.contrib.framework.load_variable(
            self.model_dir, self.global_step.name))


if __name__ == '__main__':
  tf.test.main()
