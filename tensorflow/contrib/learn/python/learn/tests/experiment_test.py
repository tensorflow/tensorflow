#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Tests for TaskRunner and Experiment class."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
# importing to get flags.
from tensorflow.contrib.learn.python.learn import runner_flags  # pylint: disable=unused-import


class TestEstimator(tf.contrib.learn.Evaluable, tf.contrib.learn.Trainable):

  def __init__(self):
    self.eval_count = 0
    self.fit_count = 0
    self.monitors = []

  def evaluate(self, **kwargs):
    tf.logging.info('evaluate called with args: %s' % kwargs)
    self.eval_count += 1
    if self.eval_count > 5:
      tf.logging.info('Ran 6 evals. Done.')
      raise StopIteration()
    return [(key, kwargs[key]) for key in sorted(kwargs.keys())]

  def fit(self, **kwargs):
    tf.logging.info('fit called with args: %s' % kwargs)
    self.fit_count += 1
    if 'monitors' in kwargs:
      self.monitors = kwargs['monitors']
    return [(key, kwargs[key]) for key in sorted(kwargs.keys())]


class ExperimentTest(tf.test.TestCase):

  def test_train(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     train_steps='train_steps',
                                     eval_input_fn='eval_input',
                                     eval_metrics='eval_metrics')
    fit_args = ex.train(delay_secs=0)
    self.assertEquals(1, est.fit_count)
    self.assertIn(('max_steps', 'train_steps'), fit_args)
    self.assertEquals(0, est.eval_count)

  def test_train_delay(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input')
    for delay in [0, 1, 3]:
      start = time.time()
      ex.train(delay_secs=delay)
      duration = time.time() - start
      self.assertAlmostEqual(duration, delay, delta=0.5)

  def test_train_default_delay(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input')
    for task in [0, 1, 3]:
      start = time.time()
      tf.flags.FLAGS.task = task
      ex.train()
      duration = time.time() - start
      self.assertAlmostEqual(duration, task*5, delta=0.5)

  def test_evaluate(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input',
                                     eval_metrics='eval_metrics',
                                     eval_steps='steps',
                                     eval_delay_secs=0)
    ex.evaluate()
    self.assertEquals(1, est.eval_count)
    self.assertEquals(0, est.fit_count)

  def test_evaluate_delay(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input')

    for delay in [0, 1, 3]:
      start = time.time()
      ex.evaluate(delay_secs=delay)
      duration = time.time() - start
      tf.logging.info('eval duration (expected %f): %f', delay, duration)
      self.assertAlmostEqual(duration, delay, delta=0.5)

  def test_continuous_eval(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input',
                                     eval_metrics='eval_metrics',
                                     eval_delay_secs=0,
                                     continuous_eval_throttle_secs=0)
    self.assertRaises(StopIteration, ex.continuous_eval)
    self.assertEquals(6, est.eval_count)
    self.assertEquals(0, est.fit_count)

  def test_continuous_eval_throttle_delay(self):
    for delay in [0, 1, 2]:
      est = TestEstimator()
      ex = tf.contrib.learn.Experiment(
          est,
          train_input_fn='train_input',
          eval_input_fn='eval_input',
          eval_metrics='eval_metrics',
          continuous_eval_throttle_secs=delay,
          eval_delay_secs=0)
      start = time.time()
      self.assertRaises(StopIteration, ex.continuous_eval)
      duration = time.time() - start
      expected = 5 * delay
      tf.logging.info('eval duration (expected %f): %f', expected, duration)
      self.assertAlmostEqual(duration, expected, delta=0.5)

  def test_run_local(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input',
                                     eval_metrics='eval_metrics',
                                     train_steps=100,
                                     eval_steps=100,
                                     local_eval_frequency=10)
    ex.local_run()
    self.assertEquals(1, est.fit_count)
    self.assertEquals(1, est.eval_count)
    self.assertEquals(1, len(est.monitors))
    self.assertTrue(isinstance(est.monitors[0],
                               tf.contrib.learn.monitors.ValidationMonitor))

  def test_test(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input')
    ex.test()
    self.assertEquals(1, est.fit_count)
    self.assertEquals(1, est.eval_count)


if __name__ == '__main__':
  tf.test.main()
