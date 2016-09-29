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

from tensorflow.contrib.learn.python.learn import run_config
from tensorflow.python.util.all_util import reveal_undocumented


class TestEstimator(tf.contrib.learn.Evaluable, tf.contrib.learn.Trainable):

  def __init__(self, config=None):
    self.eval_count = 0
    self.fit_count = 0
    self.monitors = []
    self._config = config or run_config.RunConfig()

  @property
  def config(self):
    return self._config

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

  def setUp(self):
    # The official name is tf.train, so tf.training was obliterated.
    reveal_undocumented('tensorflow.python.training')

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
    config = run_config.RunConfig()
    est = TestEstimator(config)
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input')
    for task in [0, 1, 3]:
      start = time.time()
      config.task = task
      ex.train()
      duration = time.time() - start
      self.assertAlmostEqual(duration, task*5, delta=0.5)

  @tf.test.mock.patch('tensorflow.python.training.server_lib.Server')  # pylint: disable=line-too-long
  def test_train_starts_server(self, mock_server):
    # Arrange.
    config = tf.contrib.learn.RunConfig(
        master='host4:2222',
        cluster_spec=tf.train.ClusterSpec(
            {'ps': ['host1:2222', 'host2:2222'],
             'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
        ),
        job_name='worker',
        task=1,
        num_cores=15,
        gpu_memory_fraction=0.314,
    )

    est = TestEstimator(config)
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input')

    # Act.
    # We want to make sure we discount the time it takes to start the server
    # in our accounting of the delay, so we set a small delay here.
    start = time.time()
    ex.train(delay_secs=1)
    duration = time.time() - start

    # Assert.
    expected_config_proto = tf.ConfigProto()
    expected_config_proto.inter_op_parallelism_threads = 15
    expected_config_proto.intra_op_parallelism_threads = 15
    expected_config_proto.gpu_options.per_process_gpu_memory_fraction = 0.314
    mock_server.assert_called_with(
        config.cluster_spec,
        job_name='worker',
        task_index=1,
        config=expected_config_proto,
        start=False)
    mock_server.assert_has_calls([tf.test.mock.call().start()])

    # Ensure that the delay takes into account the time to start the server.
    self.assertAlmostEqual(duration, 1.0, delta=0.5)

  @tf.test.mock.patch('tensorflow.python.training.server_lib.Server')  # pylint: disable=line-too-long
  def test_train_server_does_not_start_without_cluster_spec(self, mock_server):
    config = tf.contrib.learn.RunConfig(master='host4:2222')
    ex = tf.contrib.learn.Experiment(TestEstimator(config),
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input')
    ex.train()

    # The server should not have started because there was no ClusterSpec.
    self.assertFalse(mock_server.called)

  @tf.test.mock.patch('tensorflow.python.training.server_lib.Server')  # pylint: disable=line-too-long
  def test_train_server_does_not_start_with_empty_master(self, mock_server):
    config = tf.contrib.learn.RunConfig(
        cluster_spec=tf.train.ClusterSpec(
            {'ps': ['host1:2222', 'host2:2222'],
             'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
        ),
        master='',)
    ex = tf.contrib.learn.Experiment(TestEstimator(config),
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input')
    ex.train()

    # The server should not have started because master was the empty string.
    self.assertFalse(mock_server.called)

  def test_train_raises_if_job_name_is_missing(self):
    no_job_name = tf.contrib.learn.RunConfig(
        cluster_spec=tf.train.ClusterSpec(
            {'ps': ['host1:2222', 'host2:2222'],
             'worker': ['host3:2222', 'host4:2222', 'host5:2222']},
        ),
        task=1,
        master='host3:2222',  # Normally selected by job_name
    )
    with self.assertRaises(ValueError):
      ex = tf.contrib.learn.Experiment(TestEstimator(no_job_name),
                                       train_input_fn='train_input',
                                       eval_input_fn='eval_input')
      ex.train()

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

  def test_train_and_evaluate(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input',
                                     eval_metrics='eval_metrics',
                                     train_steps=100,
                                     eval_steps=100)
    ex.train_and_evaluate()
    self.assertEquals(1, est.fit_count)
    self.assertEquals(1, est.eval_count)
    self.assertEquals(1, len(est.monitors))
    self.assertTrue(isinstance(est.monitors[0],
                               tf.contrib.learn.monitors.ValidationMonitor))

  @tf.test.mock.patch('tensorflow.python.training.server_lib.Server')  # pylint: disable=line-too-long
  def test_run_std_server(self, mock_server):
    # Arrange.
    config = tf.contrib.learn.RunConfig(
        master='host2:2222',
        cluster_spec=tf.train.ClusterSpec(
            {'ps': ['host1:2222', 'host2:2222'],
             'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
        ),
        job_name='ps',
        task=1,
        num_cores=15,
        gpu_memory_fraction=0.314,
    )
    est = TestEstimator(config)
    ex = tf.contrib.learn.Experiment(est,
                                     train_input_fn='train_input',
                                     eval_input_fn='eval_input')

    # Act.
    ex.run_std_server()

    # Assert.
    mock_server.assert_has_calls([tf.test.mock.call().start(),
                                  tf.test.mock.call().join()])

  @tf.test.mock.patch('tensorflow.python.training.server_lib.Server')  # pylint: disable=line-too-long
  def test_run_std_server_raises_without_cluster_spec(self, mock_server):
    config = tf.contrib.learn.RunConfig(master='host4:2222')
    with self.assertRaises(ValueError):
      ex = tf.contrib.learn.Experiment(TestEstimator(config),
                                       train_input_fn='train_input',
                                       eval_input_fn='eval_input')
      ex.run_std_server()

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
