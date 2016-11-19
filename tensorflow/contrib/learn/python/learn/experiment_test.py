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

import json
import threading
import time

import tensorflow as tf

from tensorflow.contrib.learn.python.learn import run_config
from tensorflow.contrib.learn.python.learn.experiment import _GlobalStepWaiterHook
from tensorflow.python.util.all_util import reveal_undocumented

patch = tf.test.mock.patch


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


class GlobalStepWaiterHookTest(tf.test.TestCase):

  def test_not_wait_for_step_zero(self):
    with tf.Graph().as_default():
      tf.contrib.framework.get_or_create_global_step()
      hook = _GlobalStepWaiterHook(wait_until_step=0)
      hook.begin()
      with tf.Session() as sess:
        # Before run should return without waiting gstep increment.
        hook.before_run(
            tf.train.SessionRunContext(
                original_args=None, session=sess))

  def test_wait_for_step(self):
    with tf.Graph().as_default():
      gstep = tf.contrib.framework.get_or_create_global_step()
      hook = _GlobalStepWaiterHook(wait_until_step=1000)
      hook.begin()
      with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        waiter = threading.Thread(
            target=hook.before_run,
            args=(tf.train.SessionRunContext(
                original_args=None, session=sess),))
        waiter.daemon = True
        waiter.start()
        time.sleep(1.0)
        self.assertTrue(waiter.is_alive())
        sess.run(tf.assign(gstep, 500))
        time.sleep(1.0)
        self.assertTrue(waiter.is_alive())
        sess.run(tf.assign(gstep, 1100))
        time.sleep(1.2)
        self.assertFalse(waiter.is_alive())


class ExperimentTest(tf.test.TestCase):

  def setUp(self):
    # The official name is tf.train, so tf.training was obliterated.
    reveal_undocumented('tensorflow.python.training')

  def _cluster_spec(self):
    return {
        tf.contrib.learn.TaskType.PS: ['host1:2222', 'host2:2222'],
        tf.contrib.learn.TaskType.WORKER:
            ['host3:2222', 'host4:2222', 'host5:2222']
    }

  def test_train(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(
        est,
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
    ex = tf.contrib.learn.Experiment(
        est, train_input_fn='train_input', eval_input_fn='eval_input')
    for delay in [0, 1, 3]:
      start = time.time()
      ex.train(delay_secs=delay)
      duration = time.time() - start
      self.assertAlmostEqual(duration, delay, delta=0.5)

  def test_train_default_delay(self):
    for task_id in [0, 1, 3]:
      tf_config = {'task': {'index': task_id}}
      with patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}):
        config = run_config.RunConfig()
      est = TestEstimator(config)
      ex = tf.contrib.learn.Experiment(
          est, train_input_fn='train_input', eval_input_fn='eval_input')

      start = time.time()
      ex.train()
      duration = time.time() - start
      self.assertAlmostEqual(duration, task_id * 5, delta=0.5)

  @tf.test.mock.patch('tensorflow.python.training.server_lib.Server')  # pylint: disable=line-too-long
  def test_train_starts_server(self, mock_server):
    # Arrange.
    tf_config = {
        'cluster': self._cluster_spec(),
        'environment': tf.contrib.learn.Environment.CLOUD,
        'task': {
            'type': tf.contrib.learn.TaskType.WORKER,
            'index': 1
        }
    }
    with patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}):
      config = tf.contrib.learn.RunConfig(
          master='host4:2222', num_cores=15, gpu_memory_fraction=0.314)

    est = TestEstimator(config)
    ex = tf.contrib.learn.Experiment(
        est, train_input_fn='train_input', eval_input_fn='eval_input')

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
        job_name=tf.contrib.learn.TaskType.WORKER,
        task_index=1,
        config=expected_config_proto,
        start=False)
    mock_server.assert_has_calls([tf.test.mock.call().start()])

    # Ensure that the delay takes into account the time to start the server.
    self.assertAlmostEqual(duration, 1.0, delta=0.5)

  @tf.test.mock.patch('tensorflow.python.training.server_lib.Server')  # pylint: disable=line-too-long
  def test_train_server_does_not_start_without_cluster_spec(self, mock_server):
    config = tf.contrib.learn.RunConfig(master='host4:2222')
    ex = tf.contrib.learn.Experiment(
        TestEstimator(config),
        train_input_fn='train_input',
        eval_input_fn='eval_input')
    ex.train()

    # The server should not have started because there was no ClusterSpec.
    self.assertFalse(mock_server.called)

  @tf.test.mock.patch('tensorflow.python.training.server_lib.Server')  # pylint: disable=line-too-long
  def test_train_server_does_not_start_with_empty_master(self, mock_server):
    tf_config = {'cluster': self._cluster_spec()}
    with patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}):
      config = tf.contrib.learn.RunConfig(master='')
    ex = tf.contrib.learn.Experiment(
        TestEstimator(config),
        train_input_fn='train_input',
        eval_input_fn='eval_input')
    ex.train()

    # The server should not have started because master was the empty string.
    self.assertFalse(mock_server.called)

  def test_train_raises_if_job_name_is_missing(self):
    tf_config = {
        'cluster': self._cluster_spec(),
        'environment': tf.contrib.learn.Environment.CLOUD,
        'task': {
            'index': 1
        }
    }
    with patch.dict(
        'os.environ',
        {'TF_CONFIG': json.dumps(tf_config)}), self.assertRaises(ValueError):
      config = tf.contrib.learn.RunConfig(
          master='host3:2222'  # Normally selected by task type.
      )
      ex = tf.contrib.learn.Experiment(
          TestEstimator(config),
          train_input_fn='train_input',
          eval_input_fn='eval_input')
      ex.train()

  def test_evaluate(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(
        est,
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
    ex = tf.contrib.learn.Experiment(
        est, train_input_fn='train_input', eval_input_fn='eval_input')

    for delay in [0, 1, 3]:
      start = time.time()
      ex.evaluate(delay_secs=delay)
      duration = time.time() - start
      tf.logging.info('eval duration (expected %f): %f', delay, duration)
      self.assertAlmostEqual(duration, delay, delta=0.5)

  def test_continuous_eval(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(
        est,
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
    ex = tf.contrib.learn.Experiment(
        est,
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
    self.assertTrue(
        isinstance(est.monitors[0],
                   tf.contrib.learn.monitors.ValidationMonitor))

  def test_train_and_evaluate(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(
        est,
        train_input_fn='train_input',
        eval_input_fn='eval_input',
        eval_metrics='eval_metrics',
        train_steps=100,
        eval_steps=100)
    ex.train_and_evaluate()
    self.assertEquals(1, est.fit_count)
    self.assertEquals(1, est.eval_count)
    self.assertEquals(1, len(est.monitors))
    self.assertTrue(
        isinstance(est.monitors[0],
                   tf.contrib.learn.monitors.ValidationMonitor))

  @tf.test.mock.patch('tensorflow.python.training.server_lib.Server')  # pylint: disable=line-too-long
  def test_run_std_server(self, mock_server):
    # Arrange.
    tf_config = {
        'cluster': self._cluster_spec(),
        'task': {
            'type': tf.contrib.learn.TaskType.PS,
            'index': 1
        }
    }
    with patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}):
      config = tf.contrib.learn.RunConfig(
          master='host2:2222',
          num_cores=15,
          gpu_memory_fraction=0.314,)
    est = TestEstimator(config)
    ex = tf.contrib.learn.Experiment(
        est, train_input_fn='train_input', eval_input_fn='eval_input')

    # Act.
    ex.run_std_server()

    # Assert.
    mock_server.assert_has_calls(
        [tf.test.mock.call().start(), tf.test.mock.call().join()])

  @tf.test.mock.patch('tensorflow.python.training.server_lib.Server')  # pylint: disable=line-too-long
  def test_run_std_server_raises_without_cluster_spec(self, mock_server):
    config = tf.contrib.learn.RunConfig(master='host4:2222')
    with self.assertRaises(ValueError):
      ex = tf.contrib.learn.Experiment(
          TestEstimator(config),
          train_input_fn='train_input',
          eval_input_fn='eval_input')
      ex.run_std_server()

  def test_test(self):
    est = TestEstimator()
    ex = tf.contrib.learn.Experiment(
        est, train_input_fn='train_input', eval_input_fn='eval_input')
    ex.test()
    self.assertEquals(1, est.fit_count)
    self.assertEquals(1, est.eval_count)


if __name__ == '__main__':
  tf.test.main()
