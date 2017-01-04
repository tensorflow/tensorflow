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
import os
import sys
import tempfile
import threading

# TODO: #6568 Remove this hack that makes dlopen() not crash.
if hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'):
  import ctypes
  sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

from tensorflow.contrib.learn.python.learn import evaluable
from tensorflow.contrib.learn.python.learn import experiment
from tensorflow.contrib.learn.python.learn import monitors
from tensorflow.contrib.learn.python.learn import run_config
from tensorflow.contrib.learn.python.learn import trainable
from tensorflow.contrib.learn.python.learn.estimators import run_config as run_config_lib
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat
from tensorflow.python.util.all_util import reveal_undocumented


class SheepCounter(object):
  """To be patched in for time.sleep, in order to capture how long was slept."""

  def __init__(self):
    self._total_time = 0
    self._sleeptimes = []

  def __call__(self, t):
    self._total_time += t
    self._sleeptimes += [t]

  @property
  def total_time(self):
    return self._total_time

  @property
  def sleep_times(self):
    return self._sleeptimes


class TestEstimator(evaluable.Evaluable, trainable.Trainable):

  def __init__(self, config=None, max_evals=5):
    self.eval_count = 0
    self.fit_count = 0
    self._max_evals = max_evals
    self.export_count = 0
    self.monitors = []
    self._config = config or run_config.RunConfig()
    self._model_dir = tempfile.mkdtemp()

  @property
  def model_dir(self):
    return self._model_dir

  @property
  def config(self):
    return self._config

  def evaluate(self, **kwargs):
    tf_logging.info('evaluate called with args: %s' % kwargs)
    self.eval_count += 1
    if self.eval_count > self._max_evals:
      tf_logging.info('Ran %d evals. Done.' % self.eval_count)
      raise StopIteration()
    return [(key, kwargs[key]) for key in sorted(kwargs.keys())]

  def fake_checkpoint(self):
    save_path = os.path.join(self.model_dir, 'model.ckpt')
    with session.Session() as sess:
      var = variables.Variable(1.0, name='var0')
      save = saver.Saver({var.op.name: var})
      var.initializer.run()
      save.save(sess, save_path, global_step=0)

  def fit(self, **kwargs):
    self.fake_checkpoint()
    tf_logging.info('fit called with args: %s' % kwargs)
    self.fit_count += 1
    if 'monitors' in kwargs:
      self.monitors = kwargs['monitors']
    return [(key, kwargs[key]) for key in sorted(kwargs.keys())]

  def export_savedmodel(self, export_dir_base, export_input_fn, **kwargs):
    tf_logging.info('export_savedmodel called with args: %s, %s, %s' %
                    (export_dir_base, export_input_fn, kwargs))
    self.export_count += 1
    return os.path.join(
        compat.as_bytes(export_dir_base), compat.as_bytes('bogus_timestamp'))


class ExperimentTest(test.TestCase):

  def _cluster_spec(self):
    return {
        run_config_lib.TaskType.PS: ['host1:2222', 'host2:2222'],
        run_config_lib.TaskType.WORKER:
            ['host3:2222', 'host4:2222', 'host5:2222']
    }

  def test_train(self):
    est = TestEstimator()
    ex = experiment.Experiment(
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
    ex = experiment.Experiment(
        est, train_input_fn='train_input', eval_input_fn='eval_input')
    for delay in [0, 1, 3]:
      with test.mock.patch('time.sleep', SheepCounter()) as sheep:
        ex.train(delay_secs=delay)
        self.assertAlmostEqual(delay, sheep.total_time, delta=0.1)

  def test_train_default_delay(self):
    for task_id in [0, 1, 3]:
      tf_config = {'task': {'index': task_id}}
      with test.mock.patch.dict('os.environ',
                                {'TF_CONFIG': json.dumps(tf_config)}):
        config = run_config.RunConfig()
      est = TestEstimator(config)
      ex = experiment.Experiment(
          est, train_input_fn='train_input', eval_input_fn='eval_input')

      with test.mock.patch('time.sleep', SheepCounter()) as sheep:
        ex.train()
        self.assertAlmostEqual(task_id * 5, sheep.total_time, delta=0.1)

  @test.mock.patch.object(server_lib, 'Server')
  def test_train_starts_server(self, mock_server):
    # Arrange.
    tf_config = {
        'cluster': self._cluster_spec(),
        'environment': run_config_lib.Environment.CLOUD,
        'task': {
            'type': run_config_lib.TaskType.WORKER,
            'index': 1
        }
    }
    with test.mock.patch.dict('os.environ',
                              {'TF_CONFIG': json.dumps(tf_config)}):
      config = run_config_lib.RunConfig(
          master='host4:2222', num_cores=15, gpu_memory_fraction=0.314)

    est = TestEstimator(config)
    ex = experiment.Experiment(
        est, train_input_fn='train_input', eval_input_fn='eval_input')

    # Act.
    # We want to make sure we discount the time it takes to start the server
    # in our accounting of the delay, so we set a small delay here.
    with test.mock.patch('time.sleep', SheepCounter()) as sheep:
      ex.train(delay_secs=1)
      # Ensure that the delay takes into account the time to start the server.
      self.assertAlmostEqual(1, sheep.total_time, delta=0.1)

    # Assert.
    expected_config_proto = config_pb2.ConfigProto()
    expected_config_proto.inter_op_parallelism_threads = 15
    expected_config_proto.intra_op_parallelism_threads = 15
    expected_config_proto.gpu_options.per_process_gpu_memory_fraction = 0.314
    mock_server.assert_called_with(
        config.cluster_spec,
        job_name=run_config_lib.TaskType.WORKER,
        task_index=1,
        config=expected_config_proto,
        start=False)
    mock_server.assert_has_calls([test.mock.call().start()])

  @test.mock.patch.object(server_lib, 'Server')
  def test_train_server_does_not_start_without_cluster_spec(self, mock_server):
    config = run_config_lib.RunConfig(master='host4:2222')
    ex = experiment.Experiment(
        TestEstimator(config),
        train_input_fn='train_input',
        eval_input_fn='eval_input')
    ex.train()

    # The server should not have started because there was no ClusterSpec.
    self.assertFalse(mock_server.called)

  @test.mock.patch.object(server_lib, 'Server')
  def test_train_server_does_not_start_with_empty_master(self, mock_server):
    tf_config = {'cluster': self._cluster_spec()}
    with test.mock.patch.dict('os.environ',
                              {'TF_CONFIG': json.dumps(tf_config)}):
      config = run_config_lib.RunConfig(master='')
    ex = experiment.Experiment(
        TestEstimator(config),
        train_input_fn='train_input',
        eval_input_fn='eval_input')
    ex.train()

    # The server should not have started because master was the empty string.
    self.assertFalse(mock_server.called)

  def test_train_raises_if_job_name_is_missing(self):
    tf_config = {
        'cluster': self._cluster_spec(),
        'environment': run_config_lib.Environment.CLOUD,
        'task': {
            'index': 1
        }
    }
    with test.mock.patch.dict(
        'os.environ',
        {'TF_CONFIG': json.dumps(tf_config)}), self.assertRaises(ValueError):
      config = run_config_lib.RunConfig(
          master='host3:2222'  # Normally selected by task type.
      )
      ex = experiment.Experiment(
          TestEstimator(config),
          train_input_fn='train_input',
          eval_input_fn='eval_input')
      ex.train()

  def test_evaluate(self):
    est = TestEstimator()
    est.fake_checkpoint()
    ex = experiment.Experiment(
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
    est.fake_checkpoint()
    ex = experiment.Experiment(
        est, train_input_fn='train_input', eval_input_fn='eval_input')

    for delay in [0, 1, 3]:
      with test.mock.patch('time.sleep', SheepCounter()) as sheep:
        ex.evaluate(delay_secs=delay)
      self.assertAlmostEqual(delay, sheep.total_time, delta=0.1)

  def test_continuous_eval(self):
    est = TestEstimator()
    est.fake_checkpoint()
    ex = experiment.Experiment(
        est,
        train_input_fn='train_input',
        eval_input_fn='eval_input',
        eval_metrics='eval_metrics',
        eval_delay_secs=0,
        continuous_eval_throttle_secs=0)
    self.assertRaises(
        StopIteration, ex.continuous_eval, evaluate_checkpoint_only_once=False)
    self.assertEquals(6, est.eval_count)
    self.assertEquals(0, est.fit_count)

  def test_continuous_eval_throttle_delay(self):
    for delay in [0, 1, 2]:
      est = TestEstimator()
      est.fake_checkpoint()
      ex = experiment.Experiment(
          est,
          train_input_fn='train_input',
          eval_input_fn='eval_input',
          eval_metrics='eval_metrics',
          continuous_eval_throttle_secs=delay,
          eval_delay_secs=0)
      with test.mock.patch('time.sleep', SheepCounter()) as sheep:
        self.assertRaises(
            StopIteration,
            ex.continuous_eval,
            evaluate_checkpoint_only_once=False)
        self.assertAlmostEqual(5 * delay, sheep.total_time, delta=0.1)

  def test_run_local(self):
    est = TestEstimator()
    ex = experiment.Experiment(
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
    self.assertTrue(isinstance(est.monitors[0], monitors.ValidationMonitor))

  def test_train_and_evaluate(self):
    est = TestEstimator()
    export_strategy = saved_model_export_utils.make_export_strategy(
        est, 'export_input')
    ex = experiment.Experiment(
        est,
        train_input_fn='train_input',
        eval_input_fn='eval_input',
        eval_metrics='eval_metrics',
        train_steps=100,
        eval_steps=100,
        export_strategies=export_strategy)
    ex.train_and_evaluate()
    self.assertEquals(1, est.fit_count)
    self.assertEquals(1, est.eval_count)
    self.assertEquals(1, est.export_count)
    self.assertEquals(1, len(est.monitors))
    self.assertTrue(isinstance(est.monitors[0], monitors.ValidationMonitor))

  @test.mock.patch.object(server_lib, 'Server')
  def test_run_std_server(self, mock_server):
    # Arrange.
    tf_config = {
        'cluster': self._cluster_spec(),
        'task': {
            'type': run_config_lib.TaskType.PS,
            'index': 1
        }
    }
    with test.mock.patch.dict('os.environ',
                              {'TF_CONFIG': json.dumps(tf_config)}):
      config = run_config_lib.RunConfig(
          master='host2:2222',
          num_cores=15,
          gpu_memory_fraction=0.314,)
    est = TestEstimator(config)
    ex = experiment.Experiment(
        est, train_input_fn='train_input', eval_input_fn='eval_input')

    # Act.
    ex.run_std_server()

    # Assert.
    mock_server.assert_has_calls(
        [test.mock.call().start(), test.mock.call().join()])

  @test.mock.patch.object(server_lib, 'Server')
  def test_run_std_server_raises_without_cluster_spec(self, mock_server):
    config = run_config_lib.RunConfig(master='host4:2222')
    with self.assertRaises(ValueError):
      ex = experiment.Experiment(
          TestEstimator(config),
          train_input_fn='train_input',
          eval_input_fn='eval_input')
      ex.run_std_server()

  def test_test(self):
    est = TestEstimator()
    ex = experiment.Experiment(
        est, train_input_fn='train_input', eval_input_fn='eval_input')
    ex.test()
    self.assertEquals(1, est.fit_count)
    self.assertEquals(1, est.eval_count)

  def test_continuous_eval_evaluates_checkpoint_once(self):
    # Temporarily disabled until we figure out the threading story on Jenkins.
    return
    # pylint: disable=unreachable

    # The TestEstimator will raise StopIteration the second time evaluate is
    # called.
    ex = experiment.Experiment(
        TestEstimator(max_evals=1),
        train_input_fn='train_input',
        eval_input_fn='eval_input')

    # This should not happen if the logic restricting evaluation of the same
    # checkpoint works. We do need some checkpoint though, otherwise Experiment
    # will never evaluate.
    ex.estimator.fake_checkpoint()

    # Start a separate thread with continuous eval
    thread = threading.Thread(
        target=lambda: ex.continuous_eval(delay_secs=0, throttle_delay_secs=0))
    thread.start()

    # The thread will die if it evaluates twice, and we should never evaluate
    # twice since we don't write another checkpoint. Since we did not enable
    # throttling, if it hasn't died after two seconds, we're good.
    thread.join(2)
    self.assertTrue(thread.is_alive())

    # But we should have evaluated once.
    count = ex.estimator.eval_count
    self.assertEquals(1, count)


if __name__ == '__main__':
  test.main()
