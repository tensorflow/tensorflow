# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for training.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import json
import time

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.estimator import training
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_run_hook

_DEFAULT_EVAL_STEPS = 100
_DEFAULT_EVAL_DELAY_SECS = 120
_DEFAULT_EVAL_THROTTLE_SECS = 600
_DELAY_SECS_PER_WORKER = 5
_GLOBAL_STEP_KEY = ops.GraphKeys.GLOBAL_STEP
_INVALID_INPUT_FN_MSG = '`input_fn` must be callable'
_INVALID_HOOK_MSG = 'All hooks must be `SessionRunHook` instances'
_INVALID_MAX_STEPS_MSG = 'Must specify max_steps > 0'
_INVALID_STEPS_MSG = 'Must specify steps > 0'
_INVALID_NAME_MSG = '`name` must be string'
_INVALID_EVAL_DELAY_SECS_MSG = 'Must specify delay_secs >= 0'
_INVALID_EVAL_THROTTLE_SECS_MSG = 'Must specify throttle_secs >= 0'
_INVALID_ESTIMATOR_MSG = '`estimator` must have type `tf.estimator.Estimator`'
_INVALID_TRAIN_SPEC_MSG = '`train_spec` must have type `tf.estimator.TrainSpec`'
_INVALID_EVAL_SPEC_MSG = '`eval_spec` must have type `tf.estimator.EvalSpec`'
_INVALID_CONFIG_FOR_STD_SERVER_MSG = 'Could not start server; .*TF_CONFIG'

_TF_CONFIG_FOR_CHIEF = {
    'cluster': {
        run_config_lib.TaskType.CHIEF: ['host0:0'],
        run_config_lib.TaskType.PS: ['host1:1', 'host2:2'],
        run_config_lib.TaskType.WORKER: ['host3:3', 'host4:4']
    },
    'task': {
        'type': run_config_lib.TaskType.CHIEF,
        'index': 0
    }
}

_TF_CONFIG_FOR_WORKER = {
    'cluster': {
        run_config_lib.TaskType.CHIEF: ['host0:0'],
        run_config_lib.TaskType.PS: ['host1:1', 'host2:2'],
        run_config_lib.TaskType.WORKER: ['host3:3', 'host4:4']
    },
    'task': {
        'type': run_config_lib.TaskType.WORKER,
        'index': 1
    }
}

_TF_CONFIG_FOR_PS = {
    'cluster': {
        run_config_lib.TaskType.CHIEF: ['host0:0'],
        run_config_lib.TaskType.PS: ['host1:1', 'host2:2'],
        run_config_lib.TaskType.WORKER: ['host3:3', 'host4:4']
    },
    'task': {
        'type': run_config_lib.TaskType.PS,
        'index': 1
    }
}

_TF_CONFIG_FOR_GOOGLE = {'environment': 'google'}


class _FakeHook(session_run_hook.SessionRunHook):
  """Fake implementation of `SessionRunHook`."""


class _InvalidHook(object):
  """Invalid hook (not a subclass of `SessionRunHook`)."""


def _create_run_config_with_cluster_spec(tf_config):
  with test.mock.patch.dict('os.environ', {'TF_CONFIG': json.dumps(tf_config)}):
    return run_config_lib.RunConfig()


class TrainSpecTest(test.TestCase):
  """Tests TrainSpec."""

  def testRequiredArgumentsSet(self):
    """Tests that no errors are raised when all required arguments are set."""
    spec = training.TrainSpec(input_fn=lambda: 1)
    self.assertEqual(1, spec.input_fn())
    self.assertIsNone(spec.max_steps)
    self.assertEqual(0, len(spec.hooks))

  def testAllArgumentsSet(self):
    """Tests that no errors are raised when all arguments are set."""
    hooks = [_FakeHook()]
    spec = training.TrainSpec(input_fn=lambda: 1, max_steps=2, hooks=hooks)
    self.assertEqual(1, spec.input_fn())
    self.assertEqual(2, spec.max_steps)
    self.assertEqual(tuple(hooks), spec.hooks)

  def testInvalidInputFn(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_INPUT_FN_MSG):
      training.TrainSpec(input_fn='invalid')

  def testInvalidMaxStep(self):
    with self.assertRaisesRegexp(ValueError, _INVALID_MAX_STEPS_MSG):
      training.TrainSpec(input_fn=lambda: 1, max_steps=0)

  def testInvalidHook(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_HOOK_MSG):
      training.TrainSpec(input_fn=lambda: 1, hooks=[_InvalidHook()])


class EvalSpecTest(test.TestCase):
  """Tests EvalSpec."""

  def testRequiredArgumentsSet(self):
    """Tests that no errors are raised when all required arguments are set."""
    spec = training.EvalSpec(input_fn=lambda: 1)
    self.assertEqual(1, spec.input_fn())
    self.assertEqual(_DEFAULT_EVAL_STEPS, spec.steps)
    self.assertIsNone(spec.name)
    self.assertEqual(0, len(spec.hooks))
    self.assertEqual(0, len(spec.export_strategies))
    self.assertEqual(_DEFAULT_EVAL_DELAY_SECS, spec.delay_secs)
    self.assertEqual(_DEFAULT_EVAL_THROTTLE_SECS, spec.throttle_secs)

  def testAllArgumentsSet(self):
    """Tests that no errors are raised when all arguments are set."""
    hooks = [_FakeHook()]

    # TODO(b/65169058): Replace the export_strategies with valid instances.
    spec = training.EvalSpec(input_fn=lambda: 1, steps=2, name='name',
                             hooks=hooks, export_strategies=hooks,
                             delay_secs=3, throttle_secs=4)
    self.assertEqual(1, spec.input_fn())
    self.assertEqual(2, spec.steps)
    self.assertEqual('name', spec.name)
    self.assertEqual(tuple(hooks), spec.hooks)
    self.assertEqual(tuple(hooks), spec.export_strategies)
    self.assertEqual(3, spec.delay_secs)
    self.assertEqual(4, spec.throttle_secs)

  def testInvalidInputFn(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_INPUT_FN_MSG):
      training.EvalSpec(input_fn='invalid')

  def testInvalidMaxStep(self):
    with self.assertRaisesRegexp(ValueError, _INVALID_STEPS_MSG):
      training.EvalSpec(input_fn=lambda: 1, steps=0)

  def testInvalidName(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_NAME_MSG):
      training.EvalSpec(input_fn=lambda: 1, name=123)

  def testInvalidHook(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_HOOK_MSG):
      training.EvalSpec(input_fn=lambda: 1, hooks=[_InvalidHook()])

  def testInvalidDelaySecs(self):
    with self.assertRaisesRegexp(ValueError, _INVALID_EVAL_DELAY_SECS_MSG):
      training.EvalSpec(input_fn=lambda: 1, delay_secs=-1)

  def testInvalidThrottleSecs(self):
    with self.assertRaisesRegexp(ValueError, _INVALID_EVAL_THROTTLE_SECS_MSG):
      training.EvalSpec(input_fn=lambda: 1, throttle_secs=-1)


class TrainingExecutorConstructorTest(test.TestCase):
  """Tests constructor of _TrainingExecutor."""

  def testRequiredArgumentsSet(self):
    estimator = estimator_lib.Estimator(model_fn=lambda features: features)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = training.EvalSpec(input_fn=lambda: 1)

    executor = training._TrainingExecutor(estimator, train_spec, eval_spec)
    self.assertEqual(estimator, executor.estimator)

  def test_invalid_estimator(self):
    invalid_estimator = object()
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = training.EvalSpec(input_fn=lambda: 1)

    with self.assertRaisesRegexp(TypeError, _INVALID_ESTIMATOR_MSG):
      training._TrainingExecutor(invalid_estimator, train_spec, eval_spec)

  def test_invalid_train_spec(self):
    estimator = estimator_lib.Estimator(model_fn=lambda features: features)
    invalid_train_spec = object()
    eval_spec = training.EvalSpec(input_fn=lambda: 1)

    with self.assertRaisesRegexp(TypeError, _INVALID_TRAIN_SPEC_MSG):
      training._TrainingExecutor(estimator, invalid_train_spec, eval_spec)

  def test_invalid_eval_spec(self):
    estimator = estimator_lib.Estimator(model_fn=lambda features: features)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    invalid_eval_spec = object()

    with self.assertRaisesRegexp(TypeError, _INVALID_EVAL_SPEC_MSG):
      training._TrainingExecutor(estimator, train_spec, invalid_eval_spec)


class _TrainingExecutorTrainingTest(object):
  """Tests training of _TrainingExecutor."""

  def __init__(self, run_config):
    self._run_config = run_config

  def _run_task(self, executor):
    return getattr(executor, 'run_' + self._run_config.task_type)()

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_train_with_train_spec(self, mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = self._run_config
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=2, hooks=[_FakeHook()])
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)
    mock_server_instance = mock_server.return_value

    executor = training._TrainingExecutor(mock_est, train_spec, mock_eval_spec)
    self._run_task(executor)

    mock_server.assert_called_with(
        mock_est.config.cluster_spec,
        job_name=mock_est.config.task_type,
        task_index=mock_est.config.task_id,
        config=test.mock.ANY,
        start=False)

    self.assertTrue(mock_server_instance.start.called)

    mock_est.train.assert_called_with(input_fn=train_spec.input_fn,
                                      max_steps=train_spec.max_steps,
                                      hooks=train_spec.hooks)
    mock_est.evaluate.assert_not_called()
    mock_est.export_savedmodel.assert_not_called()

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_no_server_startup_in_google(self, mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = self._run_config
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)
    tf_config = {'TF_CONFIG': json.dumps(_TF_CONFIG_FOR_GOOGLE)}
    with test.mock.patch.dict('os.environ', tf_config):
      self._run_task(executor)
      mock_server.assert_not_called()

  def test_fail_with_empty_cluster_spec(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = None
    mock_est.config.master = 'grpc://...'
    mock_est.config.task_type = 'worker'
    mock_est.config.task_id = 2

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      self._run_task(training._TrainingExecutor(mock_est, mock_train_spec,
                                                mock_eval_spec))

  def test_fail_with_empty_master(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = {'worker': 'dummy'}
    mock_est.config.master = ''
    mock_est.config.task_type = 'worker'
    mock_est.config.task_id = 2

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      self._run_task(training._TrainingExecutor(mock_est, mock_train_spec,
                                                mock_eval_spec))

  def test_fail_with_empty_task_type(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = {'worker': 'dummy'}
    mock_est.config.master = 'grpc://...'
    mock_est.config.task_type = ''
    mock_est.config.task_id = 2

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      self._run_task(training._TrainingExecutor(mock_est, mock_train_spec,
                                                mock_eval_spec))

  def test_fail_with_none_task_id(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = {'worker': 'dummy'}
    mock_est.config.master = 'grpc://...'
    mock_est.config.task_type = 'worker'
    mock_est.config.task_id = None

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      self._run_task(training._TrainingExecutor(mock_est, mock_train_spec,
                                                mock_eval_spec))


class TrainingExecutorRunWorkerTest(_TrainingExecutorTrainingTest,
                                    test.TestCase):
  """Tests run_worker of _TrainingExecutor."""

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    _TrainingExecutorTrainingTest.__init__(
        self,
        run_config=_create_run_config_with_cluster_spec(_TF_CONFIG_FOR_WORKER))

  @test.mock.patch.object(server_lib, 'Server')
  def test_delay_for_worker(self, _):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = self._run_config
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)

    expected_secs = (self._run_config.task_id + 1) * _DELAY_SECS_PER_WORKER
    with test.mock.patch.object(time, 'sleep') as mock_sleep:
      mock_sleep.side_effect = lambda s: self.assertEqual(expected_secs, s)
      self._run_task(executor)
      self.assertTrue(mock_sleep.called)


class TrainingExecutorRunChiefTest(_TrainingExecutorTrainingTest,
                                   test.TestCase):
  """Tests run_chief of _TrainingExecutor."""

  def __init__(self, methodName='runTest'):  # pylint: disable=invalid-name
    test.TestCase.__init__(self, methodName)
    _TrainingExecutorTrainingTest.__init__(
        self,
        run_config=_create_run_config_with_cluster_spec(_TF_CONFIG_FOR_CHIEF))

  @test.mock.patch.object(server_lib, 'Server')
  def test_no_delay_for_chief(self, _):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = self._run_config
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)

    with test.mock.patch.object(time, 'sleep') as mock_sleep:
      self._run_task(executor)
      mock_sleep.assert_not_called()


class TrainingExecutorRunEvaluatorTest(test.TestCase):
  """Tests run_evaluator of _TrainingExecutor."""

  def _set_up_mock_est_to_train_and_evaluate_once(self, mock_est,
                                                  mock_train_spec):
    """Sets global step in eval result to end the while True eval loop."""
    training_max_step = 200
    mock_est.evaluate.return_value = {_GLOBAL_STEP_KEY: training_max_step}
    mock_train_spec.max_steps = training_max_step

  @test.mock.patch.object(saver, 'latest_checkpoint')
  def test_evaluate_with_evaluate_spec(self, mock_latest_ckpt):
    latest_ckpt_path = mock_latest_ckpt.return_value

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    self._set_up_mock_est_to_train_and_evaluate_once(mock_est, mock_train_spec)

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, steps=2, hooks=[_FakeHook()], name='cont_eval',
        delay_secs=0, throttle_secs=0)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    executor.run_evaluator()

    mock_est.evaluate.assert_called_with(
        name='cont_eval',
        input_fn=eval_spec.input_fn,
        steps=eval_spec.steps,
        checkpoint_path=latest_ckpt_path,
        hooks=eval_spec.hooks)
    self.assertFalse(mock_est.train.called)

  @test.mock.patch.object(saver, 'latest_checkpoint')
  def test_evaluate_multiple_times(self, mock_latest_ckpt):
    training_max_step = 200

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate.side_effect = [
        {_GLOBAL_STEP_KEY: training_max_step // 2},
        {_GLOBAL_STEP_KEY: training_max_step}
    ]
    mock_latest_ckpt.side_effect = ['path_1', 'path_2']

    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_train_spec.max_steps = training_max_step

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, delay_secs=0, throttle_secs=0)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    executor.run_evaluator()
    self.assertEqual(2, mock_est.evaluate.call_count)

  @test.mock.patch.object(saver, 'latest_checkpoint')
  def test_skip_evaluation_due_to_ckpt(self, mock_latest_ckpt):
    training_max_step = 200
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate.side_effect = [
        {_GLOBAL_STEP_KEY: training_max_step // 2},
        {_GLOBAL_STEP_KEY: training_max_step}
    ]
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_train_spec.max_steps = training_max_step

    self._set_up_mock_est_to_train_and_evaluate_once(mock_est, mock_train_spec)

    # First two items are invalid, next two items are same.
    mock_latest_ckpt.side_effect = [None, '', 'same', 'same', 'path_2']

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, delay_secs=0, throttle_secs=0)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    with test.mock.patch.object(logging, 'warning') as mock_log:
      executor.run_evaluator()

    # Three checkpoint paths are invalid.
    self.assertEqual(5, mock_latest_ckpt.call_count)
    self.assertEqual(2, mock_est.evaluate.call_count)

    # Two warning logs are expected (last warning time is reset after a
    # successuful evaluation)
    self.assertEqual(2, mock_log.call_count)

  @test.mock.patch.object(saver, 'latest_checkpoint')
  def test_sleep_delay_secs(self, mock_latest_ckpt):
    training_max_step = 200
    delay_secs = 123

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate.return_value = {_GLOBAL_STEP_KEY: training_max_step}
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_train_spec.max_steps = training_max_step

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, steps=2, hooks=[_FakeHook()], name='cont_eval',
        delay_secs=delay_secs, throttle_secs=0)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    with test.mock.patch.object(time, 'sleep') as mock_sleep:
      executor.run_evaluator()
      mock_sleep.assert_called_with(delay_secs)
      self.assertTrue(mock_est.evaluate.called)

  @test.mock.patch.object(time, 'time')
  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(saver, 'latest_checkpoint')
  def test_throttle_secs(self, mock_latest_ckpt, mock_sleep, mock_time):
    throttle_secs = 123
    operation_secs = 12

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    self._set_up_mock_est_to_train_and_evaluate_once(mock_est, mock_train_spec)

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, delay_secs=0, throttle_secs=throttle_secs)

    mock_time.side_effect = [921, 921 + operation_secs]

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    # Disable logging as it calls time.time also.
    with test.mock.patch.object(logging, 'info'):
      executor.run_evaluator()
    mock_sleep.assert_called_with(throttle_secs - operation_secs)
    self.assertTrue(mock_est.evaluate.called)


class TrainingExecutorRunPsTest(test.TestCase):
  """Tests run_ps of _TrainingExecutor."""

  @test.mock.patch.object(server_lib, 'Server')
  def test_std_server(self, mock_server):
    mock_server_instance = test.mock.Mock()
    mock_server.return_value = mock_server_instance

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = _create_run_config_with_cluster_spec(_TF_CONFIG_FOR_PS)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)
    executor.run_ps()

    mock_server.assert_called_with(
        mock_est.config.cluster_spec,
        job_name=mock_est.config.task_type,
        task_index=mock_est.config.task_id,
        config=test.mock.ANY,
        start=False)

    self.assertTrue(mock_server_instance.start.called)
    self.assertTrue(mock_server_instance.join.called)

  def test_fail_with_empty_cluster_spec(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = None
    mock_est.config.master = 'grpc://...'
    mock_est.config.task_type = 'gs'
    mock_est.config.task_id = 2

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      training._TrainingExecutor(mock_est, mock_train_spec,
                                 mock_eval_spec).run_ps()

  def test_fail_with_empty_master(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = {'gs': 'dummy'}
    mock_est.config.master = ''
    mock_est.config.task_type = 'gs'
    mock_est.config.task_id = 2

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      training._TrainingExecutor(mock_est, mock_train_spec,
                                 mock_eval_spec).run_ps()

  def test_fail_with_empty_task_type(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = {'gs': 'dummy'}
    mock_est.config.master = 'grpc://...'
    mock_est.config.task_type = ''
    mock_est.config.task_id = 2

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      training._TrainingExecutor(mock_est, mock_train_spec,
                                 mock_eval_spec).run_ps()

  def test_fail_with_none_task_id(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = {'gs': 'dummy'}
    mock_est.config.master = 'grpc://...'
    mock_est.config.task_type = 'gs'
    mock_est.config.task_id = None

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      training._TrainingExecutor(mock_est, mock_train_spec,
                                 mock_eval_spec).run_ps()


if __name__ == '__main__':
  test.main()
