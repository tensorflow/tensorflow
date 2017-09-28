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
from tensorflow.python.estimator import export_strategy as export_strategy_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.estimator import training
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import compat

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
_INVALID_EXPORT_STRATEGY_MSG = '`export_strategies` must be an ExportStrategy'
_INVALID_TRAIN_SPEC_MSG = '`train_spec` must have type `tf.estimator.TrainSpec`'
_INVALID_EVAL_SPEC_MSG = '`eval_spec` must have type `tf.estimator.EvalSpec`'
_INVALID_CONFIG_FOR_STD_SERVER_MSG = 'Could not start server; .*TF_CONFIG'
_INVALID_LOCAL_TASK_WITH_CLUSTER = '`task.type` in TF_CONFIG cannot be `local`'
_INVALID_TASK_TYPE = '`estimator.config` must have task_type set.'
# The message should NOT have 'local' word as part of it. As (?!word) is looking
# ahead, so, the $ (ending) check is required; otherwise, it will match
# partially and return successuful.
_INVALID_TASK_TO_RUN = (
    'Task type .* is not supported. Supported task types are ((?!local).)*$')

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

_TF_CONFIG_FOR_EVALUATOR = {
    'cluster': {
        run_config_lib.TaskType.CHIEF: ['host0:0'],
        run_config_lib.TaskType.PS: ['host1:1', 'host2:2'],
        run_config_lib.TaskType.WORKER: ['host3:3', 'host4:4']
    },
    'task': {
        'type': run_config_lib.TaskType.EVALUATOR,
        'index': 1
    }
}

_TF_CONFIG_FOR_GOOGLE = {'environment': 'google'}


class _FakeHook(session_run_hook.SessionRunHook):
  """Fake implementation of `SessionRunHook`."""


class _InvalidHook(object):
  """Invalid hook (not a subclass of `SessionRunHook`)."""


def _create_fake_export_strategy():
  def export_fn(estimator, export_path):
    del estimator, export_path

  return export_strategy_lib.ExportStrategy(name='fake_export_strategy',
                                            export_fn=export_fn)


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
    export_strategy = _create_fake_export_strategy()

    spec = training.EvalSpec(input_fn=lambda: 1, steps=2, name='name',
                             hooks=hooks, export_strategies=export_strategy,
                             delay_secs=3, throttle_secs=4)
    self.assertEqual(1, spec.input_fn())
    self.assertEqual(2, spec.steps)
    self.assertEqual('name', spec.name)
    self.assertEqual(tuple(hooks), spec.hooks)
    self.assertEqual((export_strategy,), spec.export_strategies)
    self.assertEqual(3, spec.delay_secs)
    self.assertEqual(4, spec.throttle_secs)

  def testListOfExportStrategies(self):
    """Tests that no errors are raised with multiple export strategies."""
    export_strategies = [_create_fake_export_strategy(),
                         _create_fake_export_strategy()]

    spec = training.EvalSpec(input_fn=lambda: 1,
                             export_strategies=export_strategies)
    self.assertEqual(1, spec.input_fn())
    self.assertEqual(tuple(export_strategies), spec.export_strategies)

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

  def testInvalidTypeOfListOfExportStrategies(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_EXPORT_STRATEGY_MSG):
      training.EvalSpec(input_fn=lambda: 1,
                        export_strategies=[_create_fake_export_strategy(),
                                           _FakeHook()])

  def testInvalidTypeOfIndividualExportStrategy(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_EXPORT_STRATEGY_MSG):
      training.EvalSpec(input_fn=lambda: 1, export_strategies=_FakeHook())


class TrainAndEvaluteTest(test.TestCase):

  def _mock_executor_instance(self):
    def task_fn(name):
      def _fn():
        return name
      return _fn

    mock_instance = test.mock.Mock()
    mock_instance.run_chief = task_fn('chief')
    mock_instance.run_master = task_fn('master')
    mock_instance.run_ps = task_fn('ps')
    mock_instance.run_evaluator = task_fn('evaluator')
    mock_instance.run_worker = task_fn('worker')
    mock_instance.run_local = task_fn('local')

    return mock_instance

  def _test_run_task_in_distributed_training(self, run_config):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = run_config
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    with test.mock.patch.object(training, '_TrainingExecutor') as mock_executor:
      mock_executor.return_value = self._mock_executor_instance()
      return_value = training.train_and_evaluate(
          mock_est, mock_train_spec, mock_eval_spec)

      self.assertEqual(mock_est.config.task_type, return_value)
      mock_executor.assert_called_with(estimator=mock_est,
                                       train_spec=mock_train_spec,
                                       eval_spec=mock_eval_spec)

  def test_run_chief(self):
    self._test_run_task_in_distributed_training(
        run_config=_create_run_config_with_cluster_spec(_TF_CONFIG_FOR_CHIEF))

  def test_run_worker(self):
    self._test_run_task_in_distributed_training(
        run_config=_create_run_config_with_cluster_spec(_TF_CONFIG_FOR_WORKER))

  def test_run_ps(self):
    self._test_run_task_in_distributed_training(
        run_config=_create_run_config_with_cluster_spec(_TF_CONFIG_FOR_PS))

  def test_run_evaluator(self):
    self._test_run_task_in_distributed_training(
        run_config=_create_run_config_with_cluster_spec(
            _TF_CONFIG_FOR_EVALUATOR))

  def test_run_local(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = run_config_lib.RunConfig()
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    with test.mock.patch.object(training, '_TrainingExecutor') as mock_executor:
      mock_executor.return_value = self._mock_executor_instance()
      return_value = training.train_and_evaluate(
          mock_est, mock_train_spec, mock_eval_spec)

      self.assertEqual('local', return_value)
      mock_executor.assert_called_with(estimator=mock_est,
                                       train_spec=mock_train_spec,
                                       eval_spec=mock_eval_spec)

  def test_invalid_local_task(self):
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host0:0'],
            'local': ['hos1:1'],
        },
        'task': {
            'type': 'local',
            'index': 0
        }
    }
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = _create_run_config_with_cluster_spec(tf_config)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    with self.assertRaisesRegexp(ValueError, _INVALID_LOCAL_TASK_WITH_CLUSTER):
      training.train_and_evaluate(mock_est, mock_train_spec, mock_eval_spec)

  def test_unsupported_task_due_to_missing_run_task(self):
    unsupported_task = 'alloc'
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host0:0'],
            unsupported_task: ['hos1:1'],
        },
        'task': {
            'type': unsupported_task,
            'index': 0
        }
    }
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = _create_run_config_with_cluster_spec(tf_config)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    with test.mock.patch.object(training, '_TrainingExecutor') as mock_executor:
      # mock_instance has no run_alloc method.
      mock_instance = self._mock_executor_instance()
      mock_executor.return_value = mock_instance
      with self.assertRaisesRegexp(ValueError, _INVALID_TASK_TO_RUN):
        training.train_and_evaluate(mock_est, mock_train_spec, mock_eval_spec)

  def test_unsupported_task_due_to_not_callable(self):
    unsupported_task = 'alloc'
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host0:0'],
            unsupported_task: ['hos1:1'],
        },
        'task': {
            'type': unsupported_task,
            'index': 0
        }
    }
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = _create_run_config_with_cluster_spec(tf_config)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    with test.mock.patch.object(training, '_TrainingExecutor') as mock_executor:
      mock_instance = self._mock_executor_instance()
      mock_instance.run_alloc = 123  # not callable
      mock_executor.return_value = mock_instance
      with self.assertRaisesRegexp(ValueError, _INVALID_TASK_TO_RUN):
        training.train_and_evaluate(mock_est, mock_train_spec, mock_eval_spec)

  def test_invalid_estimator(self):
    invalid_estimator = object()
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    with self.assertRaisesRegexp(TypeError, _INVALID_ESTIMATOR_MSG):
      training.train_and_evaluate(invalid_estimator, mock_train_spec,
                                  mock_eval_spec)

  def test_invalid_task_type(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = test.mock.Mock()
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.Mock()
    mock_est.config.cluster_spec = {'1': 'dummy'}
    mock_est.config.task_type = ''

    with self.assertRaisesRegexp(ValueError, _INVALID_TASK_TYPE):
      training.train_and_evaluate(mock_est, mock_train_spec, mock_eval_spec)


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

  def test_evaluate_with_evaluate_spec(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.latest_checkpoint.return_value = 'latest_it_is'
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
        checkpoint_path='latest_it_is',
        hooks=eval_spec.hooks)
    self.assertFalse(mock_est.train.called)

  def test_evaluate_multiple_times(self):
    training_max_step = 200

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.model_dir = compat.as_bytes(test.get_temp_dir())
    mock_est.evaluate.side_effect = [
        {_GLOBAL_STEP_KEY: training_max_step // 2},
        {_GLOBAL_STEP_KEY: training_max_step}
    ]
    mock_est.latest_checkpoint.side_effect = ['path_1', 'path_2']

    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_train_spec.max_steps = training_max_step

    mock_est.times_export_fn_was_called = 0
    def export_fn(estimator, *args, **kwargs):
      del args, kwargs
      estimator.times_export_fn_was_called += 1

    export_strategy = export_strategy_lib.ExportStrategy(
        name='see_whether_export_fn_is_called', export_fn=export_fn)

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1,
        delay_secs=0,
        throttle_secs=0,
        export_strategies=export_strategy)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    executor.run_evaluator()

    self.assertEqual(2, mock_est.evaluate.call_count)
    self.assertEqual(2, mock_est.times_export_fn_was_called)

  def test_skip_evaluation_due_to_ckpt(self):
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
    mock_est.latest_checkpoint.side_effect = [
        None, '', 'same', 'same', 'path_2'
    ]

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, delay_secs=0, throttle_secs=0)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    with test.mock.patch.object(logging, 'warning') as mock_log:
      executor.run_evaluator()

    # Three checkpoint paths are invalid.
    self.assertEqual(5, mock_est.latest_checkpoint.call_count)
    self.assertEqual(2, mock_est.evaluate.call_count)

    # Two warning logs are expected (last warning time is reset after a
    # successuful evaluation)
    self.assertEqual(2, mock_log.call_count)

  def test_sleep_delay_secs(self):
    training_max_step = 200
    delay_secs = 123

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate.return_value = {_GLOBAL_STEP_KEY: training_max_step}
    mock_est.model_dir = compat.as_bytes(test.get_temp_dir())
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
  def test_throttle_secs(self, mock_sleep, mock_time):
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

  @test.mock.patch.object(saver, 'latest_checkpoint')
  def test_that_export_fn_is_called(self, mock_latest_ckpt):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    self._set_up_mock_est_to_train_and_evaluate_once(mock_est, mock_train_spec)

    def export_fn(estimator, *args, **kwargs):
      del args, kwargs
      estimator.export_fn_was_called = True

    export_strategy = export_strategy_lib.ExportStrategy(
        name='see_whether_export_fn_is_called', export_fn=export_fn)

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1,
        steps=2,
        delay_secs=0,
        throttle_secs=0,
        export_strategies=export_strategy)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    executor.run_evaluator()

    # Verify that export_fn was called on the right estimator.
    self.assertTrue(mock_est.export_fn_was_called)


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


class StopAtSecsHookTest(test.TestCase):
  """Tests StopAtSecsHook."""

  @test.mock.patch.object(time, 'time')
  def test_stops_after_time(self, mock_time):
    mock_time.return_value = 1484695987.209386
    hook = training._StopAtSecsHook(1000)
    with ops.Graph().as_default():
      no_op = control_flow_ops.no_op()
      # some time passed before training starts
      mock_time.return_value += 250
      with monitored_session.MonitoredSession(hooks=[hook]) as sess:
        self.assertFalse(sess.should_stop())
        sess.run(no_op)
        self.assertFalse(sess.should_stop())
        mock_time.return_value += 500
        sess.run(no_op)
        self.assertFalse(sess.should_stop())
        mock_time.return_value += 400
        sess.run(no_op)
        self.assertFalse(sess.should_stop())
        mock_time.return_value += 200
        sess.run(no_op)
        self.assertTrue(sess.should_stop())


class TrainingExecutorRunLocalTest(test.TestCase):
  """Tests run_local of _TrainingExecutor."""

  def test_send_stop_at_secs_to_train(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=2, hooks=[_FakeHook()])
    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, hooks=[_FakeHook()], throttle_secs=100)
    mock_est.evaluate.return_value = {_GLOBAL_STEP_KEY: train_spec.max_steps}

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    executor.run_local()

    stop_hook = mock_est.train.call_args[1]['hooks'][-1]
    self.assertIsInstance(stop_hook, training._StopAtSecsHook)
    self.assertEqual(eval_spec.throttle_secs, stop_hook._stop_after_secs)

  def test_runs_in_a_loop_until_max_steps(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=300, hooks=[_FakeHook()])
    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, hooks=[_FakeHook()], throttle_secs=100)
    # should be called 3 times.
    mock_est.evaluate.side_effect = [{
        _GLOBAL_STEP_KEY: train_spec.max_steps - 100
    }, {
        _GLOBAL_STEP_KEY: train_spec.max_steps - 50
    }, {
        _GLOBAL_STEP_KEY: train_spec.max_steps
    }]

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    executor.run_local()

    self.assertEqual(3, mock_est.train.call_count)
    self.assertEqual(3, mock_est.evaluate.call_count)

  def test_train_and_evaluate_args(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=300, hooks=[_FakeHook()])
    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, steps=2, hooks=[_FakeHook()], name='local_eval')
    mock_est.evaluate.return_value = {_GLOBAL_STEP_KEY: train_spec.max_steps}

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    executor.run_local()

    mock_est.evaluate.assert_called_with(
        name=eval_spec.name,
        input_fn=eval_spec.input_fn,
        steps=eval_spec.steps,
        hooks=eval_spec.hooks)

    train_args = mock_est.train.call_args[1]
    self.assertEqual(list(train_spec.hooks), list(train_args['hooks'][:-1]))
    self.assertEqual(train_spec.input_fn, train_args['input_fn'])
    self.assertEqual(train_spec.max_steps, train_args['max_steps'])

  def test_errors_out_if_throttle_secs_is_zero(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = training.EvalSpec(input_fn=lambda: 1, throttle_secs=0)

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    with self.assertRaisesRegexp(ValueError, 'throttle_secs'):
      executor.run_local()


if __name__ == '__main__':
  test.main()
