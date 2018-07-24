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

import glob
import json
import os
import random
import shutil
import tempfile
import time

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import exporter as exporter_lib
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.estimator import training
from tensorflow.python.estimator.canned import dnn
from tensorflow.python.estimator.canned import prediction_keys
from tensorflow.python.estimator.export import export as export_lib
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import metrics as metrics_lib
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
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
_INVALID_EVAL_DELAY_SECS_MSG = 'Must specify start_delay_secs >= 0'
_INVALID_EVAL_THROTTLE_SECS_MSG = 'Must specify throttle_secs >= 0'
_INVALID_ESTIMATOR_MSG = '`estimator` must have type `tf.estimator.Estimator`'
_STALE_CHECKPOINT_MSG = 'There was no new checkpoint after the training.'
_INVALID_EXPORTER_MSG = '`exporters` must be an Exporter'
_INVALID_EXPORTER_NAME_TYPE_MSG = 'An Exporter must have a string name'
_DUPLICATE_EXPORTER_NAMES_MSG = '`exporters` must have unique names.'
_NONE_EXPORTER_NAME_MSG = (
    'An Exporter cannot have a name that is `None` or empty.')
_INVALID_TRAIN_SPEC_MSG = '`train_spec` must have type `tf.estimator.TrainSpec`'
_INVALID_EVAL_SPEC_MSG = '`eval_spec` must have type `tf.estimator.EvalSpec`'
_EVAL_SPEC_OR_NONE_MSG = (
    '`eval_spec` must be either `None` or have type `tf.estimator.EvalSpec`')
_INVALID_EVAL_LISTENER_MSG = 'must have type `_ContinuousEvalListener`'
_INVALID_CONFIG_FOR_STD_SERVER_MSG = 'Could not start server; .*TF_CONFIG'
_INVALID_LOCAL_TASK_WITH_CLUSTER = '`task.type` in TF_CONFIG cannot be `local`'
_INVALID_TASK_TYPE = '`estimator.config` must have task_type set.'
# The message should NOT have 'local' word as part of it. As (?!word) is looking
# ahead, so, the $ (ending) check is required; otherwise, it will match
# partially and return successuful.
_INVALID_TASK_TO_RUN = (
    'Task type .* is not supported. Supported task types are ((?!local).)*$')
_INVALID_EMPTY_EVAL_RESULT_ERR = (
    'Internal error: `Estimator.evaluate` should never return empty metrics')
_INVALID_EVAL_RESULT_TYPE_ERR = '`Estimator.evaluate` should return dict.'
_MISSING_GLOBAL_STEP_IN_EVAL_RESULT_ERR = (
    'Internal error: `Estimator.evaluate` result should have `global_step`')
_INVALID_EVAL_TASK_ID_ERR = (
    'there can only be one `evaluator` task .*with task id 0')

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

_TF_CONFIG_FOR_MASTER = {
    'cluster': {
        run_config_lib.TaskType.MASTER: ['host0:0'],
        run_config_lib.TaskType.PS: ['host1:1', 'host2:2'],
        run_config_lib.TaskType.WORKER: ['host3:3', 'host4:4']
    },
    'task': {
        'type': run_config_lib.TaskType.MASTER,
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
        'index': 0
    }
}

_TF_CONFIG_FOR_GOOGLE = {'environment': 'google'}


class _FakeHook(session_run_hook.SessionRunHook):
  """Fake implementation of `SessionRunHook`."""


class _InvalidHook(object):
  """Invalid hook (not a subclass of `SessionRunHook`)."""


def _create_exporter(name):
  class FakeExporter(exporter_lib.Exporter):

    def __init__(self, name):
      self._name = name

    @property
    def name(self):
      return self._name

    def export(self, *args, **kwargs):
      del args, kwargs

  return FakeExporter(name=name)


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
    self.assertEqual(0, len(spec.exporters))
    self.assertEqual(_DEFAULT_EVAL_DELAY_SECS, spec.start_delay_secs)
    self.assertEqual(_DEFAULT_EVAL_THROTTLE_SECS, spec.throttle_secs)

  def testAllArgumentsSet(self):
    """Tests that no errors are raised when all arguments are set."""
    hooks = [_FakeHook()]
    exporter = _create_exporter('a')

    spec = training.EvalSpec(
        input_fn=lambda: 1,
        steps=2,
        name='name',
        hooks=hooks,
        exporters=exporter,
        start_delay_secs=3,
        throttle_secs=4)
    self.assertEqual(1, spec.input_fn())
    self.assertEqual(2, spec.steps)
    self.assertEqual('name', spec.name)
    self.assertEqual(tuple(hooks), spec.hooks)
    self.assertEqual((exporter,), spec.exporters)
    self.assertEqual(3, spec.start_delay_secs)
    self.assertEqual(4, spec.throttle_secs)

  def testListOfExporters(self):
    """Tests that no errors are raised with multiple exporters."""
    exporters = [_create_exporter('a'), _create_exporter('b')]

    spec = training.EvalSpec(input_fn=lambda: 1, exporters=exporters)
    self.assertEqual(1, spec.input_fn())
    self.assertEqual(tuple(exporters), spec.exporters)

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
      training.EvalSpec(input_fn=lambda: 1, start_delay_secs=-1)

  def testInvalidThrottleSecs(self):
    with self.assertRaisesRegexp(ValueError, _INVALID_EVAL_THROTTLE_SECS_MSG):
      training.EvalSpec(input_fn=lambda: 1, throttle_secs=-1)

  def testInvalidTypeOfListOfExporters(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_EXPORTER_MSG):
      training.EvalSpec(
          input_fn=lambda: 1, exporters=[_create_exporter('a'),
                                         _FakeHook()])

  def testInvalidTypeOfIndividualExporter(self):
    with self.assertRaisesRegexp(TypeError, _INVALID_EXPORTER_MSG):
      training.EvalSpec(input_fn=lambda: 1, exporters=_FakeHook())

  def testInvalidTypeOfExporterName(self):
    with self.assertRaisesRegexp(ValueError, _INVALID_EXPORTER_NAME_TYPE_MSG):
      training.EvalSpec(input_fn=lambda: 1,
                        exporters=_create_exporter(name=123))

  def testMultipleExportersWithTheSameName(self):
    with self.assertRaisesRegexp(ValueError, _DUPLICATE_EXPORTER_NAMES_MSG):
      training.EvalSpec(
          input_fn=lambda: 1,
          exporters=[_create_exporter('a'), _create_exporter('a')])

  def testMultipleExportersAndOneWithoutAName(self):
    with self.assertRaisesRegexp(ValueError, _NONE_EXPORTER_NAME_MSG):
      training.EvalSpec(
          input_fn=lambda: 1,
          exporters=[_create_exporter('a'),
                     _create_exporter(None)])

  def testSingleExporterWithoutAName(self):
    with self.assertRaisesRegexp(ValueError, _NONE_EXPORTER_NAME_MSG):
      training.EvalSpec(input_fn=lambda: 1, exporters=_create_exporter(None))


class TrainAndEvaluateTest(test.TestCase):

  def test_run_task(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    with test.mock.patch.object(training, '_TrainingExecutor') as mock_executor:
      mock_executor_instance = test.mock.Mock()
      mock_executor.return_value = mock_executor_instance
      training.train_and_evaluate(mock_est, mock_train_spec, mock_eval_spec)
      mock_executor.assert_called_with(estimator=mock_est,
                                       train_spec=mock_train_spec,
                                       eval_spec=mock_eval_spec)
      self.assertTrue(mock_executor_instance.run.called)

  def test_error_out_if_evaluator_task_id_is_non_zero(self):
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host0:0'],
        },
        'task': {
            'type': run_config_lib.TaskType.EVALUATOR,
            'index': 1
        }
    }

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = _create_run_config_with_cluster_spec(tf_config)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    with self.assertRaisesRegexp(ValueError, _INVALID_EVAL_TASK_ID_ERR):
      training.train_and_evaluate(mock_est, mock_train_spec, mock_eval_spec)

  def test_invalid_estimator(self):
    invalid_estimator = object()
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    with self.assertRaisesRegexp(TypeError, _INVALID_ESTIMATOR_MSG):
      training.train_and_evaluate(invalid_estimator, mock_train_spec,
                                  mock_eval_spec)

  def test_fail_fast_if_invalid_eval_spec(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    invalid_eval_spec = object()

    with test.mock.patch.object(training, '_TrainingExecutor') as mock_executor:
      with self.assertRaisesRegexp(TypeError, _INVALID_EVAL_SPEC_MSG):
        training.train_and_evaluate(mock_est, mock_train_spec,
                                    invalid_eval_spec)

      mock_executor.assert_not_called()


class TrainingExecutorConstructorTest(test.TestCase):
  """Tests constructor of _TrainingExecutor."""

  def test_required_arguments_set(self):
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

    with self.assertRaisesRegexp(TypeError, _EVAL_SPEC_OR_NONE_MSG):
      training._TrainingExecutor(estimator, train_spec, invalid_eval_spec)

  def test_eval_spec_none(self):
    estimator = estimator_lib.Estimator(model_fn=lambda features: features)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = None

    # Tests that no error is raised.
    training._TrainingExecutor(estimator, train_spec, eval_spec)

  def test_invalid_train_hooks(self):
    estimator = estimator_lib.Estimator(model_fn=lambda features: features)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = training.EvalSpec(input_fn=lambda: 1)
    invalid_train_hooks = [object()]

    with self.assertRaisesRegexp(TypeError, _INVALID_HOOK_MSG):
      training._TrainingExecutor(
          estimator, train_spec, eval_spec, train_hooks=invalid_train_hooks)

  def test_invalid_continuous_eval_listener(self):
    estimator = estimator_lib.Estimator(model_fn=lambda features: features)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = training.EvalSpec(input_fn=lambda: 1)
    invalid_continuous_eval_listener = object()

    with self.assertRaisesRegexp(TypeError, _INVALID_EVAL_LISTENER_MSG):
      training._TrainingExecutor(
          estimator,
          train_spec,
          eval_spec,
          continuous_eval_listener=invalid_continuous_eval_listener)


class _TrainingExecutorTrainingTest(object):
  """Tests training of _TrainingExecutor."""

  def __init__(self, run_config):
    self._run_config = run_config

  def _run_task(self, executor):
    # We should not call executor.run as the test here is intended to test
    # run_foo explicitly (foo is the task type).
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
        protocol=None,
        start=False)

    self.assertTrue(mock_server_instance.start.called)

    mock_est.train.assert_called_with(
        input_fn=train_spec.input_fn,
        max_steps=train_spec.max_steps,
        hooks=list(train_spec.hooks),
        saving_listeners=test.mock.ANY)
    mock_est.evaluate.assert_not_called()
    mock_est.export_savedmodel.assert_not_called()

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_train_with_no_eval_spec(self, mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = self._run_config
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=2, hooks=[_FakeHook()])
    eval_spec = None
    mock_server_instance = mock_server.return_value

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    self._run_task(executor)

    mock_server.assert_called_with(
        mock_est.config.cluster_spec,
        job_name=mock_est.config.task_type,
        task_index=mock_est.config.task_id,
        config=test.mock.ANY,
        protocol=None,
        start=False)

    self.assertTrue(mock_server_instance.start.called)

    mock_est.train.assert_called_with(
        input_fn=train_spec.input_fn,
        max_steps=train_spec.max_steps,
        hooks=list(train_spec.hooks),
        saving_listeners=test.mock.ANY)
    mock_est.evaluate.assert_not_called()
    mock_est.export_savedmodel.assert_not_called()

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_train_with_train_hooks(self, unused_mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = self._run_config
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=2, hooks=[_FakeHook()])
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)
    extra_hooks = [_FakeHook()]

    executor = training._TrainingExecutor(
        mock_est, train_spec, mock_eval_spec, train_hooks=extra_hooks)
    self._run_task(executor)

    mock_est.train.assert_called_with(
        input_fn=train_spec.input_fn,
        max_steps=train_spec.max_steps,
        hooks=list(train_spec.hooks) + extra_hooks,
        saving_listeners=test.mock.ANY)

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_no_server_startup_in_google(self, mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = self._run_config
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec, hooks=[])
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
    mock_est.config.cluster_spec = server_lib.ClusterSpec(
        {'worker': ['dummy', 'dummy1']})
    mock_est.config.master = ''
    mock_est.config.task_type = 'worker'
    mock_est.config.task_id = 2

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      self._run_task(training._TrainingExecutor(mock_est, mock_train_spec,
                                                mock_eval_spec))

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_single_worker_node_with_empty_tf_master(
      self, mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec, hooks=[])
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    # Single node cluster.
    mock_est.config.cluster_spec = server_lib.ClusterSpec({'worker': ['dummy']})
    mock_est.config.master = ''
    mock_est.config.task_type = 'worker'
    mock_est.config.task_id = 2

    self._run_task(training._TrainingExecutor(mock_est, mock_train_spec,
                                              mock_eval_spec))
    self.assertTrue(mock_est.train.called)
    mock_server.assert_not_called()

  def test_fail_with_empty_task_type(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = server_lib.ClusterSpec({'worker': ['dummy']})
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
    mock_est.config.cluster_spec = server_lib.ClusterSpec({'worker': ['dummy']})
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
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec, hooks=[])
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
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec, hooks=[])
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)

    with test.mock.patch.object(time, 'sleep') as mock_sleep:
      self._run_task(executor)
      mock_sleep.assert_not_called()


class TrainingExecutorRunMasterTest(test.TestCase):
  """Tests run_chief of _TrainingExecutor."""

  def setUp(self):
    self._run_config = _create_run_config_with_cluster_spec(
        _TF_CONFIG_FOR_MASTER)

  @test.mock.patch.object(server_lib, 'Server')
  def test_no_delay_for_master(self, _):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate = lambda *args, **kw: {ops.GraphKeys.GLOBAL_STEP: 123}
    mock_est.config = self._run_config
    mock_train_spec = test.mock.Mock(
        spec=training.TrainSpec, max_steps=123, hooks=[])
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec, exporters=[])

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)

    with test.mock.patch.object(time, 'sleep') as mock_sleep:
      executor.run_master()
      mock_sleep.assert_not_called()

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_train_with_train_spec(self, mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate = lambda *args, **kw: {ops.GraphKeys.GLOBAL_STEP: 123}
    mock_est.config = self._run_config
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=2, hooks=[_FakeHook()])
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec, exporters=[])
    mock_server_instance = mock_server.return_value

    executor = training._TrainingExecutor(mock_est, train_spec, mock_eval_spec)
    executor.run_master()

    mock_server.assert_called_with(
        mock_est.config.cluster_spec,
        job_name=mock_est.config.task_type,
        task_index=mock_est.config.task_id,
        config=test.mock.ANY,
        protocol=None,
        start=False)

    self.assertTrue(mock_server_instance.start.called)

    mock_est.train.assert_called_with(
        input_fn=train_spec.input_fn,
        max_steps=train_spec.max_steps,
        hooks=list(train_spec.hooks),
        saving_listeners=test.mock.ANY)
    mock_est.export_savedmodel.assert_not_called()

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_train_with_no_eval_spec_fails(self, mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate = lambda *args, **kw: {ops.GraphKeys.GLOBAL_STEP: 123}
    mock_est.config = self._run_config
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=2, hooks=[_FakeHook()])
    eval_spec = None

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    with self.assertRaisesRegexp(TypeError, _INVALID_EVAL_SPEC_MSG):
      executor.run_master()

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_train_with_train_hooks(self, mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate = lambda *args, **kw: {ops.GraphKeys.GLOBAL_STEP: 123}
    mock_est.config = self._run_config
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=2, hooks=[_FakeHook()])
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec, exporters=[])
    extra_hooks = [_FakeHook()]

    executor = training._TrainingExecutor(
        mock_est, train_spec, mock_eval_spec, train_hooks=extra_hooks)
    executor.run_master()

    mock_est.train.assert_called_with(
        input_fn=train_spec.input_fn,
        max_steps=train_spec.max_steps,
        hooks=list(train_spec.hooks) + extra_hooks,
        saving_listeners=test.mock.ANY)

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_no_server_startup_in_google(self, mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate = lambda *args, **kw: {ops.GraphKeys.GLOBAL_STEP: 123}
    mock_est.config = self._run_config
    mock_train_spec = test.mock.Mock(
        spec=training.TrainSpec, max_steps=123, hooks=[])
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec, exporters=[])

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)
    tf_config = {'TF_CONFIG': json.dumps(_TF_CONFIG_FOR_GOOGLE)}
    with test.mock.patch.dict('os.environ', tf_config):
      executor.run_master()
      mock_server.assert_not_called()

  def test_fail_with_empty_cluster_spec(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = None
    mock_est.config.master = 'grpc://...'
    mock_est.config.task_type = 'master'
    mock_est.config.task_id = 2

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      training._TrainingExecutor(
          mock_est, mock_train_spec, mock_eval_spec).run_master()

  def test_fail_with_empty_master(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = server_lib.ClusterSpec(
        {'master': ['dummy'], 'worker': ['dummy1']})
    mock_est.config.master = ''
    mock_est.config.task_type = 'master'
    mock_est.config.task_id = 0

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      training._TrainingExecutor(
          mock_est, mock_train_spec, mock_eval_spec).run_master()

  @test.mock.patch.object(time, 'sleep')
  @test.mock.patch.object(server_lib, 'Server')
  def test_single_master_node_with_empty_tf_master(
      self, mock_server, unused_mock_sleep):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate = lambda *args, **kw: {ops.GraphKeys.GLOBAL_STEP: 123}

    mock_train_spec = test.mock.Mock(
        spec=training.TrainSpec, max_steps=123, hooks=[])
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec, exporters=[])

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = server_lib.ClusterSpec(
        {'master': ['dummy']})
    mock_est.config.master = ''
    mock_est.config.task_type = 'master'
    mock_est.config.task_id = 0

    executor = training._TrainingExecutor(
        mock_est, mock_train_spec, mock_eval_spec)
    executor.run_master()

    mock_server.assert_not_called()
    self.assertTrue(mock_est.train.called)

  def test_fail_with_empty_task_type(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = server_lib.ClusterSpec({'master': ['dummy']})
    mock_est.config.master = 'grpc://...'
    mock_est.config.task_type = ''
    mock_est.config.task_id = 2

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      training._TrainingExecutor(
          mock_est, mock_train_spec, mock_eval_spec).run_master()

  def test_fail_with_none_task_id(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.PropertyMock(spec=run_config_lib.RunConfig)
    mock_est.config.cluster_spec = server_lib.ClusterSpec({'master': ['dummy']})
    mock_est.config.master = 'grpc://...'
    mock_est.config.task_type = 'master'
    mock_est.config.task_id = None

    with self.assertRaisesRegexp(RuntimeError,
                                 _INVALID_CONFIG_FOR_STD_SERVER_MSG):
      training._TrainingExecutor(
          mock_est, mock_train_spec, mock_eval_spec).run_master()

  @test.mock.patch.object(server_lib, 'Server')
  def test_run_master_triggers_evaluate_and_export(self, _):

    def estimator_train(saving_listeners, *args, **kwargs):
      #  There shalt be a saving_listener.  Estimator is going to call
      # `after_save`.
      del args, kwargs
      saving_listeners[0].begin()
      saving_listeners[0].after_save(session=None, global_step_value=0)
      saving_listeners[0].after_save(session=None, global_step_value=10)

    mock_est = test.mock.Mock(
        spec=estimator_lib.Estimator, model_dir='path/', train=estimator_train)
    mock_est.latest_checkpoint.return_value = 'checkpoint_path/'
    mock_est.config = self._run_config

    exporter = test.mock.PropertyMock(spec=exporter_lib.Exporter)
    exporter.name = 'see_whether_export_is_called'

    train_spec = training.TrainSpec(input_fn=lambda: 1, max_steps=300)
    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, steps=2, exporters=exporter)
    eval_result = {_GLOBAL_STEP_KEY: train_spec.max_steps}
    mock_est.evaluate.return_value = eval_result

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    executor.run_master()

    mock_est.evaluate.assert_called_with(
        name=eval_spec.name,
        input_fn=eval_spec.input_fn,
        steps=eval_spec.steps,
        checkpoint_path='checkpoint_path/',
        hooks=eval_spec.hooks)
    self.assertEqual(1, exporter.export.call_count)
    exporter.export.assert_called_with(
        estimator=mock_est,
        export_path=os.path.join('path/', 'export', exporter.name),
        checkpoint_path='checkpoint_path/',
        eval_result=eval_result,
        is_the_final_export=True)

  @test.mock.patch.object(basic_session_run_hooks, 'SecondOrStepTimer')
  @test.mock.patch.object(server_lib, 'Server')
  def test_run_master_throttle_eval(self, _, mock_timer_class):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, model_dir='path/')

    mock_timer = test.mock.Mock()
    mock_timer_class.return_value = mock_timer

    def estimator_train(saving_listeners, *args, **kwargs):
      del args, kwargs
      saving_listeners[0].begin()

      # Call four times.
      mock_timer.should_trigger_for_step.return_value = True
      saving_listeners[0].after_save(session=None, global_step_value=None)

      mock_timer.should_trigger_for_step.return_value = True
      saving_listeners[0].after_save(session=None, global_step_value=None)

      mock_timer.should_trigger_for_step.return_value = False
      saving_listeners[0].after_save(session=None, global_step_value=None)

      mock_timer.should_trigger_for_step.return_value = True
      saving_listeners[0].after_save(session=None, global_step_value=None)

    mock_est.train = estimator_train
    mock_est.latest_checkpoint.side_effect = ['ckpt1', 'ckpt2']
    mock_est.config = self._run_config

    exporter = test.mock.PropertyMock(spec=exporter_lib.Exporter)
    exporter.name = 'see_whether_export_is_called'

    train_spec = training.TrainSpec(input_fn=lambda: 1, max_steps=300)
    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, steps=2, exporters=exporter, throttle_secs=10)

    mock_est.evaluate.side_effect = [
        {_GLOBAL_STEP_KEY: train_spec.max_steps //2},
        {_GLOBAL_STEP_KEY: train_spec.max_steps}
    ]

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    executor.run_master()

    self.assertEqual(2, mock_est.evaluate.call_count)
    self.assertEqual(2, exporter.export.call_count)

    is_final_export_list = [call[1]['is_the_final_export']
                            for call in exporter.export.call_args_list]
    self.assertEqual([False, True], is_final_export_list)

  @test.mock.patch.object(basic_session_run_hooks, 'SecondOrStepTimer')
  @test.mock.patch.object(server_lib, 'Server')
  def test_run_master_throttle_eval_which_skips_final_ckpt(
      self, _, mock_timer_class):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, model_dir='path/')

    mock_timer = test.mock.Mock()
    mock_timer_class.return_value = mock_timer

    def estimator_train(saving_listeners, *args, **kwargs):
      del args, kwargs
      saving_listeners[0].begin()

      # Call tree times (one for first saving).
      mock_timer.should_trigger_for_step.return_value = True
      saving_listeners[0].after_save(session=None, global_step_value=0)

      mock_timer.should_trigger_for_step.return_value = True
      saving_listeners[0].after_save(session=None, global_step_value=125)

      mock_timer.should_trigger_for_step.return_value = False
      saving_listeners[0].after_save(session=None, global_step_value=250)

      # At the end evaluate should be called even if throttle secs prevents it.
      mock_timer.should_trigger_for_step.return_value = False
      saving_listeners[0].end(session=None, global_step_value=300)

    mock_est.train = estimator_train
    mock_est.latest_checkpoint.side_effect = ['ckpt1', 'ckpt2']
    mock_est.config = self._run_config

    exporter = test.mock.PropertyMock(spec=exporter_lib.Exporter)
    exporter.name = 'see_whether_export_is_called'

    train_spec = training.TrainSpec(input_fn=lambda: 1, max_steps=300)
    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, steps=2, exporters=exporter, throttle_secs=10)

    mock_est.evaluate.side_effect = [
        {_GLOBAL_STEP_KEY: train_spec.max_steps //2},
        {_GLOBAL_STEP_KEY: train_spec.max_steps}
    ]

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    executor.run_master()

    self.assertEqual(2, mock_est.evaluate.call_count)
    self.assertEqual(2, exporter.export.call_count)

    is_final_export_list = [call[1]['is_the_final_export']
                            for call in exporter.export.call_args_list]
    self.assertEqual([False, True], is_final_export_list)


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
        start_delay_secs=0, throttle_secs=0)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    executor.run_evaluator()

    mock_est.evaluate.assert_called_with(
        name='cont_eval',
        input_fn=eval_spec.input_fn,
        steps=eval_spec.steps,
        checkpoint_path='latest_it_is',
        hooks=eval_spec.hooks)
    self.assertFalse(mock_est.train.called)

  def test_evaluate_with_no_eval_spec_fails(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.latest_checkpoint.return_value = 'latest_it_is'
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    self._set_up_mock_est_to_train_and_evaluate_once(mock_est, mock_train_spec)

    eval_spec = None

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)

    with self.assertRaisesRegexp(TypeError, _INVALID_EVAL_SPEC_MSG):
      executor.run_evaluator()

  def test_evaluate_with_train_hooks(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.latest_checkpoint.return_value = 'latest_it_is'
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    self._set_up_mock_est_to_train_and_evaluate_once(mock_est, mock_train_spec)

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1,
        steps=2,
        hooks=[_FakeHook()],
        name='cont_eval',
        start_delay_secs=0,
        throttle_secs=0)

    # The train_hooks will not be called during eval.
    mock_hook = test.mock.Mock(spec=session_run_hook.SessionRunHook)
    executor = training._TrainingExecutor(
        mock_est, mock_train_spec, eval_spec, train_hooks=[mock_hook])
    executor.run_evaluator()

    mock_hook.begin.assert_not_called()

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

    exporter = test.mock.PropertyMock(spec=exporter_lib.Exporter)
    exporter.name = 'see_how_many_times_export_is_called'

    mock_est.times_export_was_called = 0
    mock_est.times_final_export_was_called = 0
    def export(estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
      del export_path, checkpoint_path, eval_result
      estimator.times_export_was_called += 1
      # final_export is happened at the end.
      self.assertEqual(0, estimator.times_final_export_was_called)
      if is_the_final_export:
        estimator.times_final_export_was_called += 1

    exporter.export = export

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1,
        start_delay_secs=0,
        throttle_secs=0,
        exporters=exporter)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    executor.run_evaluator()

    self.assertEqual(2, mock_est.evaluate.call_count)
    self.assertEqual(2, mock_est.times_export_was_called)
    self.assertEqual(1, mock_est.times_final_export_was_called)

  def test_evaluate_listener_before_eval(self):
    training_max_step = 200

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.model_dir = compat.as_bytes(test.get_temp_dir())
    # Without early stopping, this eval will be run twice.
    mock_est.evaluate.side_effect = [{
        _GLOBAL_STEP_KEY: training_max_step // 2
    }, {
        _GLOBAL_STEP_KEY: training_max_step
    }]
    mock_est.latest_checkpoint.side_effect = ['path_1', 'path_2']

    mock_train_spec = test.mock.Mock(spec=training.TrainSpec, hooks=[])
    mock_train_spec.max_steps = training_max_step

    class _Listener(training._ContinuousEvalListener):

      def __init__(self):
        self.call_count = 0

      def before_eval(self):
        self.call_count += 1
        return  self.call_count == 1

    listener = _Listener()

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, start_delay_secs=0, throttle_secs=0)

    training._TrainingExecutor(
        mock_est, mock_train_spec, eval_spec,
        continuous_eval_listener=listener).run_evaluator()

    # Before_eval returns False during the second time, so, evaluate will be
    # called once.
    self.assertEqual(1, mock_est.evaluate.call_count)
    self.assertEqual(2, listener.call_count)

  def test_evaluate_listener_after_eval(self):
    training_max_step = 200

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.model_dir = compat.as_bytes(test.get_temp_dir())
    # Without early stopping, this eval will be run twice.
    expected_eval_metrics = [{
        _GLOBAL_STEP_KEY: training_max_step // 2
    }, {
        _GLOBAL_STEP_KEY: training_max_step
    }]
    mock_est.evaluate.side_effect = expected_eval_metrics
    mock_est.latest_checkpoint.side_effect = ['path_1', 'path_2']

    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_train_spec.max_steps = training_max_step

    class _Listener(training._ContinuousEvalListener):

      def __init__(self):
        self.call_count = 0

      def after_eval(self, eval_result):
        self.call_count += 1
        self.eval_result = eval_result
        return False

    listener = _Listener()

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, start_delay_secs=0, throttle_secs=0)

    training._TrainingExecutor(
        mock_est, mock_train_spec, eval_spec,
        continuous_eval_listener=listener).run_evaluator()

    # after_eval returns False during the first time, so, evaluate will be
    # called once.
    self.assertEqual(1, mock_est.evaluate.call_count)
    self.assertEqual(1, listener.call_count)
    self.assertAllEqual(expected_eval_metrics[0], listener.eval_result.metrics)
    self.assertEqual('path_1', listener.eval_result.checkpoint_path)

  def test_final_export_is_true_in_the_end(self):
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
    mock_est.times_the_final_export_was_true = 0
    def export(estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
      del export_path, checkpoint_path, eval_result
      estimator.times_export_fn_was_called += 1
      if is_the_final_export:
        estimator.times_the_final_export_was_true += 1

    exporter = test.mock.PropertyMock(spec=exporter_lib.Exporter)
    exporter.name = 'see_how_many_times_export_is_called'
    exporter.export = export

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1,
        start_delay_secs=0,
        throttle_secs=0,
        exporters=exporter)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    executor.run_evaluator()

    self.assertEqual(2, mock_est.evaluate.call_count)
    self.assertEqual(2, mock_est.times_export_fn_was_called)
    self.assertEqual(1, mock_est.times_the_final_export_was_true)

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
        input_fn=lambda: 1, start_delay_secs=0, throttle_secs=0)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    with test.mock.patch.object(logging, 'warning') as mock_log:
      executor.run_evaluator()

    # Three checkpoint paths are invalid.
    self.assertEqual(5, mock_est.latest_checkpoint.call_count)
    self.assertEqual(2, mock_est.evaluate.call_count)

    # Two warning logs are expected (last warning time is reset after a
    # successuful evaluation)
    self.assertEqual(2, mock_log.call_count)

  def test_continuous_eval_listener_eval_result(self):
    training_max_step = 200
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    expected_eval_metrics = [{
        _GLOBAL_STEP_KEY: training_max_step // 2
    }, {
        _GLOBAL_STEP_KEY: training_max_step
    }]
    mock_est.evaluate.side_effect = expected_eval_metrics
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_train_spec.max_steps = training_max_step

    class _Listener(training._ContinuousEvalListener):

      def __init__(self):
        self.eval_results = []

      def after_eval(self, eval_result):
        self.eval_results.append(eval_result)
        return True

    continuous_eval_listener = _Listener()

    self._set_up_mock_est_to_train_and_evaluate_once(mock_est, mock_train_spec)

    # First two items are invalid, next two items are same.
    mock_est.latest_checkpoint.side_effect = [
        None, '', 'same', 'same', 'path_2'
    ]
    expected_eval_results = [
        training._EvalResult(training._EvalStatus.MISSING_CHECKPOINT),
        training._EvalResult(training._EvalStatus.MISSING_CHECKPOINT),
        training._EvalResult(
            training._EvalStatus.EVALUATED,
            metrics=expected_eval_metrics[0],
            checkpoint_path='same'),
        training._EvalResult(training._EvalStatus.NO_NEW_CHECKPOINT),
        training._EvalResult(
            training._EvalStatus.EVALUATED,
            metrics=expected_eval_metrics[1],
            checkpoint_path='path_2'),
    ]

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, start_delay_secs=0, throttle_secs=0)

    executor = training._TrainingExecutor(
        mock_est,
        mock_train_spec,
        eval_spec,
        continuous_eval_listener=continuous_eval_listener)
    executor.run_evaluator()

    # Three checkpoint paths are invalid.
    self.assertEqual(5, mock_est.latest_checkpoint.call_count)
    self.assertEqual(2, mock_est.evaluate.call_count)

    self.assertEqual(5, len(continuous_eval_listener.eval_results))
    for i, result in enumerate(continuous_eval_listener.eval_results):
      self.assertEqual(expected_eval_results[i].status, result.status)
      self.assertAllEqual(expected_eval_results[i].metrics, result.metrics)
      self.assertEqual(expected_eval_results[i].checkpoint_path,
                       result.checkpoint_path)

  def test_sleep_start_delay_secs(self):
    training_max_step = 200
    start_delay_secs = 123

    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.evaluate.return_value = {_GLOBAL_STEP_KEY: training_max_step}
    mock_est.model_dir = compat.as_bytes(test.get_temp_dir())
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_train_spec.max_steps = training_max_step

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1, steps=2, hooks=[_FakeHook()], name='cont_eval',
        start_delay_secs=start_delay_secs, throttle_secs=0)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    with test.mock.patch.object(time, 'sleep') as mock_sleep:
      executor.run_evaluator()
      mock_sleep.assert_called_with(start_delay_secs)
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
        input_fn=lambda: 1, start_delay_secs=0, throttle_secs=throttle_secs)

    mock_time.side_effect = [921, 921 + operation_secs]

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    # Disable logging as it calls time.time also.
    with test.mock.patch.object(logging, 'info'):
      executor.run_evaluator()
    mock_sleep.assert_called_with(throttle_secs - operation_secs)
    self.assertTrue(mock_est.evaluate.called)

  def test_that_export_is_called(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    self._set_up_mock_est_to_train_and_evaluate_once(mock_est, mock_train_spec)

    def export(estimator, *args, **kwargs):
      del args, kwargs
      estimator.export_was_called = True

    exporter = test.mock.PropertyMock(spec=exporter_lib.Exporter)
    exporter.name = 'see_whether_export_is_called'
    exporter.export = export

    eval_spec = training.EvalSpec(
        input_fn=lambda: 1,
        steps=2,
        start_delay_secs=0,
        throttle_secs=0,
        exporters=exporter)

    executor = training._TrainingExecutor(mock_est, mock_train_spec, eval_spec)
    executor.run_evaluator()

    # Verify that export was called on the right estimator.
    self.assertTrue(mock_est.export_was_called)

  def test_errors_out_if_evaluate_returns_empty_dict(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = training.EvalSpec(input_fn=(lambda: 1),
                                  start_delay_secs=0, throttle_secs=0)
    mock_est.evaluate.return_value = {}

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    with self.assertRaisesRegexp(ValueError, _INVALID_EMPTY_EVAL_RESULT_ERR):
      executor.run_evaluator()

  def test_errors_out_if_evaluate_returns_non_dict(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = training.EvalSpec(input_fn=(lambda: 1),
                                  start_delay_secs=0, throttle_secs=0)
    mock_est.evaluate.return_value = 123

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    with self.assertRaisesRegexp(TypeError, _INVALID_EVAL_RESULT_TYPE_ERR):
      executor.run_evaluator()

  def test_errors_out_if_evaluate_returns_dict_without_global_step(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    train_spec = training.TrainSpec(input_fn=lambda: 1)
    eval_spec = training.EvalSpec(input_fn=(lambda: 1),
                                  start_delay_secs=0, throttle_secs=0)
    mock_est.evaluate.return_value = {'loss': 123}

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    with self.assertRaisesRegexp(ValueError,
                                 _MISSING_GLOBAL_STEP_IN_EVAL_RESULT_ERR):
      executor.run_evaluator()


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
        protocol=None,
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
    mock_est.config.task_type = 'ps'
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
    mock_est.config.cluster_spec = server_lib.ClusterSpec({'ps': ['dummy']})
    mock_est.config.master = ''
    mock_est.config.task_type = 'ps'
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
    mock_est.config.cluster_spec = server_lib.ClusterSpec({'ps': ['dummy']})
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
    mock_est.config.cluster_spec = server_lib.ClusterSpec({'ps': ['dummy']})
    mock_est.config.master = 'grpc://...'
    mock_est.config.task_type = 'ps'
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

  def _model_fn(self, features, labels, mode):
    del labels
    with ops.control_dependencies([features]):
      train_op = state_ops.assign_add(training_util.get_global_step(), 1)
    return model_fn_lib.EstimatorSpec(
        mode,
        loss=constant_op.constant(0.),
        train_op=train_op,
        predictions=constant_op.constant([[10.]]),
        eval_metric_ops={'mean_of_features': metrics_lib.mean(features)})

  def _input_fn(self, repeat=True):
    ds = dataset_ops.Dataset.from_tensors([1])
    if repeat:
      return ds.repeat()
    return ds

  def unique_checkpoint_every_time_fn(self):
    return 'checkpoint_path_%s/' % random.random()

  def test_runs_evaluate_with_every_new_checkpoint(self):
    est = estimator_lib.Estimator(
        model_fn=self._model_fn,
        config=run_config_lib.RunConfig(save_checkpoints_steps=10))
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, wraps=est)

    mock_est.times_export_was_called = 0
    mock_est.times_final_export_was_called = 0
    def export(estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
      del export_path, checkpoint_path, eval_result
      estimator.times_export_was_called += 1
      # final_export is happened at the end.
      self.assertEqual(0, estimator.times_final_export_was_called)
      if is_the_final_export:
        estimator.times_final_export_was_called += 1

    exporter = test.mock.PropertyMock(spec=exporter_lib.Exporter)
    exporter.name = 'see_how_many_times_export_is_called'
    exporter.export = export

    train_spec = training.TrainSpec(input_fn=self._input_fn, max_steps=22)
    eval_spec = training.EvalSpec(
        input_fn=lambda: self._input_fn(repeat=False),
        throttle_secs=0,
        exporters=exporter)

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    executor.run_local()

    self.assertEqual(1, mock_est.train.call_count)
    self.assertEqual(3, mock_est.evaluate.call_count)
    self.assertEqual(3, mock_est.times_export_was_called)
    self.assertEqual(1, mock_est.times_final_export_was_called)

  def test_runs_with_eval_listener_before_eval(self):
    est = estimator_lib.Estimator(
        model_fn=self._model_fn,
        config=run_config_lib.RunConfig(save_checkpoints_steps=10))
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, wraps=est)
    mock_est.latest_checkpoint = self.unique_checkpoint_every_time_fn

    train_spec = training.TrainSpec(input_fn=self._input_fn, max_steps=12)
    eval_spec = training.EvalSpec(input_fn=lambda: self._input_fn(repeat=False))
    mock_est.evaluate.side_effect = [{_GLOBAL_STEP_KEY: train_spec.max_steps}]

    class _Listener(training._ContinuousEvalListener):

      def __init__(self):
        self.call_count = 0

      def before_eval(self):
        self.call_count += 1
        return False  # Will stop the run_local before first eval.

    listener = _Listener()

    executor = training._TrainingExecutor(
        mock_est, train_spec, eval_spec, continuous_eval_listener=listener)
    executor.run_local()

    self.assertEqual(1, mock_est.train.call_count)
    self.assertEqual(0, mock_est.evaluate.call_count)

  def test_runs_with_eval_listener_after_eval(self):
    est = estimator_lib.Estimator(
        model_fn=self._model_fn,
        config=run_config_lib.RunConfig(save_checkpoints_steps=10))
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, wraps=est)

    train_spec = training.TrainSpec(input_fn=self._input_fn, max_steps=3000)
    eval_spec = training.EvalSpec(
        input_fn=lambda: self._input_fn(repeat=False), throttle_secs=0)

    class _Listener(training._ContinuousEvalListener):

      def __init__(self):
        self.call_count = 0

      def after_eval(self, eval_result):
        self.call_count += 1
        return False  # Will stop the run_local after first eval.

    listener = _Listener()

    executor = training._TrainingExecutor(
        mock_est, train_spec, eval_spec, continuous_eval_listener=listener)
    metrics, _ = executor.run_local()  # pylint: disable=assignment-from-no-return

    self.assertEqual(1, mock_est.train.call_count)
    self.assertEqual(1, mock_est.evaluate.call_count)
    self.assertEqual(1, listener.call_count)
    # Should be less than max_steps since listener did early stopping.
    self.assertLess(metrics[_GLOBAL_STEP_KEY], train_spec.max_steps)

  def test_handles_no_new_checkpoint_found(self):
    est = estimator_lib.Estimator(
        model_fn=self._model_fn,
        # disable saving checkpoint
        config=run_config_lib.RunConfig(
            save_checkpoints_steps=None, save_checkpoints_secs=None))
    train_spec = training.TrainSpec(
        input_fn=self._input_fn, max_steps=300, hooks=[_FakeHook()])
    eval_spec = training.EvalSpec(
        input_fn=lambda: self._input_fn(repeat=False),
        hooks=[_FakeHook()],
        throttle_secs=100)

    executor = training._TrainingExecutor(est, train_spec, eval_spec)
    with self.assertRaisesRegexp(ValueError,
                                 'There should be a CheckpointSaverHook'):
      executor.run_local()

  def test_final_export_is_true_in_the_end(self):
    est = estimator_lib.Estimator(
        model_fn=self._model_fn,
        config=run_config_lib.RunConfig(save_checkpoints_steps=10))
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, wraps=est)

    mock_est.times_export_fn_was_called = 0
    mock_est.times_the_final_export_was_true = 0
    def export(estimator, export_path, checkpoint_path, eval_result,
               is_the_final_export):
      del export_path, checkpoint_path, eval_result
      estimator.times_export_fn_was_called += 1
      if is_the_final_export:
        estimator.times_the_final_export_was_true += 1

    exporter = test.mock.PropertyMock(spec=exporter_lib.Exporter)
    exporter.name = 'see_how_many_times_export_is_called'
    exporter.export = export

    train_spec = training.TrainSpec(
        input_fn=self._input_fn, max_steps=12, hooks=[_FakeHook()])
    eval_spec = training.EvalSpec(
        input_fn=lambda: self._input_fn(repeat=False),
        throttle_secs=0,
        exporters=exporter)
    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    executor.run_local()

    self.assertEqual(1, mock_est.train.call_count)
    self.assertEqual(2, mock_est.evaluate.call_count)
    self.assertEqual(2, mock_est.times_export_fn_was_called)
    self.assertEqual(1, mock_est.times_the_final_export_was_true)

  def test_train_and_evaluate_args(self):
    est = estimator_lib.Estimator(model_fn=self._model_fn)
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, wraps=est)
    train_spec = training.TrainSpec(
        input_fn=self._input_fn, max_steps=300, hooks=[_FakeHook()])
    eval_spec = training.EvalSpec(
        input_fn=lambda: self._input_fn(repeat=False),
        steps=2,
        hooks=[_FakeHook()],
        name='local_eval')

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    executor.run_local()

    mock_est.evaluate.assert_called_with(
        name=eval_spec.name,
        input_fn=eval_spec.input_fn,
        steps=eval_spec.steps,
        checkpoint_path=est.latest_checkpoint(),
        hooks=eval_spec.hooks)

    train_args = mock_est.train.call_args[1]
    self.assertEqual(list(train_spec.hooks), list(train_args['hooks']))
    self.assertEqual(train_spec.input_fn, train_args['input_fn'])
    self.assertEqual(train_spec.max_steps, train_args['max_steps'])

  def test_train_with_no_eval_spec_fails(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=300, hooks=[_FakeHook()])
    eval_spec = None

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)

    with self.assertRaisesRegexp(TypeError, _INVALID_EVAL_SPEC_MSG):
      executor.run_local()

  def test_train_hooks(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, model_dir='path/')
    mock_est.latest_checkpoint.return_value = 'checkpoint_path/'
    train_spec = training.TrainSpec(
        input_fn=lambda: 1, max_steps=300, hooks=[_FakeHook()])
    eval_spec = training.EvalSpec(input_fn=lambda: 1, steps=2)
    mock_est.evaluate.return_value = {_GLOBAL_STEP_KEY: train_spec.max_steps}
    extra_hooks = [_FakeHook()]

    executor = training._TrainingExecutor(
        mock_est, train_spec, eval_spec, train_hooks=extra_hooks)
    executor.run_local()

    train_args = mock_est.train.call_args[1]
    self.assertEqual(
        list(train_spec.hooks) + extra_hooks, [
            h for h in train_args['hooks']
            if not isinstance(h, training._StopAtSecsHook)
        ])

  def test_that_export_is_called_with_run_local(self):
    est = estimator_lib.Estimator(model_fn=self._model_fn)
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, wraps=est)
    train_spec = training.TrainSpec(input_fn=self._input_fn, max_steps=12)
    mock_est.evaluate.return_value = {_GLOBAL_STEP_KEY: train_spec.max_steps}

    def export(estimator, *args, **kwargs):
      del args, kwargs
      estimator.export_was_called = True
      return 'path_to_export'

    exporter = test.mock.PropertyMock(spec=exporter_lib.Exporter)
    exporter.name = 'see_whether_export_is_called'
    exporter.export = export

    eval_spec = training.EvalSpec(
        input_fn=lambda: self._input_fn(repeat=False),
        steps=2,
        start_delay_secs=0,
        throttle_secs=213,
        exporters=exporter)

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    # pylint: disable=assignment-from-no-return
    _, export_results = executor.run_local()
    # pylint: enable=assignment-from-no-return

    self.assertTrue(mock_est.export_was_called)
    self.assertEqual(export_results, ['path_to_export'])

  def test_errors_out_if_evaluate_returns_empty_dict(self):
    est = estimator_lib.Estimator(
        model_fn=self._model_fn,
        config=run_config_lib.RunConfig(save_checkpoints_steps=2))
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, wraps=est)
    train_spec = training.TrainSpec(input_fn=self._input_fn)
    eval_spec = training.EvalSpec(
        input_fn=lambda: self._input_fn(repeat=False), throttle_secs=0)
    mock_est.evaluate.return_value = {}

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    with self.assertRaisesRegexp(ValueError, _INVALID_EMPTY_EVAL_RESULT_ERR):
      executor.run_local()

  def test_errors_out_if_evaluate_returns_non_dict(self):
    est = estimator_lib.Estimator(
        model_fn=self._model_fn,
        config=run_config_lib.RunConfig(save_checkpoints_steps=2))
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, wraps=est)
    train_spec = training.TrainSpec(input_fn=self._input_fn)
    eval_spec = training.EvalSpec(
        input_fn=lambda: self._input_fn(repeat=False), throttle_secs=0)
    mock_est.evaluate.return_value = 123
    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    with self.assertRaisesRegexp(TypeError, _INVALID_EVAL_RESULT_TYPE_ERR):
      executor.run_local()

  def test_errors_out_if_evaluate_returns_dict_without_global_step(self):
    est = estimator_lib.Estimator(
        model_fn=self._model_fn,
        config=run_config_lib.RunConfig(save_checkpoints_steps=2))
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, wraps=est)
    train_spec = training.TrainSpec(input_fn=self._input_fn)
    eval_spec = training.EvalSpec(
        input_fn=lambda: self._input_fn(repeat=False), throttle_secs=0)
    mock_est.evaluate.return_value = {'loss': 123}

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    with self.assertRaisesRegexp(ValueError,
                                 _MISSING_GLOBAL_STEP_IN_EVAL_RESULT_ERR):
      executor.run_local()

  def test_train_and_evaluate_return_metrics(self):
    est = estimator_lib.Estimator(model_fn=self._model_fn)
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator, wraps=est)
    train_spec = training.TrainSpec(
        input_fn=self._input_fn, max_steps=12, hooks=[_FakeHook()])
    eval_spec = training.EvalSpec(
        input_fn=lambda: self._input_fn(repeat=False),
        steps=2,
        hooks=[_FakeHook()],
        name='local_eval')

    executor = training._TrainingExecutor(mock_est, train_spec, eval_spec)
    # pylint: disable=assignment-from-no-return
    metrics, _ = executor.run_local()
    # pylint: enable=assignment-from-no-return
    self.assertEqual(metrics['global_step'], 12)


class TrainAndEvaluateRunTest(test.TestCase):

  def _test_run_task_and_executor(self, run_config):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = run_config
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)

    executor.call_task = {}

    def task_fn(name):

      def _fn():
        executor.call_task[name] = 1

      return _fn

    executor.run_chief = task_fn('chief')
    executor.run_master = task_fn('master')
    executor.run_ps = task_fn('ps')
    executor.run_evaluator = task_fn('evaluator')
    executor.run_worker = task_fn('worker')
    executor.run_local = task_fn('local')
    return executor

  def test_run_chief(self):
    executor = self._test_run_task_and_executor(
        run_config=_create_run_config_with_cluster_spec(_TF_CONFIG_FOR_CHIEF))
    executor.run()
    self.assertEqual(1, executor.call_task['chief'])

  def test_run_worker(self):
    executor = self._test_run_task_and_executor(
        run_config=_create_run_config_with_cluster_spec(_TF_CONFIG_FOR_WORKER))
    executor.run()
    self.assertEqual(1, executor.call_task['worker'])

  def test_run_ps(self):
    executor = self._test_run_task_and_executor(
        run_config=_create_run_config_with_cluster_spec(_TF_CONFIG_FOR_PS))
    executor.run()
    self.assertEqual(1, executor.call_task['ps'])

  def test_run_evaluator(self):
    executor = self._test_run_task_and_executor(
        run_config=_create_run_config_with_cluster_spec(
            _TF_CONFIG_FOR_EVALUATOR))
    executor.run()
    self.assertEqual(1, executor.call_task['evaluator'])

  def test_run_local(self):
    executor = self._test_run_task_and_executor(
        run_config=run_config_lib.RunConfig())
    executor.run()
    self.assertEqual(1, executor.call_task['local'])

  def test_invalid_local_task(self):
    tf_config = {
        'cluster': {
            run_config_lib.TaskType.CHIEF: ['host0:0'],
            'local': ['hos1:1'],
        },
        'task': {
            'type': 'local',  # invalid task type.
            'index': 0
        }
    }
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = _create_run_config_with_cluster_spec(tf_config)
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)
    with self.assertRaisesRegexp(ValueError, _INVALID_LOCAL_TASK_WITH_CLUSTER):
      executor.run()

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

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)
    with self.assertRaisesRegexp(ValueError, _INVALID_TASK_TO_RUN):
      executor.run()

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

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)
    executor.run_alloc = 123  # not callable
    with self.assertRaisesRegexp(ValueError, _INVALID_TASK_TO_RUN):
      executor.run()

  def test_invalid_task_type(self):
    mock_est = test.mock.Mock(spec=estimator_lib.Estimator)
    mock_est.config = test.mock.Mock()
    mock_train_spec = test.mock.Mock(spec=training.TrainSpec)
    mock_eval_spec = test.mock.Mock(spec=training.EvalSpec)

    mock_est.config = test.mock.Mock()
    mock_est.config.cluster_spec = server_lib.ClusterSpec({'1': ['dummy']})
    mock_est.config.task_type = ''

    executor = training._TrainingExecutor(mock_est, mock_train_spec,
                                          mock_eval_spec)
    with self.assertRaisesRegexp(ValueError, _INVALID_TASK_TYPE):
      executor.run()


class TrainAndEvaluateIntegrationTest(test.TestCase):

  def setUp(self):
    self._model_dir = tempfile.mkdtemp()

  def tearDown(self):
    if self._model_dir:
      shutil.rmtree(self._model_dir)

  def _as_label(self, data_in_float):
    return np.rint(data_in_float).astype(np.int64)

  def _get_exporter(self, name, fc):
    feature_spec = feature_column.make_parse_example_spec(fc)
    serving_input_receiver_fn = (
        export_lib.build_parsing_serving_input_receiver_fn(feature_spec))
    return exporter_lib.LatestExporter(
        name, serving_input_receiver_fn=serving_input_receiver_fn)

  def _extract_loss_and_global_step(self, event_folder):
    """Returns the loss and global step in last event."""
    event_paths = sorted(glob.glob(os.path.join(event_folder, 'events*')))

    loss = None
    global_step_count = None

    for e in summary_iterator.summary_iterator(event_paths[-1]):
      current_loss = None
      for v in e.summary.value:
        if v.tag == 'loss':
          current_loss = v.simple_value

      # If loss is not found, global step is meaningless.
      if current_loss is None:
        continue

      current_global_step = e.step
      if global_step_count is None or current_global_step > global_step_count:
        global_step_count = current_global_step
        loss = current_loss

    return (loss, global_step_count)

  def test_complete_flow_with_non_distributed_configuration(self):
    n_classes = 3
    input_dimension = 2
    batch_size = 10

    eval_name = 'foo'
    exporter_name = 'saved_model_exporter'

    # max_steps should be larger than save_summary_steps
    max_steps = 10
    save_summary_steps = 9

    data = np.linspace(
        0., n_classes - 1., batch_size * input_dimension, dtype=np.float32)
    x_data = data.reshape(batch_size, input_dimension)
    y_data = np.reshape(self._as_label(data[:batch_size]), (batch_size, 1))

    # learn y = x
    def train_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(({
          'x': x_data
      }, y_data)).batch(batch_size).repeat().shuffle(1000)

    def eval_input_fn():
      return dataset_ops.Dataset.from_tensor_slices(({
          'x': x_data
      }, y_data)).batch(batch_size)

    def predict_input_fn():
      return dataset_ops.Dataset.from_tensor_slices({
          'x': x_data
      }).batch(batch_size)

    feature_columns = [
        feature_column.numeric_column('x', shape=(input_dimension,))]

    est = dnn.DNNClassifier(
        hidden_units=(2, 2),
        feature_columns=feature_columns,
        n_classes=n_classes,
        config=run_config_lib.RunConfig(save_summary_steps=save_summary_steps),
        model_dir=self._model_dir)

    train_spec = training.TrainSpec(input_fn=train_input_fn,
                                    max_steps=max_steps)

    eval_spec = training.EvalSpec(
        name=eval_name,
        input_fn=eval_input_fn,
        steps=None,
        exporters=self._get_exporter(exporter_name, feature_columns),
        throttle_secs=0)

    training.train_and_evaluate(est, train_spec, eval_spec)

    # Make sure nothing is stuck in limbo.
    writer_cache.FileWriterCache.clear()

    # Examine the training events.
    training_loss, training_global_step = self._extract_loss_and_global_step(
        est.model_dir)
    self.assertIsNotNone(training_loss)
    # Training summaries are logged for steps 1 and 10, so we see final step.
    self.assertEqual(max_steps, training_global_step)

    # Examine the eval events. The global step should be accurate.
    eval_loss, eval_global_step = self._extract_loss_and_global_step(
        event_folder=est.eval_dir(eval_name))
    self.assertIsNotNone(eval_loss)
    self.assertEqual(max_steps, eval_global_step)

    # Examine the export folder.
    export_dir = os.path.join(os.path.join(est.model_dir, 'export'),
                              exporter_name)
    self.assertTrue(gfile.Exists(export_dir))

    # Examine the ckpt for predict.
    predicted_proba = np.array([
        x[prediction_keys.PredictionKeys.PROBABILITIES]
        for x in est.predict(predict_input_fn)
    ])
    self.assertAllEqual((batch_size, n_classes), predicted_proba.shape)


if __name__ == '__main__':
  test.main()
