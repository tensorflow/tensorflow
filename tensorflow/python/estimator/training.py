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

"""Classes and functions related to train_and_evaluate."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
import time

import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_run_hook


_MAX_DELAY_SECS = 60
_DELAY_SECS_PER_WORKER = 5
_TF_CONFIG_ENV = 'TF_CONFIG'
_ENVIRONMENT_KEY = 'environment'
_ENVIRONMENT_GOOGLE_VALUE = 'google'


def _validate_input_fn(input_fn):
  """Validates the `input_fn`."""
  if not callable(input_fn):
    raise TypeError(
        '`input_fn` must be callable, given: {}'.format(input_fn))


def _validate_hooks(hooks):
  """Validates the `hooks`."""
  hooks = tuple(hooks or [])
  for hook in hooks:
    if not isinstance(hook, session_run_hook.SessionRunHook):
      raise TypeError(
          'All hooks must be `SessionRunHook` instances, given: {}'.format(
              hook))
  return hooks


def _is_google_env():
  """Detects whether current environment is google."""
  tf_config = json.loads(os.environ.get(_TF_CONFIG_ENV) or '{}')
  if not tf_config:
    logging.warn('TF_CONFIG should not be empty in distributed environment.')
  return tf_config.get(_ENVIRONMENT_KEY) == _ENVIRONMENT_GOOGLE_VALUE


class TrainSpec(
    collections.namedtuple('TrainSpec', ['input_fn', 'max_steps', 'hooks'])):
  """Objects passed to `train_and_evaluate`.

  `TrainSpec` fully defines the objects to be run by `Estimator.train`.
  """

  def __new__(cls,
              input_fn,
              max_steps=None,
              hooks=None):
    """Creates a validated `TrainSpec` instance.

    Args:
      input_fn: Training input function returning a tuple of:
          features - `Tensor` or dictionary of string feature name to `Tensor`.
          labels - `Tensor` or dictionary of `Tensor` with labels.
      max_steps: Int. Number of total steps for which to train model. If `None`,
        train forever or train until `input_fn` generates the `OutOfRange` error
        or `StopIteration` exception. See `Estimator.train` for details.
      hooks: Iterable of `tf.train.SessionRunHook` objects to run
        on all workers (including chief) during training.

    Returns:
      A validated `TrainSpec` object.

    Raises:
      ValueError: If validation fails.
      TypeError: If any of the arguments is not the expected type.
    """
    # Validate input_fn.
    _validate_input_fn(input_fn)

    # Validate max_steps.
    if max_steps is not None and max_steps <= 0:
      raise ValueError(
          'Must specify max_steps > 0, given: {}'.format(max_steps))

    # Validate hooks.
    hooks = _validate_hooks(hooks)

    return super(TrainSpec, cls).__new__(
        cls,
        input_fn=input_fn,
        max_steps=max_steps,
        hooks=hooks)


class EvalSpec(
    collections.namedtuple('EvalSpec', [
        'input_fn', 'steps', 'name', 'hooks', 'export_strategies',
        'delay_secs', 'throttle_secs'
    ])):
  """Objects passed to `train_and_evaluate`.

  `EvalSpec` fully defines the objects to be run by `Estimator.evaluate` and
  `Estimator.export_savedmodel`.
  """

  def __new__(cls,
              input_fn,
              steps=100,
              name=None,
              hooks=None,
              export_strategies=None,
              delay_secs=120,
              throttle_secs=600):
    """Creates a validated `EvalSpec` instance.

    Args:
      input_fn: Training input function returning a tuple of:
          features - `Tensor` or dictionary of string feature name to `Tensor`.
          labels - `Tensor` or dictionary of `Tensor` with labels.
      steps: Int. Number of total steps for which to train model. If `None`,
        train forever or train until `input_fn` generates the `OutOfRange` error
        or `StopIteration` exception. See `Estimator.train` for details.
      name: String. Name of the evaluation if user needs to run multiple
        evaluations on different data sets. Metrics for different evaluations
        are saved in separate folders, and appear separately in tensorboard.
      hooks: Iterable of `tf.train.SessionRunHook` objects to run
        on all workers (including chief) during training.
      export_strategies: Iterable of `ExportStrategy`s, or a single one, or
        `None`. `export_strategies` will be invoked after each evaluation.
      delay_secs: Int. Start evaluating after waiting for this many seconds.
      throttle_secs: Int. Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago. Of course, evaluation does not
        occur if no new checkpoint is available, hence, this is the minimum.

    Returns:
      A validated `TrainSpec` object.

    Raises:
      ValueError: If validation fails.
      TypeError: If any of the arguments is not the expected type.
    """
    # Validate input_fn.
    _validate_input_fn(input_fn)

    # Validate steps.
    if steps is not None and steps <= 0:
      raise ValueError('Must specify steps > 0, given: {}'.format(steps))

    # Validate name.
    if name is not None and not isinstance(name, six.string_types):
      raise TypeError('`name` must be string, given: {}'.format(name))

    # Validate hooks.
    hooks = _validate_hooks(hooks)

    # Validate export_strategies.
    export_strategies = tuple(export_strategies or [])
    # TODO(b/65169058): Validate export_strategies once `ExportStratey` defined.

    # Validate delay_secs.
    if delay_secs < 0:
      raise ValueError(
          'Must specify delay_secs >= 0, given: {}'.format(delay_secs))

    # Validate throttle_secs.
    if throttle_secs < 0:
      raise ValueError(
          'Must specify throttle_secs >= 0, given: {}'.format(throttle_secs))

    return super(EvalSpec, cls).__new__(
        cls,
        input_fn=input_fn,
        steps=steps,
        name=name,
        hooks=hooks,
        export_strategies=export_strategies,
        delay_secs=delay_secs,
        throttle_secs=throttle_secs)


class UnimplementedError(Exception):
  pass


class _TrainingExecutor(object):
  """The executor to run `Estimator` training and evaluation.

  This implementation supports both distributed and non-distributed (aka local)
  training and evaluation based on the setting in `tf.estimator.RunConfig`.
  """

  def __init__(self, estimator, train_spec, eval_spec):
    if not isinstance(estimator, estimator_lib.Estimator):
      raise TypeError('`estimator` must have type `tf.estimator.Estimator`.')
    self._estimator = estimator

    if not isinstance(train_spec, TrainSpec):
      raise TypeError('`train_spec` must have type `tf.estimator.TrainSpec`.')
    self._train_spec = train_spec

    if not isinstance(eval_spec, EvalSpec):
      raise TypeError('`eval_spec` must have type `tf.estimator.EvalSpec`.')
    self._eval_spec = eval_spec

  @property
  def estimator(self):
    return self._estimator

  def run_chief(self):
    """Runs task chief."""
    # TODO(xiejw): To allow execution framework to add train hooks.
    return self._start_distributed_training()

  def run_worker(self):
    """Runs task (training) worker."""
    # TODO(xiejw): To allow execution framework to add train hooks.
    return self._start_distributed_training()

  def run_evaluator(self):
    """Runs task evaluator."""
    # TODO(xiejw): To allow execution framework to add continuous eval listener.
    return self._start_continuous_evaluation()

  def run_ps(self):
    """Runs task parameter server (in training cluster spec)."""
    config = self._estimator.config
    server = self._start_std_server(config)
    server.join()

  def run_local(self):
    """Runs training and evaluation locally (non-distributed)."""
    raise UnimplementedError('Method run_local has not been implemented.')

  def _start_std_server(self, config):
    """Creates, starts, and returns a server_lib.Server."""
    if (not config.cluster_spec or not config.task_type or not config.master or
        config.task_id is None):
      raise RuntimeError('Could not start server; be sure to specify '
                         'cluster_spec, task_type, master, and task in '
                         'RunConfig or set the TF_CONFIG environment variable.')
    server = server_lib.Server(
        config.cluster_spec,
        job_name=config.task_type,
        task_index=config.task_id,
        config=config_pb2.ConfigProto(log_device_placement=False),
        start=False)
    server.start()
    return server

  def _start_distributed_training(self):
    """Calls `Estimator` train in a distributed setting."""
    config = self._estimator.config

    # Start in-process TensorFlow server if needed. It's important to start the
    # server before we (optionally) sleep. Otherwise, the servers will wait to
    # connect to each other before starting to train.
    if not _is_google_env():
      self._start_std_server(config)

    # Delay worker to start. For asynchronous training, this usually helps model
    # to converge faster.  Chief starts the training immediately, so, worker
    # with task id x (0-based) should wait (x+1) * _DELAY_SECS_PER_WORKER.
    delay_secs = 0
    if config.task_type == run_config_lib.TaskType.WORKER:
      # TODO(xiejw): Replace the hard code logic (task_id + 1) with unique id in
      # training cluster.
      delay_secs = min(_MAX_DELAY_SECS,
                       (config.task_id + 1) * _DELAY_SECS_PER_WORKER)
    if delay_secs > 0:
      logging.info('Waiting %d secs before starting training.', delay_secs)
      time.sleep(delay_secs)

    self._estimator.train(input_fn=self._train_spec.input_fn,
                          max_steps=self._train_spec.max_steps,
                          hooks=self._train_spec.hooks)

  def _start_continuous_evaluation(self):
    """Repeatedly calls `Estimator` evaluate and export until training ends."""
    delay_secs = self._eval_spec.delay_secs
    if delay_secs:
      logging.info('Waiting %f secs before starting eval.', delay_secs)
      time.sleep(delay_secs)

    latest_eval_result = None
    evaluator = _TrainingExecutor._Evaluator(self._estimator, self._eval_spec)

    while True:
      if latest_eval_result:
        global_step = latest_eval_result.get(ops.GraphKeys.GLOBAL_STEP)
        if (global_step and self._train_spec.max_steps and
            global_step >= self._train_spec.max_steps):
          logging.info(
              'Exiting evaluation, global_step=%s >= train max_steps=%s',
              global_step,
              self._train_spec.max_steps)
          return

      start = time.time()
      latest_eval_result = evaluator.evaluate_and_export()

      # Throttle if necessary.
      elapsed_time = time.time() - start
      difference = self._eval_spec.throttle_secs  - elapsed_time
      if difference > 0:
        logging.info('Waiting %f secs before starting next eval run.',
                     difference)
        time.sleep(difference)

  class _Evaluator(object):
    """A helper class to call `Estimator.evaluate` and export model."""

    def __init__(self, estimator, eval_spec):
      self._estimator = estimator
      self._eval_spec = eval_spec
      self._previous_ckpt_path = None
      self._last_warning_time = 0

    def evaluate_and_export(self):
      """Evaluate and (maybe) export the current model.

      Returns:
        Evaluation results. Returns `None` if current round of evaluation is
        skipped.
      """
      latest_ckpt_path = saver.latest_checkpoint(self._estimator.model_dir)
      if not latest_ckpt_path:
        self._log_err_msg('Estimator is not trained yet. Will start an '
                          'evaluation when a checkpoint is ready.')
        return None

      if latest_ckpt_path == self._previous_ckpt_path:
        self._log_err_msg(
            'No new checkpoint ready for evaluation. Skip the current '
            'evaluation pass as evaluation results are expected to be same '
            'for the same checkpoint.')
        return None

      eval_result = self._estimator.evaluate(
          input_fn=self._eval_spec.input_fn,
          steps=self._eval_spec.steps,
          name=self._eval_spec.name,
          checkpoint_path=latest_ckpt_path,
          hooks=self._eval_spec.hooks)

      if not eval_result:
        self._log_err_msg('Estimator evaluate returns empty result.')
        return None

      # TODO(b/65169058): Adds export once export strategies are moved.

      self._last_warning_time = 0
      self._previous_ckpt_path = latest_ckpt_path
      return eval_result

    def _log_err_msg(self, message):
      """Prints warning `message` every 10 mins."""
      current_time = time.time()
      if current_time - self._last_warning_time > 600:
        logging.warning(message)
        self._last_warning_time = current_time
