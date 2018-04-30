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
from tensorflow.python.estimator import exporter as exporter_lib
from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

_MAX_DELAY_SECS = 60
_DELAY_SECS_PER_WORKER = 5
_TF_CONFIG_ENV = 'TF_CONFIG'
_ENVIRONMENT_KEY = 'environment'
_ENVIRONMENT_GOOGLE_VALUE = 'google'
_TRAINER_JOBS = (run_config_lib.TaskType.CHIEF, run_config_lib.TaskType.MASTER,
                 run_config_lib.TaskType.WORKER)


def _validate_input_fn(input_fn):
  """Validates the `input_fn`."""
  if not callable(input_fn):
    raise TypeError('`input_fn` must be callable, given: {}'.format(input_fn))


def _validate_hooks(hooks):
  """Validates the `hooks`."""
  hooks = tuple(hooks or [])
  for hook in hooks:
    if not isinstance(hook, session_run_hook.SessionRunHook):
      raise TypeError(
          'All hooks must be `SessionRunHook` instances, given: {}'.format(
              hook))
  return hooks


def _validate_exporters(exporters):
  """Validates `exporters` and returns them as a tuple."""
  if not exporters:
    return ()

  if isinstance(exporters, exporter_lib.Exporter):
    exporters = [exporters]

  unique_names = []  # `Exporter`s should have unique names.
  try:
    for exporter in exporters:
      if not isinstance(exporter, exporter_lib.Exporter):
        # Error message will be printed out by the outer try/except.
        raise TypeError

      if not exporter.name:
        full_list_of_names = [e.name for e in exporters]
        raise ValueError('An Exporter cannot have a name that is `None` or'
                         ' empty. All exporter names:'
                         ' {}'.format(full_list_of_names))

      if not isinstance(exporter.name, six.string_types):
        raise ValueError('An Exporter must have a string name. Given: '
                         '{}'.format(type(exporter.name)))

      if exporter.name in unique_names:
        full_list_of_names = [e.name for e in exporters]
        raise ValueError(
            '`exporters` must have unique names. Such a name cannot be `None`.'
            ' All exporter names: {}'.format(full_list_of_names))
      unique_names.append(exporter.name)
  except TypeError:
    # Two possibilities:
    # - `exporters` is neither `Exporter` nor iterable.  Python has
    #   raised a `TypeError` when iterating over `exporters`.
    # - an `exporter` was None or not of type `Exporter`, so we raised a
    #   `TypeError`.
    raise TypeError('`exporters` must be an Exporter,'
                    ' an iterable of Exporter, or `None`,'
                    ' found %s.' % exporters)

  return tuple(exporters)


def _is_google_env():
  """Detects whether current environment is google."""
  tf_config = json.loads(os.environ.get(_TF_CONFIG_ENV) or '{}')
  if not tf_config:
    logging.warn('TF_CONFIG should not be empty in distributed environment.')
  return tf_config.get(_ENVIRONMENT_KEY) == _ENVIRONMENT_GOOGLE_VALUE


@tf_export('estimator.TrainSpec')
class TrainSpec(
    collections.namedtuple('TrainSpec', ['input_fn', 'max_steps', 'hooks'])):
  """Configuration for the "train" part for the `train_and_evaluate` call.

  `TrainSpec` determines the input data for the training, as well as the
  duration. Optional hooks run at various stages of training.
  """

  def __new__(cls, input_fn, max_steps=None, hooks=None):
    """Creates a validated `TrainSpec` instance.

    Args:
      input_fn: A function that provides input data for training as minibatches.
        See @{$get_started/premade_estimators#create_input_functions} for more
        information. The function should construct and return one of
        the following:
          * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
            tuple (features, labels) with same constraints as below.
          * A tuple (features, labels): Where features is a `Tensor` or a
            dictionary of string feature name to `Tensor` and labels is a
            `Tensor` or a dictionary of string label name to `Tensor`.

      max_steps: Int. Positive number of total steps for which to train model.
        If `None`, train forever. The training `input_fn` is not expected to
        generate `OutOfRangeError` or `StopIteration` exceptions. See the
        `train_and_evaluate` stop condition section for details.
      hooks: Iterable of `tf.train.SessionRunHook` objects to run
        on all workers (including chief) during training.

    Returns:
      A validated `TrainSpec` object.

    Raises:
      ValueError: If any of the input arguments is invalid.
      TypeError: If any of the arguments is not of the expected type.
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
        cls, input_fn=input_fn, max_steps=max_steps, hooks=hooks)


@tf_export('estimator.EvalSpec')
class EvalSpec(
    collections.namedtuple('EvalSpec', [
        'input_fn', 'steps', 'name', 'hooks', 'exporters', 'start_delay_secs',
        'throttle_secs'
    ])):
  """Configuration for the "eval" part for the `train_and_evaluate` call.

  `EvalSpec` combines details of evaluation of the trained model as well as its
  export. Evaluation consists of computing metrics to judge the performance of
  the trained model.  Export writes out the trained model on to external
  storage.
  """

  def __new__(cls,
              input_fn,
              steps=100,
              name=None,
              hooks=None,
              exporters=None,
              start_delay_secs=120,
              throttle_secs=600):
    """Creates a validated `EvalSpec` instance.

    Args:
      input_fn: A function that constructs the input data for evaluation.
        See @{$get_started/premade_estimators#create_input_functions} for more
        information. The function should construct and return one of
        the following:
          * A 'tf.data.Dataset' object: Outputs of `Dataset` object must be a
            tuple (features, labels) with same constraints as below.
          * A tuple (features, labels): Where features is a `Tensor` or a
            dictionary of string feature name to `Tensor` and labels is a
            `Tensor` or a dictionary of string label name to `Tensor`.
            
      steps: Int. Positive number of steps for which to evaluate model. If
        `None`, evaluates until `input_fn` raises an end-of-input exception.
        See `Estimator.evaluate` for details.
      name: String. Name of the evaluation if user needs to run multiple
        evaluations on different data sets. Metrics for different evaluations
        are saved in separate folders, and appear separately in tensorboard.
      hooks: Iterable of `tf.train.SessionRunHook` objects to run
        during evaluation.
      exporters: Iterable of `Exporter`s, or a single one, or `None`.
        `exporters` will be invoked after each evaluation.
      start_delay_secs: Int. Start evaluating after waiting for this many
        seconds.
      throttle_secs: Int. Do not re-evaluate unless the last evaluation was
        started at least this many seconds ago. Of course, evaluation does not
        occur if no new checkpoints are available, hence, this is the minimum.

    Returns:
      A validated `EvalSpec` object.

    Raises:
      ValueError: If any of the input arguments is invalid.
      TypeError: If any of the arguments is not of the expected type.
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

    # Validate exporters.
    exporters = _validate_exporters(exporters)

    # Validate start_delay_secs.
    if start_delay_secs < 0:
      raise ValueError('Must specify start_delay_secs >= 0, given: {}'.format(
          start_delay_secs))

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
        exporters=exporters,
        start_delay_secs=start_delay_secs,
        throttle_secs=throttle_secs)


@tf_export('estimator.train_and_evaluate')
def train_and_evaluate(estimator, train_spec, eval_spec):
  """Train and evaluate the `estimator`.

  This utility function trains, evaluates, and (optionally) exports the model by
  using the given `estimator`. All training related specification is held in
  `train_spec`, including training `input_fn` and training max steps, etc. All
  evaluation and export related specification is held in `eval_spec`, including
  evaluation `input_fn`, steps, etc.

  This utility function provides consistent behavior for both local
  (non-distributed) and distributed configurations. Currently, the only
  supported distributed training configuration is between-graph replication.

  Overfitting: In order to avoid overfitting, it is recommended to set up the
  training `input_fn` to shuffle the training data properly. It is also
  recommended to train the model a little longer, say multiple epochs, before
  performing evaluation, as the input pipeline starts from scratch for each
  training. It is particularly important for local training and evaluation.

  Stop condition: In order to support both distributed and non-distributed
  configuration reliably, the only supported stop condition for model
  training is `train_spec.max_steps`. If `train_spec.max_steps` is `None`, the
  model is trained forever. *Use with care* if model stop condition is
  different. For example, assume that the model is expected to be trained with
  one epoch of training data, and the training `input_fn` is configured to throw
  `OutOfRangeError` after going through one epoch, which stops the
  `Estimator.train`. For a three-training-worker distributed configuration, each
  training worker is likely to go through the whole epoch independently. So, the
  model will be trained with three epochs of training data instead of one epoch.

  Example of local (non-distributed) training:
  ```python
  # Set up feature columns.
  categorial_feature_a = categorial_column_with_hash_bucket(...)
  categorial_feature_a_emb = embedding_column(
      categorical_column=categorial_feature_a, ...)
  ...  # other feature columns

  estimator = DNNClassifier(
      feature_columns=[categorial_feature_a_emb, ...],
      hidden_units=[1024, 512, 256])

  # Or set up the model directory
  #   estimator = DNNClassifier(
  #       config=tf.estimator.RunConfig(
  #           model_dir='/my_model', save_summary_steps=100),
  #       feature_columns=[categorial_feature_a_emb, ...],
  #       hidden_units=[1024, 512, 256])

  # Input pipeline for train and evaluate.
  def train_input_fn: # returns x, y
    # please shuffle the data.
    pass
  def eval_input_fn_eval: # returns x, y
    pass

  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1000)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  ```

  Example of distributed training:

  Regarding the example of distributed training, the code above can be used
  without a change (Please do make sure that the `RunConfig.model_dir` for all
  workers is set to the same directory, i.e., a shared file system all workers
  can read and write). The only extra work to do is setting the environment
  variable `TF_CONFIG` properly for each worker correspondingly.

  Also see
  [Distributed TensorFlow](https://www.tensorflow.org/deploy/distributed).

  Setting environment variable depends on the platform. For example, on Linux,
  it can be done as follows (`$` is the shell prompt):
  ```
  $ TF_CONFIG='<replace_with_real_content>' python train_model.py
  ```

  For the content in `TF_CONFIG`, assume that the training cluster spec looks
  like:
  ```
  cluster = {"chief": ["host0:2222"],
             "worker": ["host1:2222", "host2:2222", "host3:2222"],
             "ps": ["host4:2222", "host5:2222"]}
  ```

  Example of `TF_CONFIG` for chief training worker (must have one and only one):
  ```
  # This should be a JSON string, which is set as environment variable. Usually
  # the cluster manager handles that.
  TF_CONFIG='{
      "cluster": {
          "chief": ["host0:2222"],
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "chief", "index": 0}
  }'
  ```
  Note that the chief worker also does the model training job, similar to other
  non-chief training workers (see next paragraph). In addition to the model
  training, it manages some extra work, e.g., checkpoint saving and restoring,
  writing summaries, etc.

  Example of `TF_CONFIG` for non-chief training worker (optional, could be
  multiple):
  ```
  # This should be a JSON string, which is set as environment variable. Usually
  # the cluster manager handles that.
  TF_CONFIG='{
      "cluster": {
          "chief": ["host0:2222"],
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "worker", "index": 0}
  }'
  ```
  where the `task.index` should be set as 0, 1, 2, in this example, respectively
  for non-chief training workers.

  Example of `TF_CONFIG` for parameter server, aka ps (could be multiple):
  ```
  # This should be a JSON string, which is set as environment variable. Usually
  # the cluster manager handles that.
  TF_CONFIG='{
      "cluster": {
          "chief": ["host0:2222"],
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "ps", "index": 0}
  }'
  ```
  where the `task.index` should be set as 0 and 1, in this example, respectively
  for parameter servers.

  Example of `TF_CONFIG` for evaluator task. Evaluator is a special task that is
  not part of the training cluster. There could be only one. It is used for
  model evaluation.
  ```
  # This should be a JSON string, which is set as environment variable. Usually
  # the cluster manager handles that.
  TF_CONFIG='{
      "cluster": {
          "chief": ["host0:2222"],
          "worker": ["host1:2222", "host2:2222", "host3:2222"],
          "ps": ["host4:2222", "host5:2222"]
      },
      "task": {"type": "evaluator", "index": 0}
  }'
  ```

  Args:
    estimator: An `Estimator` instance to train and evaluate.
    train_spec: A `TrainSpec` instance to specify the training specification.
    eval_spec: A `EvalSpec` instance to specify the evaluation and export
      specification.

  Raises:
    ValueError: if environment variable `TF_CONFIG` is incorrectly set.
  """
  executor = _TrainingExecutor(
      estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)

  config = estimator.config
  if (config.task_type == run_config_lib.TaskType.EVALUATOR and
      config.task_id > 0):
    raise ValueError(
        'For distributed training, there can only be one `evaluator` task '
        '(with task id 0).  Given task id {}'.format(config.task_id))

  executor.run()


class _StopAtSecsHook(session_run_hook.SessionRunHook):
  """Stops given secs after begin is called."""

  def __init__(self, stop_after_secs):
    self._stop_after_secs = stop_after_secs
    self._start_time = None

  def begin(self):
    self._start_time = time.time()

  def after_run(self, run_context, run_values):
    del run_values
    if time.time() - self._start_time >= self._stop_after_secs:
      run_context.request_stop()


class _TrainingExecutor(object):
  """The executor to run `Estimator` training and evaluation.

  This implementation supports both distributed and non-distributed (aka local)
  training and evaluation based on the setting in `tf.estimator.RunConfig`.
  """

  def __init__(self,
               estimator,
               train_spec,
               eval_spec,
               train_hooks=None,
               continuous_eval_listener=None):
    if not isinstance(estimator, estimator_lib.Estimator):
      raise TypeError(
          '`estimator` must have type `tf.estimator.Estimator`. '
          'Got: {}'.format(type(estimator)))
    self._estimator = estimator

    if not isinstance(train_spec, TrainSpec):
      raise TypeError(
          '`train_spec` must have type `tf.estimator.TrainSpec`. '
          'Got: {}'.format(type(train_spec)))
    self._train_spec = train_spec

    if not isinstance(eval_spec, EvalSpec):
      raise TypeError(
          '`eval_spec` must have type `tf.estimator.EvalSpec`. '
          'Got: {}'.format(type(eval_spec)))
    self._eval_spec = eval_spec

    self._train_hooks = _validate_hooks(train_hooks)

    if (continuous_eval_listener and
        not isinstance(continuous_eval_listener, _ContinuousEvalListener)):
      raise TypeError('`continuous_eval_listener` must have type '
                      '`_ContinuousEvalListener`.')
    self._continuous_eval_listener = (
        continuous_eval_listener or _ContinuousEvalListener())

  @property
  def estimator(self):
    return self._estimator

  def run(self):
    """Executes the run_foo for task type `foo`.

    `_TrainingExecutor` predefines the procedure for task type 'chief',
    'worker', 'ps', and 'evaluator'. For task type `foo`, the corresponding
    procedure is `run_foo'. This `run` method invoke the procedure base on the
    `RunConfig.task_type`.

    Raises:
      ValueError: if the estimator.config is mis-configured.
    """
    config = self._estimator.config

    if (not config.cluster_spec and
        config.task_type != run_config_lib.TaskType.EVALUATOR):
      logging.info('Running training and evaluation locally (non-distributed).')
      self.run_local()
      return

    # Distributed case.
    if not config.task_type:
      # TODO(xiejw): Improve the error message about how to set the TF_CONFIG
      # correctly.
      raise ValueError(
          '`estimator.config` must have task_type set. This usually means '
          'TF_CONFIG environment is not set correctly.')

    if config.task_type == 'local':
      raise ValueError(
          '`task.type` in TF_CONFIG cannot be `local`. Leaving `cluster` and '
          '`task` properties in TF_CONFIG absent triggers train and evaluate '
          '`Estimator` locally (non-distributed).')

    # For task type foo, call executor.run_foo.
    available_tasks = [
        x for x in dir(self)
        if x.startswith('run_') and x != 'run_local' and
        callable(getattr(self, x))
    ]
    task_to_run = 'run_' + config.task_type
    if task_to_run not in available_tasks:
      raise ValueError(
          'Task type {} is not supported. Supported task types are {}'.format(
              config.task_type, [x[len('run_'):] for x in available_tasks]))
    getattr(self, task_to_run)()

  def run_chief(self):
    """Runs task chief."""
    # TODO(xiejw): To allow execution framework to add train hooks.
    return self._start_distributed_training()

  def run_worker(self):
    """Runs task (training) worker."""
    # TODO(xiejw): To allow execution framework to add train hooks.
    return self._start_distributed_training()

  def run_master(self):
    """Runs task master."""

    class NewCheckpointListener(
        basic_session_run_hooks.CheckpointSaverListener):

      def __init__(self, evaluator, eval_throttle_secs):
        self._evaluator = evaluator
        self._eval_throttle_secs = eval_throttle_secs

      def begin(self):
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_secs=self._eval_throttle_secs)

      def after_save(self, session, global_step_value):
        del session  # unused; required by signature.

        if self._timer.should_trigger_for_step(global_step_value):
          self._timer.update_last_triggered_step(global_step_value)
          self._evaluator.evaluate_and_export()
        else:
          logging.info('Skip the current checkpoint eval due to throttle secs '
                       '({} secs).'.format(self._eval_throttle_secs))

    # Final export signal: For any eval result with global_step >= train
    # max_steps, the evaluator will send the final export signal. There is a
    # small chance that the Estimator.train stopping logic sees a different
    # global_step value (due to global step race condition and the fact the
    # saver sees a larger value for checkpoing saving), which does not end
    # the training. When the training ends, a new checkpoint is generated, which
    # triggers the listener again. So, it could be the case the final export is
    # triggered twice.
    #
    # But here, throttle_secs will skip the next intermediate checkpoint and,
    # so, the double final export chance is very small.
    evaluator = _TrainingExecutor._Evaluator(self._estimator, self._eval_spec,
                                             self._train_spec.max_steps)

    # When the underlying `Estimator` object saves a new checkpoint, we would
    # like this callback to be called so that evaluation and export can trigger.
    saving_listeners = [
        NewCheckpointListener(evaluator, self._eval_spec.throttle_secs)
    ]
    self._start_distributed_training(saving_listeners=saving_listeners)

    if not evaluator.is_final_export_triggered:
      logging.info('Training has already ended. But the last eval is skipped '
                   'due to eval throttle_secs. Now evaluating the final '
                   'checkpoint.')
      evaluator.evaluate_and_export()

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

    def _should_stop_local_train(global_step):
      if self._train_spec.max_steps is None:
        return False
      if global_step >= self._train_spec.max_steps:
        return True
      return False

    if self._eval_spec.throttle_secs <= 0:
      raise ValueError('eval_spec.throttle_secs should be positive, given: {}.'
                       'It is used do determine how long each training '
                       'iteration should go when train and evaluate '
                       'locally.'.format(self._eval_spec.throttle_secs))

    stop_hook = _StopAtSecsHook(self._eval_spec.throttle_secs)
    train_hooks = (
        list(self._train_spec.hooks) + [stop_hook] + list(self._train_hooks))
    logging.info('Start train and evaluate loop. The evaluate will happen '
                 'after {} secs (eval_spec.throttle_secs) or training is '
                 'finished.'.format(self._eval_spec.throttle_secs))

    evaluator = _TrainingExecutor._Evaluator(self._estimator, self._eval_spec,
                                             self._train_spec.max_steps)

    while True:
      self._estimator.train(
          input_fn=self._train_spec.input_fn,
          max_steps=self._train_spec.max_steps,
          hooks=train_hooks)

      # Final export signal: For any eval result with global_step >= train
      # max_steps, the evaluator will send the final export signal. The
      # _should_stop_local_train will then end the while True as the stopping
      # condition is satisfied (both checks use the same global_step value,
      # i.e., no race condition)
      eval_result = evaluator.evaluate_and_export()

      if eval_result.status != _EvalStatus.EVALUATED:
        #  This is unexpected; should never happen.
        #  Training should always end with a new checkpoint.
        raise RuntimeError('There was no new checkpoint after the training. '
                           'Eval status: {}'.format(eval_result.status))

      if _should_stop_local_train(
          eval_result.metrics[ops.GraphKeys.GLOBAL_STEP]):
        break

  def _start_std_server(self, config):
    """Creates, starts, and returns a server_lib.Server."""
    if (not config.cluster_spec or not config.task_type or
        config.task_id is None):
      raise RuntimeError('Could not start server; be sure to specify '
                         'cluster_spec, task_type, and task in '
                         'RunConfig or set the TF_CONFIG environment variable.')

    if not config.master:
      jobs = config.cluster_spec.jobs
      if (len(jobs) == 1 and
          len(config.cluster_spec.job_tasks(jobs[0])) == 1 and
          config.task_type in _TRAINER_JOBS):
        # For distributed training, config.master is empty if and only if it has
        # a single node in the cluster spec. In this case, we should not start
        # the server.
        logging.info('Skip starting Tensorflow server as there is only one '
                     'node in the cluster.')
        return
      else:
        raise RuntimeError(
            'Could not start server; be sure to specify master in '
            'RunConfig or set the TF_CONFIG environment variable.')

    logging.info('Start Tensorflow server.')

    if config.session_config is None:
      session_config = config_pb2.ConfigProto(log_device_placement=False)
    else:
      session_config = config_pb2.ConfigProto(
          log_device_placement=False,
          gpu_options=config.session_config.gpu_options)

    server = server_lib.Server(
        config.cluster_spec,
        job_name=config.task_type,
        task_index=config.task_id,
        config=session_config,
        start=False)
    server.start()
    return server

  def _start_distributed_training(self, saving_listeners=None):
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
    start_delay_secs = 0
    if config.task_type == run_config_lib.TaskType.WORKER:
      # TODO(xiejw): Replace the hard code logic (task_id + 1) with unique id in
      # training cluster.
      start_delay_secs = min(_MAX_DELAY_SECS,
                             (config.task_id + 1) * _DELAY_SECS_PER_WORKER)
    if start_delay_secs > 0:
      logging.info('Waiting %d secs before starting training.',
                   start_delay_secs)
      time.sleep(start_delay_secs)

    self._estimator.train(
        input_fn=self._train_spec.input_fn,
        max_steps=self._train_spec.max_steps,
        hooks=list(self._train_spec.hooks) + list(self._train_hooks),
        saving_listeners=saving_listeners)

  def _start_continuous_evaluation(self):
    """Repeatedly calls `Estimator` evaluate and export until training ends."""
    start_delay_secs = self._eval_spec.start_delay_secs
    if start_delay_secs:
      logging.info('Waiting %f secs before starting eval.', start_delay_secs)
      time.sleep(start_delay_secs)

    latest_eval_result = None
    evaluator = _TrainingExecutor._Evaluator(self._estimator, self._eval_spec,
                                             self._train_spec.max_steps)

    should_early_stop = False
    while not should_early_stop:
      if (latest_eval_result and
          latest_eval_result.status == _EvalStatus.EVALUATED):
        global_step = latest_eval_result.metrics.get(ops.GraphKeys.GLOBAL_STEP)
        if (global_step and self._train_spec.max_steps and
            global_step >= self._train_spec.max_steps):
          logging.info(
              'Exiting evaluation, global_step=%s >= train max_steps=%s',
              global_step, self._train_spec.max_steps)
          return

      latest_eval_result, should_early_stop = self._execute_evaluator_once(
          evaluator, self._continuous_eval_listener,
          self._eval_spec.throttle_secs)

  def _execute_evaluator_once(self, evaluator, continuous_eval_listener,
                              throttle_secs):
    """Executes the `evaluator`."""
    start = time.time()

    eval_result = None
    should_early_stop = False

    if not continuous_eval_listener.before_eval():
      logging.info('Exiting evaluation, as requested by '
                   '_ContinuousEvalListener.before_eval.')
      should_early_stop = True
      return (eval_result, should_early_stop)

    # Final export signal: For any eval result with global_step >= train
    # max_steps, the evaluator will send the final export signal. The next
    # iteration of while loop will end the continuous eval as the stopping
    # condition is satisfied (both checks use the same global_step value,
    # i.e., no race condition)
    eval_result = evaluator.evaluate_and_export()

    if not self._continuous_eval_listener.after_eval(eval_result):
      logging.info('Exiting evaluation, as requested by '
                   '_ContinuousEvalListener.after_eval.')
      should_early_stop = True
      return (eval_result, should_early_stop)

    # Throttle if necessary.
    elapsed_time = time.time() - start
    difference = throttle_secs - elapsed_time
    if difference > 0:
      logging.info('Waiting %f secs before starting next eval run.', difference)
      time.sleep(difference)

    return (eval_result, should_early_stop)

  class _Evaluator(object):
    """A helper class to call `Estimator.evaluate` and export model."""

    def __init__(self, estimator, eval_spec, max_training_steps):
      self._estimator = estimator
      self._eval_spec = eval_spec
      self._is_final_export_triggered = False
      self._previous_ckpt_path = None
      self._last_warning_time = 0
      self._max_training_steps = max_training_steps

    @property
    def is_final_export_triggered(self):
      return self._is_final_export_triggered

    def evaluate_and_export(self):
      """Evaluate and (maybe) export the current model.

      Returns:
        An `EvalResult` instance.

      Raises:
        RuntimeError: for any unexpected internal error.
        TypeError: if evaluation result has wrong type.
      """
      latest_ckpt_path = self._estimator.latest_checkpoint()
      if not latest_ckpt_path:
        self._log_err_msg('Estimator is not trained yet. Will start an '
                          'evaluation when a checkpoint is ready.')
        return _EvalResult(status=_EvalStatus.MISSING_CHECKPOINT)

      if latest_ckpt_path == self._previous_ckpt_path:
        self._log_err_msg(
            'No new checkpoint ready for evaluation. Skip the current '
            'evaluation pass as evaluation results are expected to be same '
            'for the same checkpoint.')
        return _EvalResult(status=_EvalStatus.NO_NEW_CHECKPOINT)

      metrics = self._estimator.evaluate(
          input_fn=self._eval_spec.input_fn,
          steps=self._eval_spec.steps,
          name=self._eval_spec.name,
          checkpoint_path=latest_ckpt_path,
          hooks=self._eval_spec.hooks)

      # _EvalResult validates the metrics.
      eval_result = _EvalResult(
          status=_EvalStatus.EVALUATED,
          metrics=metrics,
          checkpoint_path=latest_ckpt_path)

      is_the_final_export = (
          eval_result.metrics[ops.GraphKeys.GLOBAL_STEP] >=
          self._max_training_steps if self._max_training_steps else False)
      self._export_eval_result(eval_result, is_the_final_export)

      if is_the_final_export:
        logging.debug('Calling exporter with the `is_the_final_export=True`.')
        self._is_final_export_triggered = True

      self._last_warning_time = 0
      self._previous_ckpt_path = latest_ckpt_path
      return eval_result

    def _log_err_msg(self, message):
      """Prints warning `message` every 10 mins."""
      current_time = time.time()
      if current_time - self._last_warning_time > 600:
        logging.warning(message)
        self._last_warning_time = current_time

    def _export_eval_result(self, eval_result, is_the_final_export):
      """Export `eval_result` according to exporters in `EvalSpec`."""
      export_dir_base = os.path.join(
          compat.as_str_any(self._estimator.model_dir),
          compat.as_str_any('export'))

      for exporter in self._eval_spec.exporters:
        exporter.export(
            estimator=self._estimator,
            export_path=os.path.join(
                compat.as_str_any(export_dir_base),
                compat.as_str_any(exporter.name)),
            checkpoint_path=eval_result.checkpoint_path,
            eval_result=eval_result.metrics,
            is_the_final_export=is_the_final_export)


class _EvalStatus(object):
  """The status of an evaluation event.

  For local training and evaluation, the status can only be `EVALUATED` as
  `Estimator.train` always generates a new checkpoint.

  For distributed training and evaluation, a separated evaluator keeps looking
  for new checkpoint. So, multiple situations might occur:

  - EVALUATED: A new checkpoint is found since last evaluation.
      `Estimator.evaluate` will be invoked.
  - MISSING_CHECKPOINT: No checkpoint can be found. Typically, this means
      the trainer has not yet produced any checkpoint.
  - NO_NEW_CHECKPOINT: No new checkpoint can be found since last evaluation.
      Typically, this means the trainer has not yet produced any new checkpoint.
  """

  EVALUATED = 'evaluated'
  MISSING_CHECKPOINT = 'missing checkpoint'
  NO_NEW_CHECKPOINT = 'no new checkpoint'


class _EvalResult(
    collections.namedtuple('EvalResult',
                           ['status', 'metrics', 'checkpoint_path'])):
  """_EvalResult holds the result of an evaluation event."""

  def __new__(cls, status, metrics=None, checkpoint_path=None):
    """Creates a validated `_EvalResult`.

    Args:
      status: See `_EvalStatus`.
      metrics: The evaluation results returned by `Estimator.evaluate`. Only set
          if status is `EVALUATED`.
      checkpoint_path: The corresponding checkpoint path for the `metrics`. Only
          set if status is `EVALUATED`.
    Returns:
      A validated `_EvalResult` object.

    Raises:
      ValueError: If validation fails.
      TypeError: If any of the arguments is not the expected type.
    """

    if status != _EvalStatus.EVALUATED:
      if metrics:
        raise ValueError(
            'metrics must be `None` if status is not {}; got status {},'
            ' metrics {}'.format(_EvalStatus.EVALUATED, status, metrics))
      if checkpoint_path:
        raise ValueError(
            'checkpoint must be `None` if status is not {}; got status {}, '
            'checkpoint_path {}'.format(_EvalStatus.EVALUATED, status,
                                        checkpoint_path))
      return super(_EvalResult, cls).__new__(cls, status, metrics,
                                             checkpoint_path)

    # Now, evaluated case.
    assert status == _EvalStatus.EVALUATED

    # Validates metrics.
    if not metrics:
      raise ValueError(
          'Internal error: `Estimator.evaluate` should never return empty '
          'metrics.')
    if not isinstance(metrics, dict):
      raise TypeError(
          '`Estimator.evaluate` should return dict. Given {}.'.format(
              type(metrics)))
    if ops.GraphKeys.GLOBAL_STEP not in metrics:
      raise ValueError(
          'Internal error: `Estimator.evaluate` result should have '
          '`global_step` in result. Given {}'.format(metrics))

    # Validates checkpoint_path.
    if not checkpoint_path:
      raise ValueError(
          'Internal error: `checkpoint_path` should never be empty.')

    return super(_EvalResult, cls).__new__(cls, status, metrics,
                                           checkpoint_path)


class _ContinuousEvalListener(object):
  """Interface for listeners that take action before or after evaluation."""

  def before_eval(self):
    """Called before evaluation.

    Returns:
      `False` if you want to skip the current evaluation and early stop the
      continuous evaluation; `True` otherwise.
    """
    return True

  def after_eval(self, eval_result):
    """Called after the evaluation is executed.

    Args:
      eval_result: An `_EvalResult` instance.

    Returns:
      False if you want to early stop continuous evaluation; `True` otherwise.
    """
    del eval_result
    return True
