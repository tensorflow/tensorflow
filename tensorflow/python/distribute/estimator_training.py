# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Training utilities for Estimator to use Distribute Coordinator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import six

from tensorflow.python.distribute import distribute_coordinator as dc
from tensorflow.python.distribute import distribute_coordinator_context as dc_context
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib

# pylint: disable=protected-access
CHIEF = dc._TaskType.CHIEF
EVALUATOR = dc._TaskType.EVALUATOR
PS = dc._TaskType.PS
WORKER = dc._TaskType.WORKER

# pylint: enable=protected-access


def _count_ps(cluster_spec):
  """Counts the number of parameter servers in cluster_spec."""
  if not cluster_spec:
    raise RuntimeError(
        'Internal error: `_count_ps` does not expect empty cluster_spec.')

  return len(cluster_spec.as_dict().get(PS, []))


def _count_worker(cluster_spec, chief_task_type):
  """Counts the number of workers (including chief) in cluster_spec."""
  if not cluster_spec:
    raise RuntimeError(
        'Internal error: `_count_worker` does not expect empty cluster_spec.')

  return (len(cluster_spec.as_dict().get(WORKER, [])) + len(
      cluster_spec.as_dict().get(chief_task_type, [])))


def _get_global_id(cluster_spec, task_type, task_id, chief_task_type):
  """Returns the global id of the given task type in a cluster."""
  if not task_type:
    return 0

  # Sort task names in cluster by "chief"/"master", "evaluator", "worker"
  # and "ps". More details can be found at the documentation of
  # `tf.estimator.RunConfig.global_id_in_cluster`.
  task_type_ordered_list = []
  if chief_task_type in cluster_spec.jobs:
    task_type_ordered_list = [chief_task_type]
  task_type_ordered_list.extend([
      t for t in sorted(cluster_spec.jobs) if t != chief_task_type and t != PS
  ])
  if PS in cluster_spec.jobs:
    task_type_ordered_list.append(PS)

  # Find the right gloabl_id for current task.
  next_global_id = 0
  for t in task_type_ordered_list:
    if t == task_type:
      return next_global_id + task_id
    # `cluster_spec.job_tasks` returns all task addresses of type `t`.
    next_global_id += len(cluster_spec.job_tasks(t))

  # It is unexpected that it passes through all task_types in
  # `task_type_ordered_list`.
  raise RuntimeError('Internal Error: `task_type` ({}) is not in '
                     'cluster_spec ({}).'.format(task_type, cluster_spec))


def _init_run_config_from_worker_context(config, worker_context):
  """Initializes run config from distribute coordinator's worker context."""

  # pylint: disable=protected-access
  config._service = None
  config._cluster_spec = worker_context.cluster_spec
  config._task_type = worker_context.task_type
  config._task_id = worker_context.task_id
  config._evaluation_master = worker_context.master_target
  config._master = worker_context.master_target
  config._is_chief = worker_context.is_chief

  if config._cluster_spec:
    # Distributed mode.
    if config._task_type != EVALUATOR:

      config._num_ps_replicas = _count_ps(config._cluster_spec)
      config._num_worker_replicas = _count_worker(
          config._cluster_spec, chief_task_type=CHIEF)
      config._global_id_in_cluster = _get_global_id(
          config._cluster_spec,
          config._task_type,
          config._task_id,
          chief_task_type=CHIEF)
    else:
      # Evaluator task should not be aware of the other tasks.
      config._cluster_spec = server_lib.ClusterSpec({})
      config._num_ps_replicas = 0
      config._num_worker_replicas = 0
      config._global_id_in_cluster = None  # undefined
  else:
    # Local mode.
    config._global_id_in_cluster = 0
    config._num_ps_replicas = 0
    config._num_worker_replicas = 1


def init_run_config(config, tf_config):
  """Initializes RunConfig for distribution strategies."""
  # pylint: disable=protected-access
  if (config._experimental_distribute and
      config._experimental_distribute.train_distribute):
    if config._train_distribute:
      raise ValueError('Either `train_distribute` or'
                       '`experimental_distribute.train_distribute` can be set.')
    config._train_distribute = config._experimental_distribute.train_distribute

  if (config._experimental_distribute and
      config._experimental_distribute.eval_distribute):
    if config._eval_distribute:
      raise ValueError('Either `eval_distribute` or'
                       '`experimental_distribute.eval_distribute` can be set.')
    config._eval_distribute = config._experimental_distribute.eval_distribute

  cluster_spec = server_lib.ClusterSpec(tf_config.get('cluster', {}))
  config._init_distributed_setting_from_environment_var({})

  # Use distribute coordinator with STANDALONE_CLIENT mode if
  # `experimental_distribute.remote_cluster` is set.
  if (config._train_distribute and config._experimental_distribute and
      config._experimental_distribute.remote_cluster):
    if cluster_spec:
      raise ValueError('Cannot set both "cluster_spec" of TF_CONFIG and '
                       '`experimental_distribute.remote_cluster`')
    config._distribute_coordinator_mode = dc.CoordinatorMode.STANDALONE_CLIENT
    config._cluster_spec = config._experimental_distribute.remote_cluster
    logging.info('RunConfig initialized for Distribute Coordinator with '
                 'STANDALONE_CLIENT mode')
    return

  # Don't use distribute coordinator if it is local training or cluster has a
  # MASTER job or `train_distribute` is not specifed.
  if (not tf_config or 'master' in cluster_spec.jobs or
      not config._train_distribute):
    config._distribute_coordinator_mode = None
    config._init_distributed_setting_from_environment_var(tf_config)
    config._maybe_overwrite_session_config_for_distributed_training()
    logging.info('Not using Distribute Coordinator.')
    return

  # Use distribute coordinator with INDEPENDENT_WORKER mode otherwise.
  assert tf_config

  # Set the cluster_spec only since the distributed setting will come from
  # distribute coordinator.
  config._cluster_spec = cluster_spec
  config._distribute_coordinator_mode = dc.CoordinatorMode.INDEPENDENT_WORKER
  logging.info('RunConfig initialized for Distribute Coordinator with '
               'INDEPENDENT_WORKER mode')


def should_run_distribute_coordinator(config):
  """Checks the config to see whether to run distribute coordinator."""
  # pylint: disable=protected-access
  if (not hasattr(config, '_distribute_coordinator_mode') or
      config._distribute_coordinator_mode is None):
    logging.info('Not using Distribute Coordinator.')
    return False
  if (not isinstance(config._distribute_coordinator_mode, six.string_types) or
      config._distribute_coordinator_mode not in [
          dc.CoordinatorMode.STANDALONE_CLIENT,
          dc.CoordinatorMode.INDEPENDENT_WORKER
      ]):
    logging.warning('Unexpected distribute_coordinator_mode: %r',
                    config._distribute_coordinator_mode)
    return False
  if not config.cluster_spec:
    logging.warning('Running `train_and_evaluate` locally, ignoring '
                    '`experimental_distribute_coordinator_mode`.')
    return False
  return True


def train_and_evaluate(estimator, train_spec, eval_spec, executor_cls):
  """Run distribute coordinator for Estimator's `train_and_evaluate`.

  Args:
    estimator: An `Estimator` instance to train and evaluate.
    train_spec: A `TrainSpec` instance to specify the training specification.
    eval_spec: A `EvalSpec` instance to specify the evaluation and export
      specification.
    executor_cls: the evaluation executor class of Estimator.

  Raises:
    ValueError: if `distribute_coordinator_mode` is None in RunConfig.
  """
  run_config = estimator.config
  if not run_config._distribute_coordinator_mode:  # pylint: disable=protected-access
    raise ValueError(
        'Distribute coordinator mode is not specified in `RunConfig`.')

  def _worker_fn(strategy):
    """Function for worker task."""
    local_estimator = copy.deepcopy(estimator)
    # pylint: disable=protected-access
    local_estimator._config._train_distribute = strategy
    context = dc_context.get_current_worker_context()
    _init_run_config_from_worker_context(local_estimator._config, context)
    logging.info('Updated config: %s', str(vars(local_estimator._config)))
    local_estimator._train_distribution = strategy
    # pylint: enable=protected-access

    # In the standalone client, we don't need to run hooks on all threads
    # because logging hooks on all threads may be too much on the screen; also
    # tensor passed to one hook can only be fetched with the graph where the
    # tensor is defined. Other hooks such as checkpointing hooks will added by
    # MonitoredTrainingSession.
    # TODO(yuefengz): Is there a hook that does need to run on all threads in
    # standalone client mode?
    if (run_config._distribute_coordinator_mode ==  # pylint: disable=protected-access
        dc.CoordinatorMode.INDEPENDENT_WORKER or context.is_chief):
      hooks = list(train_spec.hooks)
    else:
      hooks = []

    # Prevent estimator.train from calling distribute coordinator again. This
    # function calls estimator.train which will use distribute coordinator path
    # again if `_distribute_coordinator_mode` is set.
    local_estimator._config._distribute_coordinator_mode = None  # pylint: disable=protected-access
    local_estimator.train(
        input_fn=train_spec.input_fn,
        max_steps=train_spec.max_steps,
        hooks=hooks)

  def _eval_fn(strategy):
    """Function for evaluator task."""
    local_estimator = copy.deepcopy(estimator)
    # pylint: disable=protected-access
    local_estimator._config._eval_distribute = strategy
    _init_run_config_from_worker_context(
        local_estimator._config, dc_context.get_current_worker_context())
    logging.info('Updated config: %s', str(vars(local_estimator._config)))
    local_estimator._eval_distribution = strategy

    # Prevent estimator.evaluate from calling distribute coordinator again. This
    # function calls estimator.evaluate which will use distribute coordinator
    # path again if `_distribute_coordinator_mode` is set.
    local_estimator._config._distribute_coordinator_mode = None  # pylint: disable=protected-access

    executor = executor_cls(local_estimator, train_spec, eval_spec)
    executor._start_continuous_evaluation()
    # pylint: enable=protected-access

  # pylint: disable=protected-access
  if (run_config._distribute_coordinator_mode ==
      dc.CoordinatorMode.STANDALONE_CLIENT):
    cluster_spec = run_config.cluster_spec
    assert cluster_spec
  else:
    # The cluster_spec comes from TF_CONFIG environment variable if it is
    # INDEPENDENT_WORKER mode.
    cluster_spec = None

  dc.run_distribute_coordinator(
      _worker_fn,
      run_config.train_distribute,
      _eval_fn,
      run_config.eval_distribute,
      mode=run_config._distribute_coordinator_mode,
      cluster_spec=cluster_spec,
      session_config=run_config.session_config)


# TODO(yuefengz): maybe merge the following two functions?
# pylint: disable=protected-access
def estimator_train(estimator, train_distributed_fn, hooks):
  """Run distribute coordinator for Estimator's `train` method."""
  assert estimator._config._distribute_coordinator_mode
  run_config = estimator._config
  assert estimator._config.cluster_spec
  cluster_spec = multi_worker_util.normalize_cluster_spec(
      estimator._config.cluster_spec)
  assert estimator._config._train_distribute

  if 'evaluator' in cluster_spec.jobs:
    raise ValueError("'evaluator' job is not supported if you don't use "
                     '`train_and_evaluate`')

  if (estimator._config._distribute_coordinator_mode !=  # pylint: disable=protected-access
      dc.CoordinatorMode.STANDALONE_CLIENT):
    raise ValueError('Only `STANDALONE_CLIENT` mode is supported when you call '
                     '`estimator.train`')

  if estimator._config._train_distribute.extended.experimental_between_graph:
    # TODO(yuefengz): remove this limitation once we figure out how to merge
    # return values from `_worker_fn`s.
    raise ValueError('`Estimator.train` API is not supported for %s with '
                     '`STANDALONE_CLIENT` mode.' %
                     estimator._config._train_distribute.__class__.__name__)

  def _worker_fn(strategy):
    """Function for worker task."""
    local_estimator = copy.deepcopy(estimator)
    local_estimator._config._train_distribute = strategy
    context = dc_context.get_current_worker_context()
    _init_run_config_from_worker_context(local_estimator._config, context)
    logging.info('Updated config: %s', str(vars(local_estimator._config)))
    local_estimator._train_distribution = strategy

    if context.is_chief:
      chief_hooks = hooks
    else:
      chief_hooks = []
    train_distributed_fn(local_estimator, strategy, chief_hooks)
    return local_estimator

  return dc.run_distribute_coordinator(
      _worker_fn,
      estimator._config.train_distribute,
      mode=run_config._distribute_coordinator_mode,
      cluster_spec=cluster_spec,
      session_config=run_config.session_config)


def estimator_evaluate(estimator, evaluate_distributed_fn, hooks):
  """Run distribute coordinator for Estimator's `evaluate` method."""
  assert estimator._config._distribute_coordinator_mode
  run_config = estimator._config
  assert estimator._config.cluster_spec
  cluster_spec = multi_worker_util.normalize_cluster_spec(
      estimator._config.cluster_spec)
  assert estimator._config._eval_distribute

  if 'evaluator' in cluster_spec.jobs:
    raise ValueError("'evaluator' job is not supported if you don't use "
                     '`train_and_evaluate`')

  if (estimator._config._distribute_coordinator_mode !=
      dc.CoordinatorMode.STANDALONE_CLIENT):
    raise ValueError('Only `STANDALONE_CLIENT` mode is supported when you call '
                     '`Estimator.evaluate`')

  if estimator._config._eval_distribute.extended.experimental_between_graph:
    # TODO(yuefengz): remove this limitation once we figure out how to merge
    # return values from `_worker_fn`s.
    raise ValueError('`Estimator.evaluate` API is not supported for %s with '
                     '`STANDALONE_CLIENT` mode.' %
                     estimator._config._eval_distribute.__class__.__name__)

  def _worker_fn(strategy):
    """Function for evaluation."""
    local_estimator = copy.deepcopy(estimator)
    local_estimator._config._eval_distribute = strategy
    context = dc_context.get_current_worker_context()
    _init_run_config_from_worker_context(local_estimator._config, context)
    logging.info('Updated config: %s', str(vars(local_estimator._config)))
    local_estimator._eval_distribution = strategy

    if context.is_chief:
      chief_hooks = hooks
    else:
      chief_hooks = []
    return evaluate_distributed_fn(local_estimator, strategy, chief_hooks)

  return dc.run_distribute_coordinator(
      _worker_fn,
      estimator._config.eval_distribute,
      mode=run_config._distribute_coordinator_mode,
      cluster_spec=cluster_spec,
      session_config=run_config.session_config)

# pylint: enable=protected-access
