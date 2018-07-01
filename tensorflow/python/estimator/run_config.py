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
"""Environment configuration object for Estimators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import os

import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat_internal
from tensorflow.python.util import function_utils
from tensorflow.python.util.tf_export import estimator_export


_USE_DEFAULT = object()
_VALID_DEVICE_FN_ARGS = set(['op'])

# A list of the property names in RunConfig that the user is allowed to change.
_DEFAULT_REPLACEABLE_LIST = [
    'model_dir',
    'tf_random_seed',
    'save_summary_steps',
    'save_checkpoints_steps',
    'save_checkpoints_secs',
    'session_config',
    'keep_checkpoint_max',
    'keep_checkpoint_every_n_hours',
    'log_step_count_steps',
    'train_distribute',
    'device_fn'
]

_SAVE_CKPT_ERR = (
    '`save_checkpoints_steps` and `save_checkpoints_secs` cannot be both set.'
)

_TF_CONFIG_ENV = 'TF_CONFIG'
_TASK_ENV_KEY = 'task'
_TASK_TYPE_KEY = 'type'
_TASK_ID_KEY = 'index'
_CLUSTER_KEY = 'cluster'
_SERVICE_KEY = 'service'
_SESSION_MASTER_KEY = 'session_master'
_EVAL_SESSION_MASTER_KEY = 'eval_session_master'
_MODEL_DIR_KEY = 'model_dir'
_LOCAL_MASTER = ''
_GRPC_SCHEME = 'grpc://'


def _get_session_master(cluster_spec, task_type, task_id, tf_config):
  """Returns the appropriate address for TensorFlow master.

  The order of precedence to deteremine the TF session master is as follows:
  1. If `tf_session_master` is set in TF_CONFIG environment variable, takes it.
  2. If the cluster has only one node, returns empty string ''.
  3. Returns the grpc address according to the task type and id in the cluster.
     This is between-graph replication.

  Note: task_type and task_id must be validated. Typically, validated using
  `_validate_task_type_and_task_id`.

  Args:
    cluster_spec: A `ClusterSpec` instance.
    task_type: String. Task type for current node.
    task_id: Int. Task id for current node.
    tf_config: Dict. Python dict for the TF_CONFIG environment variable.

  Raises:
    RuntimeError: If `cluster_spec` is not set.

  """
  if _SESSION_MASTER_KEY in tf_config:
    return tf_config[_SESSION_MASTER_KEY]

  if not cluster_spec:
    raise RuntimeError('Internal error: `_get_session_master` '
                       'does not expect empty cluster_spec.')

  jobs = cluster_spec.jobs

  # If there is only one node in the cluster, do things locally by setting
  # master to ''.  If a service or user sets TF_CONFIG with a single node, it's
  # more performant to use a direct master rather than an RPC service.
  if len(jobs) == 1 and len(cluster_spec.job_tasks(jobs[0])) == 1:
    return _LOCAL_MASTER

  # Lookup the master in cluster_spec using task_type and task_id,
  # if possible.
  addresses = cluster_spec.job_tasks(task_type)
  return _GRPC_SCHEME + addresses[task_id]


def _get_eval_session_master(task_type, tf_config):
  """Returns the appropriate address for TensorFlow evaluation master."""
  if task_type == TaskType.EVALUATOR:
    return tf_config.get(_EVAL_SESSION_MASTER_KEY, _LOCAL_MASTER)

  if _EVAL_SESSION_MASTER_KEY in tf_config:
    raise ValueError('Key ({}) should not be set for task type other than {}. '
                     'Task type: {}'.format(_EVAL_SESSION_MASTER_KEY,
                                            TaskType.EVALUATOR, task_type))
  return _LOCAL_MASTER


def _count_ps(cluster_spec):
  """Counts the number of parameter servers in cluster_spec."""
  if not cluster_spec:
    raise RuntimeError(
        'Internal error: `_count_ps` does not expect empty cluster_spec.')

  return len(cluster_spec.as_dict().get(TaskType.PS, []))


def _count_worker(cluster_spec, chief_task_type):
  """Counts the number of workers (including chief) in cluster_spec."""
  if not cluster_spec:
    raise RuntimeError(
        'Internal error: `_count_worker` does not expect empty cluster_spec.')

  return (len(cluster_spec.as_dict().get(TaskType.WORKER, [])) +
          len(cluster_spec.as_dict().get(chief_task_type, [])))


def _validate_service(service):
  """Validates the service key."""
  if service is not None and not isinstance(service, dict):
    raise TypeError(
        'If "service" is set in TF_CONFIG, it must be a dict. Given %s' %
        type(service))
  return service


def _validate_task_type_and_task_id(cluster_spec, task_env, chief_task_type):
  """Validates the task type and index in `task_env` according to cluster."""
  if chief_task_type not in cluster_spec.jobs:
    raise ValueError(
        'If "cluster" is set in TF_CONFIG, it must have one "%s" node.' %
        chief_task_type)
  if len(cluster_spec.job_tasks(chief_task_type)) > 1:
    raise ValueError(
        'The "cluster" in TF_CONFIG must have only one "%s" node.' %
        chief_task_type)

  task_type = task_env.get(_TASK_TYPE_KEY, None)
  task_id = task_env.get(_TASK_ID_KEY, None)

  if not task_type:
    raise ValueError(
        'If "cluster" is set in TF_CONFIG, task type must be set.')
  if task_id is None:
    raise ValueError(
        'If "cluster" is set in TF_CONFIG, task index must be set.')

  task_id = int(task_id)

  # Check the task id bounds. Upper bound is not necessary as
  # - for evaluator, there is no upper bound.
  # - for non-evaluator, task id is upper bounded by the number of jobs in
  # cluster spec, which will be checked later (when retrieving the `master`)
  if task_id < 0:
    raise ValueError('Task index must be non-negative number.')

  # Evaluator is not part of the training cluster.
  if task_type == TaskType.EVALUATOR:
    return task_type, task_id

  if task_type not in cluster_spec.jobs:
    raise ValueError(
        '%s is not a valid task_type in the cluster_spec:\n'
        '%s\n\n'
        'Note that these values may be coming from the TF_CONFIG environment '
        'variable.' % (task_type, cluster_spec))
  addresses = cluster_spec.job_tasks(task_type)
  if not 0 <= task_id < len(addresses):
    raise ValueError(
        '%d is not a valid task_id for task_type %s in the cluster_spec:\n'
        '%s\n\n'
        'Note that these values may be coming from the TF_CONFIG environment '
        'variable.' % (task_id, task_type, cluster_spec))

  return task_type, task_id


def _get_global_id_in_cluster(
    cluster_spec, task_type, task_id, chief_task_type):
  """Returns the global id in cluster."""
  # Note: This is implementation details, which user should not rely on.
  # The first id is 0, which is always for the `chief` node. All other nodes,
  # except `ps`, are ordered alphabetical based on task type (alphabetically)
  # and task id (ascendingly). `ps` are ordered last.

  # Sort task names in cluster
  task_type_ordered_list = [chief_task_type]
  task_type_ordered_list.extend([
      t for t in sorted(cluster_spec.jobs)
      if t != chief_task_type and t != TaskType.PS
  ])
  if TaskType.PS in cluster_spec.jobs:
    task_type_ordered_list.append(TaskType.PS)

  next_global_id = 0
  for t in task_type_ordered_list:
    if t == task_type:
      return next_global_id + task_id
    next_global_id += len(cluster_spec.job_tasks(t))

  # This should never happen.
  raise RuntimeError('Internal Error: `task_type` ({}) is not in '
                     'cluster_spec ({}).'.format(task_type, cluster_spec))


def _validate_save_ckpt_with_replaced_keys(new_copy, replaced_keys):
  """Validates the save ckpt properties."""
  # Ensure one (and only one) of save_steps and save_secs is not None.
  # Also, if user sets one save ckpt property, say steps, the other one (secs)
  # should be set as None to improve usability.

  save_steps = new_copy.save_checkpoints_steps
  save_secs = new_copy.save_checkpoints_secs

  if ('save_checkpoints_steps' in replaced_keys and
      'save_checkpoints_secs' in replaced_keys):
    # If user sets both properties explicitly, we need to error out if both
    # are set or neither of them are set.
    if save_steps is not None and save_secs is not None:
      raise ValueError(_SAVE_CKPT_ERR)
  elif 'save_checkpoints_steps' in replaced_keys and save_steps is not None:
    new_copy._save_checkpoints_secs = None  # pylint: disable=protected-access
  elif 'save_checkpoints_secs' in replaced_keys and save_secs is not None:
    new_copy._save_checkpoints_steps = None  # pylint: disable=protected-access


def _validate_properties(run_config):
  """Validates the properties."""
  def _validate(property_name, cond, message):
    property_value = getattr(run_config, property_name)
    if property_value is not None and not cond(property_value):
      raise ValueError(message)

  _validate('model_dir', lambda dir: dir,
            message='model_dir should be non-empty')

  _validate('save_summary_steps', lambda steps: steps >= 0,
            message='save_summary_steps should be >= 0')

  _validate('save_checkpoints_steps', lambda steps: steps >= 0,
            message='save_checkpoints_steps should be >= 0')
  _validate('save_checkpoints_secs', lambda secs: secs >= 0,
            message='save_checkpoints_secs should be >= 0')

  _validate('session_config',
            lambda sc: isinstance(sc, config_pb2.ConfigProto),
            message='session_config must be instance of ConfigProto')

  _validate('keep_checkpoint_max', lambda keep_max: keep_max >= 0,
            message='keep_checkpoint_max should be >= 0')
  _validate('keep_checkpoint_every_n_hours', lambda keep_hours: keep_hours > 0,
            message='keep_checkpoint_every_n_hours should be > 0')
  _validate('log_step_count_steps', lambda num_steps: num_steps > 0,
            message='log_step_count_steps should be > 0')

  _validate('tf_random_seed', lambda seed: isinstance(seed, six.integer_types),
            message='tf_random_seed must be integer.')

  _validate('device_fn', lambda device_fn: six.callable(device_fn) and
            set(function_utils.fn_args(device_fn)) == _VALID_DEVICE_FN_ARGS,
            message='device_fn must be callable with exactly'
                    ' one argument "op".')


class TaskType(object):
  MASTER = 'master'
  PS = 'ps'
  WORKER = 'worker'
  CHIEF = 'chief'
  EVALUATOR = 'evaluator'


@estimator_export('estimator.RunConfig')
class RunConfig(object):
  """This class specifies the configurations for an `Estimator` run."""

  def __init__(self,
               model_dir=None,
               tf_random_seed=None,
               save_summary_steps=100,
               save_checkpoints_steps=_USE_DEFAULT,
               save_checkpoints_secs=_USE_DEFAULT,
               session_config=None,
               keep_checkpoint_max=5,
               keep_checkpoint_every_n_hours=10000,
               log_step_count_steps=100,
               train_distribute=None,
               device_fn=None):
    """Constructs a RunConfig.

    All distributed training related properties `cluster_spec`, `is_chief`,
    `master` , `num_worker_replicas`, `num_ps_replicas`, `task_id`, and
    `task_type` are set based on the `TF_CONFIG` environment variable, if the
    pertinent information is present. The `TF_CONFIG` environment variable is a
    JSON object with attributes: `cluster` and `task`.

    `cluster` is a JSON serialized version of `ClusterSpec`'s Python dict from
    `server_lib.py`, mapping task types (usually one of the `TaskType` enums) to
    a list of task addresses.

    `task` has two attributes: `type` and `index`, where `type` can be any of
    the task types in `cluster`. When `TF_CONFIG` contains said information,
    the following properties are set on this class:

    * `cluster_spec` is parsed from `TF_CONFIG['cluster']`. Defaults to {}. If
      present, must have one and only one node in the `chief` attribute of
      `cluster_spec`.
    * `task_type` is set to `TF_CONFIG['task']['type']`. Must set if
      `cluster_spec` is present; must be `worker` (the default value) if
      `cluster_spec` is not set.
    * `task_id` is set to `TF_CONFIG['task']['index']`. Must set if
      `cluster_spec` is present; must be 0 (the default value) if
      `cluster_spec` is not set.
    * `master` is determined by looking up `task_type` and `task_id` in the
      `cluster_spec`. Defaults to ''.
    * `num_ps_replicas` is set by counting the number of nodes listed
      in the `ps` attribute of `cluster_spec`. Defaults to 0.
    * `num_worker_replicas` is set by counting the number of nodes listed
      in the `worker` and `chief` attributes of `cluster_spec`. Defaults to 1.
    * `is_chief` is determined based on `task_type` and `cluster`.

    There is a special node with `task_type` as `evaluator`, which is not part
    of the (training) `cluster_spec`. It handles the distributed evaluation job.

    Example of non-chief node:
    ```
      cluster = {'chief': ['host0:2222'],
                 'ps': ['host1:2222', 'host2:2222'],
                 'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
      os.environ['TF_CONFIG'] = json.dumps(
          {'cluster': cluster,
           'task': {'type': 'worker', 'index': 1}})
      config = RunConfig()
      assert config.master == 'host4:2222'
      assert config.task_id == 1
      assert config.num_ps_replicas == 2
      assert config.num_worker_replicas == 4
      assert config.cluster_spec == server_lib.ClusterSpec(cluster)
      assert config.task_type == 'worker'
      assert not config.is_chief
    ```

    Example of chief node:
    ```
      cluster = {'chief': ['host0:2222'],
                 'ps': ['host1:2222', 'host2:2222'],
                 'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
      os.environ['TF_CONFIG'] = json.dumps(
          {'cluster': cluster,
           'task': {'type': 'chief', 'index': 0}})
      config = RunConfig()
      assert config.master == 'host0:2222'
      assert config.task_id == 0
      assert config.num_ps_replicas == 2
      assert config.num_worker_replicas == 4
      assert config.cluster_spec == server_lib.ClusterSpec(cluster)
      assert config.task_type == 'chief'
      assert config.is_chief
    ```

    Example of evaluator node (evaluator is not part of training cluster):
    ```
      cluster = {'chief': ['host0:2222'],
                 'ps': ['host1:2222', 'host2:2222'],
                 'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
      os.environ['TF_CONFIG'] = json.dumps(
          {'cluster': cluster,
           'task': {'type': 'evaluator', 'index': 0}})
      config = RunConfig()
      assert config.master == ''
      assert config.evaluator_master == ''
      assert config.task_id == 0
      assert config.num_ps_replicas == 0
      assert config.num_worker_replicas == 0
      assert config.cluster_spec == {}
      assert config.task_type == 'evaluator'
      assert not config.is_chief
    ```

    N.B.: If `save_checkpoints_steps` or `save_checkpoints_secs` is set,
    `keep_checkpoint_max` might need to be adjusted accordingly, especially in
    distributed training. For example, setting `save_checkpoints_secs` as 60
    without adjusting `keep_checkpoint_max` (defaults to 5) leads to situation
    that checkpoint would be garbage collected after 5 minutes. In distributed
    training, the evaluation job starts asynchronously and might fail to load or
    find the checkpoint due to race condition.

    Args:
      model_dir: directory where model parameters, graph, etc are saved. If
        `PathLike` object, the path will be resolved. If `None`, will use a
        default value set by the Estimator.
      tf_random_seed: Random seed for TensorFlow initializers.
        Setting this value allows consistency between reruns.
      save_summary_steps: Save summaries every this many steps.
      save_checkpoints_steps: Save checkpoints every this many steps. Can not be
          specified with `save_checkpoints_secs`.
      save_checkpoints_secs: Save checkpoints every this many seconds. Can not
          be specified with `save_checkpoints_steps`. Defaults to 600 seconds if
          both `save_checkpoints_steps` and `save_checkpoints_secs` are not set
          in constructor.  If both `save_checkpoints_steps` and
          `save_checkpoints_secs` are None, then checkpoints are disabled.
      session_config: a ConfigProto used to set session parameters, or None.
      keep_checkpoint_max: The maximum number of recent checkpoint files to
        keep. As new files are created, older files are deleted. If None or 0,
        all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent
        checkpoint files are kept.)
      keep_checkpoint_every_n_hours: Number of hours between each checkpoint
        to be saved. The default value of 10,000 hours effectively disables
        the feature.
      log_step_count_steps: The frequency, in number of global steps, that the
        global step/sec and the loss will be logged during training.
      train_distribute: an optional instance of
        `tf.contrib.distribute.DistributionStrategy`. If specified,
        then Estimator will distribute the user's model during training,
        according to the policy specified by that strategy.
      device_fn: A callable invoked for every `Operation` that takes the
        `Operation` and returns the device string. If `None`, defaults to
        the device function returned by `tf.train.replica_device_setter`
        with round-robin strategy.

    Raises:
      ValueError: If both `save_checkpoints_steps` and `save_checkpoints_secs`
      are set.
    """
    if (save_checkpoints_steps == _USE_DEFAULT and
        save_checkpoints_secs == _USE_DEFAULT):
      save_checkpoints_steps = None
      save_checkpoints_secs = 600
    elif save_checkpoints_secs == _USE_DEFAULT:
      save_checkpoints_secs = None
    elif save_checkpoints_steps == _USE_DEFAULT:
      save_checkpoints_steps = None
    elif (save_checkpoints_steps is not None and
          save_checkpoints_secs is not None):
      raise ValueError(_SAVE_CKPT_ERR)

    tf_config = json.loads(os.environ.get(_TF_CONFIG_ENV, '{}'))
    if tf_config:
      logging.info('TF_CONFIG environment variable: %s', tf_config)

    model_dir = _get_model_dir(tf_config,
                               compat_internal.path_to_str(model_dir))

    RunConfig._replace(
        self,
        allowed_properties_list=_DEFAULT_REPLACEABLE_LIST,
        model_dir=model_dir,
        tf_random_seed=tf_random_seed,
        save_summary_steps=save_summary_steps,
        save_checkpoints_steps=save_checkpoints_steps,
        save_checkpoints_secs=save_checkpoints_secs,
        session_config=session_config,
        keep_checkpoint_max=keep_checkpoint_max,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        log_step_count_steps=log_step_count_steps,
        train_distribute=train_distribute,
        device_fn=device_fn)

    self._init_distributed_setting_from_environment_var(tf_config)

    # Get session_config only for distributed mode (cluster_spec is present).
    if not self._session_config and self._cluster_spec:
      RunConfig._replace(
          self,
          allowed_properties_list=_DEFAULT_REPLACEABLE_LIST,
          session_config=self._get_default_session_config())

  def _get_default_session_config(self):
    """Returns None or tf.ConfigProto instance with default device_filters set.

    Device filters are set such that chief/master and worker communicates with
    only ps. session_config=None for evaluators or any other TaskType.
    """

    rewrite_opts = rewriter_config_pb2.RewriterConfig(
        meta_optimizer_iterations=rewriter_config_pb2.RewriterConfig.ONE)
    graph_opts = config_pb2.GraphOptions(rewrite_options=rewrite_opts)

    device_filters = None
    if self._task_type == TaskType.MASTER:
      device_filters = ['/job:ps', '/job:master']
    elif self._task_type == TaskType.CHIEF:
      device_filters = ['/job:ps', '/job:chief']
    elif self._task_type == TaskType.WORKER:
      device_filters = ['/job:ps', '/job:worker/task:%d' % self._task_id]
    elif self._task_type == TaskType.PS:
      device_filters = ['/job:ps', '/job:worker', '/job:master']
    else:
      # If the task_type is `EVALUATOR` or something other than the ones in
      # TaskType then don't set any device filters.
      return None

    return config_pb2.ConfigProto(
        allow_soft_placement=True,
        graph_options=graph_opts,
        device_filters=device_filters)

  def _init_distributed_setting_from_environment_var(self, tf_config):
    """Initialize distributed properties based on `tf_config`."""

    self._service = _validate_service(tf_config.get(_SERVICE_KEY))
    self._cluster_spec = server_lib.ClusterSpec(tf_config.get(_CLUSTER_KEY, {}))
    task_env = tf_config.get(_TASK_ENV_KEY, {})

    if self._cluster_spec and TaskType.MASTER in self._cluster_spec.jobs:
      return self._init_distributed_setting_from_environment_var_with_master(
          tf_config)

    if self._cluster_spec:
      # Distributed mode.
      self._task_type, self._task_id = _validate_task_type_and_task_id(
          self._cluster_spec, task_env, TaskType.CHIEF)

      self._evaluation_master = _get_eval_session_master(
          self._task_type, tf_config)

      if self._task_type != TaskType.EVALUATOR:
        self._master = _get_session_master(self._cluster_spec, self._task_type,
                                           self._task_id, tf_config)
        self._num_ps_replicas = _count_ps(self._cluster_spec)
        self._num_worker_replicas = _count_worker(
            self._cluster_spec, chief_task_type=TaskType.CHIEF)
        self._global_id_in_cluster = _get_global_id_in_cluster(
            self._cluster_spec,
            self._task_type,
            self._task_id,
            chief_task_type=TaskType.CHIEF)
      else:
        # Evaluator is not part of the training cluster.
        self._cluster_spec = server_lib.ClusterSpec({})
        self._master = _LOCAL_MASTER
        self._num_ps_replicas = 0
        self._num_worker_replicas = 0
        self._global_id_in_cluster = None  # undefined

      self._is_chief = self._task_type == TaskType.CHIEF
    else:
      # Local mode.
      self._task_type = task_env.get(_TASK_TYPE_KEY, TaskType.WORKER)
      self._task_id = int(task_env.get(_TASK_ID_KEY, 0))
      self._global_id_in_cluster = 0

      if self._task_type != TaskType.WORKER:
        raise ValueError(
            'If "cluster" is not set in TF_CONFIG, task type must be WORKER.')
      if self._task_id != 0:
        raise ValueError(
            'If "cluster" is not set in TF_CONFIG, task index must be 0.')

      self._master = tf_config.get(_SESSION_MASTER_KEY, _LOCAL_MASTER)
      self._evaluation_master = tf_config.get(_EVAL_SESSION_MASTER_KEY,
                                              _LOCAL_MASTER)
      self._is_chief = True
      self._num_ps_replicas = 0
      self._num_worker_replicas = 1

  def _init_distributed_setting_from_environment_var_with_master(self,
                                                                 tf_config):
    """Initialize distributed properties for legacy cluster with `master`."""
    # There is no tech reason, why user cannot have chief and master in the same
    # cluster, but it is super confusing (which is really the chief?). So, block
    # this case.
    if TaskType.CHIEF in self._cluster_spec.jobs:
      raise ValueError('If `master` node exists in `cluster`, job '
                       '`chief` is not supported.')

    task_env = tf_config.get(_TASK_ENV_KEY, {})

    self._task_type, self._task_id = _validate_task_type_and_task_id(
        self._cluster_spec, task_env, TaskType.MASTER)

    if self._task_type == TaskType.EVALUATOR:
      raise ValueError('If `master` node exists in `cluster`, task_type '
                       '`evaluator` is not supported.')

    self._global_id_in_cluster = _get_global_id_in_cluster(
        self._cluster_spec,
        self._task_type,
        self._task_id,
        chief_task_type=TaskType.MASTER)

    self._master = _get_session_master(self._cluster_spec, self._task_type,
                                       self._task_id, tf_config)
    self._evaluation_master = _get_eval_session_master(self._task_type,
                                                       tf_config)
    self._num_ps_replicas = _count_ps(self._cluster_spec)
    self._num_worker_replicas = _count_worker(
        self._cluster_spec, chief_task_type=TaskType.MASTER)

    self._is_chief = self._task_type == TaskType.MASTER

  @property
  def cluster_spec(self):
    return self._cluster_spec

  @property
  def device_fn(self):
    """Returns the device_fn.

    If device_fn is not `None`, it overrides the default
    device function used in `Estimator`.
    Otherwise the default one is used.
    """
    return self._device_fn

  @property
  def evaluation_master(self):
    return self._evaluation_master

  @property
  def is_chief(self):
    return self._is_chief

  @property
  def master(self):
    return self._master

  @property
  def num_ps_replicas(self):
    return self._num_ps_replicas

  @property
  def num_worker_replicas(self):
    return self._num_worker_replicas

  @property
  def task_id(self):
    return self._task_id

  @property
  def global_id_in_cluster(self):
    """The global id in the training cluster.

    All global ids in the training cluster are assigned from an increasing
    sequence of consecutive integers. The first id is 0.

    Note: Task id (the property field `task_id`) is tracking the index of the
    node among all nodes with the SAME task type. For example, given the cluster
    definition as follows:

    ```
      cluster = {'chief': ['host0:2222'],
                 'ps': ['host1:2222', 'host2:2222'],
                 'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
    ```

    Nodes with task type `worker` can have id 0, 1, 2.  Nodes with task type
    `ps` can have id, 0, 1. So, `task_id` is not unique, but the pair
    (`task_type`, `task_id`) can uniquely determine a node in the cluster.

    Global id, i.e., this field, is tracking the index of the node among ALL
    nodes in the cluster. It is uniquely assigned.  For example, for the cluster
    spec given above, the global ids are assigned as:
    ```
      task_type  | task_id  |  global_id
      --------------------------------
      chief      | 0        |  0
      worker     | 0        |  1
      worker     | 1        |  2
      worker     | 2        |  3
      ps         | 0        |  4
      ps         | 1        |  5
    ```

    Returns:
      An integer id.
    """
    return self._global_id_in_cluster

  @property
  def task_type(self):
    return self._task_type

  @property
  def tf_random_seed(self):
    return self._tf_random_seed

  @property
  def save_summary_steps(self):
    return self._save_summary_steps

  @property
  def save_checkpoints_secs(self):
    return self._save_checkpoints_secs

  @property
  def session_config(self):
    return self._session_config

  @property
  def save_checkpoints_steps(self):
    return self._save_checkpoints_steps

  @property
  def keep_checkpoint_max(self):
    return self._keep_checkpoint_max

  @property
  def keep_checkpoint_every_n_hours(self):
    return self._keep_checkpoint_every_n_hours

  @property
  def log_step_count_steps(self):
    return self._log_step_count_steps

  @property
  def model_dir(self):
    return self._model_dir

  @property
  def service(self):
    """Returns the platform defined (in TF_CONFIG) service dict."""
    return self._service

  @property
  def train_distribute(self):
    """Returns the optional `tf.contrib.distribute.DistributionStrategy` object.
    """
    return self._train_distribute

  def replace(self, **kwargs):
    """Returns a new instance of `RunConfig` replacing specified properties.

    Only the properties in the following list are allowed to be replaced:

      - `model_dir`,
      - `tf_random_seed`,
      - `save_summary_steps`,
      - `save_checkpoints_steps`,
      - `save_checkpoints_secs`,
      - `session_config`,
      - `keep_checkpoint_max`,
      - `keep_checkpoint_every_n_hours`,
      - `log_step_count_steps`,
      - `train_distribute`,
      - `device_fn`.

    In addition, either `save_checkpoints_steps` or `save_checkpoints_secs`
    can be set (should not be both).

    Args:
      **kwargs: keyword named properties with new values.

    Raises:
      ValueError: If any property name in `kwargs` does not exist or is not
        allowed to be replaced, or both `save_checkpoints_steps` and
        `save_checkpoints_secs` are set.

    Returns:
      a new instance of `RunConfig`.
    """
    return RunConfig._replace(
        copy.deepcopy(self),
        allowed_properties_list=_DEFAULT_REPLACEABLE_LIST,
        **kwargs)

  @staticmethod
  def _replace(config, allowed_properties_list=None, **kwargs):
    """See `replace`.

    N.B.: This implementation assumes that for key named "foo", the underlying
    property the RunConfig holds is "_foo" (with one leading underscore).

    Args:
      config: The RunConfig to replace the values of.
      allowed_properties_list: The property name list allowed to be replaced.
      **kwargs: keyword named properties with new values.

    Raises:
      ValueError: If any property name in `kwargs` does not exist or is not
        allowed to be replaced, or both `save_checkpoints_steps` and
        `save_checkpoints_secs` are set.

    Returns:
      a new instance of `RunConfig`.
    """

    allowed_properties_list = allowed_properties_list or []

    for key, new_value in six.iteritems(kwargs):
      if key in allowed_properties_list:
        setattr(config, '_' + key, new_value)
        continue

      raise ValueError(
          'Replacing {} is not supported. Allowed properties are {}.'.format(
              key, allowed_properties_list))

    _validate_save_ckpt_with_replaced_keys(config, kwargs.keys())
    _validate_properties(config)
    return config


def _get_model_dir(tf_config, model_dir):
  """Returns `model_dir` based user provided `tf_config` or `model_dir`."""
  # pylint: disable=g-explicit-bool-comparison

  # Empty string is treated as False in Python condition check, which triggers
  # some confusing error messages. For example, 'a or b' returns None if a is ''
  # and b is None. `None` is allowed for model_dir but '' is not allowed. Here,
  # explicitly check empty string to provide clear error message.
  if model_dir == '':
    raise ValueError('model_dir should be non-empty.')

  model_dir_in_tf_config = tf_config.get('model_dir')
  if model_dir_in_tf_config == '':
    raise ValueError('model_dir in TF_CONFIG should be non-empty.')

  if model_dir_in_tf_config:
    if model_dir and model_dir_in_tf_config != model_dir:
      raise ValueError(
          '`model_dir` provided in RunConfig construct, if set, '
          'must have the same value as the model_dir in TF_CONFIG. '
          'model_dir: {}\nTF_CONFIG["model_dir"]: {}.\n'.format(
              model_dir, model_dir_in_tf_config))

    logging.info('Using model_dir in TF_CONFIG: %s', model_dir_in_tf_config)

  return model_dir or model_dir_in_tf_config
