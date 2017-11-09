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
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib


_USE_DEFAULT = object()

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
    'log_step_count_steps'
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
_LOCAL_MASTER = ''
_GRPC_SCHEME = 'grpc://'


def _get_master(cluster_spec, task_type, task_id):
  """Returns the appropriate string for the TensorFlow master."""
  if not cluster_spec:
    raise RuntimeError(
        'Internal error: `_get_master` does not expect empty cluster_spec.')

  jobs = cluster_spec.jobs
  # Lookup the master in cluster_spec using task_type and task_id,
  # if possible.
  if task_type not in jobs:
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
  return _GRPC_SCHEME + addresses[task_id]


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
  return task_type, task_id


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


class TaskType(object):
  MASTER = 'master'
  PS = 'ps'
  WORKER = 'worker'
  CHIEF = 'chief'
  EVALUATOR = 'evaluator'


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
               log_step_count_steps=100):
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
    the task types in `cluster`. ` When `TF_CONFIG` contains said information,
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
      config = ClusterConfig()
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
      config = ClusterConfig()
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
      config = ClusterConfig()
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
        `None`, will use a default value set by the Estimator.
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
        global step/sec will be logged during training.


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
        log_step_count_steps=log_step_count_steps)

    self._init_distributed_setting_from_environment_var()

  def _init_distributed_setting_from_environment_var(self):
    """Initialize distributed properties based on environment variable."""

    tf_config = json.loads(os.environ.get(_TF_CONFIG_ENV) or '{}')
    if tf_config:
      logging.info('TF_CONFIG environment variable: %s', tf_config)

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

      if self._task_type != TaskType.EVALUATOR:
        self._master = _get_master(
            self._cluster_spec, self._task_type, self._task_id)
        self._num_ps_replicas = _count_ps(self._cluster_spec)
        self._num_worker_replicas = _count_worker(
            self._cluster_spec, chief_task_type=TaskType.CHIEF)
      else:
        # Evaluator is not part of the training cluster.
        self._cluster_spec = server_lib.ClusterSpec({})
        self._master = _LOCAL_MASTER
        self._num_ps_replicas = 0
        self._num_worker_replicas = 0

      self._is_chief = self._task_type == TaskType.CHIEF
    else:
      # Local mode.
      self._task_type = task_env.get(_TASK_TYPE_KEY, TaskType.WORKER)
      self._task_id = int(task_env.get(_TASK_ID_KEY, 0))

      if self._task_type != TaskType.WORKER:
        raise ValueError(
            'If "cluster" is not set in TF_CONFIG, task type must be WORKER.')
      if self._task_id != 0:
        raise ValueError(
            'If "cluster" is not set in TF_CONFIG, task index must be 0.')

      self._master = ''
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

    self._master = _get_master(
        self._cluster_spec, self._task_type, self._task_id)
    self._num_ps_replicas = _count_ps(self._cluster_spec)
    self._num_worker_replicas = _count_worker(
        self._cluster_spec, chief_task_type=TaskType.MASTER)

    self._is_chief = self._task_type == TaskType.MASTER

  @property
  def cluster_spec(self):
    return self._cluster_spec

  @property
  def evaluation_master(self):
    return ''

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

  def replace(self, **kwargs):
    """Returns a new instance of `RunConfig` replacing specified properties.

    Only the properties in the following list are allowed to be replaced:

      - `model_dir`.
      - `tf_random_seed`,
      - `save_summary_steps`,
      - `save_checkpoints_steps`,
      - `save_checkpoints_secs`,
      - `session_config`,
      - `keep_checkpoint_max`,
      - `keep_checkpoint_every_n_hours`,
      - `log_step_count_steps`,

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
