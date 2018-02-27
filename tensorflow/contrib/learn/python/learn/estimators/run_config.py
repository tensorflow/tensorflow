# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Run Config (deprecated, use tf.estimator.RunConfig instead).

This module and all its submodules are deprecated. See
[contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
for migration instructions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os

import six

from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.estimator import run_config as core_run_config
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib
from tensorflow.python.util.deprecation import deprecated


# A list of the property names in RunConfig user allows to change. They will
# not affect the execution framework, so when execution framework checks the
# `uid` of the RunConfig, it should be ignored.
_DEFAULT_UID_WHITE_LIST = [
    'tf_random_seed',
    'save_summary_steps',
    'save_checkpoints_steps',
    'save_checkpoints_secs',
    'session_config',
    'keep_checkpoint_max',
    'keep_checkpoint_every_n_hours',
    'log_step_count_steps',
]


class Environment(object):
  """DEPRECATED CLASS."""
  # For running general distributed training.
  CLOUD = 'cloud'
  # For running Google-internal distributed training.
  GOOGLE = 'google'
  # For running on local desktop.
  LOCAL = 'local'


class TaskType(object):
  """DEPRECATED CLASS."""
  MASTER = 'master'
  PS = 'ps'
  WORKER = 'worker'


class ClusterConfig(object):
  """This class specifies the configurations for a distributed run.

  THIS CLASS IS DEPRECATED. Use tf.estimator.RunConfig instead.

  If you're using an `Estimator`, you should probably use the subclass
  RunConfig instead.
  """

  def __init__(self, master=None, evaluation_master=None):
    """Constructor.

    Sets the properties `cluster_spec`, `is_chief`, `master` (if `None` in the
    args), `num_ps_replicas`, `task_id`, and `task_type` based on the
    `TF_CONFIG` environment variable, if the pertinent information is
    present. The `TF_CONFIG` environment variable is a JSON object with
    attributes: `cluster`, `environment`, and `task`.

    `cluster` is a JSON serialized version of `ClusterSpec`'s Python dict from
    `server_lib.py`, mapping task types (usually one of the TaskType enums) to a
    list of task addresses.

    `environment` specifies the runtime environment for the job (usually one of
    the `Environment` enums). Defaults to `LOCAL`.

    `task` has two attributes: `type` and `index`, where `type` can be any of
    the task types in `cluster`. When `TF_CONFIG` contains said information, the
    following properties are set on this class:

    * `task_type` is set to `TF_CONFIG['task']['type']`. Defaults to `None`.
    * `task_id` is set to `TF_CONFIG['task']['index']`. Defaults to 0.
    * `cluster_spec` is parsed from `TF_CONFIG['cluster']`. Defaults to {}.
    * `master` is determined by looking up `task_type` and `task_id` in the
      `cluster_spec`. Defaults to ''.
    * `num_ps_replicas` is set by counting the number of nodes listed
      in the `ps` attribute of `cluster_spec`. Defaults to 0.
    * `num_worker_replicas` is set by counting the number of nodes listed
      in the `worker` attribute of `cluster_spec`. Defaults to 0.
    * `is_chief` is deteremined based on `task_type`, `type_id`, and
      `environment`.

    Example:
    ```
      cluster = {'ps': ['host1:2222', 'host2:2222'],
                 'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
      os.environ['TF_CONFIG'] = json.dumps(
          {'cluster': cluster,
           'task': {'type': 'worker', 'index': 1}})
      config = ClusterConfig()
      assert config.master == 'host4:2222'
      assert config.task_id == 1
      assert config.num_ps_replicas == 2
      assert config.num_worker_replicas == 3
      assert config.cluster_spec == server_lib.ClusterSpec(cluster)
      assert config.task_type == 'worker'
      assert not config.is_chief
    ```

    Args:
      master: TensorFlow master. Defaults to empty string for local.
      evaluation_master: The master on which to perform evaluation.
    """
    # If not explicitly specified in the constructor and the TF_CONFIG
    # environment variable is present, load cluster_spec from TF_CONFIG.
    config = json.loads(os.environ.get('TF_CONFIG') or '{}')

    # Set task_type and task_id if the TF_CONFIG environment variable is
    # present.  Otherwise, use the respective default (None / 0).
    task_env = config.get('task', {})
    self._task_type = task_env.get('type', None)
    self._task_id = self.get_task_id()

    self._cluster_spec = server_lib.ClusterSpec(config.get('cluster', {}))
    self._master = (master if master is not None else
                    _get_master(self._cluster_spec, self._task_type,
                                self._task_id) or '')
    self._num_ps_replicas = _count_ps(self._cluster_spec) or 0
    self._num_worker_replicas = _count_worker(self._cluster_spec) or 0

    # Set is_chief.
    self._environment = config.get('environment', Environment.LOCAL)
    self._is_chief = None
    if self._task_type is None:
      self._is_chief = (self._task_id == 0)
    elif self._environment == Environment.CLOUD:
      # When the TF_CONFIG environment variable is set, we can set the
      # default of is_chief to 0 when task_type is "master" and task_id is 0.
      self._is_chief = (self._task_type == TaskType.MASTER and
                        self._task_id == 0)
    else:
      # Legacy behavior is that is_chief is None if task_id == 0.
      self._is_chief = (self._task_type == TaskType.WORKER and
                        self._task_id == 0)

    self._evaluation_master = evaluation_master or ''

  @property
  def cluster_spec(self):
    return self._cluster_spec

  @property
  def environment(self):
    return self._environment

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
  def task_type(self):
    return self._task_type

  @staticmethod
  def get_task_id():
    """Returns task index from `TF_CONFIG` environmental variable.

    If you have a ClusterConfig instance, you can just access its task_id
    property instead of calling this function and re-parsing the environmental
    variable.

    Returns:
      `TF_CONFIG['task']['index']`. Defaults to 0.
    """
    config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    task_env = config.get('task', {})
    task_index = task_env.get('index')
    return int(task_index) if task_index else 0


class RunConfig(ClusterConfig, core_run_config.RunConfig):
  """This class specifies the configurations for an `Estimator` run.

  This class is a deprecated implementation of @{tf.estimator.RunConfig}
  interface.
  """
  _USE_DEFAULT = 0

  @deprecated(None, 'When switching to tf.estimator.Estimator, use'
              ' tf.estimator.RunConfig instead.')
  def __init__(self,
               master=None,
               num_cores=0,
               log_device_placement=False,
               gpu_memory_fraction=1,
               tf_random_seed=None,
               save_summary_steps=100,
               save_checkpoints_secs=_USE_DEFAULT,
               save_checkpoints_steps=None,
               keep_checkpoint_max=5,
               keep_checkpoint_every_n_hours=10000,
               log_step_count_steps=100,
               evaluation_master='',
               model_dir=None,
               session_config=None):
    """Constructor.

    The superclass `ClusterConfig` may set properties like `cluster_spec`,
    `is_chief`, `master` (if `None` in the args), `num_ps_replicas`, `task_id`,
    and `task_type` based on the `TF_CONFIG` environment variable. See
    `ClusterConfig` for more details.

    N.B.: If `save_checkpoints_steps` or `save_checkpoints_secs` is set,
    `keep_checkpoint_max` might need to be adjusted accordingly, especially in
    distributed training. For example, setting `save_checkpoints_secs` as 60
    without adjusting `keep_checkpoint_max` (defaults to 5) leads to situation
    that checkpoint would be garbage collected after 5 minutes. In distributed
    training, the evaluation job starts asynchronously and might fail to load or
    find the checkpoint due to race condition.

    Args:
      master: TensorFlow master. Defaults to empty string for local.
      num_cores: Number of cores to be used. If 0, the system picks an
        appropriate number (default: 0).
      log_device_placement: Log the op placement to devices (default: False).
      gpu_memory_fraction: Fraction of GPU memory used by the process on
        each GPU uniformly on the same machine.
      tf_random_seed: Random seed for TensorFlow initializers.
        Setting this value allows consistency between reruns.
      save_summary_steps: Save summaries every this many steps.
      save_checkpoints_secs: Save checkpoints every this many seconds. Can not
          be specified with `save_checkpoints_steps`.
      save_checkpoints_steps: Save checkpoints every this many steps. Can not be
          specified with `save_checkpoints_secs`.
      keep_checkpoint_max: The maximum number of recent checkpoint files to
        keep. As new files are created, older files are deleted. If None or 0,
        all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent
        checkpoint files are kept.)
      keep_checkpoint_every_n_hours: Number of hours between each checkpoint
        to be saved. The default value of 10,000 hours effectively disables
        the feature.
      log_step_count_steps: The frequency, in number of global steps, that the
        global step/sec will be logged during training.
      evaluation_master: the master on which to perform evaluation.
      model_dir: directory where model parameters, graph etc are saved. If
        `None`, will use `model_dir` property in `TF_CONFIG` environment
        variable. If both are set, must have same value. If both are `None`, see
        `Estimator` about where the model will be saved.
      session_config: a ConfigProto used to set session parameters, or None.
        Note - using this argument, it is easy to provide settings which break
        otherwise perfectly good models. Use with care.
    """
    super(RunConfig, self).__init__(
        master=master, evaluation_master=evaluation_master)

    gpu_options = config_pb2.GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    self._tf_config = config_pb2.ConfigProto(
        log_device_placement=log_device_placement,
        inter_op_parallelism_threads=num_cores,
        intra_op_parallelism_threads=num_cores,
        gpu_options=gpu_options)

    self._tf_random_seed = tf_random_seed
    self._save_summary_steps = save_summary_steps
    self._save_checkpoints_secs = save_checkpoints_secs
    self._log_step_count_steps = log_step_count_steps
    self._session_config = session_config
    if save_checkpoints_secs == RunConfig._USE_DEFAULT:
      if save_checkpoints_steps is None:
        self._save_checkpoints_secs = 600
      else:
        self._save_checkpoints_secs = None
    self._save_checkpoints_steps = save_checkpoints_steps

    # TODO(weiho): Remove these after ModelFn refactoring, when users can
    # create Scaffold and Saver in their model_fn to set these.
    self._keep_checkpoint_max = keep_checkpoint_max
    self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self._model_dir = _get_model_dir(model_dir)

  @experimental
  def uid(self, whitelist=None):
    """Generates a 'Unique Identifier' based on all internal fields.

    Caller should use the uid string to check `RunConfig` instance integrity
    in one session use, but should not rely on the implementation details, which
    is subject to change.

    Args:
      whitelist: A list of the string names of the properties uid should not
        include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
        includes most properties user allowes to change.

    Returns:
      A uid string.
    """
    if whitelist is None:
      whitelist = _DEFAULT_UID_WHITE_LIST

    state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
    # Pop out the keys in whitelist.
    for k in whitelist:
      state.pop('_' + k, None)

    ordered_state = collections.OrderedDict(
        sorted(state.items(), key=lambda t: t[0]))
    # For class instance without __repr__, some special cares are required.
    # Otherwise, the object address will be used.
    if '_cluster_spec' in ordered_state:
      ordered_state['_cluster_spec'] = collections.OrderedDict(
          sorted(ordered_state['_cluster_spec'].as_dict().items(),
                 key=lambda t: t[0]))
    return ', '.join(
        '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state))

  @property
  def model_dir(self):
    return self._model_dir

  @property
  def tf_config(self):
    return self._tf_config

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
  def save_checkpoints_steps(self):
    return self._save_checkpoints_steps

  @property
  def session_config(self):
    return self._session_config

  @property
  def keep_checkpoint_max(self):
    return self._keep_checkpoint_max

  @property
  def keep_checkpoint_every_n_hours(self):
    return self._keep_checkpoint_every_n_hours

  @property
  def log_step_count_steps(self):
    return self._log_step_count_steps


def _count_ps(cluster_spec):
  """Counts the number of parameter servers in cluster_spec."""
  return len(cluster_spec.as_dict().get('ps', [])) if cluster_spec else 0


def _count_worker(cluster_spec):
  """Counts the number of workers in cluster_spec.

  Workers with TaskType.WORKER and TaskType.MASTER are included in the return
  value.

  Args:
    cluster_spec: a ClusterSpec instance that describes current deployment.

  Returns:
    The total number of eligible workers.

    If 'cluster_spec' was None, then 0 is returned.
  """
  return (len(cluster_spec.as_dict().get('worker', [])) +
          len(cluster_spec.as_dict().get('master', []))) if cluster_spec else 0


def _get_master(cluster_spec, task_type, task_id):
  """Returns the appropriate string for the TensorFlow master."""
  if not cluster_spec:
    return ''

  # If there is only one node in the cluster, do things locally.
  jobs = cluster_spec.jobs
  if len(jobs) == 1 and len(cluster_spec.job_tasks(jobs[0])) == 1:
    return ''

  # Lookup the master in cluster_spec using task_type and task_id,
  # if possible.
  if task_type:
    if task_type not in jobs:
      raise ValueError(
          '%s is not a valid task_type in the cluster_spec:\n'
          '%s\n\n'
          'Note that these values may be coming from the TF_CONFIG environment '
          'variable.' % (task_type, cluster_spec))
    addresses = cluster_spec.job_tasks(task_type)
    if task_id >= len(addresses) or task_id < 0:
      raise ValueError(
          '%d is not a valid task_id for task_type %s in the '
          'cluster_spec:\n'
          '%s\n\n'
          'Note that these value may be coming from the TF_CONFIG environment '
          'variable.' % (task_id, task_type, cluster_spec))
    return 'grpc://' + addresses[task_id]

  # For backwards compatibility, we return empty string if task_type was
  # not set (task_type did not previously exist).
  return ''


def _get_model_dir(model_dir):
  """Returns `model_dir` based user provided `model_dir` or `TF_CONFIG`."""

  model_dir_in_tf_config = json.loads(
      os.environ.get('TF_CONFIG') or '{}').get('model_dir', None)
  if model_dir_in_tf_config is not None:
    if model_dir is not None and model_dir_in_tf_config != model_dir:
      raise ValueError(
          '`model_dir` provided in RunConfig construct, if set, '
          'must have the same value as the model_dir in TF_CONFIG. '
          'model_dir: {}\nTF_CONFIG["model_dir"]: {}.\n'.format(
              model_dir, model_dir_in_tf_config))

    logging.info('Using model_dir in TF_CONFIG: %s', model_dir_in_tf_config)

  return model_dir or model_dir_in_tf_config
