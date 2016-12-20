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
"""Run Config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
from tensorflow.contrib.framework import deprecated
from tensorflow.python import ConfigProto
from tensorflow.python import GPUOptions
from tensorflow.python.training.server_lib import ClusterSpec


class Environment(object):
  # For running general distributed training.
  CLOUD = 'cloud'
  # For running Google-internal distributed training.
  GOOGLE = 'google'
  # For running on local desktop.
  LOCAL = 'local'


class TaskType(object):
  MASTER = 'master'
  PS = 'ps'
  WORKER = 'worker'


class ClusterConfig(object):
  """This class specifies the configurations for a distributed run.

  If you're using `tf.learn` `Estimators`, you should probably use the subclass
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
    * `is_chief` is deteremined based on `task_type`, `type_id`, and
      `environment`.

    Example:
    ```
      cluster = {'ps': ['host1:2222', 'host2:2222'],
                 'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
      os.environ['TF_CONFIG'] = json.dumps({
          {'cluster': cluster,
           'task_id': {'type': 'worker', 'index': 1}}})
      config = ClusterConfig()
      assert config.master == 'host4:2222'
      assert config.task_id == 1
      assert config.num_ps_replicas == 2
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

    self._cluster_spec = ClusterSpec(config.get('cluster', {}))
    self._master = (master if master is not None else
                    _get_master(self._cluster_spec, self._task_type,
                                self._task_id) or '')
    self._num_ps_replicas = _count_ps(self._cluster_spec) or 0

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


class RunConfig(ClusterConfig):
  """This class specifies the configurations for an `Estimator` run.

  If you're a Google-internal user using command line flags with
  `learn_runner.py` (for instance, to do distributed training or to use
  parameter servers), you probably want to use `learn_runner.EstimatorConfig`
  instead.
  """

  def __init__(self,
               master=None,
               num_cores=0,
               log_device_placement=False,
               gpu_memory_fraction=1,
               tf_random_seed=None,
               save_summary_steps=100,
               save_checkpoints_secs=600,
               save_checkpoints_steps=None,
               keep_checkpoint_max=5,
               keep_checkpoint_every_n_hours=10000,
               evaluation_master=''):
    """Constructor.

    Note that the superclass `ClusterConfig` may set properties like
    `cluster_spec`, `is_chief`, `master` (if `None` in the args),
    `num_ps_replicas`, `task_id`, and `task_type` based on the `TF_CONFIG`
    environment variable. See `ClusterConfig` for more details.

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
      evaluation_master: the master on which to perform evaluation.
    """
    super(RunConfig, self).__init__(
        master=master, evaluation_master=evaluation_master)

    gpu_options = GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    self._tf_config = ConfigProto(
        log_device_placement=log_device_placement,
        inter_op_parallelism_threads=num_cores,
        intra_op_parallelism_threads=num_cores,
        gpu_options=gpu_options)

    self._tf_random_seed = tf_random_seed
    self._save_summary_steps = save_summary_steps
    self._save_checkpoints_secs = save_checkpoints_secs
    self._save_checkpoints_steps = save_checkpoints_steps

    # TODO(weiho): Remove these after ModelFn refactoring, when users can
    # create Scaffold and Saver in their model_fn to set these.
    self._keep_checkpoint_max = keep_checkpoint_max
    self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

  @property
  def tf_config(self):
    return self._tf_config

  @tf_config.setter
  @deprecated(
      '2017-01-08',
      'RunConfig will be made immutable, please pass all args to constructor.')
  def tf_config(self, value):
    self._tf_config = value

  @property
  def tf_random_seed(self):
    return self._tf_random_seed

  @tf_random_seed.setter
  @deprecated(
      '2017-01-08',
      'RunConfig will be made immutable, please pass all args to constructor.')
  def tf_random_seed(self, value):
    self._tf_random_seed = value

  @property
  def save_summary_steps(self):
    return self._save_summary_steps

  @save_summary_steps.setter
  @deprecated(
      '2017-01-08',
      'RunConfig will be made immutable, please pass all args to constructor.')
  def save_summary_steps(self, value):
    self._save_summary_steps = value

  @property
  def save_checkpoints_secs(self):
    return self._save_checkpoints_secs

  @save_checkpoints_secs.setter
  @deprecated(
      '2017-01-08',
      'RunConfig will be made immutable, please pass all args to constructor.')
  def save_checkpoints_secs(self, value):
    self._save_checkpoints_secs = value

  @property
  def save_checkpoints_steps(self):
    return self._save_checkpoints_steps

  @save_checkpoints_steps.setter
  @deprecated(
      '2017-01-08',
      'RunConfig will be made immutable, please pass all args to constructor.')
  def save_checkpoints_steps(self, value):
    self._save_checkpoints_steps = value

  @property
  def keep_checkpoint_max(self):
    return self._keep_checkpoint_max

  @keep_checkpoint_max.setter
  @deprecated(
      '2017-01-08',
      'RunConfig will be made immutable, please pass all args to constructor.')
  def keep_checkpoint_max(self, value):
    self._keep_checkpoint_max = value

  @property
  def keep_checkpoint_every_n_hours(self):
    return self._keep_checkpoint_every_n_hours

  @keep_checkpoint_every_n_hours.setter
  @deprecated(
      '2017-01-08',
      'RunConfig will be made immutable, please pass all args to constructor.')
  def keep_checkpoint_every_n_hours(self, value):
    self._keep_checkpoint_every_n_hours = value


def _count_ps(cluster_spec):
  """Counts the number of parameter servers in cluster_spec."""
  return len(cluster_spec.as_dict().get('ps', [])) if cluster_spec else 0


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
