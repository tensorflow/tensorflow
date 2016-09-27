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

from tensorflow.python import ConfigProto
from tensorflow.python import GPUOptions
from tensorflow.python.training.server_lib import ClusterSpec


class RunConfig(object):
  """This class specifies the specific configurations for the run.

  If you're a Google-internal user using command line flags with learn_runner.py
  (for instance, to do distributed training or to use parameter servers), you
  probably want to use learn_runner.EstimatorConfig instead.
  """

  # TODO(wicke): Move options out once functionality is covered by monitors
  def __init__(self,
               master=None,
               task=None,
               num_ps_replicas=None,
               num_cores=0,
               log_device_placement=False,
               gpu_memory_fraction=1,
               cluster_spec=None,
               tf_random_seed=None,
               save_summary_steps=100,
               save_checkpoints_secs=600,
               keep_checkpoint_max=5,
               keep_checkpoint_every_n_hours=10000,
               job_name=None,
               is_chief=None,
               evaluation_master=''):
    """Constructor.

    If set to None, `master`, `task`, `num_ps_replicas`, `cluster_spec`,
    `job_name`, and `is_chief` are set based on the TF_CONFIG environment
    variable, if the pertinent information is present; otherwise, the defaults
    listed in the Args section apply.

    The TF_CONFIG environment variable is a JSON object with two relevant
    attributes: `task` and `cluster_spec`. `cluster_spec` is a JSON serialized
    version of the Python dict described in server_lib.py. `task` has two
    attributes: `type` and `index`, where `type` can be any of the task types
    in the cluster_spec. When TF_CONFIG contains said information, the
    following properties are set on this class:

      * `job_name` is set to [`task`][`type`]
      * `task` is set to [`task`][`index`]
      * `cluster_spec` is parsed from [`cluster`]
      * 'master' is determined by looking up `job_name` and `task` in the
        cluster_spec.
      * `num_ps_replicas` is set by counting the number of nodes listed
        in the `ps` job of `cluster_spec`.
      * `is_chief`: true when `job_name` == "master" and `task` == 0.

    Example:
    ```
      cluster = {'ps': ['host1:2222', 'host2:2222'],
                 'worker': ['host3:2222', 'host4:2222', 'host5:2222']}
      os.environ['TF_CONFIG'] = json.dumps({
          {'cluster': cluster,
           'task': {'type': 'worker', 'index': 1}}})
      config = RunConfig()
      assert config.master == 'host4:2222'
      assert config.task == 1
      assert config.num_ps_replicas == 2
      assert config.cluster_spec == server_lib.ClusterSpec(cluster)
      assert config.job_name == 'worker'
      assert not config.is_chief
    ```

    Args:
      master: TensorFlow master. Defaults to empty string for local.
      task: Task id of the replica running the training (default: 0).
      num_ps_replicas: Number of parameter server tasks to use (default: 0).
      num_cores: Number of cores to be used. If 0, the system picks an
        appropriate number (default: 0).
      log_device_placement: Log the op placement to devices (default: False).
      gpu_memory_fraction: Fraction of GPU memory used by the process on
        each GPU uniformly on the same machine.
      cluster_spec: a `tf.train.ClusterSpec` object that describes the cluster
        in the case of distributed computation. If missing, reasonable
        assumptions are made for the addresses of jobs.
      tf_random_seed: Random seed for TensorFlow initializers.
        Setting this value allows consistency between reruns.
      save_summary_steps: Save summaries every this many steps.
      save_checkpoints_secs: Save checkpoints every this many seconds.
      keep_checkpoint_max: The maximum number of recent checkpoint files to
        keep. As new files are created, older files are deleted. If None or 0,
        all checkpoint files are kept. Defaults to 5 (that is, the 5 most recent
        checkpoint files are kept.)
      keep_checkpoint_every_n_hours: Number of hours between each checkpoint
        to be saved. The default value of 10,000 hours effectively disables
        the feature.
      job_name: the type of task, e.g., 'ps', 'worker', etc. The `job_name`
        must exist in the `cluster_spec.jobs`.
      is_chief: whether or not this task (as identified by the other parameters)
        should be the chief task.
      evaluation_master: the master on which to perform evaluation.

    Raises:
      ValueError: if num_ps_replicas and cluster_spec are set (cluster_spec
        may fome from the TF_CONFIG environment variable).
    """
    # If not explicitly specified in the constructor and the TF_CONFIG
    # environment variable is present, load cluster_spec from TF_CONFIG.
    config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    if not cluster_spec and 'cluster' in config:
      cluster_spec = ClusterSpec(config['cluster'])
    self.cluster_spec = cluster_spec

    # Set job_name and task. If explicitly specified, use those values,
    # otherwise, if the TF_CONFIG environment variable is present, use that.
    # Otherwise, use the respective default (None / 0).
    task_env = config.get('task', {})
    self._job_name = job_name or task_env.get('type') or None
    self.task = task if task is not None else task_env.get('index') or 0

    self.master = (master or _get_master(self.cluster_spec, self.job_name,
                                         self.task) or '')

    if num_ps_replicas is not None and self.cluster_spec:
      raise ValueError('Cannot specify both num_ps_replicas and cluster_spec. '
                       'Note: cluster_spec may have been set in the TF_CONFIG '
                       'environment variable.')
    self.num_ps_replicas = num_ps_replicas or _count_ps(self.cluster_spec) or 0

    # Set is_chief.
    self._is_chief = is_chief
    if self._is_chief is None:
      if not self._job_name:
        self._is_chief = (self.task == 0)
      elif config:
        # When the TF_CONFIG environment variable is set, we can set the
        # default of is_chief to 0 when job_name is "master" and task is 0.
        self._is_chief = (self._job_name == 'master' and self.task == 0)
      else:
        # Legacy behavior is that is_chief is None if task == 0.
        self._is_chief = (self._job_name == 'worker' and self.task == 0)

    # Enforce that is_chief is only applicable to workers or masters
    # (Cloud ML) with task == 0.
    if self._is_chief:
      if self.task != 0:
        raise ValueError(
            'Task is %d, but only task 0 may be chief. Please check is_chief '
            'and task, which may have been set in TF_CONFIG environment '
            'variable.' % (self.task,))
      if self._job_name not in (None, 'master', 'worker'):
        raise ValueError(
            'job_name is \'%s\', but only masters or workers may be chiefs. '
            'Please check is_chief and job_name, which may have been set in '
            'TF_CONFIG environment variable.' % (self._job_name,))
    elif (self._is_chief is False and self._job_name == 'master' and
          self.task == 0):
      raise ValueError(
          'Master task 0 must be chief. Please check is_chief, job_name, and '
          'task, which may have been set in TF_CONFIG environment variable.')

    self.evaluation_master = evaluation_master or ''

    gpu_options = GPUOptions(
        per_process_gpu_memory_fraction=gpu_memory_fraction)
    self.tf_config = ConfigProto(
        log_device_placement=log_device_placement,
        inter_op_parallelism_threads=num_cores,
        intra_op_parallelism_threads=num_cores,
        gpu_options=gpu_options)

    self.tf_random_seed = tf_random_seed
    self.save_summary_steps = save_summary_steps
    self.save_checkpoints_secs = save_checkpoints_secs
    self.keep_checkpoint_max = keep_checkpoint_max
    self.keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours

  @property
  def is_chief(self):
    return self._is_chief

  @property
  def job_name(self):
    return self._job_name


def _count_ps(cluster_spec):
  """Counts the number of parameter servers in cluster_spec."""
  return len(cluster_spec.as_dict().get('ps', [])) if cluster_spec else 0


def _get_master(cluster_spec, job_name, task_index):
  """Returns the appropriate string for the TensorFlow master."""
  if not cluster_spec:
    return ''

  # If there is only one node in the cluster, do things locally.
  jobs = cluster_spec.jobs
  if len(jobs) == 1 and len(cluster_spec.job_tasks(jobs[0])) == 1:
    return ''

  # Lookup the master in cluster_spec using job_name and task_index,
  # if possible.
  if job_name:
    if job_name not in jobs:
      raise ValueError(
          '%s is not a valid task in the cluster_spec:\n'
          '%s\n\n'
          'Note that these values may be coming from the TF_CONFIG environment '
          'variable.' % (job_name, cluster_spec))
    addresses = cluster_spec.job_tasks(job_name)
    if task_index >= len(addresses) or task_index < 0:
      raise ValueError(
          '%d is not a valid task index for task type %s in the '
          'cluster_spec:\n'
          '%s\n\n'
          'Note that these value may be coming from the TF_CONFIG environment '
          'variable.' % (task_index, job_name, cluster_spec))
    return 'grpc://' + addresses[task_index]

  # For backwards compatibility, we return empty string if job_name was
  # not set (job_name did not previously exist).
  return ''
