# Copyright 2018-2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Cluster Resolvers for Slurm workload manager."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import subprocess

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export


def expand_hostlist(hostlist):
  """Create a list of hosts out of a SLURM hostlist.

  The order of nodes is preserved and no deduplication is done
  Input: 'n[1-2],m5,o[3-4,6,7-9]')
  Output: ['n1', 'n2', 'm5', 'o3', 'o4', 'o6', 'o7', 'o8', 'o9']
  """

  def split_hostlist(hostlist):
    """Split hostlist at commas outside of range expressions ('[3-5]')."""
    in_brackets = False
    cur_host = ''
    for c in hostlist:
      if in_brackets:
        assert c != '['
        if c == ']':
          in_brackets = False
      elif c == '[':
        in_brackets = True
      elif c == ',':
        assert cur_host != ''
        yield cur_host
        cur_host = ''
        continue
      cur_host += c
    if cur_host:
      yield cur_host

  def expand_range_expression(range_exp):
    """Expand a range expression like '3-5' to values 3,4,5."""
    for part in range_exp.split(','):
      sub_range = part.split('-')
      if len(sub_range) == 1:
        sub_range = sub_range * 2
      else:
        assert len(sub_range) == 2
      for i in range(int(sub_range[0]), int(sub_range[1]) + 1):
        yield i

  hosts = []
  try:
    for part in split_hostlist(hostlist):
      # Match prefix (anything but a range expression) and range expression
      # Both are optional
      m = re.match(r'([^,[\]]*)(\[([^\]]+)\])?$', part)
      if m is None:
        raise ValueError('Invalid part: %s' % part)
      prefix = m.group(1) or ''
      if m.group(3) is None:
        hosts.append(prefix)
      else:
        hosts.extend(
            prefix + str(i) for i in expand_range_expression(m.group(3)))
  except Exception as e:
    raise ValueError('Invalid hostlist format "%s": %s' % (hostlist, e))
  return hosts


def expand_tasks_per_node(tasks_per_node):
  """Expands the tasks per node expression from SLURM.

  The order is preserved so it can be matched to the hostlist
  Input: '3(x2),2,1'
  Output: [3, 3, 2, 1]
  """
  result = []
  try:
    for part in tasks_per_node.split(','):
      m = re.match(r'(\d+)(\(x(\d+)\))?$', part)
      assert m is not None
      num_tasks = int(m.group(1))
      num_repetitions = int(m.group(3) or 1)
      result.extend([num_tasks] * num_repetitions)
  except Exception as e:
    raise ValueError('Invalid tasks-per-node list format "%s": %s' %
                     (tasks_per_node, e))
  return result


def _get_slurm_var(name):
  """Gets the SLURM variable from the environment.

  Args:
    name: Name of the step variable

  Returns:
    SLURM_<name> from os.environ
  Raises:
    RuntimeError if variable is not found
  """
  name = 'SLURM_' + name
  try:
    return os.environ[name]
  except KeyError:
    raise RuntimeError('%s not found in environment. '
                       'Not running inside a SLURM step?' % name)


def _get_num_slurm_tasks():
  """Returns the number of SLURM tasks of the current job step.

  Returns:
    The number of tasks as an int
  """
  return int(_get_slurm_var('STEP_NUM_TASKS'))


def _get_num_nvidia_gpus():
  """Gets the number of NVIDIA GPUs by using CUDA_VISIBLE_DEVICES and nvidia-smi.

  Returns:
    Number of GPUs available on the node
  Raises:
    RuntimeError if executing nvidia-smi failed
  """
  try:
    return len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
  except KeyError:
    pass  # Ignore and fallback to using nvidia-smi
  try:
    output = subprocess.check_output(['nvidia-smi', '--list-gpus'],
                                     encoding='utf-8')
    return sum(l.startswith('GPU ') for l in output.strip().split('\n'))
  except subprocess.CalledProcessError as e:
    raise RuntimeError('Could not get number of GPUs from nvidia-smi. '
                       'Maybe it is missing?\nOutput: %s' % e.output)


def get_num_gpus():
  """Returns the number of GPUs visible on the current node.

  Currently only implemented for NVIDIA GPUs.
  """
  return _get_num_nvidia_gpus()


@tf_export('distribute.cluster_resolver.SlurmClusterResolver')
class SlurmClusterResolver(ClusterResolver):
  """ClusterResolver for system with Slurm workload manager.

  This is an implementation of ClusterResolver for Slurm clusters. This allows
  the specification of jobs and task counts, number of tasks per node, number
  of GPUs on each node and number of GPUs for each task. It retrieves system
  attributes by Slurm environment variables, resolves allocated computing node
  names, constructs a cluster and returns a ClusterResolver object which can be
  used for distributed TensorFlow.
  """

  def __init__(self,
               jobs=None,
               port_base=8888,
               gpus_per_node=None,
               gpus_per_task=None,
               tasks_per_node=None,
               auto_set_gpu=True,
               rpc_layer='grpc'):
    """Creates a new SlurmClusterResolver object.

    For any parameter not set it will query the environment for the value.
    It uses those parameters to check which nodes have processes reside on and
    resolves their hostnames.
    With the number tasks per node it offsets the port number for each process.
    With the number of GPUs per node and per task it allocates GPUs to tasks by
    setting environment variables.
    Using the resolver works best (and is easier) with homogeneous tasks but
    heterogeneous tasks (number of tasks varying per node) are also possible as
    long as the number of GPUs per task stays constant.

    Used environment variables:
      - SLURM_PROCID
      - (opt) SLURM_STEP_NUM_TASKS
      - (opt) SLURM_STEP_NODELIST
      - (opt) SLURM_STEP_TASKS_PER_NODE

    Args:
      jobs: Dictionary with job names as key and number of tasks in the job as
        value. Defaults to as many 'worker's as there are (Slurm) tasks.
      port_base: The first port number to start with for processes on a node.
      gpus_per_node: Number of GPUs available on each node. Defaults to the
        number of GPUs reported by nvidia-smi
      gpus_per_task: Number of GPUs to be used for each task. Default is to
        evenly distribute the gpus_per_node to tasks_per_node.
      tasks_per_node: Number of tasks running on each node. Can be an integer if
        the number of tasks per node is constant or a dictionary mapping
        hostnames to number of tasks on that node. If not set the Slurm
        environment is queried for the correct mapping.
      auto_set_gpu: Set the visible CUDA devices automatically while resolving
        the cluster by setting CUDA_VISIBLE_DEVICES environment variable.
        Defaults to True.
      rpc_layer: The protocol TensorFlow used to communicate between nodes.
        Defaults to 'grpc'.

    Returns:
      A ClusterResolver object which can be used with distributed TensorFlow.

    Raises:
      RuntimeError: If requested more GPUs per node then available or
        requested more tasks then assigned tasks or
        resolving missing values from the environment failed.
    """

    self._rank = self._resolve_own_rank()

    if jobs is None:
      jobs = {'worker': self._resolve_num_tasks()}

    self._jobs = jobs
    self._port_base = port_base

    if tasks_per_node is None:
      self._task_configuration = self._resolve_task_configuration()
    elif isinstance(tasks_per_node, dict):
      # User can pass in an explicit configuration as a dict
      self._task_configuration = tasks_per_node
    else:
      # User can pass a fixed number of tasks per node
      hostlist = self._resolve_hostlist()
      self._task_configuration = {
          host: int(tasks_per_node) for host in hostlist
      }

    max_tasks_per_node = max(self._task_configuration.values())
    num_tasks = sum(self._task_configuration.values())

    if gpus_per_node is None:
      gpus_per_node = get_num_gpus()
    if gpus_per_task is None:
      gpus_per_task = gpus_per_node // max_tasks_per_node
    self._gpus_per_node = gpus_per_node
    self._gpus_per_task = gpus_per_task

    self._auto_set_gpu = auto_set_gpu
    self.task_type = None
    self.task_id = None
    self.rpc_layer = rpc_layer

    self._gpu_allocation = []
    self._cluster_allocation = {}

    if max_tasks_per_node * self._gpus_per_task > self._gpus_per_node:
      raise RuntimeError('Requested more GPUs per node then available.')

    if sum(self._jobs.values()) != num_tasks:
      raise RuntimeError('Requested {} tasks but only {} were assigned.'.format(
          sum(self._jobs.values()), num_tasks))

  def _resolve_own_rank(self):
    """Returns the rank of the current task in range [0, num_tasks)."""
    return int(_get_slurm_var('PROCID'))

  def _resolve_num_tasks(self):
    """Returns the number of tasks for the current job step."""
    return _get_num_slurm_tasks()

  def _resolve_hostlist(self):
    """Returns a list of hostnames for nodes running the current job step."""
    return expand_hostlist(_get_slurm_var('STEP_NODELIST'))

  def _resolve_task_configuration(self):
    """Creates a mapping of hostnames to the number of tasks allocated on it.

    Reads the SLURM environment to determine the nodes involved in the current
    job step and number of tasks running on each node.

    Returns a dictionary mapping each hostname to the number of tasks.
    """
    hostlist = self._resolve_hostlist()
    tasks_per_node = expand_tasks_per_node(
        _get_slurm_var('STEP_TASKS_PER_NODE'))
    return {
        host: num_tasks for (host, num_tasks) in zip(hostlist, tasks_per_node)
    }

  def cluster_spec(self):
    """Returns a ClusterSpec object based on the latest instance group info.

    This returns a ClusterSpec object for use based on information from the
    specified initialization parameters and Slurm environment variables. The
    cluster specification is resolved each time this function is called. The
    resolver extract hostnames of nodes by scontrol and pack tasks in that
    order until a node a has number of tasks that is equal to specification.
    GPUs on nodes are allocated to tasks by specification through setting
    CUDA_VISIBLE_DEVICES environment variable.

    Returns:
      A ClusterSpec containing host information retrieved from Slurm's
        environment variables.
    """

    task_list = []
    self._gpu_allocation = []
    self._cluster_allocation = {}

    # Sort to make sure the order is the same for each run
    for host, num_tasks in sorted(self._task_configuration.items()):
      for port_offset, gpu_offset in zip(
          range(num_tasks), range(0, self._gpus_per_node, self._gpus_per_task)):

        host_addr = '%s:%d' % (host, self._port_base + port_offset)
        task_list.append(host_addr)
        gpu_id_list = []

        for gpu_id in range(gpu_offset, gpu_offset + self._gpus_per_task):
          gpu_id_list.append(str(gpu_id))

        self._gpu_allocation.append(','.join(gpu_id_list))

    cluster_rank_offset_start = 0
    cluster_rank_offset_end = 0

    # Sort to make sure the order is the same for each run
    for task_type, num_tasks in sorted(self._jobs.items()):
      cluster_rank_offset_end = cluster_rank_offset_start + num_tasks

      self._cluster_allocation[task_type] = (
          task_list[cluster_rank_offset_start:cluster_rank_offset_end])

      if cluster_rank_offset_start <= self._rank < cluster_rank_offset_end:
        self.task_type = task_type
        self.task_id = self._rank - cluster_rank_offset_start

      cluster_rank_offset_start = cluster_rank_offset_end

    if self._auto_set_gpu:
      os.environ['CUDA_VISIBLE_DEVICES'] = self._gpu_allocation[self._rank]

    return ClusterSpec(self._cluster_allocation)

  def get_task_info(self):
    """Returns job name and task_id for the process which calls this.

    This returns the job name and task index for the process which calls this
    function according to its rank and cluster specification. The job name and
    task index are set after a cluster is constructed by cluster_spec otherwise
    defaults to None.

    Returns:
      A string specifying job name the process belongs to and an integer
        specifying the task index the process belongs to in that job.
    """
    return self.task_type, self.task_id

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Returns the master string for connecting to a TensorFlow master.

    Args:
      task_type: (Optional) Overrides the default auto-selected task type.
      task_id: (Optional) Overrides the default auto-selected task index.
      rpc_layer: (Optional) Overrides the default RPC protocol TensorFlow uses
        to communicate across nodes.

    Returns:
      A connection string for connecting to a TensorFlow master.
    """
    task_type = task_type if task_type is not None else self.task_type
    task_id = task_id if task_id is not None else self.task_id

    if task_type is not None and task_id is not None:
      return format_master_url(
          self.cluster_spec().task_address(task_type, task_id),
          rpc_layer or self.rpc_layer)

    return ''

  def num_accelerators(self,
                       task_type=None,
                       task_id=None,
                       config_proto=None):
    # Unused, since this is set in __init__ manually.
    del task_type, task_id, config_proto
    return {'GPU': self._gpus_per_task}
