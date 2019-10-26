# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

import collections
import os
import subprocess

from tensorflow.python.distribute.cluster_resolver.cluster_resolver import ClusterResolver
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.util.tf_export import tf_export


@tf_export('distribute.cluster_resolver.SlurmClusterResolver')
class SlurmClusterResolver(ClusterResolver):
  """ClusterResolver for system with Slurm workload manager.

  This is an implementation of cluster resolvers for Slurm clusters. This allows
  the specification of jobs and task counts, number of tasks per node, number of
  GPUs on each node and number of GPUs for each task. It retrieves system
  attributes by Slurm environment variables, resolves allocated computing node
  names, constructs a cluster and returns a ClusterResolver object which can be
  use for distributed TensorFlow.
  """

  def _resolve_hostnames(self):
    """Resolve host names of nodes allocated in current jobs.

    Returns:
      A list of node names as strings.
    """
    hostlist = (subprocess.check_output(['scontrol', 'show', 'hostname']).
                decode('utf-8').strip().split('\n'))
    return hostlist

  def __init__(self,
               jobs,
               port_base=8888,
               gpus_per_node=1,
               gpus_per_task=1,
               tasks_per_node=None,
               auto_set_gpu=True,
               rpc_layer='grpc'):
    """Creates a new SlurmClusterResolver object.

    This takes in parameters and creates a SlurmClusterResolver object. It uses
    those parameters to check which nodes will processes reside on and resolves
    their hostnames. With the number of the GPUs on each node and number of GPUs
    for each task it offsets the port number for each process and allocates
    GPUs to tasks by setting environment variables. The resolver currently
    supports homogeneous tasks and default Slurm process allocation.

    Args:
      jobs: Dictionary with job names as key and number of tasks in the job as
        value.
      port_base: The first port number to start with for processes on a node.
      gpus_per_node: Number of GPUs available on each node.
      gpus_per_task: Number of GPUs to be used for each task.
      tasks_per_node: Number of tasks to run on each node, if not set defaults
        to Slurm's output environment variable SLURM_NTASKS_PER_NODE.
      auto_set_gpu: Set the visible CUDA devices automatically while resolving
        the cluster by setting CUDA_VISIBLE_DEVICES environment variable.
        Defaults to True.
      rpc_layer: (Optional) The protocol TensorFlow uses to communicate between
        nodes. Defaults to 'grpc'.

    Returns:
      A ClusterResolver object which can be used with distributed TensorFlow.

    Raises:
      RuntimeError: If requested more GPUs per node then available or requested
      more tasks then assigned tasks.
    """

    # check if launched by mpirun
    if 'OMPI_COMM_WORLD_RANK' in os.environ:
      self._rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
      num_tasks = int(os.environ['OMPI_COMM_WORLD_SIZE'])
    else:
      self._rank = int(os.environ['SLURM_PROCID'])
      num_tasks = int(os.environ['SLURM_NTASKS'])

    self._jobs = collections.OrderedDict(sorted(jobs.items()))
    self._port_base = port_base

    # user specification overrides SLURM specification
    if tasks_per_node is not None:
      self._tasks_per_node = tasks_per_node
    elif tasks_per_node is None and 'SLURM_NTASKS_PER_NODE' in os.environ:
      self._tasks_per_node = int(os.environ['SLURM_NTASKS_PER_NODE'])
    else:
      raise RuntimeError('Neither `tasks_per_node` or '
                         'SLURM_NTASKS_PER_NODE is set.')

    self._gpus_per_node = gpus_per_node
    self._gpus_per_task = gpus_per_task

    self._auto_set_gpu = auto_set_gpu
    self.task_type = None
    self.task_id = None
    self.rpc_layer = rpc_layer

    self._gpu_allocation = []
    self._cluster_allocation = {}

    if self._tasks_per_node * self._gpus_per_task > self._gpus_per_node:
      raise RuntimeError('Requested more GPUs per node then available.')

    if sum(self._jobs.values()) != num_tasks:
      raise RuntimeError('Requested more tasks then assigned tasks.')

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
    hostlist = self._resolve_hostnames()

    task_list = []
    self._gpu_allocation = []
    self._cluster_allocation = {}

    for host in hostlist:
      for port_offset, gpu_offset in zip(
          range(self._tasks_per_node),
          range(0, self._gpus_per_node, self._gpus_per_task)):

        host_addr = '%s:%d' % (host, self._port_base + port_offset)
        task_list.append(host_addr)
        gpu_id_list = []

        for gpu_id in range(gpu_offset, gpu_offset + self._gpus_per_task):
          gpu_id_list.append(str(gpu_id))

        self._gpu_allocation.append(','.join(gpu_id_list))

    cluster_rank_offset_start = 0
    cluster_rank_offset_end = 0

    for task_type, num_tasks in self._jobs.items():
      cluster_rank_offset_end = cluster_rank_offset_start + num_tasks

      self._cluster_allocation[task_type] = (
          task_list[cluster_rank_offset_start:cluster_rank_offset_end])

      if cluster_rank_offset_start <= self._rank < cluster_rank_offset_end:
        self.task_type = task_type
        self.task_id = self._rank - cluster_rank_offset_start

      cluster_rank_offset_start = cluster_rank_offset_end

    if self._auto_set_gpu is True:
      os.environ['CUDA_VISIBLE_DEVICES'] = self._gpu_allocation[self._rank]

    return ClusterSpec(self._cluster_allocation)

  def get_task_info(self):
    """Returns job name and task_id for the process which calls this.

    This returns the job name and task index for the process which calls this
    function according to its rank and cluster specification. The job name and
    task index are set after a cluster is constructed by cluster_spec otherwise
    defaults to None.

    Returns:
      A string specifying job name the process belongs to and an integner
        specifying the task index the process belongs to in that job.
    """
    return self.task_type, self.task_id

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Returns the master string for connecting to a TensorFlow master.

    Args:
      task_type: (Optional) Overrides the default auto-selected task type.
      task_id: (Optional) Overrides the default auto-slected task index.
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
    return {'GPU': self._gpus_per_node}
