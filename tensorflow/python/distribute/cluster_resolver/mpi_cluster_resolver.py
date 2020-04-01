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
"""Implementation of a Cluster Resolver for MPI jobs."""

from socket import gethostname
from slurm_cluster_resolver import SlurmClusterResolver

from tensorflow.python.util.tf_export import tf_export


@tf_export('distribute.cluster_resolver.MPIClusterResolver')
class MPIClusterResolver(SlurmClusterResolver):
  """ClusterResolver for programs run via MPI (mpirun, srun, ...)

  This is an implementation of ClusterResolver for MPI programs which can
  be used for distributed TensorFlow.
  When no explicit values are set it will retrieve all information from MPI.
  For rank and number of Tasks the values are gotten from the MPI_COMM_WORLD
  communicator. It also automatically resolves hostnames which requires some
  communication on startup.
  """

  def __init__(self,
               jobs=None,
               port_base=8888,
               gpus_per_node=None,
               gpus_per_task=None,
               auto_set_gpu=True,
               rpc_layer='grpc'):
    """Creates a new MPIClusterResolver object.

    For any parameter not set it will query MPI_COMM_WORLD for the value.
    With the number of GPUs per node and per task it allocates GPUs to tasks by
    setting environment variables.
    Using the resolver works best (and is easier) with homogeneous tasks but
    heterogeneous tasks (number of tasks varying per node) are also possible as
    long as the number of GPUs per task stays constant.

    Args:
      jobs: Dictionary with job names as key and number of tasks in the job as
        value. Defaults to as many 'worker's as there are MPI tasks.
      port_base: The first port number to start with for processes on a node.
      gpus_per_node: Number of GPUs available on each node. Defaults to the
        number of GPUs reported by nvidia-smi
      gpus_per_task: Number of GPUs to be used for each task. Default is to
        evenly distribute the gpus_per_node to tasks_per_node.
      auto_set_gpu: Set the visible CUDA devices automatically while resolving
        the cluster by setting CUDA_VISIBLE_DEVICES environment variable.
        Defaults to True.
      rpc_layer: The protocol TensorFlow used to communicate between nodes.
        Defaults to 'grpc'.

    Returns:
      A MPIResolver object which can be used with distributed TensorFlow.

    Raises:
      RuntimeError: If requested more GPUs per node then available or
        requested more tasks then assigned tasks.
    """
    from mpi4py import MPI # pylint: disable=import-outside-toplevel
    self._comm = MPI.COMM_WORLD # pylint: disable=c-extension-no-member
    super().__init__(jobs, port_base, gpus_per_node, gpus_per_task,
                     tasks_per_node=None, auto_set_gpu=auto_set_gpu,
                     rpc_layer=rpc_layer)

  def _resolve_own_rank(self):
    """Return the rank of the current task in range [0, num_tasks)"""
    return self._comm.Get_rank()

  def _resolve_num_tasks(self):
    """Return the number of tasks for the current job step"""
    return self._comm.Get_size()

  def _resolve_hostlist(self):
    """Return a list of hostnames for nodes running the current job step"""
    own_hostname = gethostname()
    return list(set(self._comm.allgather(own_hostname)))

  def _resolve_task_configuration(self):
    """Create a mapping of hostnames to the number of tasks allocated on it
    Returns a dictionary mapping each hostname to the number of tasks.
    """
    own_hostname = gethostname()
    hostnames = self._comm.allgather(own_hostname)
    return {host: hostnames.count(host) for host in set(hostnames)}
