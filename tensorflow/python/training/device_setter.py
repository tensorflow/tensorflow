# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Device function for replicated training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import server_lib


class _RoundRobinStrategy(object):
  """Returns the next ps task index for placement in round-robin order.

  This class is not to be used directly by users.  See instead
  `replica_device_setter()` below.
  """

  def __init__(self, num_tasks):
    """Create a new `_RoundRobinStrategy`.

    Args:
      num_tasks: Number of ps tasks to cycle among.
    """
    self._num_tasks = num_tasks
    self._next_task = 0

  def __call__(self, unused_op):
    """Choose a ps task index for the given `Operation`.

    Args:
      unused_op: An `Operation` to be placed on ps.

    Returns:
      The next ps task index to use for the `Operation`. Returns the next
      index, in the range `[offset, offset + num_tasks)`.
    """
    task = self._next_task
    self._next_task = (self._next_task + 1) % self._num_tasks
    return task


class _ReplicaDeviceChooser(object):
  """Class to choose devices for Ops in a replicated training setup.

  This class is not to be used directly by users.  See instead
  `replica_device_setter()` below.
  """

  def __init__(self, ps_tasks, ps_device, worker_device, merge_devices, ps_ops,
               ps_strategy):
    """Create a new `_ReplicaDeviceChooser`.

    Args:
      ps_tasks: Number of tasks in the `ps` job.
      ps_device: String.  Name of the `ps` job.
      worker_device: String.  Name of the `worker` job.
      merge_devices: Boolean. Set to True to allow merging of device specs.
      ps_ops: List of strings representing `Operation` types that need to be
        placed on `ps` devices.
      ps_strategy: A callable invoked for every ps `Operation` (i.e. matched by
        `ps_ops`), that takes the `Operation` and returns the ps task index to
        use.
    """
    self._ps_tasks = ps_tasks
    self._ps_device = ps_device
    self._worker_device = worker_device
    self._merge_devices = merge_devices
    self._ps_ops = ps_ops
    self._ps_strategy = ps_strategy

  def device_function(self, op):
    """Choose a device for `op`.

    Args:
      op: an `Operation`.

    Returns:
      The device to use for the `Operation`.
    """
    # If we don't return early here, either merge_devices is True, or op.device
    # is empty (in which case merging is a no-op). So we can always merge below.
    if not self._merge_devices and op.device:
      return op.device

    current_device = pydev.DeviceSpec.from_string(op.device or "")

    # The ps_device will be used for specified ops (ps_ops) whenever it is
    # present and ps_tasks is non-zero. However, its task number will only be
    # set (using ps_strategy) if there is a job field in ps_device that won't be
    # changed by the job field (if present) in current_device.
    node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
    if self._ps_tasks and self._ps_device and node_def.op in self._ps_ops:
      ps_device = pydev.DeviceSpec.from_string(self._ps_device)

      current_job, ps_job = current_device.job, ps_device.job
      if ps_job and (not current_job or current_job == ps_job):
        ps_device.task = self._ps_strategy(op)

      ps_device.merge_from(current_device)
      return ps_device.to_string()

    worker_device = pydev.DeviceSpec.from_string(self._worker_device or "")
    worker_device.merge_from(current_device)
    return worker_device.to_string()


def replica_device_setter(ps_tasks=0, ps_device="/job:ps",
                          worker_device="/job:worker", merge_devices=True,
                          cluster=None, ps_ops=None, ps_strategy=None):
  """Return a `device function` to use when building a Graph for replicas.

  Device Functions are used in `with tf.device(device_function):` statement to
  automatically assign devices to `Operation` objects as they are constructed,
  Device constraints are added from the inner-most context first, working
  outwards. The merging behavior adds constraints to fields that are yet unset
  by a more inner context. Currently the fields are (job, task, cpu/gpu).

  If `cluster` is `None`, and `ps_tasks` is 0, the returned function is a no-op.
  Otherwise, the value of `ps_tasks` is derived from `cluster`.

  By default, only Variable ops are placed on ps tasks, and the placement
  strategy is round-robin over all ps tasks. A custom `ps_strategy` may be used
  to do more intelligent placement, such as
  `tf.contrib.training.GreedyLoadBalancingStrategy`.

  For example,

  ```python
  # To build a cluster with two ps jobs on hosts ps0 and ps1, and 3 worker
  # jobs on hosts worker0, worker1 and worker2.
  cluster_spec = {
      "ps": ["ps0:2222", "ps1:2222"],
      "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]}
  with tf.device(tf.train.replica_device_setter(cluster=cluster_spec)):
    # Build your graph
    v1 = tf.Variable(...)  # assigned to /job:ps/task:0
    v2 = tf.Variable(...)  # assigned to /job:ps/task:1
    v3 = tf.Variable(...)  # assigned to /job:ps/task:0
  # Run compute
  ```

  Args:
    ps_tasks: Number of tasks in the `ps` job.  Ignored if `cluster` is
      provided.
    ps_device: String.  Device of the `ps` job.  If empty no `ps` job is used.
      Defaults to `ps`.
    worker_device: String.  Device of the `worker` job.  If empty no `worker`
      job is used.
    merge_devices: `Boolean`. If `True`, merges or only sets a device if the
      device constraint is completely unset. merges device specification rather
      than overriding them.
    cluster: `ClusterDef` proto or `ClusterSpec`.
    ps_ops: List of strings representing `Operation` types that need to be
      placed on `ps` devices.  If `None`, defaults to `["Variable"]`.
    ps_strategy: A callable invoked for every ps `Operation` (i.e. matched by
      `ps_ops`), that takes the `Operation` and returns the ps task index to
      use.  If `None`, defaults to a round-robin strategy across all `ps`
      devices.

  Returns:
    A function to pass to `tf.device()`.

  Raises:
    TypeError if `cluster` is not a dictionary or `ClusterDef` protocol buffer,
    or if `ps_strategy` is provided but not a callable.
  """
  if cluster is not None:
    if isinstance(cluster, server_lib.ClusterSpec):
      cluster_spec = cluster.as_dict()
    else:
      cluster_spec = server_lib.ClusterSpec(cluster).as_dict()
    # Get ps_job_name from ps_device by stripping "/job:".
    ps_job_name = pydev.DeviceSpec.from_string(ps_device).job
    if ps_job_name not in cluster_spec or cluster_spec[ps_job_name] is None:
      return None
    ps_tasks = len(cluster_spec[ps_job_name])

  if ps_tasks == 0:
    return None

  if ps_ops is None:
    # TODO(sherrym): Variables in the LOCAL_VARIABLES collection should not be
    # placed in the parameter server.
    ps_ops = ["Variable", "VariableV2", "VarHandleOp"]

  if not merge_devices:
    logging.warning(
        "DEPRECATION: It is recommended to set merge_devices=true in "
        "replica_device_setter")
  if ps_strategy is None:
    ps_strategy = _RoundRobinStrategy(ps_tasks)
  if not six.callable(ps_strategy):
    raise TypeError("ps_strategy must be callable")
  chooser = _ReplicaDeviceChooser(
      ps_tasks, ps_device, worker_device, merge_devices, ps_ops, ps_strategy)
  return chooser.device_function
