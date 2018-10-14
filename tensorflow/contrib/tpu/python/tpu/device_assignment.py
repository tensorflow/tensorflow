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
# ======================================
"""Library of TPU helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.tpu.python.tpu.topology import Topology


def _tpu_device_name(job, task, device):
  """Returns the device name for the TPU `device` on `task` of `job`."""
  if job is None:
    return "/task:%d/device:TPU:%d" % (task, device)
  else:
    return "/job:%s/task:%d/device:TPU:%d" % (job, task, device)


def _tpu_host_device_name(job, task):
  """Returns the device name for the CPU device on `task` of `job`."""
  if job is None:
    return "/task:%d/device:CPU:0" % task
  else:
    return "/job:%s/task:%d/device:CPU:0" % (job, task)


class DeviceAssignment(object):
  """Mapping from logical cores in a computation to the physical TPU topology.

  Prefer to use the `device_assignment()` helper to construct a
  `DeviceAssignment`; it is easier if less flexible than constructing a
  `DeviceAssignment` directly.
  """

  def __init__(self, topology, core_assignment):
    """Constructs a `DeviceAssignment` object.

    Args:
      topology: A `Topology` object that describes the physical TPU topology.
      core_assignment: A logical to physical core mapping, represented as a
        rank 3 numpy array. See the description of the `core_assignment`
        property for more details.

    Raises:
      ValueError: If `topology` is not `Topology` object.
      ValueError: If `core_assignment` is not a rank 3 numpy array.
    """
    if not isinstance(topology, Topology):
      raise ValueError("topology must be a Topology object, got {}".format(
          type(topology)))
    core_assignment = np.asarray(core_assignment, dtype=np.int32)

    self._topology = topology
    self._topology_tasks, self._topology_devices = (
        self._invert_topology(topology))

    topology_rank = self._topology_tasks.ndim
    if core_assignment.ndim != 3:
      raise ValueError("core_assignment must be a rank 3 numpy array, "
                       "got shape {}".format(core_assignment.shape))

    self._num_replicas = core_assignment.shape[0]
    self._num_cores_per_replica = core_assignment.shape[1]

    if core_assignment.shape[-1] != topology_rank:
      raise ValueError(
          "minor dimension of core_assignment must have size equal to topology "
          "rank ({}), got shape {}".format(topology_rank,
                                           core_assignment.shape))

    self._core_assignment = core_assignment
    self._task_and_cores_to_replicas = self._compute_task_and_cores_to_replicas(
        self._core_assignment, self._topology_tasks)

  def _invert_topology(self, topology):
    """Inverts a [task,device,axis] topology to [x,y,z] -> task/device maps."""
    mesh_shape = topology.mesh_shape
    tasks = np.full(list(mesh_shape), -1, dtype=np.int32)
    devices = np.full(list(mesh_shape), -1, dtype=np.int32)
    for task in xrange(topology.device_coordinates.shape[0]):
      for device in xrange(topology.device_coordinates.shape[1]):
        x, y, z = topology.device_coordinates[task, device, :]
        tasks[x, y, z] = task
        devices[x, y, z] = device
    return tasks, devices

  def _compute_task_and_cores_to_replicas(self, core_assignment,
                                          topology_tasks):
    """Computes a nested dict which maps task and logical core to replicas."""
    task_and_cores_to_replicas = {}
    for replica in xrange(core_assignment.shape[0]):
      for logical_core in xrange(core_assignment.shape[1]):
        x, y, z = core_assignment[replica, logical_core, :]
        task_id = topology_tasks[x, y, z]
        if task_id not in task_and_cores_to_replicas:
          task_and_cores_to_replicas[task_id] = {}
        if logical_core not in task_and_cores_to_replicas[task_id]:
          task_and_cores_to_replicas[task_id][logical_core] = set()

        task_and_cores_to_replicas[task_id][logical_core].add(replica)

    task_to_sorted_replica_id = {}

    for task, core_to_replicas in task_and_cores_to_replicas.items():
      core_to_sorted_replicas = {}
      for core, replicas in core_to_replicas.items():
        core_to_sorted_replicas[core] = sorted(replicas)

      task_to_sorted_replica_id[task] = core_to_sorted_replicas
    return task_to_sorted_replica_id

  @property
  def topology(self):
    """A `Topology` that describes the TPU topology."""
    return self._topology

  @property
  def num_cores_per_replica(self):
    """The number of cores per replica."""
    return self._num_cores_per_replica

  @property
  def num_replicas(self):
    """The number of replicas of the computation."""
    return self._num_replicas

  @property
  def core_assignment(self):
    """The logical to physical core mapping.

    Returns:
      An integer numpy array of rank 3, with shape
      `[num_replicas, num_cores_per_replica, topology_rank]`. Maps
      (replica, logical core) pairs to physical topology coordinates.
    """
    return self._core_assignment

  def _coordinates(self, replica, logical_core):
    """Returns the physical topology coordinates of a logical core."""
    return tuple(self.core_assignment[replica, logical_core, :])

  def lookup_replicas(self, task_id, logical_core):
    """Lookup replica ids by task number and logical core.

    Args:
      task_id: TensorFlow task number.
      logical_core: An integer, identifying a logical core.
    Returns:
      A sorted list of the replicas that are attached to that task and
      logical_core.
    Raises:
      ValueError: If no replica exists in the task which contains the logical
      core.
    """
    try:
      return self._task_and_cores_to_replicas[task_id][logical_core]
    except KeyError:
      raise ValueError(
          "Can not find any replica in task: {} contains logical_core: {} ".
          format(task_id, logical_core))

  def tpu_ordinal(self, replica=0, logical_core=0):
    """Returns the ordinal of the TPU device assigned to a logical core."""
    coordinates = self._coordinates(replica, logical_core)
    return self._topology_devices[coordinates]

  def host_device(self, replica=0, logical_core=0, job=None):
    """Returns the CPU device attached to a logical core."""
    coordinates = self._coordinates(replica, logical_core)
    return _tpu_host_device_name(job, self._topology_tasks[coordinates])

  def tpu_device(self, replica=0, logical_core=0, job=None):
    """Returns the name of the TPU device assigned to a logical core."""
    coordinates = self._coordinates(replica, logical_core)
    return _tpu_device_name(job, self._topology_tasks[coordinates],
                            self._topology_devices[coordinates])


def device_assignment(topology,
                      computation_shape=None,
                      computation_stride=None,
                      num_replicas=1):
  """Computes a device_assignment of a computation across a TPU topology.

  Attempts to choose a compact grid of cores for locality.

  Returns a `DeviceAssignment` that describes the cores in the topology assigned
  to each core of each replica.

  `computation_shape` and `computation_stride` values should be powers of 2 for
  optimal packing.

  Args:
    topology: A `Topology` object that describes the TPU cluster topology.
      To obtain a TPU topology, evaluate the `Tensor` returned by
      `initialize_system` using `Session.run`. Either a serialized
      `TopologyProto` or a `Topology` object may be passed. Note: you must
      evaluate the `Tensor` first; you cannot pass an unevaluated `Tensor` here.
    computation_shape: A rank 1 int32 numpy array with size equal to the
      topology rank, describing the shape of the computation's block of cores.
      If None, the `computation_shape` is `[1] * topology_rank`.
    computation_stride: A rank 1 int32 numpy array of size `topology_rank`,
      describing the inter-core spacing of the `computation_shape` cores in the
      TPU topology. If None, the `computation_stride` is `[1] * topology_rank`.
    num_replicas: The number of computation replicas to run. The replicas will
      be packed into the free spaces of the topology.

  Returns:
    A DeviceAssignment object, which describes the mapping between the logical
    cores in each computation replica and the physical cores in the TPU
    topology.

  Raises:
    ValueError: If `topology` is not a valid `Topology` object.
    ValueError: If `computation_shape` or `computation_stride` are not 1D int32
      numpy arrays with shape [3] where all values are positive.
    ValueError: If computation's replicas cannot fit into the TPU topology.
  """
  # Deserialize the Topology proto, if it is a string.
  if isinstance(topology, bytes):
    topology = Topology(serialized=topology)

  if not isinstance(topology, Topology):
    raise ValueError("`topology` is not a Topology object; got {}".format(
        type(topology)))

  topology_rank = len(topology.mesh_shape)
  mesh_shape = topology.mesh_shape
  if computation_shape is None:
    computation_shape = np.array([1] * topology_rank, dtype=np.int32)
  else:
    computation_shape = np.asarray(computation_shape, dtype=np.int32)

  if computation_stride is None:
    computation_stride = np.array([1] * topology_rank, dtype=np.int32)
  else:
    computation_stride = np.asarray(computation_stride, dtype=np.int32)

  if computation_shape.shape != (topology_rank,):
    raise ValueError("computation_shape must have shape [{}]; got {}".format(
        topology_rank, computation_shape.shape))
  if computation_stride.shape != (topology_rank,):
    raise ValueError("computation_stride must have shape [{}]; got {}".format(
        topology_rank, computation_stride.shape))

  if any(computation_shape < 1):
    raise ValueError(
        "computation_shape must be positive; got computation_shape={}".format(
            computation_shape))
  if any(computation_stride < 1):
    raise ValueError(
        "computation_stride must be positive; got computation_stride={}".format(
            computation_stride))

  # Computes the physical size of one computation instance.
  computation_footprint = computation_shape * computation_stride
  if any(computation_footprint > mesh_shape):
    raise ValueError(
        "computation footprint {} does not fit in TPU topology shape {}".format(
            computation_footprint, mesh_shape))

  # Computes how many copies of the computation footprint fit in the mesh.
  block_counts = mesh_shape // computation_footprint

  replica_counts = block_counts * computation_stride
  max_replicas = np.prod(replica_counts)
  if num_replicas > max_replicas:
    raise ValueError(
        "requested {} replicas but only {} replicas with shape {} and "
        "computation_stride {} fit in a TPU mesh of shape {}".format(
            num_replicas, max_replicas, computation_shape, computation_stride,
            mesh_shape))

  def ceil_of_ratio(n, m):
    return (n + m - 1) // m

  replica_shape = [0] * topology_rank
  if num_replicas > 0:
    remaining_replicas = num_replicas
    remaining_dims = topology_rank

    # Choose dimensions as close to an equal cube as possible, in order of
    # increasing dimension size. By visiting dimensions in increasing size, we
    # assign the most constrained dimension first, so we won't make infeasible
    # choices.
    #
    # As a secondary sort order, visit the dimensions in reverse order. This
    # means we try to use both cores on the same chip in preference to two cores
    # on different chips.
    for x, ni in sorted(((x, -i) for (i, x) in enumerate(replica_counts))):
      i = -ni
      target_size = int(math.ceil(remaining_replicas**(1.0 / remaining_dims)))
      replica_shape[i] = min(target_size, x)
      remaining_replicas = ceil_of_ratio(remaining_replicas, replica_shape[i])
      remaining_dims -= 1

    assert remaining_replicas == 1 and remaining_dims == 0

  # Assigns an offset to each replica such that no two replicas overlap.
  replica_offsets = np.full([num_replicas, topology_rank], -1, dtype=np.int32)
  for replica in xrange(num_replicas):
    # Chooses a replica number in each axis.
    t = replica
    pos = []
    for dim in replica_shape[::-1]:
      pos.append(t % dim)
      t //= dim
    replica_pos = np.array(pos[::-1], dtype=np.int32)

    # Determines where that replica starts in each axis.
    outer = replica_pos // computation_stride
    inner = replica_pos % computation_stride
    replica_offsets[replica, :] = outer * computation_footprint + inner

  # Computes a complete logical core -> physical core mapping for each replica.
  indices = [
      np.arange(0, computation_shape[i] * computation_stride[i],
                computation_stride[i]) for i in xrange(topology_rank)
  ]
  indices = np.concatenate(
      [i[..., np.newaxis] for i in np.meshgrid(*indices, indexing="ij")],
      axis=-1)
  indices = indices.reshape((-1, topology_rank))
  assignment = indices + replica_offsets[:, np.newaxis, :]
  return DeviceAssignment(topology, core_assignment=assignment)
