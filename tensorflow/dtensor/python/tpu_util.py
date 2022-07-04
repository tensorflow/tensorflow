# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""TPU-specific utilities for DTensor."""

import functools
import time
from typing import List, Optional, Dict

from absl import flags
import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import heartbeat
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import multi_client_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import topology
from tensorflow.python.util.tf_export import tf_export

_INITIALIZED_TPU_SYSTEMS = {}
_MESH_DIM_X = "x"
_TPU_DEVICE_TYPE = "TPU"

# A dedicated, hidden device used to make C++ API calls.
_dtensor_device = None

# `_topology._mesh_shape` contains the TPU hardware slice size.
# `_topology.device_coordinates` maps TF task-device ordinals to TPU core IDs.
_tpu_topology = None

# Cache core ID <-> location mappings so we need not make repeated C++ calls.
# Both are indexed by TF task-device ordinals.
_all_core_ids = None
_all_core_locations = None


class _CoreLocation:
  """Represents a TPU core's location in the mesh."""

  def __init__(self, x: int = 0, y: int = 0, z: int = 0, core: int = 0):
    self.x = x
    self.y = y
    self.z = z
    self.core = core

  def __eq__(self, other):
    if not isinstance(other, _CoreLocation):
      return False
    return self.x == other.x and self.y == other.y and self.z == other.z and self.core == other.core

  def __ne__(self, other):
    if not isinstance(other, _CoreLocation):
      return True
    return not self == other

  def __hash__(self):
    return hash((self.x, self.y, self.z, self.core))

  def __repr__(self):
    return f"{type(self).__name__}(x={self.x}, y={self.y}, z={self.z}, core={self.core})"

  def to_list(self):
    return [self.x, self.y, self.z, self.core]


def _create_device_array(shape, device_type, host_id, local_device_ids=None):
  """Returns ID and device lists that can be used to create a mesh."""
  num_global_devices = api.num_global_devices(device_type)
  global_device_ids = np.arange(num_global_devices).reshape(shape)
  local_device_list = api.local_devices(device_type)

  # User can specify local_device_ids or use default list for multi host.
  num_local_devices = len(local_device_list)
  local_device_ids = [
      x + host_id * num_local_devices for x in range(num_local_devices)
  ] if not local_device_ids else local_device_ids

  return global_device_ids, local_device_ids, local_device_list


def _create_tpu_topology(core_locations: List[_CoreLocation], num_tasks: int,
                         num_devices_per_task: int) -> topology.Topology:
  """Returns a Topology object build from a _CoreLocation list.

  Args:
    core_locations: A list of _CoreLocation objects sorted first by TF task ID
      and then by per-task device ordinals.
    num_tasks: The number of TF tasks in the cluster.
    num_devices_per_task: The number of TPU devices local to each task.
  """

  assert min([l.x for l in core_locations]) == 0
  assert min([l.y for l in core_locations]) == 0
  assert min([l.z for l in core_locations]) == 0
  assert min([l.core for l in core_locations]) == 0
  x_max = max([l.x for l in core_locations])
  y_max = max([l.y for l in core_locations])
  z_max = max([l.z for l in core_locations])
  core_max = max([l.core for l in core_locations])
  mesh_shape = [x_max + 1, y_max + 1, z_max + 1, core_max + 1]

  device_coordinates = [[l.x, l.y, l.z, l.core] for l in core_locations]
  device_coordinates = np.asarray(device_coordinates).reshape(
      num_tasks, num_devices_per_task, 4)

  return topology.Topology(
      mesh_shape=mesh_shape, device_coordinates=device_coordinates)


@tf_export("experimental.dtensor.shutdown_tpu_system", v1=[])
def dtensor_shutdown_tpu_system():
  """Shutdown TPU system."""

  @def_function.function
  def _shutdown_tpu_system():
    return gen_dtensor_ops.shutdown_tpu_system()

  success = _shutdown_tpu_system() if context.is_tfrt_enabled() else True
  if success:
    logging.info("TPU system shut down.")
  else:
    logging.warning("TPU system fails to shut down.")


@tf_export("experimental.dtensor.initialize_tpu_system", v1=[])
def dtensor_initialize_tpu_system(enable_coordination_service=False):
  """Initialize the TPU devices.

  This functions performs additional TPU related initialization after
  calling `dtensor.initialize_multi_client` to initialize multi-client DTensor.
  Refer to `dtensor.initialize_multi_client` for relevant environment
  variables that controls the initialization of multi-client DTensor.

  Args:
    enable_coordination_service: If true, enable distributed coordination
      service to make sure that workers know the devices on each other, a
      prerequisite for data transfer through cross-worker rendezvous.

  Raises:
    RuntimeError: If running inside a tf.function.
    NotFoundError: If no TPU devices found in eager mode.
  """

  assert context.executing_eagerly()

  # Reconfigure TensorFlow to use TFRT TPU runtime if requested.
  _configure_tpu_runtime()

  in_multi_client_mode = api.job_name() != "localhost"

  # Collective GRPC servers are only necessary in mutli-client setup.
  # Single clients can use local mode of collectives.
  if in_multi_client_mode:
    if api.jobs() is None:
      raise ValueError(
          "DTENSOR_JOBS environment variable is required when"
          "using multi-client to properly set up communications between servers"
      )
    multi_client_util.initialize_multi_client_cluster(
        job_name=api.job_name(),
        dtensor_jobs=api.jobs(),
        client_id=api.client_id(),
        collective_leader=api.full_job_name(task_id=0),
        enable_coordination_service=enable_coordination_service)

  # Make sure the server change is fully propagated before attempting to run
  # the core ID merging logic below.
  context.ensure_initialized()
  context.async_wait()
  context.context()._clear_caches()  # pylint: disable=protected-access

  @function.defun
  def _tpu_init_fn():
    return gen_dtensor_ops.configure_and_initialize_global_tpu()

  try:
    with ops.device("/job:" + api.full_job_name() + "/device:TPU_SYSTEM:0"):  # pylint: disable=protected-access
      my_core_ids = _tpu_init_fn()
    logging.info("TPU core IDs: %s", my_core_ids)
    context.initialize_logical_devices()

    # Configure virtual CPUs that is 1:1 mapped to TPU cores.
    context.context().set_logical_cpu_devices(
        len(api.local_devices(_TPU_DEVICE_TYPE)),
        tf_device.DeviceSpec(
            job=api.job_name(), replica=0, task=api.client_id()).to_string())

    # `my_core_ids` contains the IDs of TPU cores attached to this host.
    #
    # To generate correct and efficient XLA AllReduce group assignment, we must
    # merge these arrays from all hosts and broadcast the result back to all
    # hosts, so all hosts can use these mappings in their MLIR passes.
    #
    # This is essentially doing what WaitForDistributedTpuOp and
    # SetGlobalTPUArrayOp do, in our multi-client environment.
    task_id = api.client_id()
    num_tasks = api.num_clients()
    num_devices = api.num_global_devices(_TPU_DEVICE_TYPE)
    num_devices_per_task = int(num_devices / num_tasks)

    # Create a one-time use mesh and layout just for merging core IDs.
    mesh = layout_lib.Mesh([_MESH_DIM_X],
                           *_create_device_array((num_devices,),
                                                 _TPU_DEVICE_TYPE,
                                                 api.client_id()))
    layout = layout_lib.Layout([_MESH_DIM_X, layout_lib.UNSHARDED], mesh)
    device = dtensor_device.DTensorDevice(meshes=[mesh])
    logging.info("TPU core locations: %s",
                 device.tpu_core_ids_to_locations(my_core_ids))

    # At this point, we don't know which cores are attached to other hosts.
    # The core ID mappings in the runtime haven't been set yet.
    #
    # The core ID merging AllReduce below is carefully written so it works
    # without needing correct core mappings to be set in the runtime. We will
    # use this AllReduce's result to set the core ID mappings, and all future
    # user-initiated AllReduces will use the mappings.
    #
    # The runtime is hard-coded to ignore core ID mappings on this AllReduce.
    all_core_ids = np.zeros([num_devices], dtype=np.int32)
    for i in range(len(my_core_ids)):
      all_core_ids[task_id * num_devices_per_task + i] = my_core_ids[i]

    # Only one local device gets valid input: 8 local core IDs among
    # (num_tasks - 1) * 8 zeros. The 8 core IDs are set using task ID as offset.
    # The other 7 local devices get zero inputs. All devices on all host
    # participate in one AllReduce, whose result will be core IDs arranged by
    # task-device ordinals.
    all_core_ids = constant_op.constant([all_core_ids])
    zeros = array_ops.zeros_like(all_core_ids)
    all_core_ids = [all_core_ids] + [zeros] * (num_devices_per_task - 1)

    with ops.device(device.name):
      all_core_ids = device.pack(all_core_ids, layout)
      all_core_ids = math_ops.reduce_sum(all_core_ids, axis=[0])
      unpacked_all_tpu_ids = device.unpack(all_core_ids)

    all_core_ids = list(unpacked_all_tpu_ids[0].numpy())
    logging.info("All TPU core IDs: %s", all_core_ids)

    # Set the default core ID mappings in the runtime for legacy code and tests.
    #
    # Legacy code and tests create TPU meshes directly without using the
    # `create_tpu_mesh` function below. Those meshes have global device IDs
    # equal to TF task-device ordinals. The `all_core_ids` array happens to
    # arrange core IDs by TF task-device ordinals. Using this array on those
    # meshes guarantee correct although inefficient results.
    device.set_tpu_core_ids("", all_core_ids)

    # Remember enough global, immutable information to be able to build any ring
    # we want prescribed by `create_tpu_mesh` in the future.
    global _all_core_ids
    _all_core_ids = all_core_ids

    all_core_locations = device.tpu_core_ids_to_locations(all_core_ids)
    all_core_locations = [
        _CoreLocation(l[0], l[1], l[2], l[3]) for l in all_core_locations
    ]
    global _all_core_locations
    _all_core_locations = all_core_locations
    logging.info("All TPU core locations: %s", all_core_locations)

    tpu_topology = _create_tpu_topology(all_core_locations, num_tasks,
                                        num_devices_per_task)
    global _tpu_topology
    _tpu_topology = tpu_topology
    logging.vlog(1, "TPU Topology: %s, %s", tpu_topology.mesh_shape,
                 tpu_topology.device_coordinates)

    global _dtensor_device
    _dtensor_device = device

    context.async_wait()

  except errors.InvalidArgumentError as e:
    raise errors.NotFoundError(
        None, None,
        "Initialization failed, no valid TPUs found. " + str(e)) from e

  except errors.InternalError as e:
    logging.error("Hit internal error during TPU system initialization. "
                  + "It is likely hareware failure. \nPlease check the error "
                  + "messages above to see whether that's the case. \nIf so, "
                  + "consider to restart the job or try another machine.")
    raise e

  # Optionally exchange heartbeats between workers every minute.
  if in_multi_client_mode and api.heartbeat_enabled():
    logging.info(
        "Starting DTensor heartbeat service exchanging signals every 10 minutes"
    )
    heartbeat.start(period=180)

  # Clear out the eager context caches since the memory is invalid now.
  logging.info("Clearing out eager caches")
  context.context()._clear_caches()  # pylint: disable=protected-access


def _enumerate_cores(bounds: List[int], ring_bounds: List[int],
                     ring_sizes: List[int], host_bounds: List[int],
                     host_sizes: List[int]) -> List[List[int]]:
  """Enumerates cores within `bounds` from fatest to slowest varying axes.

  Args:
    bounds: Upper bounds of axes, from fastest to slowest varying.
    ring_bounds: Upper bounds of ring size per axis in the same axis order.
    ring_sizes: Number consecutive cores in the ring built so far, cumulatively.
    host_bounds: Number of axis values per host in the same axis order.
    host_sizes: Number consecutive cores on one host, cumulatively.

  Returns:
    Cores represented as a list of 4 integers in the same axis order.
  """
  if not bounds:
    return [[]]

  # Recursively enumerate cores under all but the slowest varying axis.
  partials = _enumerate_cores(bounds[:-1], ring_bounds[:-1], ring_sizes[:-1],
                              host_bounds[:-1], host_sizes[:-1])

  # Append the slowest varying axis to the end of all partial results.
  # From ring_i|j to host_i|j to core_i|j, use progressively smaller or equal
  # iteration groupings until every one of the bounds[-1] * len(partials)
  # combinations is iterated on.
  # Despite the six levels of nested loops below, the total time complexity for
  # this invocation is O(N), where N is the number of cores in the topology.
  results = []
  for ring_i in range(0, bounds[-1], ring_bounds[-1]):
    for ring_j in range(0, len(partials), ring_sizes[-1]):
      for host_i in range(ring_i, ring_i + ring_bounds[-1], host_bounds[-1]):
        for host_j in range(ring_j, ring_j + ring_sizes[-1], host_sizes[-1]):
          for i in range(host_i, host_i + host_bounds[-1]):
            for j in range(host_j, host_j + host_sizes[-1]):
              results.append(partials[j] + [i])
  return results


def _enumerate_core_locations(bounds: List[int], ring_bounds: List[int],
                              axes: List[str],
                              can_split_host_across_rings: bool,
                              ring_size: int) -> List[_CoreLocation]:
  """Enumerates all possible core locations under the axis iteration order.

  Args:
    bounds: A list of 4 positive integers, upper bound values for x, y, z, core.
    ring_bounds: A list of 4 positive integers, upper bound values for ring size
      in x, y, z, core axes.
    axes: A permutation of ["x", "y", "z", "core"], the axis iteration order.
    can_split_host_across_rings: If true, devices attached to the same host may
      get assigned to different rings.
    ring_size: Number of devices in a ring, only for argument validation.

  Returns:
    A list of all CoreLocation objects defined in a TPU slice of shape `bounds`,
    sorted by axis iteration order specified by `axes`.

    For example, given bounds=[2, 2, 1, 2] and axes=["core", "z", "y", "x"],
    return 8 core locations expressed in (x, y, z, core) format but iterated in
    core -> z -> y -> x order (fatest to slowest varying):

    [_CoreLocation(0, 0, 0, 0),
     _CoreLocation(0, 0, 0, 1),
     _CoreLocation(0, 1, 0, 0),
     _CoreLocation(0, 1, 0, 1),
     _CoreLocation(1, 0, 0, 0),
     _CoreLocation(1, 0, 0, 1),
     _CoreLocation(1, 1, 0, 0),
     _CoreLocation(1, 1, 0, 1)]

  Raises:
    ValueError: If ring_size cannot be fulfilled without splitting hosts.
  """

  num_cores_per_chip = bounds[3]
  if num_cores_per_chip != 1 and num_cores_per_chip != 2:
    raise ValueError("Unsupported TPU slice size: %s" % bounds)

  # Translate `axes` from string to integer format.
  axes = [{"x": 0, "y": 1, "z": 2, "core": 3}[axis] for axis in axes]
  # Reorder bounds from fastest to slowest varying axes.
  bounds = [bounds[i] for i in axes]

  # Set and validate host_bounds.
  if can_split_host_across_rings:
    # If we can split hosts, shrink every host to effectively contain 1 device.
    host_bounds = [1, 1, 1, 1]
  elif np.prod(bounds) <= 2:
    # We must be running on 1x1 or 1x1x1 Forge.
    host_bounds = [[1, 1, 1, num_cores_per_chip][i] for i in axes]
  else:
    # Other cases including 2x2 Forge and Borg must use a full donut.
    host_bounds = [[2, 2, 1, num_cores_per_chip][i] for i in axes]
  # host_sizes is the cumulative products of host_bounts.
  host_sizes = [1]
  for host_bound in host_bounds:
    host_sizes.append(host_sizes[-1] * host_bound)
  host_size = host_sizes.pop()
  # When can_split_host_across_rings is false, a ring must contain at least as
  # many devices as a host has.
  if ring_size < host_size:
    assert not can_split_host_across_rings
    raise ValueError(
        "Rings too small for can_split_host_across_rings = False: %d" %
        ring_size)

  # Reorder ring_bounds and validate it's element-wise >= host_bounds.
  ring_bounds = [ring_bounds[i] for i in axes]
  if ring_bounds < host_bounds:
    raise ValueError("ring_bounds %s should be >= host_bounds %s" %
                     (ring_bounds, host_bounds))
  ring_sizes = [1]
  # ring_sizes is the cumulative products of ring_bounds.
  for ring_bound in ring_bounds:
    ring_sizes.append(ring_sizes[-1] * ring_bound)
  ring_sizes.pop()

  # Enumerate cores in the given iteration order. Each core is represented as a
  # list of int, which are offsets from fatest to slowest varying axes.
  cores = _enumerate_cores(bounds, ring_bounds, ring_sizes, host_bounds,
                           host_sizes)
  # Reorder offsets of each core back to the x, y, z, core order.
  core_locations = []
  for core in cores:
    core = [core[axes.index(i)] for i in range(4)]
    core_locations.append(_CoreLocation(core[0], core[1], core[2], core[3]))
  return core_locations


def _build_all_reduce_ring(core_locations: List[_CoreLocation],
                           rotate: bool = False) -> List[int]:
  """Reorders a list of TPU cores to optimize for AllReduce performance.

  This is ported from the C++ tensorflow::BuildAllReduceRing function,
  mixed with some logic from TF TPU's device_assignment._ring_3d.

  Args:
    core_locations: A list of core locations expressed as [x, y, z, core].
    rotate: If true, scan the cores in a column-major order. False by default.

  Returns:
    A permutation of the input list such that neighbors in the sequence are
    nearby in the TPU topology.
  """

  permutation = list(range(len(core_locations)))
  if not permutation:
    return permutation
  logging.vlog(2, "Core locations in: %s", core_locations)

  first_column = min([l.x for l in core_locations])
  first_row = min([l.y for l in core_locations])
  same_z = (len(set([l.z for l in core_locations])) == 1)
  logging.vlog(2, "first_column: %d", first_column)
  logging.vlog(2, "first_row: %d", first_row)
  logging.vlog(2, "same_z: %s", same_z)

  def _cmp_2d(ia: int, ib: int) -> int:
    if not rotate:
      a = core_locations[ia]
      b = core_locations[ib]

      # Order the first column last in the sequence, except for the first row.
      a_first = (a.x == first_column and a.y != first_row)
      b_first = (b.x == first_column and b.y != first_row)
      if a_first != b_first:
        return -1 if b_first else 1

      # Order rows in increasing order, unless in the first column.
      if a.y != b.y:
        return b.y - a.y if a_first else a.y - b.y

      # Order even rows left to right, odd rows right to left.
      if a.x != b.x:
        return a.x - b.x if a.y % 2 == 0 else b.x - a.x

      # Order cores in increasing order.
      return a.core - b.core
    else:
      a = core_locations[ia]
      b = core_locations[ib]

      # Order the first row last in the sequence, except for the first column.
      a_first = (a.y == first_row and a.x != first_column)
      b_first = (b.y == first_row and b.x != first_column)
      if a_first != b_first:
        return -1 if b_first else 1

      # Order columns in increasing order, unless in the first row.
      if a.x != b.x:
        return b.x - a.x if a_first else a.x - b.x

      # Order even columns top down, odd columns bottom up.
      if a.y != b.y:
        return a.y - b.y if a.x % 2 == 0 else b.y - a.y

      # Order cores in increasing order.
      return a.core - b.core

  def _cmp_3d(ia: int, ib: int) -> int:
    a = core_locations[ia]
    b = core_locations[ib]

    a_corner = (a.x == first_column and a.y == first_row)
    b_corner = (b.x == first_column and b.y == first_row)

    # If both are in the corner, order in reverse z then core order.
    if a_corner and b_corner:
      return b.z - a.z if a.z != b.z else a.core - b.core

    # Corner cores always go after non-corner cores.
    if a_corner != b_corner:
      return -1 if b_corner else 1

    # Both non-corner cores are on the same z-plane. Reverse odd z-planes.
    if a.z == b.z:
      return _cmp_2d(ia, ib) if a.z % 2 == 0 else -_cmp_2d(ia, ib)

    # Both non-corner cores are on different z-planes. Smaller z goes first.
    return a.z - b.z

  # If all cores are on the same z-plane, order as usual. Otherwise, order
  # neighbor z-planes in opposite orders. Stack all z-planes along the z axis
  # and connect them in one corner.
  if same_z:
    permutation.sort(key=functools.cmp_to_key(_cmp_2d))
  else:
    permutation.sort(key=functools.cmp_to_key(_cmp_3d))
  logging.vlog(2, "Permutation out: %s", permutation)
  return permutation


def _build_orthogonal_rings(
    core_locations: List[_CoreLocation], ring_size: int,
    rotate_ring_across_rings: bool) -> List[_CoreLocation]:
  """Build two all-reduce rings orthogonal to each other.

  One ring includes every `ring_size` consecutive core locations. It is usually
  applied to the model-parallel dimension of a mesh to achieve best 1D
  all-reduce performance. The other ring includes core locations separated by
  a stride of `ring_size`. It is usually applied to the data-parallel dimension
  of a mesh to get predictable strided all-reduce performance.

  Args:
    core_locations: A list of core locations expressed as [x, y, z, core].
    ring_size: The number of core locations in the consecutive ring.
    rotate_ring_across_rings: Build column-major secondary rings.

  Returns:
    A permutation of the input list forming the described rings.
  """
  # Build a ring for the first `ring_size` cores, and apply that permutation to
  # every group of `ring_size` cores.
  num_cores = len(core_locations)
  permutation = _build_all_reduce_ring(core_locations[:ring_size])
  for r in range(0, num_cores, ring_size):
    core_locations[r:r + ring_size] = [
        core_locations[r + permutation[i]] for i in range(ring_size)
    ]
  logging.vlog(1, "Permutated core locations: %s", core_locations)

  # Build a "ring" for the collection of devices consisting of the 0th device
  # from every group, and apply that permutation to every i-th device group.
  # This is achieved by transposing the list and back.
  transposed = []
  for i in range(ring_size):
    transposed += [
        core_locations[g + i] for g in range(0, num_cores, ring_size)
    ]

  num_rings = int(num_cores / ring_size)
  permutation = _build_all_reduce_ring(
      transposed[:num_rings], rotate=rotate_ring_across_rings)
  for r in range(0, num_cores, num_rings):
    transposed[r:r + num_rings] = [
        transposed[r + permutation[i]] for i in range(num_rings)
    ]

  untransposed = []
  for i in range(num_rings):
    untransposed += [transposed[g + i] for g in range(0, num_cores, num_rings)]
  logging.vlog(1, "Stride-permutated core locations: %s", untransposed)

  return untransposed


def create_tpu_mesh(mesh_dim_names: List[str],
                    mesh_shape: List[int],
                    mesh_name: str,
                    ring_dims: Optional[int] = None,
                    ring_axes: Optional[List[str]] = None,
                    ring_bounds: Optional[List[int]] = None,
                    can_split_host_across_rings: bool = True,
                    build_ring_across_rings: bool = False,
                    rotate_ring_across_rings: bool = False) -> layout_lib.Mesh:
  """Returns a TPU mesh optimized for AllReduce ring reductions.

  Only as many as leading axes specified by `ring_axes` as necessary will be
  used to build rings, as long as the subslice formed by these axes have enough
  cores to contain a ring of the required size. The leftover axes in `ring_axes`
  won't affect results.

  Args:
    mesh_dim_names: List of mesh dimension names.
    mesh_shape: Shape of the mesh.
    mesh_name: A unique name for the mesh. If empty, internally generate one.
    ring_dims: Optional; The number of leading (ring_dims > 0) or trailing
      (ring_dims < 0) mesh dimensions to build rings for. If unspecified, build
      rings for all but the first dimension.
    ring_axes: Optional; A permutation of ["x", "y", "z", "core"], specifying
      the order of TPU topology axes to build rings in. If unspecified, default
      to ["core", "x", "y", "z"].
    ring_bounds: Optional; The maximum number of devices on each axis, in the x,
      y, z, core order. If unspecified, default to physical topology limits.
    can_split_host_across_rings: Optional; If true, devices attached to the same
      host (i.e., DTensor client) may get assigned to different rings. Setting
      it to false may cause some combinations of arguments to be infeasible; see
      DeviceAssignmentTest.testCreateMesh[No]SplittingHosts* for examples.
    build_ring_across_rings: Optional; If true, also build a data-parallel ring
      across model-parallel rings. This ring could be strided.
    rotate_ring_across_rings: Optional; If true, build the data-parallel ring in
      column-major instead of row-major order.
  """

  logging.info("Building a TPU mesh %s of shape %s", mesh_name, mesh_shape)
  logging.info("Requested ring_dims: %s", ring_dims)
  logging.info("Requested ring_axes: %s", ring_axes)
  logging.info("Requested ring_bounds: %s", ring_bounds)
  logging.info("Requested can_split_host_across_rings: %s",
               can_split_host_across_rings)
  if not mesh_name:
    mesh_name = "mesh_%f" % time.time()
  logging.info("Requested mesh_name: %s", mesh_name)

  # By default, build rings for all but the first (usually batch) dimension.
  if ring_dims is None:
    ring_dims = 1 - len(mesh_shape)
  elif ring_dims < -len(mesh_shape) or ring_dims > len(mesh_shape):
    raise ValueError("Invalid ring_dims value: %d" % ring_dims)
  logging.info("Actual ring_dims: %s", ring_dims)

  # By default, vary axes in the core -> x -> y -> z order.
  if ring_axes is None:
    ring_axes = ["core", "x", "y", "z"]
  elif len(ring_axes) != 4:
    raise ValueError("Expected 4 elements in ring_axes, got %s" % ring_axes)
  elif sorted(ring_axes) != ["core", "x", "y", "z"]:
    raise ValueError("Invalid ring_axes value: %s" % ring_axes)
  logging.info("Actual ring_axes: %s", ring_axes)

  # Validate ring_bounds values.
  global _tpu_topology
  if _tpu_topology is None:
    raise ValueError(
        "Invalid TPU topology, run dtensor.initialize_tpu_system() first")
  topology_shape = list(_tpu_topology.mesh_shape)
  if ring_bounds is None:
    ring_bounds = topology_shape
  elif len(ring_bounds) != 4:
    raise ValueError("Expected 4 elements in ring_bounds, got %s" % ring_bounds)
  elif ring_bounds > topology_shape:
    raise ValueError("ring_bounds %s should be <= topology sizes %s" %
                     (ring_bounds, topology_shape))
  logging.info("Actual ring_bounds: %s", ring_bounds)

  # Compute ring_size, the number of cores in a ring.
  if ring_dims > 0:
    ring_size = np.prod(mesh_shape[:ring_dims])
  elif ring_dims < 0:
    ring_size = np.prod(mesh_shape[ring_dims:])
  else:
    ring_size = 1  # single-core rings
  logging.info("Actual ring_size: %d", ring_size)

  # Rearrange all cores according to the axis iteration order.
  global_core_locations = _enumerate_core_locations(
      topology_shape, ring_bounds, ring_axes, can_split_host_across_rings,
      ring_size)
  logging.vlog(1, "Enumerated core locations: %s", global_core_locations)
  num_cores = len(global_core_locations)

  # The mesh to be created must use all TPU cores in the system.
  mesh_size = np.prod(mesh_shape)
  if mesh_size != num_cores:
    raise ValueError(
        "Invalid mesh size: mesh shape %s cannot 1:1 map to %d TPU cores" %
        (mesh_shape, num_cores))

  # Build a ring for the `ring_size` dimension and, if required, a strided ring
  # for the orthogonal dimension.
  if build_ring_across_rings:
    global_core_locations = _build_orthogonal_rings(global_core_locations,
                                                    ring_size,
                                                    rotate_ring_across_rings)
  else:
    permutation = _build_all_reduce_ring(global_core_locations[:ring_size])
    for r in range(0, num_cores, ring_size):
      global_core_locations[r:r + ring_size] = [
          global_core_locations[r + permutation[i]] for i in range(ring_size)
      ]
    logging.vlog(1, "Permutated core locations: %s", global_core_locations)

  # For this point on, change from List[CoreLocation] to List[List[int]] for
  # easier interaction with the C++ API.
  global_core_locations = [l.to_list() for l in global_core_locations]
  global _dtensor_device
  if _dtensor_device is None:
    raise ValueError(
        "Invalid system device, run dtensor.initialize_tpu_system() first")
  global_core_ids = _dtensor_device.tpu_core_locations_to_ids(
      global_core_locations)

  # Store a per-mesh mapping in the runtime.
  _dtensor_device.set_tpu_core_ids(mesh_name, global_core_ids)

  # Create the mesh by manually specifying local_device_ids.
  local_core_locations = _tpu_topology.device_coordinates[api.client_id()]
  indexes = [
      global_core_locations.index(list(local_core_location))
      for local_core_location in local_core_locations
  ]
  global_device_ids, local_device_ids, local_device_list = _create_device_array(
      mesh_shape, _TPU_DEVICE_TYPE, None, local_device_ids=indexes)
  return layout_lib.Mesh(mesh_dim_names, global_device_ids, local_device_ids,
                         local_device_list, mesh_name)


def get_device_ids(mesh: layout_lib.Mesh,
                   client_id: Optional[int] = None) -> List[int]:
  """Returns the device IDs of all TPU cores local to the given client.

  A device ID is a non-negative integer that uniquely identifies a device in the
  mesh. For example, for a 2x2 mesh ('x', 'y'), this function returns a
  permutation of [0, 1, 2, 3].

  Note that device IDs and device locations are equivalent. The former is a
  linearization of the latter along mesh dimensions.

  Args:
    mesh: A TPU mesh.
    client_id: Optional; A DTensor client ID. If empty, query this client.
  """

  if mesh.device_type() != _TPU_DEVICE_TYPE:
    raise ValueError("The mesh must be a TPU mesh")

  if client_id is None or client_id == api.client_id():
    return mesh.local_device_ids()

  # It's not clear we should ever allow a client to query other clients for
  # their device IDs.
  raise NotImplementedError(
      "Looking up other clients' device IDs is not supported")


def get_device_locations(
    mesh: layout_lib.Mesh,
    client_id: Optional[int] = None) -> List[Dict[str, int]]:
  """Returns the device locations of all TPU cores local to the given client.

  A device location is a dictionary from dimension names to indices on those
  dimensions. For example, for a 2x2 mesh ('x', 'y'), this function returns a
  permutation of this list:

    [{'x': 0, 'y': 0},
     {'x': 0, 'y': 1},
     {'x': 1, 'y': 0},
     {'x': 1, 'y': 1}].

  Note that device IDs and device locations are equivalent. The former is a
  linearization of the latter along mesh dimensions.

  Args:
    mesh: A TPU mesh.
    client_id: Optional; A DTensor client ID. If empty, query this client.
  """

  if mesh.device_type() != _TPU_DEVICE_TYPE:
    raise ValueError("The mesh must be a TPU mesh")

  if client_id is None or client_id == api.client_id():
    return mesh.local_device_locations()

  # It's not clear we should ever allow a client to query other clients for
  # their device locations.
  raise NotImplementedError(
      "Looking up other clients' device locations is not supported")


def _configure_tpu_runtime():
  was_enabled = context.is_tfrt_enabled()
  if ("tpu_use_tfrt" in flags.FLAGS and flags.FLAGS["tpu_use_tfrt"].value):
    tfrt_utils.set_tfrt_enabled(True)
  if not was_enabled:
    context._reset_context()  # pylint:disable=protected-access

