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
"""Utilities to help with mesh creation."""

from typing import List, Optional, Tuple
from absl import logging
import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import multi_client_util
from tensorflow.dtensor.python import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


def _print_context(num_global_devices: int, num_clients: int, client_id: int,
                   device_type: str, mesh: layout.Mesh) -> None:
  logging.info('This is client %d of %d clients', client_id, num_clients)
  logging.info('Number of global %s devices: %d', device_type.upper(),
               num_global_devices)
  # pylint: disable=protected-access
  logging.info('Global device IDs: %s', mesh._global_device_ids)
  logging.info('Local device IDs: %s', mesh._local_device_ids)
  logging.info('Local devices: %s',
               [d.to_string() for d in mesh._local_devices])
  # pylint: enable=protected-access


def _make_device_specs(
    devices: Optional[List[str]] = None,
    device_type: Optional[str] = None
) -> Tuple[List[tf_device.DeviceSpec], str]:
  """Makes device specs from local devices names or number of global devices."""
  if devices is None:
    if device_type is None:
      device_type = 'CPU'
    devices = [
        tf_device.DeviceSpec.from_string(d.name)
        for d in tf_config.list_logical_devices(device_type)
    ]
  else:
    devices = [tf_device.DeviceSpec.from_string(d) for d in devices]
    if device_type is None:
      device_type = devices[0].device_type

    if device_type.upper() != devices[0].device_type.upper():
      raise ValueError(
          f'Conflicting devices {str(devices)} and device_type {device_type}')

  return devices, device_type


@tf_export('experimental.dtensor.create_mesh', v1=[])
def create_mesh(mesh_dims: Optional[List[Tuple[str, int]]] = None,
                mesh_name: str = '',
                devices: Optional[List[str]] = None,
                device_type: Optional[str] = None) -> layout.Mesh:
  """Creates a single-client mesh.

  If both `mesh_dims` and `devices` are specified, they must match each otehr.
  As a special case, when all arguments are missing, this creates a 1D CPU mesh
  with an empty name, assigning all available devices to that dimension.

  Args:
    mesh_dims: A list of (dim_name, dim_size) tuples. Defaults to a single
      batch-parallel dimension called 'x' using all devices. As a special case,
      a single-element mesh_dims whose dim_size is -1 also uses all devices.
    mesh_name: Name of the created mesh. Defaults to ''.
    devices: String representations of devices to use. This is the device part
      of tf.DeviceSpec, e.g. 'CPU:0'. Defaults to all available logical devices.
    device_type: If `devices` is missing, the type of devices to use. Defaults
      to 'CPU'.

  Returns:
    A single-client mesh created from specified or default arguments.
  """
  device_specs, device_type = _make_device_specs(devices, device_type)

  local_spec = tf_device.DeviceSpec(job=config.job_name(), replica=0, task=0)
  device_specs = [local_spec.make_merged_spec(d) for d in device_specs]

  if mesh_dims is None:
    mesh_dims = [('x', len(device_specs))]
  elif len(mesh_dims) == 1 and mesh_dims[0][1] == -1:
    # Replace -1 dim_size in a 1D mesh will the number of all devices.
    mesh_dims[0] = (mesh_dims[0][0], len(device_specs))

  dim_names = [d[0] for d in mesh_dims]
  shape = [d[1] for d in mesh_dims]

  if np.prod(shape) != len(device_specs):
    raise ValueError(f'length of devices ({len(device_specs)}) must be '
                     f'equal to total size of the mesh of shape {shape}')

  global_device_ids = np.arange(len(device_specs)).reshape(shape)
  local_device_ids = np.ravel(global_device_ids).tolist()
  mesh = layout.Mesh(
      dim_names=dim_names,
      global_device_ids=global_device_ids,
      local_device_ids=local_device_ids,
      local_devices=device_specs,
      mesh_name=mesh_name)
  _print_context(
      num_global_devices=len(device_specs),
      num_clients=1,
      client_id=0,
      device_type=device_type,
      mesh=mesh)
  return mesh


@tf_export('experimental.dtensor.create_distributed_mesh', v1=[])
def create_distributed_mesh(mesh_dims: List[Tuple[str, int]],
                            mesh_name: str = '',
                            local_devices: Optional[List[str]] = None,
                            device_type: Optional[str] = None) -> layout.Mesh:
  """Creates a distributed mesh.

  This is similar to `create_mesh`, but with a different set of arguments to
  create a mesh that spans evenly across a multi-client DTensor cluster.

  For CPU and GPU meshes, users can choose to use fewer local devices than what
  is available `local_devices`.

  For TPU, only meshes that uses all TPU cores is supported by the DTensor
  runtime.

  Args:
    mesh_dims: A list of (dim_name, dim_size) tuples.
    mesh_name: Name of the created mesh. Defaults to ''.
    local_devices: String representations of devices to use. This is the device
      part of tf.DeviceSpec, e.g. 'CPU:0'. Defaults to all available local
      logical devices.
    device_type: Type of device to build the mesh for. Defaults to 'CPU'.
      Supported values are 'CPU', 'GPU', 'TPU'.

  Returns:
    A mesh that spans evenly across all DTensor clients in the cluster.
  """
  dim_names, shape = zip(*mesh_dims)

  if device_type and device_type.upper() == 'TPU':
    # TODO(b/185940495): Allow multi-mesh and partial on TPU.
    # TPU meshes can only be configured through environment variables that
    # reflect the actual TPU topology. Do not let users specify custom args.
    if local_devices is not None:
      raise ValueError(
          f'Do not specify devices for {device_type.upper()} meshes. '
          f'Using a partial list of devices for {device_type.upper()} '
          f'is not supported.')

  device_specs, device_type = _make_device_specs(local_devices, device_type)

  if device_type.upper() in ['CPU', 'GPU']:
    # For CPU and GPU meshes, user-specified args take precedence over env vars.
    # This is particularly useful on single clients when users want to create
    # meshes that use fewer logical devices than what's available.

    if config.num_clients() > 1 and not multi_client_util.is_initialized():
      raise ValueError('Invalid multi-client topology, please run '
                       'dtensor.initialize_multi_client() first.')

    local_spec = tf_device.DeviceSpec(
        job=config.job_name(), replica=0, task=config.client_id())
    device_specs = [local_spec.make_merged_spec(d) for d in device_specs]

    # Assumes identical number of local devices per client.
    num_global_devices = len(device_specs) * config.num_clients()

    if np.prod(shape) != num_global_devices:
      raise ValueError(
          f'Global number of devices '
          f'({len(device_specs)} per client * {config.num_clients()} clients '
          f'= {num_global_devices}) must be '
          f'equal to total size of the mesh of shape {shape}')

    global_device_ids = np.arange(num_global_devices).reshape(shape)
    flattened = np.ravel(global_device_ids).tolist()
    start_idx = len(device_specs) * config.client_id()
    local_device_ids = flattened[start_idx:start_idx + len(device_specs)]

    mesh = layout.Mesh(
        dim_names=dim_names,
        global_device_ids=global_device_ids,
        local_device_ids=local_device_ids,
        local_devices=device_specs,
        mesh_name=mesh_name)
    _print_context(num_global_devices, config.num_clients(), config.client_id(),
                   device_type, mesh)
    return mesh

  if device_type.upper() == 'TPU':
    mesh = tpu_util.create_tpu_mesh(dim_names, shape, mesh_name)
    _print_context(
        api.num_global_devices(device_type), config.num_clients(),
        config.client_id(), device_type, mesh)
    return mesh

  raise ValueError(f'Device type {device_type} is not CPU, GPU or TPU')


@tf_export('experimental.dtensor.initialize_multi_client', v1=[])
def dtensor_initialize_multi_client(
    enable_coordination_service: Optional[bool] = False) -> None:
  """Initializes Multi Client DTensor.

  The following environment variables controls the behavior of this function.
  If the variables are unset, DTensor will be configured to run in single-client
  mode.

  - DTENSOR_CLIENT_ID: integer, between 0 to num_clients - 1, to identify the
      client id of the current process. The default value is 0.
  - DTENSOR_NUM_CLIENTS: integer, the number of clients. The default value is 1.
  - DTENSOR_JOB_NAME: string, a hostname like string for the name of the dtensor
      job. The default is `localhost` when number of clients is 1, and `worker`
      when the number of clients is greater than 1.
      The job name controls the job name section of the TensorFlow DeviceSpecs,
      e.g., `job:worker` in `/job:worker/replica:0/task:0/device:TPU:0` when
      the job name is `worker`.
  - DTENSOR_JOBS: string, a comma separated list. Each item in the list is
      of format `{hostname}:{port}` and the items must be sorted in alphabet
      order. The implication is the RPC port numbers of the clients from
      the same host must be ordered by the client ID.
      Examples of valid DTENSOR_JOBS values:
      - 4 clients on localhost:
        `localhost:10000,localhost:10001,localhost:10002,localhost:10003`
      - 2 clients on host1, 2 clients on host2
        `host1:10000,host1:10001,host2:10000,host2:10003`

  Args:
    enable_coordination_service: If true, enable distributed coordination
      service to make sure that workers know the devices on each other, a
      prerequisite for data transfer through cross-worker rendezvous.
  """
  assert context.executing_eagerly()

  # Collective GRPC servers are only necessary in multi-client setup.
  # Single clients can use local mode of collectives.
  if config.num_clients() > 1:
    multi_client_util.initialize_multi_client_cluster(
        job_name=config.job_name(),
        dtensor_jobs=config.jobs(),
        client_id=config.client_id(),
        collective_leader=config.full_job_name(task_id=0),
        enable_coordination_service=enable_coordination_service)

  # Make sure the server change is fully propagated before returning.
  context.ensure_initialized()
  context.async_wait()
  context.context()._clear_caches()  # pylint: disable=protected-access

  # Unlike TPU, do not enable heartbeat service.
  # They tend to interfere with regular GPU/CPU collective Ops.


@tf_export('experimental.dtensor.barrier', v1=[])
def barrier(mesh: layout.Mesh, barrier_name: Optional[str] = None):
  """Runs a barrier on the mesh.

  Upon returning from the barrier, all operations run before the barrier
  would have completed across all clients. Currently we allocate a fully
  sharded tensor with mesh shape and run an all_reduce on it.

  Example:

  A barrier can be used before application exit to ensure completion of pending
  ops.

  ```python

  x = [1, 2, 3]
  x = dtensor.relayout(x, dtensor.Layout.batch_sharded(mesh, 'batch', 1))
  dtensor.barrier(mesh)

  # At this point all devices on all clients in the mesh have completed
  # operations before the barrier. Therefore it is OK to tear down the clients.
  sys.exit()
  ```

  Args:
    mesh: The mesh to run the barrier on.
    barrier_name: The name of the barrier. mainly used for logging purpose.
  """
  if barrier_name is None:
    barrier_name = '(barrier)'

  logging.info('entering barrier before op: %s', barrier_name)

  # Make sure all ops are consumed before running the sync.
  context.async_wait()

  # Reduction on a fully sharded tensor requires all devices to participate
  # and serves as a barrier on the mesh.
  component = array_ops.reshape(1.0, [1] * len(mesh.shape()))
  ones = api.pack([component] * mesh.num_local_devices(),
                  layout.Layout(mesh.dim_names, mesh))

  mesh_size = math_ops.reduce_sum(ones)
  if mesh_size != mesh.size:
    raise ValueError(
        'Global barrier produced wrong mesh size : {0} while mesh has actual'
        'size : {1}'.format(mesh_size, mesh.size))

  # TODO(hthu): This isn't strictly needed but might cause confusing behaviors
  # from users. Consider dropping this if there is a `big` performance hit.
  context.async_wait()

  logging.info('finished running barrier across all clients after '
               'op: %s', barrier_name)
