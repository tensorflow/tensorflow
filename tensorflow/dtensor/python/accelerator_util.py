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
"""Utility for working with accelerator systems."""

from typing import List, Optional

from absl import logging

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import tpu_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.platform import remote_utils
from tensorflow.python.util.tf_export import tf_export

_INITIALIZED_ACCELERATOR_SYSTEM_TYPE = None


def is_initialized() -> bool:
  """Returns whether accelerator system has been initialized."""
  return bool(_INITIALIZED_ACCELERATOR_SYSTEM_TYPE)


def set_initialized(value):
  """Sets if accelerator system has been initialized."""
  global _INITIALIZED_ACCELERATOR_SYSTEM_TYPE
  _INITIALIZED_ACCELERATOR_SYSTEM_TYPE = value


def initialize_multi_client_cluster(job_name: str,
                                    dtensor_jobs: List[str],
                                    client_id: int,
                                    collective_leader: str,
                                    port: Optional[int] = None,
                                    gpu_use_nccl_communication: bool = False,
                                    enable_coordination_service: bool = True):
  """Initialize GRPC servers and collectives for multi-client DTensor setup.

  This function can be used to initialize a multi-client cluster and enable
  collective ops. GRPC servers are necessary in the multi-client mode, even
  when the number of clientis is 1.

  NOTE: this function must be called in an eager context.

  Args:
    job_name: The job name used by all clients in the DTensor cluster.
    dtensor_jobs: A list of the DTensor client jobs participating in the
      cluster. Must be strings of the form "hostname:port".
    client_id: The ID of the DTensor client this function is being called in.
    collective_leader: The job/task that will be used to run collectives.
    port: The port this client's GRPC server will run on. If omitted, use the
      port from dtensor_jobs for this client.
    gpu_use_nccl_communication: if True, configure TensorFlow to use NCCL by
      default.
    enable_coordination_service: If true, enable distributed coordination
      service to make sure that workers know the devices on each other, a
      prerequisite for data transfer through cross-worker rendezvous.

  Raises:
    RuntimeError: If running inside a tf.function.
  """
  assert context.executing_eagerly()

  if not collective_leader.startswith("/job:"):
    collective_leader = "/job:" + collective_leader

  context.context().configure_collective_ops(
      use_nccl_communication=gpu_use_nccl_communication,
      collective_leader=collective_leader)
  if enable_coordination_service:
    context.context().configure_coordination_service(
        service_type="standalone", service_leader=collective_leader)

  config_proto = context.get_config()

  # Construct server def from the host directly instead of relying on
  # TF_CONFIG.
  cluster_def = cluster_pb2.ClusterDef()
  # Note that for bns addresses, we will currently rely on the sorted string
  # of job name as the order of assigning task ids. This might be brittle once
  # we have jobs across multiple cells.
  cluster_def.job.add(name=job_name, tasks=dict(enumerate(dtensor_jobs)))
  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_def,
      default_session_config=config_proto,
      job_name=job_name,
      task_index=client_id,
      protocol=remote_utils.get_default_communication_protocol(),
      port=port)
  server_def.default_session_config.rpc_options.num_channels_per_target = 4
  server_def.default_session_config.experimental.recv_buf_max_chunk = -1

  logging.info("Enabling collectives with server_def: %s", server_def)

  context.context().enable_collective_ops(server_def)

  context.ensure_initialized()


@tf_export(
    "experimental.dtensor.initialize_accelerator_system",
    "experimental.dtensor.initialize_tpu_system",
    "experimental.dtensor.initialize_multi_client",
    v1=[])
def initialize_accelerator_system(
    device_type: Optional[str] = None,
    enable_coordination_service: Optional[bool] = True,
    experimental_reset_context: Optional[bool] = False,
) -> str:
  """Initializes accelerators and communication fabrics for DTensor.

  DTensor configures TensorFlow to run in the local mode or multi-client mode.
  - In local mode, a mesh can only use devices attached to the current process.
  - In multi-client mode, a mesh can span across devices from multiple clients.

  If `DTENSOR_JOBS` is non-empty, DTensor configures TensorFlow to run in the
  multi-client mode using the distributed runtime. In multi-client mode devices
  on different clients can communicate with each other.

  The following environment variables controls the behavior of this function.

  - `DTENSOR_JOBS`: string, a comma separated list. Each item in the list is
      of format `{hostname}:{port}`. If empty, DTensor runs in the local mode.
      Examples of valid `DTENSOR_JOBS` values:
      - 4 clients on localhost:
        `localhost:10000,localhost:10001,localhost:10002,localhost:10003`
      - 2 clients on host1, 2 clients on host2
        `host1:10000,host1:10001,host2:10000,host2:10003`
      If the hostnames are BNS addresses, the items must be sorted in
      alphabetical order.
  - `DTENSOR_CLIENT_ID`: integer, between `0` to `num_clients - 1`, to identify
      the client id of the current process. The default value is `0`.
  - `DTENSOR_JOB_NAME`: string, a string for the name of the TensorFlow job.
      The job name controls the job name section of the TensorFlow DeviceSpecs,
      e.g., `job:worker` in `/job:worker/replica:0/task:0/device:TPU:0` when
      the job name is `worker`.
      The default value is `localhost` in local mode, and
      `worker` when in the multi-client mode. All DTensor clients within the
      same multi-client cluster share the same job name.
  - `DTENSOR_USE_PARALLEL_EXECUTOR`: string, with its value being `pw` to
      specify that the backend is Pathways, and TensorFlow otherwise.

  Args:
    device_type: Type of accelerator to use, can be CPU, GPU, or TPU. If None,
      uses `tf.experimental.dtensor.preferred_device_type()`.
    enable_coordination_service: If true, enable distributed coordination
      service to make sure that workers know the devices on each other, when
      there is more than 1 client.
    experimental_reset_context: Reset the tensorflow context. Behaviors of
      existing TensorFlow objects (e.g. Tensors) are undefined. Set this to True
      as an escape hatch, if there is no clear way to refactor your code to call
      initialize_accelerator_system() before calling TensorFlow APIs that
      initialize the context.

  Returns:
    device_type: the type of accelerator that was initialized.
  """
  global _INITIALIZED_ACCELERATOR_SYSTEM_TYPE
  assert context.executing_eagerly()

  if is_initialized():
    raise ValueError(
        "Accelerator system has already been initialized. "
        "Call tf.experimental.dtensor.shutdown_accelerator_system() first.")

  if experimental_reset_context:
    logging.warn(
        "experimental_reset_context is True. "
        "Resetting TensorFlow context. Existing TensorFlow objects "
        "(e.g. Tensors and resources) are invalidated."
    )
    context.context().ensure_uninitialized()  # pylint: disable=protected-access

  if context.context()._initialized:  # pylint: disable=protected-access
    raise ValueError(
        "TensorFlow has already been initialized. "
        "tf.experimental.dtensor.initialize_accelerator_system() must be "
        "called before TensorFlow is initialized.")

  context.context()._clear_caches()  # pylint: disable=protected-access

  if device_type is None:
    device_type = config.preferred_device_type()

  device_type = device_type.upper()
  if device_type not in {"CPU", "GPU", "TPU"}:
    raise ValueError(f"Unknown device_type {device_type}. "
                     "Allowed values are CPU, GPU, or TPU")

  if config.gpu_use_nccl_communication():
    logical_gpu_count = config.num_local_devices("GPU")
    physical_gpu_count = len(tf_config.list_physical_devices("GPU"))
    if logical_gpu_count > physical_gpu_count:
      raise ValueError(
          "DTENSOR_GPU_USE_NCCL_COMMUNICATION is set for using NCCL. "
          "NCCL Collectives require one to one mapping between logical and "
          "physical GPUs. "
          f"The number of logical GPU ({logical_gpu_count}) "
          f"is more than the number of physical GPU ({physical_gpu_count})."
      )

  # Configure logical host CPU devices for accelerators.
  if device_type in ("GPU", "TPU"):
    num_local_devices = config.num_local_devices(device_type)
    if config.num_local_devices("CPU") < num_local_devices:
      tf_config.set_logical_device_configuration(
          tf_config.list_physical_devices("CPU")[0],
          [context.LogicalDeviceConfiguration()] * num_local_devices)

  if not config.is_local_mode():
    initialize_multi_client_cluster(
        job_name=config.job_name(),
        dtensor_jobs=config.jobs(),
        client_id=config.client_id(),
        collective_leader=config.full_job_name(task_id=0),
        gpu_use_nccl_communication=config.gpu_use_nccl_communication(),
        enable_coordination_service=enable_coordination_service)
  else:
    if device_type == "GPU":
      # Enables Nccl on local mode.
      context.context(  # pylint: disable=protected-access
      )._collective_use_nccl_communication = config.gpu_use_nccl_communication(
      )

  if device_type == "TPU" and not config.backend_is_pw():
    tpu_util.initialize_tpu_system()

  _INITIALIZED_ACCELERATOR_SYSTEM_TYPE = device_type

  return device_type


@tf_export(
    "experimental.dtensor.shutdown_accelerator_system",
    "experimental.dtensor.shutdown_tpu_system",
    v1=[])
def shutdown_accelerator_system() -> None:
  """Shuts down the accelerator system."""
  global _INITIALIZED_ACCELERATOR_SYSTEM_TYPE
  context.async_wait()

  if not is_initialized():
    raise ValueError(
        "Accelerator system is not initialized. Call "
        "tf.experimental.dtensor.initialize_accelerator_system first.")

  device_type = _INITIALIZED_ACCELERATOR_SYSTEM_TYPE

  if not config.is_local_mode():
    raise ValueError(
        "Shutting down accelerator system under multi-client mode is "
        "not supported.")

  if device_type == "TPU" and not config.backend_is_pw():
    tpu_util.shutdown_tpu_system()

  # reset TF context to stop gRPC servers.
  context._reset_context()  # pylint: disable=protected-access
  context.context()._clear_caches()  # pylint: disable=protected-access
  _INITIALIZED_ACCELERATOR_SYSTEM_TYPE = None
