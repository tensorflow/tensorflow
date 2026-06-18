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
"""DTensor Configuration API."""

import os
from typing import List, Optional, Union

from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.util.tf_export import tf_export

_DT_CLIENT_ID = "DTENSOR_CLIENT_ID"
# DTENSOR_NUM_CLIENTS is removed, but some DTensor users still use this symbol.
_DT_NUM_CLIENTS = "DTENSOR_NUM_CLIENTS"
_DT_JOB_NAME = "DTENSOR_JOB_NAME"
_DT_JOBS = "DTENSOR_JOBS"
_DT_HEARTBEAT_ENABLED = "DTENSOR_ENABLE_HEARTBEAT"


# All functions in this file can be used before calling
# `tf.experimental.dtensor.initialize_accelerator_system`.


# -----------------------------------------------------------------------------
# Distributed training-related methods.
#
# Most users should use DTensor utility methods to create a mesh. The methods
# here are only for advanced users who want to fully customize their meshes.
# Note that local_devices and num_local_devices return the actual number of
# locally attached devices. The others are set through environment variables.


@tf_export("experimental.dtensor.local_devices", v1=[])
def local_devices(
    device_type: str, for_client_id: Optional[int] = None
) -> List[tf_device.DeviceSpec]:
  """Returns a list of device specs configured on this client."""
  if device_type.upper() not in ["CPU", "GPU", "TPU"]:
    raise ValueError(f"Device type {device_type} is not CPU, GPU, or TPU.")

  if for_client_id is None:
    for_client_id = client_id()

  # Return fully qualified device specs, sorted by increasing device index.
  return [
      tf_device.DeviceSpec(  # pylint: disable=g-complex-comprehension
          job=job_name(),
          replica=0,  # replica is deprecated and mostly hard-coded now.
          task=for_client_id,
          device_type=device_type,
          device_index=i,
      )
      for i in range(num_local_devices(device_type))
  ]


@tf_export("experimental.dtensor.num_local_devices", v1=[])
def num_local_devices(device_type: str) -> int:
  """Returns the number of devices of device_type configured on this client."""

  # Reads from config because CPU and GPU can use logical devices.
  if device_type.upper() in ["CPU", "GPU"]:
    context_config = context.get_config()
    return context_config.device_count[device_type.upper()]

  return len(tf_config.list_physical_devices(device_type))


@tf_export("experimental.dtensor.num_global_devices", v1=[])
def num_global_devices(device_type: str) -> int:
  """Returns the number of devices of device_type in this DTensor cluster."""
  return num_local_devices(device_type) * num_clients()


@tf_export("experimental.dtensor.client_id", v1=[])
def client_id() -> int:
  """Returns this client's ID."""
  # If missing, assume running with a single client with client_id of 0.
  client_id_value = int(os.environ.get(_DT_CLIENT_ID, "0"))
  if client_id_value < 0:
    raise ValueError(
        f"Environment variable {_DT_CLIENT_ID} "
        f"must be >= 0, got {client_id_value}. "
    )
  if client_id_value >= num_clients():
    raise ValueError(
        f"Environment variable {_DT_CLIENT_ID} "
        f"must be < {num_clients()}, got {client_id_value}"
    )
  return client_id_value


@tf_export("experimental.dtensor.num_clients", v1=[])
def num_clients() -> int:
  """Returns the number of clients in this DTensor cluster."""
  if is_local_mode():
    return 1
  return len(jobs())


@tf_export("experimental.dtensor.job_name", v1=[])
def job_name() -> str:
  """Returns the job name used by all clients in this DTensor cluster."""
  # If missing, assumes the program runs locally and use localhost as job name
  # per TensorFlow convention.
  return os.environ.get(
      _DT_JOB_NAME, "localhost" if num_clients() == 1 else "worker"
  )


@tf_export("experimental.dtensor.full_job_name", v1=[])
def full_job_name(task_id: Optional[int] = None) -> str:
  """Returns the fully qualified TF job name for this or another task."""
  # If task_id is None, use this client's ID, which is equal to its task ID.
  if task_id is None:
    task_id = client_id()
  # In local runs and unit tests, there should be exactly one client running
  # on one TF task.
  if num_clients() == 1 and task_id != 0:
    raise ValueError(f"Unexpected task ID {task_id} in local runs")
  return f"{job_name()}/replica:0/task:{task_id}"


def _bns_task_id(job: str) -> Union[int, str]:
  """Tries to extract an integer task ID from a job name.

  For example, for `job` = '/.../tpu_worker/0:port_name', return 0.

  Args:
    job: A job name to extract task ID from.

  Returns:
    The task ID on success, or the original job name on failure.
  """
  maybe_task_id = job.rsplit("/")[-1].rsplit(":")[0]
  try:
    return int(maybe_task_id)
  except ValueError:
    return job


@tf_export("experimental.dtensor.jobs", v1=[])
def jobs() -> List[str]:
  """Returns a list of job names of all clients in this DTensor cluster."""
  d_jobs = os.environ.get(_DT_JOBS)
  if d_jobs is None:
    return []
  d_jobs_list = d_jobs.split(",")

  # Validate ordering for BNS style job names.
  # For definition of BNS, refer to https://research.google/pubs/pub43438/.
  if any([name.startswith("/bns/") for name in d_jobs_list]):
    if d_jobs_list != sorted(d_jobs_list, key=_bns_task_id):
      raise ValueError(
          f"Unexpected DTENSOR_JOBS content {d_jobs}. Sort entries "
          "in DTENSOR_JOBS because cluster construction relies on "
          "the order."
      )

  return d_jobs_list


@tf_export("experimental.dtensor.heartbeat_enabled", v1=[])
def heartbeat_enabled() -> bool:
  """Returns true if DTensor heartbeat service is enabled."""
  return os.environ.get(_DT_HEARTBEAT_ENABLED, "true").lower() in ("true", "1")


def is_local_mode() -> bool:
  """Returns true if DTensor shall run in local mode."""
  return not jobs()


def is_tpu_present() -> bool:
  """Returns true if TPU devices are present."""
  # Check if TPU is present from initialized context.
  # TPU_SYSTEM is a device that indicates TPUs are present.
  tpu_system_devices = tf_config.list_physical_devices("TPU_SYSTEM")
  return bool(tpu_system_devices)


def is_gpu_present() -> bool:
  """Returns true if TPU devices are present."""
  return bool(tf_config.list_physical_devices("GPU"))


@tf_export("experimental.dtensor.preferred_device_type", v1=[])
def preferred_device_type() -> str:
  """Returns the preferred device type for the accelerators.

  The returned device type is determined by checking the first present device
  type from all supported device types in the order of 'TPU', 'GPU', 'CPU'.
  """
  if is_tpu_present():
    return "TPU"
  elif is_gpu_present():
    return "GPU"

  return "CPU"


def use_multi_device_mode() -> bool:
  """Return True if environment indicates multi-device mode is enabled."""
  return os.environ.get("DTENSOR_ENABLE_MULTI_DEVICE_EXPANSION", "0") != "0"


def gpu_use_nccl_communication() -> bool:
  """Return True if environment indicates NCCL shall be used for GPU."""
  return os.environ.get("DTENSOR_GPU_USE_NCCL_COMMUNICATION", "0") != "0"
