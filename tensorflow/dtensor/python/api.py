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
"""Core DTensor Python API."""

import contextlib
import os
import threading
from typing import Any, Callable, List, Optional, Sequence, Union

from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import ops
from tensorflow.python.util.tf_export import tf_export


_DT_CLIENT_ID = "DTENSOR_CLIENT_ID"
_DT_NUM_CLIENTS = "DTENSOR_NUM_CLIENTS"
_DT_JOB_NAME = "DTENSOR_JOB_NAME"
_DT_JOBS = "DTENSOR_JOBS"
_DT_CPU_COUNT = "DTENSOR_CPU_CORE_COUNT"
_DT_GPU_COUNT = "DTENSOR_GPU_CORE_COUNT"
_DT_TPU_COUNT = "DTENSOR_TPU_CORE_COUNT"
_DT_HEARTBEAT_ENABLED = "DTENSOR_ENABLE_HEARTBEAT"

_dtensor_singleton = None
_dtensor_singleton_lock = threading.Lock()

# -----------------------------------------------------------------------------
# Main methods to launch DTensor computations.


@tf_export("experimental.dtensor.call_with_layout", v1=[])
def call_with_layout(fn: Callable[...,
                                  Any], layout: Optional[layout_lib.Layout],
                     *args, **kwargs) -> Any:
  """Calls a function in the DTensor device scope if `layout` is not None.

  If `layout` is not None, `fn` consumes DTensor(s) as input and produces a
  DTensor as output; a DTensor is a tf.Tensor with layout-related attributes.

  If `layout` is None, `fn` consumes and produces regular tf.Tensors.

  Args:
    fn: A supported TF API function such as tf.zeros.
    layout: Optional, the layout of the output DTensor.
    *args:  Arguments given to `fn`.
    **kwargs: Keyword arguments given to `fn`.

  Returns:
    The return value of `fn` transformed to a DTensor if requested.
  """
  if layout is not None:
    if not context.executing_eagerly():
      # This is a workaround for b/199324097, where functions such as tf.ones
      # could attach an incorrect layout to the tf.const generated under the
      # hood. The op runs successfully in eager mode, but in graph mode, MLIR
      # passes sometimes attach the default layout to a scalar constant.
      # %cst = tf.Const([1])  -- With the given layout
      # %0 = "tf.DTensorLayout"(%cst). -- Fails in MLIR pass since shape for
      #                                -- layout could be different than
      #                                -- shape[0] for %cst.
      # %1 = tf.Fill(%0, 1)
      result = fn(*args, **kwargs)
      return relayout(result, layout)
    else:
      with run_on(layout.mesh):
        with _dtensor_device()._default_layout(layout):  # pylint: disable=protected-access
          return fn(*args, **kwargs)
  return fn(*args, **kwargs)


@tf_export("experimental.dtensor.run_on", v1=[])
@contextlib.contextmanager
def run_on(layout_or_mesh: Union[layout_lib.Layout, layout_lib.Mesh]):
  """Runs enclosed functions in the DTensor device scope.

  This function returns a scope. All the ops and tf.functions in this scope will
  run on the DTensor device using the mesh provided or attached to the layout.
  This is useful for wrapping any tf.function that doesn't take a DTensor as
  input but would like to produce DTensor as result. The scope will also make
  sure all small constants be replicated as DTensor.

  Args:
    layout_or_mesh: A Layout or Mesh instance to extract a default mesh from.

  Yields:
    A context in which all ops and tf.functions will run on the DTensor device.
  """
  if isinstance(layout_or_mesh, layout_lib.Layout):
    mesh = layout_or_mesh.mesh
  elif isinstance(layout_or_mesh, layout_lib.Mesh):
    mesh = layout_or_mesh
  else:
    raise ValueError("Expect `layout_or_mesh` to be either `Layout` or `Mesh`, "
                     f"got {type(layout_or_mesh)}")

  with _dtensor_device()._experimental_default_mesh(mesh):  # pylint: disable=protected-access
    with ops.device(device_name()):
      yield


@tf_export("experimental.dtensor.device_name", v1=[])
def device_name() -> str:
  """Returns the singleton DTensor device's name.

  This function can be used in the following way:

  ```python
  import tensorflow as tf

  with tf.device(dtensor.device_name()):
    # ...
  ```
  """
  return _dtensor_device().name


# -----------------------------------------------------------------------------
# Data transfer methods.


@tf_export("experimental.dtensor.copy_to_mesh", v1=[])
def copy_to_mesh(
    tensor: Any,
    layout: layout_lib.Layout,
    source_layout: Optional[layout_lib.Layout] = None) -> ops.Tensor:
  """Copies a tf.Tensor onto the DTensor device with the given layout.

  Copies a regular tf.Tensor onto the DTensor device. Use the mesh attached to
  `layout` as target mesh. This method currently only supports replicated
  layouts. To get a DTensor with a sharded layout, use the `pack` method.

  Args:
    tensor: A regular tf.Tensor to be copied as a DTensor.
    layout: Target layout (and mesh) for the result DTensor.
    source_layout: Source layout of the tensor before copy, used for backward
      passes.

  Returns:
    A DTensor on the DTensor device with the given layout.
  """
  return _dtensor_device().copy_to_mesh(tensor, layout, source_layout)


@tf_export("experimental.dtensor.pack", v1=[])
def pack(tensors: Sequence[Any], layout: layout_lib.Layout) -> Any:
  """Packs tf.Tensor components into a DTensor."""
  return _dtensor_device().pack(tensors, layout)


@tf_export("experimental.dtensor.unpack", v1=[])
def unpack(tensor: Any) -> Sequence[Any]:
  """Unpacks a DTensor into tf.Tensor components."""
  return _dtensor_device().unpack(tensor)


# -----------------------------------------------------------------------------
# Layout-related methods.


@tf_export("experimental.dtensor.fetch_layout", v1=[])
def fetch_layout(tensor: ops.Tensor) -> layout_lib.Layout:
  """Returns the layout of a DTensor."""
  return _dtensor_device().fetch_layout(tensor)


@tf_export("experimental.dtensor.check_layout", v1=[])
def check_layout(tensor: ops.Tensor, layout: layout_lib.Layout) -> None:
  """Asserts that the layout of `tensor` is `layout`."""
  if fetch_layout(tensor) != layout:
    raise ValueError("Layout of tensor: " + str(fetch_layout(tensor)) +
                     ", did not match expected layout: " + str(layout))


@tf_export("experimental.dtensor.relayout", v1=[])
def relayout(tensor: ops.Tensor, layout: layout_lib.Layout) -> ops.Tensor:
  """Changes the layout of `tensor`.

  Changes the layout of `tensor` to `layout`. This is used to fine-tune the
  behavior of ops following/connected to `tensor`, such as choosing one SPMD
  expansion pattern over another. This works by forward propagating `layout`
  to connected TensorFlow computation graphs during layout propagation.

  Currently, only converting layouts from replicated to sharded or sharded to
  replicated per mesh dimension is supported. That is, "x, y" -> "unsharded, y"
  is supported, while "x, y" -> "z, y" is not supported.

  We also support a special "match" sharding spec, which instructs the relayout
  to act as an identity operation with respect to any sharding on these
  mesh dimensions.

  Relayout is internally lowered to a set of Split and/or AllToAll ops. When
  tensor layouts are converted from replicated to sharded, the cost is
  comparatively low because we only insert Split ops and no cross-device
  communication is needed. However, when tensor layouts are converted from
  sharded to replicated, cross-device communication may occur, causing potential
  performance impact.

  Args:
    tensor: A DTensor to specify a new layout for.
    layout: A Layout object specifying a new sharding spec.

  Returns:
    A DTensor output from the Relayout op.
  """
  layout_str = layout.to_string()
  return gen_dtensor_ops.relayout(tensor, layout_str)


# -----------------------------------------------------------------------------
# Distributed training-related methods.
#
# Most users should use DTensor utility methods to create a mesh. The methods
# here are only for advanced users who want to fully customize their meshes.
# Note that local_devices and num_local_devices return the actual number of
# locally attached devices. The others are set through environment variables.


@tf_export("experimental.dtensor.client_id", v1=[])
def client_id() -> int:
  """Returns this client's ID."""
  # If missing, likely in unit tests and local runs, 0 is a good default.
  return int(os.environ.get(_DT_CLIENT_ID, "0"))


@tf_export("experimental.dtensor.num_clients", v1=[])
def num_clients() -> int:
  """Returns the number of clients in this DTensor cluster."""
  # If missing, likely in unit tests and local runs, 1 is a good default.
  return int(os.environ.get(_DT_NUM_CLIENTS, "1"))


@tf_export("experimental.dtensor.local_devices", v1=[])
def local_devices(
    device_type: str,
    for_client_id: Optional[int] = None) -> List[tf_device.DeviceSpec]:
  """Returns a list of device specs of device_type attached to this client."""
  if device_type.upper() not in ["CPU", "GPU", "TPU"]:
    raise ValueError(f"Device type {device_type} is not CPU, GPU, or TPU.")
  if for_client_id is None:
    for_client_id = client_id()

  logical_devices = [
      tf_device.DeviceSpec.from_string(d.name)
      for d in tf_config.list_logical_devices(device_type)
  ]

  # Get the number of local devices.
  device_count = 0
  for d in logical_devices:
    # d might have a partial name, e.g. /device:TPU:0.
    if (d.job is None or d.job == job_name()) and (d.task is None or
                                                   d.task == for_client_id):
      device_count = device_count + 1

  # Return fully qualified device specs, sorted by increasing device index.
  return [
      tf_device.DeviceSpec(  # pylint: disable=g-complex-comprehension
          job=job_name(),
          replica=0,  # replica is deprecated and mostly hard-coded now.
          task=for_client_id,
          device_type=device_type,
          device_index=i) for i in range(device_count)
  ]


@tf_export("experimental.dtensor.num_local_devices", v1=[])
def num_local_devices(device_type: str) -> int:
  """Returns the number of devices of device_type attached to this client."""
  return len(local_devices(device_type))


@tf_export("experimental.dtensor.num_global_devices", v1=[])
def num_global_devices(device_type: str) -> int:
  """Returns the number of devices of device_type in this DTensor cluster."""
  num_devices = 0

  if device_type.upper() == "CPU":
    num_devices = int(os.environ.get(_DT_CPU_COUNT, "0"))
  elif device_type.upper() == "GPU":
    num_devices = int(os.environ.get(_DT_GPU_COUNT, "0"))
  elif device_type.upper() == "TPU":
    num_devices = int(os.environ.get(_DT_TPU_COUNT, "0"))
  else:
    raise ValueError(f"Device type {device_type} is not CPU, GPU, or TPU.")

  if num_devices > 0:
    return num_devices

  # If missing, likely in unit tests and local runs, assume there is only one
  # client and all global devices are local.
  return num_local_devices(device_type)


@tf_export("experimental.dtensor.job_name", v1=[])
def job_name() -> str:
  """Returns the job name used by all clients in this DTensor cluster."""
  # If missing, the program is likely running locally or in a unit test.
  return os.environ.get(_DT_JOB_NAME, "localhost")


@tf_export("experimental.dtensor.full_job_name", v1=[])
def full_job_name(task_id: Optional[int] = None) -> str:
  """Returns the fully qualified TF job name for this or another task."""
  # If task_id is None, use this client's ID, which is equal to its task ID.
  if task_id is None:
    task_id = client_id()
  # In local runs and unit tests, there should be exactly one client running
  # on one TF task.
  if job_name() == "localhost" and task_id != 0:
    raise ValueError(f"Unexpected task ID {task_id} in local runs")
  return f"{job_name()}/replica:0/task:{task_id}"


def _task_id(job: str) -> Union[int, str]:
  """Tries to extract an integer task ID from a Borg job name.

  For example, for `job` = '/bns/.../tpu_worker/0:port_name', return 0.

  Args:
    job: A BNS job name to extract task ID from.

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
  """Returns a list of Borg job names of all clients in this DTensor cluster."""
  d_jobs = os.environ.get(_DT_JOBS)
  if d_jobs is None:
    return []
  d_jobs_list = d_jobs.split(",")
  if d_jobs_list != sorted(d_jobs_list, key=_task_id):
    raise ValueError(f"Unexpected DTENSOR_JOBS content {d_jobs}. Sort entries "
                     "in DTENSOR_JOBS because cluster construction relies on "
                     "the order.")
  return d_jobs_list


@tf_export("experimental.dtensor.heartbeat_enabled", v1=[])
def heartbeat_enabled() -> bool:
  """Returns true if DTensor heartbeat service is enabled."""
  return os.environ.get(_DT_HEARTBEAT_ENABLED, "true").lower() in ("true", "1")


# -----------------------------------------------------------------------------
# Private methods.


def _dtensor_device() -> dtensor_device.DTensorDevice:
  global _dtensor_singleton
  with _dtensor_singleton_lock:
    if _dtensor_singleton is None:
      _dtensor_singleton = dtensor_device.DTensorDevice(meshes=[])
  return _dtensor_singleton


def _reset() -> None:
  global _dtensor_singleton
  if _dtensor_singleton is not None:
    _dtensor_singleton.clear_tpu_core_ids()
  with _dtensor_singleton_lock:
    _dtensor_singleton = None
