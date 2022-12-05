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
"""Keras utilities for DTensor unit test."""
# TODO(feyu): Consolidate this file after moving the dtensor test base
# from learning/brain to tensorflow/dtensor.

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import api as dtensor_api
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import device_spec
from tensorflow.python.platform import test

_DEFAULT_GPU_MEMORY_LIMIT = 200  # MB


class DTensorBaseTest(test.TestCase, parameterized.TestCase):
  """Provides comparison helper for dtensor vs local results."""

  def tearDown(self):
    super().tearDown()
    # Make sure all async ops finish.
    context.async_wait()

    # TODO(hthu): Remove the reset once we fixed the CopyToMesh with
    # DefaultMesh placement issue.
    reset_dtensor()

  @classmethod
  def configTestMesh(cls, device_type_mesh_map):
    """Configs corresponding mesh given test context.

    If runs on a CPU mesh, set virtual device on CPU.
    If runs on a GPU mesh, sets virtual device on GPU with proper memory
    limits.
    if runs on a TPU mesh, initializes TPU system.

    Args:
      device_type_mesh_map: A dictionary containing device_type -> mesh mapping.

    Returns:
      A properly configured mesh for use in test.
    """
    reset_context()

    def get_mesh(device_type):
      mesh = device_type_mesh_map.get(device_type, None)
      if mesh is None:
        dt = device_type
        raise ValueError(f"Requires a {dt} mesh to run test on {dt}.")
      return mesh

    if config.list_physical_devices("GPU"):
      mesh = get_mesh("GPU")
      reset_logical_devices("GPU", np.prod(mesh.shape()))
    else:
      mesh = get_mesh("CPU")
      reset_logical_devices("CPU", np.prod(mesh.shape()))

    context.ensure_initialized()
    return mesh


def create_device_array(shape, device_type):
  device_count = np.prod(shape)
  return np.asarray([
      device_spec.DeviceSpecV2(  # pylint: disable=g-complex-comprehension
          job="localhost/replica:0/task:0",
          device_type=device_type,
          device_index=i,
      ) for i in range(device_count)
  ]).reshape(shape)


def create_device_list(shape, device_type):
  devices = create_device_array(shape, device_type)
  return np.ravel(devices).tolist()


def create_device_ids_array(shape):
  device_count = np.prod(shape)
  return np.arange(device_count).reshape(shape)


def reset_context():
  context._reset_context()  # pylint: disable=protected-access


def reset_logical_devices(device_type, count):
  """Resets logical devices for CPU/GPU.

  Logical devices can only be instantiated once on a particular context. For
  now, context re-use is triggering some function duplication errors, so we
  reset the context on each call.

  Args:
    device_type: The device_type to reset.
    count: numbers of virtual device to reset to.
  """
  reset_context()
  devices = config.list_physical_devices(device_type)
  if device_type.upper() == "CPU":
    config.set_logical_device_configuration(
        devices[0],
        [
            context.LogicalDeviceConfiguration(),
        ] * count,
    )
  elif device_type.upper() == "GPU":
    config.set_logical_device_configuration(
        devices[0],
        [
            context.LogicalDeviceConfiguration(
                memory_limit=_DEFAULT_GPU_MEMORY_LIMIT),
        ] * count,
    )
  else:
    dt = device_type
    raise ValueError(
        f"resetting logical device for non-supported device type: {dt}")


def reset_dtensor():
  dtensor_api._reset()  # pylint: disable=protected-access
