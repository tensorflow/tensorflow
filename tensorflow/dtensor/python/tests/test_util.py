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
"""Utility methods for DTensor testing."""
import collections
import copy
import itertools
import json
import os
import typing

from absl import flags
from absl.testing import parameterized

import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.config import is_gpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import is_tpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import preferred_device_type  # pylint: disable=unused-import
from tensorflow.dtensor.python.tests.test_backend_name import DTensorTestUtilBackend
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test as tf_test
# pylint: enable=g-direct-tensorflow-import

DEFAULT_TOL = 1e-5

_DEFAULT_GPU_MEMORY_LIMIT = 200  # MB


DTENSOR_TEST_UTIL_BACKEND = DTensorTestUtilBackend(
    os.getenv('DTENSOR_TEST_UTIL_BACKEND', default='unspecified'))


def create_device_ids_array(shape):
  device_count = np.prod(shape)
  return np.arange(device_count).reshape(shape)


def create_device_array(shape, device_type):
  device_count = np.prod(shape)
  return np.asarray([
      tf_device.DeviceSpec(  # pylint: disable=g-complex-comprehension
          job='localhost/replica:0/task:0',
          device_type=device_type,
          device_index=i) for i in range(device_count)
  ]).reshape(shape)


def create_device_list(shape, device_type):
  devices = create_device_array(shape, device_type)
  return np.ravel(devices).tolist()


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
  devices = tf_config.list_physical_devices(device_type)
  if device_type.upper() not in ('CPU', 'GPU'):
    raise ValueError('resetting logical device for non-supported device type : '
                     '%s' % device_type)

  if count < len(devices):
    raise ValueError(f'Cannot set {count} logical devices, which is '
                     f'less than ({len(devices)}) physical devices.')

  for i, device in enumerate(devices):
    n = (i + 1) * count // len(devices) - i * count // len(devices)
    assert n > 0  # guaranteed if count >= len(devices)
    configs = []
    for ordinal in range(n):
      if device_type.upper() == 'GPU':
        dev_config = context.LogicalDeviceConfiguration(
            memory_limit=_DEFAULT_GPU_MEMORY_LIMIT,
            experimental_device_ordinal=ordinal)
      else:
        dev_config = context.LogicalDeviceConfiguration()
      configs.append(dev_config)

    tf_config.set_logical_device_configuration(device, configs)


def list_local_logical_devices(device_type):
  """Returns a list of local logial devices."""

  # When coordinator service is enabled, list_logical_devices returns
  # a global list.
  devices = tf_config.list_logical_devices(device_type)

  def is_local(device):
    spec = tf_device.DeviceSpec.from_string(device.name)
    if spec.job is None or spec.job == 'localhost':
      return True
    elif spec.job == config.job_name() and spec.task == config.client_id():
      return True
    return False

  return [d for d in devices if is_local(d)]


def is_tfrt_enabled():
  return context.is_tfrt_enabled()


FLAGS = flags.FLAGS


class DTensorBaseTest(tf_test.TestCase, parameterized.TestCase):
  """Provides comparison helper for dtensor vs local results."""

  @classmethod
  def setUpClass(cls):
    super(DTensorBaseTest, cls).setUpClass()

  def tearDown(self):
    super().tearDown()
    # Make sure all async ops finish.
    context.async_wait()

    self.maybeShutdownTpuSystem()
    # TODO(hthu): Remove the reset once we fixed the CopyToMesh with
    # DefaultMesh placement issue.
    reset_dtensor()

  @staticmethod
  def configTestMesh(  # pylint: disable=invalid-name
      device_type_mesh_map: typing.Dict[typing.Text,
                                        layout_lib.Mesh]) -> layout_lib.Mesh:
    """Configs corresponding mesh given test context.

    If runs on a CPU mesh, set virtual device on CPU.
    If runs on a GPU mesh, sets virtual device on GPU with proper memory limits.
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
        raise ValueError('Requires a %s mesh to run test on %s.' %
                         (device_type, device_type))
      return mesh

    mesh = None
    if is_tpu_present():
      mesh = get_mesh('TPU')
      reset_context()
      accelerator_util.initialize_accelerator_system('TPU')
    elif tf_config.list_physical_devices('GPU'):
      mesh = get_mesh('GPU')
      reset_logical_devices('GPU', np.prod(mesh.shape()))
      accelerator_util.initialize_accelerator_system('GPU')
    else:
      mesh = get_mesh('CPU')
      reset_logical_devices('CPU', np.prod(mesh.shape()))
      accelerator_util.initialize_accelerator_system('CPU')

    return mesh

  @staticmethod
  def maybeShutdownTpuSystem():  # pylint: disable=invalid-name
    """Shuts down the TPU System if present.

    This is usually called at the unit test tear down phase to reset the TPU
    system before running the next test.
    """
    # Only need to explicitly shuts down TPU system in TFRT since in current
    # runtime, the shutdown is done in initialization process.
    if accelerator_util.is_initialized():
      accelerator_util.shutdown_accelerator_system()

  def skipForDeviceType(  # pylint: disable=invalid-name
      self,
      device_type: typing.List[str],
      reason: str,
      unless_device_count_equals_to=None):
    """Skip the test for the specific device_type.

    Args:
      device_type: list of device types, one of "CPU", "GPU", or "TPU".
      reason: string that describe the reason for skipping the test.
      unless_device_count_equals_to: Optional int. This parameter only works if
        device_type is "TPU". If set, the test will be skipped unless the number
        of TPUs equals to the specified count.
    """
    physical_device_types = set(
        [d.device_type for d in tf_config.list_physical_devices()])
    for device in device_type:
      if device == 'TPU' and is_tpu_present():
        if unless_device_count_equals_to is None:
          self.skipTest(reason)
        elif len(list_local_logical_devices(
            device)) != unless_device_count_equals_to:
          self.skipTest(reason)
      if device == 'CPU' and len(
          physical_device_types) == 1 and 'CPU' in physical_device_types:
        # Make sure we skip when only `CPU` is present.
        self.skipTest(reason)
      if device == 'GPU' and 'GPU' in physical_device_types:
        self.skipTest(reason)

  def skipForTfrt(self, reason: str):  # pylint: disable=invalid-name
    if is_tfrt_enabled():
      self.skipTest(reason)

  def skipTest(self, reason):  # pylint: disable=invalid-name
    self.maybeShutdownTpuSystem()
    super().skipTest(reason)

  def assertDTensorEqual(
      self,  # pylint: disable=invalid-name
      expected_result,
      expected_layout,
      result_dtensor,
      tol=DEFAULT_TOL):
    """Asserts DTensor is of the particular value."""
    if issubclass(
        type(result_dtensor), resource_variable_ops.BaseResourceVariable):
      result_dtensor = result_dtensor.value()
    if expected_layout is not None:
      # This, the assertEqual, is a pure proto raw bytes comparison. To make it
      # human-readable, use the `to_string` api for Layout for the dedicated msg
      # field.
      #
      # Futhurmore, as the mesh part is very long and usually identical. Try to
      # cut them as well, to make it easier to read.
      expected_str = expected_layout.to_string()
      got_str = api.fetch_layout(result_dtensor).to_string()
      index_for_mesh = expected_str.find('mesh:')
      if index_for_mesh != -1 and got_str.find(
          expected_str[index_for_mesh:]) != -1:
        # the mesh part is same. cut them so it is more readable.
        expected_str = expected_str[:index_for_mesh]
        got_str = got_str[:got_str.find('mesh:')]

      self.assertEqual(
          api.fetch_layout(result_dtensor),
          expected_layout,
          msg='========\nexpected layout is\n  {}\n\nwhile got layout is\n  {}\n'
          .format(expected_str, got_str))

    layout = api.fetch_layout(result_dtensor)
    unpacked = [t.numpy() for t in api.unpack(result_dtensor)]

    # Check dtype.
    self.assertEqual(expected_result.dtype, result_dtensor.dtype,
                     result_dtensor)
    # Check global shape.
    self.assertAllEqual(expected_result.shape, result_dtensor.shape)

    result_dtensor = np.block(layout.unravel(unpacked).tolist())

    # Check value on concatenated result DTensor.
    self.assertAllClose(expected_result, result_dtensor, atol=tol, rtol=tol)

    # In addition to check the 'concatenated' DTensor, we also check all
    # "replicated" parts are same.
    #
    # The algorithm is simple:
    # 1. For a mesh with topology (x,y,z,p), and a DTensor with layout ('',z,x).
    # 2. Create some data structures:
    #   - create a mapping from device id (called offset below) to mesh
    #     location. For the mesh above, loc {x:1,y:2,z:2,p:0} means the device
    #     is located at that coordinates in the 4-D mesh.
    #   - create a mapping from mesh location to device id.
    # 3. Find all replicated mesh dimension names, i.e., 'y' and `p` in the
    #     example above.
    # 4. Iterate over all unpacked components, translate the offset (device id)
    #    to mesh location, called (x',y',z',p').
    #   - For `y`, which is replicated dim in the mesh, check all unpacked
    #     components at (x',*,z',p') are same as the component at (x',0,z',p').
    #   - For `p`, which is also replicated dim in the mesh, check all unpacked
    #     components at (x',y',z',*) are same as the component at (x',y',z',0).

    def hash_key(loc):
      """Hash key for Python dict."""
      # Python dict is unhashable. Creates a sorted dict and dumps as json str.
      d = collections.OrderedDict(sorted(loc.items(), key=lambda x: x[0]))
      return json.dumps(d)

    offset_to_mesh_loc_dict = layout.mesh.unravel_index()
    mesh_loc_to_offset_dict = {}
    for offset, loc in offset_to_mesh_loc_dict.items():
      mesh_loc_to_offset_dict[hash_key(loc)] = offset

    # pylint: disable=protected-access
    replicated_dims = [
        x for x in layout.mesh._dim_names if x not in layout.sharding_specs
    ]
    # pylint: enable=protected-access

    for offset, tensor in enumerate(unpacked):
      mesh_loc = offset_to_mesh_loc_dict[offset]
      for dim_sharding in replicated_dims:
        if mesh_loc[dim_sharding] != 0:
          mesh_loc = copy.deepcopy(mesh_loc)  # deepcopy as we will mutate
          mesh_loc[dim_sharding] = 0
          offset = mesh_loc_to_offset_dict[hash_key(mesh_loc)]
          # tol is be as low as possible as they should match "exactly". so, we
          # ignore the `tol` passed by caller and choose the default one.
          self.assertAllClose(tensor, unpacked[offset])


def product(*lists):
  """Makes a product of names parameters list."""
  # Each element lists should be a tuple of tuples of the form
  # (("test1", ...), ("test2", ...), ...).
  # Function returns the product of the lists with the labels concatenated.
  return [  # pylint: disable=g-complex-comprehension
      (''.join(p[0]
               for p in elt), *sum((p[1:]
                                    for p in elt), ()))
      for elt in itertools.product(*lists)
  ]


def reset_dtensor():
  """Resets the singleton DTensor Device.

  This behavior is not generally exposed and only meant to be used in tests.
  """
  api._reset()  # pylint: disable=protected-access


__all__ = [
    'DEFAULT_TOL',
    'DTensorTestUtilBackend',
    'DTENSOR_TEST_UTIL_BACKEND',
    'create_device_ids_array',
    'create_device_array',
    'create_device_list',
    'reset_context',
    'reset_logical_devices',
    'list_local_logical_devices',
    'is_tfrt_enabled',
    'FLAGS',
    'DTensorBaseTest',
    'product',
    'reset_dtensor',
    'is_tpu_present',
    'is_gpu_present',
]
