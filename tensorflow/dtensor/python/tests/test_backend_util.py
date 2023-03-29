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
"""Utility to set up DTensor backend in tests."""

# LINT.IfChange
import multiprocessing
import os

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.python.platform import test as tf_test


class DTensorTestBackendConfigurator:
  """Configurate test backends."""

  def __init__(self, test_case: tf_test.TestCase):
    self._test_case = test_case
    # TODO(b/260771689): Refactor common backend set up logic to here.

  def tearDown(self):
    # Only need to explicitly shuts down TPU system in TFRT since in current
    # runtime, the shutdown is done in initialization process.
    if accelerator_util.is_initialized():
      accelerator_util.shutdown_accelerator_system()


def config_test_mesh(mesh: layout_lib.Mesh):
  """No Op.

  Args:
    mesh: The DTensor mesh.
  """
  if config.backend_is_pw():
    del mesh


def slice_host_devices_for_multiworker(num_clients, client_id, ports):
  """Configure the current process to only use a slice of devices."""
  if num_clients == 0:
    # All GPUs are visible to the client.
    del os.environ['CUDA_VISIBLE_DEVICES']
    del os.environ['HIP_VISIBLE_DEVICES']
  else:    
    # Make the client_id-th GPU visible to the client.
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{client_id}'
    os.environ['HIP_VISIBLE_DEVICES'] = f'{client_id}'
    # Make the client_id-th (4x) TPU cores visible to the client.
    os.environ['CLOUD_TPU_TASK_ID'] = f'{client_id}'
    if 'tpu' in DTENSOR_TEST_UTIL_BACKEND.value:
      del ports  # Unused due to lack of implementation.
      # We need to find out if there is a way to slice a CloudTPU host to
      # multiple workers.
      raise NotImplementedError(
          'OSS multi-client tests of TPU is not supported.'
      )


def get_mp_context():
  return multiprocessing.get_context('forkserver')


def handle_test_main(main):
  main()


# LINT.ThenChange(test_backend_util.py)
