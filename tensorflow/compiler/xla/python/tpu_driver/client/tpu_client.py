# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""XLA LocalClient interface for interacting with TPUs via the TPU driver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

from tensorflow.compiler.xla.python import xla_client
from tensorflow.compiler.xla.python import xla_extension as _xla
from tensorflow.compiler.xla.python.tpu_driver.client import tpu_client_extension as _tpu_client


class TpuBackend(xla_client.Backend):
  """XLA backend implemented using the Tpu driver API."""

  # Cache the backends to prevent double driver initializations.
  _local_backend = None

  def __init__(self, client):
    """Creates a new TpuBackend.

    Args:
      client: A _tpu_client.TpuClient object.
    """
    super(TpuBackend, self).__init__('tpu')
    self.client = client

  @staticmethod
  def create(worker=None, force=False):
    # `force` == True will skip caching any backends (if applicable) and will
    # always try to create a new client.
    if worker is None:
      raise ValueError(
          'Failed to create TpuBackend. The `worker` parameter must not be '
          '`None`. Use `local` to connect to a local TPU or '
          '`grpc://host:port` to connect to a remote TPU.')

    if worker == 'local' or 'local://' in worker:
      # We usually want to cache for local backends to prevent double
      # initialization, except where `force` == True.
      if worker == 'local':
        worker = 'local://'
      if force:
        return TpuBackend(_tpu_client.TpuClient.Get(worker))
      if TpuBackend._local_backend is None:
        logging.info('Starting the local TPU driver.')
        TpuBackend._local_backend = TpuBackend(
            _tpu_client.TpuClient.Get(worker))
      return TpuBackend._local_backend
    else:
      # We do not cache for non-local backends.
      return TpuBackend(_tpu_client.TpuClient.Get(worker))

  def device_count(self):
    return self.client.device_count()

  def local_device_count(self):
    return self.client.local_device_count()

  def local_devices(self):
    return self.client.local_devices()

  def devices(self):
    return self.client.devices()

  def host_id(self):
    return self.client.host_id()

  def buffer_from_pyval(self, pyval, device=None):
    if device is None:
      device = self.client.local_devices()[0]
    return _tpu_client.PyTpuBuffer.from_python(pyval, self.client, device)

  def make_tuple(self, c_buffers, device):
    return _tpu_client.PyTpuBuffer.make_tuple(c_buffers, self.client, device)

  def compile(self, c_computation, compile_options):
    options = _xla.ExecutableBuildOptions()
    options.num_replicas = compile_options.num_replicas
    options.num_partitions = compile_options.num_partitions
    if compile_options.result_layout:
      options.result_layout = compile_options.result_layout
    options.debug_options.xla_cpu_fast_math_honor_infs = True
    options.debug_options.xla_cpu_fast_math_honor_nans = True
    options.debug_options.xla_cpu_fast_math_honor_division = True
    options.debug_options.xla_cpu_fast_math_honor_functions = True
    options.debug_options.xla_gpu_enable_fast_min_max = False
    return _tpu_client.TpuExecutable.Compile(c_computation,
                                             compile_options.argument_layouts,
                                             options, self.client,
                                             compile_options.device_assignment)

  def get_default_device_assignment(self, num_replicas, num_partitions):
    return self.client.GetDefaultDeviceAssignment(num_replicas, num_partitions)

  def serialize(self, executable):
    return self.client.SerializeExecutable(executable)

  def deserialize(self, serialized_executable):
    return self.client.DeserializeExecutable(serialized_executable, self.client)
