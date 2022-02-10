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

from absl import logging

# Import xla_client to load shared C++ extensions (just CompileOptions at the
# time of writing).
from tensorflow.compiler.xla.python import xla_client  # pylint: disable=unused-import
from tensorflow.compiler.xla.python.tpu_driver.client import tpu_client_extension as _tpu_client


class TpuBackend(object):
  """XLA backend implemented using the Tpu driver API."""

  # Cache the backends to prevent double driver initializations.
  _local_backend = None

  @staticmethod
  def create(worker=None, force=False):
    """Constructs a Cloud TPU backend."""
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
        return _tpu_client.TpuClient.Get(worker)
      if TpuBackend._local_backend is None:
        logging.info('Starting the local TPU driver.')
        TpuBackend._local_backend = _tpu_client.TpuClient.Get(worker)
      return TpuBackend._local_backend
    else:
      # We do not cache for non-local backends.
      return _tpu_client.TpuClient.Get(worker)
