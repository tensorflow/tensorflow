# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Classes implementing a multi-worker ps DistributionStrategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.util.tf_export import tf_export


@tf_export("distribute.experimental.CentralStorageStrategy", v1=[])
class CentralStorageStrategy(distribute_lib.Strategy):
  """A one-machine strategy that puts all variables on a single device.

  Variables are assigned to local CPU or the only GPU. If there is more
  than one GPU, compute operations (other than variable update operations)
  will be replicated across all GPUs.

  Args:
    compute_devices: an optional list of strings for device to replicate models
      on. If this is not provided, all local GPUs will be used; if there is no
      GPU, local CPU will be used.
    parameter_device: an optional device string for which device to put
      variables on. The default one is CPU or GPU if there is only one.
  """

  def __init__(self, compute_devices=None, parameter_device=None):
    extended = parameter_server_strategy.ParameterServerStrategyExtended(
        self,
        compute_devices=compute_devices,
        parameter_device=parameter_device)
    super(CentralStorageStrategy, self).__init__(extended)

  @classmethod
  def _from_num_gpus(cls, num_gpus):
    return cls(device_util.local_devices_from_num_gpus(num_gpus))


@tf_export(v1=["distribute.experimental.CentralStorageStrategy"])
class CentralStorageStrategyV1(distribute_lib.StrategyV1):

  __doc__ = CentralStorageStrategy.__doc__

  def __init__(self, compute_devices=None, parameter_device=None):
    """Initializes this strategy with default TFConfigClusterResolver."""
    super(CentralStorageStrategyV1, self).__init__(
        parameter_server_strategy.ParameterServerStrategyExtended(
            self,
            compute_devices=compute_devices,
            parameter_device=parameter_device))
