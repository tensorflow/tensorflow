# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Utility functions related to the Graphcore IPU."""

from tensorflow.core.protobuf import config_pb2


def create_ipu_config(profiling=False, num_ipus=None, tiles_per_ipu=None):
  """Create the IPU options for an IPU model device.

  Args:
    profiling:
    num_ipus:
    tiles_per_ipu:

  Returns:
    An IPUOptions configuration protobuf, suitable for using in the creation
    of the ConfigProto session options.

    ```python
    opts = create_ipu_config(True, 1, 64)
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

  """

  opts = config_pb2.IPUOptions()
  dev = opts.device_config.add()
  dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
  dev.profiling.enable_compilation_trace = profiling
  dev.profiling.enable_io_trace = profiling
  dev.profiling.enable_execution_trace = profiling

  if num_ipus:
    dev.ipu_model_config.num_ipus = num_ipus

  if tiles_per_ipu:
    dev.ipu_model_config.tiles_per_ipu = tiles_per_ipu

  return opts