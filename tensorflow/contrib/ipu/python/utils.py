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

from tensorflow.compiler.plugin.poplar.driver.trace_pb2 import IpuTraceEvent
from tensorflow.core.protobuf import config_pb2

import time

def create_ipu_config(profiling=False, num_ipus=None, tiles_per_ipu=None,
                      use_poplar_text_report=False, type='IPU_MODEL'):
  """Create the IPU options for an IPU model device.

  Args:
    profiling: Enable all IPU profiling
    num_ipus: Number of IPUs in the model
    tiles_per_ipu: Number of tiles per IPU in the model
    use_poplar_text_report: Enable the poplar textual report summary
    type: The type of hardware to target ('IPU', 'CPU', 'IPU_MODEL')

  Returns:
    An IPUOptions configuration protobuf, suitable for using in the creation
    of the ConfigProto session options.

    ```python
    opts = create_ipu_config(profiling=True, num_ipus=1, tiles_per_ipu=64)
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

  """

  opts = config_pb2.IPUOptions()
  dev = opts.device_config.add()
  if type == 'IPU':
    dev.type = config_pb2.IPUOptions.DeviceConfig.IPU
  elif type == 'CPU':
    dev.type = config_pb2.IPUOptions.DeviceConfig.CPU
  elif type == 'IPU_MODEL':
    dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
  else:
    raise Exception("type parameter must be 'IPU', 'CPU', or 'IPU_MODEL'")

  dev.profiling.enable_compilation_trace = profiling
  dev.profiling.enable_io_trace = profiling
  dev.profiling.enable_execution_trace = profiling
  dev.profiling.enable_poplar_reports_text = use_poplar_text_report

  if num_ipus:
    dev.ipu_model_config.num_ipus = num_ipus

  if tiles_per_ipu:
    dev.ipu_model_config.tiles_per_ipu = tiles_per_ipu

  return opts


def extract_all_strings_from_event_trace(events):
  """Extract a concatenation of all data strings from an IPU event trace.

  Args:
    events: An array of IPU events as returned from the `ipu_compile_summary`
            operation.

  Returns:
    A string containing the concatenation of all of the data fields of the
    events.

  """
  result = ""
  for e in events:
    evt = IpuTraceEvent.FromString(e)

    result = result + ("-" * 70) + "\n=> @ " + \
             time.strftime('%F %T %z', time.localtime(evt.timestamp)) + ": "

    if evt.type == IpuTraceEvent.COMPILE_BEGIN:
      evt_str = "Compile begin: " + evt.module_name + "\n"
    elif evt.type == IpuTraceEvent.COMPILE_END:
      evt_str = "Compile end: " + evt.module_name + "\n" + \
                "Duration: " + str(evt.data_int) + " us\n" + \
                evt.data_str.decode('utf-8')
    elif evt.type == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER:
      evt_str = "Host->Device\nHandle = " + evt.data_str.decode('utf-8') + "\n"
                #"Bytes = " + str(evt.data_int)
    elif evt.type == IpuTraceEvent.DEVICE_TO_HOST_TRANSFER:
      evt_str = "Device->Host\nHandle = " + evt.data_str.decode('utf-8') + "\n"
                #"Bytes = " + str(evt.data_int)
    elif evt.type == IpuTraceEvent.LOAD_ENGINE:
      evt_str = "Load engine: " + evt.module_name + "\n"
    elif evt.type == IpuTraceEvent.EXECUTE:
      evt_str = "Execute\n" + evt.data_str.decode('utf-8')
    else:
      evt_str = "Unknown event"

    result = result + evt_str + '\n'

  return result
