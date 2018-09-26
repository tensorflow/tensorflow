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
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import ops

import time

def create_ipu_config(type='IPU_MODEL', profiling=False, num_devices=1,
                      num_ipus=None, tiles_per_ipu=None,
                      ipu_device_config_index=None,
                      use_poplar_text_report=False,
                      report_every_nth_execution=0):
  """Create the IPU options for an IPU model device.

  The configuration describes a system connsisting of multiple Tensorflow
  devices, each with control of one of more IPUs. The devices will be labelled
  `/device:IPU:0`, `/device:IPU:1' and so on.  Up to four devices can be
  configured. The `num_devices` parameter specifies the number of Tensorflow
  devices.

  Each device can control a specific number of IPUs, given by the `num_ipus`
  parameter.   This can either be an integer or a list of integers. A single
  integer will make each device in the system have the same number of IPUs.
  A list allows each device to have a different number of IPUs.  The list
  must have the same number of elements as the number of devices.

  As an alternative to supplying the number of IPUs, the index of the poplar
  device can be specified in the `ipu_device_config_index` parameter. An
  integer or a list can be provided, like for `num_ipus`.

  Examples:

    ```python
    # Create a single device, with one IPU and 64 tiles per IPU.
    opts = create_ipu_config(profiling=True, num_ipus=1, tiles_per_ipu=64)
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

    ```python
    # Create a single device, using poplar configuration 4
    opts = create_ipu_config(ipu_device_config_index=4)
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

    ```python
    # Create two devices, with 2 IPUs per device.
    opts = create_ipu_config(num_devices=2, num_ipus=2)
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

    ```python
    # Create two devices, with 1 IPU in the first device and 2 IPUs
    # in the second device.
    opts = create_ipu_config(num_devices=2, num_ipus=[1,2])
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

    ```python
    # Create four devices, using poplar device IDs 0, 1, 2, an 3
    # in the second device.
    opts = create_ipu_config(num_devices=2, ipu_device_config_index=[0, 1, 2, 3])
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

  Args:
    :param type: The type of hardware to target ('IPU', 'CPU', 'IPU_MODEL')
    :param profiling: Enable all IPU profiling
    :param num_devices: The number of Tensorflow devices to configure
    :param num_ipus: Number of IPUs pre Tensorflow device
    :param tiles_per_ipu: Number of tiles per IPU in the model. This only
                          applies when the `type` parameter is `IPU_MODEL`.
    :param ipu_device_config_index: Index into the vector returned by
                                getDevices() without specifying the number
                                of IPUs. This only applies when the `type`
                                parameter is `IPU`.
    :param use_poplar_text_report: Enable the poplar textual report summary
    :param report_every_nth_execution: Only produce an execution report on
                                       every Nth execution.  0=One report
                                       only.
  Returns:

    :return: An IPUOptions configuration protobuf, suitable for using in the
             creation of the ConfigProto session options.
  """

  if ipu_device_config_index is not None:
    if (num_ipus is not None) or (tiles_per_ipu is not None):
      raise Exception(
        "Cannot specify `ipu_device_config_index` and `num_ipus` or "
        "`tiles_per_ipu`")

  if isinstance(num_ipus, (list, tuple)):
    if len(num_ipus) != num_devices:
      raise Exception("`num_ipus` must contain `num_devices` entries.")

  if isinstance(ipu_device_config_index, (list, tuple)):
    if len(ipu_device_config_index) != num_devices:
      raise Exception(
        "`ipu_device_config_index` must contain `num_devices` entries.")

  if num_devices > 1 and isinstance(ipu_device_config_index, (int)):
    raise Exception(
      "`ipu_device_config_index` must be a list of length `num_devices`")

  if num_devices > 4:
    raise Exception("`num_devices` must not exceed 4")

  opts = config_pb2.IPUOptions()

  for d in range(num_devices):
    dev = opts.device_config.add()
    if type == 'IPU':
      dev.type = config_pb2.IPUOptions.DeviceConfig.IPU
    elif type == 'CPU':
      dev.type = config_pb2.IPUOptions.DeviceConfig.CPU
    elif type == 'IPU_MODEL':
      dev.type = config_pb2.IPUOptions.DeviceConfig.IPU_MODEL
    else:
      raise Exception("`type` parameter must be 'IPU', 'CPU', or 'IPU_MODEL'")

    dev.profiling.enable_compilation_trace = profiling
    dev.profiling.enable_io_trace = profiling
    dev.profiling.enable_execution_trace = profiling
    dev.profiling.enable_poplar_reports_text = use_poplar_text_report
    dev.profiling.report_every_nth_execution = report_every_nth_execution

    if isinstance(num_ipus, int):
      dev.ipu_model_config.num_ipus = num_ipus
    if isinstance(num_ipus, (list, tuple)):
      dev.ipu_model_config.num_ipus = num_ipus[d]

    if isinstance(ipu_device_config_index, int):
      dev.device_config_index.has_index = True
      dev.device_config_index.index = ipu_device_config_index
    if isinstance(ipu_device_config_index, (list, tuple)):
      dev.device_config_index.has_index = True
      dev.device_config_index.index = ipu_device_config_index[d]

    if isinstance(tiles_per_ipu, int):
      dev.ipu_model_config.tiles_per_ipu = tiles_per_ipu

  return opts


def extract_all_strings_from_event_trace(events):
  """Extract a concatenation of all data strings from an IPU event trace.

  :param events: An array of IPU events as returned from the
                 `ipu_compile_summary` operation.

  :return: A string containing the concatenation of all of the data fields of
           the events.

  """
  result = ""
  for e in events:
    evt = IpuTraceEvent.FromString(e)

    result = result + ("-" * 70) + "\n=> @ " + \
             time.strftime('%F %T %z', time.localtime(evt.timestamp)) + ": "

    if evt.type == IpuTraceEvent.COMPILE_BEGIN:
      evt_str = "Compile begin: " + evt.module_name.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.COMPILE_END:
      evt_str = "Compile end: " + evt.module_name.decode('utf-8') + "\n" + \
                "Duration: " + str(evt.data_int) + " us\n" + \
                evt.data_str.decode('utf-8')
    elif evt.type == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER:
      evt_str = "Host->Device\nHandle = " + evt.data_str.decode('utf-8') + "\n"
                #"Bytes = " + str(evt.data_int)
    elif evt.type == IpuTraceEvent.DEVICE_TO_HOST_TRANSFER:
      evt_str = "Device->Host\nHandle = " + evt.data_str.decode('utf-8') + "\n"
                #"Bytes = " + str(evt.data_int)
    elif evt.type == IpuTraceEvent.LOAD_ENGINE:
      evt_str = "Load engine: " + evt.module_name.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.EXECUTE:
      evt_str = "Execute\n" + evt.data_str.decode('utf-8')
    else:
      evt_str = "Unknown event"

    result = result + evt_str + '\n'

  return result

def extract_all_types_from_event_trace(events):
  """Return a list of the types of each event in an event trace tensor

  :param events: A tensor containing a list of IPU events as protobuf strings
  :return: A list containing the type of each event
  """
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    result += [evt.type]
  return result

def extract_all_events(events):
  """Extract a list containing each event as an event object
  :param events: A tensor containing a list of IPU events as protobuf strings
  :return: A list containing IpuTraceEvent objects
  """
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    result += [evt]
  return result

def extract_all_io_events(events):
  """Extract a list of all of the IO events from an IPU event trace tensor
  :param events: A tensor containing a list of IPU events as protobuf strings
  :return: A list containing only IO events as IpuTraceEvent objects
  """
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    if evt.type in [IpuTraceEvent.HOST_TO_DEVICE_TRANSFER,
                    IpuTraceEvent.DEVICE_TO_HOST_TRANSFER]:
      try:
        payload = json.loads(evt.data_str.decode('utf-8'))
        for t in payload["tensors"]:
          result += [(evt.type, t["name"])]
      except UnicodeDecodeError:
        pass
  return result

def move_variable_initialization_to_cpu(graph=None):
  """For all variables in the VARIABLES collection, move any initialization
  ops onto the CPU.

  :param graph: Operations are moved around on this graph.  The default graph
                will be used if not specified.
  :return: None
  """
  if not graph:
    graph = ops.get_default_graph()

  init_ops = []
  dep_ops = list(map(lambda x:x.initializer.inputs[1].op, graph.get_collection('variables')))
  visited  = set()

  while len(dep_ops) > 0:
    op = dep_ops.pop()
    if not op in visited:
      visited.add(op)
      init_ops += [op]
      dep_ops += map(lambda x:x.op, op.inputs)

  for op in init_ops:
    op._set_device('/device:CPU:0')
    op._set_attr('_class', attr_value_pb2.AttrValue(s=b'loc:@cpu'))

  return
