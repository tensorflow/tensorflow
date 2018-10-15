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

import json
import re
import time

def create_ipu_config(profiling=False, use_poplar_text_report=False,
                      report_every_nth_execution=0):
  """Create an empty IPU session configuration structure.

  Args:
    :param profiling: Enable compilation reports and execution profiles.
    :param use_poplar_text_report: Enable the poplar textual report summary
    :param report_every_nth_execution: Only produce an execution report on
                                       every Nth execution.  0=One report
                                       only.

  Returns:

    :return: An empty IPUOptions configuration protobuf, suitable for using in
             the creation of the ConfigProto session options.
  """
  opts = config_pb2.IPUOptions()
  opts.ipu_model_config.enable_ipu_model = True

  opts.profiling.enable_compilation_trace = profiling
  opts.profiling.enable_io_trace = profiling
  opts.profiling.enable_execution_trace = profiling
  opts.profiling.enable_poplar_reports_text = use_poplar_text_report
  opts.profiling.report_every_nth_execution = report_every_nth_execution

  return opts

def set_compilation_options(opts, compilation_options=None):
  """Set the IPU compilation options for the session..

    ```python
    # Create a device with debug execution profile flag set to "compute_sets"
    opts = create_ipu_config()
    opts = set_compilation_options(opts,
        compilation_options={"debug.executionProfile", "compute_sets"})
    
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

  Args:
    :param opts: An IPUOptions session control protobuf.
    :param compilation_options: A dictionary of poplar compilation option flags
                                to be sent to the executor.
  Returns:

    :return: The IPUOptions configuration protobuf, with engine compilation
             options set.
  """
  if not(isinstance(compilation_options, dict)):
    raise Exception(
      "`compilation_options` must either be a dictionary")

  if (compilation_options is not None):
    for (option_name, value) in compilation_options.items():
      compilation_option = opts.compilation_options.add()
      compilation_option.option = option_name
      compilation_option.value = value

  return opts

def auto_select_ipus(opts, num_ipus):
  """Configure the IPUs to be used by the session.

  The configuration describes a system consisting of multiple Tensorflow
  devices, each with control of one of more IPUs. The devices will be labeled
  `/device:IPU:0`, `/device:IPU:1' and so on.

  Each device can control a specific number of IPUs, given by the `num_ipus`
  parameter. The system will automatically select IPU configurations from the
  available IPUs, where they match the desired number of IPUs.

  Examples:

    ```python
    # Create a single device, with one IPU
    opts = create_ipu_config()
    opts = auto_select_ipus(opts, num_ipus=1)
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

    ```python
    # Create two devices, with 2 IPUs per device.
    opts = create_ipu_config()
    opts = auto_select_ipus(opts, num_ipus=[2,2])
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

    ```python
    # Create two devices, with 1 IPU in the first device and 2 IPUs
    # in the second device.
    opts = create_ipu_config()
    opts = auto_select_ipus(opts, num_ipus=[1,2])
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

  Args:
    :param opts: An IPUOptions session control protobuf.
    :param num_ipus: List of IPUs per Tensorflow device
  Returns:

    :return: The IPUOptions configuration protobuf, configured for
             auto-selecting a set of IPU devices.
  """
  if not isinstance(num_ipus, (int, list, tuple)):
    raise Exception("`num_ipus` must be an integer, list or tuple.")

  if len(opts.device_config) > 0:
    raise Exception("IPU devices have already been configured.")

  if isinstance(num_ipus, int):
    dev = opts.device_config.add()
    dev.auto_count = num_ipus
  else:
    for n in num_ipus:
      dev = opts.device_config.add()
      dev.auto_count = n

  return opts

def select_ipus(opts, indicies):
  """Configure the IPUs to be used by the session.

  The configuration describes a system consisting of multiple Tensorflow
  devices, each with control of one of more IPUs. The devices will be labeled
  `/device:IPU:0`, `/device:IPU:1' and so on.

  Each device can control a specific number of IPUs, given by the `num_ipus`
  parameter. The system will automatically select IPU configurations from the
  available IPUs, where they match the desired number of IPUs.

  Examples:

    ```python
    # Create a single device, using IPU configuration index 0
    opts = create_ipu_config()
    opts = select_ipus(opts, indicies=[0])
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

    ```python
    # Create a single device, using IPU configuration index 8
    opts = create_ipu_config()
    opts = select_ipus(opts, indicies=[8])
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

    ```python
    # Create two devices, with IPU configuration indicies 0 and 1
    opts = create_ipu_config()
    opts = select_ipus(opts, indicies=[0,1])
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

    ```python
    # Create four devices, with IPU configuration indicies 8, 9, 10, 11
    opts = create_ipu_config()
    opts = select_ipus(opts, indicies=[8, 9, 10, 11])
    with tf.Session(config=tf.ConfigProto(ipu_options=opts)) as s:
      ...
    ```

  Args:
    :param opts: An IPUOptions session control protobuf.
    :param indicies: List of IPU configuration indicies
  Returns:

    :return: The IPUOptions configuration protobuf, with a number of devices
             selected by IPU configuration index.
  """

  if len(opts.device_config) > 0:
    raise Exception("IPU devices have already been configured.")

  if isinstance(indicies, (list, tuple)):
    raise Exception("`indicies` must be a list or tuple.")

  for i in indicies:
    dev = opts.device_config.add()
    dev.cfg_index = i

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
      evt_str = "Compile begin: " + \
                evt.compile_begin.module_name.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.COMPILE_END:
      evt_str = "Compile end: " + \
                evt.compile_end.module_name.decode('utf-8') + "\n" + \
                "Duration: " + str(evt.compile_end.duration) + " us\n" + \
                evt.compile_end.compilation_report.decode('utf-8')
    elif evt.type == IpuTraceEvent.HOST_TO_DEVICE_TRANSFER:
      evt_str = "Host->Device\n" + \
                evt.data_transfer.data_transfer.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.DEVICE_TO_HOST_TRANSFER:
      evt_str = "Device->Host\n" + \
                evt.data_transfer.data_transfer.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.LOAD_ENGINE:
      evt_str = "Load engine: " + \
                evt.load_engine.module_name.decode('utf-8') + "\n"
    elif evt.type == IpuTraceEvent.EXECUTE:
      evt_str = "Execute: " + \
                evt.execute.module_name.decode('utf-8') + "\n" + \
                evt.execute.execution_report.decode('utf-8')
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
        payload = json.loads(evt.data_transfer.data_transfer.decode('utf-8'))
        for t in payload["tensors"]:
          result += [(evt.type, t["name"])]
      except UnicodeDecodeError:
        pass
  return result

def get_memory_size_from_events(events):
  """Get the total memory consumption for the first compilation in the list
  of events.
  :param events: A list of IpuTraceEvent objects.
  :return: The total size as an integer, or None.
  """
  size = None
  for evt in events:
    if evt.type == IpuTraceEvent.COMPILE_END:
      in_memory_usage_section=False
      try:
        for l in evt.compile_end.compilation_report.decode('utf-8').split("\n"):
          l = l.strip()
          if l.startswith('Memory Usage'):
            in_memory_usage_section=True
          if l.startswith('Total') and in_memory_usage_section:
              m = re.match(r'.+:\s+(\d+)', l)
              if m:
                return int(m.group(1))
      except UnicodeDecodeError:
        pass
  return None


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
