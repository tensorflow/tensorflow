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
from tensorflow.compiler.plugin.poplar.driver.config_pb2 import IpuOptions
from tensorflow.compiler.plugin.poplar.ops import gen_ipu_ops
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging

import json
import re
import time


def configure_ipu_system(config, device="cpu"):
  """Configure an IPU system.  Passing an IpuOptions protobuf created by the
  `create_ipu_config` function.

  Args:
    :param config: An IpuOptions configuration protobuf
    :param device: The CPU device which is local to the IPU hardware
  """
  if not (isinstance(config, IpuOptions)):
    raise Exception("`config` must be an IpuOptions instance")

  g = ops.Graph()
  with g.as_default():
    with ops.device(device):
      cfg_op = gen_ipu_ops.ipu_configure_hardware(config.SerializeToString())

  with session_lib.Session(graph=g) as sess:
    sess.run(cfg_op)


def create_ipu_config(profiling=False,
                      enable_ipu_events=False,
                      use_poplar_text_report=False,
                      report_every_nth_execution=0,
                      always_rearrange_copies_on_the_host=False,
                      disable_graph_convolution_caching=False,
                      retain_control_dependencies=False):
  """Create an empty IPU session configuration structure.

  Args:
    :param profiling: Enable compilation and execution reports,and IPU trace
                      events.
    :param enable_ipu_events: Enable IPU trace events without poplar reports.
    :param use_poplar_text_report: Enable the poplar textual report summary
    :param report_every_nth_execution: Only produce an execution report on
                                       every Nth execution.  0=One report
                                       only.
    :param always_rearrange_copies_on_the_host: *** Experimental Flag ***
                                                The data which is streamed
                                                to/from the device might be
                                                stored in different layouts on
                                                the device and on the host. If
                                                that is the case the
                                                rearrangment is performed on the
                                                device by default. By enabling
                                                this option the rearrangment
                                                will be perfomed on the host at
                                                the expense of latency.
    :param disable_graph_convolution_caching: By default, the convolution  
                                              opeartion searches for an 
                                              equivalent cached operation, and
                                              uses this  instead of creating a  
                                              new convolution. Setting this flag 
                                              forces the creation of a new 
                                              convoltuion. This can improve 
                                              runtime at the expense of graph
                                              size.
    :param retain_control_dependencies: When set to true, control dependencies
                                        from the Tensorflow graph are passed
                                        through to the backend.  This can result
                                        in a different memory size due to
                                        differing constraints on the operation
                                        scheduler.

  Returns:

    :return: An empty IpuOptions configuration protobuf, suitable for using in
             the creation of the ConfigProto session options.
  """
  if profiling and enable_ipu_events:
    raise Exception(
        "`profiling` and `enable_ipu_events` are mutually exclusive")

  opts = IpuOptions()
  opts.ipu_model_config.enable_ipu_model = True
  opts.ipu_model_config.compile_ipu_code = True

  opts.profiling.enable_ipu_trace_events = profiling or enable_ipu_events
  opts.profiling.enable_compilation_trace = profiling
  opts.profiling.enable_io_trace = profiling
  opts.profiling.enable_execution_trace = profiling
  opts.profiling.enable_poplar_reports_text = use_poplar_text_report
  opts.profiling.report_every_nth_execution = report_every_nth_execution

  opts.speed_size_config.always_rearrange_copies_on_the_host = always_rearrange_copies_on_the_host

  opts.speed_size_config.disable_graph_convolution_caching = disable_graph_convolution_caching

  opts.retain_control_dependencies = retain_control_dependencies

  return opts


def set_compilation_options(opts, compilation_options=None):
  """Set the IPU compilation options for the session..

    ```python
    # Create a device with debug execution profile flag set to "compute_sets"
    opts = create_ipu_config()
    opts = set_compilation_options(opts,
        compilation_options={"debug.executionProfile": "compute_sets",
                             "target.workerStackSizeInBytes": "64"})
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

  Args:
    :param opts: An IpuOptions session control protobuf.
    :param compilation_options: A dictionary of poplar compilation option flags
                                to be sent to the executor.
  Returns:

    :return: The IpuOptions configuration protobuf, with engine compilation
             options set.
  """
  if not (isinstance(compilation_options, dict)):
    raise Exception("`compilation_options` must be a dictionary")

  if (compilation_options is not None):
    for (option_name, value) in compilation_options.items():
      compilation_option = opts.compilation_options.add()
      compilation_option.option = option_name
      compilation_option.value = value

  return opts


def set_convolution_options(opts, convolution_options=None):
  """Set the IPU convolution compilation options for the session.
  *** This is an experimental function which might be removed in the future. ***
    ```python
    # Set "tempMemoryBudget" flag to "1000000"
    opts = create_ipu_config()
    opts = set_convolution_options(opts,
        convolution_options={"tempMemoryBudget": "1000000"})
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

  Args:
    :param opts: An IpuOptions session control protobuf.
    :param convolution_options: A dictionary of poplar option flags for the
                                convolutions.
  Returns:

    :return: The IpuOptions configuration protobuf, with convolution options
             set.
  """
  if not (isinstance(convolution_options, dict)):
    raise Exception("`convolution_options` must be a dictionary")

  if (convolution_options is not None):
    for (option_name, value) in convolution_options.items():
      opt = opts.convolution_options.add()
      opt.option = option_name
      opt.value = value

  return opts


def set_pooling_options(opts, pooling_options=None):
  """Set the IPU pooling compilation options for the session.
  *** This is an experimental function which might be removed in the future. ***
    ```python
    # Set "poolUseIntrospectiveMapping" flag to "false"
    opts = create_ipu_config()
    opts = set_pooling_options(opts,
        pooling_options={"poolUseIntrospectiveMapping": "false"})
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

  Args:
    :param opts: An IpuOptions session control protobuf.
    :param pooling_options: A dictionary of poplar option flags for the
                            pooling operation.
  Returns:

    :return: The IpuOptions configuration protobuf, with pooling options
             set.
  """
  if not (isinstance(pooling_options, dict)):
    raise Exception("`pooling_options` must be a dictionary")

  if (pooling_options is not None):
    for (option_name, value) in pooling_options.items():
      opt = opts.pooling_options.add()
      opt.option = option_name
      opt.value = value

  return opts


def set_report_options(opts, report_options=None):
  """Set the options used to influence Poplar report generation.

     The options are added to both the compile and execution report generations.

    ```python
    opts = create_ipu_config()
    opts = set_report_options(opts,
        report_options={"reportOption1": "false"})
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

  Args:
    :param opts: An IpuOptions session control protobuf.
    :param report_options: A dictionary of poplar option flags for the
                           report generation.
  Returns:

    :return: The IpuOptions configuration protobuf, with convolution options
             set.
  """
  if not (isinstance(report_options, dict)):
    raise Exception("`report_options` must be a dictionary")

  if (report_options is not None):
    for (option_name, value) in report_options.items():
      opt = opts.profiling.options.add()
      opt.option = option_name
      opt.value = value

  return opts


def set_ipu_model_options(opts, compile_ipu_code=True):
  """Set the IPU Model options.

  Args:
    :param compile_ipu_code: Whether or not to actually compile real IPU code
                             for modelling.

  Returns:

    :return: The IpuOptions configuration protobuf, with IPU model options
             set.
  """
  opts.ipu_model_config.compile_ipu_code = compile_ipu_code

  return opts


def set_recomputation_options(opts, recompute_norm_inputs=True):
  """Set re-computation options.

  Args:
    :param recompute_norm_inputs: Whether or not to re-compute the norm inputs
      during training. Enabling this option can reduce memory usage at the
      expense of extra computation.

  Returns:

    :return: The IpuOptions configuration protobuf.
  """

  opts.speed_size_config.recompute_norm_inputs = recompute_norm_inputs
  opts.speed_size_config.has_recompute_norm_inputs = True

  return opts


def auto_select_ipus(opts, num_ipus, sharded=False, number_of_replicas=None):
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
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

    ```python
    # Create two devices, with 2 IPUs per device.
    opts = create_ipu_config()
    opts = auto_select_ipus(opts, num_ipus=[2,2])
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

    ```python
    # Create two devices, with 1 IPU in the first device and 2 IPUs
    # in the second device.
    opts = create_ipu_config()
    opts = auto_select_ipus(opts, num_ipus=[1,2])
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

  Args:
    :param opts: An IpuOptions session control protobuf.
    :param num_ipus: List of IPUs per Tensorflow device
    :param sharded: Deprecated.
    :param number_of_replicas: The number of replicas to divide the device into.
                               This should be a divisor of the number of IPUs.
  Returns:

    :return: The IpuOptions configuration protobuf, configured for
             auto-selecting a set of IPU devices.
  """
  if len(opts.device_config) > 0:
    raise Exception("IPU devices have already been configured.")

  if not isinstance(num_ipus, (int, list, tuple)):
    raise Exception("`num_ipus` must be an integer, list or tuple.")

  if number_of_replicas is not None:
    if type(number_of_replicas) is not type(num_ipus):
      raise Exception("`number_of_replicas` must be same type as `num_ipus`")

  if sharded:
    logging.warning("`sharded` has been deprecated.  Mark operations with a "
                    "sharding attribute to enable sharding")

  if isinstance(num_ipus, int):
    dev = opts.device_config.add()
    dev.auto_count = num_ipus
    if isinstance(number_of_replicas, (int)):
      dev.num_replicas = number_of_replicas
  else:
    for i, n in enumerate(num_ipus):
      dev = opts.device_config.add()
      dev.auto_count = n
      if isinstance(number_of_replicas, (list, tuple)):
        dev.num_replicas = number_of_replicas[i]

  return opts


def select_ipus(opts, indices, sharded=False, number_of_replicas=None):
  """Configure the IPUs to be used by the session.

  The configuration describes a system consisting of multiple Tensorflow
  devices, each with control of one of more IPUs. The Tensorflow devices will be
  labeled `/device:IPU:0`, `/device:IPU:1' and so on.

  Each Tensorflow device uses a specific configuration consisting of one or more
  IPUs from the list of devices.  These can be found by running the Graphcore
  utility `gc-info -l`.  For instance, the following listing shows the device
  configurations available on a system with 16 IPUs.

  ```
  user@host:~$ gc-info -l
  Graphcore device listing:

  -+- Id:  [0], type:      [PCIe], PCI Domain: [0000:1a:00.0]
  -+- Id:  [1], type:      [PCIe], PCI Domain: [0000:1b:00.0]
  -+- Id:  [2], type:      [PCIe], PCI Domain: [0000:1c:00.0]
  -+- Id:  [3], type:      [PCIe], PCI Domain: [0000:1d:00.0]
  -+- Id:  [4], type:      [PCIe], PCI Domain: [0000:60:00.0]
  -+- Id:  [5], type:      [PCIe], PCI Domain: [0000:61:00.0]
  -+- Id:  [6], type:      [PCIe], PCI Domain: [0000:62:00.0]
  -+- Id:  [7], type:      [PCIe], PCI Domain: [0000:63:00.0]
  -+- Id:  [8], type:      [PCIe], PCI Domain: [0000:b1:00.0]
  -+- Id:  [9], type:      [PCIe], PCI Domain: [0000:b2:00.0]
  -+- Id: [10], type:      [PCIe], PCI Domain: [0000:b3:00.0]
  -+- Id: [11], type:      [PCIe], PCI Domain: [0000:b4:00.0]
  -+- Id: [12], type:      [PCIe], PCI Domain: [0000:da:00.0]
  -+- Id: [13], type:      [PCIe], PCI Domain: [0000:db:00.0]
  -+- Id: [14], type:      [PCIe], PCI Domain: [0000:dc:00.0]
  -+- Id: [15], type:      [PCIe], PCI Domain: [0000:dd:00.0]
  -+- Id: [32], type: [Multi IPU]
   |--- PCIe Id:  [7], DNC Id: [0], PCI Domain: [0000:63:00.0]
   |--- PCIe Id:  [6], DNC Id: [1], PCI Domain: [0000:62:00.0]
   |--- PCIe Id:  [5], DNC Id: [2], PCI Domain: [0000:61:00.0]
   |--- PCIe Id:  [4], DNC Id: [3], PCI Domain: [0000:60:00.0]
   |--- PCIe Id:  [3], DNC Id: [4], PCI Domain: [0000:1d:00.0]
   |--- PCIe Id:  [2], DNC Id: [5], PCI Domain: [0000:1c:00.0]
   |--- PCIe Id:  [1], DNC Id: [6], PCI Domain: [0000:1b:00.0]
   |--- PCIe Id:  [0], DNC Id: [7], PCI Domain: [0000:1a:00.0]
   |--- PCIe Id: [11], DNC Id: [8], PCI Domain: [0000:b4:00.0]
   |--- PCIe Id: [10], DNC Id: [9], PCI Domain: [0000:b3:00.0]
   |--- PCIe Id:  [9], DNC Id: [10], PCI Domain: [0000:b2:00.0]
   |--- PCIe Id:  [8], DNC Id: [11], PCI Domain: [0000:b1:00.0]
   |--- PCIe Id: [15], DNC Id: [12], PCI Domain: [0000:dd:00.0]
   |--- PCIe Id: [14], DNC Id: [13], PCI Domain: [0000:dc:00.0]
   |--- PCIe Id: [13], DNC Id: [14], PCI Domain: [0000:db:00.0]
   |--- PCIe Id: [12], DNC Id: [15], PCI Domain: [0000:da:00.0]
  -+- Id: [33], type: [Multi IPU]
   |--- PCIe Id:  [7], DNC Id: [0], PCI Domain: [0000:63:00.0]
   |--- PCIe Id:  [6], DNC Id: [1], PCI Domain: [0000:62:00.0]
   |--- PCIe Id:  [5], DNC Id: [2], PCI Domain: [0000:61:00.0]
   |--- PCIe Id:  [4], DNC Id: [3], PCI Domain: [0000:60:00.0]
   |--- PCIe Id:  [3], DNC Id: [4], PCI Domain: [0000:1d:00.0]
   |--- PCIe Id:  [2], DNC Id: [5], PCI Domain: [0000:1c:00.0]
   |--- PCIe Id:  [1], DNC Id: [6], PCI Domain: [0000:1b:00.0]
   |--- PCIe Id:  [0], DNC Id: [7], PCI Domain: [0000:1a:00.0]
  -+- Id: [34], type: [Multi IPU]
   |--- PCIe Id: [11], DNC Id: [0], PCI Domain: [0000:b4:00.0]
   |--- PCIe Id: [10], DNC Id: [1], PCI Domain: [0000:b3:00.0]
   |--- PCIe Id:  [9], DNC Id: [2], PCI Domain: [0000:b2:00.0]
   |--- PCIe Id:  [8], DNC Id: [3], PCI Domain: [0000:b1:00.0]
   |--- PCIe Id: [15], DNC Id: [4], PCI Domain: [0000:dd:00.0]
   |--- PCIe Id: [14], DNC Id: [5], PCI Domain: [0000:dc:00.0]
   |--- PCIe Id: [13], DNC Id: [6], PCI Domain: [0000:db:00.0]
   |--- PCIe Id: [12], DNC Id: [7], PCI Domain: [0000:da:00.0]
  -+- Id: [35], type: [Multi IPU]
   |--- PCIe Id:  [7], DNC Id: [0], PCI Domain: [0000:63:00.0]
   |--- PCIe Id:  [6], DNC Id: [1], PCI Domain: [0000:62:00.0]
   |--- PCIe Id:  [5], DNC Id: [2], PCI Domain: [0000:61:00.0]
   |--- PCIe Id:  [4], DNC Id: [3], PCI Domain: [0000:60:00.0]
  -+- Id: [36], type: [Multi IPU]
   |--- PCIe Id:  [3], DNC Id: [0], PCI Domain: [0000:1d:00.0]
   |--- PCIe Id:  [2], DNC Id: [1], PCI Domain: [0000:1c:00.0]
   |--- PCIe Id:  [1], DNC Id: [2], PCI Domain: [0000:1b:00.0]
   |--- PCIe Id:  [0], DNC Id: [3], PCI Domain: [0000:1a:00.0]
  -+- Id: [37], type: [Multi IPU]
   |--- PCIe Id: [11], DNC Id: [0], PCI Domain: [0000:b4:00.0]
   |--- PCIe Id: [10], DNC Id: [1], PCI Domain: [0000:b3:00.0]
   |--- PCIe Id:  [9], DNC Id: [2], PCI Domain: [0000:b2:00.0]
   |--- PCIe Id:  [8], DNC Id: [3], PCI Domain: [0000:b1:00.0]
  -+- Id: [38], type: [Multi IPU]
   |--- PCIe Id: [15], DNC Id: [0], PCI Domain: [0000:dd:00.0]
   |--- PCIe Id: [14], DNC Id: [1], PCI Domain: [0000:dc:00.0]
   |--- PCIe Id: [13], DNC Id: [2], PCI Domain: [0000:db:00.0]
   |--- PCIe Id: [12], DNC Id: [3], PCI Domain: [0000:da:00.0]
  -+- Id: [39], type: [Multi IPU]
   |--- PCIe Id:  [7], DNC Id: [0], PCI Domain: [0000:63:00.0]
   |--- PCIe Id:  [6], DNC Id: [1], PCI Domain: [0000:62:00.0]
  -+- Id: [40], type: [Multi IPU]
   |--- PCIe Id:  [5], DNC Id: [0], PCI Domain: [0000:61:00.0]
   |--- PCIe Id:  [4], DNC Id: [1], PCI Domain: [0000:60:00.0]
  -+- Id: [41], type: [Multi IPU]
   |--- PCIe Id:  [3], DNC Id: [0], PCI Domain: [0000:1d:00.0]
   |--- PCIe Id:  [2], DNC Id: [1], PCI Domain: [0000:1c:00.0]
  -+- Id: [42], type: [Multi IPU]
   |--- PCIe Id:  [1], DNC Id: [0], PCI Domain: [0000:1b:00.0]
   |--- PCIe Id:  [0], DNC Id: [1], PCI Domain: [0000:1a:00.0]
  -+- Id: [43], type: [Multi IPU]
   |--- PCIe Id: [11], DNC Id: [0], PCI Domain: [0000:b4:00.0]
   |--- PCIe Id: [10], DNC Id: [1], PCI Domain: [0000:b3:00.0]
  -+- Id: [44], type: [Multi IPU]
   |--- PCIe Id:  [9], DNC Id: [0], PCI Domain: [0000:b2:00.0]
   |--- PCIe Id:  [8], DNC Id: [1], PCI Domain: [0000:b1:00.0]
  -+- Id: [45], type: [Multi IPU]
   |--- PCIe Id: [15], DNC Id: [0], PCI Domain: [0000:dd:00.0]
   |--- PCIe Id: [14], DNC Id: [1], PCI Domain: [0000:dc:00.0]
  -+- Id: [46], type: [Multi IPU]
   |--- PCIe Id: [13], DNC Id: [0], PCI Domain: [0000:db:00.0]
   |--- PCIe Id: [12], DNC Id: [1], PCI Domain: [0000:da:00.0]
  ```

  Examples based on the listing above:

    ```python
    # Create a single device with 1 IPU at PCI address 0000:1a:00.0 by using IPU
    # configuration index 0
    opts = create_ipu_config()
    opts = select_ipus(opts, indices=[0])
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

    ```python
    # Create a single device with 1 IPU at PCI address 0000:b1:00.0 by using IPU
    # configuration index 8
    opts = create_ipu_config()
    opts = select_ipus(opts, indices=[8])
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

    ```python
    # Create two Tensorflow devices, with one IPU each, being devices at indices
    # 0 and 1
    opts = create_ipu_config()
    opts = select_ipus(opts, indices=[0, 1])
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

    ```python
    # Create two Tensorflow devices, with four IPUs each. The device
    # configurations at indices 37 (0000:b4:00.0, 0000:b3:00.0, 0000:b2:00.0,
    # 000:b1:00.0) and 38 (0000:dd:00.0, 0000:dc:00.0, 0000:db:00.0,
    # 00:da:00.0)
    opts = create_ipu_config()
    opts = select_ipus(opts, indices=[37, 38])
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

    ```python
    # Create four Tensorflow devices each with one IPU, at addresses
    # 0000:1a:00.0, 0000:1b:00.0, 0000:1c:00.0, 0000:1d:00.0.
    opts = create_ipu_config()
    opts = select_ipus(opts, indices=[0, 1, 2, 3])
    ipu.utils.configure_ipu_system(cfg)
    with tf.Session() as s:
      ...
    ```

  Args:
    :param opts: An IpuOptions session control protobuf.
    :param indicies: List of IPU configuration indicies.
    :param sharded: Deprecated.
    :param number_of_replicas: The number of replicas to divide the device into.
                               This should be a divisor of the number of IPUs.
  Returns:

    :return: The IpuOptions configuration protobuf, with a number of devices
             selected by IPU configuration index.
  """

  if len(opts.device_config) > 0:
    raise Exception("IPU devices have already been configured.")

  if not isinstance(indices, (list, tuple)):
    raise Exception("`indicies` must be a list or tuple.")

  if number_of_replicas is not None:
    if not isinstance(number_of_replicas, (list, tuple)):
      raise Exception("`number_of_replicas` must be a list or tuple.")

  if sharded:
    logging.warning("`sharded` has been deprecated.  Mark operations with a "
                    "sharding attribute to enable sharding")

  for n, i in enumerate(indices):
    dev = opts.device_config.add()
    dev.cfg_index = i
    if isinstance(number_of_replicas, (list, tuple)):
      dev.num_replicas = number_of_replicas[n]

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
    if evt.type in [
        IpuTraceEvent.HOST_TO_DEVICE_TRANSFER,
        IpuTraceEvent.DEVICE_TO_HOST_TRANSFER
    ]:
      try:
        payload = json.loads(evt.data_transfer.data_transfer.decode('utf-8'))
        for t in payload["tensors"]:
          result += [(evt.type, t["name"])]
      except UnicodeDecodeError:
        pass
  return result


def extract_compile_reports(events):
  """Get a list of all compiler reports in the event list.
  :param events: A list of trace event serialized protobufs
  :return: A list of tuples containing the module namd and report."""
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    if evt.type == IpuTraceEvent.COMPILE_END:
      try:
        module = evt.compile_end.module_name.decode('utf-8')
        rep = evt.compile_end.compilation_report.decode('utf-8')
        result += [(module, rep)]
      except UnicodeDecodeError:
        pass
  return result


def extract_execute_reports(events):
  """Get a list of all compiler reports in the event list.
  :param events: A list of trace event serialized protobufs
  :return: A list of tuples containing the module namd and report."""
  result = []
  for e in events:
    evt = IpuTraceEvent.FromString(e)
    if evt.type == IpuTraceEvent.EXECUTE:
      try:
        module = evt.execute.module_name.decode('utf-8')
        rep = evt.execute.execution_report.decode('utf-8')
        result += [(module, rep)]
      except UnicodeDecodeError:
        pass
  return result


def extract_graphviz_from_compilation_event(evt):
  """Return the final optimized XLA graph from a COMPILE_BEGIN event.
  :param evt: An IpuTraceEvent which is of type COMPILE_BEGIN.
  :return: A DOT file string of the main XLA computation.
  """
  if evt.type != IpuTraceEvent.COMPILE_BEGIN:
    raise Exception("`evt` must be a COMPILE_BEGIN event")
  return evt.compile_begin.xla_graph


def get_memory_size_from_events(events):
  """Get the total memory consumption for the first compilation in the list
  of events.
  :param events: A list of IpuTraceEvent objects.
  :return: The total size as an integer, or None.
  """
  size = None
  for evt in events:
    if evt.type == IpuTraceEvent.COMPILE_END:
      in_memory_usage_section = False
      try:
        for l in evt.compile_end.compilation_report.decode('utf-8').split(
            "\n"):
          l = l.strip()
          if l.startswith('Memory Usage'):
            in_memory_usage_section = True
          if l.startswith('Including Gaps') and in_memory_usage_section:
            m = re.match(r'.+:\s+([\d,]+) B', l)
            if m:
              return int(m.group(1).replace(',', ''))
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
  dep_ops = list(
      map(lambda x: x.initializer.inputs[1].op,
          graph.get_collection('variables')))
  visited = set()

  while len(dep_ops) > 0:
    op = dep_ops.pop()
    if not op in visited:
      visited.add(op)
      init_ops += [op]
      dep_ops += map(lambda x: x.op, op.inputs)

  for op in init_ops:
    op._set_device('/device:CPU:0')
    op._set_attr('_class', attr_value_pb2.AttrValue(s=b'loc:@cpu'))
    op._set_attr('_XlaCompile', attr_value_pb2.AttrValue(b=False))
    op._set_attr('_XlaScope', attr_value_pb2.AttrValue(s=b''))

  return
