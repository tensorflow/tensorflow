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
"""Functions for configuring TensorFlow execution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager import context
from tensorflow.python.util.tf_export import tf_export


@tf_export('config.threading.intra_op_parallelism_threads')
def get_intra_op_parallelism_threads():
  """Get number of threads used within an individual op for parallelism.

  Certain operations like matrix multiplication and reductions can utilize
  parallel threads for speed ups. A value of 0 means the system picks an
  appropriate number.

  Returns:
    Number of parallel threads
  """
  return context.context().intra_op_parallelism_threads


@tf_export('config.threading.set_intra_op_parallelism_threads')
def set_intra_op_parallelism_threads(num_threads):
  """Set number of threads used within an individual op for parallelism.

  Certain operations like matrix multiplication and reductions can utilize
  parallel threads for speed ups. A value of 0 means the system picks an
  appropriate number.

  Args:
    num_threads: Number of parallel threads
  """
  context.context().intra_op_parallelism_threads = num_threads


@tf_export('config.threading.inter_op_parallelism_threads')
def get_inter_op_parallelism_threads():
  """Get number of threads used for parallelism between independent operations.

  Determines the number of threads used by independent non-blocking operations.
  0 means the system picks an appropriate number.

  Returns:
    Number of parallel threads
  """
  return context.context().inter_op_parallelism_threads


@tf_export('config.threading.set_inter_op_parallelism_threads')
def set_inter_op_parallelism_threads(num_threads):
  """Set number of threads used for parallelism between independent operations.

  Determines the number of threads used by independent non-blocking operations.
  0 means the system picks an appropriate number.

  Args:
    num_threads: Number of parallel threads
  """
  context.context().inter_op_parallelism_threads = num_threads


@tf_export('config.optimizer.get_jit')
def get_optimizer_jit():
  """Get if JIT compilation is enabled.

  Note that optimizations are only applied in graph mode, (within tf.function).

  Returns:
    If JIT compilation is enabled.
  """
  return context.context().optimizer_jit


@tf_export('config.optimizer.set_jit')
def set_optimizer_jit(enabled):
  """Set if JIT compilation is enabled.

  Args:
    enabled: Whether to enable JIT compilation.
  """
  context.context().optimizer_jit = enabled


@tf_export('config.optimizer.get_experimental_options')
def get_optimizer_experimental_options():
  """Get experimental optimizer options.

  Refer to tf.config.optimizer.set_experimental_options for a list of current
  options.

  Note that optimizations are only applied in graph mode, (within tf.function).
  In addition, as these are experimental options, the list is subject to change.

  Returns:
    Dictionary of configured experimental optimizer options
  """
  return context.context().get_optimizer_experimental_options()


@tf_export('config.optimizer.set_experimental_options')
def set_optimizer_experimental_options(options):
  """Set experimental optimizer options.

  Note that optimizations are only applied in graph mode, (within tf.function).
  In addition, as these are experimental options, the list is subject to change.

  Args:
    options: Dictionary of experimental optimizer options to configure.
      Valid keys:
      - layout_optimizer: Optimize tensor layouts
        e.g. This will try to use NCHW layout on GPU which is faster.
      - constant_folding: Fold constants
        Statically infer the value of tensors when possible, and materialize the
        result using constants.
      - shape_optimization: Simplify computations made on shapes.
      - remapping: Remap subgraphs onto more efficient implementations.
      - arithmetic_optimization: Simplify arithmetic ops with common
        sub-expression elimination and arithmetic simplification.
      - dependency_optimization: Control dependency optimizations. Remove
        redundant control dependencies, which may enable other optimization.
        This optimizer is also essential for pruning Identity and NoOp nodes.
      - loop_optimization: Loop optimizations.
      - function_optimization: Function optimizations and inlining.
      - debug_stripper: Strips debug-related nodes from the graph.
      - disable_model_pruning: Disable removal of unnecessary ops from the graph
      - scoped_allocator_optimization: Try to allocate some independent Op
        outputs contiguously in order to merge or eliminate downstream Ops.
      - pin_to_host_optimization: Force small ops onto the CPU.
      - implementation_selector: Enable the swap of kernel implementations based
        on the device placement.
      - disable_meta_optimizer: Disable the entire meta optimizer.
      - min_graph_nodes: The minimum number of nodes in a graph to optimizer.
        For smaller graphs, optimization is skipped.
  """
  context.context().set_optimizer_experimental_options(options)


@tf_export('config.get_soft_device_placement')
def get_soft_device_placement():
  """Get if soft device placement is enabled.

  If enabled, an op will be placed on CPU if any of the following are true
    1. there's no GPU implementation for the OP
    2. no GPU devices are known or registered
    3. need to co-locate with reftype input(s) which are from CPU

  Returns:
    If soft placement is enabled.
  """
  return context.context().soft_device_placement


@tf_export('config.set_soft_device_placement')
def set_soft_device_placement(enabled):
  """Set if soft device placement is enabled.

  If enabled, an op will be placed on CPU if any of the following are true
    1. there's no GPU implementation for the OP
    2. no GPU devices are known or registered
    3. need to co-locate with reftype input(s) which are from CPU

  Args:
    enabled: Whether to enable soft placement.
  """
  context.context().soft_device_placement = enabled


@tf_export('config.experimental.get_device_policy')
def get_device_policy():
  """Gets the current device policy.

  The device policy controls how operations requiring inputs on a specific
  device (e.g., on GPU:0) handle inputs on a different device (e.g. GPU:1).

  This function only gets the device policy for the current thread. Any
  subsequently started thread will again use the default policy.

  Returns:
    Current thread device policy
  """
  device_policy = context.context().device_policy
  if device_policy == context.DEVICE_PLACEMENT_SILENT:
    return 'silent'
  elif device_policy == context.DEVICE_PLACEMENT_SILENT_FOR_INT32:
    return 'silent_for_int32'
  elif device_policy == context.DEVICE_PLACEMENT_WARN:
    return 'warn'
  elif device_policy == context.DEVICE_PLACEMENT_EXPLICIT:
    return 'explicit'
  else:
    raise ValueError('Not a valid device policy: %r' % device_policy)


@tf_export('config.experimental.set_device_policy')
def set_device_policy(device_policy):
  """Sets the current thread device policy.

  The device policy controls how operations requiring inputs on a specific
  device (e.g., on GPU:0) handle inputs on a different device (e.g. GPU:1).

  When using the default, an appropriate policy will be picked automatically.
  The default policy may change over time.

  This function only sets the device policy for the current thread. Any
  subsequently started thread will again use the default policy.

  Args:
    device_policy: A device policy.
      Valid values:
      - None: Switch to a system default.
      - 'warn': Copies the tensors which are not on the right device and logs
          a warning.
      - 'explicit': Raises an error if the placement is not as required.
      - 'silent': Silently copies the tensors. Note that this may hide
          performance problems as there is no notification provided when
          operations are blocked on the tensor being copied between devices.
      - 'silent_for_int32': silently copies `int32` tensors, raising errors on
          the other ones.

  Raises:
      ValueError: If an invalid `device_policy` is passed.
  """
  if device_policy == 'silent':
    context.context().device_policy = context.DEVICE_PLACEMENT_SILENT
  elif device_policy == 'silent_for_int32':
    context.context().device_policy = context.DEVICE_PLACEMENT_SILENT_FOR_INT32
  elif device_policy == 'warn':
    context.context().device_policy = context.DEVICE_PLACEMENT_WARN
  elif device_policy == 'explicit':
    context.context().device_policy = context.DEVICE_PLACEMENT_EXPLICIT
  elif device_policy is None:
    context.context().device_policy = None
  else:
    raise ValueError('Not a valid device policy: %r' % device_policy)


@tf_export('config.experimental.get_synchronous_execution')
def get_synchronous_execution():
  """Gets whether operations are executed synchronously or asynchronously.

  TensorFlow can execute operations synchronously or asynchronously. If
  asynchronous execution is enabled, operations may return "non-ready" handles.

  Returns:
    Current thread execution mode
  """
  return context.context().execution_mode == context.SYNC


@tf_export('config.experimental.set_synchronous_execution')
def set_synchronous_execution(enable):
  """Specifies whether operations are executed synchronously or asynchronously.

  TensorFlow can execute operations synchronously or asynchronously. If
  asynchronous execution is enabled, operations may return "non-ready" handles.

  When `enable` is set to None, an appropriate value will be picked
  automatically. The value picked may change between TensorFlow releases.

  Args:
    enable: Whether operations should be dispatched synchronously.
      Valid values:
      - None: sets the system default.
      - True: executes each operation synchronously.
      - False: executes each operation asynchronously.
  """
  if enable is None:
    context.context().execution_mode = None
  elif enable:
    context.context().execution_mode = context.SYNC
  else:
    context.context().execution_mode = context.ASYNC


@tf_export('config.experimental.list_physical_devices')
def list_physical_devices(device_type=None):
  """Return a list of physical devices visible to the runtime.

  Physical devices are hardware devices locally present on the current machine.
  By default all discovered CPU and GPU devices are considered visible. The
  `list_physical_devices` allows querying the hardware prior to runtime
  initialization.

  The following example ensures the machine can see at least 1 GPU.

  ```python
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "No GPUs found."
  ```

  Args:
    device_type: (optional) Device type to filter by such as "CPU" or "GPU"

  Returns:
    List of PhysicalDevice objects
  """
  return context.context().list_physical_devices(device_type)


@tf_export('config.experimental.list_logical_devices')
def list_logical_devices(device_type=None):
  """Return a list of logical devices created by runtime.

  Logical devices may correspond to physical devices or remote devices in the
  cluster. Operations and tensors may be placed on these devices by using the
  `name` of the LogicalDevice.

  For example:

  ```python
  logical_devices = tf.config.experimental.list_logical_devices('GPU')
  # Allocate on GPU:0
  with tf.device(logical_devices[0].name):
    one = tf.constant(1)
  # Allocate on GPU:1
  with tf.device(logical_devices[1].name):
    two = tf.constant(2)
  ```

  Args:
    device_type: (optional) Device type to filter by such as "CPU" or "GPU"

  Returns:
    List of LogicalDevice objects
  """
  return context.context().list_logical_devices(device_type=device_type)


@tf_export('config.experimental.get_visible_devices')
def get_visible_devices(device_type=None):
  """Get the list of visible physical devices.

  Returns a list of PhysicalDevice objects that are current marked as visible to
  the runtime. Any visible devices will have LogicalDevices assigned to them
  once the runtime is initialized.

  The following example verifies all visible GPUs have been disabled:

  ```python
  physical_devices = config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  # Disable all GPUS
  tf.config.experimental.set_visible_devices([], 'GPU')
  visible_devices = tf.config.experimental.get_visible_devices()
  for device in visible_devices:
    assert device.device_type != 'GPU'
  ```

  Args:
    device_type: (optional) Device types to limit query to.

  Returns:
    List of PhysicalDevice objects
  """
  return context.context().get_visible_devices(device_type)


@tf_export('config.experimental.set_visible_devices')
def set_visible_devices(devices, device_type=None):
  """Set the list of visible devices.

  Sets the list of PhysicalDevices to be marked as visible to the runtime. Any
  devices that are not marked as visible means TensorFlow will not allocate
  memory on it and will not be able to place any operations on it as no
  LogicalDevice will be created on it. By default all discovered devices are
  marked as visible.

  The following example demonstrates disabling the first GPU on the machine.

  ```python
  physical_devices = config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  # Disable first GPU
  tf.config.experimental.set_visible_devices(physical_devices[1:], 'GPU')
  logical_devices = config.experimental.list_logical_devices('GPU')
  # Logical device was not created for first GPU
  assert len(logical_devices) == len(physical_devices) - 1
  ```

  Args:
    devices: (optional) List of PhysicalDevice objects to make visible
    device_type: (optional) Device types to limit visibility configuration to.
      Other device types will be left unaltered.
  """
  context.context().set_visible_devices(devices, device_type)


@tf_export('config.experimental.get_memory_growth')
def get_memory_growth(device):
  """Get if memory growth is enabled for a PhysicalDevice.

  A PhysicalDevice with memory growth set will not allocate all memory on the
  device upfront.

  For example:

  ```python
  physical_devices = config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  assert tf.config.experimental.get_memory_growth(physical_devices[0]) == True
  ```

  Args:
    device: PhysicalDevice to query

  Returns:
    Current memory growth setting.
  """
  return context.context().get_memory_growth(device)


@tf_export('config.experimental.set_memory_growth')
def set_memory_growth(device, enable):
  """Set if memory growth should be enabled for a PhysicalDevice.

  A PhysicalDevice with memory growth set will not allocate all memory on the
  device upfront. Memory growth cannot be configured on a PhysicalDevice with
  virtual devices configured.

  For example:

  ```python
  physical_devices = config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
  ```

  Args:
    device: PhysicalDevice to configure
    enable: Whether to enable or disable memory growth
  """
  context.context().set_memory_growth(device, enable)


@tf_export('config.experimental.get_virtual_device_configuration')
def get_virtual_device_configuration(device):
  """Get the virtual device configuration for a PhysicalDevice.

  Returns the list of VirtualDeviceConfiguration objects previously configured
  by a call to `tf.config.experimental.set_virtual_device_configuration()`.

  For example:

  ```python
  physical_devices = tf.config.experimental.list_physical_devices('CPU')
  assert len(physical_devices) == 1, "No CPUs found"
  tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(),
     tf.config.experimental.VirtualDeviceConfiguration()])
  configs = tf.config.experimental.get_virtual_device_configuration('CPU')
  assert len(configs) == 2
  ```

  Args:
    device: PhysicalDevice to query

  Returns:
    List of VirtualDeviceConfiguration objects
  """
  return context.context().get_virtual_device_configuration(device)


@tf_export('config.experimental.set_virtual_device_configuration')
def set_virtual_device_configuration(device, virtual_devices):
  """Set the virtual device configuration for a PhysicalDevice.

  A PhysicalDevice marked as visible will by default have a single LogicalDevice
  allocated to it once the runtime is configured. Specifying a list of
  tf.config.experimental.VirtualDeviceConfiguration objects allows multiple
  devices to be configured that utilize the same PhysicalDevice.

  The following example splits the CPU into 2 virtual devices:

  ```python
  physical_devices = tf.config.experimental.list_physical_devices('CPU')
  assert len(physical_devices) == 1, "No CPUs found"
  # Specify 2 virtual CPUs. Note currently memory limit is not supported.
  tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(),
     tf.config.experimental.VirtualDeviceConfiguration()])
  logical_devices = tf.config.experimental.list_logical_devices('CPU')
  assert len(logical_devices) == 2

  try:
    tf.config.experimental.set_virtual_device_configuration(
      physical_devices[0],
      [tf.config.experimental.VirtualDeviceConfiguration(),
       tf.config.experimental.VirtualDeviceConfiguration(),
       tf.config.experimental.VirtualDeviceConfiguration(),
       tf.config.experimental.VirtualDeviceConfiguration()])
  except:
    print('Cannot modify the virtual devices once they have been initialized.')
  ```

  The following example splits the GPU into 2 virtual devices with 100 MB each:

  ```python
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  assert len(physical_devices) > 0, "No GPUs found"
  tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100),
     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=100)])

  try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
  except:
    print('Cannot set memory growth when virtual devices configured')

  logical_devices = tf.config.experimental.list_logical_devices('GPU')
  assert len(logical_devices) == len(physical_devices) + 1

  try:
    tf.config.experimental.set_virtual_device_configuration(
      physical_devices[0],
      [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10),
       tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10)])
  except:
    print('Cannot modify the virtual devices once they have been initialized.')
  ```

  Args:
    device: (optional) Need to update
    virtual_devices: (optional) Need to update
  """
  context.context().set_virtual_device_configuration(device, virtual_devices)
