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
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@tf_export('config.threading.get_intra_op_parallelism_threads')
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


@tf_export('config.threading.get_inter_op_parallelism_threads')
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
      - auto_mixed_precision: Change certain float32 ops to float16 on Volta
        GPUs and above. Without the use of loss scaling, this can cause
        numerical underflow (see
        `keras.mixed_precision.experimental.LossScaleOptimizer`).
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


@tf_export('config.list_physical_devices',
           'config.experimental.list_physical_devices')
@deprecation.deprecated_endpoints(
    'config.experimental.list_physical_devices')
def list_physical_devices(device_type=None):
  """Return a list of physical devices visible to the host runtime.

  Physical devices are hardware devices present on the host machine. By default
  all discovered CPU and GPU devices are considered visible.

  This API allows querying the physical hardware resources prior to runtime
  initialization. Thus, giving an opportunity to call any additional
  configuration APIs. This is in contrast to `tf.config.list_logical_devices`,
  which triggers runtime initialization in order to list the configured devices.

  The following example lists the number of visible GPUs on the host.

  >>> physical_devices = tf.config.list_physical_devices('GPU')
  >>> print("Num GPUs:", len(physical_devices))
  Num GPUs: ...

  However, the number of GPUs available to the runtime may change during runtime
  initialization due to marking certain devices as not visible or configuring
  multiple logical devices.

  Args:
    device_type: (optional string) Only include devices matching this device
      type. For example "CPU" or "GPU".

  Returns:
    List of discovered `tf.config.PhysicalDevice` objects
  """
  return context.context().list_physical_devices(device_type)


@tf_export('config.list_logical_devices',
           'config.experimental.list_logical_devices')
@deprecation.deprecated_endpoints(
    'config.experimental.list_logical_devices')
def list_logical_devices(device_type=None):
  """Return a list of logical devices created by runtime.

  Logical devices may correspond to physical devices or remote devices in the
  cluster. Operations and tensors may be placed on these devices by using the
  `name` of the `tf.config.LogicalDevice`.

  Calling `tf.config.list_logical_devices` triggers the runtime to configure any
  `tf.config.PhysicalDevice` visible to the runtime, thereby preventing
  further configuration. To avoid runtime initialization, call
  `tf.config.list_physical_devices` instead.

  For example:

  >>> logical_devices = tf.config.list_logical_devices('GPU')
  >>> if len(logical_devices) > 0:
  ...   # Allocate on GPU:0
  ...   with tf.device(logical_devices[0].name):
  ...     one = tf.constant(1)
  ...   # Allocate on GPU:1
  ...   with tf.device(logical_devices[1].name):
  ...     two = tf.constant(2)

  Args:
    device_type: (optional string) Only include devices matching this device
      type. For example "CPU" or "GPU".

  Returns:
    List of initialized `LogicalDevice`s
  """
  return context.context().list_logical_devices(device_type=device_type)


@tf_export('config.get_visible_devices',
           'config.experimental.get_visible_devices')
@deprecation.deprecated_endpoints(
    'config.experimental.get_visible_devices')
def get_visible_devices(device_type=None):
  """Get the list of visible physical devices.

  Returns the list of `PhysicalDevice`s currently marked as visible to the
  runtime. A visible device will have at least one `LogicalDevice` associated
  with it once the runtime is initialized.

  The following example verifies all visible GPUs have been disabled:

  >>> physical_devices = tf.config.list_physical_devices('GPU')
  >>> try:
  ...   # Disable all GPUS
  ...   tf.config.set_visible_devices([], 'GPU')
  ...   visible_devices = tf.config.get_visible_devices()
  ...   for device in visible_devices:
  ...     assert device.device_type != 'GPU'
  ... except:
  ...   # Invalid device or cannot modify virtual devices once initialized.
  ...   pass

  Args:
    device_type: (optional string) Only include devices matching this device
      type. For example "CPU" or "GPU".

  Returns:
    List of visible `PhysicalDevice`s
  """
  return context.context().get_visible_devices(device_type)


@tf_export('config.set_visible_devices',
           'config.experimental.set_visible_devices')
@deprecation.deprecated_endpoints(
    'config.experimental.set_visible_devices')
def set_visible_devices(devices, device_type=None):
  """Set the list of visible devices.

  Specifies which `PhysicalDevice` objects are visible to the runtime.
  TensorFlow will only allocate memory and place operations on visible
  physical devices, as otherwise no `LogicalDevice` will be created on them.
  By default all discovered devices are marked as visible.

  The following example demonstrates disabling the first GPU on the machine.

  >>> physical_devices = tf.config.list_physical_devices('GPU')
  >>> try:
  ...   # Disable first GPU
  ...   tf.config.set_visible_devices(physical_devices[1:], 'GPU')
  ...   logical_devices = tf.config.list_logical_devices('GPU')
  ...   # Logical device was not created for first GPU
  ...   assert len(logical_devices) == len(physical_devices) - 1
  ... except:
  ...   # Invalid device or cannot modify virtual devices once initialized.
  ...   pass

  Args:
    devices: List of `PhysicalDevice`s to make visible
    device_type: (optional) Only configure devices matching this device type.
      For example "CPU" or "GPU". Other devices will be left unaltered.

  Raises:
    ValueError: If argument validation fails.
    RuntimeError: Runtime is already initialized.
  """
  context.context().set_visible_devices(devices, device_type)


@tf_export('config.experimental.get_memory_growth')
def get_memory_growth(device):
  """Get if memory growth is enabled for a `PhysicalDevice`.

  If memory growth is enabled for a `PhysicalDevice`, the runtime initialization
  will not allocate all memory on the device.

  For example:

  >>> physical_devices = tf.config.list_physical_devices('GPU')
  >>> try:
  ...   tf.config.experimental.set_memory_growth(physical_devices[0], True)
  ...   assert tf.config.experimental.get_memory_growth(physical_devices[0])
  ... except:
  ...   # Invalid device or cannot modify virtual devices once initialized.
  ...   pass

  Args:
    device: `PhysicalDevice` to query

  Returns:
    A boolean indicating the memory growth setting for the `PhysicalDevice`.

  Raises:
    ValueError: Invalid `PhysicalDevice` specified.
  """
  return context.context().get_memory_growth(device)


@tf_export('config.experimental.set_memory_growth')
def set_memory_growth(device, enable):
  """Set if memory growth should be enabled for a `PhysicalDevice`.

  If memory growth is enabled for a `PhysicalDevice`, the runtime initialization
  will not allocate all memory on the device. Memory growth cannot be configured
  on a `PhysicalDevice` with virtual devices configured.

  For example:

  >>> physical_devices = tf.config.list_physical_devices('GPU')
  >>> try:
  ...   tf.config.experimental.set_memory_growth(physical_devices[0], True)
  ... except:
  ...   # Invalid device or cannot modify virtual devices once initialized.
  ...   pass

  Args:
    device: `PhysicalDevice` to configure
    enable: (Boolean) Whether to enable or disable memory growth

  Raises:
    ValueError: Invalid `PhysicalDevice` specified.
    RuntimeError: Runtime is already initialized.
  """
  context.context().set_memory_growth(device, enable)


@tf_export('config.get_logical_device_configuration',
           'config.experimental.get_virtual_device_configuration')
@deprecation.deprecated_endpoints(
    'config.experimental.get_virtual_device_configuration')
def get_logical_device_configuration(device):
  """Get the virtual device configuration for a `tf.config.PhysicalDevice`.

  Returns the list of `tf.config.LogicalDeviceConfiguration`
  objects previously configured by a call to
  `tf.config.set_logical_device_configuration`.

  For example:

  >>> physical_devices = tf.config.list_physical_devices('CPU')
  >>> assert len(physical_devices) == 1, "No CPUs found"
  >>> configs = tf.config.get_logical_device_configuration(
  ...   physical_devices[0])
  >>> try:
  ...   assert configs is None
  ...   tf.config.set_logical_device_configuration(
  ...     physical_devices[0],
  ...     [tf.config.LogicalDeviceConfiguration(),
  ...      tf.config.LogicalDeviceConfiguration()])
  ...   configs = tf.config.get_logical_device_configuration(
  ...     physical_devices[0])
  ...   assert len(configs) == 2
  ... except:
  ...   # Cannot modify virtual devices once initialized.
  ...   pass

  Args:
    device: `PhysicalDevice` to query

  Returns:
    List of `tf.config.LogicalDeviceConfiguration` objects or
    `None` if no virtual device configuration has been set for this physical
    device.
  """
  return context.context().get_logical_device_configuration(device)


@tf_export('config.set_logical_device_configuration',
           'config.experimental.set_virtual_device_configuration')
@deprecation.deprecated_endpoints(
    'config.experimental.set_virtual_device_configuration')
def set_logical_device_configuration(device, logical_devices):
  """Set the logical device configuration for a `tf.config.PhysicalDevice`.

  A visible `tf.config.PhysicalDevice` will by default have a single
  `tf.config.LogicalDevice` associated with it once the runtime is initialized.
  Specifying a list of `tf.config.LogicalDeviceConfiguration` objects allows
  multiple devices to be created on the same `tf.config.PhysicalDevice`.

  The following example splits the CPU into 2 logical devices:

  >>> physical_devices = tf.config.list_physical_devices('CPU')
  >>> assert len(physical_devices) == 1, "No CPUs found"
  >>> # Specify 2 virtual CPUs. Note currently memory limit is not supported.
  >>> try:
  ...   tf.config.set_logical_device_configuration(
  ...     physical_devices[0],
  ...     [tf.config.LogicalDeviceConfiguration(),
  ...      tf.config.LogicalDeviceConfiguration()])
  ...   logical_devices = tf.config.list_logical_devices('CPU')
  ...   assert len(logical_devices) == 2
  ...
  ...   tf.config.set_logical_device_configuration(
  ...     physical_devices[0],
  ...     [tf.config.LogicalDeviceConfiguration(),
  ...      tf.config.LogicalDeviceConfiguration(),
  ...      tf.config.LogicalDeviceConfiguration(),
  ...      tf.config.LogicalDeviceConfiguration()])
  ... except:
  ...   # Cannot modify logical devices once initialized.
  ...   pass

  The following example splits the GPU into 2 logical devices with 100 MB each:

  >>> physical_devices = tf.config.list_physical_devices('GPU')
  >>> try:
  ...   tf.config.set_logical_device_configuration(
  ...     physical_devices[0],
  ...     [tf.config.LogicalDeviceConfiguration(memory_limit=100),
  ...      tf.config.LogicalDeviceConfiguration(memory_limit=100)])
  ...
  ...   logical_devices = tf.config.list_logical_devices('GPU')
  ...   assert len(logical_devices) == len(physical_devices) + 1
  ...
  ...   tf.config.set_logical_device_configuration(
  ...     physical_devices[0],
  ...     [tf.config.LogicalDeviceConfiguration(memory_limit=10),
  ...      tf.config.LogicalDeviceConfiguration(memory_limit=10)])
  ... except:
  ...   # Invalid device or cannot modify logical devices once initialized.
  ...   pass

  Args:
    device: The `PhysicalDevice` to configure.
    logical_devices: (optional) List of `tf.config.LogicalDeviceConfiguration`
      objects to allocate for the specified `PhysicalDevice`. If None, the
      default configuration will be used.

  Raises:
    ValueError: If argument validation fails.
    RuntimeError: Runtime is already initialized.
  """
  context.context().set_logical_device_configuration(device, logical_devices)


@tf_export('config.experimental.enable_mlir_bridge')
def enable_mlir_bridge():
  """Enables experimental MLIR-Based TensorFlow Compiler Bridge.

  DO NOT USE, DEV AND TESTING ONLY AT THE MOMENT.

  NOTE: MLIR-Based TensorFlow Compiler is under active development and has
  missing features, please refrain from using. This API exists for development
  and testing only.

  TensorFlow Compiler Bridge (TF Bridge) is responsible for translating parts
  of TensorFlow graph into a form that can be accepted as an input by a backend
  compiler such as XLA.
  """
  context.context().enable_mlir_bridge = True


@tf_export('config.experimental.disable_mlir_bridge')
def disable_mlir_bridge():
  """Disables experimental MLIR-Based TensorFlow Compiler Bridge."""
  context.context().enable_mlir_bridge = False
