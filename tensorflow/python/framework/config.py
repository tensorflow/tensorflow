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


@tf_export('config.gpu.get_per_process_memory_fraction')
def get_gpu_per_process_memory_fraction():
  """Get fraction of the available GPU memory to allocate for each process.

  1.0 means to allocate all of the GPU memory, 0.5 means the process allocates
  up to half of the available GPU memory.

  Returns:
    Current GPU per process memory fraction
  """
  return context.context().gpu_per_process_memory_fraction


@tf_export('config.gpu.set_per_process_memory_fraction')
def set_gpu_per_process_memory_fraction(fraction):
  """Set fraction of the available GPU memory to allocate for each process.

  1.0 means to allocate all of the GPU memory, 0.5 means the process allocates
  up to half of the available GPU memory.

  Args:
    fraction: Fraction of GPU memory to allocate
  """
  context.context().gpu_per_process_memory_fraction = fraction


@tf_export('config.gpu.get_per_process_memory_growth')
def get_gpu_per_process_memory_growth():
  """Get if GPU memory should be pre-allocated or allowed to grow.

  Returns:
    If GPU memory growth should be enabled
  """
  return context.context().gpu_per_process_memory_growth


@tf_export('config.gpu.set_per_process_memory_growth')
def set_gpu_per_process_memory_growth(enabled):
  """Set if GPU memory should be pre-allocated or allowed to grow.

  Args:
    enabled: Indicates if GPU memory growth should be enabled
  """
  context.context().gpu_per_process_memory_growth = enabled


@tf_export('config.threading.intra_op_parallelism_threads')
def get_intra_op_parallelism_threads():
  """Get number of threads used within an individual op for parallelism.

  Certain operations like matrix multiplication and reductions can utilize
  parellel threads for speed ups. A value of 0 means the system picks an
  appropriate number.

  Returns:
    Number of parallel threads
  """
  return context.context().intra_op_parallelism_threads


@tf_export('config.threading.set_intra_op_parallelism_threads')
def set_intra_op_parallelism_threads(num_threads):
  """Set number of threads used within an individual op for parallelism.

  Certain operations like matrix multiplication and reductions can utilize
  parellel threads for speed ups. A value of 0 means the system picks an
  appropriate number.

  Args:
    num_threads: Number of parallel threads
  """
  context.context().intra_op_parallelism_threads = num_threads


@tf_export('config.threading.inter_op_parallelism_threads')
def get_inter_op_parallelism_threads():
  """Get number of threads used for parallelism between independent operations.

  Determines the number of threads used by independent non-blokcing operations.
  0 means the system picks an appropriate number.

  Returns:
    Number of parallel threads
  """
  return context.context().inter_op_parallelism_threads


@tf_export('config.threading.set_inter_op_parallelism_threads')
def set_inter_op_parallelism_threads(num_threads):
  """Set number of threads used for parallelism between independent operations.

  Determines the number of threads used by independent non-blokcing operations.
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
