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
# ==============================================================================
"""State management for eager execution."""

import collections
import contextlib
import copy
import gc
import os
import random
import threading

from absl import logging
import numpy as np
import six

from tensorflow.core.framework import function_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import coordination_config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import executor
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import tfrt_utils
from tensorflow.python.util import compat
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export

GRAPH_MODE = 0
EAGER_MODE = 1

default_execution_mode = EAGER_MODE if tf2.enabled() else GRAPH_MODE

# Cache from (old_device_name, partial_new_device_name) -> (new_device_name,
# new_device_spec).
# Note that we do not protect this with a lock and instead rely on python's GIL
# and the idempotent nature of writes to provide thread safety.
_device_parsing_cache = {}
_starting_device_spec = pydev.DeviceSpec.from_string("")

_MAXINT32 = 2**31 - 1

DEVICE_PLACEMENT_EXPLICIT = pywrap_tfe.TFE_DEVICE_PLACEMENT_EXPLICIT
DEVICE_PLACEMENT_WARN = pywrap_tfe.TFE_DEVICE_PLACEMENT_WARN
DEVICE_PLACEMENT_SILENT = pywrap_tfe.TFE_DEVICE_PLACEMENT_SILENT
DEVICE_PLACEMENT_SILENT_FOR_INT32 = (
    pywrap_tfe.TFE_DEVICE_PLACEMENT_SILENT_FOR_INT32)

SYNC = 0
ASYNC = 1

_KEEP_ALIVE_SECS = 600

_python_eager_context_create_counter = monitoring.Counter(
    "/tensorflow/api/python/eager_context_create_counter",
    "Counter for number of eager contexts created in Python.")

# Re-exporting through context.
is_tfrt_enabled = tfrt_utils.enabled

# This flag and the associated environment var are transient and will eventually
# be removed, once this experiment is enabled by default.
_RUN_EAGER_OP_AS_FUNCTION_ENABLED = os.getenv("TF_RUN_EAGER_OP_AS_FUNCTION",
                                              "1") == "1"

# This flag and the associated environment var are transient and will eventually
# be removed, once this experiment is enabled by default.
_JIT_COMPILE_REWRITE_ENABLED = os.getenv("TF_JIT_COMPILE_REWRITE") == "1"


# This method should only be called after the context has beein initialized.
def enable_run_eager_op_as_function():
  """Execute elementary eager ops (non-function) wrapped in a call op.

  This should be functionally equivalent to running the eager op's kernel
  directly (the default) but reduces the number of codepaths for executing
  TF2 programs in the runtime, thereby improving consistency (in terms of
  optimizations and rewrites for instance) and maintainability.
  """
  global _RUN_EAGER_OP_AS_FUNCTION_ENABLED
  _RUN_EAGER_OP_AS_FUNCTION_ENABLED = True
  if context_safe() is not None:
    context_safe().run_eager_op_as_function = True


# This method should only be called after the context has been initialized.
def disable_run_eager_op_as_function():
  global _RUN_EAGER_OP_AS_FUNCTION_ENABLED
  _RUN_EAGER_OP_AS_FUNCTION_ENABLED = False
  if context_safe() is not None:
    context_safe().run_eager_op_as_function = False


def run_eager_op_as_function_enabled():
  if context_safe() is not None:
    return context_safe().run_eager_op_as_function
  return _RUN_EAGER_OP_AS_FUNCTION_ENABLED


# This method should only be called after the context has beein initialized.
def enable_jit_compile_rewrite():
  """Run jit_compile functions through rewrite pass.

  This runs jit_compile functions through all of the multidevice function
  rewrite passes.
  """
  global _JIT_COMPILE_REWRITE_ENABLED
  _JIT_COMPILE_REWRITE_ENABLED = True
  if context_safe() is not None:
    context_safe().jit_compile_rewrite = True


# This method should only be called after the context has been initialized.
def disable_jit_compile_rewrite():
  global _JIT_COMPILE_REWRITE_ENABLED
  _JIT_COMPILE_REWRITE_ENABLED = False
  if context_safe() is not None:
    context_safe().jit_compile_rewrite = False


def jit_compile_rewrite_enabled():
  if context_safe() is not None:
    return context_safe().jit_compile_rewrite
  return _JIT_COMPILE_REWRITE_ENABLED


# Expose it as internally public APIs for Keras use cases in b/171080602.
tf_export("__internal__.is_tfrt_enabled", v1=[])(is_tfrt_enabled)


class _EagerTensorCache(object):
  """Simple cache which evicts items based on length in a FIFO manner."""

  __slots__ = ["_data", "_max_items", "_max_tensor_size"]

  def __init__(self, max_items=256, max_tensor_size=10000):
    self._data = collections.OrderedDict()
    self._max_items = max_items
    self._max_tensor_size = max_tensor_size

  def put(self, key, value):
    if value._num_elements() > self._max_tensor_size:  # pylint: disable=protected-access
      return

    self._data[key] = value

    if len(self._data) > self._max_items:
      self._data.popitem(last=False)

  def get(self, key):
    return self._data.get(key, None)

  def flush(self):
    self._data.clear()


class FunctionCallOptions(object):
  """Options applied at call sites of eager functions.

  Eager functions are functions decorated with tf.contrib.eager.defun.
  """

  __slots__ = ["_config_proto_serialized", "_executor_type"]

  def __init__(self, executor_type=None, config_proto=None):
    """Constructor.

    Args:
      executor_type: (optional) name of the executor to be used to execute the
        eager function. If None or an empty string, the default Tensorflow
        executor will be used.
      config_proto: (optional) a `config_pb2.ConfigProto` proto or a serialized
        string of that proto. The config used by Grappler when optimizing the
        function graph. Each concrete function is optimized the first time is
        called. Changing config_proto after the first call has no effect. If
        config_proto is None, an empty RewriterConfig will be used.
    """
    self.config_proto_serialized = config_proto
    self.executor_type = executor_type

  @property
  def executor_type(self):
    return self._executor_type

  @executor_type.setter
  def executor_type(self, executor_type):
    self._executor_type = executor_type

  @property
  def config_proto_serialized(self):
    return self._config_proto_serialized

  @config_proto_serialized.setter
  def config_proto_serialized(self, config):
    if isinstance(config, config_pb2.ConfigProto):
      self._config_proto_serialized = config.SerializeToString(
          deterministic=True)
    elif isinstance(config, str):
      self._config_proto_serialized = config
    elif config is None:
      self._config_proto_serialized = (
          config_pb2.ConfigProto().SerializeToString())
    else:
      raise ValueError("the rewriter config must be either a "
                       "config_pb2.ConfigProto, or a serialized string of that "
                       "proto or None. got: {}".format(type(config)))


# Map from context_id (an int) to _TensorCaches.
# Dicts are thread safe in CPython.
# TODO(iga): Remove this once TensorCaches are moved to C++.
_tensor_caches_map = {}


class _TensorCaches(threading.local):
  """Thread local tensor caches."""

  __slots__ = ["_ones_rank_cache", "_zeros_cache"]

  def __init__(self):
    super(_TensorCaches, self).__init__()
    self._ones_rank_cache = None
    self._zeros_cache = None

  @property
  def ones_rank_cache(self):
    if not self._ones_rank_cache:
      self._ones_rank_cache = _EagerTensorCache()
    return self._ones_rank_cache

  @property
  def zeros_cache(self):
    if not self._zeros_cache:
      self._zeros_cache = _EagerTensorCache()
    return self._zeros_cache


ContextSwitch = collections.namedtuple(
    "ContextSwitch",
    ["is_building_function", "enter_context_fn", "device_stack"])


# `_ContextSwitchStack` is a `threading.local` to match the semantics of
# ``DefaultGraphStack`, which is also a `threading.local`.
class _ContextSwitchStack(threading.local):
  """A thread-local stack of context switches."""

  def __init__(self, eager):
    super(_ContextSwitchStack, self).__init__()
    self.stack = []
    if eager:
      # Initialize the stack with a pointer to enter the eager context; this
      # ensures that the fact that eager execution was enabled is propagated
      # across threads, since (1) `enable_eager_execution` modifies a
      # process-level flag (`default_execution_mode`) and (2) `__init__` is
      # called each time a threading.local object is used in a separate thread.
      self.push(
          is_building_function=False,
          enter_context_fn=eager_mode,
          device_stack=None)

  def push(self, is_building_function, enter_context_fn, device_stack):
    """Push metadata about a context switch onto the stack.

    A context switch can take any one of the two forms: installing a graph as
    the default graph, or entering the eager context. For each context switch,
    we record whether or not the entered context is building a function.

    Args:
      is_building_function: (bool.) Whether the context is building a function.
      enter_context_fn: (function.) A callable that executes the context switch.
        For example, `graph.as_default` or `eager_mode`.
      device_stack: If applicable, the device function stack for this graph.
        When breaking out of graphs in init_scope, the innermost nonempty device
        stack is used. Eager contexts put `None` here and the value is never
        used.
    """

    self.stack.append(
        ContextSwitch(is_building_function, enter_context_fn, device_stack))

  def pop(self):
    """Pop the stack."""

    self.stack.pop()


@tf_export("config.LogicalDevice")
class LogicalDevice(
    collections.namedtuple("LogicalDevice", ["name", "device_type"])):
  """Abstraction for a logical device initialized by the runtime.

  A `tf.config.LogicalDevice` corresponds to an initialized logical device on a
  `tf.config.PhysicalDevice` or a remote device visible to the cluster. Tensors
  and operations can be placed on a specific logical device by calling
  `tf.device` with a specified `tf.config.LogicalDevice`.

  Fields:
    name: The fully qualified name of the device. Can be used for Op or function
      placement.
    device_type: String declaring the type of device such as "CPU" or "GPU".
  """
  pass


@tf_export("config.LogicalDeviceConfiguration",
           "config.experimental.VirtualDeviceConfiguration")
class LogicalDeviceConfiguration(
    collections.namedtuple("LogicalDeviceConfiguration",
                           ["memory_limit", "experimental_priority"])):
  """Configuration class for a logical devices.

  The class specifies the parameters to configure a `tf.config.PhysicalDevice`
  as it is initialized to a `tf.config.LogicalDevice` during runtime
  initialization. Not all fields are valid for all device types.

  See `tf.config.get_logical_device_configuration` and
  `tf.config.set_logical_device_configuration` for usage examples.

  Fields:
    memory_limit: (optional) Maximum memory (in MB) to allocate on the virtual
      device. Currently only supported for GPUs.
    experimental_priority: (optional) Priority to assign to a virtual device.
      Lower values have higher priorities and 0 is the default.
      Within a physical GPU, the GPU scheduler will prioritize ops on virtual
      devices with higher priority. Currently only supported for Nvidia GPUs.
  """

  def __new__(cls, memory_limit=None, experimental_priority=None):
    return super(LogicalDeviceConfiguration,
                 cls).__new__(cls, memory_limit, experimental_priority)


@tf_export("config.PhysicalDevice")
class PhysicalDevice(
    collections.namedtuple("PhysicalDevice", ["name", "device_type"])):
  """Abstraction for a locally visible physical device.

  TensorFlow can utilize various devices such as the CPU or multiple GPUs
  for computation. Before initializing a local device for use, the user can
  customize certain properties of the device such as it's visibility or memory
  configuration.

  Once a visible `tf.config.PhysicalDevice` is initialized one or more
  `tf.config.LogicalDevice` objects are created. Use
  `tf.config.set_visible_devices` to configure the visibility of a physical
  device and `tf.config.set_logical_device_configuration` to configure multiple
  `tf.config.LogicalDevice` objects for a `tf.config.PhysicalDevice`. This is
  useful when separation between models is needed or to simulate a multi-device
  environment.

  Fields:
    name: Unique identifier for device.
    device_type: String declaring the type of device such as "CPU" or "GPU".
  """
  pass


class _AtomicCounter(object):
  """A simple atomic counter."""

  __slots__ = ["_value", "_lock"]

  def __init__(self):
    self._value = 0
    self._lock = threading.Lock()

  def increment_and_get(self):
    with self._lock:
      self._value += 1
      return self._value


_context_id_counter = _AtomicCounter()


class _TensorCacheDeleter(object):
  """Deletes tensor caches for a given context."""

  __slots__ = ["_context_id"]

  def __init__(self, context_id):
    self._context_id = context_id

  def __del__(self):
    if _tensor_caches_map is None:
      return
    if self._context_id in _tensor_caches_map:
      del _tensor_caches_map[self._context_id]


# TODO(agarwal): rename to EagerContext / EagerRuntime ?
# TODO(agarwal): consider keeping the corresponding Graph here.
class Context(object):
  """Environment in which eager operations execute."""

  # TODO(agarwal): create and link in some documentation for `execution_mode`.
  # pylint: disable=redefined-outer-name
  def __init__(self,
               config=None,
               device_policy=None,
               execution_mode=None,
               server_def=None):
    """Creates a new Context.

    Args:
      config: (Optional.) A `ConfigProto` protocol buffer with configuration
        options for the Context. Note that a lot of these options may be
        currently unimplemented or irrelevant when eager execution is enabled.
      device_policy: (Optional.) What policy to use when trying to run an
        operation on a device with inputs which are not on that device. When set
        to None, an appropriate value will be picked automatically. The value
        picked may change between TensorFlow releases.  Defaults to
        DEVICE_PLACEMENT_SILENT.
        Valid values:
        - DEVICE_PLACEMENT_EXPLICIT: raises an error if the placement is not
          correct.
        - DEVICE_PLACEMENT_WARN: copies the tensors which are not on the right
          device but raises a warning.
        - DEVICE_PLACEMENT_SILENT: silently copies the tensors. This might hide
          performance problems.
        - DEVICE_PLACEMENT_SILENT_FOR_INT32: silently copies int32 tensors,
          raising errors on the other ones.
      execution_mode: (Optional.) Policy controlling how operations dispatched
        are actually executed. When set to None, an appropriate value will be
        picked automatically. The value picked may change between TensorFlow
        releases.
        Valid values:
        - SYNC: executes each operation synchronously.
        - ASYNC: executes each operation asynchronously. These operations may
          return "non-ready" handles.
      server_def: (Optional.) A tensorflow::ServerDef proto. Enables execution
        on remote devices. GrpcServers need to be started by creating an
        identical server_def to this, and setting the appropriate task_indexes,
        so that the servers can communicate. It will then be possible to execute
        operations on remote devices.

    Raises:
     ValueError: If execution_mode is not valid.
    """
    # This _id is used only to index the tensor caches.
    # TODO(iga): Remove this when tensor caches are moved to C++.
    self._id = _context_id_counter.increment_and_get()
    self._tensor_cache_deleter = _TensorCacheDeleter(self._id)
    _tensor_caches_map[self._id] = _TensorCaches()

    self._config = config
    self._thread_local_data = pywrap_tfe.EagerContextThreadLocalData(
        self,
        is_eager=lambda: default_execution_mode == EAGER_MODE,
        device_spec=_starting_device_spec)
    self._context_switches = _ContextSwitchStack(self.executing_eagerly())
    self._context_handle = None
    self._context_devices = None
    self._seed = None
    self._initialize_lock = threading.Lock()
    self._initialized = False
    if device_policy is None:
      device_policy = DEVICE_PLACEMENT_SILENT
    self._device_policy = device_policy
    self._mirroring_policy = None
    if execution_mode not in (None, SYNC, ASYNC):
      raise ValueError("execution_mode should be None/SYNC/ASYNC. Got %s" %
                       execution_mode)
    if execution_mode is None:
      execution_mode = SYNC
    self._default_is_async = execution_mode == ASYNC
    self._use_tfrt = is_tfrt_enabled()
    self._use_tfrt_distributed_runtime = None
    self._run_eager_op_as_function = run_eager_op_as_function_enabled()
    self._jit_compile_rewrite = jit_compile_rewrite_enabled()
    self._server_def = server_def
    self._collective_ops_server_def = None
    self._collective_leader = None
    self._collective_scoped_allocator_enabled_ops = None
    self._collective_use_nccl_communication = None
    self._collective_device_filters = None
    self._coordination_service_config = None

    self._device_lock = threading.Lock()
    self._physical_devices = None
    self._physical_device_to_index = None
    self._pluggable_devices = None
    self._visible_device_list = []
    self._memory_growth_map = None
    self._virtual_device_map = {}

    # Values set after construction
    self._optimizer_jit = None
    self._intra_op_parallelism_threads = None
    self._inter_op_parallelism_threads = None
    self._soft_device_placement = None
    self._log_device_placement = None
    self._operation_timeout_in_ms = None
    self._enable_mlir_graph_optimization = None
    self._optimizer_experimental_options = {}

    _python_eager_context_create_counter.get_cell().increase_by(1)

    self._is_global_context = False

  # pylint: enable=redefined-outer-name

  def _set_global_seed(self, seed):
    """Set a global eager mode seed for random ops."""
    self._seed = seed
    # `random.Random(seed)` needs `seed` to be hashable, while values of type
    # e.g. `np.int64` or `np.ndarray` are not. We use `int(...)` to convert them
    # to int.
    try:
      hash(seed)
    except TypeError:
      seed = int(np.array(seed))
    self._rng = random.Random(seed)
    # Also clear the kernel cache, to reset any existing seeds
    if self._context_handle is not None:
      pywrap_tfe.TFE_ContextClearCaches(self._context_handle)

  def _internal_operation_seed(self):
    """Returns a fake operation seed.

      In eager mode, user shouldn't set or depend on operation seed.
      Here, we generate a random seed based on global seed to make
      operation's randomness different and depend on the global seed.

    Returns:
      A fake operation seed based on global seed.
    """
    return self._rng.randint(0, _MAXINT32)

  def _initialize_logical_devices(self):
    """Helper to initialize devices."""
    # Store list of devices
    logical_devices = []
    context_devices = []
    device_list = pywrap_tfe.TFE_ContextListDevices(self._context_handle)
    try:
      self._num_gpus = 0
      current_job, current_task = None, None
      server_def = self._server_def or self._collective_ops_server_def
      if server_def is not None:
        current_job, current_task = server_def.job_name, server_def.task_index
      for i in range(pywrap_tfe.TF_DeviceListCount(device_list)):
        dev_name = pywrap_tfe.TF_DeviceListName(device_list, i)
        context_devices.append(pydev.canonical_name(dev_name))
        spec = pydev.DeviceSpec.from_string(dev_name)
        # If the job is localhost, we assume that the cluster has not yet been
        # configured and thus clear the job, replica & task.
        if spec.job == "localhost":
          spec = spec.replace(job=None, replica=None, task=None)
        logical_devices.append(
            LogicalDevice(name=spec.to_string(), device_type=spec.device_type))
        dev_type = pywrap_tfe.TF_DeviceListType(device_list, i)
        if (dev_type == "GPU" and spec.job == current_job and
            spec.task == current_task):
          self._num_gpus += 1

    finally:
      self._logical_devices = logical_devices
      self._context_devices = context_devices
      pywrap_tfe.TF_DeleteDeviceList(device_list)

  def ensure_initialized(self):
    """Initialize handle and devices if not already done so."""
    if self._initialized:
      return
    with self._initialize_lock:
      if self._initialized:
        return
      assert self._context_devices is None
      opts = pywrap_tfe.TFE_NewContextOptions()
      try:
        config_str = self.config.SerializeToString()
        pywrap_tfe.TFE_ContextOptionsSetConfig(opts, config_str)
        if self._device_policy is not None:
          pywrap_tfe.TFE_ContextOptionsSetDevicePlacementPolicy(
              opts, self._device_policy)
        if self._mirroring_policy is not None:
          pywrap_tfe.TFE_ContextOptionsSetMirroringPolicy(
              opts, self._mirroring_policy)
        if self._default_is_async == ASYNC:
          pywrap_tfe.TFE_ContextOptionsSetAsync(opts, True)
        if self._use_tfrt is not None:
          pywrap_tfe.TFE_ContextOptionsSetTfrt(opts, self._use_tfrt)
        # pylint: disable=g-backslash-continuation
        if self._use_tfrt is not None and \
            self._use_tfrt_distributed_runtime is not None:
          pywrap_tfe.TFE_ContextOptionsSetTfrtDistributedRuntime(
              opts, self._use_tfrt_distributed_runtime)
        pywrap_tfe.TFE_ContextOptionsSetRunEagerOpAsFunction(
            opts, self._run_eager_op_as_function)
        pywrap_tfe.TFE_ContextOptionsSetJitCompileRewrite(
            opts, self._jit_compile_rewrite)
        context_handle = pywrap_tfe.TFE_NewContext(opts)
      finally:
        pywrap_tfe.TFE_DeleteContextOptions(opts)
      assert not (self._server_def and self._collective_ops_server_def), (
          "Cannot enable remote execution as well as collective ops at the "
          "moment. If this is important to you, please file an issue.")
      if self._server_def is not None:
        server_def_str = self._server_def.SerializeToString()
        pywrap_tfe.TFE_ContextSetServerDef(context_handle, _KEEP_ALIVE_SECS,
                                           server_def_str)
      elif self._collective_ops_server_def is not None:
        server_def_str = self._collective_ops_server_def.SerializeToString()
        pywrap_tfe.TFE_EnableCollectiveOps(context_handle, server_def_str)

      self._context_handle = context_handle
      self._initialize_logical_devices()
      self._initialized = True

      if self._is_global_context:
        pywrap_tfe.TFE_Py_SetCEagerContext(self._context_handle)

  def mark_as_global_context(self):
    # If the context was already initialized, publish it. Otherwise wait with
    # publication until it's initialized.
    if self._initialized:
      pywrap_tfe.TFE_Py_SetCEagerContext(self._context_handle)
    self._is_global_context = True

  def _clear_caches(self):
    self.ones_rank_cache().flush()
    self.zeros_cache().flush()
    pywrap_tfe.TFE_ClearScalarCache()

  def get_server_def(self):
    return self._server_def

  def set_server_def(self, server_def, keep_alive_secs=_KEEP_ALIVE_SECS):
    """Allow setting a server_def on the context.

    When a server def is replaced, it effectively clears a bunch of caches
    within the context. If you attempt to use a tensor object that was pointing
    to a tensor on the remote device, it will raise an error.

    Args:
      server_def: A tensorflow::ServerDef proto. Enables execution on remote
        devices.
      keep_alive_secs: Num. seconds after which the remote end will hang up. As
        long as the client is still alive, the server state for the context will
        be kept alive. If the client is killed (or there is some failure), the
        server will clean up its context keep_alive_secs after the final RPC it
        receives.

    Raises:
      ValueError: if server_def is None.
    """
    if not server_def:
      raise ValueError("server_def is None.")

    self._server_def = server_def

    if self._context_handle:
      server_def_str = server_def.SerializeToString()
      pywrap_tfe.TFE_ContextSetServerDef(self._context_handle, keep_alive_secs,
                                         server_def_str)
      self._initialize_logical_devices()

    # Clear all the caches in case there are remote tensors in them.
    self._clear_caches()

  def update_server_def(self, server_def, keep_alive_secs=_KEEP_ALIVE_SECS):
    """Update a server_def on the context.

    Args:
      server_def: A tensorflow::ServerDef proto. Enables execution on remote
        devices.
      keep_alive_secs: Num. seconds after which the remote end will hang up. As
        long as the client is still alive, the server state for the context will
        be kept alive. If the client is killed (or there is some failure), the
        server will clean up its context keep_alive_secs after the final RPC it
        receives.

    Raises:
      ValueError: if server_def is None.
    """
    if not server_def:
      raise ValueError("server_def is None.")

    self._server_def = server_def

    if self._context_handle:
      server_def_str = server_def.SerializeToString()
      pywrap_tfe.TFE_ContextUpdateServerDef(self._context_handle,
                                            keep_alive_secs, server_def_str)
      self._initialize_logical_devices()

    self._clear_caches()

  def check_alive(self, worker_name):
    """Checks whether a remote worker is alive or not.

    Args:
      worker_name: a string representing the remote worker. It must be a fully
      specified name like "/job:worker/replica:0/task:0".

    Returns:
      a boolean indicating whether the remote worker is alive or not.

    Raises:
      ValueError: if context is not initialized.
    """
    # TODO(yuefengz): support checking multiple workers.
    if self._context_handle:
      return pywrap_tfe.TFE_ContextCheckAlive(self._context_handle, worker_name)
    else:
      raise ValueError("Context is not initialized.")

  def sync_executors(self):
    """Sync both local executors and the ones on remote workers.

    In async execution mode, local function calls can return before the
    corresponding remote op/function execution requests are completed. Calling
    this method creates a synchronization barrier for remote executors. It only
    returns when all remote pending nodes are finished, potentially with errors
    if any remote executors are in error state.

    Raises:
      ValueError: if context is not initialized.
    """
    if self._context_handle:
      pywrap_tfe.TFE_ContextSyncExecutors(self._context_handle)
    else:
      raise ValueError("Context is not initialized.")

  def clear_executor_errors(self):
    """Clear errors in both local executors and remote workers.

    After receiving errors from remote workers, additional requests on the fly
    could further taint the status on the remote workers due to the async nature
    of remote execution. Calling this method block on waiting for all pending
    nodes in remote executors to finish and clear their error statuses.

    Raises:
      ValueError: if context is not initialized.
    """
    if self._context_handle:
      pywrap_tfe.TFE_ContextClearExecutors(self._context_handle)
    else:
      raise ValueError("Context is not initialized.")

  def configure_coordination_service(self,
                                     service_type,
                                     service_leader="",
                                     enable_health_check=True,
                                     cluster_register_timeout_in_ms=0,
                                     heartbeat_timeout_in_ms=0,
                                     coordinated_jobs=None):
    """Enable distributed coordination service with specified configs."""
    if self._context_handle:
      logging.warning("Configuring coordination service type may not be "
                      "effective because the context is already initialized.")
    config = coordination_config_pb2.CoordinationServiceConfig()
    config.service_type = service_type
    if service_leader:
      config.service_leader = pydev.canonical_name(service_leader)
    config.enable_health_check = enable_health_check
    config.cluster_register_timeout_in_ms = cluster_register_timeout_in_ms
    config.heartbeat_timeout_in_ms = heartbeat_timeout_in_ms
    if coordinated_jobs is not None:
      if isinstance(coordinated_jobs, list):
        config.coordinated_jobs.extend(coordinated_jobs)
      else:
        raise ValueError("`coordinated_jobs` must be a list of job names or "
                         "None, but got: %s" % (coordinated_jobs,))
    self._coordination_service_config = config

  @property
  def coordination_service(self):
    return self._coordination_service_config

  def set_config_key_value(self, key, value):
    ensure_initialized()
    pywrap_tfe.TFE_InsertConfigKeyValue(self._context_handle, key, value)

  def get_config_key_value(self, key):
    ensure_initialized()
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_GetConfigKeyValue(self._context_handle, key, buffer_)
      value = pywrap_tf_session.TF_GetBuffer(buffer_).decode("utf-8")
    return value

  def delete_config_key_value(self, key):
    ensure_initialized()
    pywrap_tfe.TFE_DeleteConfigKeyValue(self._context_handle, key)

  def report_error_to_cluster(self, error_code, error_message):
    """Report error to other members in a multi-client cluster.

    Args:
      error_code: a `tf.errors` error code.
      error_message: a string. The error message.
    """
    if self._context_handle:
      pywrap_tfe.TFE_ReportErrorToCluster(self._context_handle, error_code,
                                          error_message)
    else:
      raise ValueError("Context is not initialized.")

  def clear_kernel_cache(self):
    """Clear kernel cache and reset all stateful kernels."""
    if self._context_handle is not None:
      pywrap_tfe.TFE_ContextClearCaches(self._context_handle)

  def enable_collective_ops(self, server_def):
    """Enable distributed collective ops with an appropriate server_def.

    Args:
      server_def: A tensorflow::ServerDef proto. Enables execution on remote
        devices.

    Raises:
      ValueError: if server_def is None.
      RuntimeError: if this method is not called at program startup.
    """
    if not server_def:
      raise ValueError("server_def is None.")

    self._collective_ops_server_def = server_def

    # TODO(b/129298253): Allow creating datasets/tensors before enabling
    # collective ops.
    if self._context_handle is not None:
      logging.warning("Enabling collective ops after program startup may cause "
                      "error when accessing previously created tensors.")
      with self._initialize_lock:
        assert self._initialized
        server_def_str = self._collective_ops_server_def.SerializeToString()
        pywrap_tfe.TFE_EnableCollectiveOps(self._context_handle, server_def_str)
        self._initialize_logical_devices()
        self._clear_caches()

  def configure_collective_ops(
      self,
      collective_leader="",
      scoped_allocator_enabled_ops=("CollectiveReduce",),
      use_nccl_communication=False,
      device_filters=None):
    """Configure collective ops.

      Collective group leader is necessary for collective ops to run, other
      configurations are mainly for the purpose of performance.

    Args:
      collective_leader: a device string for collective leader, e.g.
        "/job:worker/replica:0/task:0"; empty string means local execution of
          collective ops.
      scoped_allocator_enabled_ops: a tuple or a list of op names for scoped
        allocator to run with.
      use_nccl_communication: whether to use nccl communication for collective
        ops.
      device_filters: a tuple or a list of device strings. If set, corresponding
        task can only see the devices filtered by these device filters.

    Raises:
      RuntimeError: if this method is not called at program startup.
    """
    if self._collective_leader is not None:
      if (self._collective_leader != collective_leader or
          self._collective_scoped_allocator_enabled_ops !=
          scoped_allocator_enabled_ops or
          self._collective_use_nccl_communication != use_nccl_communication or
          self._collective_device_filters != device_filters):
        raise ValueError("Collective ops are already configured.")
      else:
        return

    if self._context_handle is not None:
      raise RuntimeError("Collective ops must be configured at program startup")

    self._collective_leader = collective_leader
    self._collective_scoped_allocator_enabled_ops = scoped_allocator_enabled_ops
    self._collective_use_nccl_communication = use_nccl_communication
    self._collective_device_filters = device_filters

  def abort_collective_ops(self, code, message):
    """Abort the collective ops.

    This is intended to be used when a peer failure is detected, which allows
    the user to handle the case instead of hanging. This aborts all on-going
    collectives. After all subsequent collectives error immediately, and you
    need to reset_context() to use collectives again.

    Args:
      code: a `tf.errors` error code.
      message: a string. The error message.
    """
    self.ensure_initialized()
    pywrap_tfe.TFE_AbortCollectiveOps(self._handle, code, message)

  def check_collective_ops_peer_health(self, task, timeout_in_ms):
    """Check collective peer health.

    This probes each task to see if they're still alive. Note that restarted
    tasks are considered a different one, and they're considered not healthy.

    This should only be used in multi client multi worker training.

    Args:
      task: a task string, must be in the format of /job:xxx/replica:0/task:N.
      timeout_in_ms: an integer, the timeout. If zero, there's no timeout.

    Raises:
      tf.errors.UnavailableError: when a peer is down.
      tf.errors.FailedPreconditionError: when a peer is a different one from the
        one this task has talked to, e.g. the peer has restarted.
      tf.errors.InvalidArgumentError: when the task string is invalid.
    """
    self.ensure_initialized()
    pywrap_tfe.TFE_CollectiveOpsCheckPeerHealth(self._handle, task,
                                                timeout_in_ms)

  @property
  def _handle(self):
    if self._context_handle is None:
      raise AssertionError("Context must be initialized first.")

    return self._context_handle

  @property
  def _devices(self):
    if self._context_devices is None:
      raise AssertionError("Context must be initialized first.")

    return self._context_devices

  def __str__(self):
    if self._context_handle is None:
      return "Eager TensorFlow Context. Devices currently uninitialized."
    else:
      devices = self._devices
      lines = ["Eager TensorFlow Context with %d devices" % (len(devices))]
      for i, d in enumerate(devices):
        lines.append("   Device %d: %s" % (i, d))
      return "\n".join(lines)

  @tf_contextlib.contextmanager
  def _mode(self, mode):
    """A context manager to allow setting the mode to EAGER/GRAPH."""
    ctx = self._thread_local_data
    old_is_eager = ctx.is_eager
    ctx.is_eager = mode == EAGER_MODE
    if mode == EAGER_MODE:
      # Entering graph mode does not provide us with sufficient information to
      # record a context switch; graph-based context switches are only logged
      # when a graph is registered as the default graph.
      self.context_switches.push(False, eager_mode, None)
    try:
      yield
    finally:
      ctx.is_eager = old_is_eager
      if mode == EAGER_MODE:
        self.context_switches.pop()

  def executing_eagerly(self):
    """Returns True if current thread has eager executing enabled."""
    return self._thread_local_data.is_eager

  def ones_rank_cache(self):
    """Per-device cache for scalars."""
    return _tensor_caches_map[self._id].ones_rank_cache

  def zeros_cache(self):
    """Per-device cache for scalars."""
    return _tensor_caches_map[self._id].zeros_cache

  @property
  def scope_name(self):
    """Returns scope name for the current thread."""
    return self._thread_local_data.scope_name

  @scope_name.setter
  def scope_name(self, s):
    """Sets scope name for the current thread."""
    self._thread_local_data.scope_name = s

  @property
  def device_name(self):
    """Returns the device name for the current thread."""
    return self._thread_local_data.device_name

  @property
  def device_spec(self):
    """Returns the device spec for the current thread."""
    return self._thread_local_data.device_spec

  def _set_device(self, device_name, device_spec):
    self._thread_local_data.device_name = device_name
    self._thread_local_data.device_spec = device_spec

  def device(self, name):
    """Context-manager to force placement of operations and Tensors on a device.

    Args:
      name: Name of the device or None to get default placement.

    Returns:
      Context manager that forces device placement.

    Raises:
      ValueError: If name is not a string or is an invalid device name.
      RuntimeError: If device scopes are not properly nested.
    """
    if isinstance(name, LogicalDevice):
      name = name.name
    elif pydev.is_device_spec(name):
      name = name.to_string()
    return _EagerDeviceContext(self, name)

  def devices(self):
    """List of the names of devices available to execute operations."""
    return self._devices

  def host_address_space(self):
    self.ensure_initialized()
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_HostAddressSpace(self._context_handle, buffer_)
      address_space = pywrap_tf_session.TF_GetBuffer(buffer_).decode("utf-8")
    return address_space

  # TODO(fishx): remove this property.
  @property
  def execution_mode(self):
    """Gets execution mode for current thread."""
    return ASYNC if self.is_async() else SYNC

  @execution_mode.setter
  def execution_mode(self, mode):
    """Sets execution mode for current thread."""
    if mode not in (None, SYNC, ASYNC):
      raise ValueError("Execution mode should be None/SYNC/ASYNC. Got %s" %
                       mode)

    if mode is None:
      mode = SYNC

    enable_async = (mode == ASYNC)
    if self.is_async() != enable_async:
      # Only set the execution mode if the context has already been initialized
      if self._context_handle is not None:
        self.executor.wait()
        executor_new = executor.new_executor(enable_async)
        self._thread_local_data.executor = executor_new
        pywrap_tfe.TFE_ContextSetExecutorForThread(self._context_handle,
                                                   executor_new.handle())
      else:
        self._default_is_async = enable_async

  def is_async(self):
    if self._context_handle is not None:
      return self.executor.is_async()
    else:
      return self._default_is_async

  @property
  def executor(self):
    self.ensure_initialized()
    return executor.Executor(
        pywrap_tfe.TFE_ContextGetExecutorForThread(self._context_handle))

  @executor.setter
  def executor(self, e):
    self.ensure_initialized()
    pywrap_tfe.TFE_ContextSetExecutorForThread(self._context_handle, e.handle())

  @property
  def config(self):
    """Return the ConfigProto with all runtime deltas applied."""
    # Ensure physical devices have been discovered and config has been imported
    self._initialize_physical_devices()

    config = config_pb2.ConfigProto()
    if self._config is not None:
      config.CopyFrom(self._config)

    if self._optimizer_jit is not None:
      config.graph_options.optimizer_options.global_jit_level = (
          config_pb2.OptimizerOptions.ON_1
          if self._optimizer_jit else config_pb2.OptimizerOptions.OFF)
    if self._intra_op_parallelism_threads is not None:
      config.intra_op_parallelism_threads = self._intra_op_parallelism_threads
    if self._inter_op_parallelism_threads is not None:
      config.inter_op_parallelism_threads = self._inter_op_parallelism_threads

    if self._soft_device_placement is not None:
      config.allow_soft_placement = self._soft_device_placement
    else:
      config.allow_soft_placement = self.executing_eagerly()

    if self._log_device_placement is not None:
      config.log_device_placement = self._log_device_placement

    if self._operation_timeout_in_ms is not None:
      config.operation_timeout_in_ms = self._operation_timeout_in_ms

    is_mlir_bridge_enabled = pywrap_tfe.TF_IsMlirBridgeEnabled()
    config.experimental.mlir_bridge_rollout = is_mlir_bridge_enabled
    if (is_mlir_bridge_enabled ==
        config_pb2.ConfigProto.Experimental.MLIR_BRIDGE_ROLLOUT_ENABLED):
      config.experimental.enable_mlir_bridge = True

    if self._enable_mlir_graph_optimization is not None:
      config.experimental.enable_mlir_graph_optimization = (
          self._enable_mlir_graph_optimization)

    def rewriter_toggle(option):
      toggle = self._optimizer_experimental_options.get(option, None)
      if toggle is None:
        return

      setattr(config.graph_options.rewrite_options, option,
              (rewriter_config_pb2.RewriterConfig.ON
               if toggle else rewriter_config_pb2.RewriterConfig.OFF))

    def rewriter_bool(option):
      toggle = self._optimizer_experimental_options.get(option, None)
      if toggle is None:
        return

      setattr(config.graph_options.rewrite_options, option, toggle)

    rewriter_toggle("layout_optimizer")
    rewriter_toggle("constant_folding")
    rewriter_toggle("shape_optimization")
    rewriter_toggle("remapping")
    rewriter_toggle("arithmetic_optimization")
    rewriter_toggle("dependency_optimization")
    rewriter_toggle("loop_optimization")
    rewriter_toggle("function_optimization")
    rewriter_toggle("debug_stripper")
    rewriter_bool("disable_model_pruning")
    rewriter_toggle("scoped_allocator_optimization")
    rewriter_toggle("pin_to_host_optimization")
    rewriter_toggle("implementation_selector")
    rewriter_toggle("auto_mixed_precision")
    rewriter_toggle("use_plugin_optimizers")
    rewriter_bool("disable_meta_optimizer")
    rewriter_toggle("auto_mixed_precision_mkl")
    nodes = self._optimizer_experimental_options.get("min_graph_nodes", None)
    if nodes is not None:
      config.graph_options.rewrite_options.min_graph_nodes = nodes

    # Compute device counts
    config.device_count["CPU"] = 0
    config.device_count["GPU"] = 0
    for dev in self._physical_devices:
      if dev not in self._visible_device_list:
        continue

      virtual_devices = self._virtual_device_map.get(dev)
      if virtual_devices is None:
        config.device_count[dev.device_type] += 1
      else:
        config.device_count[dev.device_type] += len(virtual_devices)

    # Configure gpu_options
    gpu_options = self._compute_gpu_options()
    config.gpu_options.MergeFrom(gpu_options)

    # Configure collective ops
    if self._collective_leader:
      config.experimental.collective_group_leader = self._collective_leader
    if self._collective_scoped_allocator_enabled_ops:
      rewrite_options = config.graph_options.rewrite_options
      rewrite_options.scoped_allocator_optimization = (
          rewriter_config_pb2.RewriterConfig.ON)
      del rewrite_options.scoped_allocator_opts.enable_op[:]
      for op in self._collective_scoped_allocator_enabled_ops:
        rewrite_options.scoped_allocator_opts.enable_op.append(op)
    if self._collective_use_nccl_communication:
      config.experimental.collective_nccl = True
    if self._collective_device_filters:
      del config.device_filters[:]
      for f in self._collective_device_filters:
        config.device_filters.append(f)

    # Configure coordination service
    if self._coordination_service_config:
      config.experimental.coordination_config.CopyFrom(
          self._coordination_service_config)

    return config

  def _compute_gpu_options(self):
    """Build the GPUOptions proto."""
    visible_device_list = []
    virtual_devices = []
    gpu_index = -1
    memory_growths = set()
    gpu_devices = self.list_physical_devices("GPU")
    pluggable_devices = self._pluggable_devices
    compatible_devices = gpu_devices
    for dev in pluggable_devices:
      if dev not in gpu_devices:
        compatible_devices.append(dev)
    for dev in compatible_devices:
      gpu_index += 1

      if dev not in self._visible_device_list:
        continue

      growth = self._memory_growth_map[dev]
      memory_growths.add(growth)
      visible_device_list.append(str(gpu_index))

      if self._virtual_device_map:
        vdevs = self._virtual_device_map.get(dev, [])
        device_limits = []
        priority = []
        for virt_dev in vdevs:
          device_limits.append(virt_dev.memory_limit)
          if virt_dev.experimental_priority is not None:
            priority.append(virt_dev.experimental_priority)
        # If priority is specified, it must be specified for all virtual
        # devices.
        if priority and len(device_limits) != len(priority):
          raise ValueError("priority must be specified for all virtual devices")

        virtual_devices.append(
            config_pb2.GPUOptions.Experimental.VirtualDevices(
                memory_limit_mb=device_limits, priority=priority))

    # Only compute growth if virtual devices have not been configured and we
    # have GPUs
    if not virtual_devices and memory_growths:
      if len(memory_growths) > 1:
        raise ValueError("Memory growth cannot differ between GPU devices")
      allow_growth = memory_growths.pop()
    else:
      allow_growth = None

    return config_pb2.GPUOptions(
        allow_growth=allow_growth,
        visible_device_list=",".join(visible_device_list),
        experimental=config_pb2.GPUOptions.Experimental(
            virtual_devices=virtual_devices))

  @property
  def function_call_options(self):
    """Returns function call options for current thread.

    Note that the returned object is still referenced by the eager context.

    Returns: the FunctionCallOptions for current thread.
    """
    if self._thread_local_data.function_call_options is None:
      config = self.config

      # Default to soft placement for functions unless specified
      if self._soft_device_placement is None:
        config.allow_soft_placement = True
      self._thread_local_data.function_call_options = FunctionCallOptions(
          config_proto=config)

    return self._thread_local_data.function_call_options

  @function_call_options.setter
  def function_call_options(self, options):
    """Returns function call options for current thread."""
    self._thread_local_data.function_call_options = options

  def num_gpus(self):
    """The number of GPUs available to execute operations."""
    self.ensure_initialized()
    return self._num_gpus

  def add_function(self, fn):
    """Add a function definition to the context.

    Once added, the function (identified by its name) can be executed like any
    other operation.

    Args:
      fn: A wrapped TF_Function (returned from TF_GraphToFunction_wrapper).
    """
    self.ensure_initialized()
    pywrap_tfe.TFE_ContextAddFunction(self._handle, fn)

  def add_function_def(self, fdef):
    """Add a function definition to the context.

    Once added, the function (identified by its name) can be executed like any
    other operation.

    Args:
      fdef: A FunctionDef protocol buffer message.
    """
    self.ensure_initialized()
    fdef_string = fdef.SerializeToString()
    pywrap_tfe.TFE_ContextAddFunctionDef(self._handle, fdef_string,
                                         len(fdef_string))

  def get_function_def(self, name):
    """Get a function definition from the context.

    Args:
      name: function signature name.

    Returns:
      The requested FunctionDef.

    Raises:
      tf.errors.NotFoundError: if name is not the name of a registered function.
    """
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_ContextGetFunctionDef(self._handle, name, buffer_)
      proto_data = pywrap_tf_session.TF_GetBuffer(buffer_)
    function_def = function_pb2.FunctionDef()
    function_def.ParseFromString(proto_data)

    return function_def

  def register_custom_device(self, device_capsule, device_name,
                             device_info_capsule):
    """Calls TFE_RegisterCustomDevice. See the non-member function."""
    self.ensure_initialized()
    pywrap_tfe.TFE_Py_RegisterCustomDevice(self._handle, device_capsule,
                                           device_name, device_info_capsule)

  def pack_eager_tensors(self, tensors):
    """Pack multiple `EagerTensor`s of the same dtype and shape.

    Args:
      tensors: a list of EagerTensors to pack.

    Returns:
      A packed EagerTensor.
    """
    self.ensure_initialized()
    return pywrap_tfe.TFE_Py_PackEagerTensors(self._handle, tensors)

  def list_function_names(self):
    """Get a list of names of registered functions.

    Returns:
      A set of names of all registered functions for the context.
    """
    self.ensure_initialized()
    return set(pywrap_tfe.TFE_ContextListFunctionNames(self._handle))

  def remove_function(self, name):
    """Remove a function from the context.

    Once removed, the function cannot be executed anymore.

    Args:
      name: function signature name.
    """
    self.ensure_initialized()
    pywrap_tfe.TFE_ContextRemoveFunction(self._handle, name)

  def has_function(self, name):
    """Check if a function `name` is registered."""
    self.ensure_initialized()
    return bool(pywrap_tfe.TFE_ContextHasFunction(self._handle, name))

  def add_op_callback(self, callback):
    """Add a post-op callback to the context.

    A post-op callback is invoked immediately after an eager operation or
    function has finished execution or after a op has been added to a graph,
    providing access to the op's type, name input and output tensors. Multiple
    op callbacks can be added, in which case the callbacks will be invoked in
    the order in which they are added.

    Args:
      callback: a callable of the signature `f(op_type, inputs, attrs, outputs,
        op_name=None, graph=None)`. See doc strings in `op_callbacks.py` for
        details on the function signature and its semantics.
    """
    if callback not in self._thread_local_data.op_callbacks:
      self._thread_local_data.op_callbacks.append(callback)

  def remove_op_callback(self, callback):
    """Remove an already-registered op callback.

    Args:
      callback: The op callback to be removed.

    Raises:
      KeyError: If `callback` is not already registered.
    """
    if callback not in self._thread_local_data.op_callbacks:
      raise KeyError("The specified op callback has not been registered, "
                     "and hence cannot be removed.")
    del self._thread_local_data.op_callbacks[
        self._thread_local_data.op_callbacks.index(callback)]

  @property
  def op_callbacks(self):
    return self._thread_local_data.op_callbacks

  @property
  def invoking_op_callbacks(self):
    return self._thread_local_data.invoking_op_callbacks

  @invoking_op_callbacks.setter
  def invoking_op_callbacks(self, value):
    self._thread_local_data.invoking_op_callbacks = value

  def _initialize_physical_devices(self, reinitialize=False):
    """Gets local devices visible to the system.

    Args:
      reinitialize: If True, reinitializes self._physical_devices  so that
        dynamic registered devices will also be visible to the python front-end.
    """
    # We lazy initialize self._physical_devices since we do not want to do this
    # the constructor since the backend may not be initialized yet.
    with self._device_lock:
      if not reinitialize and self._physical_devices is not None:
        return

      devs = pywrap_tfe.TF_ListPhysicalDevices()
      self._physical_devices = [
          PhysicalDevice(name=d.decode(), device_type=d.decode().split(":")[1])
          for d in devs
      ]
      self._physical_device_to_index = {
          p: i for i, p in enumerate(self._physical_devices)
      }
      # We maintain a seperate list just so we can check whether the device in
      # _physical_devices is a PluggableDevice.
      pluggable_devs = pywrap_tfe.TF_ListPluggablePhysicalDevices()
      self._pluggable_devices = [
          PhysicalDevice(name=d.decode(), device_type=d.decode().split(":")[1])
          for d in pluggable_devs
      ]

      self._visible_device_list = list(self._physical_devices)
      self._memory_growth_map = {
          d: None
          for d in self._physical_devices
          if d.device_type == "GPU" or d in self._pluggable_devices
      }

    # Import device settings that may have been passed into the constructor
    self._import_config()

  def reinitialize_physical_devices(self):
    """Gets local devices visible to the system."""
    # Reinitialize the physical device list after registering
    # the pluggable device.
    self._initialize_physical_devices(True)

  def list_physical_devices(self, device_type=None):
    """List local devices visible to the system.

    This API allows a client to query the devices before they have been
    initialized by the eager runtime. Additionally a user can filter by device
    type, to get only CPUs or GPUs.

    Args:
      device_type: Optional device type to limit results to

    Returns:
      List of PhysicalDevice objects.
    """
    self._initialize_physical_devices()

    if device_type is None:
      return list(self._physical_devices)

    return [d for d in self._physical_devices if d.device_type == device_type]

  def get_device_details(self, device):  # pylint: disable=redefined-outer-name
    """Returns details about a physical devices.

    Args:
      device: A `tf.config.PhysicalDevice` returned by
        `tf.config.list_physical_devices` or `tf.config.get_visible_devices`.

    Returns:
      A dict with string keys.
    """
    if not isinstance(device, PhysicalDevice):
      raise ValueError("device must be a tf.config.PhysicalDevice, but got: "
                       "%s" % (device,))
    if (self._physical_device_to_index is None or
        device not in self._physical_device_to_index):
      raise ValueError("The PhysicalDevice must be one obtained from "
                       "calling `tf.config.list_physical_devices`, but got: "
                       "%s" % (device,))
    index = self._physical_device_to_index[device]
    details = pywrap_tfe.TF_GetDeviceDetails(index)

    # Change compute_capability from a string to a tuple
    if "compute_capability" in details:
      try:
        major, minor = details["compute_capability"].split(".")
        details["compute_capability"] = (int(major), int(minor))
      except ValueError:
        raise RuntimeError("Device returned compute capability an in invalid "
                           "format: %s" % details["compute_capability"])
    return details

  def _import_config(self):
    """Import config if passed in during construction.

    If Context was created with a ConfigProto such as when calling
    tf.compat.v1.enable_eager_execution(), then we need to pull out the
    various pieces we might be replacing and import then into our internal
    class representation.
    """
    if self._config is None:
      return

    num_cpus = self._config.device_count.get("CPU", 1)
    if num_cpus != 1:
      cpus = [d for d in self._physical_devices if d.device_type == "CPU"]
      if num_cpus == 0:
        self.set_visible_devices([], "CPU")
      elif num_cpus > 1:
        self.set_logical_device_configuration(
            cpus[0], [LogicalDeviceConfiguration() for _ in range(num_cpus)])

    # Parse GPU options
    gpus = [d for d in self._physical_devices if d.device_type == "GPU"]

    # If there are no GPUs detected, simply ignore all the GPU options passed in
    # rather than doing any validation checks.
    if not gpus:
      return

    gpu_count = self._config.device_count.get("GPU", None)

    visible_gpus = []
    # TODO(gjn): Handle importing existing virtual GPU configuration
    visible_indices = self._config.gpu_options.visible_device_list
    if visible_indices:
      for index in visible_indices.split(","):
        if int(index) >= len(gpus):
          raise ValueError("Invalid visible device index: %s" % index)
        visible_gpus.append(gpus[int(index)])
    else:
      visible_gpus = gpus

    if gpu_count is not None:
      visible_gpus = visible_gpus[:gpu_count]

    self.set_visible_devices(visible_gpus, "GPU")

  def list_logical_devices(self, device_type=None):
    """Return logical devices."""
    self.ensure_initialized()
    if device_type is None:
      return list(self._logical_devices)

    return [d for d in self._logical_devices if d.device_type == device_type]

  def get_visible_devices(self, device_type=None):
    """Get the list of visible devices."""
    self._initialize_physical_devices()

    if device_type is None:
      return list(self._visible_device_list)

    return [
        d for d in self._visible_device_list if d.device_type == device_type
    ]

  def set_visible_devices(self, devices, device_type=None):
    """Set the list of visible devices."""
    self._initialize_physical_devices()

    if not isinstance(devices, list):
      devices = [devices]

    for d in devices:
      if d not in self._physical_devices:
        raise ValueError("Unrecognized device: %s" % repr(d))
      if device_type is not None and d.device_type != device_type:
        raise ValueError("Unrecognized device: %s" % repr(d))

    visible_device_list = []
    if device_type is not None:
      visible_device_list = [
          d for d in self._visible_device_list if d.device_type != device_type
      ]

    visible_device_list += devices

    if self._visible_device_list == visible_device_list:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Visible devices cannot be modified after being initialized")

    self._visible_device_list = visible_device_list

  def get_memory_info(self, dev):
    """Returns a dict of memory info for the device."""
    self._initialize_physical_devices()
    self.ensure_initialized()
    return pywrap_tfe.TFE_GetMemoryInfo(self._context_handle, dev)

  def reset_memory_stats(self, dev):
    """Resets the tracked memory stats for the device."""
    self._initialize_physical_devices()
    self.ensure_initialized()
    pywrap_tfe.TFE_ResetMemoryStats(self._context_handle, dev)

  def get_memory_growth(self, dev):
    """Get if memory growth is enabled for a PhysicalDevice."""
    self._initialize_physical_devices()

    if dev not in self._physical_devices:
      raise ValueError("Unrecognized device: %s" % repr(dev))

    return self._memory_growth_map[dev]

  def set_memory_growth(self, dev, enable):
    """Set if memory growth should be enabled for a PhysicalDevice."""
    self._initialize_physical_devices()

    if dev not in self._physical_devices:
      raise ValueError("Unrecognized device: %s" % repr(dev))

    if dev in self._virtual_device_map:
      raise ValueError(
          "Cannot set memory growth on device when virtual devices configured")

    if dev.device_type != "GPU" and dev not in self._pluggable_devices:
      raise ValueError(
          "Cannot set memory growth on non-GPU and non-Pluggable devices")

    if self._memory_growth_map.get(dev) == enable:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Physical devices cannot be modified after being initialized")

    self._memory_growth_map[dev] = enable

  def get_logical_device_configuration(self, dev):
    """Get the virtual device configuration for a PhysicalDevice."""
    self._initialize_physical_devices()

    if dev not in self._physical_devices:
      raise ValueError("Unrecognized device: %s" % repr(dev))

    return self._virtual_device_map.get(dev)

  def set_logical_device_configuration(self, dev, virtual_devices):
    """Set the virtual device configuration for a PhysicalDevice."""
    self._initialize_physical_devices()

    if dev not in self._physical_devices:
      raise ValueError("Unrecognized device: %s" % repr(dev))

    if dev.device_type == "CPU":
      for vdev in virtual_devices:
        if vdev.memory_limit is not None:
          raise ValueError("Setting memory limit on CPU virtual devices is "
                           "currently not supported")
        if vdev.experimental_priority is not None:
          raise ValueError("Setting experimental_priority on CPU virtual "
                           " devices is currently not supported")
    elif dev.device_type == "GPU":
      for vdev in virtual_devices:
        if vdev.memory_limit is None:
          raise ValueError(
              "Setting memory limit is required for GPU virtual devices")
    else:
      raise ValueError("Virtual devices are not supported for %s" %
                       dev.device_type)

    if self._virtual_device_map.get(dev) == virtual_devices:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Virtual devices cannot be modified after being initialized")

    self._virtual_device_map[dev] = virtual_devices

  def set_logical_cpu_devices(self, num_cpus, prefix=""):
    """Set virtual CPU devices in context.

    If virtual CPU devices are already configured at context initialization
    by tf.config.set_logical_device_configuration(), this method should not be
    called.

    Args:
      num_cpus: Number of virtual CPUs.
      prefix: Device name prefix.

    Raises:
     RuntimeError: If virtual CPUs are already configured at context
     initialization.
    """
    server_def = self._server_def or self._collective_ops_server_def
    local_prefix = ["/device"]
    if server_def is not None:
      local_prefix.append("/job:%s/replica:0/task:%d" % (server_def.job_name,
                                                         server_def.task_index))
    logical_local_devices = [d for d in self.list_logical_devices("CPU") if
                             d.name.startswith(tuple(local_prefix))]
    self.ensure_initialized()
    # Error out if there are already multiple logical CPU in the context.
    if len(logical_local_devices) > 1:
      raise RuntimeError("Virtual CPUs already set, cannot modify again.")

    pywrap_tfe.TFE_SetLogicalCpuDevices(self._context_handle, num_cpus, prefix)
    self._initialize_logical_devices()

  def get_compiler_ir(self, device_name, function_name, args, stage="hlo"):
    return pywrap_tfe.TF_GetCompilerIr(self._context_handle, function_name,
                                       stage, device_name, args)

  @deprecated(
      None, "XLA:CPU and XLA:GPU devices are deprecated", warn_once=True)
  def enable_xla_devices(self):
    """Enables XLA:CPU and XLA:GPU devices registration."""
    pywrap_tfe.TF_EnableXlaDevices()

  @property
  def enable_mlir_bridge(self):
    return pywrap_tfe.TF_IsMlirBridgeEnabled()

  @property
  def enable_mlir_graph_optimization(self):
    return self._enable_mlir_graph_optimization

  @enable_mlir_bridge.setter
  def enable_mlir_bridge(self, enabled):
    pywrap_tfe.TF_EnableMlirBridge(enabled)
    self._thread_local_data.function_call_options = None

  @enable_mlir_graph_optimization.setter
  def enable_mlir_graph_optimization(self, enabled):
    self._enable_mlir_graph_optimization = enabled
    self._thread_local_data.function_call_options = None

  @property
  def optimizer_jit(self):
    level = self.config.graph_options.optimizer_options.global_jit_level
    return (level == config_pb2.OptimizerOptions.ON_1 or
            level == config_pb2.OptimizerOptions.ON_2)

  @optimizer_jit.setter
  def optimizer_jit(self, enabled):
    self._optimizer_jit = enabled

    self._thread_local_data.function_call_options = None

  def get_optimizer_experimental_options(self):
    """Get experimental options for the optimizer.

    Returns:
      Dictionary of current option values
    """
    rewrite_options = self.config.graph_options.rewrite_options
    options = {}

    def rewriter_toggle(option):
      attr = getattr(rewrite_options, option)
      if attr != 0:
        options[option] = (attr == rewriter_config_pb2.RewriterConfig.ON)

    def rewriter_bool(option):
      options[option] = getattr(rewrite_options, option)

    rewriter_toggle("layout_optimizer")
    rewriter_toggle("constant_folding")
    rewriter_toggle("shape_optimization")
    rewriter_toggle("remapping")
    rewriter_toggle("arithmetic_optimization")
    rewriter_toggle("dependency_optimization")
    rewriter_toggle("loop_optimization")
    rewriter_toggle("function_optimization")
    rewriter_toggle("debug_stripper")
    rewriter_bool("disable_model_pruning")
    rewriter_toggle("scoped_allocator_optimization")
    rewriter_toggle("pin_to_host_optimization")
    rewriter_toggle("implementation_selector")
    rewriter_toggle("auto_mixed_precision")
    rewriter_toggle("use_plugin_optimizers")
    rewriter_bool("disable_meta_optimizer")
    rewriter_toggle("auto_mixed_precision_mkl")

    if rewrite_options.min_graph_nodes != 0:
      options["min_graph_nodes"] = rewrite_options.min_graph_nodes

    return options

  def set_optimizer_experimental_options(self, options):
    """Set experimental options for the optimizer.

    Args:
      options: Dictionary of options to modify
    """
    self._optimizer_experimental_options.update(options)

    self._thread_local_data.function_call_options = None

  @property
  def intra_op_parallelism_threads(self):
    return self.config.intra_op_parallelism_threads

  @intra_op_parallelism_threads.setter
  def intra_op_parallelism_threads(self, num_threads):
    if self._intra_op_parallelism_threads == num_threads:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Intra op parallelism cannot be modified after initialization.")

    self._intra_op_parallelism_threads = num_threads

  @property
  def inter_op_parallelism_threads(self):
    return self.config.inter_op_parallelism_threads

  @inter_op_parallelism_threads.setter
  def inter_op_parallelism_threads(self, num_threads):
    if self._inter_op_parallelism_threads == num_threads:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Inter op parallelism cannot be modified after initialization.")

    self._inter_op_parallelism_threads = num_threads

  @property
  def soft_device_placement(self):
    return self.config.allow_soft_placement

  @soft_device_placement.setter
  def soft_device_placement(self, enable):
    if self._context_handle is not None:
      pywrap_tfe.TFE_ContextSetSoftDevicePlacement(self._handle, enable)

    self._soft_device_placement = enable
    self._thread_local_data.function_call_options = None

  @property
  def log_device_placement(self):
    return self.config.log_device_placement

  @log_device_placement.setter
  def log_device_placement(self, enable):
    if self._context_handle is not None:
      pywrap_tfe.TFE_ContextSetLogDevicePlacement(self._handle, enable)

    self._log_device_placement = enable
    self._thread_local_data.function_call_options = None

  @property
  def run_eager_op_as_function(self):
    return self._run_eager_op_as_function

  @run_eager_op_as_function.setter
  def run_eager_op_as_function(self, enable):
    if self._context_handle is not None:
      pywrap_tfe.TFE_ContextSetRunEagerOpAsFunction(self._handle, enable)
    self._run_eager_op_as_function = enable

  @property
  def jit_compile_rewrite(self):
    return self._jit_compile_rewrite

  @jit_compile_rewrite.setter
  def jit_compile_rewrite(self, enable):
    if self._context_handle is not None:
      pywrap_tfe.TFE_ContextSetJitCompileRewrite(self._handle, enable)
    self._jit_compile_rewrite = enable

  @property
  def device_policy(self):
    # Only get the policy from the context if it has already been initialized
    if self._context_handle is not None:
      return pywrap_tfe.TFE_ContextGetDevicePlacementPolicy(self._handle)

    return self._device_policy

  @device_policy.setter
  def device_policy(self, policy):
    if policy is None:
      policy = DEVICE_PLACEMENT_SILENT

    if self._device_policy != policy:
      self._device_policy = policy

      # Only set the policy if the context has already been initialized
      if self._context_handle is not None:
        pywrap_tfe.TFE_ContextSetThreadLocalDevicePlacementPolicy(
            self._handle, self._device_policy)

  @property
  def use_tfrt(self):
    return self._use_tfrt

  @use_tfrt.setter
  def use_tfrt(self, tfrt):
    """Sets whether to use TFRT."""
    if not isinstance(tfrt, bool):
      raise ValueError("Expecting a boolean but got %s" % type(tfrt))

    if self._use_tfrt != tfrt:
      if self._initialized:
        raise ValueError("use_tfrt should be set before being initialized.")
      self._use_tfrt = tfrt

  @property
  def use_tfrt_distributed_runtime(self):
    return self._use_tfrt_distributed_runtime

  @use_tfrt_distributed_runtime.setter
  def use_tfrt_distributed_runtime(self, enable):
    """Sets whether to use TFRT distributed runtime.

    This is only effective when use_tfrt is also true. Note that currently TFRT
    distributed runtime is not function complete and this config is for testing
    only.
    Args:
      enable: A boolean to set whether to use TFRT distributed runtime.
    """
    if not isinstance(enable, bool):
      raise ValueError("Expecting a boolean but got %s" % type(enable))

    if self._use_tfrt_distributed_runtime != enable:
      if self._initialized:
        raise ValueError("use_tfrt should be set before being initialized.")
      self._use_tfrt_distributed_runtime = enable

  @property
  def operation_timeout_in_ms(self):
    return self.config.operation_timeout_in_ms

  @operation_timeout_in_ms.setter
  def operation_timeout_in_ms(self, timeout_in_ms):
    if self._operation_timeout_in_ms == timeout_in_ms:
      return

    if self._context_handle is not None:
      raise RuntimeError(
          "Operation timeout cannot be modified after initialization.")

    self._operation_timeout_in_ms = timeout_in_ms

  def enable_run_metadata(self):
    """Enables tracing of op execution via RunMetadata.

    To retrieve the accumulated metadata call context.export_run_metadata()
    and to stop tracing call context.disable_run_metadata().
    """
    self.ensure_initialized()
    pywrap_tfe.TFE_ContextEnableRunMetadata(self._handle)

  def disable_run_metadata(self):
    """Disables tracing of op execution via RunMetadata."""
    if not self._context_handle:
      return
    pywrap_tfe.TFE_ContextDisableRunMetadata(self._context_handle)

  def enable_graph_collection(self):
    """Enables graph collection of executed functions.

    To retrieve the accumulated graphs call context.export_run_metadata()
    and to stop collecting graphs call context.disable_graph_collection().
    """
    self.ensure_initialized()
    pywrap_tfe.TFE_ContextEnableGraphCollection(self._handle)

  def disable_graph_collection(self):
    """Disables graph collection of executed functions."""
    if not self._context_handle:
      return
    pywrap_tfe.TFE_ContextDisableGraphCollection(self._context_handle)

  def export_run_metadata(self):
    """Returns a RunMetadata proto with accumulated information.

    The returned protocol buffer contains information since the most recent call
    to either enable_run_metadata or export_run_metadata.

    Returns:
      A RunMetadata protocol buffer. Or None if not enabled.
    """
    if not self._context_handle:
      return None
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tfe.TFE_ContextExportRunMetadata(self._context_handle, buffer_)
      proto_data = pywrap_tf_session.TF_GetBuffer(buffer_)
    run_metadata = config_pb2.RunMetadata()
    run_metadata.ParseFromString(compat.as_bytes(proto_data))
    return run_metadata

  @property
  def context_switches(self):
    """Returns a stack of context switches."""
    return self._context_switches


class _EagerDeviceContext(object):
  """Context-manager forcing placement of ops and Tensors on a device."""

  __slots__ = ["_device_name", "_ctx", "_stack"]

  def __init__(self, ctx, device_name):
    self._device_name = device_name
    self._ctx = ctx
    self._stack = []

  # TODO(b/189233748): Consolidate the device string parsing logic with
  # tensorflow/core/util/device_name_utils.cc.
  def __enter__(self):
    ctx = self._ctx
    old_device_name = ctx.device_name
    old_device_spec = ctx.device_spec
    new_device_name = self._device_name
    cache_key = (old_device_name, new_device_name)
    try:
      new_device_name, new_device_spec = _device_parsing_cache[cache_key]
    except TypeError:
      # Error while trying to compute the cache key.
      raise ValueError("Expecting a string device name. Got %s(%s)" %
                       (type(new_device_name), new_device_name))
    except KeyError:
      # Handle a cache miss.
      if new_device_name is not None:
        if not isinstance(new_device_name, six.string_types):
          raise ValueError("Expecting a string device name. Got %s(%s)" %
                           (type(new_device_name), new_device_name))
        device_spec = pydev.DeviceSpec.from_string(new_device_name)
        if old_device_name:
          new_device_spec = copy.copy(old_device_spec)
        else:
          ctx.ensure_initialized()
          new_device_spec = pydev.DeviceSpec.from_string(
              ctx._context_devices[0])  # pylint: disable=protected-access
        new_device_spec = new_device_spec.make_merged_spec(device_spec)
      else:
        new_device_spec = pydev.DeviceSpec.from_string("")
      new_device_name = new_device_spec.to_string()
      _device_parsing_cache[cache_key] = (new_device_name, new_device_spec)

    ctx._set_device(new_device_name, new_device_spec)  # pylint: disable=protected-access
    self._stack.append((old_device_name, old_device_spec, new_device_spec))

  def __exit__(self, *ex_info):
    ctx = self._ctx
    old_device_name, old_device_spec, new_device_spec = self._stack[-1]
    if ctx.device_spec is not new_device_spec:
      raise RuntimeError("Exiting device scope without proper scope nesting")
    del self._stack[-1]
    ctx._set_device(old_device_name, old_device_spec)  # pylint: disable=protected-access


# Do not change directly.
_context = None
_context_lock = threading.Lock()


def _set_context_locked(ctx):
  global _context
  pywrap_tfe.TFE_Py_SetEagerContext(ctx)
  ctx.mark_as_global_context()
  _context = ctx


def _set_context(ctx):
  with _context_lock:
    _set_context_locked(ctx)


def _create_context():
  with _context_lock:
    if _context is None:
      ctx = Context()
      _set_context_locked(ctx)


def _reset_context():
  """Clears and re-initializes the singleton context.

  Should only be used for testing.
  """
  global _context
  global _device_parsing_cache

  # Garbage collect and clear scalar cache to avoid Tensor from current context
  # polluting next context.
  gc.collect()
  pywrap_tfe.TFE_ClearScalarCache()
  with _context_lock:
    if _context is not None:
      _context._clear_caches()
      _context = None
  _create_context()
  _device_parsing_cache = {}


def _reset_jit_compiler_flags():
  """Clears and re-initializes the TF JIT compiler flags.

  Should only be used for testing.
  """
  pywrap_tfe.TF_ResetJitCompilerFlags()


def context():
  """Returns a singleton context object."""
  if _context is None:
    _create_context()
  return _context


def context_safe():
  """Returns current context (or None if one hasn't been initialized)."""
  return _context


def ensure_initialized():
  """Initialize the context."""
  context().ensure_initialized()


def initialize_logical_devices():
  """Initialize the virtual devices."""
  context()._initialize_logical_devices()  # pylint: disable=protected-access


def set_global_seed(seed):
  """Sets the eager mode seed."""
  context()._set_global_seed(seed)  # pylint: disable=protected-access


def global_seed():
  """Returns the eager mode seed."""
  return context()._seed  # pylint: disable=protected-access


def internal_operation_seed():
  """Returns the operation seed generated based on global seed."""
  return context()._internal_operation_seed()  # pylint: disable=protected-access


@tf_export("executing_eagerly", v1=[])
def executing_eagerly():
  """Checks whether the current thread has eager execution enabled.

  Eager execution is enabled by default and this API returns `True`
  in most of cases. However, this API might return `False` in the following use
  cases.

  *  Executing inside `tf.function`, unless under `tf.init_scope` or
     `tf.config.run_functions_eagerly(True)` is previously called.
  *  Executing inside a transformation function for `tf.dataset`.
  *  `tf.compat.v1.disable_eager_execution()` is called.

  General case:

  >>> print(tf.executing_eagerly())
  True

  Inside `tf.function`:

  >>> @tf.function
  ... def fn():
  ...   with tf.init_scope():
  ...     print(tf.executing_eagerly())
  ...   print(tf.executing_eagerly())
  >>> fn()
  True
  False

  Inside `tf.function` after `tf.config.run_functions_eagerly(True)` is called:

  >>> tf.config.run_functions_eagerly(True)
  >>> @tf.function
  ... def fn():
  ...   with tf.init_scope():
  ...     print(tf.executing_eagerly())
  ...   print(tf.executing_eagerly())
  >>> fn()
  True
  True
  >>> tf.config.run_functions_eagerly(False)

  Inside a transformation function for `tf.dataset`:

  >>> def data_fn(x):
  ...   print(tf.executing_eagerly())
  ...   return x
  >>> dataset = tf.data.Dataset.range(100)
  >>> dataset = dataset.map(data_fn)
  False

  Returns:
    `True` if the current thread has eager execution enabled.
  """
  ctx = context_safe()
  if ctx is None:
    return default_execution_mode == EAGER_MODE

  return ctx.executing_eagerly()


@tf_export(v1=["executing_eagerly"])
def executing_eagerly_v1():
  """Checks whether the current thread has eager execution enabled.

  Eager execution is typically enabled via
  `tf.compat.v1.enable_eager_execution`, but may also be enabled within the
  context of a Python function via tf.contrib.eager.py_func.

  When eager execution is enabled, returns `True` in most cases. However,
  this API might return `False` in the following use cases.

  *  Executing inside `tf.function`, unless under `tf.init_scope` or
     `tf.config.run_functions_eagerly(True)` is previously called.
  *  Executing inside a transformation function for `tf.dataset`.
  *  `tf.compat.v1.disable_eager_execution()` is called.

  >>> tf.compat.v1.enable_eager_execution()

  General case:

  >>> print(tf.executing_eagerly())
  True

  Inside `tf.function`:

  >>> @tf.function
  ... def fn():
  ...   with tf.init_scope():
  ...     print(tf.executing_eagerly())
  ...   print(tf.executing_eagerly())
  >>> fn()
  True
  False

  Inside `tf.function`
  after  `tf.config.run_functions_eagerly(True)` is called:

  >>> tf.config.run_functions_eagerly(True)
  >>> @tf.function
  ... def fn():
  ...   with tf.init_scope():
  ...     print(tf.executing_eagerly())
  ...   print(tf.executing_eagerly())
  >>> fn()
  True
  True
  >>> tf.config.run_functions_eagerly(False)

  Inside a transformation function for `tf.dataset`:

  >>> def data_fn(x):
  ...   print(tf.executing_eagerly())
  ...   return x
  >>> dataset = tf.data.Dataset.range(100)
  >>> dataset = dataset.map(data_fn)
  False

  Returns:
    `True` if the current thread has eager execution enabled.
  """
  return executing_eagerly()


def in_eager_mode():
  """Use executing_eagerly() instead. This function will be removed."""
  return executing_eagerly()


def anonymous_name():
  """Returns the anonymous shared name.

  In eager mode we create anonymous resources to avoid spurious sharing issues.
  The runtime generates a unique name on our behalf when the reserved
  anonymous shared name is used as a shared name.

  Returns:
    The anonymous shared name.
  """

  # The magic value is defined as
  # `tensorflow::ResourceHandle::ANONYMOUS_NAME` in C++.
  return "cd2c89b7-88b7-44c8-ad83-06c2a9158347"


def graph_mode():
  """Context-manager to disable eager execution for the current thread."""
  return context()._mode(GRAPH_MODE)  # pylint: disable=protected-access


# Used by b/167638505 for keras backend API and Lambda layer.
@tf_export("__internal__.eager_context.eager_mode", v1=[])
def eager_mode():
  """Context-manager to enable eager execution for the current thread."""
  return context()._mode(EAGER_MODE)  # pylint: disable=protected-access


def scope_name():
  """Name of the current scope."""
  return context().scope_name


def device(name):
  """Context-manager to force placement of operations and Tensors on a device.

  Example:
  ```python
  with tf.device('gpu:0'):
    with tf.device('cpu:0'):
      shape = tf.constant([], dtype=tf.int32)
    x = tf.random.truncated_normal(shape, tf.float32)
  ```
  will ensure that the `shape` Tensor is on CPU but the `truncated_normal`
  operation runs on GPU 0.

  Args:
    name: Name of the device (see context().devices()), or None to perform
      automatic placement.

  Returns:
    Context manager for setting the device.
  """
  ensure_initialized()
  return context().device(name)


# Expose some properties of Context as internally public APIs (b/160348781).
@tf_export("__internal__.eager_context.get_config", v1=[])
def get_config():
  """Get the ConfigProto of Context.

  Returns:
    The ConfigProto of Context.
  """
  return context().config


@tf_export("__internal__.eager_context.get_device_name", v1=[])
def get_device_name():
  """Get the device name for the current thread.

  Returns:
    The device name for the current thread.
  """
  return context().device_name


@tf_export("__internal__.eager_context.set_soft_device_placement", v1=[])
def set_soft_device_placement(enabled):
  """Set if soft device placements should be allowed.

  Args:
    enabled: Whether to enable soft device placement.
  """
  context().soft_device_placement = enabled


@tf_export("__internal__.eager_context.get_executor", v1=[])
def get_executor():
  """Get the Executor of the current thread.

  Returns:
    The Executor of the current thread.
  """
  return context().executor


@tf_export("debugging.get_log_device_placement")
def get_log_device_placement():
  """Get if device placements are logged.

  Returns:
    If device placements are logged.
  """
  return context().log_device_placement


@tf_export("debugging.set_log_device_placement")
def set_log_device_placement(enabled):
  """Turns logging for device placement decisions on or off.

  Operations execute on a particular device, producing and consuming tensors on
  that device. This may change the performance of the operation or require
  TensorFlow to copy data to or from an accelerator, so knowing where operations
  execute is useful for debugging performance issues.

  For more advanced profiling, use the [TensorFlow
  profiler](https://www.tensorflow.org/guide/profiler).

  Device placement for operations is typically controlled by a `tf.device`
  scope, but there are exceptions, for example operations on a `tf.Variable`
  which follow the initial placement of the variable. Turning off soft device
  placement (with `tf.config.set_soft_device_placement`) provides more explicit
  control.

  >>> tf.debugging.set_log_device_placement(True)
  >>> tf.ones([])
  >>> # [...] op Fill in device /job:localhost/replica:0/task:0/device:GPU:0
  >>> with tf.device("CPU"):
  ...  tf.ones([])
  >>> # [...] op Fill in device /job:localhost/replica:0/task:0/device:CPU:0
  >>> tf.debugging.set_log_device_placement(False)

  Turning on `tf.debugging.set_log_device_placement` also logs the placement of
  ops inside `tf.function` when the function is called.

  Args:
    enabled: Whether to enabled device placement logging.
  """
  context().log_device_placement = enabled


@tf_contextlib.contextmanager
def device_policy(policy):
  """Context manager for setting device placement policy for current thread."""
  ctx = context()
  old_policy = ctx.device_policy
  try:
    ctx.device_policy = policy
    yield
  finally:
    ctx.device_policy = old_policy


def set_execution_mode(mode):
  """Sets execution mode for the current thread."""
  context().execution_mode = mode


# TODO(fishx): remove this method.
@tf_contextlib.contextmanager
def execution_mode(mode):
  """Context manager for setting execution mode for current thread."""
  if mode is None:
    yield
  else:
    ctx = context()
    executor_new = executor.new_executor(mode == ASYNC)
    executor_old = ctx.executor
    try:
      executor_old.wait()
      ctx.executor = executor_new
      yield
    finally:
      ctx.executor = executor_old
      executor_new.wait()


@tf_contextlib.contextmanager
def executor_scope(e):
  """Context manager for changing executor for current thread.

  Args:
    e: A Executor to execute eager ops under this scope. Setting it to None will
      switch back to use the default executor for the context.

  Yields:
    Context manager for setting the executor for current thread.
  """
  ctx = context()
  executor_old = ctx.executor
  try:
    ctx.executor = e
    yield
  finally:
    ctx.executor = executor_old


@tf_export("experimental.function_executor_type")
@tf_contextlib.contextmanager
def function_executor_type(executor_type):
  """Context manager for setting the executor of eager defined functions.

  Eager defined functions are functions decorated by tf.contrib.eager.defun.

  Args:
    executor_type: a string for the name of the executor to be used to execute
      functions defined by tf.contrib.eager.defun.

  Yields:
    Context manager for setting the executor of eager defined functions.
  """
  current_options = context().function_call_options
  old_options = copy.copy(current_options)
  try:
    current_options.executor_type = executor_type
    yield
  finally:
    context().function_call_options = old_options


def is_async():
  """Returns true if current thread is in async mode."""
  return context().is_async()


def num_gpus():
  """Get the number of available GPU devices.

  Returns:
    The number of available GPU devices.
  """
  return context().num_gpus()


def enable_run_metadata():
  """Enables tracing of op execution via RunMetadata.

  To retrieve the accumulated metadata call context.export_run_metadata()
  and to stop tracing call context.disable_run_metadata().
  """
  context().enable_run_metadata()


def disable_run_metadata():
  """Disables tracing of op execution via RunMetadata."""
  context().disable_run_metadata()


def enable_graph_collection():
  """Enables graph collection of executed functions.

  To retrieve the accumulated graphs call context.export_run_metadata()
  and to stop collecting graphs call context.disable_graph_collection().
  """
  context().enable_graph_collection()


def disable_graph_collection():
  """Disables graph collection of executed functions."""
  context().disable_graph_collection()


def export_run_metadata():
  """Returns a RunMetadata proto with accumulated information.

  The returned protocol buffer contains information since the most recent call
  to either enable_run_metadata or export_run_metadata.

  Returns:
    A RunMetadata protocol buffer.
  """
  return context().export_run_metadata()


@contextlib.contextmanager
def collect_graphs(optimized=True):
  """Collects a flat list of pre- or post-optimization graphs.

  The collected graphs include device placements, which can be useful for
  testing.

  Usage:

  ```
  @def_function.function
  def f(x):
    return x + constant_op.constant(1.)

  with context.collect_graphs() as graphs:
    with ops.device("CPU:0"):
      f(constant_op.constant(1.))

  graph, = graphs  # `graph` contains a single GraphDef for inspection
  ```

  Args:
    optimized: whether to collect optimized graphs or non-optimized graphs

  Yields:
    A list of GraphDefs, populated when the context manager exits.
  """
  ctx = context()
  ctx.enable_graph_collection()
  try:
    graphs = []
    yield graphs
    metadata = ctx.export_run_metadata()
  finally:
    ctx.disable_graph_collection()
  for graph in metadata.function_graphs:
    if optimized:
      graphs.append(graph.post_optimization_graph)
    else:
      graphs.append(graph.pre_optimization_graph)


def get_server_def():
  return context().get_server_def()


def set_server_def(server_def):
  context().set_server_def(server_def)


def update_server_def(server_def):
  context().update_server_def(server_def)


def check_alive(worker_name):
  return context().check_alive(worker_name)


@tf_export("experimental.async_scope")
@tf_contextlib.contextmanager
def async_scope():
  """Context manager for grouping async operations.

  Ops/function calls inside the scope can return before finishing the actual
  execution. When exiting the async scope, a synchronization barrier will be
  automatically added to ensure the completion of all async op and function
  execution, potentially raising exceptions if async execution results in
  an error state.

  Users may write the following code to asynchronously invoke `train_step_fn`
  and log the `loss` metric for every `num_steps` steps in a training loop.
  `train_step_fn` internally consumes data using `iterator.get_next()`, and may
  throw OutOfRangeError when running out of data. In the case:

  ```
  try:
    with tf.experimental.async_scope():
      for _ in range(num_steps):
        # Step function updates the metric `loss` internally
        train_step_fn()
  except tf.errors.OutOfRangeError:
    tf.experimental.async_clear_error()
  logging.info('loss = %s', loss.numpy())
  ```

  Yields:
    Context manager for grouping async operations.
  """
  # TODO(haoyuzhang): replace env var once we have a config method to turn on
  # and off async streaming RPC
  remote_async_env_var = "TF_ENABLE_EAGER_CLIENT_STREAMING_ENQUEUE"
  old_policy = os.environ.get(remote_async_env_var)
  try:
    os.environ[remote_async_env_var] = str(True)
    yield
    # Note: sync local and remote executors iff the async block does not raise
    # an exception. Triggering sync after an exception may lead to derived
    # runtime errors and unexpected exception types.
    context().sync_executors()
  finally:
    if old_policy is None:
      del os.environ[remote_async_env_var]
    else:
      os.environ[remote_async_env_var] = old_policy


def async_wait():
  """Sync all async operations and raise any errors during execution.

  In async execution mode, an op/function call can return before finishing the
  actual execution. Calling this method creates a synchronization barrier for
  all async op and function execution. It only returns when all pending nodes
  are finished, potentially raising exceptions if async execution results in
  an error state. It is a no-op if the context is not initialized.
  """
  disable_async_executor_env_var = "TF_PS_DISABLE_ASYNC_EXECUTOR_GLOBALLY"
  if os.environ.get(disable_async_executor_env_var) == str(True):
    return
  if context()._context_handle is not None:  # pylint: disable=protected-access
    context().sync_executors()


@tf_export("experimental.async_clear_error")
def async_clear_error():
  """Clear pending operations and error statuses in async execution.

  In async execution mode, an error in op/function execution can lead to errors
  in subsequent ops/functions that are scheduled but not yet executed. Calling
  this method clears all pending operations and reset the async execution state.

  Example:

  ```
  while True:
    try:
      # Step function updates the metric `loss` internally
      train_step_fn()
    except tf.errors.OutOfRangeError:
      tf.experimental.async_clear_error()
      break
  logging.info('loss = %s', loss.numpy())
  ```
  """
  context().clear_executor_errors()


def add_function(fdef):
  """Add a function definition to the context."""
  context().add_function(fdef)


def remove_function(name):
  """Remove a function from the context."""
  context().remove_function(name)


def get_function_def(name):
  return context().get_function_def(name)


def register_custom_device(device_capsule, device_name, device_info_capsule):
  """Calls TFE_RegisterCustomDevice to register a custom device with Python.

  Enables using C extensions specifying a custom device from Python. See the
  experimental eager C API in tensorflow/c/eager/c_api_experimental.h for
  details.

  Note that custom devices are not currently supported inside `tf.function`s.

  Args:
    device_capsule: A PyCapsule with the name set to 'TFE_CustomDevice'
      containing a pointer to a TFE_CustomDevice struct. The capsule retains
      ownership of the memory.
    device_name: A string indicating the name to register the custom device
      under, e.g. '/job:localhost/replica:0/task:0/device:CUSTOM:0'. It may
      subsequently be passed to `with tf.device(...):`.
    device_info_capsule: A PyCapsule with the name set to
      'TFE_CustomDevice_DeviceInfo' containing a pointer to a device-specific
      struct with the initial state of the custom device (the void* device_info
      argument to TFE_RegisterCustomDevice). This method takes ownership of the
      memory and clears the capsule destructor.
  """
  context().register_custom_device(device_capsule, device_name,
                                   device_info_capsule)


# Not every user creates a Context via context.context()
# (for example, enable_eager_execution in python/framework/ops.py),
# but they do all import this file.  Note that IS_IN_GRAPH_MODE and
# in_graph_mode are both parameterless functions.
def _tmp_in_graph_mode():
  if context_safe() is None:
    # Context not yet initialized. Assume graph mode following the
    # default implementation in `is_in_graph_mode`.
    return True
  return not executing_eagerly()


is_in_graph_mode.IS_IN_GRAPH_MODE = _tmp_in_graph_mode
