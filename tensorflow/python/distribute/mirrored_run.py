# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Class MirroredStrategy implementing tf.distribute.Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools
import threading
import weakref

from tensorflow.python import pywrap_tfe
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import shared_variable_creator
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import coordinator


def call_for_each_replica(strategy, fn, args=None, kwargs=None):
  """Call `fn` on each worker devices(replica).

  It's highly recommended to wrap the call to this function inside a
  `tf.function`, otherwise the performance is poor.

  Args:
    strategy: `tf.distribute.Strategy`.
    fn: function to call on each worker devices.
    args: positional arguments to `fn`.
    kwargs: keyword arguments to `fn`.

  Returns:
    Wrapped returned value of `fn` from all replicas.
  """
  if args is None:
    args = ()
  if kwargs is None:
    kwargs = {}

  if isinstance(fn, def_function.Function):
    if strategy not in _cfer_fn_cache:
      _cfer_fn_cache[strategy] = weakref.WeakKeyDictionary()
    wrapped = _cfer_fn_cache[strategy].get(fn)
    if wrapped is None:
      # We need to wrap fn such that it triggers _call_for_each_replica inside
      # the tf.function. We use _clone() instead of @tf.function wrapped
      # call_for_each_replica() because we would like to retain the arguments to
      # the @tf.function decorator of fn.
      wrapped = fn._clone(  # pylint: disable=protected-access
          python_function=functools.partial(call_for_each_replica, strategy,
                                            fn.python_function))
      _cfer_fn_cache[strategy][fn] = wrapped
    return wrapped(args, kwargs)

  if context.executing_eagerly():
    logging.log_first_n(
        logging.WARN, "Using %s eagerly has significant "
        "overhead currently. We will be working on improving "
        "this in the future, but for now please wrap "
        "`call_for_each_replica` or `experimental_run` or "
        "`experimental_run_v2` inside a tf.function to get "
        "the best performance." % strategy.__class__.__name__, 5)
  else:
    # When a tf.function is wrapped to trigger _call_for_each_replica (see
    # the other branch above), AutoGraph stops conversion at
    # _call_for_each_replica itself (TF library functions are whitelisted).
    # This makes sure that the Python function that originally passed to
    # the tf.function is still converted.
    fn = autograph.tf_convert(fn, autograph_ctx.control_status_ctx())

  return _call_for_each_replica(strategy, fn, args, kwargs)


# Per strategy cache for call_for_each_replica def_function.Function objects.
_cfer_fn_cache = weakref.WeakKeyDictionary()


@contextlib.contextmanager
def _enter_graph(g, eager, creator_stack=None):
  """Context manager for selecting a graph and maybe eager mode."""
  if eager:
    with g.as_default(), context.eager_mode():
      if creator_stack is not None:
        g._variable_creator_stack = creator_stack  # pylint: disable=protected-access
      yield
  else:
    with g.as_default():
      if creator_stack is not None:
        g._variable_creator_stack = creator_stack  # pylint: disable=protected-access
      yield


def _cpu_device(device):
  cpu_device = tf_device.DeviceSpec.from_string(device)
  cpu_device = cpu_device.replace(device_type="CPU", device_index=0)
  return cpu_device.to_string()


class _RequestedStop(Exception):  # pylint: disable=g-bad-exception-name
  pass


def _call_for_each_replica(distribution, fn, args, kwargs):
  """Run `fn` in separate threads, once per replica/worker device.

  Args:
    distribution: the DistributionStrategy object.
    fn: function to run (will be run once per replica, each in its own thread).
    args: positional arguments for `fn`
    kwargs: keyword arguments for `fn`.

  Returns:
    Merged return value of `fn` across all replicas.

  Raises:
    RuntimeError: If fn() calls get_replica_context().merge_call() a different
        number of times from the available devices.
  """
  # TODO(josh11b): Add this option once we add synchronization to variable
  # creation. Until then, this is pretty unsafe to use.
  run_concurrently = False
  if not context.executing_eagerly():
    # Needed for per-thread device, etc. contexts in graph mode.
    ops.get_default_graph().switch_to_thread_local()

  coord = coordinator.Coordinator(clean_stop_exception_types=(_RequestedStop,))

  shared_variable_store = {}
  devices = distribution.extended.worker_devices

  # TODO(isaprykin): Create these threads once instead of during every call.
  threads = []
  for index in range(len(devices)):
    variable_creator_fn = shared_variable_creator.make_fn(
        shared_variable_store, index)
    t = _MirroredReplicaThread(
        distribution, coord, index, devices, variable_creator_fn, fn,
        values.select_replica(index, args),
        values.select_replica(index, kwargs))
    threads.append(t)

  for t in threads:
    t.start()

  # When `fn` starts `should_run` event is set on _MirroredReplicaThread
  # (`MRT`) threads. The execution waits until
  # `MRT.has_paused` is set, which indicates that either `fn` is
  # complete or a `get_replica_context().merge_call()` is called.  If `fn` is
  # complete, then `MRT.done` is set to True.  Otherwise, arguments
  # of `get_replica_context().merge_call` from all paused threads are grouped
  # and the `merge_fn` is performed.  Results of the
  # `get_replica_context().merge_call` are then set to `MRT.merge_result`.
  # Each such `get_replica_context().merge_call` call returns the
  # `MRT.merge_result` for that thread when `MRT.should_run` event
  # is reset again. Execution of `fn` resumes.

  try:
    with coord.stop_on_exception():
      all_done = False
      while not all_done and not coord.should_stop():
        done = []
        if run_concurrently:
          for t in threads:
            t.should_run.set()
          for t in threads:
            t.has_paused.wait()
            t.has_paused.clear()
            if coord.should_stop():
              return None
            done.append(t.done)
        else:
          for t in threads:
            t.should_run.set()
            t.has_paused.wait()
            t.has_paused.clear()
            if coord.should_stop():
              return None
            done.append(t.done)
        if coord.should_stop():
          return None
        all_done = all(done)
        if not all_done:
          if any(done):
            raise RuntimeError("Some replicas made a different number of "
                               "replica_context().merge_call() calls.")
          # get_replica_context().merge_call() case
          merge_args = values.regroup(tuple(t.merge_args for t in threads))
          merge_kwargs = values.regroup(tuple(t.merge_kwargs for t in threads))
          # We capture the name_scope of the MRT when we call merge_fn
          # to ensure that if we have opened a name scope in the MRT,
          # it will be respected when executing the merge function. We only
          # capture the name_scope from the first MRT and assume it is
          # the same for all other MRTs.
          mtt_captured_name_scope = threads[0].captured_name_scope
          mtt_captured_var_scope = threads[0].captured_var_scope
          # Capture and merge the control dependencies from all the threads.
          mtt_captured_control_deps = set()
          for t in threads:
            mtt_captured_control_deps.update(t.captured_control_deps)
          with ops.name_scope(mtt_captured_name_scope),\
              ops.control_dependencies(mtt_captured_control_deps), \
              variable_scope.variable_scope(mtt_captured_var_scope):
            merge_result = threads[0].merge_fn(distribution, *merge_args,
                                               **merge_kwargs)
          for r, t in enumerate(threads):
            t.merge_result = values.select_replica(r, merge_result)
  finally:
    for t in threads:
      t.should_run.set()
    coord.join(threads)

  return values.regroup(tuple(t.main_result for t in threads))


class _MirroredReplicaThread(threading.Thread):
  """A thread that runs() a function on a device."""

  def __init__(self, dist, coord, replica_id, devices, variable_creator_fn,
               fn, args, kwargs):
    super(_MirroredReplicaThread, self).__init__()
    self.coord = coord
    self.distribution = dist
    self.devices = devices
    self.replica_id = replica_id
    self.variable_creator_fn = variable_creator_fn
    # State needed to run and return the results of `fn`.
    self.main_fn = fn
    self.main_args = args
    self.main_kwargs = kwargs
    self.main_result = None
    self.done = False
    # State needed to run the next merge_call() (if any) requested via
    # ReplicaContext.
    self.merge_fn = None
    self.merge_args = None
    self.merge_kwargs = None
    self.merge_result = None
    self.captured_name_scope = None
    self.captured_var_scope = None
    # We use a thread.Event for the main thread to signal when this
    # thread should start running (`should_run`), and another for
    # this thread to transfer control back to the main thread
    # (`has_paused`, either when it gets to a
    # `get_replica_context().merge_call` or when `fn` returns). In
    # either case the event starts cleared, is signaled by calling
    # set(). The receiving thread waits for the signal by calling
    # wait() and then immediately clearing the event using clear().
    self.should_run = threading.Event()
    self.has_paused = threading.Event()
    # These fields have to do with inheriting various contexts from the
    # parent thread:
    context.ensure_initialized()
    ctx = context.context()
    self.in_eager = ctx.executing_eagerly()
    self.record_thread_local_summary_state()
    self.record_thread_local_eager_context_state()
    self.context_device_policy = (
        pywrap_tfe.TFE_ContextGetDevicePlacementPolicy(
            ctx._context_handle))  # pylint: disable=protected-access
    self.graph = ops.get_default_graph()
    with ops.init_scope():
      self._init_in_eager = context.executing_eagerly()
      self._init_graph = ops.get_default_graph()
    self._variable_creator_stack = self.graph._variable_creator_stack[:]  # pylint: disable=protected-access
    self._var_scope = variable_scope.get_variable_scope()
    # Adding a "/" at end lets us re-enter this scope later.
    self._name_scope = self.graph.get_name_scope()
    if self._name_scope:
      self._name_scope += "/"
    if self.replica_id > 0:
      if not self._name_scope:
        self._name_scope = ""
      self._name_scope += "replica_%d/" % self.replica_id

  def run(self):
    self.should_run.wait()
    self.should_run.clear()
    try:
      if self.coord.should_stop():
        return
      self.restore_thread_local_summary_state()
      self.restore_thread_local_eager_context_state()
      # TODO(josh11b): Use current logical device instead of 0 here.
      with self.coord.stop_on_exception(), \
          _enter_graph(self._init_graph, self._init_in_eager), \
          _enter_graph(self.graph, self.in_eager,
                       self._variable_creator_stack), \
          context.device_policy(self.context_device_policy), \
          _MirroredReplicaContext(self.distribution, constant_op.constant(
              self.replica_id, dtypes.int32)), \
          ops.device(self.devices[self.replica_id]), \
          ops.name_scope(self._name_scope), \
          variable_scope.variable_scope(
              self._var_scope, reuse=self.replica_id > 0), \
          variable_scope.variable_creator_scope(self.variable_creator_fn):
        self.main_result = self.main_fn(*self.main_args, **self.main_kwargs)
        self.done = True
    finally:
      self.has_paused.set()

  def record_thread_local_summary_state(self):
    """Record the thread local summary state in self."""
    # TODO(slebedev): is this still relevant? the referenced bug is closed.
    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    self._summary_step = summary_state.step
    self._summary_writer = summary_state.writer
    self._summary_recording = summary_state.is_recording
    self._summary_recording_distribution_strategy = (
        summary_state.is_recording_distribution_strategy)

  def restore_thread_local_summary_state(self):
    """Restore thread local summary state from self."""
    # TODO(slebedev): is this still relevant? the referenced bug is closed.
    summary_state = summary_ops_v2._summary_state  # pylint: disable=protected-access
    summary_state.step = self._summary_step
    summary_state.writer = self._summary_writer
    summary_state.is_recording = self._summary_recording
    summary_state.is_recording_distribution_strategy = (
        self._summary_recording_distribution_strategy)

  def record_thread_local_eager_context_state(self):
    ctx = context.context()
    eager_context_state = ctx._thread_local_data  # pylint: disable=protected-access
    self._eager_context_op_callbacks = eager_context_state.op_callbacks
    # TODO(b/125892694): record other fields in EagerContext.

  def restore_thread_local_eager_context_state(self):
    ctx = context.context()
    eager_context_state = ctx._thread_local_data  # pylint: disable=protected-access
    eager_context_state.op_callbacks = self._eager_context_op_callbacks
    # TODO(b/125892694): record other fields in EagerContext.


class _MirroredReplicaContext(distribute_lib.ReplicaContext):
  """ReplicaContext for synchronized replica."""

  def _merge_call(self, fn, args, kwargs):
    """`merge_call()` implementation for synchronized replica.

    This pauses the current replica thread and passes `fn` and its arguments to
    the main thread. The main thread will wait until all replicas pause, then
    invoke `fn` with grouped arguments. The current replica thread will continue
    after `fn` completes.

    See `_call_for_each_replica` for the logic in the main thread.

    Args:
      fn: a function that is called in cross replica context with grouped
        arguments from each replica. `fn` should returns grouped values.
      args: positional arguments to `fn`.
      kwargs: keyward arguments to `fn`.

    Returns:
      Return value of `fn` for the current replica.

    Raises:
      RuntimeError: when merge_call happens in a different graph, e.g. in a
        different tf.function, which is not supported now.
      _RequestedStop: when stop is requested.

    """
    t = threading.current_thread()
    assert isinstance(t, _MirroredReplicaThread)
    t.merge_fn = fn
    t.merge_args = args
    t.merge_kwargs = kwargs
    t.captured_name_scope = t.graph.get_name_scope()
    # Adding a "/" at end lets us re-enter this scope later.
    if t.captured_name_scope:
      t.captured_name_scope += "/"

    t.captured_var_scope = variable_scope.get_variable_scope()
    t.captured_control_deps = t.graph._current_control_dependencies()  # pylint: disable=protected-access

    # It is problematic if `merge_call` is called under a different graph other
    # than the one that `_call_for_each_replica` is called under, there are
    # 3 cases this can happen:
    #
    #   1. The `fn` passed to `_call_for_each_replica` is decorated with
    #   `tf.function` and there is a `merge_call` in `fn`. Since
    #   MirroredStrategy traces a separate function per thread (per device),
    #   and each trace takes a shared lock, the lock is never released by the
    #   first thread and subsequent replica threads cannot proceed to trace
    #   their own functions. This issue is addressed by always converting
    #   `_call_for_each_replica(tf.function(f))` to
    #   ``tf.function(_call_for_each_replica(f))`.` in
    #   `MirroredStrategy._call_for_each_replica`.
    #
    #   2. The `fn` passed to `_call_for_each_replica` contains a nested
    #   `tf.function`, and there is a `merge_call` in the nested `tf.function`.
    #   In this case each thread can successfully trace its own function, but
    #   since the `merge_fn` passed to `merge_call` is executed in the main
    #   thread (where `_call_for_each_replica` is executed), it can't access
    #   the tensors that come from different graphs.
    #
    #   3. The `fn` passed to `_call_for_each_replica` contains a control-flow
    #   statement, and there is a `merge_call` inside the control-flow body,
    #   `fn` or `_call_for_each_replica` is decorated with `tf.function`.
    #   Control flow statement creates a separate graph for its body, similar
    #   to #2, `merge_fn` executed in the main thread can't access the
    #   tensors that come from different graphs.
    #
    #   We raise an error for #2 and #3.
    if ops.get_default_graph() != t.graph:
      raise RuntimeError(
          "`merge_call` called while defining a new graph or a tf.function."
          " This can often happen if the function `fn` passed to"
          " `strategy.run()` contains a nested `@tf.function`, and the nested "
          "`@tf.function` contains a synchronization point, such as aggregating"
          " gradients (e.g, optimizer.apply_gradients), or if the function `fn`"
          " uses a control flow statement which contains a synchronization"
          " point in the body. Such behaviors are not yet supported. Instead,"
          " please avoid nested `tf.function`s or control flow statements that"
          " may potentially cross a synchronization boundary, for example,"
          " wrap the `fn` passed to `strategy.run` or the entire `strategy.run`"
          " inside a `tf.function` or move the control flow out of `fn`")

    t.has_paused.set()
    t.should_run.wait()
    t.should_run.clear()
    if t.coord.should_stop():
      raise _RequestedStop()
    return t.merge_result

  @property
  def devices(self):
    distribute_lib.require_replica_context(self)
    replica_id = tensor_util.constant_value(self._replica_id_in_sync_group)
    return [self._strategy.extended.worker_devices_by_replica[replica_id]]
