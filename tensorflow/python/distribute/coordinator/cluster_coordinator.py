# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module for `ClusterCoordinator` and relevant cluster-worker related library.

This is currently under development and the API is subject to change.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import enum
import functools
import os
import re
import threading
import time
import weakref
from six.moves import queue

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

# Maximum time for failed worker to come back is 1 hour
_WORKER_MAXIMUM_RECOVERY_SEC = 3600

# Maximum size for queued closures, "infinite" if set to 0.
# When the maximum queue size is reached, further schedule calls will become
# blocking until some previously queued closures are executed on workers.
# Note that using an "infinite" queue size can take a non-trivial portion of
# memory, and even lead to coordinator OOM. Modify the size to a smaller value
# for coordinator with constrained memory resource (only recommended for
# advanced users). Also used in unit tests to ensure the correctness when the
# queue is full.
_CLOSURE_QUEUE_MAX_SIZE = 256 * 1024

# RPC error message from PS
_RPC_ERROR_FROM_PS = "GRPC error information from remote target /job:ps"

# InvalidArgumentError (unknown device) will not have "GRPC error..." string.
_JOB_WORKER_STRING_IDENTIFIER = "/job:worker"


class _RemoteValueStatus(enum.Enum):
  """The status of a `RemoteValue` object.

  A `RemoteValue` object can have three states:
    1) not ready: no value, no non-retryable error and not aborted;
    2) aborted: i.e. the execution of function was aborted because of task
       failure, but can be retried;
    3) ready: i.e. has value or has non-tryable error;

  The initial state of a `RemoteValue` is "not ready". When its corresponding
  closure has
  been executed at least once, it will become aborted or ready. The state
  transitions are:
    1) not ready -> 2) aborted:
      when the corresponding closure is aborted due to worker failure, and the
      worker failure is not immediately handled.
    1) not ready -> 3) ready:
      when the corresponding closure has been executed successfully.
    2) aborted -> 3) ready:
      when the `RemoteValue` is rebuilt by rerunning the corresponding closure
      and the closure has been executed successfully.
    3) ready -> 2) aborted:
      when the corresponding closure had been executed successfully but later
      the corresponding remote worker failed. This is currently only implemented
      for resource `RemoteValue` like iterators.
  """
  NOT_READY = "NOT_READY"
  ABORTED = "ABORTED"
  READY = "READY"


@tf_export("distribute.experimental.coordinator.RemoteValue", v1=[])
class RemoteValue(object):
  """An asynchronously available value of a scheduled function.

  This class is used as the return value of
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` where
  the underlying value becomes available at a later time once the function has
  been executed.

  Using `tf.distribute.experimental.coordinator.RemoteValue` as an input to
  a subsequent function scheduled with
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` is
  currently not supported.

  Example:

  ```python
  strategy = tf.distribute.experimental.ParameterServerStrategy(
      cluster_resolver=...)
  coordinator = (
      tf.distribute.experimental.coordinator.ClusterCoordinator(strategy))

  with strategy.scope():
    v1 = tf.Variable(initial_value=0.0)
    v2 = tf.Variable(initial_value=1.0)

  @tf.function
  def worker_fn():
    v1.assign_add(0.1)
    v2.assign_sub(0.2)
    return v1.read_value() / v2.read_value()

  result = coordinator.schedule(worker_fn)
  # Note that `fetch()` gives the actual result instead of a `tf.Tensor`.
  assert result.fetch() == 0.125

  for _ in range(10):
    # `worker_fn` will be run on arbitrary workers that are available. The
    # `result` value will be available later.
    result = coordinator.schedule(worker_fn)
  ```
  """

  def fetch(self):
    """Wait for the result of `RemoteValue` to be ready and return the result.

    This makes the value concrete by copying the remote value to local.

    Returns:
      The actual output of the `tf.function` associated with this `RemoteValue`,
      previously by a
      `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule` call.
      This can be a single value, or a structure of values, depending on the
      output of the `tf.function`.

    Raises:
      tf.errors.CancelledError: If the function that produces this `RemoteValue`
        is aborted or cancelled due to failure.
    """
    raise NotImplementedError("Must be implemented in subclasses.")


class RemoteValueImpl(RemoteValue):
  """Implementation of `RemoteValue`."""

  def __init__(self, closure, type_spec):  # pylint: disable=super-init-not-called
    """Initializes a `RemoteValueImpl`.

    Args:
      closure: The closure from which the `RemoteValue` is created.
      type_spec: The type spec for this `RemoteValue` which is used to trace
        functions that take this `RemoteValue` as input.
    """
    self._closure = closure
    self._type_spec = type_spec
    self._values = None
    self._fetched_numpys = None
    self._error = None
    self._status_available_event = threading.Event()
    self._status = _RemoteValueStatus.NOT_READY

  def _set_aborted(self):
    self._status = _RemoteValueStatus.ABORTED
    self._values = None
    self._error = None

    # Wake up any waiting thread and clear the event.
    self._status_available_event.set()

  def _rebuild_on(self, worker):
    self._status_available_event.clear()
    # TODO(yuefengz): we may need to rebuild its inputs as well.
    self._closure.execute_on(worker)

  def _set_values(self, tensors):
    self._status = _RemoteValueStatus.READY
    self._values = tensors
    self._error = None
    self._status_available_event.set()

  def _set_error(self, exception):
    self._status = _RemoteValueStatus.READY
    self._values = None
    self._error = exception
    self._status_available_event.set()

  def _get_values(self):
    self._status_available_event.wait()
    return self._values

  def _get_error(self):
    self._status_available_event.wait()
    return self._error

  def fetch(self):
    self._status_available_event.wait()
    if self._status is _RemoteValueStatus.ABORTED:
      raise errors.CancelledError(
          None, None,
          "The corresponding function is aborted. Please reschedule the "
          "function.")
    if self._error is not None:
      raise self._error
    if self._fetched_numpys is None:
      self._fetched_numpys = nest.map_structure(
          lambda x: x.numpy() if hasattr(x, "numpy") else x, self._values)
    return self._fetched_numpys


class InputError(Exception):

  def __init__(self, original_exception):
    message = ("Input has an error, the original exception is %r, "
               "error message is %s." %
               (original_exception, str(original_exception)))
    super().__init__(message)


def _maybe_rebuild_remote_values(worker, structure):
  """Attempts to return errors from `RemoteValue`s. Rebuilds them if needed."""
  errors_in_structure = []

  def _get_error(val):
    if isinstance(val, RemoteValue):
      if val._status is _RemoteValueStatus.ABORTED:  # pylint: disable=protected-access
        try:
          with worker.failure_handler.wait_on_failure(
              on_recovery_fn=functools.partial(val._rebuild_on, worker),  # pylint: disable=protected-access
              worker_device_name=worker.device_name):
            val._rebuild_on(worker)  # pylint: disable=protected-access
        except Exception as e:  # pylint: disable=broad-except
          val._set_error(e)  # pylint: disable=protected-access

      error = val._get_error()  # pylint: disable=protected-access
      if error:
        errors_in_structure.append(error)

  nest.map_structure(_get_error, structure)
  if errors_in_structure:
    return errors_in_structure[0]
  else:
    return None


def _maybe_get_remote_value(val):
  """Gets the value of `val` if it is a `RemoteValue`."""
  if isinstance(val, RemoteValue):
    error = val._get_error()  # pylint: disable=protected-access
    if error:
      raise AssertionError(
          "RemoteValue doesn't have a value because it has errors.")
    else:
      return val._get_values()  # pylint: disable=protected-access
  else:
    return val


def _maybe_as_type_spec(val):
  if isinstance(val, RemoteValue):
    if val._type_spec is None:  # pylint: disable=protected-access
      raise ValueError("Output of a scheduled function that is not "
                       "tf.function cannot be the input of another function.")
    return val._type_spec  # pylint: disable=protected-access
  else:
    return val


@tf_export("distribute.experimental.coordinator.PerWorkerValues", v1=[])
class PerWorkerValues(object):
  """A container that holds a list of values, one value per worker.

  `tf.distribute.experimental.coordinator.PerWorkerValues` contains a collection
  of values, where each of the values is located on its corresponding worker,
  and upon being used as one of the `args` or `kwargs` of
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule()`, the
  value specific to a worker will be passed into the function being executed at
  that corresponding worker.

  Currently, the only supported path to create an object of
  `tf.distribute.experimental.coordinator.PerWorkerValues` is through calling
  `iter` on a `ClusterCoordinator.create_per_worker_dataset`-returned
  distributed dataset instance. The mechanism to create a custom
  `tf.distribute.experimental.coordinator.PerWorkerValues` is not yet supported.
  """

  def __init__(self, values):
    self._values = tuple(values)


def _select_worker_slice(worker_id, structured):
  """Selects the worker slice of each of the items in `structured`."""

  def _get(x):
    return x._values[worker_id] if isinstance(x, PerWorkerValues) else x  # pylint: disable=protected-access

  return nest.map_structure(_get, structured)


def _disallow_remote_value_as_input(structured):
  """Raises if any element of `structured` is a RemoteValue."""

  def _raise_if_remote_value(x):
    if isinstance(x, RemoteValue):
      raise ValueError(
          "`tf.distribute.experimental.coordinator.RemoteValue` used "
          "as an input to scheduled function is not yet "
          "supported.")

  nest.map_structure(_raise_if_remote_value, structured)


class Closure(object):
  """Hold a function to be scheduled and its arguments."""

  def __init__(self, function, cancellation_mgr, args=None, kwargs=None):
    if not callable(function):
      raise ValueError("Function passed to `ClusterCoordinator.schedule` must "
                       "be a callable object.")
    self._args = args or ()
    self._kwargs = kwargs or {}

    _disallow_remote_value_as_input(self._args)
    _disallow_remote_value_as_input(self._kwargs)

    if isinstance(function, def_function.Function):
      replica_args = _select_worker_slice(0, self._args)
      replica_kwargs = _select_worker_slice(0, self._kwargs)

      # Note: no need to handle function registration failure since this kind of
      # failure will not raise exceptions as designed in the runtime. The
      # coordinator has to rely on subsequent operations that raise to catch
      # function registration failure.

      # Record the function tracing overhead. Note that we pass in the tracing
      # count of the def_function.Function as a state tracker, so that metrics
      # will only record the time for actual function tracing (i.e., excluding
      # function cache lookups).
      with metric_utils.monitored_timer(
          "function_tracing", state_tracker=function._get_tracing_count):  # pylint: disable=protected-access
        self._concrete_function = function.get_concrete_function(
            *nest.map_structure(_maybe_as_type_spec, replica_args),
            **nest.map_structure(_maybe_as_type_spec, replica_kwargs))
    elif isinstance(function, tf_function.ConcreteFunction):
      self._concrete_function = function

    if hasattr(self, "_concrete_function"):
      # If we have a concrete function, we get to retrieve the output type spec
      # via the structured_output.
      output_type_spec = func_graph.convert_structure_to_signature(
          self._concrete_function.structured_outputs)
      self._function = cancellation_mgr.get_cancelable_function(
          self._concrete_function)
    else:
      # Otherwise (i.e. what is passed in is a regular python function), we have
      # no such information.
      output_type_spec = None
      self._function = function

    self.output_remote_value = RemoteValueImpl(self, output_type_spec)

  def mark_cancelled(self):
    self.output_remote_value._set_error(  # pylint: disable=protected-access
        errors.CancelledError(
            None, None, "The corresponding function is "
            "cancelled. Please reschedule the function."))

  def execute_on(self, worker):
    """Executes the closure on the given worker.

    Args:
      worker: a `Worker` object.
    """
    replica_args = _select_worker_slice(worker.worker_index, self._args)
    replica_kwargs = _select_worker_slice(worker.worker_index, self._kwargs)

    e = (
        _maybe_rebuild_remote_values(worker, replica_args) or
        _maybe_rebuild_remote_values(worker, replica_kwargs))
    if e:
      if not isinstance(e, InputError):
        e = InputError(e)
      self.output_remote_value._set_error(e)  # pylint: disable=protected-access
      return

    with ops.device(worker.device_name):
      with context.executor_scope(worker.executor):
        with metric_utils.monitored_timer("closure_execution"):
          output_values = self._function(
              *nest.map_structure(_maybe_get_remote_value, replica_args),
              **nest.map_structure(_maybe_get_remote_value, replica_kwargs))
    self.output_remote_value._set_values(output_values)  # pylint: disable=protected-access


class _CoordinatedClosureQueue(object):
  """Manage a queue of closures, inflight count and errors from execution.

  This class is thread-safe.
  """

  def __init__(self):
    # `self._inflight_closure_count` only tracks the number of inflight closures
    # that are "in generation". Once an error occurs, error generation is
    # incremented and all subsequent arriving closures (from inflight) are
    # considered "out of generation".
    self._inflight_closure_count = 0

    self._queue_lock = threading.Lock()

    # Condition indicating that all pending closures (either queued or inflight)
    # have been processed, failed, or cancelled.
    self._stop_waiting_condition = threading.Condition(self._queue_lock)

    # Condition indicating that an item becomes available in queue (not empty).
    self._closures_queued_condition = threading.Condition(self._queue_lock)
    self._should_process_closures = True

    # Condition indicating that a queue slot becomes available (not full).
    # Note that even with "infinite" queue size, there is still a "practical"
    # size limit for the queue depending on host memory capacity, and thus the
    # queue will eventually become full with a lot of enqueued closures.
    self._queue_free_slot_condition = threading.Condition(self._queue_lock)

    # Condition indicating there is no inflight closures.
    self._no_inflight_closure_condition = threading.Condition(self._queue_lock)

    # Use to cancel in-flight closures.
    self._cancellation_mgr = cancellation.CancellationManager()

    if _CLOSURE_QUEUE_MAX_SIZE <= 0:
      logging.warning(
          "In a `ClusterCoordinator`, creating an infinite closure queue can "
          "consume a significant amount of memory and even lead to OOM.")
    self._queue = queue.Queue(maxsize=_CLOSURE_QUEUE_MAX_SIZE)
    self._error = None

    # The following is a lock to make sure when `wait` is called and before it
    # returns no `put` can be executed during this period. It is because `wait`
    # won't know what to do with newly put closures. This lock adds an cutoff
    # for `wait` so that closures put into the queue while waiting would not be
    # taken responsible by this `wait`.
    #
    # We cannot reuse the `self._queue_lock` since when `wait` waits for a
    # condition, the `self._queue_lock` will be released.
    #
    # We don't use a reader/writer's lock on purpose to reduce the complexity
    # of the code.
    self._put_wait_lock = threading.Lock()

  def stop(self):
    with self._queue_lock:
      self._should_process_closures = False
      self._closures_queued_condition.notifyAll()

  def _cancel_all_closures(self):
    """Clears the queue and sets remaining closures cancelled error.

    This method expects self._queue_lock to be held prior to entry.
    """
    self._cancellation_mgr.start_cancel()
    while self._inflight_closure_count > 0:
      self._no_inflight_closure_condition.wait()
    while True:
      try:
        closure = self._queue.get(block=False)
        self._queue_free_slot_condition.notify()
        closure.mark_cancelled()
      except queue.Empty:
        break
    # The cancellation manager cannot be reused once cancelled. After all
    # closures (queued or inflight) are cleaned up, recreate the cancellation
    # manager with clean state.
    # Note on thread-safety: this is triggered when one of theses
    # ClusterCoordinator APIs are called: `schedule`, `wait`, and `done`. At the
    # same time, no new closures can be constructed (which reads the
    # _cancellation_mgr to get cancellable functions).
    self._cancellation_mgr = cancellation.CancellationManager()

  def _raise_if_error(self):
    """Raises the error if one exists.

    If an error exists, cancel the closures in queue, raises it, and clear
    the error.

    This method expects self._queue_lock to be held prior to entry.
    """
    if self._error:
      logging.error("Start cancelling closures due to error %r: %s",
                    self._error, self._error)
      self._cancel_all_closures()
      try:
        raise self._error  # pylint: disable=raising-bad-type
      finally:
        self._error = None

  def put(self, closure):
    """Put a closure into the queue for later execution.

    If `mark_failed` was called before `put`, the error from the first
    invocation of `mark_failed` will be raised.

    Args:
      closure: The `Closure` to put into the queue.
    """
    with self._put_wait_lock, self._queue_lock:
      self._queue_free_slot_condition.wait_for(lambda: not self._queue.full())
      self._queue.put(closure, block=False)
      self._raise_if_error()
      self._closures_queued_condition.notify()

  def get(self, timeout=None):
    """Return a closure from the queue to be executed."""
    with self._queue_lock:
      while self._queue.empty() and self._should_process_closures:
        if not self._closures_queued_condition.wait(timeout=timeout):
          return None
      if not self._should_process_closures:
        return None
      closure = self._queue.get(block=False)
      self._queue_free_slot_condition.notify()
      self._inflight_closure_count += 1
      return closure

  def mark_finished(self):
    """Let the queue know that a closure has been successfully executed."""
    with self._queue_lock:
      if self._inflight_closure_count < 1:
        raise AssertionError("There is no inflight closures to mark_finished.")
      self._inflight_closure_count -= 1
      if self._inflight_closure_count == 0:
        self._no_inflight_closure_condition.notifyAll()
      if self._queue.empty() and self._inflight_closure_count == 0:
        self._stop_waiting_condition.notifyAll()

  def put_back(self, closure):
    """Put the closure back into the queue as it was not properly executed."""
    with self._queue_lock:
      if self._inflight_closure_count < 1:
        raise AssertionError("There is no inflight closures to put_back.")
      if self._error:
        closure.mark_cancelled()
      else:
        self._queue_free_slot_condition.wait_for(lambda: not self._queue.full())
        self._queue.put(closure, block=False)
        self._closures_queued_condition.notify()
      self._inflight_closure_count -= 1
      if self._inflight_closure_count == 0:
        self._no_inflight_closure_condition.notifyAll()

  def wait(self, timeout=None):
    """Wait for all closures to be finished before returning.

    If `mark_failed` was called before or during `wait`, the error from the
    first invocation of `mark_failed` will be raised.

    Args:
      timeout: A float specifying a timeout for the wait in seconds.

    Returns:
      True unless the given timeout expired, in which case it returns False.
    """
    with self._put_wait_lock, self._queue_lock:
      while (not self._error and
             (not self._queue.empty() or self._inflight_closure_count > 0)):
        if not self._stop_waiting_condition.wait(timeout=timeout):
          return False
      self._raise_if_error()
      return True

  def mark_failed(self, e):
    """Sets error and unblocks any wait() call."""
    with self._queue_lock:
      # TODO(yuefengz): maybe record all failure and give users more
      # information?
      if self._inflight_closure_count < 1:
        raise AssertionError("There is no inflight closures to mark_failed.")
      if self._error is None:
        self._error = e
      self._inflight_closure_count -= 1
      if self._inflight_closure_count == 0:
        self._no_inflight_closure_condition.notifyAll()
      self._stop_waiting_condition.notifyAll()

  def done(self):
    """Returns true if the queue is empty and there is no inflight closure.

    If `mark_failed` was called before `done`, the error from the first
    invocation of `mark_failed` will be raised.
    """
    with self._queue_lock:
      self._raise_if_error()
      return self._queue.empty() and self._inflight_closure_count == 0


class WorkerPreemptionHandler(object):
  """Handles worker preemptions."""

  def __init__(self, server_def, cluster):
    self._server_def = server_def
    self._cluster = cluster
    self._cluster_update_lock = threading.Lock()
    self._cluster_due_for_update_or_finish = threading.Event()
    self._worker_up_cond = threading.Condition(self._cluster_update_lock)
    self._error_from_recovery = None
    self._should_preemption_thread_run = True
    self._preemption_handler_thread = threading.Thread(
        target=self._preemption_handler,
        name="WorkerPreemptionHandler",
        daemon=True)
    self._preemption_handler_thread.start()

  def stop(self):
    """Ensure the worker preemption thread is closed."""
    self._should_preemption_thread_run = False
    with self._cluster_update_lock:
      self._cluster_due_for_update_or_finish.set()
    # TODO(yuefengz): The preemption handler thread shouldn't be terminated
    # asynchronously since it touches eager context which is a process-wide
    # singleton. The problem is in OSS unit tests will time out.

  def _validate_preemption_failure(self, e):
    """Validates that the given exception represents worker preemption."""
    if _is_worker_failure(e):
      return
    raise e

  @contextlib.contextmanager
  def wait_on_failure(self,
                      on_failure_fn=None,
                      on_recovery_fn=None,
                      worker_device_name="(unknown)"):
    """Catches worker preemption error and wait until failed workers are back.

    Args:
      on_failure_fn: an optional function to run if preemption happens.
      on_recovery_fn: an optional function to run when a worker is recovered
        from preemption.
      worker_device_name: the device name of the worker instance that is passing
        through the failure.

    Yields:
      None.
    """
    assert self._should_preemption_thread_run
    try:
      yield
    except errors.OpError as e:
      # If the error is due to temporary connectivity issues between worker and
      # ps, put back closure, ignore error and do not mark worker as failure.
      if self._cluster._record_and_ignore_transient_ps_failure(e):  # pylint: disable=protected-access
        if on_failure_fn:
          on_failure_fn()
        return

      self._validate_preemption_failure(e)
      logging.error("Worker %s failed with error: %s", worker_device_name, e)
      if on_failure_fn:
        on_failure_fn()

      with self._cluster_update_lock:
        self._cluster_due_for_update_or_finish.set()
        self._worker_up_cond.wait(_WORKER_MAXIMUM_RECOVERY_SEC)
        if self._error_from_recovery:
          # TODO(yuefengz): there is only one worker that will get this error.
          # Ideally we shuold let all workers notified by `_worker_up_cond` get
          # this error.
          try:
            raise self._error_from_recovery
          finally:
            self._error_from_recovery = None
        logging.info("Worker %s has been recovered.", worker_device_name)

      if on_recovery_fn:
        with self.wait_on_failure(
            on_recovery_fn=on_recovery_fn,
            worker_device_name=worker_device_name):
          on_recovery_fn()

  def _preemption_handler(self):
    """A loop that handles preemption.

    This loop waits for signal of worker preemption and upon worker preemption,
    it waits until all workers are back and updates the cluster about the
    restarted workers.
    """
    assert self._should_preemption_thread_run
    while True:
      self._cluster_due_for_update_or_finish.wait()
      if not self._should_preemption_thread_run:
        logging.info("Stopping the failure handing thread.")
        break

      with self._cluster_update_lock:
        try:
          # TODO(haoyuzhang): support partial cluster recovery
          logging.info("Cluster now being recovered.")
          context.context().update_server_def(self._server_def)

          # Cluster updated successfully, clear the update signal, and notify
          # all workers that they are recovered from failure.
          logging.info("Cluster successfully recovered.")
          self._worker_up_cond.notify_all()
          # The check for _should_preemption_thread_run is necessary since the
          # `stop` may have already set _cluster_due_for_update_or_finish.
          if self._should_preemption_thread_run:
            self._cluster_due_for_update_or_finish.clear()
        except Exception as e:  # pylint: disable=broad-except
          try:
            self._validate_preemption_failure(e)
          except Exception as e:  # pylint: disable=broad-except
            # In this case, a parameter server fails. So we raise this error to
            # the caller of `wait_on_failure`.
            self._error_from_recovery = e
            self._worker_up_cond.notify_all()
            if self._should_preemption_thread_run:
              self._cluster_due_for_update_or_finish.clear()
          # NOTE: Since the first RPC (GetStatus) of update_server_def is
          # currently blocking by default, error should only happen if:
          # (1) More workers failed while waiting for the previous workers to
          #     come back;
          # (2) Worker failed when exchanging subsequent RPCs after the first
          #     RPC returns.
          # Consider adding backoff retry logic if we see the error logged
          # too frequently.
          logging.error("Cluster update failed with error: %s. Retrying...", e)


class Worker(object):
  """A worker in a cluster.

  Attributes:
    worker_index: The index of the worker in the cluster.
    device_name: The device string of the worker, e.g. "/job:worker/task:1".
    executor: The worker's executor for remote function execution.
    failure_handler: The failure handler used to handler worker preemption
      failure.
  """

  def __init__(self, worker_index, device_name, cluster):
    self.worker_index = worker_index
    self.device_name = device_name
    self.executor = executor.new_executor(enable_async=False)
    self.failure_handler = cluster.failure_handler
    self._cluster = cluster
    self._resource_remote_value_refs = []
    self._should_worker_thread_run = True

    # Worker threads need to start after `Worker`'s initialization.
    threading.Thread(target=self._process_queue,
                     name="WorkerClosureProcessingLoop-%d" % self.worker_index,
                     daemon=True).start()

  def stop(self):
    """Ensure the worker thread is closed."""
    self._should_worker_thread_run = False

  def _set_resources_aborted(self):
    # TODO(yuefengz): maybe we can query whether a tensor is valid or not
    # instead of marking a tensor aborted?
    for weakref_resource in self._resource_remote_value_refs:
      resource = weakref_resource()
      if resource:
        resource._set_aborted()  # pylint: disable=protected-access

  def _set_dead(self):
    raise NotImplementedError("_set_dead is not implemented.")

  def _process_closure(self, closure):
    """Runs a closure with preemption handling."""
    assert closure is not None
    try:
      with self._cluster.failure_handler.wait_on_failure(
          on_failure_fn=lambda: self._cluster._closure_queue.put_back(closure),  # pylint: disable=protected-access
          on_recovery_fn=self._set_resources_aborted,
          worker_device_name=self.device_name):
        closure.execute_on(self)
        # TODO(yuefengz): we don't have to materialize results every step.
        with metric_utils.monitored_timer("remote_value_fetch"):
          closure.output_remote_value.fetch()
        self._cluster._closure_queue.mark_finished()  # pylint: disable=protected-access
    except Exception as e:  # pylint: disable=broad-except
      # Avoid logging the derived cancellation error
      if not isinstance(e, errors.CancelledError):
        logging.error(
            "/job:worker/task:%d encountered the following error when "
            "processing closure: %r:%s", self.worker_index, e, e)
      closure.output_remote_value._set_error(e)  # pylint: disable=protected-access
      self._cluster._closure_queue.mark_failed(e)  # pylint: disable=protected-access

  def _maybe_delay(self):
    """Delay if corresponding env vars are set."""
    # If the following two env vars variables are set. Scheduling for workers
    # will start in a staggered manner. Worker i will wait for
    # `TF_COORDINATOR_SCHEDULE_START_DELAY` * i seconds, not exceeding
    # `TF_COORDINATOR_SCHEDULE_START_DELAY_MAX`.
    delay_secs = int(os.environ.get("TF_COORDINATOR_SCHEDULE_START_DELAY", "0"))
    delay_cap = int(
        os.environ.get("TF_COORDINATOR_SCHEDULE_START_DELAY_MAX", "0"))
    if delay_cap:
      delay_secs = min(delay_secs * self.worker_index, delay_cap)
    if delay_secs > 0:
      logging.info("Worker %d sleeping for %d seconds before running function",
                   self.worker_index, delay_secs)
    time.sleep(delay_secs)

  def _process_queue(self):
    """Function running in a worker thread to process closure queues."""
    self._maybe_delay()
    while self._should_worker_thread_run:
      closure = self._cluster._closure_queue.get()  # pylint: disable=protected-access
      if not self._should_worker_thread_run or closure is None:
        return
      self._process_closure(closure)
      # To properly stop the worker and preemption threads, it is important that
      # `ClusterCoordinator` object is not held onto so its `__del__` can be
      # called. By removing the reference to the `closure` that has already been
      # processed, we ensure that the `closure` object is released, while
      # getting the next `closure` at above `self._cluster._closure_queue.get()`
      # call.
      del closure

  def _create_resource(self, function, args=None, kwargs=None):
    """Synchronously creates a per-worker resource represented by a `RemoteValue`.

    Args:
      function: the resource function to be run remotely. It should be a
        `tf.function`, a concrete function or a Python function.
      args: positional arguments to be passed to the function.
      kwargs: keyword arguments to be passed to the function.

    Returns:
      one or several RemoteValue objects depending on the function return
      values.
    """
    # Some notes about the concurrency: currently all the activities related to
    # the same worker such as creating resources, setting resources' aborted
    # status, and executing closures happen on the same thread. This allows us
    # to have simpler logic of concurrency.
    closure = Closure(
        function,
        self._cluster._closure_queue._cancellation_mgr,  # pylint: disable=protected-access
        args=args,
        kwargs=kwargs)
    resource_remote_value = closure.output_remote_value
    self._register_resource(resource_remote_value)

    # The following is a short-term solution to lazily create resources in
    # parallel.
    # TODO(b/160343165): we should create resources eagerly, i.e. schedule the
    # resource creation function as soon as users call this method.
    resource_remote_value._set_aborted()  # pylint: disable=protected-access
    return resource_remote_value

  def _register_resource(self, resource_remote_value):
    if not isinstance(resource_remote_value, RemoteValue):
      raise ValueError("Resource being registered is not of type "
                       "`tf.distribute.experimental.coordinator.RemoteValue`.")
    self._resource_remote_value_refs.append(weakref.ref(resource_remote_value))


class Cluster(object):
  """A cluster with workers.

  We assume all function errors are fatal and based on this assumption our
  error reporting logic is:
  1) Both `schedule` and `join` can raise a non-retryable error which is the
  first error seen by the coordinator from any previously scheduled functions.
  2) When an error is raised, there is no guarantee on how many previously
  scheduled functions have been executed; functions that have not been executed
  will be thrown away and marked as cancelled.
  3) After an error is raised, the internal state of error will be cleared.
  I.e. functions can continue to be scheduled and subsequent calls of `schedule`
  or `join` will not raise the same error again.

  Attributes:
    failure_handler: The failure handler used to handler worker preemption
      failure.
    workers: a list of `Worker` objects in the cluster.
  """

  def __init__(self, strategy):
    """Initializes the cluster instance."""

    self._num_workers = strategy._num_workers
    self._num_ps = strategy._num_ps

    # Ignore PS failures reported by workers due to transient connection errors.
    # Transient connectivity issues between workers and PS are relayed by the
    # workers to the coordinator, leading the coordinator to believe that there
    # are PS failures. The difference between transient vs. permanent PS failure
    # is the number of reports from the workers. When this env var is set to a
    # positive integer K, the coordinator ignores up to K reports of a failed PS
    # task, i.e., only when there are more than K trials of executing closures
    # fail due to errors from the same PS instance do we consider the PS
    # instance encounters a failure.
    # TODO(b/164279603): Remove this workaround when the underlying connectivity
    # issue in gRPC server is resolved.
    self._transient_ps_failures_threshold = int(
        os.environ.get("TF_COORDINATOR_IGNORE_TRANSIENT_PS_FAILURES", 3))
    self._potential_ps_failures_lock = threading.Lock()
    self._potential_ps_failures_count = [0] * self._num_ps

    self._closure_queue = _CoordinatedClosureQueue()
    self.failure_handler = WorkerPreemptionHandler(context.get_server_def(),
                                                   self)
    worker_device_strings = [
        "/job:worker/replica:0/task:%d" % i for i in range(self._num_workers)
    ]
    self.workers = [
        Worker(i, w, self) for i, w in enumerate(worker_device_strings)
    ]

  def stop(self):
    """Stop worker, worker preemption threads, and the closure queue."""
    self.failure_handler.stop()

    for worker in self.workers:
      worker.stop()
    self._closure_queue.stop()

  def _record_and_ignore_transient_ps_failure(self, e):
    """Records potential PS failures and return if failure should be ignored."""
    if self._transient_ps_failures_threshold <= 0 or not _is_ps_failure(e):
      return False

    ps_tasks = _extract_failed_ps_instances(str(e))
    with self._potential_ps_failures_lock:
      for t in ps_tasks:
        self._potential_ps_failures_count[t] += 1
        # The number of UnavailableError encountered on this PS task exceeds the
        # maximum number of ignored error
        if (self._potential_ps_failures_count[t] >=
            self._transient_ps_failures_threshold):
          return False
    return True

  def schedule(self, function, args, kwargs):
    """Schedules `function` to be dispatched to a worker for execution.

    Args:
      function: The function to be dispatched to a worker for execution
        asynchronously.
      args: Positional arguments for `fn`.
      kwargs: Keyword arguments for `fn`.

    Returns:
      A `RemoteValue` object.
    """
    closure = Closure(
        function,
        self._closure_queue._cancellation_mgr,  # pylint: disable=protected-access
        args=args,
        kwargs=kwargs)
    self._closure_queue.put(closure)
    return closure.output_remote_value

  def join(self):
    """Blocks until all scheduled functions are executed."""
    self._closure_queue.wait()

  def done(self):
    """Returns true if all scheduled functions are executed."""
    return self._closure_queue.done()


@tf_export("distribute.experimental.coordinator.ClusterCoordinator", v1=[])
class ClusterCoordinator(object):
  """An object to schedule and coordinate remote function execution.

  This class is used to create fault-tolerant resources and dispatch functions
  to remote TensorFlow servers.

  Currently, this class is not supported to be used in a standalone manner. It
  should be used in conjunction with a `tf.distribute` strategy that is designed
  to work with it. The `ClusterCoordinator` class currently only works
  `tf.distribute.experimental.ParameterServerStrategy`.

  __The `schedule`/`join` APIs__

  The most important APIs provided by this class is the `schedule`/`join` pair.
  The `schedule` API is non-blocking in that it queues a `tf.function` and
  returns a `RemoteValue` immediately. The queued functions will be dispatched
  to remote workers in background threads and their `RemoteValue`s will be
  filled asynchronously. Since `schedule` doesn’t require worker assignment, the
  `tf.function` passed in can be executed on any available worker. If the worker
  it is executed on becomes unavailable before its completion, it will be
  migrated to another worker. Because of this fact and function execution is not
  atomic, a function may be executed more than once.

  __Handling Task Failure__

  This class when used with
  `tf.distribute.experimental.ParameterServerStrategy`, comes with built-in
  fault tolerance for worker failures. That is, when some workers are not
  available for any reason to be reached from the coordinator, the training
  progress continues to be made with the remaining workers. Upon recovery of a
  failed worker, it will be added for function execution after datasets created
  by `create_per_worker_dataset` are re-built on it.

  When a parameter server fails, a `tf.errors.UnavailableError` is raised by
  `schedule`, `join` or `done`. In this case, in addition to bringing back the
  failed parameter server, users should restart the coordinator so that it
  reconnects to workers and parameter servers, re-creates the variables, and
  loads checkpoints. If the coordinator fails, after the user brings it back,
  the program will automatically connect to workers and parameter servers, and
  continue the progress from a checkpoint.

  It is thus essential that in user's program, a checkpoint file is periodically
  saved, and restored at the start of the program. If an
  `tf.keras.optimizers.Optimizer` is checkpointed, after restoring from a
  checkpoiont, its `iterations` property roughly indicates the number of steps
  that have been made. This can be used to decide how many epochs and steps are
  needed before the training completion.

  See `tf.distribute.experimental.ParameterServerStrategy` docstring for an
  example usage of this API.

  This is currently under development, and the API as well as implementation
  are subject to changes.
  """

  def __new__(cls, strategy):
    # `ClusterCoordinator` is kept as a single instance to a given `Strategy`.
    # TODO(rchao): Needs a lock for thread-safety
    if strategy._cluster_coordinator is None:
      strategy._cluster_coordinator = super(
          ClusterCoordinator, cls).__new__(cls)
    return strategy._cluster_coordinator

  def __init__(self, strategy):
    """Initialization of a `ClusterCoordinator` instance.

    Args:
      strategy: a supported `tf.distribute.Strategy` object. Currently, only
        `tf.distribute.experimental.ParameterServerStrategy` is supported.

    Raises:
      ValueError: if the strategy being used is not supported.
    """
    if not isinstance(strategy,
                      parameter_server_strategy_v2.ParameterServerStrategyV2):
      raise ValueError(
          "Only `tf.distribute.experimental.ParameterServerStrategy` "
          "is supported to work with "
          "`tf.distribute.experimental.coordinator.ClusterCoordinator` "
          "currently.")
    self._strategy = strategy
    self.strategy.extended._used_with_coordinator = True
    self._cluster = Cluster(strategy)

  def __del__(self):
    self._cluster.stop()

  @property
  def strategy(self):
    """Returns the `Strategy` associated with the `ClusterCoordinator`."""
    return self._strategy

  def schedule(self, fn, args=None, kwargs=None):
    """Schedules `fn` to be dispatched to a worker for asynchronous execution.

    This method is non-blocking in that it queues the `fn` which will be
    executed later and returns a
    `tf.distribute.experimental.coordinator.RemoteValue` object immediately.
    `fetch` can be called on it to wait for the function execution to finish
    and retrieve its output from a remote worker. On the other hand, call
    `tf.distribute.experimental.coordinator.ClusterCoordinator.join` to wait for
    all scheduled functions to finish.

    `schedule` guarantees that `fn` will be executed on a worker at least once;
    it could be more than once if its corresponding worker fails in the middle
    of its execution. Note that since worker can fail at any point when
    executing the function, it is possible that the function is partially
    executed, but `tf.distribute.experimental.coordinator.ClusterCoordinator`
    guarantees that in those events, the function will eventually be executed on
    any worker that is available.

    If any previously scheduled function raises an error, `schedule` will raise
    any one of those errors, and clear the errors collected so far. What happens
    here, some of the previously scheduled functions may have not been executed.
    User can call `fetch` on the returned
    `tf.distribute.experimental.coordinator.RemoteValue` to inspect if they have
    executed, failed, or cancelled, and reschedule the corresponding function if
    needed.

    When `schedule` raises, it guarantees that there is no function that is
    still being executed.

    At this time, there is no support of worker assignment for function
    execution, or priority of the workers.

    `args` and `kwargs` are the arguments passed into `fn`, when `fn` is
    executed on a worker. They can be
    `tf.distribute.experimental.coordinator.PerWorkerValues` and in this case,
    the argument will be substituted with the corresponding component on the
    target worker. Arguments that are not
    `tf.distribute.experimental.coordinator.PerWorkerValues` will be passed into
    `fn` as-is. Currently, `tf.distribute.experimental.coordinator.RemoteValue`
    is not supported to be input `args` or `kwargs`.

    Args:
      fn: A `tf.function`; the function to be dispatched to a worker for
        execution asynchronously. Regular python funtion is not supported to be
        scheduled.
      args: Positional arguments for `fn`.
      kwargs: Keyword arguments for `fn`.

    Returns:
      A `tf.distribute.experimental.coordinator.RemoteValue` object that
      represents the output of the function scheduled.

    Raises:
      Exception: one of the exceptions caught by the coordinator from any
        previously scheduled function, since the last time an error was thrown
        or since the beginning of the program.
    """
    if not isinstance(fn,
                      (def_function.Function, tf_function.ConcreteFunction)):
      raise TypeError(
          "`tf.distribute.experimental.coordinator.ClusterCoordinator.schedule`"
          " only accepts a `tf.function` or a concrete function.")
    # Slot variables are usually created during function tracing time; thus
    # `schedule` needs to be called within the `strategy.scope()`.
    with self.strategy.scope():
      self.strategy.extended._being_scheduled = True  # pylint: disable=protected-access
      remote_value = self._cluster.schedule(fn, args=args, kwargs=kwargs)
      self.strategy.extended._being_scheduled = False  # pylint: disable=protected-access
      return remote_value

  def join(self):
    """Blocks until all the scheduled functions have finished execution.

    If any previously scheduled function raises an error, `join` will fail by
    raising any one of those errors, and clear the errors collected so far. If
    this happens, some of the previously scheduled functions may have not been
    executed. Users can call `fetch` on the returned
    `tf.distribute.experimental.coordinator.RemoteValue` to inspect if they have
    executed, failed, or cancelled. If some that have been cancelled need to be
    rescheduled, users should call `schedule` with the function again.

    When `join` returns or raises, it guarantees that there is no function that
    is still being executed.

    Raises:
      Exception: one of the exceptions caught by the coordinator by any
        previously scheduled function since the last time an error was thrown or
        since the beginning of the program.
    """
    self._cluster.join()

  def done(self):
    """Returns whether all the scheduled functions have finished execution.

    If any previously scheduled function raises an error, `done` will fail by
    raising any one of those errors.

    When `done` returns True or raises, it guarantees that there is no function
    that is still being executed.

    Returns:
      Whether all the scheduled functions have finished execution.
    Raises:
      Exception: one of the exceptions caught by the coordinator by any
        previously scheduled function since the last time an error was thrown or
        since the beginning of the program.
    """
    return self._cluster.done()

  def create_per_worker_dataset(self, dataset_fn):
    """Create dataset on workers by calling `dataset_fn` on worker devices.

    This creates the given dataset generated by dataset_fn on workers
    and returns an object that represents the collection of those individual
    datasets. Calling `iter` on such collection of datasets returns a
    `tf.distribute.experimental.coordinator.PerWorkerValues`, which is a
    collection of iterators, where the iterators have been placed on respective
    workers.

    Calling `next` on a `PerWorkerValues` of iterator is unsupported. The
    iterator is meant to be passed as an argument into
    `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule`. When
    the scheduled function is about to be executed by a worker, the
    function will receive the individual iterator that corresponds to the
    worker. The `next` method can be called on an iterator inside a
    scheduled function when the iterator is an input of the function.

    Currently the `schedule` method assumes workers are all the same and thus
    assumes the datasets on different workers are the same, except they may be
    shuffled differently if they contain a `dataset.shuffle` operation and a
    random seed is not set. Because of this, we also recommend the datasets to
    be repeated indefinitely and schedule a finite number of steps instead of
    relying on the `OutOfRangeError` from a dataset.


    Example:

    ```python
    strategy = tf.distribute.experimental.ParameterServerStrategy(
        cluster_resolver=...)
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy=strategy)

    @tf.function
    def worker_fn(iterator):
      return next(iterator)

    def per_worker_dataset_fn():
      return strategy.distribute_datasets_from_function(
          lambda x: tf.data.Dataset.from_tensor_slices([3] * 3))

    per_worker_dataset = coordinator.create_per_worker_dataset(
        per_worker_dataset_fn)
    per_worker_iter = iter(per_worker_dataset)
    remote_value = coordinator.schedule(worker_fn, args=(per_worker_iter,))
    assert remote_value.fetch() == 3
    ```

    NOTE: A known limitation is `tf.data.Options` is ignored in dataset created
    by `create_per_worker_dataset`.

    Args:
      dataset_fn: The dataset function that returns a dataset. This is to be
        executed on the workers.

    Returns:
      An object that represents the collection of those individual
      datasets. `iter` is expected to be called on this object that returns
      a `tf.distribute.experimental.coordinator.PerWorkerValues` of the
      iterators (that are on the workers).
    """
    input_workers = input_lib.InputWorkers([
        (w.device_name, [w.device_name]) for w in self._cluster.workers
    ])

    return _PerWorkerDistributedDataset(dataset_fn, input_workers, self)

  def _create_per_worker_resources(self, fn, args=None, kwargs=None):
    """Synchronously create resources on the workers.

    The resources are represented by
    `tf.distribute.experimental.coordinator.RemoteValue`s.

    Args:
      fn: The function to be dispatched to all workers for execution
        asynchronously.
      args: Positional arguments for `fn`.
      kwargs: Keyword arguments for `fn`.

    Returns:
      A `tf.distribute.experimental.coordinator.PerWorkerValues` object, which
      wraps a tuple of `tf.distribute.experimental.coordinator.RemoteValue`
      objects.
    """
    results = []
    for w in self._cluster.workers:
      results.append(w._create_resource(fn, args=args, kwargs=kwargs))  # pylint: disable=protected-access
    return PerWorkerValues(tuple(results))

  def fetch(self, val):
    """Blocking call to fetch results from the remote values.

    This is a wrapper around
    `tf.distribute.experimental.coordinator.RemoteValue.fetch` for a
    `RemoteValue` structure; it returns the execution results of
    `RemoteValue`s. If not ready, wait for them while blocking the caller.

    Example:
    ```python
    strategy = ...
    coordinator = tf.distribute.experimental.coordinator.ClusterCoordinator(
        strategy)

    def dataset_fn():
      return tf.data.Dataset.from_tensor_slices([1, 1, 1])

    with strategy.scope():
      v = tf.Variable(initial_value=0)

    @tf.function
    def worker_fn(iterator):
      def replica_fn(x):
        v.assign_add(x)
        return v.read_value()
      return strategy.run(replica_fn, args=(next(iterator),))

    distributed_dataset = coordinator.create_per_worker_dataset(dataset_fn)
    distributed_iterator = iter(distributed_dataset)
    result = coordinator.schedule(worker_fn, args=(distributed_iterator,))
    assert coordinator.fetch(result) == 1
    ```

    Args:
      val: The value to fetch the results from. If this is structure of
        `tf.distribute.experimental.coordinator.RemoteValue`, `fetch()` will be
        called on the individual
        `tf.distribute.experimental.coordinator.RemoteValue` to get the result.

    Returns:
      If `val` is a `tf.distribute.experimental.coordinator.RemoteValue` or a
      structure of `tf.distribute.experimental.coordinator.RemoteValue`s,
      return the fetched `tf.distribute.experimental.coordinator.RemoteValue`
      values immediately if they are available, or block the call until they are
      available, and return the fetched
      `tf.distribute.experimental.coordinator.RemoteValue` values with the same
      structure. If `val` is other types, return it as-is.
    """

    def _maybe_fetch(val):
      if isinstance(val, RemoteValue):
        return val.fetch()
      else:
        return val

    # TODO(yuefengz): we should fetch values in a batch.
    return nest.map_structure(_maybe_fetch, val)


class _PerWorkerDistributedDataset(object):
  """Represents worker-distributed datasets created from dataset function."""

  def __init__(self, dataset_fn, input_workers, coordinator):
    """Makes an iterable from datasets created by the given function.

    Args:
      dataset_fn: A function that returns a `Dataset`.
      input_workers: an `InputWorkers` object.
      coordinator: a `ClusterCoordinator` object, used to create dataset
        resources.
    """
    def disallow_variable_creation(next_creator, **kwargs):
      raise ValueError("Creating variables in `dataset_fn` is not allowed.")

    if isinstance(dataset_fn, def_function.Function):
      with variable_scope.variable_creator_scope(disallow_variable_creation):
        dataset_fn = dataset_fn.get_concrete_function()
    elif not isinstance(dataset_fn, tf_function.ConcreteFunction):
      with variable_scope.variable_creator_scope(disallow_variable_creation):
        dataset_fn = def_function.function(dataset_fn).get_concrete_function()
    self._dataset_fn = dataset_fn
    self._input_workers = input_workers
    self._coordinator = coordinator
    self._element_spec = None

  def __iter__(self):
    # We would like users to create iterators outside `tf.function`s so that we
    # can track them.
    if (not context.executing_eagerly() or
        ops.get_default_graph().building_function):
      raise RuntimeError(
          "__iter__() is not supported inside of tf.function or in graph mode.")

    def _create_per_worker_iterator():
      dataset = self._dataset_fn()
      return iter(dataset)

    # If _PerWorkerDistributedDataset.__iter__ is called multiple
    # times, for the same object it should only create and register resource
    # once. Using object id to distinguish different iterator resources.
    per_worker_iterator = self._coordinator._create_per_worker_resources(
        _create_per_worker_iterator)

    # Setting type_spec of each RemoteValue so that functions taking these
    # RemoteValues as inputs can be traced.
    for iterator_remote_value in per_worker_iterator._values:
      iterator_remote_value._type_spec = (  # pylint: disable=protected-access
          iterator_ops.IteratorSpec(
              self._dataset_fn.structured_outputs.element_spec))
    return _PerWorkerDistributedIterator(per_worker_iterator._values)

  @property
  def element_spec(self):
    """The type specification of an element of this dataset.

    This property is subject to change without notice.
    """
    if not isinstance(self._dataset_fn, tf_function.ConcreteFunction):
      raise NotImplementedError(
          "`element_spec` is not supported when the `dataset_fn` is not "
          "a `ConcreteFunction`.")
    return self._dataset_fn.structured_outputs.element_spec


class _PerWorkerDistributedIterator(PerWorkerValues):
  """Distributed iterator for `ClusterCoordinator`."""

  def __next__(self):
    return self.get_next()

  def get_next(self, name=None):
    """Returns the next input from the iterator for all replicas."""
    raise NotImplementedError("Iterating over an `AsyncDistributedIterator` "
                              "is not supported right now.")


def _extract_failed_ps_instances(err_msg):
  """Return a set of potentially failing ps instances from error message."""
  tasks = re.findall("/job:ps/replica:0/task:[0-9]+", err_msg)
  return set(int(t.split(":")[-1]) for t in tasks)


def _is_ps_failure(error):
  """Whether the error is considered a parameter server failure."""
  return (isinstance(error, errors.UnavailableError) and
          _RPC_ERROR_FROM_PS in str(error))


def _is_worker_failure(error):
  """Whether the error is considered a worker failure."""
  if _JOB_WORKER_STRING_IDENTIFIER not in str(error):
    return False
  if _RPC_ERROR_FROM_PS in str(error):
    return False

  # TODO(haoyuzhang): Consider using special status code if error from a
  # remote is derived from RPC errors originated from other hosts.
  if isinstance(error, (errors.UnavailableError, errors.AbortedError)):
    return True

  # The following error could happen when the remote task fails and restarts
  # in a very short interval during which no RPCs were exchanged to detect the
  # failure. In that case, gRPC allows channel (which is different from a
  # connection) to be reused for a replaced server listening to same address.
  if isinstance(error, errors.InvalidArgumentError):
    if ("unknown device" in str(error) or
        "Unable to find the relevant tensor remote_handle" in str(error)):
      # TODO(b/159961667): Fix "Unable to find the relevant tensor
      # remote_handle" part.
      return True

  # TODO(b/162541228): The following 2 types of errors are very rare and only
  # observed in large-scale testing. The types of errors should be reduced.
  # This could happen when the function registration fails. In the observed
  # cases this only happens to the dataset related functions.
  if isinstance(error, errors.NotFoundError):
    if ("is neither a type of a primitive operation nor a name of a function "
        "registered" in str(error)):
      return True

  return False
