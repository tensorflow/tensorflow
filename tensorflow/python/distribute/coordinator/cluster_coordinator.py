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

import collections
import contextlib
import os
import re
import threading
import time
import weakref

from six.moves import queue

from tensorflow.python.distribute.coordinator import coordinator_context
from tensorflow.python.distribute.coordinator import metric_utils
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.distribute.coordinator import utils
from tensorflow.python.distribute.coordinator import values as values_lib
from tensorflow.python.distribute.coordinator import watchdog
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

# Maximum time for failed worker to come back is 1 hour
_WORKER_MAXIMUM_RECOVERY_SEC = 3600
# How often to poll task states from the coordination service. In testing, a
# value of 1 led to some spurious reports of unavailability, so a higher value
# is used. Refer to the discussion in b/249134783 for more.
_POLL_FREQ_IN_SEC = 5

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


RemoteValueStatus = remote_value.RemoteValueStatus
RemoteValue = remote_value.RemoteValue
RemoteValueImpl = values_lib.RemoteValueImpl
PerWorkerValues = values_lib.PerWorkerValues


class ClosureInputError(Exception):
  """Wrapper for errors from resource building.

  When a closure starts, it first checks for errors in any of its inputs, which
  are RemoteValues from resource closures. If there were any errors, it wraps
  the exception in this class and raises so it can be handled by the worker
  failure handler.

  Attributes:
    original_exception:
  """

  def __init__(self, original_exception):
    # Avoid doubly-nested errors
    if isinstance(original_exception,
                  (ClosureInputError, ClosureAbortedError)):
      self.original_exception = original_exception.original_exception
    else:
      self.original_exception = original_exception
    message = ("Input has an error, the original exception is %r, "
               "error message is %s." %
               (self.original_exception, str(self.original_exception)))
    super().__init__(message)
    self.with_traceback(original_exception.__traceback__)


class ClosureAbortedError(Exception):
  """Wrapper for errors from training closures, to attach to resource closures.

  This wrapper is used when a dependent training closure fails to set errors on
  its required resource closures.

  Attributes:
    original_exception: The Exception to wrap
  """

  def __init__(self, original_exception):
    # Avoid doubly-nested errors
    if isinstance(original_exception,
                  (ClosureInputError, ClosureAbortedError)):
      self.original_exception = original_exception.original_exception
    else:
      self.original_exception = original_exception
    message = ("Other function has an execution error, as a result, the "
               "current value is not available. The original exception is %r, "
               "error message is %s." %
               (self.original_exception, str(self.original_exception)))
    super().__init__(message)
    self.with_traceback(original_exception.__traceback__)


class PSUnavailableError(errors.UnavailableError):
  """Specifies that a parameter server is the unavailable task."""

  def __init__(self, original_exception):
    assert isinstance(original_exception, errors.UnavailableError)
    super().__init__(
        original_exception.node_def,
        original_exception.op,
        original_exception.message,
    )


def _get_error_from_remote_values(structure):
  """Attempts to return errors from `RemoteValue`s. Rebuilds them if needed."""
  errors_in_structure = []

  def _get_error(val):
    if isinstance(val, RemoteValue):
      error = val._get_error()  # pylint: disable=protected-access
      if error:
        errors_in_structure.append(error)

  nest.map_structure(_get_error, structure)
  if errors_in_structure:
    return errors_in_structure[0]
  else:
    return None


def _maybe_as_type_spec(val):
  if isinstance(val, (RemoteValue, PerWorkerValues)):
    if val._type_spec is None:  # pylint: disable=protected-access
      raise ValueError("Output of a scheduled function that is not "
                       "tf.function cannot be the input of another function.")
    return val._type_spec  # pylint: disable=protected-access
  else:
    return val


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
      self._output_type_spec = func_graph.convert_structure_to_signature(
          self._concrete_function.structured_outputs)
      self._function = cancellation_mgr.get_cancelable_function(
          self._concrete_function)
    else:
      # Otherwise (i.e. what is passed in is a regular python function), we have
      # no such information.
      self._output_type_spec = None
      self._function = function

    self._output_remote_value_ref = None

  def build_output_remote_value(self):
    if self._output_remote_value_ref is None:
      ret = RemoteValueImpl(None, self._output_type_spec)
      self._output_remote_value_ref = weakref.ref(ret)
      return ret
    else:
      raise ValueError(
          "The output of the Closure cannot be built more than once.")

  def maybe_call_with_output_remote_value(self, method):
    if self._output_remote_value_ref is None:
      return None
    output_remote_value = self._output_remote_value_ref()
    if output_remote_value is not None:
      return method(output_remote_value)
    return None

  def mark_cancelled(self):
    e = errors.CancelledError(
        None, None, "The corresponding function is "
        "cancelled. Please reschedule the function.")
    self.maybe_call_with_output_remote_value(lambda r: r._set_error(e))  # pylint: disable=protected-access

  def execute_on(self, worker):
    """Executes the closure on the given worker.

    Args:
      worker: a `Worker` object.
    """
    replica_args = _select_worker_slice(worker.worker_index, self._args)
    replica_kwargs = _select_worker_slice(worker.worker_index, self._kwargs)

    e = (
        _get_error_from_remote_values(replica_args) or
        _get_error_from_remote_values(replica_kwargs))
    if e:
      if not isinstance(e, ClosureInputError):
        e = ClosureInputError(e)
      raise e

    with ops.device(worker.device_name):
      with context.executor_scope(worker.executor):
        with coordinator_context.with_dispatch_context(worker):
          with metric_utils.monitored_timer("closure_execution"):
            output_values = self._function(
                *nest.map_structure(coordinator_context.maybe_get_remote_value,
                                    replica_args),
                **nest.map_structure(coordinator_context.maybe_get_remote_value,
                                     replica_kwargs))
    self.maybe_call_with_output_remote_value(
        lambda r: r._set_values(output_values))  # pylint: disable=protected-access


class ResourceClosure(Closure):

  def build_output_remote_value(self):
    if self._output_remote_value_ref is None:
      # We need to remember the Closure object in the `RemoteValue` here.
      ret = RemoteValueImpl(self, self._output_type_spec)
      self._output_remote_value_ref = weakref.ref(ret)
      return ret
    else:
      return self._output_remote_value_ref()


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
    self._tagged_queue = collections.defaultdict(queue.Queue)
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

    self._watchdog = watchdog.WatchDog(on_triggered=self._on_watchdog_timeout)

  def _on_watchdog_timeout(self):
    logging.info("inflight_closure_count is %d", self._inflight_closure_count)
    logging.info("current error is %s:%r", self._error, self._error)

  def stop(self):
    with self._queue_lock:
      self._should_process_closures = False
      self._cancellation_mgr.start_cancel()
      self._closures_queued_condition.notify_all()
    self._watchdog.stop()

  def _cancel_all_closures(self):
    """Clears the queue and sets remaining closures cancelled error.

    This method expects self._queue_lock to be held prior to entry.
    """
    self._cancellation_mgr.start_cancel()
    logging.info("Canceling all closures: waiting for inflight closures to "
                 "finish")
    while self._inflight_closure_count > 0:
      self._no_inflight_closure_condition.wait()
    logging.info("Canceling all closures: canceling remaining closures on the "
                 "queue")
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

  def put(self, closure, tag=None):
    """Put a closure into the queue for later execution.

    If `mark_failed` was called before `put`, the error from the first
    invocation of `mark_failed` will be raised.

    Args:
      closure: The `Closure` to put into the queue.
      tag: if not None, put into a queue with the given tag.
    """
    closure.tag = tag
    if tag is not None:
      with self._queue_lock:
        self._tagged_queue[tag].put(closure, block=False)
        self._closures_queued_condition.notify_all()
    else:
      with self._put_wait_lock, self._queue_lock:
        self._queue_free_slot_condition.wait_for(lambda: not self._queue.full())
        self._queue.put(closure, block=False)
        self._raise_if_error()
        self._closures_queued_condition.notify()

  def get(self, timeout=None, tag=None):
    """Return a closure from the queue to be executed.

    It will try to fetch an item from the queue with the given tag. If this
    queue is empty, it will then check the global queue.

    Args:
      timeout: timeout when waiting for a closure to be put.
      tag: optional tag to specify which queue to query first before querying
        the global queue.

    Returns:
      a closure or None after timeout.
    """
    with self._queue_lock:
      while (self._should_process_closures and self._queue.empty() and
             (tag is None or self._tagged_queue[tag].empty())):
        if not self._closures_queued_condition.wait(timeout=timeout):
          return None
      if not self._should_process_closures:
        return None
      if tag is not None and not self._tagged_queue[tag].empty():
        closure = self._tagged_queue[tag].get(block=False)
        return closure
      closure = self._queue.get(block=False)
      assert closure.tag is None
      assert tag is None or self._tagged_queue[tag].empty()
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
        self._no_inflight_closure_condition.notify_all()
      if self._queue.empty() and self._inflight_closure_count == 0:
        self._stop_waiting_condition.notify_all()
      self._watchdog.report_closure_done()

  def put_back(self, closure):
    """Put the closure back into the queue as it was not properly executed."""
    assert closure.tag is None
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
        self._no_inflight_closure_condition.notify_all()

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
      logging.info("Waiting for all global closures to be finished.")
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
        self._no_inflight_closure_condition.notify_all()
      self._stop_waiting_condition.notify_all()

  def done(self):
    """Returns true if the queue is empty and there is no inflight closure.

    If `mark_failed` was called before `done`, the error from the first
    invocation of `mark_failed` will be raised.
    """
    with self._queue_lock:
      self._raise_if_error()
      return self._queue.empty() and self._inflight_closure_count == 0

  def clear_tag_unlocked(self, tag):
    self._tagged_queue[tag] = queue.Queue()


class CoordinationServicePreemptionHandler(object):
  """Handles preemptions of workers and parameter servers.

  Starts a thread to regularly poll the coordination service (hosted on PS 0)
  for task states. When a worker's task state reflects an error, it inspects the
  error. If the error is recoverable (i.e. a preemption), it waits for the
  worker to recover, then updates the server def. Otherwise, it raises the error
  to the user.

  A worker error is detected to be recoverable if it is the result of missing a
  heartbeat that workers regularly send to the coordination service.

  The thread also checks for parameter server errors. If these are detected, the
  thread and coordinator shutdown. To resume training in this case, the whole
  job must be restarted and resumed from the latest checkpoint.
  """

  def __init__(self, server_def, cluster):
    self._server_def = server_def
    self._cluster = cluster
    self._cluster_update_lock = threading.Lock()
    self._cluster_due_for_update_or_finish = threading.Event()
    self._worker_up_cond = threading.Condition(self._cluster_update_lock)

    self._next_task_state_cond = threading.Condition()
    self._task_states = None

    self._error_from_recovery = None
    self._should_preemption_thread_run = True
    self._task_state_poller_thread = utils.RepeatedTimer(
        interval=_POLL_FREQ_IN_SEC,
        function=self._get_task_states)
    self._preemption_handler_thread = threading.Thread(
        target=self._preemption_handler,
        name="WorkerPreemptionHandler",
        daemon=True)
    self._preemption_handler_thread.start()

    self._num_workers = self._cluster._num_workers
    self._num_ps = self._cluster._num_ps

  def stop(self):
    """Ensure the worker preemption thread is closed."""
    self._task_state_poller_thread.stop()
    self._should_preemption_thread_run = False
    with self._cluster_update_lock:
      self._cluster_due_for_update_or_finish.set()
    # TODO(yuefengz): The preemption handler thread shouldn't be terminated
    # asynchronously since it touches eager context which is a process-wide
    # singleton. The problem is in OSS unit tests will time out.

  @contextlib.contextmanager
  def wait_on_failure(self,
                      on_failure_fn=None,
                      on_transient_failure_fn=None,
                      on_recovery_fn=None,
                      worker_device_name="(unknown)"):
    """Catches errors during closure execution and handles them.

    Args:
      on_failure_fn: an optional function to run if preemption happens.
      on_transient_failure_fn: an optional function to run if transient failure
        happens.
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
    except (errors.OpError, ClosureInputError,
            ClosureAbortedError) as e:
      # The next state could reflect stale heartbeats, so wait for two rounds.
      # Example:
      # - Worker sends healthy heartbeat at T=0.
      # - Coordination service receives healthy heartbeat at T=0.
      # - Worker gets preempted at T=0.1.
      # - Coordinator catches error at T=0.2, and waits here for next states.
      # - Coordinator polls states at T=1.9. Heartbeat time has not elapsed yet,
      #   so coordination service does not know it is down yet.
      # - Coordination service learns of worker unavailability at T=2, the next
      #   heartbeat.
      # - Coordinator polls states at T=3.9 and learns of worker unavailability.
      with self._next_task_state_cond:
        # Give some buffer time to make sure task states are updated during the
        # wait interval
        self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 1.25)
      with self._next_task_state_cond:
        self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 1.25)

      # Check for coordination service failure
      if not self._task_states:
        self._log_ps_failure_and_raise(e, 0)

      worker_states = self._task_states[:self._num_workers]
      ps_states = self._task_states[self._num_workers:]

      # Check for PS failure
      if any(ps_states):
        failed_ps_index = [
            ix for ix, ps_state in enumerate(ps_states) if ps_state
        ]
        self._log_ps_failure_and_raise(e, failed_ps_index[0])

      # Check for preemption of this worker
      worker_ix = int(worker_device_name.split(":")[-1])
      if worker_states[worker_ix]:
        # Raise error if all closures are being cancelled
        if self._cluster.closure_queue._cancellation_mgr.is_cancelled:  # pylint: disable=protected-access
          if isinstance(e, errors.CancelledError):
            raise e
          # It's possible the caught error `e` here is due to worker preemption
          # and is thus not a `CancelledError`, because a different
          # unrecoverable error on another worker caused closure cancellation,
          # while this thread was waiting for task states. So raise a new
          # CancelledError.
          else:
            raise errors.CancelledError(
                None, None, "The corresponding function was cancelled while "
                "attempting to recover from worker failure.")
        # Else, preemption
        self._handle_failure_and_recovery(e, on_failure_fn,
                                          on_transient_failure_fn,
                                          on_recovery_fn, worker_device_name)
        return

      #  else, if timeout: log
      if self._cluster._record_and_ignore_transient_timeouts(e):  # pylint: disable=protected-access
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "This derived error is ignored and not reported to users.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return
      raise e

  def _handle_failure_and_recovery(self,
                                   e,
                                   on_failure_fn,
                                   on_transient_failure_fn,
                                   on_recovery_fn,
                                   worker_device_name):
    """Call failure fn, wait for cluster to recover, then call recovery fn.

    Args:
      e: the Exception thrown during closure execution.
      on_failure_fn: an optional function to run if preemption happens.
      on_transient_failure_fn: an optional function to run if transient failure
        happens.
      on_recovery_fn: an optional function to run when a worker is recovered
        from preemption.
      worker_device_name: the device name of the worker instance that is passing
        through the failure.
    """
    if on_failure_fn:
      on_failure_fn(e)
    # update server def
    with self._cluster_update_lock:
      self._cluster_due_for_update_or_finish.set()
      self._worker_up_cond.wait(_WORKER_MAXIMUM_RECOVERY_SEC)
      if self._error_from_recovery:
        # TODO(yuefengz): there is only one worker that will get this error.
        # Ideally we should let all workers notified by `_worker_up_cond` get
        # this error.
        try:
          raise self._error_from_recovery
        finally:
          self._error_from_recovery = None
      logging.info("Worker %s has been recovered.", worker_device_name)

    if on_recovery_fn:
      logging.info("Worker %s calling on_recovery_fn", worker_device_name)
      with self.wait_on_failure(
          on_recovery_fn=on_recovery_fn,
          on_transient_failure_fn=on_transient_failure_fn,
          worker_device_name=worker_device_name):
        on_recovery_fn()

  def _log_ps_failure_and_raise(self, e, ps_index):
    logging.info("Parameter server failure detected at PS task %d", ps_index)
    self.stop()
    raise PSUnavailableError(e)

  def _get_task_states(self):
    try:
      self._task_states = context.context().get_task_states(
          [("worker", self._num_workers), ("ps", self._num_ps)]
      )
    except errors.UnavailableError:
      # Coordination service is down
      self._task_states = None
    with self._next_task_state_cond:
      self._next_task_state_cond.notify_all()

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
          self._notify_cluster_update()
        except Exception as e:  # pylint: disable=broad-except
          logging.info("Error occurred while updating server def: %s", e)
          # Wait for the next set of states from the task state poller
          with self._next_task_state_cond:
            self._next_task_state_cond.wait(_POLL_FREQ_IN_SEC * 2)
          # If a PS is preempted, set the error
          if not self._task_states:
            self._error_from_recovery = e
          else:
            ps_states = self._task_states[self._num_workers:]
            # Check for PS failure
            if any(ps_states):
              self._error_from_recovery = e
          # Else, likely another worker failed. Just log and retry
          self._notify_cluster_update()
          # NOTE: Since the first RPC (GetStatus) of update_server_def is
          # currently blocking by default, error should only happen if:
          # (1) More workers failed while waiting for the previous workers to
          #     come back;
          # (2) Worker failed when exchanging subsequent RPCs after the first
          #     RPC returns.
          # Consider adding backoff retry logic if we see the error logged
          # too frequently.
          logging.error("Cluster update failed with error: %s. Retrying...", e)

  def _notify_cluster_update(self):
    self._worker_up_cond.notify_all()
    # The check for _should_preemption_thread_run is necessary since the
    # `stop` may have already set _cluster_due_for_update_or_finish.
    if self._should_preemption_thread_run:
      self._cluster_due_for_update_or_finish.clear()


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

    # Only categorize the failure as a worker preemption if the cancellation
    # manager did not attempt to cancel the blocking operations.
    if _is_worker_failure(e) and (
        not self._cluster.closure_queue._cancellation_mgr.is_cancelled):  # pylint: disable=protected-access
      return
    raise e

  @contextlib.contextmanager
  def wait_on_failure(self,
                      on_failure_fn=None,
                      on_transient_failure_fn=None,
                      on_recovery_fn=None,
                      worker_device_name="(unknown)"):
    """Catches worker preemption error and wait until failed workers are back.

    Args:
      on_failure_fn: an optional function to run if preemption happens.
      on_transient_failure_fn: an optional function to run if transient failure
        happens.
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
    except (errors.OpError, ClosureInputError,
            ClosureAbortedError, TypeError) as e:
      # If the error is due to temporary connectivity issues between worker and
      # ps, put back closure, ignore error and do not mark worker as failure.
      if self._cluster._record_and_ignore_transient_ps_failure(e):  # pylint: disable=protected-access
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "It is treated as a transient connectivity failure for now.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return

      # If the error is due to temporary connectivity issues that cause the
      # server-side RPCs to be cancelled, TF might not abort the step and the
      # closure might timeout. The coordinator ignores certain amount of such
      # failures without marking worker as failure.
      if self._cluster._record_and_ignore_transient_timeouts(e):  # pylint: disable=protected-access
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "This derived error is ignored and not reported to users.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return

      # Ignoring derived CancelledErrors to tolerate transient failures in
      # PS-worker communication, which initially exposed as an UnavailableError
      # and then lead to sub-function cancellation, subsequently getting
      # reported from worker to chief as CancelledError.
      # We do not mark either worker or PS as failed due to only CancelledError.
      # If there are real (non-transient) failures, they must also be reported
      # as other errors (UnavailableError most likely) in closure executions.
      if isinstance(e, errors.CancelledError) and "/job:" in str(e):
        logging.error(
            "Remote function on worker %s failed with %r:%s\n"
            "This derived error is ignored and not reported to users.",
            worker_device_name, e, e)
        if on_transient_failure_fn:
          on_transient_failure_fn()
        return

      # This reraises the error, if it's not considered recoverable; otherwise,
      # the following failure recovery logic run. At this time, only worker
      # unavailability is recoverable. PS unavailability as well as other
      # errors in the user function is not recoverable.
      self._validate_preemption_failure(e)

      logging.error("Worker %s failed with %r:%s", worker_device_name, e, e)
      if on_failure_fn:
        on_failure_fn(e)

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
        logging.info("Worker %s calling on_recovery_fn", worker_device_name)
        with self.wait_on_failure(
            on_recovery_fn=on_recovery_fn,
            on_transient_failure_fn=on_transient_failure_fn,
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
          with metric_utils.monitored_timer("server_def_update"):
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
          logging.info("Error occurred while updating server def: %s", e)
          try:
            self._validate_preemption_failure(e)
          except Exception as ps_e:  # pylint: disable=broad-except
            logging.info("Error that occurred while updating server def is not "
                         "a worker failure. So set it as _error_from_recovery")
            # In this case, a parameter server fails. So we raise this error to
            # the caller of `wait_on_failure`.
            self._error_from_recovery = ps_e
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
    self._resource_tracking_lock = threading.Lock()
    self._resource_remote_value_refs = []
    self._is_dead_with_error = None
    self._should_worker_thread_run = True

    # Worker threads need to start after `Worker`'s initialization.
    threading.Thread(target=self._process_queue,
                     name="WorkerClosureProcessingLoop-%d" % self.worker_index,
                     daemon=True).start()

  def stop(self):
    """Ensure the worker thread is closed."""
    self._should_worker_thread_run = False

  def _schedule_resource(self, closure):
    self._cluster.closure_queue.put(closure, tag=self.worker_index)

  def _set_resources_aborted(self, e):
    """Set the resource ABORTED and add an error to it."""
    # TODO(yuefengz): maybe we can query whether a tensor is valid or not
    # instead of marking a tensor aborted?
    logging.info("[Worker %d] Clearing all resources.", self.worker_index)
    for weakref_resource in self._resource_remote_value_refs:
      resource = weakref_resource()
      if resource:
        # It is important to set an error on an aborted RemoteValue from a
        # ResourceClosure because its failure will not trigger the worker thread
        # to raise error immediately and the worker may continue executing
        # closures taking it as an input. The error will then be correctly
        # reported to users.
        resource._set_aborted(ClosureAbortedError(e))  # pylint: disable=protected-access

  def _on_closure_failure(self, closure, e):
    logging.info("[Worker %d] Putting back a closure after it failed.",
                 self.worker_index)
    self._cluster.closure_queue.put_back(closure)

    with self._resource_tracking_lock:
      self._is_dead_with_error = e
      self._set_resources_aborted(e)

  def _on_resource_closure_failure(self, e):
    """Clear tagged queue to ensure resource closures are rebuilt.

    Args:
      e: The exception arisen from the resource closure.
    """
    logging.info("[Worker %d] Clearing tagged queue after resource closure "
                 "failure.", self.worker_index)
    with self._resource_tracking_lock:
      self._is_dead_with_error = e
      # No locking on queue is needed since
      #  * get will not happen concurrently here.
      #  * put to the specific tagged queue will be guarded by
      #    `self._resource_tracking_lock`.
      self._cluster.closure_queue.clear_tag_unlocked(self.worker_index)
      self._set_resources_aborted(e)

  def _on_worker_recovery(self):
    logging.info("[Worker %d] calling _on_worker_recovery", self.worker_index)
    with self._resource_tracking_lock:
      for weakref_resource in self._resource_remote_value_refs:
        resource = weakref_resource()
        if resource:
          self._schedule_resource(resource._closure)  # pylint: disable=protected-access
      self._is_dead_with_error = False

  def _process_closure(self, closure):
    """Runs a closure with preemption handling."""
    try:
      with self.failure_handler.wait_on_failure(
          on_failure_fn=lambda e: self._on_closure_failure(closure, e),
          on_transient_failure_fn=(
              lambda: self._cluster.closure_queue.put_back(closure)),
          on_recovery_fn=self._on_worker_recovery,
          worker_device_name=self.device_name):
        closure.execute_on(self)
        with metric_utils.monitored_timer("remote_value_fetch"):
          # Copy the remote tensor to local (the coordinator) in case worker
          # becomes unavailable at a later time.
          closure.maybe_call_with_output_remote_value(lambda r: r.get())
        self._cluster.closure_queue.mark_finished()
    except Exception as e:  # pylint: disable=broad-except
      # Avoid logging the derived cancellation error
      if not isinstance(e, errors.CancelledError):
        logging.error(
            " /job:worker/task:%d encountered the following error when "
            "processing closure: %r:%s", self.worker_index, e, e)
      closure.maybe_call_with_output_remote_value(lambda r: r._set_error(e))  # pylint: disable=protected-access
      self._cluster.closure_queue.mark_failed(e)

  def _process_resource_closure(self, closure):
    """Run the given resource closure with preemption handling."""
    assert closure.tag == self.worker_index
    try:
      with self.failure_handler.wait_on_failure(
          on_failure_fn=self._on_resource_closure_failure,
          on_transient_failure_fn=(
              lambda: self._process_resource_closure(closure)),
          on_recovery_fn=self._on_worker_recovery,
          worker_device_name=self.device_name):
        closure.execute_on(self)
    except Exception as e:  # pylint: disable=broad-except
      # Avoid logging the derived cancellation error
      logging.info("[Worker %d] got an exception when processing resource "
                   "closure", self.worker_index)
      if not isinstance(e, errors.CancelledError):
        logging.error(
            " /job:worker/task:%d encountered the following error when "
            "processing resource closure: %r:%s", self.worker_index, e, e)
      closure.maybe_call_with_output_remote_value(lambda r: r._set_error(e))  # pylint: disable=protected-access

  def _maybe_delay(self):
    """Delay if corresponding env vars are set."""
    # If the following two env vars variables are set. Scheduling for workers
    # will start in a staggered manner. Worker i will wait for
    # `TF_COORDINATOR_SCHEDULE_START_DELAY` * i seconds, not exceeding
    # `TF_COORDINATOR_SCHEDULE_START_DELAY_MAX`.
    delay_secs = int(os.environ.get("TF_COORDINATOR_SCHEDULE_START_DELAY", "0"))
    delay_secs *= self.worker_index
    delay_cap = int(
        os.environ.get("TF_COORDINATOR_SCHEDULE_START_DELAY_MAX", "0"))
    if delay_cap:
      delay_secs = min(delay_secs, delay_cap)
    if delay_secs > 0:
      logging.info(" Worker %d sleeping for %d seconds before running function",
                   self.worker_index, delay_secs)
    time.sleep(delay_secs)

  def _process_queue(self):
    """Function running in a worker thread to process closure queues."""
    self._maybe_delay()
    while self._should_worker_thread_run:
      closure = self._cluster.closure_queue.get(tag=self.worker_index)
      if not self._should_worker_thread_run or closure is None:
        if closure is not None:
          closure.mark_cancelled()
        return
      if isinstance(closure, ResourceClosure):
        self._process_resource_closure(closure)
      else:
        self._process_closure(closure)
      # To properly stop the worker and preemption threads, it is important that
      # `ClusterCoordinator` object is not held onto so its `__del__` can be
      # called. By removing the reference to the `closure` that has already been
      # processed, we ensure that the `closure` object is released, while
      # getting the next `closure` at above `self._cluster.closure_queue.get()`
      # call.
      del closure

  def create_resource(self, function, args=None, kwargs=None):
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

    closure = ResourceClosure(
        function,
        self._cluster.resource_cancellation_mgr,
        args=args,
        kwargs=kwargs)
    resource_remote_value = closure.build_output_remote_value()
    with self._resource_tracking_lock:
      self._register_resource(resource_remote_value)
      if self._is_dead_with_error:
        resource_remote_value._set_aborted(  # pylint: disable=protected-access
            ClosureAbortedError(self._is_dead_with_error))
      else:
        self._schedule_resource(closure)
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
    closure_queue: the global Closure queue.
    resource_cancellation_mgr: the cancellation manager used to cancel resource
      closures.
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

    # Ignore worker timeouts due to transient connection errors.
    # Transient connectivity issues might cause the server side to unexpectedly
    # cancel RPC handling logic, leading to closure execution timeouts. When
    # the _transient_timeout_threshold is set to a positive number, the cluster
    # coordinator ignores DeadlineExceeded errors from workers for the specified
    # times before raising the error to users.
    self._transient_timeouts_threshold = int(
        os.environ.get("TF_COORDINATOR_IGNORE_TRANSIENT_TIMEOUTS",
                       self._num_workers // 10))
    self._transient_timeouts_lock = threading.Lock()
    self._transient_timeouts_count = 0

    self.closure_queue = _CoordinatedClosureQueue()
    # Set this environment variable to use an experimental
    # integration with the runtime coordination service to aid in failure
    # detection and handling. This will not affect the functionality of
    # the strategy or cluster coordinator, but is off by default.
    if os.getenv("TF_PSS_ENABLE_COORDINATION_SERVICE"):
      self.failure_handler = CoordinationServicePreemptionHandler(
          context.get_server_def(), self,
      )
    else:
      self.failure_handler = WorkerPreemptionHandler(context.get_server_def(),
                                                     self)
    worker_device_strings = [
        "/job:worker/replica:0/task:%d" % i for i in range(self._num_workers)
    ]
    self.workers = [
        Worker(i, w, self) for i, w in enumerate(worker_device_strings)
    ]

    # Cancellation manager for all resource closures.
    self.resource_cancellation_mgr = cancellation.CancellationManager()

  def stop(self):
    """Stop worker, worker preemption threads, and the closure queue."""
    logging.info("Stopping cluster, starting with failure handler")
    self.failure_handler.stop()

    logging.info("Stopping workers")
    for worker in self.workers:
      worker.stop()
    logging.info("Stopping queue")
    self.closure_queue.stop()
    logging.info("Start cancelling remote resource-building functions")
    self.resource_cancellation_mgr.start_cancel()

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

  def _record_and_ignore_transient_timeouts(self, e):
    """Records observed timeout error and return if it should be ignored."""
    if self._transient_timeouts_threshold <= 0:
      return False
    if not isinstance(e, errors.DeadlineExceededError):
      return False
    with self._transient_timeouts_lock:
      self._transient_timeouts_count += 1
      if self._transient_timeouts_count >= self._transient_timeouts_threshold:
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
        self.closure_queue._cancellation_mgr,  # pylint: disable=protected-access
        args=args,
        kwargs=kwargs)
    ret = closure.build_output_remote_value()
    self.closure_queue.put(closure)
    return ret

  def join(self):
    """Blocks until all scheduled functions are executed."""
    self.closure_queue.wait()

  def done(self):
    """Returns true if all scheduled functions are executed."""
    return self.closure_queue.done()


@tf_export("distribute.experimental.coordinator.ClusterCoordinator",
           "distribute.coordinator.ClusterCoordinator", v1=[])
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
    if not getattr(self, "_has_initialized", False):
      if not hasattr(strategy, "_is_parameter_server_strategy_v2"):
        raise ValueError(
            "Only `tf.distribute.experimental.ParameterServerStrategy` "
            "is supported to work with "
            "`tf.distribute.experimental.coordinator.ClusterCoordinator` "
            "currently.")
      self._strategy = strategy
      self.strategy.extended._used_with_coordinator = True
      self._cluster = Cluster(strategy)
      self._has_initialized = True

  def __del__(self):
    logging.info("ClusterCoordinator destructor: stopping cluster")
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
        execution asynchronously. Regular python function is not supported to be
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
      schedule_remote_value = self._cluster.schedule(
          fn, args=args, kwargs=kwargs)
      self.strategy.extended._being_scheduled = False  # pylint: disable=protected-access
      return schedule_remote_value

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
    """Create dataset on each worker.

    This creates dataset on workers from the input which can be either a
    `tf.data.Dataset`, a `tf.distribute.DistributedDataset` or a function which
    returns a dataset, and returns an object that represents the collection of
    those individual datasets. Calling `iter` on such collection of datasets
    returns a `tf.distribute.experimental.coordinator.PerWorkerValues`, which is
    a collection of iterators, where the iterators have been placed on
    respective workers.

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

    Args:
      dataset_fn: The dataset function that returns a dataset. This is to be
        executed on the workers.

    Returns:
      An object that represents the collection of those individual
      datasets. `iter` is expected to be called on this object that returns
      a `tf.distribute.experimental.coordinator.PerWorkerValues` of the
      iterators (that are on the workers).
    """
    return values_lib.get_per_worker_dataset(dataset_fn, self)

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
      results.append(w.create_resource(fn, args=args, kwargs=kwargs))
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


def _extract_failed_ps_instances(err_msg):
  """Return a set of potentially failing ps instances from error message."""
  tasks = re.findall("/job:ps/replica:0/task:[0-9]+", err_msg)
  return set(int(t.split(":")[-1]) for t in tasks)


def _is_ps_failure(error):
  """Whether the error is considered a parameter server failure."""

  # For an `ClosureInputError` or `ClosureAbortedError`, extract
  # the original error and assess it accordingly.
  if isinstance(error, (ClosureInputError, ClosureAbortedError)):
    error = error.original_exception

  if _RPC_ERROR_FROM_PS not in str(error):
    return False

  if isinstance(error, (errors.UnavailableError, errors.AbortedError)):
    return True

  # The following error could happen when the remote task fails and restarts
  # in a very short interval during which no RPCs were exchanged to detect the
  # failure. In that case, gRPC allows channel (which is different from a
  # connection) to be reused for a replaced server listening to same address.
  if isinstance(error, errors.InvalidArgumentError):
    if ("unknown device" in str(error).lower() or
        "Unable to find the relevant tensor remote_handle" in str(error)):
      return True

  return False


def _handle_graph_execution_error_as_worker_failure():
  return int(os.environ.get("TF_PS_HANDLE_UNKNOWN_ERROR", "0")) > 0


def _is_worker_failure(error):
  """Whether the error is considered a worker failure."""

  # TODO(b/216666282): Understand why worker failure can manifest as a
  # "Graph execution error" `UnknownError`.
  if (_handle_graph_execution_error_as_worker_failure() and
      isinstance(error, errors.UnknownError) and
      "Graph execution error" in str(error)):
    logging.info(f"Handling {type(error)}: {str(error)} as worker failure.")
    return True

  # For an `ClosureInputError` or `ClosureAbortedError`, extract
  # the original error and assess it accordingly.
  if isinstance(error, (ClosureInputError, ClosureAbortedError)):
    error = error.original_exception

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
    if ("unknown device" in str(error).lower() or
        "Primary device is not remote" in str(error) or
        "Unable to find the relevant tensor remote_handle" in str(error)):
      return True

  # TODO(b/162541228): The following 2 types of errors are very rare and only
  # observed in large-scale testing. The types of errors should be reduced.
  # This could happen when the function registration fails. In the observed
  # cases this only happens to the dataset related functions.
  if isinstance(error, errors.NotFoundError):
    if ("is neither a type of a primitive operation nor a name of a function "
        "registered" in str(error)):
      return True

  # NOTE(b/179061495): During worker preemptions, if multiple functions are
  # running concurrently (especially with subfunctions spanning chief/PS),
  # CancelledError can be returned due to chief/PS cancelling outstanding RPCs
  # to the failing workers.
  if isinstance(error, errors.CancelledError):
    return True

  # This can occur when preparing closures for execution when doing exact
  # evaluation, because the iterator creation, which occurs within the
  # tf.function, needs to access the worker device, so it fails if the worker is
  # down.
  if isinstance(error, TypeError) and "Binding inputs to tf.function" in str(
      error):
    return True

  return False
