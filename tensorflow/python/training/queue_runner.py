# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Create threads to run multiple enqueue ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.core.protobuf import queue_runner_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging


class QueueRunner(object):
  """Holds a list of enqueue operations for a queue, each to be run in a thread.

  Queues are a convenient TensorFlow mechanism to compute tensors
  asynchronously using multiple threads. For example in the canonical 'Input
  Reader' setup one set of threads generates filenames in a queue; a second set
  of threads read records from the files, processes them, and enqueues tensors
  on a second queue; a third set of threads dequeues these input records to
  construct batches and runs them through training operations.

  There are several delicate issues when running multiple threads that way:
  closing the queues in sequence as the input is exhausted, correctly catching
  and reporting exceptions, etc.

  The `QueueRunner`, combined with the `Coordinator`, helps handle these issues.
  """

  def __init__(self, queue=None, enqueue_ops=None, close_op=None,
               cancel_op=None, queue_closed_exception_types=None,
               queue_runner_def=None):
    """Create a QueueRunner.

    On construction the `QueueRunner` adds an op to close the queue.  That op
    will be run if the enqueue ops raise exceptions.

    When you later call the `create_threads()` method, the `QueueRunner` will
    create one thread for each op in `enqueue_ops`.  Each thread will run its
    enqueue op in parallel with the other threads.  The enqueue ops do not have
    to all be the same op, but it is expected that they all enqueue tensors in
    `queue`.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      queue_closed_exception_types: Optional tuple of Exception types that
        indicate that the queue has been closed when raised during an enqueue
        operation.  Defaults to `(tf.errors.OutOfRangeError,)`.  Another common
        case includes `(tf.errors.OutOfRangeError, tf.errors.CancelledError)`,
        when some of the enqueue ops may dequeue from other Queues.
      queue_runner_def: Optional `QueueRunnerDef` protocol buffer. If specified,
        recreates the QueueRunner from its contents. `queue_runner_def` and the
        other arguments are mutually exclusive.

    Raises:
      ValueError: If both `queue_runner_def` and `queue` are both specified.
      ValueError: If `queue` or `enqueue_ops` are not provided when not
        restoring from `queue_runner_def`.
    """
    if queue_runner_def:
      if queue or enqueue_ops:
        raise ValueError("queue_runner_def and queue are mutually exclusive.")
      self._init_from_proto(queue_runner_def)
    else:
      self._init_from_args(
          queue=queue, enqueue_ops=enqueue_ops,
          close_op=close_op, cancel_op=cancel_op,
          queue_closed_exception_types=queue_closed_exception_types)
    # Protect the count of runs to wait for.
    self._lock = threading.Lock()
    self._runs = 0
    # List of exceptions raised by the running threads.
    self._exceptions_raised = []

  def _init_from_args(self, queue=None, enqueue_ops=None, close_op=None,
                      cancel_op=None, queue_closed_exception_types=None):
    """Create a QueueRunner from arguments.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      queue_closed_exception_types: Tuple of exception types, which indicate
        the queue has been safely closed.

    Raises:
      ValueError: If `queue` or `enqueue_ops` are not provided when not
        restoring from `queue_runner_def`.
      TypeError: If `queue_closed_exception_types` is provided, but is not
        a non-empty tuple of error types (subclasses of `tf.errors.OpError`).
    """
    if not queue or not enqueue_ops:
      raise ValueError("Must provide queue and enqueue_ops.")
    self._queue = queue
    self._enqueue_ops = enqueue_ops
    self._close_op = close_op
    self._cancel_op = cancel_op
    if queue_closed_exception_types is not None:
      if (not isinstance(queue_closed_exception_types, tuple)
          or not queue_closed_exception_types
          or not all(issubclass(t, errors.OpError)
                     for t in queue_closed_exception_types)):
        raise TypeError(
            "queue_closed_exception_types, when provided, "
            "must be a non-empty list of tf.error types, but saw: %s"
            % queue_closed_exception_types)
    self._queue_closed_exception_types = queue_closed_exception_types
    # Close when no more will be produced, but pending enqueues should be
    # preserved.
    if self._close_op is None:
      self._close_op = self._queue.close()
    # Close and cancel pending enqueues since there was an error and we want
    # to unblock everything so we can cleanly exit.
    if self._cancel_op is None:
      self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
    if not self._queue_closed_exception_types:
      self._queue_closed_exception_types = (errors.OutOfRangeError,)
    else:
      self._queue_closed_exception_types = tuple(
          self._queue_closed_exception_types)

  def _init_from_proto(self, queue_runner_def):
    """Create a QueueRunner from `QueueRunnerDef`.

    Args:
      queue_runner_def: Optional `QueueRunnerDef` protocol buffer.
    """
    assert isinstance(queue_runner_def, queue_runner_pb2.QueueRunnerDef)
    g = ops.get_default_graph()
    self._queue = g.as_graph_element(queue_runner_def.queue_name)
    self._enqueue_ops = [g.as_graph_element(op) for op
                         in queue_runner_def.enqueue_op_name]
    self._close_op = g.as_graph_element(queue_runner_def.close_op_name)
    self._cancel_op = g.as_graph_element(queue_runner_def.cancel_op_name)
    self._queue_closed_exception_types = tuple(
        errors.exception_type_from_error_code(code)
        for code in queue_runner_def.queue_closed_exception_types)
    # Legacy support for old QueueRunnerDefs created before this field
    # was added.
    if not self._queue_closed_exception_types:
      self._queue_closed_exception_types = (errors.OutOfRangeError,)

  @property
  def queue(self):
    return self._queue

  @property
  def enqueue_ops(self):
    return self._enqueue_ops

  @property
  def close_op(self):
    return self._close_op

  @property
  def cancel_op(self):
    return self._cancel_op

  @property
  def queue_closed_exception_types(self):
    return self._queue_closed_exception_types

  @property
  def exceptions_raised(self):
    """Exceptions raised but not handled by the `QueueRunner` threads.

    Exceptions raised in queue runner threads are handled in one of two ways
    depending on whether or not a `Coordinator` was passed to
    `create_threads()`:

    * With a `Coordinator`, exceptions are reported to the coordinator and
      forgotten by the `QueueRunner`.
    * Without a `Coordinator`, exceptions are captured by the `QueueRunner` and
      made available in this `exceptions_raised` property.

    Returns:
      A list of Python `Exception` objects.  The list is empty if no exception
      was captured.  (No exceptions are captured when using a Coordinator.)
    """
    return self._exceptions_raised

  @property
  def name(self):
    """The string name of the underlying Queue."""
    return self._queue.name

  # pylint: disable=broad-except
  def _run(self, sess, enqueue_op, coord=None):
    """Execute the enqueue op in a loop, close the queue in case of error.

    Args:
      sess: A Session.
      enqueue_op: The Operation to run.
      coord: Optional Coordinator object for reporting errors and checking
        for stop conditions.
    """
    if coord:
      coord.register_thread(threading.current_thread())
    decremented = False
    try:
      while True:
        if coord and coord.should_stop():
          break
        try:
          sess.run(enqueue_op)
        except self._queue_closed_exception_types:  # pylint: disable=catching-non-exception
          # This exception indicates that a queue was closed.
          with self._lock:
            self._runs -= 1
            decremented = True
            if self._runs == 0:
              try:
                sess.run(self._close_op)
              except Exception as e:
                # Intentionally ignore errors from close_op.
                logging.vlog(1, "Ignored exception: %s", str(e))
            return
    except Exception as e:
      # This catches all other exceptions.
      if coord:
        coord.request_stop(e)
      else:
        logging.error("Exception in QueueRunner: %s", str(e))
        with self._lock:
          self._exceptions_raised.append(e)
        raise
    finally:
      # Make sure we account for all terminations: normal or errors.
      if not decremented:
        with self._lock:
          self._runs -= 1

  def _close_on_stop(self, sess, cancel_op, coord):
    """Close the queue when the Coordinator requests stop.

    Args:
      sess: A Session.
      cancel_op: The Operation to run.
      coord: Coordinator.
    """
    coord.register_thread(threading.current_thread())
    coord.wait_for_stop()
    try:
      sess.run(cancel_op)
    except Exception as e:
      # Intentionally ignore errors from cancel_op.
      logging.vlog(1, "Ignored exception: %s", str(e))
  # pylint: enable=broad-except

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    """Create threads to run the enqueue ops.

    This method requires a session in which the graph was launched.  It creates
    a list of threads, optionally starting them.  There is one thread for each
    op passed in `enqueue_ops`.

    The `coord` argument is an optional coordinator, that the threads will use
    to terminate together and report exceptions.  If a coordinator is given,
    this method starts an additional thread to close the queue when the
    coordinator requests a stop.

    This method may be called again as long as all threads from a previous call
    have stopped.

    Args:
      sess: A `Session`.
      coord: Optional `Coordinator` object for reporting errors and checking
        stop conditions.
      daemon: Boolean.  If `True` make the threads daemon threads.
      start: Boolean.  If `True` starts the threads.  If `False` the
        caller must call the `start()` method of the returned threads.

    Returns:
      A list of threads.

    Raises:
      RuntimeError: If threads from a previous call to `create_threads()` are
      still running.
    """
    with self._lock:
      if self._runs > 0:
        # Already started: no new threads to return.
        return []
      self._runs = len(self._enqueue_ops)
      self._exceptions_raised = []

    ret_threads = [threading.Thread(target=self._run, args=(sess, op, coord))
                   for op in self._enqueue_ops]
    if coord:
      ret_threads.append(threading.Thread(target=self._close_on_stop,
                                          args=(sess, self._cancel_op, coord)))
    for t in ret_threads:
      if daemon:
        t.daemon = True
      if start:
        t.start()
    return ret_threads

  def to_proto(self):
    """Converts this `QueueRunner` to a `QueueRunnerDef` protocol buffer.

    Returns:
      A `QueueRunnerDef` protocol buffer.
    """
    queue_runner_def = queue_runner_pb2.QueueRunnerDef()
    queue_runner_def.queue_name = self.queue.name
    for enqueue_op in self.enqueue_ops:
      queue_runner_def.enqueue_op_name.append(enqueue_op.name)
    queue_runner_def.close_op_name = self.close_op.name
    queue_runner_def.cancel_op_name = self.cancel_op.name
    queue_runner_def.queue_closed_exception_types.extend([
        errors.error_code_from_exception_type(cls)
        for cls in self._queue_closed_exception_types])
    return queue_runner_def

  @staticmethod
  def from_proto(queue_runner_def):
    """Returns a `QueueRunner` object created from `queue_runner_def`."""
    return QueueRunner(queue_runner_def=queue_runner_def)


def add_queue_runner(qr, collection=ops.GraphKeys.QUEUE_RUNNERS):
  """Adds a `QueueRunner` to a collection in the graph.

  When building a complex model that uses many queues it is often difficult to
  gather all the queue runners that need to be run.  This convenience function
  allows you to add a queue runner to a well known collection in the graph.

  The companion method `start_queue_runners()` can be used to start threads for
  all the collected queue runners.

  Args:
    qr: A `QueueRunner`.
    collection: A `GraphKey` specifying the graph collection to add
      the queue runner to.  Defaults to `GraphKeys.QUEUE_RUNNERS`.
  """
  ops.add_to_collection(collection, qr)


def start_queue_runners(sess=None, coord=None, daemon=True, start=True,
                        collection=ops.GraphKeys.QUEUE_RUNNERS):
  """Starts all queue runners collected in the graph.

  This is a companion method to `add_queue_runner()`.  It just starts
  threads for all queue runners collected in the graph.  It returns
  the list of all threads.

  Args:
    sess: `Session` used to run the queue ops.  Defaults to the
      default session.
    coord: Optional `Coordinator` for coordinating the started threads.
    daemon: Whether the threads should be marked as `daemons`, meaning
      they don't block program exit.
    start: Set to `False` to only create the threads, not start them.
    collection: A `GraphKey` specifying the graph collection to
      get the queue runners from.  Defaults to `GraphKeys.QUEUE_RUNNERS`.

  Returns:
    A list of threads.
  """
  if sess is None:
    sess = ops.get_default_session()
    if not sess:
      raise ValueError("Cannot start queue runners: No default session is "
                       "registered. Use `with sess.as_default()` or pass an "
                       "explicit session to tf.start_queue_runners(sess=sess)")
  with sess.graph.as_default():
    threads = []
    for qr in ops.get_collection(collection):
      threads.extend(qr.create_threads(sess, coord=coord, daemon=daemon,
                                       start=start))
  return threads


ops.register_proto_function(ops.GraphKeys.QUEUE_RUNNERS,
                            proto_type=queue_runner_pb2.QueueRunnerDef,
                            to_proto=QueueRunner.to_proto,
                            from_proto=QueueRunner.from_proto)
