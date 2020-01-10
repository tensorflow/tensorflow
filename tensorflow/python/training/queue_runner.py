"""Create threads to run multiple enqueue ops."""
import threading

import tensorflow.python.platform

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import logging


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

  def __init__(self, queue, enqueue_ops):
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
    """
    self._queue = queue
    self._enqueue_ops = enqueue_ops
    # Close when no more will be produced, but pending enqueues should be
    # preserved.
    self._close_op = self._queue.close()
    # Close and cancel pending enqueues since there was an error and we want
    # to unblock everything so we can cleanly exit.
    self._cancel_op = self._queue.close(cancel_pending_enqueues=True)
    # Protect the count of runs to wait for.
    self._lock = threading.Lock()
    self._runs = 0
    # List of exceptions raised by the running threads.
    self._exceptions_raised = []

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

  # pylint: disable=broad-except
  def _run(self, sess, enqueue_op, coord=None):
    """Execute the enqueue op in a loop, close the queue in case of error.

    Args:
      sess: A Session.
      enqueue_op: The Operation to run.
      coord: Optional Coordinator object for reporting errors and checking
        for stop conditions.
    """
    decremented = False
    try:
      while True:
        if coord and coord.should_stop():
          break
        try:
          sess.run(enqueue_op)
        except errors.OutOfRangeError:
          # This exception indicates that a queue was closed.
          with self._lock:
            self._runs -= 1
            decremented = True
            if self._runs == 0:
              try:
                sess.run(self._close_op)
              except Exception, e:
                # Intentionally ignore errors from close_op.
                logging.vlog(1, "Ignored exception: %s", str(e))
            return
    except Exception, e:
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
    coord.wait_for_stop()
    try:
      sess.run(cancel_op)
    except Exception, e:
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
        raise RuntimeError(
            "Threads are already running from a previous call to Threads() "
            "for this queue runner.")
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
  threads = []
  for qr in ops.get_collection(collection):
    threads.extend(qr.create_threads(sess, coord=coord, daemon=daemon,
                                     start=start))
  return threads
