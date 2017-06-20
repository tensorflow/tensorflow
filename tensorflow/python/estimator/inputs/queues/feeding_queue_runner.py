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

"""A `QueueRunner` that takes a feed function as an argument."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import queue_runner as qr


class _FeedingQueueRunner(qr.QueueRunner):
  """A queue runner that allows the feeding of values such as numpy arrays."""

  def __init__(self, queue=None, enqueue_ops=None, close_op=None,
               cancel_op=None, feed_fns=None,
               queue_closed_exception_types=None):
    """Initialize the queue runner.

    For further documentation, see `queue_runner.py`. Note that
    `FeedingQueueRunner` does not support construction from protobuffer nor
    serialization to protobuffer.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      feed_fns: a list of functions that return a dictionary mapping fed
        `Tensor`s to values. Must be the same length as `enqueue_ops`.
      queue_closed_exception_types: Optional tuple of Exception types that
        indicate that the queue has been closed when raised during an enqueue
        operation.  Defaults to
        `(tf.errors.OutOfRangeError, tf.errors.CancelledError)`.

    Raises:
      ValueError: `feed_fns` is not `None` and has different length than
        `enqueue_ops`.
    """
    if queue_closed_exception_types is None:
      queue_closed_exception_types = (
          errors.OutOfRangeError, errors.CancelledError)
    super(_FeedingQueueRunner, self).__init__(
        queue, enqueue_ops, close_op,
        cancel_op, queue_closed_exception_types=queue_closed_exception_types)
    if feed_fns is None:
      self._feed_fns = [None for _ in enqueue_ops]
    else:
      if len(feed_fns) != len(enqueue_ops):
        raise ValueError(
            "If feed_fns is not None, it must have the same length as "
            "enqueue_ops.")
      self._feed_fns = feed_fns

  # pylint: disable=broad-except
  def _run(self, sess, enqueue_op, feed_fn, coord=None):
    """Execute the enqueue op in a loop, close the queue in case of error.

    Args:
      sess: A `Session`.
      enqueue_op: The `Operation` to run.
      feed_fn: the feed function to pass to `sess.run`.
      coord: Optional `Coordinator` object for reporting errors and checking
        for stop conditions.

    """
    # TODO(jamieas): Reduce code duplication with `QueueRunner`.
    if coord:
      coord.register_thread(threading.current_thread())
    decremented = False
    try:
      while True:
        if coord and coord.should_stop():
          break
        try:
          feed_dict = None if feed_fn is None else feed_fn()
          sess.run(enqueue_op, feed_dict=feed_dict)
        except (errors.OutOfRangeError, errors.CancelledError):
          # This exception indicates that a queue was closed.
          with self._lock:
            self._runs_per_session[sess] -= 1
            decremented = True
            if self._runs_per_session[sess] == 0:
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
          self._runs_per_session[sess] -= 1

  def create_threads(self, sess, coord=None, daemon=False, start=False):
    """Create threads to run the enqueue ops for the given session.

    This method requires a session in which the graph was launched.  It creates
    a list of threads, optionally starting them.  There is one thread for each
    op passed in `enqueue_ops`.

    The `coord` argument is an optional coordinator, that the threads will use
    to terminate together and report exceptions.  If a coordinator is given,
    this method starts an additional thread to close the queue when the
    coordinator requests a stop.

    If previously created threads for the given session are still running, no
    new threads will be created.

    Args:
      sess: A `Session`.
      coord: Optional `Coordinator` object for reporting errors and checking
        stop conditions.
      daemon: Boolean.  If `True` make the threads daemon threads.
      start: Boolean.  If `True` starts the threads.  If `False` the
        caller must call the `start()` method of the returned threads.

    Returns:
      A list of threads.
    """
    with self._lock:
      try:
        if self._runs_per_session[sess] > 0:
          # Already started: no new threads to return.
          return []
      except KeyError:
        # We haven't seen this session yet.
        pass
      self._runs_per_session[sess] = len(self._enqueue_ops)
      self._exceptions_raised = []

    ret_threads = [threading.Thread(target=self._run,
                                    args=(sess, op, feed_fn, coord))
                   for op, feed_fn in zip(self._enqueue_ops, self._feed_fns)]
    if coord:
      ret_threads.append(threading.Thread(target=self._close_on_stop,
                                          args=(sess, self._cancel_op, coord)))
    for t in ret_threads:
      if daemon:
        t.daemon = True
      if start:
        t.start()
    return ret_threads

  def _init_from_proto(self, queue_runner_def):
    raise NotImplementedError(
        "{} does not support initialization from proto.".format(type(
            self).__name__))

  def to_proto(self):
    raise NotImplementedError(
        "{} does not support serialization to proto.".format(type(
            self).__name__))
