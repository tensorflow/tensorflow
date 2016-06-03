# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import queue_runner as qr


class FeedingQueueRunner(qr.QueueRunner):
  """A queue runner that allows the feeding of values such as numpy arrays."""

  def __init__(self, queue=None, enqueue_ops=None, close_op=None,
               cancel_op=None, feed_fn=None):
    """Initialize the queue runner.

    For further documentation, see `queue_runner.py`. Note that
    `FeedingQueueRunner` does not support construction from protobuffer nor
    serialization to protobuffer.

    Args:
      queue: A `Queue`.
      enqueue_ops: List of enqueue ops to run in threads later.
      close_op: Op to close the queue. Pending enqueue ops are preserved.
      cancel_op: Op to close the queue and cancel pending enqueue ops.
      feed_fn: a function that returns a dictionary mapping fed `Tensor`s to
      values.
    """
    super(FeedingQueueRunner, self).__init__(queue, enqueue_ops, close_op,
                                             cancel_op)
    self._feed_fn = feed_fn

  # pylint: disable=broad-except
  def _run(self, sess, enqueue_op, coord=None):
    """Execute the enqueue op in a loop, close the queue in case of error.

    Args:
      sess: A `Session`.
      enqueue_op: The `Operation` to run.
      coord: Optional `Coordinator` object for reporting errors and checking
        for stop conditions.

    """
    # TODO(jamieas): Reduce code duplication with `QueueRunner`.
    decremented = False
    try:
      while True:
        if coord and coord.should_stop():
          break
        try:
          feed_dict = None if self._feed_fn is None else self._feed_fn()
          sess.run(enqueue_op, feed_dict=feed_dict)
        except errors.OutOfRangeError:
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

  def _init_from_proto(self, queue_runner_def):
    raise NotImplementedError(
        "{} does not support initialization from proto.".format(type(
            self).__name__))

  def to_proto(self):
    raise NotImplementedError(
        "{} does not support serialization to proto.".format(type(
            self).__name__))
