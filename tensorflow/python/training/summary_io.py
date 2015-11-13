"""Reads Summaries from and writes Summaries to event files."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import Queue
import threading
import time

import six

from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import gfile


class SummaryWriter(object):
  """Writes `Summary` protocol buffers to event files.

  The `SummaryWriter` class provides a mechanism to create an event file in a
  given directory and add summaries and events to it. The class updates the
  file contents asynchronously. This allows a training program to call methods
  to add data to the file directly from the training loop, without slowing down
  training.

  @@__init__

  @@add_summary
  @@add_event
  @@add_graph

  @@flush
  @@close
  """

  def __init__(self, logdir, graph_def=None, max_queue=10, flush_secs=120):
    """Creates a `SummaryWriter` and an event file.

    On construction the summary writer creates a new event file in `logdir`.
    This event file will contain `Event` protocol buffers constructed when you
    call one of the following functions: `add_summary()`, `add_event()`, or
    `add_graph()`.

    If you pass a `graph_def` protocol buffer to the constructor it is added to
    the event file. (This is equivalent to calling `add_graph()` later).

    TensorBoard will pick the graph from the file and display it graphically so
    you can interactively explore the graph you built. You will usually pass
    the graph from the session in which you launched it:

    ```python
    ...create a graph...
    # Launch the graph in a session.
    sess = tf.Session()
    # Create a summary writer, add the 'graph_def' to the event file.
    writer = tf.train.SummaryWriter(<some-directory>, sess.graph_def)
    ```

    The other arguments to the constructor control the asynchronous writes to
    the event file:

    *  `flush_secs`: How often, in seconds, to flush the added summaries
       and events to disk.
    *  `max_queue`: Maximum number of summaries or events pending to be
       written to disk before one of the 'add' calls block.

    Args:
      logdir: A string. Directory where event file will be written.
      graph_def: A `GraphDef` protocol buffer.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
    """
    self._logdir = logdir
    if not gfile.IsDirectory(self._logdir):
      gfile.MakeDirs(self._logdir)
    self._event_queue = Queue.Queue(max_queue)
    self._ev_writer = pywrap_tensorflow.EventsWriter(
        os.path.join(self._logdir, "events"))
    self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                      flush_secs)
    self._worker.start()
    if graph_def is not None:
      self.add_graph(graph_def)

  def add_summary(self, summary, global_step=None):
    """Adds a `Summary` protocol buffer to the event file.

    This method wraps the provided summary in an `Event` procotol buffer
    and adds it to the event file.

    You can pass the result of evaluating any summary op, using
    [`Session.run()`](client.md#Session.run] or
    [`Tensor.eval()`](framework.md#Tensor.eval), to this
    function. Alternatively, you can pass a `tf.Summary` protocol
    buffer that you populate with your own data. The latter is
    commonly done to report evaluation results in event files.

    Args:
      summary: A `Summary` protocol buffer, optionally serialized as a string.
      global_step: Number. Optional global step value to record with the
        summary.
    """
    if isinstance(summary, six.binary_type):
      summ = summary_pb2.Summary()
      summ.ParseFromString(summary)
      summary = summ
    event = event_pb2.Event(wall_time=time.time(), summary=summary)
    if global_step is not None:
      event.step = int(global_step)
    self.add_event(event)

  def add_event(self, event):
    """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
    self._event_queue.put(event)

  def add_graph(self, graph_def, global_step=None):
    """Adds a `GraphDef` protocol buffer to the event file.

    The graph described by the protocol buffer will be displayed by
    TensorBoard. Most users pass a graph in the constructor instead.

    Args:
      graph_def: A `GraphDef` protocol buffer.
      global_step: Number. Optional global step counter to record with the
        graph.
    """
    event = event_pb2.Event(wall_time=time.time(), graph_def=graph_def)
    if global_step is not None:
      event.step = int(global_step)
    self._event_queue.put(event)

  def flush(self):
    """Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    """
    self._event_queue.join()
    self._ev_writer.Flush()

  def close(self):
    """Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    """
    self.flush()
    self._ev_writer.Close()


class _EventLoggerThread(threading.Thread):
  """Thread that logs events."""

  def __init__(self, queue, ev_writer, flush_secs):
    """Creates an _EventLoggerThread.

    Args:
      queue: a Queue from which to dequeue events.
      ev_writer: an event writer. Used to log brain events for
       the visualizer.
      flush_secs: How often, in seconds, to flush the
        pending file to disk.
    """
    threading.Thread.__init__(self)
    self.daemon = True
    self._queue = queue
    self._ev_writer = ev_writer
    self._flush_secs = flush_secs
    # The first event will be flushed immediately.
    self._next_event_flush_time = 0

  def run(self):
    while True:
      event = self._queue.get()
      try:
        self._ev_writer.WriteEvent(event)
        # Flush the event writer every so often.
        now = time.time()
        if now > self._next_event_flush_time:
          self._ev_writer.Flush()
          # Do it again in two minutes.
          self._next_event_flush_time = now + self._flush_secs
      finally:
        self._queue.task_done()


def summary_iterator(path):
  """An iterator for reading `Event` protocol buffers from an event file.

  You can use this function to read events written to an event file. It returns
  a Python iterator that yields `Event` protocol buffers.

  Example: Print the contents of an events file.

  ```python
  for e in tf.summary_iterator(path to events file):
      print e
  ```

  Example: Print selected summary values.

  ```python
  # This example supposes that the events file contains summaries with a
  # summary value tag 'loss'.  These could have been added by calling
  # `add_summary()`, passing the output of a scalar summary op created with
  # with: `tf.scalar_summary(['loss'], loss_tensor)`.
  for e in tf.summary_iterator(path to events file):
      for v in e.summary.value:
          if v.tag == 'loss':
              print v.simple_value
  ```

  See the protocol buffer definitions of
  [Event](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/util/event.proto)
  and
  [Summary](https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/core/framework/summary.proto)
  for more information about their attributes.

  Args:
    path: The path to an event file created by a `SummaryWriter`.

  Yields:
    `Event` protocol buffers.
  """
  for r in tf_record.tf_record_iterator(path):
    yield event_pb2.Event.FromString(r)
