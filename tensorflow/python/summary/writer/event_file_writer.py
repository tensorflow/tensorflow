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
"""Writes events to disk in a logdir."""

import collections
import os.path
import sys
import threading
import time

from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat


class EventFileWriter:
  """Writes `Event` protocol buffers to an event file.

  The `EventFileWriter` class creates an event file in the specified directory,
  and asynchronously writes Event protocol buffers to the file. The Event file
  is encoded using the tfrecord format, which is similar to RecordIO.

  This class is not thread-safe.
  """

  def __init__(self, logdir, max_queue=10, flush_secs=120,
               filename_suffix=None):
    """Creates a `EventFileWriter` and an event file to write to.

    On construction the summary writer creates a new event file in `logdir`.
    This event file will contain `Event` protocol buffers, which are written to
    disk via the add_event method.

    The other arguments to the constructor control the asynchronous writes to
    the event file:

    *  `flush_secs`: How often, in seconds, to flush the added summaries
       and events to disk.
    *  `max_queue`: Maximum number of summaries or events pending to be
       written to disk before one of the 'add' calls block.

    Args:
      logdir: A string. Directory where event file will be written.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      filename_suffix: A string. Every event file's name is suffixed with
        `filename_suffix`.
    """
    self._logdir = str(logdir)
    gfile.MakeDirs(self._logdir)
    self._max_queue = max_queue
    self._flush_secs = flush_secs
    self._flush_complete = threading.Event()
    self._flush_sentinel = object()
    self._close_sentinel = object()
    self._ev_writer = _pywrap_events_writer.EventsWriter(
        compat.as_bytes(os.path.join(self._logdir, "events")))
    if filename_suffix:
      self._ev_writer.InitWithSuffix(compat.as_bytes(filename_suffix))
    self._initialize()
    self._closed = False

  def _initialize(self):
    """Initializes or re-initializes the queue and writer thread.

    The EventsWriter itself does not need to be re-initialized explicitly,
    because it will auto-initialize itself if used after being closed.
    """
    self._event_queue = CloseableQueue(self._max_queue)
    self._worker = _EventLoggerThread(self._event_queue, self._ev_writer,
                                      self._flush_secs, self._flush_complete,
                                      self._flush_sentinel,
                                      self._close_sentinel)

    self._worker.start()

  def get_logdir(self):
    """Returns the directory where event file will be written."""
    return self._logdir

  def reopen(self):
    """Reopens the EventFileWriter.

    Can be called after `close()` to add more events in the same directory.
    The events will go into a new events file.

    Does nothing if the EventFileWriter was not closed.
    """
    if self._closed:
      self._initialize()
      self._closed = False

  def add_event(self, event):
    """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
    if not self._closed:
      self._try_put(event)

  def _try_put(self, item):
    """Attempts to enqueue an item to the event queue.

    If the queue is closed, this will close the EventFileWriter and reraise the
    exception that caused the queue closure, if one exists.

    Args:
      item: the item to enqueue
    """
    try:
      self._event_queue.put(item)
    except QueueClosedError:
      self._internal_close()
      if self._worker.failure_exc_info:
        _, exception, _ = self._worker.failure_exc_info
        raise exception from None

  def flush(self):
    """Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    """
    if not self._closed:
      # Request a flush operation by enqueuing a sentinel and then waiting for
      # the writer thread to mark the flush as complete.
      self._flush_complete.clear()
      self._try_put(self._flush_sentinel)
      self._flush_complete.wait()
      if self._worker.failure_exc_info:
        self._internal_close()
        _, exception, _ = self._worker.failure_exc_info
        raise exception

  def close(self):
    """Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    """
    if not self._closed:
      self.flush()
      self._try_put(self._close_sentinel)
      self._internal_close()

  def _internal_close(self):
    self._closed = True
    self._worker.join()
    self._ev_writer.Close()


class _EventLoggerThread(threading.Thread):
  """Thread that logs events."""

  def __init__(self, queue, ev_writer, flush_secs, flush_complete,
               flush_sentinel, close_sentinel):
    """Creates an _EventLoggerThread.

    Args:
      queue: A CloseableQueue from which to dequeue events. The queue will be
        closed just before the thread exits, whether due to `close_sentinel` or
        any exception raised in the writing loop.
      ev_writer: An event writer. Used to log brain events for
        the visualizer.
      flush_secs: How often, in seconds, to flush the
        pending file to disk.
      flush_complete: A threading.Event that will be set whenever a flush
        operation requested via `flush_sentinel` has been completed.
      flush_sentinel: A sentinel element in queue that tells this thread to
        flush the writer and mark the current flush operation complete.
      close_sentinel: A sentinel element in queue that tells this thread to
        terminate and close the queue.
    """
    threading.Thread.__init__(self, name="EventLoggerThread")
    self.daemon = True
    self._queue = queue
    self._ev_writer = ev_writer
    self._flush_secs = flush_secs
    # The first event will be flushed immediately.
    self._next_event_flush_time = 0
    self._flush_complete = flush_complete
    self._flush_sentinel = flush_sentinel
    self._close_sentinel = close_sentinel
    # Populated when writing logic raises an exception and kills the thread.
    self.failure_exc_info = ()

  def run(self):
    try:
      while True:
        event = self._queue.get()
        if event is self._close_sentinel:
          return
        elif event is self._flush_sentinel:
          self._ev_writer.Flush()
          self._flush_complete.set()
        else:
          self._ev_writer.WriteEvent(event)
          # Flush the event writer every so often.
          now = time.time()
          if now > self._next_event_flush_time:
            self._ev_writer.Flush()
            self._next_event_flush_time = now + self._flush_secs
    except Exception as e:
      logging.error("EventFileWriter writer thread error: %s", e)
      self.failure_exc_info = sys.exc_info()
      raise
    finally:
      # When exiting the thread, always complete any pending flush operation
      # (to unblock flush() calls) and close the queue (to unblock add_event()
      # calls, including those used by flush() and close()), which ensures that
      # code using EventFileWriter doesn't deadlock if this thread dies.
      self._flush_complete.set()
      self._queue.close()


class CloseableQueue:
  """Stripped-down fork of the standard library Queue that is closeable."""

  def __init__(self, maxsize=0):
    """Create a queue object with a given maximum size.

    Args:
      maxsize: int size of queue. If <= 0, the queue size is infinite.
    """
    self._maxsize = maxsize
    self._queue = collections.deque()
    self._closed = False
    # Mutex must be held whenever queue is mutating; shared by conditions.
    self._mutex = threading.Lock()
    # Notify not_empty whenever an item is added to the queue; a
    # thread waiting to get is notified then.
    self._not_empty = threading.Condition(self._mutex)
    # Notify not_full whenever an item is removed from the queue;
    # a thread waiting to put is notified then.
    self._not_full = threading.Condition(self._mutex)

  def get(self):
    """Remove and return an item from the queue.

    If the queue is empty, blocks until an item is available.

    Returns:
      an item from the queue
    """
    with self._not_empty:
      while not self._queue:
        self._not_empty.wait()
      item = self._queue.popleft()
      self._not_full.notify()
      return item

  def put(self, item):
    """Put an item into the queue.

    If the queue is closed, fails immediately.

    If the queue is full, blocks until space is available or until the queue
    is closed by a call to close(), at which point this call fails.

    Args:
      item: an item to add to the queue

    Raises:
      QueueClosedError: if insertion failed because the queue is closed
    """
    with self._not_full:
      if self._closed:
        raise QueueClosedError()
      if self._maxsize > 0:
        while len(self._queue) == self._maxsize:
          self._not_full.wait()
          if self._closed:
            raise QueueClosedError()
      self._queue.append(item)
      self._not_empty.notify()

  def close(self):
    """Closes the queue, causing any pending or future `put()` calls to fail."""
    with self._not_full:
      self._closed = True
      self._not_full.notify_all()


class QueueClosedError(Exception):
  """Raised when CloseableQueue.put() fails because the queue is closed."""
