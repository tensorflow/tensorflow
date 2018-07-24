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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import session as tf_session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import gfile


class EventFileWriterV2(object):
  """Writes `Event` protocol buffers to an event file via the graph.

  The `EventFileWriterV2` class is backed by the summary file writer in the v2
  summary API (currently in tf.contrib.summary), so it uses a shared summary
  writer resource and graph ops to write events.

  As with the original EventFileWriter, this class will asynchronously write
  Event protocol buffers to the backing file. The Event file is encoded using
  the tfrecord format, which is similar to RecordIO.
  """

  def __init__(self, session, logdir, max_queue=10, flush_secs=120,
               filename_suffix=''):
    """Creates an `EventFileWriterV2` and an event file to write to.

    On construction, this calls `tf.contrib.summary.create_file_writer` within
    the default graph, which finds and returns a shared summary writer resource
    for `logdir` if one exists, and creates one if not. Creating the summary
    writer resource in turn creates a new event file in `logdir` to be filled
    with `Event` protocol buffers passed to `add_event`. Graph ops to control
    this writer resource are added to the default graph during this init call;
    stateful methods on this class will call `session.run()` on these ops.

    Note that because the underlying resource is shared, it is possible that
    other parts of the code using the same session may interact independently
    with the resource, e.g. by flushing or even closing it. It is the caller's
    responsibility to avoid any undesirable sharing in this regard.

    The remaining arguments to the constructor (`flush_secs`, `max_queue`, and
    `filename_suffix`) control the construction of the shared writer resource
    if one is created. If an existing resource is reused, these arguments have
    no effect.  See `tf.contrib.summary.create_file_writer` for details.

    Args:
      session: A `tf.Session`, or a callable that provides one which will be
        called on-demand. The session will hold the shared writer resource.
      logdir: A string. Directory where event file will be written.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      filename_suffix: A string. Every event file's name is suffixed with
        `filename_suffix`.

    Raises:
      ValueError: if `session` is not a `tf.Session` or a callable
    """
    if isinstance(session, tf_session.SessionInterface):
      self._session = lambda: session
    elif callable(session):
      self._session = session
    else:
      raise ValueError('session must be tf.Session or callable')
    self._logdir = logdir
    self._initialized = False
    self._closed = False
    if not gfile.IsDirectory(self._logdir):
      gfile.MakeDirs(self._logdir)

    with ops.name_scope('filewriter'):
      file_writer = summary_ops_v2.create_file_writer(
          logdir=self._logdir,
          max_queue=max_queue,
          flush_millis=flush_secs * 1000,
          filename_suffix=filename_suffix)
      with summary_ops_v2.always_record_summaries(), file_writer.as_default():
        self._event_placeholder = array_ops.placeholder_with_default(
            constant_op.constant('unused', dtypes.string),
            shape=[])
        self._add_event_op = summary_ops_v2.import_event(
            self._event_placeholder)
      self._init_op = file_writer.init()
      self._flush_op = file_writer.flush()
      self._close_op = file_writer.close()

  def _init_if_needed(self):
    if not self._initialized:
      self._session().run(self._init_op)
      self._initialized = True

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
      self._closed = False

  def add_event(self, event):
    """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
    if not self._closed:
      self._init_if_needed()
      event_pb = event.SerializeToString()
      self._session().run(
          self._add_event_op, feed_dict={self._event_placeholder: event_pb})

  def flush(self):
    """Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    """
    if not self._closed:
      self._init_if_needed()
      self._session().run(self._flush_op)

  def close(self):
    """Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    """
    if not self._closed:
      self._init_if_needed()
      self.flush()
      self._session().run(self._close_op)
      self._closed = True
      self._initialized = False
