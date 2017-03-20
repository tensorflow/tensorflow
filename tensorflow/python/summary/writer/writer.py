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
"""Provides an API for generating Event protocol buffers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import summary_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import plugin_asset
from tensorflow.python.summary.writer.event_file_writer import EventFileWriter


_PLUGINS_DIR = "plugins"


class SummaryToEventTransformer(object):
  """Abstractly implements the SummaryWriter API.

  This API basically implements a number of endpoints (add_summary,
  add_session_log, etc). The endpoints all generate an event protobuf, which is
  passed to the contained event_writer.
  """

  def __init__(self, event_writer, graph=None, graph_def=None):
    """Creates a `SummaryWriter` and an event file.

    On construction the summary writer creates a new event file in `logdir`.
    This event file will contain `Event` protocol buffers constructed when you
    call one of the following functions: `add_summary()`, `add_session_log()`,
    `add_event()`, or `add_graph()`.

    If you pass a `Graph` to the constructor it is added to
    the event file. (This is equivalent to calling `add_graph()` later).

    TensorBoard will pick the graph from the file and display it graphically so
    you can interactively explore the graph you built. You will usually pass
    the graph from the session in which you launched it:

    ```python
    ...create a graph...
    # Launch the graph in a session.
    sess = tf.Session()
    # Create a summary writer, add the 'graph' to the event file.
    writer = tf.summary.FileWriter(<some-directory>, sess.graph)
    ```


    Args:
      event_writer: An EventWriter. Implements add_event and get_logdir.
      graph: A `Graph` object, such as `sess.graph`.
      graph_def: DEPRECATED: Use the `graph` argument instead.
    """
    self.event_writer = event_writer
    # For storing used tags for session.run() outputs.
    self._session_run_tags = {}
    if graph is not None or graph_def is not None:
      # Calling it with both graph and graph_def for backward compatibility.
      self.add_graph(graph=graph, graph_def=graph_def)
      # Also export the meta_graph_def in this case.
      # graph may itself be a graph_def due to positional arguments
      maybe_graph_as_def = (
          graph.as_graph_def(add_shapes=True) if isinstance(graph, ops.Graph)
          else graph)
      self.add_meta_graph(
          meta_graph.create_meta_graph_def(
              graph_def=graph_def or maybe_graph_as_def))

  def add_summary(self, summary, global_step=None):
    """Adds a `Summary` protocol buffer to the event file.

    This method wraps the provided summary in an `Event` protocol buffer
    and adds it to the event file.

    You can pass the result of evaluating any summary op, using
    @{tf.Session.run} or
    @{tf.Tensor.eval}, to this
    function. Alternatively, you can pass a `tf.Summary` protocol
    buffer that you populate with your own data. The latter is
    commonly done to report evaluation results in event files.

    Args:
      summary: A `Summary` protocol buffer, optionally serialized as a string.
      global_step: Number. Optional global step value to record with the
        summary.
    """
    if isinstance(summary, bytes):
      summ = summary_pb2.Summary()
      summ.ParseFromString(summary)
      summary = summ
    event = event_pb2.Event(summary=summary)
    self._add_event(event, global_step)

  def add_session_log(self, session_log, global_step=None):
    """Adds a `SessionLog` protocol buffer to the event file.

    This method wraps the provided session in an `Event` protocol buffer
    and adds it to the event file.

    Args:
      session_log: A `SessionLog` protocol buffer.
      global_step: Number. Optional global step value to record with the
        summary.
    """
    event = event_pb2.Event(session_log=session_log)
    self._add_event(event, global_step)

  def _add_graph_def(self, graph_def, global_step=None):
    graph_bytes = graph_def.SerializeToString()
    event = event_pb2.Event(graph_def=graph_bytes)
    self._add_event(event, global_step)

  def add_graph(self, graph, global_step=None, graph_def=None):
    """Adds a `Graph` to the event file.

    The graph described by the protocol buffer will be displayed by
    TensorBoard. Most users pass a graph in the constructor instead.

    Args:
      graph: A `Graph` object, such as `sess.graph`.
      global_step: Number. Optional global step counter to record with the
        graph.
      graph_def: DEPRECATED. Use the `graph` parameter instead.

    Raises:
      ValueError: If both graph and graph_def are passed to the method.
    """

    if graph is not None and graph_def is not None:
      raise ValueError("Please pass only graph, or graph_def (deprecated), "
                       "but not both.")

    if isinstance(graph, ops.Graph) or isinstance(graph_def, ops.Graph):
      # The user passed a `Graph`.

      # Check if the user passed it via the graph or the graph_def argument and
      # correct for that.
      if not isinstance(graph, ops.Graph):
        logging.warning("When passing a `Graph` object, please use the `graph`"
                        " named argument instead of `graph_def`.")
        graph = graph_def

      # Serialize the graph with additional info.
      true_graph_def = graph.as_graph_def(add_shapes=True)
      self._write_plugin_assets(graph)
    elif (isinstance(graph, graph_pb2.GraphDef) or
          isinstance(graph_def, graph_pb2.GraphDef)):
      # The user passed a `GraphDef`.
      logging.warning("Passing a `GraphDef` to the SummaryWriter is deprecated."
                      " Pass a `Graph` object instead, such as `sess.graph`.")

      # Check if the user passed it via the graph or the graph_def argument and
      # correct for that.
      if isinstance(graph, graph_pb2.GraphDef):
        true_graph_def = graph
      else:
        true_graph_def = graph_def

    else:
      # The user passed neither `Graph`, nor `GraphDef`.
      raise TypeError("The passed graph must be an instance of `Graph` "
                      "or the deprecated `GraphDef`")
    # Finally, add the graph_def to the summary writer.
    self._add_graph_def(true_graph_def, global_step)

  def _write_plugin_assets(self, graph):
    plugin_assets = plugin_asset.get_all_plugin_assets(graph)
    logdir = self.event_writer.get_logdir()
    for asset_container in plugin_assets:
      plugin_name = asset_container.plugin_name
      plugin_dir = os.path.join(logdir, _PLUGINS_DIR, plugin_name)
      gfile.MakeDirs(plugin_dir)
      assets = asset_container.assets()
      for (asset_name, content) in assets.items():
        asset_path = os.path.join(plugin_dir, asset_name)
        with gfile.Open(asset_path, "w") as f:
          f.write(content)

  def add_meta_graph(self, meta_graph_def, global_step=None):
    """Adds a `MetaGraphDef` to the event file.

    The `MetaGraphDef` allows running the given graph via
    `saver.import_meta_graph()`.

    Args:
      meta_graph_def: A `MetaGraphDef` object, often as returned by
        `saver.export_meta_graph()`.
      global_step: Number. Optional global step counter to record with the
        graph.

    Raises:
      TypeError: If both `meta_graph_def` is not an instance of `MetaGraphDef`.
    """
    if not isinstance(meta_graph_def, meta_graph_pb2.MetaGraphDef):
      raise TypeError("meta_graph_def must be type MetaGraphDef, saw type: %s"
                      % type(meta_graph_def))
    meta_graph_bytes = meta_graph_def.SerializeToString()
    event = event_pb2.Event(meta_graph_def=meta_graph_bytes)
    self._add_event(event, global_step)

  def add_run_metadata(self, run_metadata, tag, global_step=None):
    """Adds a metadata information for a single session.run() call.

    Args:
      run_metadata: A `RunMetadata` protobuf object.
      tag: The tag name for this metadata.
      global_step: Number. Optional global step counter to record with the
        StepStats.

    Raises:
      ValueError: If the provided tag was already used for this type of event.
    """
    if tag in self._session_run_tags:
      raise ValueError("The provided tag was already used for this event type")
    self._session_run_tags[tag] = True

    tagged_metadata = event_pb2.TaggedRunMetadata()
    tagged_metadata.tag = tag
    # Store the `RunMetadata` object as bytes in order to have postponed
    # (lazy) deserialization when used later.
    tagged_metadata.run_metadata = run_metadata.SerializeToString()
    event = event_pb2.Event(tagged_run_metadata=tagged_metadata)
    self._add_event(event, global_step)

  def _add_event(self, event, step):
    event.wall_time = time.time()
    if step is not None:
      event.step = int(step)
    self.event_writer.add_event(event)


class FileWriter(SummaryToEventTransformer):
  """Writes `Summary` protocol buffers to event files.

  The `FileWriter` class provides a mechanism to create an event file in a
  given directory and add summaries and events to it. The class updates the
  file contents asynchronously. This allows a training program to call methods
  to add data to the file directly from the training loop, without slowing down
  training.
  """

  def __init__(self,
               logdir,
               graph=None,
               max_queue=10,
               flush_secs=120,
               graph_def=None):
    """Creates a `FileWriter` and an event file.

    On construction the summary writer creates a new event file in `logdir`.
    This event file will contain `Event` protocol buffers constructed when you
    call one of the following functions: `add_summary()`, `add_session_log()`,
    `add_event()`, or `add_graph()`.

    If you pass a `Graph` to the constructor it is added to
    the event file. (This is equivalent to calling `add_graph()` later).

    TensorBoard will pick the graph from the file and display it graphically so
    you can interactively explore the graph you built. You will usually pass
    the graph from the session in which you launched it:

    ```python
    ...create a graph...
    # Launch the graph in a session.
    sess = tf.Session()
    # Create a summary writer, add the 'graph' to the event file.
    writer = tf.summary.FileWriter(<some-directory>, sess.graph)
    ```

    The other arguments to the constructor control the asynchronous writes to
    the event file:

    *  `flush_secs`: How often, in seconds, to flush the added summaries
       and events to disk.
    *  `max_queue`: Maximum number of summaries or events pending to be
       written to disk before one of the 'add' calls block.

    Args:
      logdir: A string. Directory where event file will be written.
      graph: A `Graph` object, such as `sess.graph`.
      max_queue: Integer. Size of the queue for pending events and summaries.
      flush_secs: Number. How often, in seconds, to flush the
        pending events and summaries to disk.
      graph_def: DEPRECATED: Use the `graph` argument instead.
    """
    event_writer = EventFileWriter(logdir, max_queue, flush_secs)
    super(FileWriter, self).__init__(event_writer, graph, graph_def)

  def get_logdir(self):
    """Returns the directory where event file will be written."""
    return self.event_writer.get_logdir()

  def add_event(self, event):
    """Adds an event to the event file.

    Args:
      event: An `Event` protocol buffer.
    """
    self.event_writer.add_event(event)

  def flush(self):
    """Flushes the event file to disk.

    Call this method to make sure that all pending events have been written to
    disk.
    """
    self.event_writer.flush()

  def close(self):
    """Flushes the event file to disk and close the file.

    Call this method when you do not need the summary writer anymore.
    """
    self.event_writer.close()

  def reopen(self):
    """Reopens the EventFileWriter.

    Can be called after `close()` to add more events in the same directory.
    The events will go into a new events file.

    Does nothing if the EventFileWriter was not closed.
    """
    self.event_writer.reopen()
