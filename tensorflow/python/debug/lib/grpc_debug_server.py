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
"""gRPC debug server in Python."""
# pylint: disable=g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import threading
import time

from concurrent import futures
import grpc
from six.moves import queue

from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat

DebugWatch = collections.namedtuple("DebugWatch",
                                    ["node_name", "output_slot", "debug_op"])


def _state_change(new_state, node_name, output_slot, debug_op):
  state_change = debug_service_pb2.EventReply.DebugOpStateChange()
  state_change.state = new_state
  state_change.node_name = node_name
  state_change.output_slot = output_slot
  state_change.debug_op = debug_op
  return state_change


class EventListenerBaseStreamHandler(object):
  """Per-stream handler of EventListener gRPC streams."""

  def __init__(self):
    """Constructor of EventListenerBaseStreamHandler."""

  def on_core_metadata_event(self, event):
    """Callback for core metadata.

    Args:
      event: The Event proto that carries a JSON string in its
        `log_message.message` field.

    Returns:
      `None` or an `EventReply` proto to be sent back to the client. If `None`,
      an `EventReply` proto construct with the default no-arg constructor will
      be sent back to the client.
    """
    raise NotImplementedError(
        "on_core_metadata_event() is not implemented in the base servicer "
        "class")

  def on_graph_def(self, graph_def, device_name, wall_time):
    """Callback for Event proto received through the gRPC stream.

    This Event proto carries a GraphDef, encoded as bytes, in its graph_def
    field.

    Args:
      graph_def: A GraphDef object.
      device_name: Name of the device on which the graph was created.
      wall_time: An epoch timestamp (in microseconds) for the graph.

    Returns:
      `None` or an `EventReply` proto to be sent back to the client. If `None`,
      an `EventReply` proto construct with the default no-arg constructor will
      be sent back to the client.
    """
    raise NotImplementedError(
        "on_graph_def() is not implemented in the base servicer class")

  def on_value_event(self, event):
    """Callback for Event proto received through the gRPC stream.

    This Event proto carries a Tensor in its summary.value[0] field.

    Args:
      event: The Event proto from the stream to be processed.
    """
    raise NotImplementedError(
        "on_value_event() is not implemented in the base servicer class")


class EventListenerBaseServicer(debug_service_pb2_grpc.EventListenerServicer):
  """Base Python class for gRPC debug server."""

  def __init__(self, server_port, stream_handler_class):
    """Constructor.

    Args:
      server_port: (int) Port number to bind to.
      stream_handler_class: A class of the base class
        `EventListenerBaseStreamHandler` that will be used to constructor
        stream handler objects during `SendEvents` calls.
    """

    self._server_port = server_port
    self._stream_handler_class = stream_handler_class

    self._server_lock = threading.Lock()
    self._server_started = False
    self._stop_requested = False

    self._debug_ops_state_change_queue = queue.Queue()
    self._gated_grpc_debug_watches = set()
    self._breakpoints = set()

  def SendEvents(self, request_iterator, context):
    """Implementation of the SendEvents service method.

    This method receives streams of Event protos from the client, and processes
    them in ways specified in the on_event() callback. The stream is
    bi-directional, but currently only the client-to-server stream (i.e., the
    stream from the debug ops to the server) is used.

    Args:
      request_iterator: The incoming stream of Event protos.
      context: Server context.

    Raises:
      ValueError: If there are more than one core metadata events.

    Yields:
      An empty stream of responses.
    """
    core_metadata_count = 0

    # A map from GraphDef hash to a list of received chunks.
    graph_def_chunks = {}
    tensor_chunks = {}

    stream_handler = None
    for event in request_iterator:
      if not stream_handler:
        stream_handler = self._stream_handler_class()

      if event.summary and event.summary.value:
        # An Event proto carrying a tensor value.
        maybe_tensor_event = self._process_tensor_event_in_chunks(
            event, tensor_chunks)
        if maybe_tensor_event:
          event_reply = stream_handler.on_value_event(maybe_tensor_event)
          if event_reply is not None:
            yield self._process_debug_op_state_changes(event_reply)
      else:
        # Non-tensor-value Event.
        if event.graph_def:
          # GraphDef-carrying Event.
          maybe_graph_def, maybe_device_name, maybe_wall_time = (
              self._process_encoded_graph_def_in_chunks(
                  event, graph_def_chunks))
          if maybe_graph_def:
            reply = stream_handler.on_graph_def(
                maybe_graph_def, maybe_device_name, maybe_wall_time)
            yield self._process_debug_op_state_changes(reply)
        elif event.log_message.message:
          # Core metadata-carrying Event.
          core_metadata_count += 1
          if core_metadata_count > 1:
            raise ValueError(
                "Expected one core metadata event; received multiple")
          reply = stream_handler.on_core_metadata_event(event)
          yield self._process_debug_op_state_changes(reply)

  def _process_debug_op_state_changes(self, event_reply=None):
    """Dequeue and process all the queued debug-op state change protos.

    Include all the debug-op state change protos in a `EventReply` proto.

    Args:
      event_reply: An `EventReply` to add the `DebugOpStateChange` protos to,
        or `None`.

    Returns:
      An `EventReply` proto with the dequeued `DebugOpStateChange` protos (if
        any) added.
    """
    if event_reply is None:
      event_reply = debug_service_pb2.EventReply()
    while not self._debug_ops_state_change_queue.empty():
      state_change = self._debug_ops_state_change_queue.get()
      debug_node_key = (state_change.node_name, state_change.output_slot,
                        state_change.debug_op)
      if (state_change.state ==
          debug_service_pb2.EventReply.DebugOpStateChange.READ_WRITE):
        logging.info("Adding breakpoint %s:%d:%s", state_change.node_name,
                     state_change.output_slot, state_change.debug_op)
        self._breakpoints.add(debug_node_key)
      elif (state_change.state ==
            debug_service_pb2.EventReply.DebugOpStateChange.READ_ONLY):
        logging.info("Adding watchpoint %s:%d:%s", state_change.node_name,
                     state_change.output_slot, state_change.debug_op)
        if debug_node_key in self._breakpoints:
          self._breakpoints.discard(debug_node_key)
      elif (state_change.state ==
            debug_service_pb2.EventReply.DebugOpStateChange.DISABLED):
        logging.info("Removing watchpoint or breakpoint: %s:%d:%s",
                     state_change.node_name, state_change.output_slot,
                     state_change.debug_op)
        if debug_node_key in self._breakpoints:
          self._breakpoints.discard(debug_node_key)
        else:
          logging.warn(
              "Attempting to remove a non-existent debug node key: %s",
              debug_node_key)
      new_state_change = event_reply.debug_op_state_changes.add()
      new_state_change.CopyFrom(state_change)
    return event_reply

  def _process_tensor_event_in_chunks(self, event, tensor_chunks):
    """Possibly reassemble event chunks.

    Due to gRPC's message size limit, a large tensor can be encapsulated in
    multiple Event proto chunks to be sent through the debugger stream. This
    method keeps track of the chunks that have arrived, reassemble all chunks
    corresponding to a tensor when they have arrived and return the reassembled
    Event proto.

    Args:
      event: The single Event proto that has arrived.
      tensor_chunks: A dict used to keep track of the Event protos that have
        arrived but haven't been reassembled.

    Returns:
      If all Event protos corresponding to a tensor have arrived, returns the
      reassembled Event proto. Otherwise, return None.
    """

    value = event.summary.value[0]
    debugger_plugin_metadata = json.loads(
        compat.as_text(value.metadata.plugin_data.content))
    device_name = debugger_plugin_metadata["device"]
    num_chunks = debugger_plugin_metadata["numChunks"]
    chunk_index = debugger_plugin_metadata["chunkIndex"]

    if num_chunks <= 1:
      return event

    debug_node_name = value.node_name
    timestamp = int(event.wall_time)
    tensor_key = "%s_%s_%d" % (device_name, debug_node_name, timestamp)

    if tensor_key not in tensor_chunks:
      tensor_chunks[tensor_key] = [None] * num_chunks

    chunks = tensor_chunks[tensor_key]
    if value.tensor.tensor_content:
      chunks[chunk_index] = value.tensor
    elif value.tensor.string_val:
      chunks[chunk_index] = event

    if None not in chunks:
      if value.tensor.tensor_content:
        event.summary.value[0].tensor.tensor_content = b"".join(
            chunk.tensor_content for chunk in chunks)
        del tensor_chunks[tensor_key]
        return event
      elif value.tensor.string_val:
        merged_event = chunks[0]
        for chunk in chunks[1:]:
          merged_event.summary.value[0].tensor.string_val.extend(
              list(chunk.summary.value[0].tensor.string_val))
        return merged_event

  def _process_encoded_graph_def_in_chunks(self,
                                           event,
                                           graph_def_chunks):
    """Process an Event proto containing a chunk of encoded GraphDef.

    Args:
      event: the Event proto containing the chunk of encoded GraphDef.
      graph_def_chunks: A dict mapping keys for GraphDefs (i.e.,
      "<graph_def_hash>,<device_name>,<wall_time>") to a list of chunks of
      encoded GraphDefs.

    Returns:
      If all chunks of the GraphDef have arrived,
        return decoded GraphDef proto, device name, wall_time.
      Otherwise,
        return None, None, None.
    """
    graph_def = graph_pb2.GraphDef()
    index_bar_0 = event.graph_def.find(b"|")
    index_bar_1 = event.graph_def.find(b"|", index_bar_0 + 1)
    index_bar_2 = event.graph_def.find(b"|", index_bar_1 + 1)
    graph_def_hash_device_timestamp = event.graph_def[:index_bar_0]
    chunk_index = int(event.graph_def[index_bar_0 + 1 : index_bar_1])
    num_chunks = int(event.graph_def[index_bar_1 + 1 : index_bar_2])
    if graph_def_hash_device_timestamp not in graph_def_chunks:
      graph_def_chunks[graph_def_hash_device_timestamp] = [None] * num_chunks
    graph_def_chunks[graph_def_hash_device_timestamp][
        chunk_index] = event.graph_def[index_bar_2 + 1:]
    if all(graph_def_chunks[graph_def_hash_device_timestamp]):
      device_name = graph_def_hash_device_timestamp.split(b",")[1]
      wall_time = int(graph_def_hash_device_timestamp.split(b",")[2])
      graph_def.ParseFromString(
          b"".join(graph_def_chunks[graph_def_hash_device_timestamp]))
      del graph_def_chunks[graph_def_hash_device_timestamp]
      self._process_graph_def(graph_def)
      return graph_def, device_name, wall_time
    else:
      return None, None, None

  def _process_graph_def(self, graph_def):
    for node_def in graph_def.node:
      if (debug_graphs.is_debug_node(node_def.name) and
          node_def.attr["gated_grpc"].b):
        node_name, output_slot, _, debug_op = (
            debug_graphs.parse_debug_node_name(node_def.name))
        self._gated_grpc_debug_watches.add(
            DebugWatch(node_name, output_slot, debug_op))

  def run_server(self, blocking=True):
    """Start running the server.

    Args:
      blocking: If `True`, block until `stop_server()` is invoked.

    Raises:
      ValueError: If server stop has already been requested, or if the server
        has already started running.
    """
    self._server_lock.acquire()
    try:
      if self._stop_requested:
        raise ValueError("Server has already stopped")
      if self._server_started:
        raise ValueError("Server has already started running")

      no_max_message_sizes = [("grpc.max_receive_message_length", -1),
                              ("grpc.max_send_message_length", -1)]
      self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                                options=no_max_message_sizes)
      debug_service_pb2_grpc.add_EventListenerServicer_to_server(self,
                                                                 self.server)
      self.server.add_insecure_port("[::]:%d" % self._server_port)
      self.server.start()
      self._server_started = True
    finally:
      self._server_lock.release()

    if blocking:
      while not self._stop_requested:
        time.sleep(1.0)

  def stop_server(self, grace=1.0):
    """Request server stopping.

    Once stopped, server cannot be stopped or started again. This method is
    non-blocking. Call `wait()` on the returned event to block until the server
    has completely stopped.

    Args:
      grace: Grace period in seconds to be used when calling `server.stop()`.

    Raises:
      ValueError: If server stop has already been requested, or if the server
        has not started running yet.

    Returns:
      A threading.Event that will be set when the server has completely stopped.
    """
    self._server_lock.acquire()
    try:
      if not self._server_started:
        raise ValueError("Server has not started running")
      if self._stop_requested:
        raise ValueError("Server has already stopped")

      self._stop_requested = True
      return self.server.stop(grace=grace)
    finally:
      self._server_lock.release()

  def request_watch(self, node_name, output_slot, debug_op, breakpoint=False):  # pylint: disable=redefined-builtin
    """Request enabling a debug tensor watchpoint or breakpoint.

    This will let the server send a EventReply to the client side
    (i.e., the debugged TensorFlow runtime process) to request adding a watch
    key (i.e., <node_name>:<output_slot>:<debug_op>) to the list of enabled
    watch keys. The list applies only to debug ops with the attribute
    gated_grpc=True.

    To disable the watch, use `request_unwatch()`.

    Args:
      node_name: (`str`) name of the node that the to-be-watched tensor belongs
        to, e.g., "hidden/Weights".
      output_slot: (`int`) output slot index of the tensor to watch.
      debug_op: (`str`) name of the debug op to enable. This should not include
        any attribute substrings.
      breakpoint: (`bool`) Iff `True`, the debug op will block and wait until it
        receives an `EventReply` response from the server. The `EventReply`
        proto may carry a TensorProto that modifies the value of the debug op's
        output tensor.
    """
    self._debug_ops_state_change_queue.put(
        _state_change(
            debug_service_pb2.EventReply.DebugOpStateChange.READ_WRITE
            if breakpoint
            else debug_service_pb2.EventReply.DebugOpStateChange.READ_ONLY,
            node_name, output_slot, debug_op))

  def request_unwatch(self, node_name, output_slot, debug_op):
    """Request disabling a debug tensor watchpoint or breakpoint.

    This is the opposite of `request_watch()`.

    Args:
      node_name: (`str`) name of the node that the to-be-watched tensor belongs
        to, e.g., "hidden/Weights".
      output_slot: (`int`) output slot index of the tensor to watch.
      debug_op: (`str`) name of the debug op to enable. This should not include
        any attribute substrings.
    """
    self._debug_ops_state_change_queue.put(
        _state_change(
            debug_service_pb2.EventReply.DebugOpStateChange.DISABLED, node_name,
            output_slot, debug_op))

  @property
  def breakpoints(self):
    """Get a set of the currently-activated breakpoints.

    Returns:
      A `set` of 3-tuples: (node_name, output_slot, debug_op), e.g.,
        {("MatMul", 0, "DebugIdentity")}.
    """
    return self._breakpoints

  def gated_grpc_debug_watches(self):
    """Get the list of debug watches with attribute gated_grpc=True.

    Since the server receives `GraphDef` from the debugged runtime, it can only
    return such debug watches that it has received so far.

    Returns:
      A `list` of `DebugWatch` `namedtuples` representing the debug watches with
      gated_grpc=True. Each `namedtuple` element has the attributes:
        `node_name` as a `str`,
        `output_slot` as an `int`,
        `debug_op` as a `str`.
    """
    return list(self._gated_grpc_debug_watches)

  def SendTracebacks(self, request, context):
    """Base implementation of the handling of SendTracebacks calls.

    The base implementation does nothing with the incoming request.
    Override in an implementation of the server if necessary.

    Args:
      request: A `CallTraceback` proto, containing information about the
        type (e.g., graph vs. eager execution) and source-code traceback of the
        call and (any) associated `tf.Graph`s.
      context: Server context.

    Returns:
      A `EventReply` proto.
    """
    return debug_service_pb2.EventReply()

  def SendSourceFiles(self, request, context):
    """Base implementation of the handling of SendSourceFiles calls.

    The base implementation does nothing with the incoming request.
    Override in an implementation of the server if necessary.

    Args:
      request: A `DebuggedSourceFiles` proto, containing the path, content, size
        and last-modified timestamp of source files.
      context: Server context.

    Returns:
      A `EventReply` proto.
    """
    return debug_service_pb2.EventReply()
