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
"""GRPC debug server for testing."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import errno
import functools
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time

import portpicker

from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_server
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.ops import variables
from tensorflow.python.util import compat


def _get_dump_file_path(dump_root, device_name, debug_node_name):
  """Get the file path of the dump file for a debug node.

  Args:
    dump_root: (str) Root dump directory.
    device_name: (str) Name of the device that the debug node resides on.
    debug_node_name: (str) Name of the debug node, e.g.,
      cross_entropy/Log:0:DebugIdentity.

  Returns:
    (str) Full path of the dump file.
  """

  dump_root = os.path.join(
      dump_root, debug_data.device_name_to_device_path(device_name))
  if "/" in debug_node_name:
    dump_dir = os.path.join(dump_root, os.path.dirname(debug_node_name))
    dump_file_name = re.sub(":", "_", os.path.basename(debug_node_name))
  else:
    dump_dir = dump_root
    dump_file_name = re.sub(":", "_", debug_node_name)

  now_microsec = int(round(time.time() * 1000 * 1000))
  dump_file_name += "_%d" % now_microsec

  return os.path.join(dump_dir, dump_file_name)


class EventListenerTestStreamHandler(
    grpc_debug_server.EventListenerBaseStreamHandler):
  """Implementation of EventListenerBaseStreamHandler that dumps to file."""

  def __init__(self, dump_dir, event_listener_servicer):
    super(EventListenerTestStreamHandler, self).__init__()
    self._dump_dir = dump_dir
    self._event_listener_servicer = event_listener_servicer
    if self._dump_dir:
      self._try_makedirs(self._dump_dir)

    self._grpc_path = None
    self._cached_graph_defs = []
    self._cached_graph_def_device_names = []
    self._cached_graph_def_wall_times = []

  def on_core_metadata_event(self, event):
    self._event_listener_servicer.toggle_watch()

    core_metadata = json.loads(event.log_message.message)

    if not self._grpc_path:
      grpc_path = core_metadata["grpc_path"]
      if grpc_path:
        if grpc_path.startswith("/"):
          grpc_path = grpc_path[1:]
      if self._dump_dir:
        self._dump_dir = os.path.join(self._dump_dir, grpc_path)

        # Write cached graph defs to filesystem.
        for graph_def, device_name, wall_time in zip(
            self._cached_graph_defs,
            self._cached_graph_def_device_names,
            self._cached_graph_def_wall_times):
          self._write_graph_def(graph_def, device_name, wall_time)

    if self._dump_dir:
      self._write_core_metadata_event(event)
    else:
      self._event_listener_servicer.core_metadata_json_strings.append(
          event.log_message.message)

  def on_graph_def(self, graph_def, device_name, wall_time):
    """Implementation of the tensor value-carrying Event proto callback.

    Args:
      graph_def: A GraphDef object.
      device_name: Name of the device on which the graph was created.
      wall_time: An epoch timestamp (in microseconds) for the graph.
    """
    if self._dump_dir:
      if self._grpc_path:
        self._write_graph_def(graph_def, device_name, wall_time)
      else:
        self._cached_graph_defs.append(graph_def)
        self._cached_graph_def_device_names.append(device_name)
        self._cached_graph_def_wall_times.append(wall_time)
    else:
      self._event_listener_servicer.partition_graph_defs.append(graph_def)

  def on_value_event(self, event):
    """Implementation of the tensor value-carrying Event proto callback.

    Writes the Event proto to the file system for testing. The path written to
    follows the same pattern as the file:// debug URLs of tfdbg, i.e., the
    name scope of the op becomes the directory structure under the dump root
    directory.

    Args:
      event: The Event proto carrying a tensor value.

    Returns:
      If the debug node belongs to the set of currently activated breakpoints,
      a `EventReply` proto will be returned.
    """
    if self._dump_dir:
      self._write_value_event(event)
    else:
      value = event.summary.value[0]
      tensor_value = debug_data.load_tensor_from_event(event)
      self._event_listener_servicer.debug_tensor_values[value.node_name].append(
          tensor_value)

      items = event.summary.value[0].node_name.split(":")
      node_name = items[0]
      output_slot = int(items[1])
      debug_op = items[2]
      if ((node_name, output_slot, debug_op) in
          self._event_listener_servicer.breakpoints):
        return debug_service_pb2.EventReply()

  def _try_makedirs(self, dir_path):
    if not os.path.isdir(dir_path):
      try:
        os.makedirs(dir_path)
      except OSError as error:
        if error.errno != errno.EEXIST:
          raise

  def _write_core_metadata_event(self, event):
    core_metadata_path = os.path.join(
        self._dump_dir,
        debug_data.METADATA_FILE_PREFIX + debug_data.CORE_METADATA_TAG +
        "_%d" % event.wall_time)
    self._try_makedirs(self._dump_dir)
    with open(core_metadata_path, "wb") as f:
      f.write(event.SerializeToString())

  def _write_graph_def(self, graph_def, device_name, wall_time):
    encoded_graph_def = graph_def.SerializeToString()
    graph_hash = int(hashlib.md5(encoded_graph_def).hexdigest(), 16)
    event = event_pb2.Event(graph_def=encoded_graph_def, wall_time=wall_time)
    graph_file_path = os.path.join(
        self._dump_dir,
        debug_data.device_name_to_device_path(device_name),
        debug_data.METADATA_FILE_PREFIX + debug_data.GRAPH_FILE_TAG +
        debug_data.HASH_TAG + "%d_%d" % (graph_hash, wall_time))
    self._try_makedirs(os.path.dirname(graph_file_path))
    with open(graph_file_path, "wb") as f:
      f.write(event.SerializeToString())

  def _write_value_event(self, event):
    value = event.summary.value[0]

    # Obtain the device name from the metadata.
    summary_metadata = event.summary.value[0].metadata
    if not summary_metadata.plugin_data:
      raise ValueError("The value lacks plugin data.")
    try:
      content = json.loads(compat.as_text(summary_metadata.plugin_data.content))
    except ValueError as err:
      raise ValueError("Could not parse content into JSON: %r, %r" % (content,
                                                                      err))
    device_name = content["device"]

    dump_full_path = _get_dump_file_path(
        self._dump_dir, device_name, value.node_name)
    self._try_makedirs(os.path.dirname(dump_full_path))
    with open(dump_full_path, "wb") as f:
      f.write(event.SerializeToString())


class EventListenerTestServicer(grpc_debug_server.EventListenerBaseServicer):
  """An implementation of EventListenerBaseServicer for testing."""

  def __init__(self, server_port, dump_dir, toggle_watch_on_core_metadata=None):
    """Constructor of EventListenerTestServicer.

    Args:
      server_port: (int) The server port number.
      dump_dir: (str) The root directory to which the data files will be
        dumped. If empty or None, the received debug data will not be dumped
        to the file system: they will be stored in memory instead.
      toggle_watch_on_core_metadata: A list of
        (node_name, output_slot, debug_op) tuples to toggle the
        watchpoint status during the on_core_metadata calls (optional).
    """
    self.core_metadata_json_strings = []
    self.partition_graph_defs = []
    self.debug_tensor_values = collections.defaultdict(list)
    self._initialize_toggle_watch_state(toggle_watch_on_core_metadata)

    grpc_debug_server.EventListenerBaseServicer.__init__(
        self, server_port,
        functools.partial(EventListenerTestStreamHandler, dump_dir, self))

  def _initialize_toggle_watch_state(self, toggle_watches):
    self._toggle_watches = toggle_watches
    self._toggle_watch_state = dict()
    if self._toggle_watches:
      for watch_key in self._toggle_watches:
        self._toggle_watch_state[watch_key] = False

  def toggle_watch(self):
    for watch_key in self._toggle_watch_state:
      node_name, output_slot, debug_op = watch_key
      if self._toggle_watch_state[watch_key]:
        self.request_unwatch(node_name, output_slot, debug_op)
      else:
        self.request_watch(node_name, output_slot, debug_op)
      self._toggle_watch_state[watch_key] = (
          not self._toggle_watch_state[watch_key])

  def clear_data(self):
    self.core_metadata_json_strings = []
    self.partition_graph_defs = []
    self.debug_tensor_values = collections.defaultdict(list)


def start_server_on_separate_thread(dump_to_filesystem=True,
                                    server_start_delay_sec=0.0,
                                    poll_server=False,
                                    blocking=True,
                                    toggle_watch_on_core_metadata=None):
  """Create a test gRPC debug server and run on a separate thread.

  Args:
    dump_to_filesystem: (bool) whether the debug server will dump debug data
      to the filesystem.
    server_start_delay_sec: (float) amount of time (in sec) to delay the server
      start up for.
    poll_server: (bool) whether the server will be polled till success on
      startup.
    blocking: (bool) whether the server should be started in a blocking mode.
    toggle_watch_on_core_metadata: A list of
        (node_name, output_slot, debug_op) tuples to toggle the
        watchpoint status during the on_core_metadata calls (optional).

  Returns:
    server_port: (int) Port on which the server runs.
    debug_server_url: (str) grpc:// URL to the server.
    server_dump_dir: (str) The debug server's dump directory.
    server_thread: The server Thread object.
    server: The `EventListenerTestServicer` object.

  Raises:
    ValueError: If polling the server process for ready state is not successful
      within maximum polling count.
  """
  server_port = portpicker.pick_unused_port()
  debug_server_url = "grpc://localhost:%d" % server_port

  server_dump_dir = tempfile.mkdtemp() if dump_to_filesystem else None
  server = EventListenerTestServicer(
      server_port=server_port,
      dump_dir=server_dump_dir,
      toggle_watch_on_core_metadata=toggle_watch_on_core_metadata)

  def delay_then_run_server():
    time.sleep(server_start_delay_sec)
    server.run_server(blocking=blocking)

  server_thread = threading.Thread(target=delay_then_run_server)
  server_thread.start()

  if poll_server:
    if not _poll_server_till_success(
        50,
        0.2,
        debug_server_url,
        server_dump_dir,
        server,
        gpu_memory_fraction=0.1):
      raise ValueError(
          "Failed to start test gRPC debug server at port %d" % server_port)
    server.clear_data()
  return server_port, debug_server_url, server_dump_dir, server_thread, server


def _poll_server_till_success(max_attempts,
                              sleep_per_poll_sec,
                              debug_server_url,
                              dump_dir,
                              server,
                              gpu_memory_fraction=1.0):
  """Poll server until success or exceeding max polling count.

  Args:
    max_attempts: (int) How many times to poll at maximum
    sleep_per_poll_sec: (float) How many seconds to sleep for after each
      unsuccessful poll.
    debug_server_url: (str) gRPC URL to the debug server.
    dump_dir: (str) Dump directory to look for files in. If None, will directly
      check data from the server object.
    server: The server object.
    gpu_memory_fraction: (float) Fraction of GPU memory to be
      allocated for the Session used in server polling.

  Returns:
    (bool) Whether the polling succeeded within max_polls attempts.
  """
  poll_count = 0

  config = config_pb2.ConfigProto(gpu_options=config_pb2.GPUOptions(
      per_process_gpu_memory_fraction=gpu_memory_fraction))
  with session.Session(config=config) as sess:
    for poll_count in range(max_attempts):
      server.clear_data()
      print("Polling: poll_count = %d" % poll_count)

      x_init_name = "x_init_%d" % poll_count
      x_init = constant_op.constant([42.0], shape=[1], name=x_init_name)
      x = variables.Variable(x_init, name=x_init_name)

      run_options = config_pb2.RunOptions()
      debug_utils.add_debug_tensor_watch(
          run_options, x_init_name, 0, debug_urls=[debug_server_url])
      try:
        sess.run(x.initializer, options=run_options)
      except errors.FailedPreconditionError:
        pass

      if dump_dir:
        if os.path.isdir(
            dump_dir) and debug_data.DebugDumpDir(dump_dir).size > 0:
          shutil.rmtree(dump_dir)
          print("Poll succeeded.")
          return True
        else:
          print("Poll failed. Sleeping for %f s" % sleep_per_poll_sec)
          time.sleep(sleep_per_poll_sec)
      else:
        if server.debug_tensor_values:
          print("Poll succeeded.")
          return True
        else:
          print("Poll failed. Sleeping for %f s" % sleep_per_poll_sec)
          time.sleep(sleep_per_poll_sec)

    return False
