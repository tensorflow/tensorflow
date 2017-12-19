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
"""Communicating tracebacks and source code with debug server."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import socket

import grpc

from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import debug_pb2
from tensorflow.python.debug.lib import debug_service_pb2_grpc
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.platform import gfile
from tensorflow.python.profiler import tfprof_logger


def _load_debugged_source_file(file_path, source_file_proto):
  file_stat = gfile.Stat(file_path)
  source_file_proto.host = socket.gethostname()
  source_file_proto.file_path = file_path
  source_file_proto.last_modified = file_stat.mtime_nsec
  source_file_proto.bytes = file_stat.length
  try:
    with gfile.Open(file_path, "r") as f:
      source_lines = f.readlines()
      for line in source_lines:
        source_file_proto.lines.append(line.strip())
  except IOError:
    pass


def _string_to_id(string, string_to_id):
  if string not in string_to_id:
    string_to_id[string] = len(string_to_id)
  return string_to_id[string]


def _format_origin_stack(origin_stack, call_traceback_proto):
  """Format a traceback stack for a `CallTraceback` proto.

  Args:
    origin_stack: The stack list as returned by `traceback.extract_stack()`.
    call_traceback_proto: A `CallTraceback` proto whose fields are to be
      populated.
  """
  string_to_id = dict()
  string_to_id[None] = 0
  for frame in origin_stack:
    file_path, lineno, func_name, line_text = frame
    call_traceback_proto.origin_stack.traces.add(
        file_id=_string_to_id(file_path, string_to_id),
        lineno=lineno,
        function_id=_string_to_id(func_name, string_to_id),
        line_id=_string_to_id(line_text, string_to_id))

  id_to_string = call_traceback_proto.origin_id_to_string
  for key, value in string_to_id.items():
    id_to_string[value] = key if key is not None else ""


def _source_file_paths_outside_tensorflow_py_library(code_defs, id_to_string):
  """Extract source file paths outside TensorFlow Python library.

  Args:
    code_defs: An iterable of `CodeDef` protos, i.e., an iterable of stack
      traces.
    id_to_string: A proto map from integer ids to strings.

  Returns:
    An iterable of source file paths outside the TensorFlow Python library.
  """
  file_ids = set()
  for code_def in code_defs:
    for trace in code_def.traces:
      file_ids.add(trace.file_id)
  non_tf_files = (id_to_string[file_id] for file_id in file_ids)
  non_tf_files = (
      f for f in non_tf_files
      if not source_utils.guess_is_tensorflow_py_library(f) and gfile.Exists(f))
  return non_tf_files


def _send_call_tracebacks(destinations,
                          origin_stack,
                          is_eager_execution=False,
                          call_key=None,
                          graph=None,
                          send_source=True):
  """Send the tracebacks of a TensorFlow execution call.

  To gRPC debug server(s). This applies to graph execution (`tf.Session.run()`)
  calls and eager execution calls.

  If `send_source`, also sends the underlying source files outside the
  TensorFlow library.

  Args:
    destinations: gRPC destination addresses, a `str` or a `list` of `str`s,
      e.g., "localhost:4242". If a `list`, gRPC requests containing the same
      `CallTraceback` proto payload will be sent to all the destinations.
    origin_stack: The traceback stack for the origin of the execution call. For
      graph execution, this is the traceback of the `tf.Session.run()`
      invocation. For eager execution, this is the traceback of the Python
      line that executes the eager opertion.
    is_eager_execution: (`bool`) whether an eager execution call (i.e., not a
      `tf.Session.run` or derived methods) is being sent.
    call_key: The key of the execution call, as a string. For graph execution,
      this is a string describing the feeds, fetches (and targets) names of the
      `tf.Session.run` call. For eager execution, this is ignored.
    graph: A Python `tf.Graph` object (i.e., *not* a `tf.GraphDef`), which
      contains op tracebacks, if applicable.
    send_source: Whether the source files involved in the op tracebacks but
      outside the TensorFlow library are to be sent.
  """
  if not isinstance(destinations, list):
    destinations = [destinations]

  call_type = (debug_service_pb2.CallTraceback.EAGER_EXECUTION
               if is_eager_execution
               else debug_service_pb2.CallTraceback.GRAPH_EXECUTION)
  graph_traceback = tfprof_logger.merge_default_with_oplog(
      graph, add_trainable_var=False) if graph else None
  call_traceback = debug_service_pb2.CallTraceback(
      call_type=call_type, call_key=call_key, graph_traceback=graph_traceback,
      graph_version=graph.version if graph else None)

  _format_origin_stack(origin_stack, call_traceback)

  if send_source:
    source_file_paths = set()
    source_file_paths.update(_source_file_paths_outside_tensorflow_py_library(
        (log_entry.code_def for log_entry
         in call_traceback.graph_traceback.log_entries),
        call_traceback.graph_traceback.id_to_string))
    source_file_paths.update(_source_file_paths_outside_tensorflow_py_library(
        [call_traceback.origin_stack], call_traceback.origin_id_to_string))

    debugged_source_files = debug_pb2.DebuggedSourceFiles()
    for file_path in source_file_paths:
      _load_debugged_source_file(
          file_path, debugged_source_files.source_files.add())

  for destination in destinations:
    channel = grpc.insecure_channel(destination)
    stub = debug_service_pb2_grpc.EventListenerStub(channel)
    stub.SendTracebacks(call_traceback)
    if send_source:
      stub.SendSourceFiles(debugged_source_files)


def send_graph_tracebacks(destinations,
                          run_key,
                          origin_stack,
                          graph,
                          send_source=True):
  """Send the tracebacks of a graph execution call to debug server(s).

  Args:
    destinations: gRPC destination addresses, a `str` or a `list` of `str`s,
      e.g., "localhost:4242". If a `list`, gRPC requests containing the same
      `CallTraceback` proto payload will be sent to all the destinations.
    run_key: A string describing the feeds, fetches (and targets) names of the
      `tf.Session.run` call.
    origin_stack: The traceback of the `tf.Session.run()` invocation.
    graph: A Python `tf.Graph` object (i.e., *not* a `tf.GraphDef`), which
      contains op tracebacks.
    send_source: Whether the source files involved in the op tracebacks but
      outside the TensorFlow library are to be sent.
  """
  _send_call_tracebacks(
      destinations, origin_stack, is_eager_execution=False, call_key=run_key,
      graph=graph, send_source=send_source)


def send_eager_tracebacks(destinations,
                          origin_stack,
                          send_source=True):
  """Send the tracebacks of an eager execution call to debug server(s).

  Args:
    destinations: gRPC destination addresses, a `str` or a `list` of `str`s,
      e.g., "localhost:4242". If a `list`, gRPC requests containing the same
    origin_stack: The traceback of the eager operation invocation.
    send_source: Whether the source files involved in the op tracebacks but
      outside the TensorFlow library are to be sent.
  """
  _send_call_tracebacks(
      destinations, origin_stack, is_eager_execution=True,
      send_source=send_source)
