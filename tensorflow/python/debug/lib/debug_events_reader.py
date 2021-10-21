# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Reader class for tfdbg v2 debug events."""

import collections
import os
import threading

import six

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat


DebugEventWithOffset = collections.namedtuple(
    "DebugEventWithOffset", "debug_event offset")


class DebugEventsReader(object):
  """Reader class for a tfdbg v2 DebugEvents directory."""

  # Number of digests after which a read lock is released and re-acquired during
  # serial reading of digests for SourceFiles, Execution, and
  # GraphExecutionTrace. This allows us to avoid releasing and re-acquiring the
  # lock too often (i.e., after each digest) and to minimize performance
  # penalty.
  _READER_RELEASE_PER = 100

  _METADATA_SUFFIX = ".metadata"
  _SOURCE_FILE_SUFFIX = ".source_files"
  _STACK_FRAMES_SUFFIX = ".stack_frames"
  _GRAPHS_SUFFIX = ".graphs"
  _EXECUTION_SUFFIX = ".execution"
  _GRAPH_EXECUTION_TRACES_SUFFIX = ".graph_execution_traces"

  def __init__(self, dump_root):
    if not file_io.is_directory(dump_root):
      raise ValueError("Specified dump_root is not a directory: %s" % dump_root)
    self._dump_root = dump_root
    self._metadata_paths = self._load_metadata_files()

    prefixes = [
        metadata_path[:-len(self._METADATA_SUFFIX)]
        for metadata_path in self._metadata_paths
    ]
    prefix = prefixes[0]  # This is the prefix of the main file set.
    self._source_files_path = compat.as_bytes(prefix + self._SOURCE_FILE_SUFFIX)
    self._stack_frames_path = compat.as_bytes(prefix +
                                              self._STACK_FRAMES_SUFFIX)
    self._graphs_path = compat.as_bytes(prefix + self._GRAPHS_SUFFIX)
    self._execution_path = compat.as_bytes(prefix + self._EXECUTION_SUFFIX)
    # There can be multiple .graph_execution_trace files each belonging
    # to a file set generated on an individual host, in the case of
    # a distributed TensorFlow job.
    # This is different from the other debug event files in the file set.
    self._graph_execution_traces_paths = [
        compat.as_bytes(prefix + self._GRAPH_EXECUTION_TRACES_SUFFIX)
        for prefix in prefixes
    ]
    self._readers = dict()  # A map from file path to reader.
    # A map from file path to current reading offset.
    self._reader_offsets = dict()
    # Lock for reader creation.
    self._readers_lock = threading.Lock()
    # Locks for read operation on individual readers.
    self._reader_read_locks = dict()

    self._offsets = dict()

  def _load_metadata_files(self):
    """Load and parse metadata files in the dump root.

    Check that all metadata files have a common tfdbg_run_id, and raise
    a ValueError if their tfdbg_run_ids differ.

    Returns:
      A list of metadata file paths in ascending order of their starting
        wall_time timestamp.
    """

    metadata_paths = file_io.get_matching_files(
        os.path.join(self._dump_root, "*%s" % self._METADATA_SUFFIX))
    if not metadata_paths:
      raise ValueError("Cannot find any tfdbg metadata file in directory: %s" %
                       self._dump_root)
    wall_times = []
    run_ids = []
    tensorflow_versions = []
    file_versions = []
    for metadata_path in metadata_paths:
      reader = tf_record.tf_record_random_reader(metadata_path)
      try:
        record = reader.read(0)[0]
        debug_event = debug_event_pb2.DebugEvent.FromString(record)
        wall_times.append(debug_event.wall_time)
        run_ids.append(debug_event.debug_metadata.tfdbg_run_id)
        tensorflow_versions.append(
            debug_event.debug_metadata.tensorflow_version)
        file_versions.append(debug_event.debug_metadata.file_version)
      finally:
        reader.close()
    self._starting_wall_time = wall_times[0]
    self._tfdbg_run_id = run_ids[0]
    self._tensorflow_version = tensorflow_versions[0]
    self._file_version = file_versions[0]
    if len(metadata_paths) == 1:
      # Fast path for a common case (only one DebugEvent file set.)
      return metadata_paths

    num_no_id = len([run_id for run_id in run_ids if not run_id])
    if num_no_id:
      paths_without_run_id = [
          metadata_path
          for metadata_path, run_id in zip(metadata_paths, run_ids)
          if not run_id
      ]
      raise ValueError(
          "Found %d tfdbg metadata files and %d of them do not "
          "have tfdbg run ids. The metadata files without run ids are: %s" %
          (len(run_ids), num_no_id, paths_without_run_id))
    elif len(set(run_ids)) != 1:
      raise ValueError(
          "Unexpected: Found multiple (%d) tfdbg2 runs in directory %s" %
          (len(set(run_ids)), self._dump_root))
    # Return the metadata files in ascending order of their timestamps.
    paths_and_timestamps = sorted(
        zip(metadata_paths, wall_times), key=lambda t: t[1])
    self._starting_wall_time = paths_and_timestamps[0][1]
    return [path[0] for path in paths_and_timestamps]

  def starting_wall_time(self):
    """Get the starting timestamp of the instrumented TensorFlow program.

    When there are multiple hosts (i.e., multiple tfdbg file sets), the earliest
    timestamp among the file sets is returned. It is assumed to be the job that
    starts first (e.g., the coordinator).

    Returns:
      Starting timestamp in seconds since the epoch, as a float.
    """
    return self._starting_wall_time

  def tfdbg_run_id(self):
    """Get the run ID of the instrumented TensorFlow program."""
    return self._tfdbg_run_id

  def tensorflow_version(self):
    """Get the version string of TensorFlow that the debugged program ran on."""
    return self._tensorflow_version

  def tfdbg_file_version(self):
    """Get the tfdbg file format version."""
    return self._file_version

  def __enter__(self):
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    del exception_type, exception_value, traceback  # Unused
    self.close()

  def _generic_iterator(self, file_path):
    """A helper method that makes an iterator given a debug-events file path.

    Repeated calls to this method create iterators that remember the last
    successful reading position (offset) for each given `file_path`. So the
    iterators are meant for incremental reading of the file.

    Args:
      file_path: Path to the file to create the iterator for.

    Yields:
      A tuple of (offset, debug_event_proto) on each `next()` call.
    """
    yield_count = 0
    reader = self._get_reader(file_path)
    read_lock = self._reader_read_locks[file_path]
    read_lock.acquire()
    try:
      while True:
        current_offset = self._reader_offsets[file_path]
        try:
          record, self._reader_offsets[file_path] = reader.read(current_offset)
        except (errors.DataLossError, IndexError):
          # We ignore partial read exceptions, because a record may be
          # truncated. The PyRandomRecordReader throws an `IndexError` when
          # offset goes out of bound.
          break
        yield DebugEventWithOffset(
            debug_event=debug_event_pb2.DebugEvent.FromString(record),
            offset=current_offset)
        yield_count += 1
        # The read lock must be periodically released to allow for concurrent
        # random reads. But we do so at a number of reads, instead of after
        # every single read, in order to minimize the performance penalty.
        if yield_count % self._READER_RELEASE_PER == 0:
          read_lock.release()
          read_lock.acquire()
    finally:
      read_lock.release()

  def _get_reader(self, file_path):
    """Get a random-access reader for TFRecords file at file_path."""
    file_path = compat.as_bytes(file_path)
    # The following code uses the double-checked locking pattern to optimize
    # the common case (where the reader is already initialized).
    if file_path not in self._readers:  # 1st check, without lock.
      with self._readers_lock:
        if file_path not in self._readers:  # 2nd check, with lock.
          self._readers[file_path] = tf_record.tf_record_random_reader(
              file_path)
          self._reader_read_locks[file_path] = threading.Lock()
          self._reader_offsets[file_path] = 0
    return self._readers[file_path]

  def source_files_iterator(self):
    return self._generic_iterator(self._source_files_path)

  def stack_frames_iterator(self):
    return self._generic_iterator(self._stack_frames_path)

  def graphs_iterator(self):
    return self._generic_iterator(self._graphs_path)

  def read_source_files_event(self, offset):
    """Read a DebugEvent proto at given offset from the .source_files file."""
    with self._reader_read_locks[self._source_files_path]:
      proto_string = self._get_reader(self._source_files_path).read(offset)[0]
    return debug_event_pb2.DebugEvent.FromString(proto_string)

  def read_graphs_event(self, offset):
    """Read a DebugEvent proto at a given offset from the .graphs file.

    Args:
      offset: Offset to read the DebugEvent proto from.

    Returns:
      A DebugEventProto.

    Raises:
      `errors.DataLossError` if offset is at a wrong location.
      `IndexError` if offset is out of range of the file.
    """
    return debug_event_pb2.DebugEvent.FromString(
        self._get_reader(self._graphs_path).read(offset)[0])

  def execution_iterator(self):
    return self._generic_iterator(self._execution_path)

  def read_execution_event(self, offset):
    """Read a DebugEvent proto at a given offset from the .execution file.

    Args:
      offset: Offset to read the DebugEvent proto from.

    Returns:
      A DebugEventProto.

    Raises:
      `errors.DataLossError` if offset is at a wrong location.
      `IndexError` if offset is out of range of the file.
    """
    with self._reader_read_locks[self._execution_path]:
      proto_string = self._get_reader(self._execution_path).read(offset)[0]
    return debug_event_pb2.DebugEvent.FromString(proto_string)

  def graph_execution_traces_iterators(self):
    return [
        self._generic_iterator(path)
        for path in self._graph_execution_traces_paths
    ]

  def read_graph_execution_traces_event(self, locator):
    """Read DebugEvent at given offset from given .graph_execution_traces file.

    Args:
      locator: A (file_index, offset) tuple that locates the DebugEvent
        containing the graph execution trace.

    Returns:
      A DebugEventProto.

    Raises:
      `errors.DataLossError` if offset is at a wrong location.
      `IndexError` if offset is out of range of the file.
    """
    file_index, offset = locator
    graph_execution_traces_path = self._graph_execution_traces_paths[file_index]
    with self._reader_read_locks[graph_execution_traces_path]:
      proto_string = self._get_reader(graph_execution_traces_path).read(
          offset)[0]
    return debug_event_pb2.DebugEvent.FromString(proto_string)

  def close(self):
    with self._readers_lock:
      file_paths = list(self._readers.keys())
      for file_path in file_paths:
        self._readers[file_path].close()
        del self._readers[file_path]


class BaseDigest(object):
  """Base class for digest.

  Properties:
    wall_time: A timestamp for the digest as a `float` (unit: s).
    locator: A datum that allows tracng the digest to its original
      location. It can be either of the two:
       1. Bytes offset from the beginning of the file as a single integer,
          for the case of all digests of the same kind coming from the same
          file.
       2. A tuple of a file index and a byte offset. This applies to case
          in which the same type of debugger data may come from multple files,
          e.g., graph execution traces.
  """

  def __init__(self, wall_time, locator):
    self._wall_time = wall_time
    self._locator = locator

  @property
  def wall_time(self):
    return self._wall_time

  @property
  def locator(self):
    return self._locator

  def to_json(self):
    return {"wall_time": self.wall_time}


class ExecutionDigest(BaseDigest):
  """Light-weight digest summarizing top-level execution event.

  Use `DebugDataReader.read_execution(execution_digest)` to load the more
  detailed data object concerning the execution event (`Execution`).

  Properties:
    op_type: Type name of the executed op. In the case of the eager execution of
      an individual op, it is the name of the op (e.g., "MatMul").
      In the case of the execution of a tf.function (FuncGraph), this is the
      internally-generated name of the function (e.g.,
      "__inference_my_func_123").
    output_tensor_device_ids: IDs of the devices on which the output tensors of
      the execution reside. For no-output execution, this is `None`.
  """

  def __init__(self,
               wall_time,
               locator,
               op_type,
               output_tensor_device_ids=None):
    super(ExecutionDigest, self).__init__(wall_time, locator)
    self._op_type = op_type
    self._output_tensor_device_ids = _tuple_or_none(output_tensor_device_ids)

  @property
  def op_type(self):
    return self._op_type

  @property
  def output_tensor_device_ids(self):
    return self._output_tensor_device_ids

  def to_json(self):
    output = super(ExecutionDigest, self).to_json()
    output.update({
        "op_type": self.op_type,
        "output_tensor_device_ids": self.output_tensor_device_ids,
    })
    return output


def _tuple_or_none(data):
  return tuple(data) if data else None


class Execution(ExecutionDigest):
  """Detailed data relating to a top-level execution event.

  The execution is of an individual op or a tf.function, which may have any
  number of output tensors.

  Properties (beyond the base class `ExecutionDigest`):
    host_name: Name of the host on which the execution happened.
    stack_frame_ids: Reference IDs for stack frames, ordered from bottommost to
      topmost. Use `DebugDataReader.read_execution_stack_trace()` to load the
      detailed stack frames (filepath, lineno and function name).
    tensor_debug_mode: TensorDebugMode enum value, as an `int`.
    graph_id: ID of the executed FuncGraph (applicable only the execution of a
      tf.function). `None` for the eager execution of an individual op.
    input_tensor_ids: IDs of the input (eager) tensor(s) for this execution, if
      any. If the eager execution has no input tensor, this is `None`. Else,
      this is a `tuple` of `int`s.
    output_tensor_ids: IDs of the output (eager) tensor(s) from this execution,
      if any. If the eager execution produces no output tensor, this is `None`.
      Else, this is a `tuple` of `int`s.
    debug_tensor_values: Values of the debug tensor(s), applicable only to
      non-FULL_TENSOR tensor debug mode. A tuple of list of numbers. Each
      element of the tuple corresponds to an output tensor of the execution.
      See documentation of the various TensorDebugModes for the semantics of the
      numbers. If the eager execution produces no output tensor, this is
      `None`. Else, this is a `tuple` of `list`s.
  """

  def __init__(self,
               execution_digest,
               host_name,
               stack_frame_ids,
               tensor_debug_mode,
               graph_id=None,
               input_tensor_ids=None,
               output_tensor_ids=None,
               debug_tensor_values=None):
    super(Execution, self).__init__(
        execution_digest.wall_time,
        execution_digest.locator,
        execution_digest.op_type,
        output_tensor_device_ids=execution_digest.output_tensor_device_ids)
    self._host_name = host_name
    self._stack_frame_ids = tuple(stack_frame_ids)
    self._tensor_debug_mode = tensor_debug_mode
    self._graph_id = graph_id
    self._input_tensor_ids = _tuple_or_none(input_tensor_ids)
    self._output_tensor_ids = _tuple_or_none(output_tensor_ids)
    self._debug_tensor_values = _tuple_or_none(debug_tensor_values)

  @property
  def host_name(self):
    return self._host_name

  @property
  def stack_frame_ids(self):
    return self._stack_frame_ids

  @property
  def tensor_debug_mode(self):
    return self._tensor_debug_mode

  @property
  def graph_id(self):
    return self._graph_id

  @property
  def input_tensor_ids(self):
    return self._input_tensor_ids

  @property
  def num_outputs(self):
    return len(self._output_tensor_ids) if self._output_tensor_ids else 0

  @property
  def output_tensor_ids(self):
    return self._output_tensor_ids

  @property
  def debug_tensor_values(self):
    return self._debug_tensor_values

  def to_json(self):
    output = super(Execution, self).to_json()
    output.update({
        "host_name": self.host_name,
        "stack_frame_ids": self.stack_frame_ids,
        "tensor_debug_mode": self.tensor_debug_mode,
        "graph_id": self.graph_id,
        "input_tensor_ids": self.input_tensor_ids,
        "output_tensor_ids": self.output_tensor_ids,
        "debug_tensor_values": self.debug_tensor_values,
    })
    return output


class DebuggedGraph(object):
  """Data object representing debugging information about a tf.Graph.

  Includes `FuncGraph`s.

  Properties:
    name: Name of the graph (if any). May be `None` for non-function graphs.
    graph_id: Debugger-generated ID for the graph.
    inner_graph_ids: A list of the debugger-generated IDs for the graphs
      enclosed by this graph.
    outer_graph_id: If this graph is nested within an outer graph, ID of the
      outer graph. If this is an outermost graph, `None`.
  """

  def __init__(self,
               name,
               graph_id,
               outer_graph_id=None):
    self._name = name
    self._graph_id = graph_id
    self._outer_graph_id = outer_graph_id
    self._inner_graph_ids = []
    # A dictionary from op name to GraphOpCreationDigest.
    self._op_by_name = dict()
    # A dictionary mapping op to immediate downstream consumers.
    self._op_consumers = collections.defaultdict(list)

  def add_inner_graph_id(self, inner_graph_id):
    """Add the debugger-generated ID of a graph nested within this graph.

    Args:
      inner_graph_id: The debugger-generated ID of the nested inner graph.
    """
    assert isinstance(inner_graph_id, six.string_types)
    self._inner_graph_ids.append(inner_graph_id)

  def add_op(self, graph_op_creation_digest):
    """Add an op creation data object.

    Args:
      graph_op_creation_digest: A GraphOpCreationDigest data object describing
        the creation of an op inside this graph.
    """
    if graph_op_creation_digest.op_name in self._op_by_name:
      raise ValueError(
          "Duplicate op name: %s (op type: %s)" %
          (graph_op_creation_digest.op_name, graph_op_creation_digest.op_type))
    self._op_by_name[
        graph_op_creation_digest.op_name] = graph_op_creation_digest

  def add_op_consumer(self, src_op_name, src_slot, dst_op_name, dst_slot):
    """Add a consuming op for this op.

    Args:
      src_op_name: Name of the op of which the output tensor is being consumed.
      src_slot: 0-based output slot of the op being consumed.
      dst_op_name: Name of the consuming op (e.g., "Conv2D_3/BiasAdd")
      dst_slot: 0-based input slot of the consuming op that receives the tensor
        from this op.
    """
    self._op_consumers[src_op_name].append((src_slot, dst_op_name, dst_slot))

  @property
  def name(self):
    return self._name

  @property
  def graph_id(self):
    return self._graph_id

  @property
  def outer_graph_id(self):
    return self._outer_graph_id

  @property
  def inner_graph_ids(self):
    return self._inner_graph_ids

  def get_tensor_id(self, op_name, output_slot):
    """Get the ID of a symbolic tensor in this graph."""
    return self._op_by_name[op_name].output_tensor_ids[output_slot]

  def get_op_creation_digest(self, op_name):
    """Get the GraphOpCreationDigest for a op in the graph."""
    return self._op_by_name[op_name]

  def get_op_consumers(self, src_op_name):
    """Get all the downstream consumers of this op.

    Only data (non-control) edges are tracked.

    Args:
      src_op_name: Name of the op providing the tensor being consumed.

    Returns:
      A list of (src_slot, dst_op_name, dst_slot) tuples. In each item of
      the list:
        src_slot: 0-based output slot of the op of which the output tensor
          is being consumed.
        dst_op_name: Name of the consuming op (e.g., "Conv2D_3/BiasAdd")
        dst_slot: 0-based input slot of the consuming op that receives
          the tensor from this op.
    """
    return self._op_consumers[src_op_name]

  def to_json(self):
    return {
        "name": self.name,
        "graph_id": self.graph_id,
        "outer_graph_id": self._outer_graph_id,
        "inner_graph_ids": self._inner_graph_ids,
    }


class DebuggedDevice(object):
  """Debugger data regarding a device involved in the debugged program.

  Properties:
    device_name: Name of the device, as a str.
    device_id: An integer ID for the device, unique for each device within
      the scope of the debugged TensorFlow program.
  """

  def __init__(self,
               device_name,
               device_id):
    self._device_name = device_name
    self._device_id = device_id

  @property
  def device_name(self):
    return self._device_name

  @property
  def device_id(self):
    return self._device_id

  def to_json(self):
    return {
        "device_name": self._device_name,
        "device_id": self._device_id,
    }


class GraphOpCreationDigest(BaseDigest):
  """Data object describing the creation of an op inside a graph.

  For size efficiency, this digest object does not contain any stack frames or
  any references to them. To obtain the stack frames, use
  `DataReader.read_graph_op_creation_stack_trace()`.

  Properties (beyond the base class):
    graph_id: Debugger-generated ID of the immediately-enclosing graph.
    op_type: Type name of the op (e.g., "MatMul").
    op_name: Name of the op (e.g., "dense_1/MatMul").
    output_tensor_ids: Debugger-generated IDs for the output(s) of the op.
      If the op produces no output tensor, this is `None`. Else, this is a
      `tuple` of `int`s.
    input_names: Names of the input tensors to the op.
    device_name: The name of the device that the op is placed on (if available).
    host_name: Name of the host on which the op is created.
    stack_frame_ids: IDs of the frames of the stack trace at which the op
      is created.
  """

  def __init__(self,
               wall_time,
               locator,
               graph_id,
               op_type,
               op_name,
               output_tensor_ids,
               host_name,
               stack_frame_ids,
               input_names=None,
               device_name=None):
    super(GraphOpCreationDigest, self).__init__(wall_time, locator)
    self._graph_id = graph_id
    self._op_type = op_type
    self._op_name = op_name
    self._output_tensor_ids = _tuple_or_none(output_tensor_ids)
    self._host_name = host_name
    self._stack_frame_ids = stack_frame_ids
    self._input_names = _tuple_or_none(input_names)
    self._device_name = device_name

  @property
  def graph_id(self):
    return self._graph_id

  @property
  def op_type(self):
    return self._op_type

  @property
  def op_name(self):
    return self._op_name

  @property
  def output_tensor_ids(self):
    return self._output_tensor_ids

  @property
  def num_outputs(self):
    return len(self._output_tensor_ids) if self.output_tensor_ids else 0

  @property
  def input_names(self):
    return self._input_names

  @property
  def device_name(self):
    return self._device_name

  @property
  def host_name(self):
    return self._host_name

  @property
  def stack_frame_ids(self):
    return self._stack_frame_ids

  def to_json(self):
    output = super(GraphOpCreationDigest, self).to_json()
    output.update({
        "graph_id": self.graph_id,
        "op_type": self.op_type,
        "op_name": self.op_name,
        "output_tensor_ids": self.output_tensor_ids,
        "host_name": self.host_name,
        "stack_frame_ids": self.stack_frame_ids,
        "input_names": self.input_names,
        "device_name": self.device_name,
    })
    return output


class GraphExecutionTraceDigest(BaseDigest):
  """Light-weight summary of a intra-graph tensor execution event.

  Use `DebugDataReader.read_graph_execution_trace()` on this object to read more
  detailed data (`GraphExecutionTrace`).

  Properties (beyond the base class):
    op_type: Type name of the executed op (e.g., "Conv2D").
    op_name: Name of the op (e.g., "conv_2d_3/Conv2D").
    output_slot: Output slot index of the tensor.
    graph_id: The debugger-generated ID of the innermost (immediately-enclosing)
      graph.
  """

  def __init__(self, wall_time, locator, op_type, op_name, output_slot,
               graph_id):
    super(GraphExecutionTraceDigest, self).__init__(wall_time, locator)
    self._op_type = op_type
    self._op_name = op_name
    self._output_slot = output_slot
    self._graph_id = graph_id

  @property
  def op_type(self):
    return self._op_type

  @property
  def op_name(self):
    return self._op_name

  @property
  def output_slot(self):
    return self._output_slot

  @property
  def graph_id(self):
    return self._graph_id

  def to_json(self):
    output = super(GraphExecutionTraceDigest, self).to_json()
    output.update({
        "op_type": self.op_type,
        "op_name": self.op_name,
        "output_slot": self.output_slot,
        "graph_id": self.graph_id,
    })
    return output


class GraphExecutionTrace(GraphExecutionTraceDigest):
  """Detailed data object describing an intra-graph tensor execution.

  Attributes (in addition to GraphExecutionTraceDigest):
    graph_ids: The debugger-generated IDs of the graphs that enclose the
      executed op (tensor), ordered from the outermost to the innermost.
    graph_id: The debugger-generated ID of the innermost (immediately-enclosing)
      graph.
    tensor_debug_mode: TensorDebugMode enum value.
    debug_tensor_value: Debug tensor values (only for non-FULL_TENSOR
      tensor_debug_mode). A list of numbers. See the documentation of the
      TensorDebugModes for the semantics of the numbers.
    device_name: Device on which the tensor resides (if available)
  """

  def __init__(self,
               graph_execution_trace_digest,
               graph_ids,
               tensor_debug_mode,
               debug_tensor_value=None,
               device_name=None):
    super(GraphExecutionTrace,
          self).__init__(graph_execution_trace_digest.wall_time,
                         graph_execution_trace_digest.locator,
                         graph_execution_trace_digest.op_type,
                         graph_execution_trace_digest.op_name,
                         graph_execution_trace_digest.output_slot,
                         graph_execution_trace_digest.graph_id)
    self._graph_ids = tuple(graph_ids)
    self._tensor_debug_mode = tensor_debug_mode
    self._debug_tensor_value = debug_tensor_value
    self._device_name = device_name

  @property
  def graph_ids(self):
    return self._graph_ids

  @property
  def graph_id(self):
    return self._graph_ids[-1]

  @property
  def tensor_debug_mode(self):
    return self._tensor_debug_mode

  @property
  def debug_tensor_value(self):
    return _tuple_or_none(self._debug_tensor_value)

  @property
  def device_name(self):
    return self._device_name

  def to_json(self):
    output = super(GraphExecutionTrace, self).to_json()
    output.update({
        "graph_ids": self.graph_ids,
        "tensor_debug_mode": self.tensor_debug_mode,
        "debug_tensor_value": self.debug_tensor_value,
        "device_name": self.device_name,
    })
    return output


def _parse_tensor_value(tensor_proto, return_list=False):
  """Helper method for reading a tensor value from a tensor proto.

  The rationale for the distinction between `True` and `False value of
  `return_list` is as follows:
  - `return_list=True` is used for TensorDebugMode values other than
    FULL_TENSOR, e.g., CONCISE_HEALTH, SHAPE and FULL_HEATLH. Under
    those modes, the value is guaranteed (by contract) to be a 1D float64
    tensor.
  - `return_list=False` is used for the FULL_HEALTH TensorDebugMode
    specifically. Instead, we use `numpy.ndarray` to maximally preserve
    the shape, dtype and value information regarding the underlying tensor
    value. Under that mode, we don't use a python list to represent the
    tensor value because that can lead to loss of information (e.g., both
    float16 and float32 dtypes get mapped to Python floats).

  Args:
    tensor_proto: The TensorProto instance from which the tensor value will be
      loaded.
    return_list: Whether the return value will be a nested Python list that
      comes out from `numpy.ndarray.tolist()`.

  Returns:
    If parsing is successful, the tensor value as a `numpy.ndarray` or the
      nested Python list converted from it.
    If parsing fails, `None`.
  """
  try:
    ndarray = tensor_util.MakeNdarray(tensor_proto)
    return ndarray.tolist() if return_list else ndarray
  except TypeError:
    # Depending on tensor_debug_mode, certain dtype of tensors don't
    # have logged debug tensor values.
    return None


def _execution_digest_from_debug_event_proto(debug_event, locator):
  """Convert a DebugEvent proto into an ExecutionDigest data object."""
  return ExecutionDigest(
      debug_event.wall_time,
      locator,
      debug_event.execution.op_type,
      output_tensor_device_ids=(debug_event.execution.output_tensor_device_ids
                                or None))


def _execution_from_debug_event_proto(debug_event, locator):
  """Convert a DebugEvent proto into an Execution data object."""
  execution_proto = debug_event.execution

  debug_tensor_values = None
  if (execution_proto.tensor_debug_mode ==
      debug_event_pb2.TensorDebugMode.FULL_TENSOR):
    pass  # TODO(cais): Build tensor store.
  elif (execution_proto.tensor_debug_mode !=
        debug_event_pb2.TensorDebugMode.NO_TENSOR):
    debug_tensor_values = []
    for tensor_proto in execution_proto.tensor_protos:
      # TODO(cais): Refactor into a helper method.
      debug_tensor_values.append(
          _parse_tensor_value(tensor_proto, return_list=True))
  return Execution(
      _execution_digest_from_debug_event_proto(debug_event, locator),
      execution_proto.code_location.host_name,
      tuple(execution_proto.code_location.stack_frame_ids),
      execution_proto.tensor_debug_mode,
      graph_id=execution_proto.graph_id,
      input_tensor_ids=tuple(execution_proto.input_tensor_ids),
      output_tensor_ids=tuple(execution_proto.output_tensor_ids),
      debug_tensor_values=_tuple_or_none(debug_tensor_values))


class DebugDataReader(object):
  """A reader that reads structured debugging data in the tfdbg v2 format.

  The set of data read by an object of this class concerns the execution history
  of a tfdbg2-instrumented TensorFlow program.

  Note:
    - An object of this class incrementally reads data from files that belong to
      the tfdbg v2 DebugEvent file set. Calling `update()` triggers the reading
      from the last-successful reading positions in the files.
    - This object can be used as a context manager. Its `__exit__()` call
      closes the file readers cleanly.
  """

  def __init__(self, dump_root):
    self._reader = DebugEventsReader(dump_root)

    # TODO(cais): Implement pagination for memory constraints.
    self._execution_digests = []

    # Mapping (host_name, file_path) tuple to offset in the .source_files file.
    self._host_name_file_path_to_offset = collections.OrderedDict()
    # A dict mapping id to (host_name, file_path, lineno, func) tuple.
    self._stack_frame_by_id = dict()
    # Stores unprocessed stack frame IDs. This is necessary to handle the
    # case in which reading of the .stack_frames file gets ahead of the reading
    # of the .source_files file.
    self._unprocessed_stack_frames = dict()
    # A dict mapping id to DebuggedDevice objects.
    self._device_by_id = dict()
    # A dict mapping id to DebuggedGraph objects.
    self._graph_by_id = dict()
    self._graph_op_digests = []
    # TODO(cais): Implement pagination for memory constraints.
    self._graph_execution_trace_digests = []

    self._monitors = []

  def _add_monitor(self, monitor):
    self._monitors.append(monitor)

  def _load_source_files(self):
    """Incrementally read the .source_files DebugEvent file."""
    source_files_iter = self._reader.source_files_iterator()
    for debug_event, offset in source_files_iter:
      source_file = debug_event.source_file
      self._host_name_file_path_to_offset[
          (source_file.host_name, source_file.file_path)] = offset

  def _load_stack_frames(self):
    """Incrementally read the .stack_frames file.

    This must be called after _load_source_files().
    It assumes that the following contract is honored by the writer of the tfdbg
    v2 data file set:
      - Before a stack frame is written to the .stack_frames file, the
        corresponding source file information must have been written to the
        .source_files file first.
    """
    stack_frames_iter = self._reader.stack_frames_iterator()
    for debug_event, _ in stack_frames_iter:
      stack_frame_with_id = debug_event.stack_frame_with_id
      file_line_col = stack_frame_with_id.file_line_col
      self._unprocessed_stack_frames[stack_frame_with_id.id] = file_line_col
    # We do the processing in a separate stage, because the reading in the
    # .source_files file may sometimes get ahead of the .source_files file.
    unprocessed_stack_frame_ids = tuple(self._unprocessed_stack_frames.keys())
    for stack_frame_id in unprocessed_stack_frame_ids:
      file_line_col = self._unprocessed_stack_frames[stack_frame_id]
      if len(self._host_name_file_path_to_offset) > file_line_col.file_index:
        host_name, file_path = list(self._host_name_file_path_to_offset.keys())[
            file_line_col.file_index]
        self._stack_frame_by_id[stack_frame_id] = (
            host_name, file_path, file_line_col.line, file_line_col.func)
      del self._unprocessed_stack_frames[stack_frame_id]

  def _load_graphs(self):
    """Incrementally read the .graphs file.

    Compiles the DebuggedGraph and GraphOpCreation data.
    """
    graphs_iter = self._reader.graphs_iterator()
    for debug_event, offset in graphs_iter:
      if debug_event.graph_op_creation.ByteSize():
        op_creation_proto = debug_event.graph_op_creation
        op_digest = GraphOpCreationDigest(
            debug_event.wall_time,
            offset,
            op_creation_proto.graph_id,
            op_creation_proto.op_type,
            op_creation_proto.op_name,
            tuple(op_creation_proto.output_tensor_ids),
            op_creation_proto.code_location.host_name,
            tuple(op_creation_proto.code_location.stack_frame_ids),
            input_names=tuple(op_creation_proto.input_names))
        self._graph_op_digests.append(op_digest)
        debugged_graph = self._graph_by_id[op_creation_proto.graph_id]
        debugged_graph.add_op(op_digest)
        for dst_slot, input_name in enumerate(op_creation_proto.input_names):
          src_op_name, src_slot = input_name.split(":")
          debugged_graph.add_op_consumer(src_op_name, int(src_slot),
                                         op_creation_proto.op_name, dst_slot)

      elif debug_event.debugged_graph.ByteSize():
        graph_proto = debug_event.debugged_graph
        graph = DebuggedGraph(
            graph_proto.graph_name or None,
            graph_proto.graph_id,
            outer_graph_id=graph_proto.outer_context_id or None)
        self._graph_by_id[graph_proto.graph_id] = graph
        if graph_proto.outer_context_id:
          self._graph_by_id[
              graph_proto.outer_context_id].add_inner_graph_id(graph.graph_id)
      elif debug_event.debugged_device.ByteSize():
        device_proto = debug_event.debugged_device
        self._device_by_id[device_proto.device_id] = DebuggedDevice(
            device_proto.device_name, device_proto.device_id)

  def _load_graph_execution_traces(self):
    """Incrementally load the .graph_execution_traces file."""
    for i, traces_iter in enumerate(
        self._reader.graph_execution_traces_iterators()):
      for debug_event, offset in traces_iter:
        self._graph_execution_trace_digests.append(
            self._graph_execution_trace_digest_from_debug_event_proto(
                debug_event, (i, offset)))
        if self._monitors:
          graph_execution_trace = (
              self._graph_execution_trace_from_debug_event_proto(
                  debug_event, (i, offset)))
          for monitor in self._monitors:
            monitor.on_graph_execution_trace(
                len(self._graph_execution_trace_digests) - 1,
                graph_execution_trace)

  def _graph_execution_trace_digest_from_debug_event_proto(
      self, debug_event, locator):
    trace_proto = debug_event.graph_execution_trace
    op_name = trace_proto.op_name
    op_type = self._lookup_op_type(trace_proto.tfdbg_context_id, op_name)
    return GraphExecutionTraceDigest(
        debug_event.wall_time, locator, op_type, op_name,
        trace_proto.output_slot,
        debug_event.graph_execution_trace.tfdbg_context_id)

  def _graph_execution_trace_from_debug_event_proto(self, debug_event, locator):
    """Convert a DebugEvent proto into a GraphExecutionTrace data object."""
    trace_proto = debug_event.graph_execution_trace
    graph_ids = [trace_proto.tfdbg_context_id]
    # Walk up the chain of outer contexts (graphs), so as to include all of
    # their IDs
    while True:
      graph = self.graph_by_id(graph_ids[0])
      if graph.outer_graph_id:
        graph_ids.insert(0, graph.outer_graph_id)
      else:
        break

    if (trace_proto.tensor_debug_mode ==
        debug_event_pb2.TensorDebugMode.FULL_TENSOR):
      debug_tensor_value = None
    else:
      debug_tensor_value = _parse_tensor_value(
          trace_proto.tensor_proto, return_list=True)
    return GraphExecutionTrace(
        self._graph_execution_trace_digest_from_debug_event_proto(
            debug_event, locator),
        graph_ids=graph_ids,
        tensor_debug_mode=trace_proto.tensor_debug_mode,
        debug_tensor_value=debug_tensor_value,
        device_name=trace_proto.device_name or None)

  def _lookup_op_type(self, graph_id, op_name):
    """Lookup the type of an op by name and the immediately enclosing graph.

    Args:
      graph_id: Debugger-generated ID of the immediately-enclosing graph.
      op_name: Name of the op.

    Returns:
      Op type as a str.
    """
    return self._graph_by_id[graph_id].get_op_creation_digest(op_name).op_type

  def _load_execution(self):
    """Incrementally read the .execution file."""
    execution_iter = self._reader.execution_iterator()
    for debug_event, offset in execution_iter:
      self._execution_digests.append(
          _execution_digest_from_debug_event_proto(debug_event, offset))
      if self._monitors:
        execution = _execution_from_debug_event_proto(debug_event, offset)
        for monitor in self._monitors:
          monitor.on_execution(len(self._execution_digests) - 1, execution)

  def update(self):
    """Perform incremental read of the file set."""
    self._load_source_files()
    self._load_stack_frames()
    self._load_graphs()
    self._load_graph_execution_traces()
    self._load_execution()

  def source_file_list(self):
    """Get a list of source files known to the debugger data reader.

    Returns:
      A tuple of `(host_name, file_path)` tuples.
    """
    return tuple(self._host_name_file_path_to_offset.keys())

  def source_lines(self, host_name, file_path):
    """Read the line-by-line content of a source file.

    Args:
      host_name: Host name on which the source file is located.
      file_path: File path at which the source file is located.

    Returns:
      Lines of the source file as a `list` of `str`s.
    """
    offset = self._host_name_file_path_to_offset[(host_name, file_path)]
    return list(self._reader.read_source_files_event(offset).source_file.lines)

  def starting_wall_time(self):
    """Wall timestamp for when the debugged TensorFlow program started.

    Returns:
      Stating wall time as seconds since the epoch, as a `float`.
    """
    return self._reader.starting_wall_time()

  def tensorflow_version(self):
    """TensorFlow version used in the debugged TensorFlow program.

    Note: this is not necessarily the same as the version of TensorFlow used to
    load the DebugEvent file set.

    Returns:
      TensorFlow version used by the debugged program, as a `str`.
    """
    return self._reader.tensorflow_version()

  def tfdbg_run_id(self):
    """Get the debugger run ID of the debugged TensorFlow program."""
    return self._reader.tfdbg_run_id()

  def outermost_graphs(self):
    """Get the number of outer most graphs read so far."""
    return [graph for graph in self._graph_by_id.values()
            if not graph.outer_graph_id]

  def graph_by_id(self, graph_id):
    """Get a DebuggedGraph object by its ID."""
    return self._graph_by_id[graph_id]

  def device_name_by_id(self, device_id):
    """Get the name of a device by the debugger-generated ID of the device."""
    return self._device_by_id[device_id].device_name

  def device_name_map(self):
    """Get a map mapping device IDs to device names."""
    return {device_id: self._device_by_id[device_id].device_name
            for device_id in self._device_by_id}

  def graph_op_digests(self, op_type=None):
    """Get the list of the digests for graph-op creation so far.

    Args:
      op_type: Optional op type to filter the creation events with.

    Returns:
      A list of `GraphOpCreationDigest` objects.
    """
    if op_type is not None:
      return [digest for digest in self._graph_op_digests
              if digest.op_type == op_type]
    else:
      return self._graph_op_digests

  def graph_execution_traces(self, digest=False, begin=None, end=None):
    """Get all the intra-graph execution tensor traces read so far.

    Args:
      digest: Whether the results will be returned in the more light-weight
        digest form.
      begin: Optional beginning index for the requested traces or their digests.
        Python-style negative indices are supported.
      end: Optional ending index for the requested traces or their digests.
        Python-style negative indices are supported.

    Returns:
      If `digest`: a `list` of `GraphExecutionTraceDigest` objects.
      Else: a `list` of `GraphExecutionTrace` objects.
    """
    digests = self._graph_execution_trace_digests
    if begin is not None or end is not None:
      begin = begin or 0
      end = end or len(digests)
      digests = digests[begin:end]
    if digest:
      return digests
    else:
      return [self.read_graph_execution_trace(digest) for digest in digests]

  def num_graph_execution_traces(self):
    """Get the number of graph execution traces read so far."""
    return len(self._graph_execution_trace_digests)

  def executions(self, digest=False, begin=None, end=None):
    """Get `Execution`s or `ExecutionDigest`s this reader has read so far.

    Args:
      digest: Whether the results are returned in a digest form, i.e.,
        `ExecutionDigest` format, instead of the more detailed `Execution`
        format.
      begin: Optional beginning index for the requested execution data objects
        or their digests. Python-style negative indices are supported.
      end: Optional ending index for the requested execution data objects or
        their digests. Python-style negative indices are supported.

    Returns:
      If `digest`: a `list` of `ExecutionDigest` objects.
      Else: a `list` of `Execution` objects.
    """
    digests = self._execution_digests
    if begin is not None or end is not None:
      begin = begin or 0
      end = end or len(digests)
      digests = digests[begin:end]
    if digest:
      return digests
    else:
      # TODO(cais): Optimizer performance removing repeated file open/close.
      return [self.read_execution(digest) for digest in digests]

  def num_executions(self):
    """Get the number of execution events read so far."""
    return len(self._execution_digests)

  def read_execution(self, execution_digest):
    """Read a detailed Execution object."""
    debug_event = self._reader.read_execution_event(execution_digest.locator)
    return _execution_from_debug_event_proto(debug_event,
                                             execution_digest.locator)

  def read_graph_execution_trace(self, graph_execution_trace_digest):
    """Read the detailed graph execution trace.

    Args:
      graph_execution_trace_digest: A `GraphExecutionTraceDigest` object.

    Returns:
      The corresponding `GraphExecutionTrace` object.
    """
    debug_event = self._reader.read_graph_execution_traces_event(
        graph_execution_trace_digest.locator)
    return self._graph_execution_trace_from_debug_event_proto(
        debug_event, graph_execution_trace_digest.locator)

  def read_execution_stack_trace(self, execution):
    """Read the stack trace of a given Execution object.

    Args:
      execution: The Execution object of interest.

    Returns:
      1. The host name.
      2. The stack trace, as a list of (file_path, lineno, func) tuples.
    """
    host_name = self._stack_frame_by_id[execution.stack_frame_ids[0]][0]
    return (host_name, [
        self._stack_frame_by_id[frame_id][1:]
        for frame_id in execution.stack_frame_ids])

  def read_graph_op_creation_stack_trace(self, graph_op_creation_digest):
    """Read the stack trace of a given graph op creation object.

    Args:
      graph_op_creation_digest: The GraphOpCreationDigest object of interest.

    Returns:
      A tuple consisting of:
        1. The host name.
        2. The stack trace, as a list of (file_path, lineno, func) tuples.
    """
    return graph_op_creation_digest.host_name, [
        self._stack_frame_by_id[frame_id][1:]
        for frame_id in graph_op_creation_digest.stack_frame_ids
    ]

  # TODO(cais): Add graph_execution_digests() with an ExecutionDigest
  #   as a kwarg, to establish the association between top-level and intra-graph
  #   execution events.

  def execution_to_tensor_values(self, execution):
    """Read the full tensor values from an Execution or ExecutionDigest.

    Args:
      execution: An `ExecutionDigest` or `ExeuctionDigest` object.

    Returns:
      A list of numpy arrays representing the output tensor values of the
        execution event.
    """
    debug_event = self._reader.read_execution_event(execution.locator)
    return [_parse_tensor_value(tensor_proto)
            for tensor_proto in debug_event.execution.tensor_protos]

  def graph_execution_trace_to_tensor_value(self, trace):
    """Read full tensor values from an Execution or ExecutionDigest.

    Args:
      trace: An `GraphExecutionTraceDigest` or `GraphExecutionTrace` object.

    Returns:
      A numpy array representing the output tensor value of the intra-graph
        tensor execution event.
    """
    debug_event = self._reader.read_graph_execution_traces_event(trace.locator)
    return _parse_tensor_value(debug_event.graph_execution_trace.tensor_proto)

  def symbolic_tensor_id(self, graph_id, op_name, output_slot):
    """Get the ID of a symbolic tensor.

    Args:
      graph_id: The ID of the immediately-enclosing graph.
      op_name: Name of the op.
      output_slot: Output slot as an int.

    Returns:
      The ID of the symbolic tensor as an int.
    """
    return self._graph_by_id[graph_id].get_tensor_id(op_name, output_slot)

  def graph_execution_trace_to_tensor_id(self, trace):
    """Get symbolic tensor ID from a GraphExecutoinTraceDigest object."""
    return self.symbolic_tensor_id(
        trace.graph_id, trace.op_name, trace.output_slot)

  def __enter__(self):
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    del exception_type, exception_value, traceback  # Unused
    self._reader.close()
