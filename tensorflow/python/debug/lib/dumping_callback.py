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
"""Dumping op callbacks: Enables dump-based features in tfdbg v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import atexit
import collections
import re
import socket
import threading
import uuid

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.core.protobuf import graph_debug_info_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export

DumpingConfig = collections.namedtuple(
    "DumpingConfig",
    "dump_root tensor_debug_mode circular_buffer_size "
    "op_regex tensor_dtypes")
_state = threading.local()


@ops.RegisterGradient("DebugIdentityV2")
def _debug_identity_v2_grad(op, dy):
  """Gradient function for the DebugIdentityV2 op."""
  del op  # Unused
  return dy


def _get_writer():
  """Get the debug events writer for the currently configured dump root."""
  # TODO(cais): Explore caching the object for possible performance gain.
  # TODO(cais): Rename circular_buffer_size to circular_buffer_size in C++ and
  #   Python-bindng code.
  return debug_events_writer.DebugEventsWriter(
      _state.config.dump_root,
      circular_buffer_size=_state.config.circular_buffer_size)


def _get_id():
  """Get a short unique ID."""
  return str(uuid.uuid4())


def _get_context_id(context):
  """Get a unique ID for an op-construction context (e.g., a graph).

  If the graph has been encountered before, reuse the same unique ID.

  Args:
    context: A context to get the unique ID for. Must be hashable. E.g., a Graph
      object.

  Returns:
    A unique ID for the context.
  """
  if context not in _state.context_to_id:
    _state.context_to_id[context] = _get_id()
  return _state.context_to_id[context]


def _write_source_file_content(file_path):
  """Send the content of a source file via debug-events writer.

  Args:
    file_path: Path to the source file.

  Returns:
    An int index for the file.
  """
  if file_path not in _state.source_file_paths:
    lines = None
    if source_utils.is_extension_uncompiled_python_source(file_path):
      try:
        lines, _ = source_utils.load_source(file_path)
      except IOError:
        # Accept the fact that some source files are not readable. Here we use
        # best effort to send the source-file contents.
        pass
    writer = _get_writer()
    writer.WriteSourceFile(debug_event_pb2.SourceFile(
        file_path=file_path, host_name=_state.hostname, lines=lines))
    _state.source_file_paths.append(file_path)
  return _state.source_file_paths.index(file_path)


def _process_stack_frames():
  """Process stack frames.

  Send the content of source-files, on a best-effort basis.

  Returns:
    A list of stack frame IDs.
  """
  stack_frames = tf_stack.extract_stack()
  stack_frame_ids = []
  writer = None
  for file_path, lineno, func, _ in stack_frames:
    if (file_path, lineno, func) not in _state.stack_frame_to_id:
      stack_frame_id = _get_id()
      _state.stack_frame_to_id[(file_path, lineno, func)] = stack_frame_id
      file_index = _write_source_file_content(file_path)
      file_line_col = graph_debug_info_pb2.GraphDebugInfo.FileLineCol(
          file_index=file_index, line=lineno, func=func)
      stack_frame_with_id = debug_event_pb2.StackFrameWithId(
          id=stack_frame_id, file_line_col=file_line_col)
      writer = _get_writer()
      writer.WriteStackFrameWithId(stack_frame_with_id)
    stack_frame_ids.append(_state.stack_frame_to_id[(file_path, lineno, func)])

  code_location = debug_event_pb2.CodeLocation(
      host_name=_state.hostname, stack_frame_ids=stack_frame_ids)
  return code_location


def _should_dump_tensor(op_type, dtype):
  """Determine if the given tensor's value will be dumped.

  The determination is made given the configurations such as `op_regex`,
  `tensor_dtypes`.

  Args:
    op_type: Name of the op's type, as a string (e.g., "MatMul").
    dtype: The dtype of the tensor, as a `dtypes.DType` object.

  Returns:
    A bool indicating whether the tensor's value will be dumped.
  """
  should_dump = True
  if _state.config.op_regex:
    should_dump = (should_dump and
                   re.match(_state.config.op_regex, op_type))
  if _state.config.tensor_dtypes:
    if isinstance(_state.config.tensor_dtypes, (list, tuple)):
      should_dump = (should_dump and
                     any(dtype == dtype_item for dtype_item
                         in _state.config.tensor_dtypes))
    else:  # A callable that takes a DType argument and return a boolean.
      should_dump = should_dump and _state.config.tensor_dtypes(dtype)
  return should_dump


def _instrument_symbolic_tensors(tensors, op_type, op_name, tfdbg_context_id):
  """Add debugging instrumentation for symbolic (i.e., non-eager) tensors.

  The detailed fashion in which the tensors are instrumented is determined
  by the tensor_debug_mode configured for the currently enabled dumping
  callback.

  Args:
    tensors: A tuple of Tensors to instrument. It is assumed that their ordering
      corresponds to the ordering of output tensors of an original op. Output
      slot indices (0-based) will be generated based on the ordering.
    op_type: Name of the op type of the node that emits `tensors` (e.g.,
      "MatMul"), as a string.
    op_name: Name of the node that emits `tensors` (e.g., "dense_1/MatMul"), as
      a string.
    tfdbg_context_id: A unique ID for the context that the op belongs to (e.g.,
      a graph).

  Returns:
    Non-eager Tensors that override the `tensors` as the output of the op
    that originally generated `tensors`. In some cases (e.g., non-V1 graph
    mode), this may be `None`, as the instrumentation can simply rely on
    automatic control dependencies (see `auto_control_deps.py`) instead of
    tensor overriding.
  """
  tensor_debug_mode = _state.config.tensor_debug_mode
  debug_urls = ["file://%s" % _state.config.dump_root]
  is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
  instrumented_tensors = [] if is_v1_graph_mode else None
  for output_slot, tensor in enumerate(tensors):
    if not _should_dump_tensor(op_type, tensor.dtype):
      if is_v1_graph_mode:
        instrumented_tensors.append(tensor)
      continue

    if tensor_debug_mode == debug_event_pb2.TensorDebugMode.NO_TENSOR:
      # Except in V1 graph mode + control flow, debug_identity_v2 trigger auto
      # control dependency because it's a stateful op.
      debug_tensor = gen_debug_ops.debug_identity_v2(
          # Use an empty (shape=[0]) float32 tensor for the NO_TENSOR mode
          # as a low-overhead placeholder, since no actual tensor value is
          # traced.
          constant_op.constant([], dtype=dtypes.float32),
          tfdbg_context_id=tfdbg_context_id,
          op_name=op_name,
          output_slot=output_slot,
          tensor_debug_mode=_state.config.tensor_debug_mode,
          debug_urls=debug_urls)
      if is_v1_graph_mode:
        # TODO(cais): Evaluate performance optimization options. For the
        # `NO_TENSOR` debug mode, an alternative is to add `debug_tensor` as a
        # control dependency of `tensor.op` without an additional identity op.
        identity = array_ops.identity(tensor)
        identity.op._add_control_input(  # pylint: disable=protected-access
            debug_tensor.op)
        instrumented_tensors.append(identity)
    elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
      debug_tensor = gen_debug_ops.debug_identity_v2(
          tensor,
          tfdbg_context_id=tfdbg_context_id,
          op_name=op_name,
          output_slot=output_slot,
          tensor_debug_mode=_state.config.tensor_debug_mode,
          debug_urls=debug_urls)
      if is_v1_graph_mode:
        instrumented_tensors.append(debug_tensor)
    else:
      raise NotImplementedError(
          "Symbolic tensor instrumentation is not implemented for debug "
          "mode %s" % _state.config.tensor_debug_mode)
  return instrumented_tensors


def _dump_eager_tensors(tensors, op_type, input_tensor_ids):
  """Dump the value of eager tensors.

  The destination of the dumping is determined by the dump_root of the currently
  enabled dumping callback. The tensors may be transformed prior to dumping
  (e.g., reduced as summary statistics such as minimum, maximum and arithmetic
  mean). The details of this transformation (if any) depends on the
  tensor_debug_mode of the currently enabled dumping callback.

  Args:
    tensors: The EagerTensors whose values are to be dumped, with or without
      value transform.
    op_type: Type of the op that generates the tensors, as a string.
    input_tensor_ids: IDs of the input EagerTensors to the op.

  Returns:
    A tfdbg Execution protocol buffer.
  """
  tensor_debug_mode = _state.config.tensor_debug_mode
  output_tensor_ids = [
      t._id for t in tensors]  # pylint:disable=protected-access
  if tensor_debug_mode == debug_event_pb2.TensorDebugMode.NO_TENSOR:
    return debug_event_pb2.Execution(
        op_type=op_type,
        num_outputs=len(tensors),
        input_tensor_ids=input_tensor_ids,
        output_tensor_ids=output_tensor_ids,
        tensor_debug_mode=tensor_debug_mode,
        code_location=_process_stack_frames())
  elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
    execution_proto = debug_event_pb2.Execution(
        op_type=op_type,
        num_outputs=len(tensors),
        input_tensor_ids=input_tensor_ids,
        output_tensor_ids=output_tensor_ids,
        tensor_debug_mode=tensor_debug_mode,
        code_location=_process_stack_frames())
    for tensor in tensors:
      if (_should_dump_tensor(op_type, tensor.dtype) and
          tensor.dtype.is_numpy_compatible):
        execution_proto.tensor_protos.append(
            tensor_util.make_tensor_proto(tensor.numpy()))
    return execution_proto
  else:
    raise NotImplementedError(
        "Tensor instrumentation is not implemented for debug mode %s yet " %
        _state.config.tensor_debug_mode)


def _dumping_callback(op_type,
                      inputs,
                      attrs,
                      outputs,
                      op_name=None,
                      graph=None):
  """Op callback for tracing a TF program's execution."""
  del attrs  # Unused

  writer = _get_writer()
  if graph:
    context_id = _get_context_id(graph)
    assert op_name is not None
    graph_op_creation = debug_event_pb2.GraphOpCreation(
        op_type=op_type,
        op_name=op_name,
        graph_name=graph.name if hasattr(graph, "name") else None,
        graph_id=context_id,
        input_names=[input_tensor.name for input_tensor in inputs],
        num_outputs=len(outputs),
        code_location=_process_stack_frames())
    writer.WriteGraphOpCreation(graph_op_creation)
    if outputs and compat.as_bytes(
        op_type) not in op_callbacks_common.OP_CALLBACK_SKIP_OPS:
      return _instrument_symbolic_tensors(outputs, op_type, op_name, context_id)
  else:
    input_ids = [t._id for t in inputs]  # pylint:disable=protected-access
    writer.WriteExecution(_dump_eager_tensors(outputs, op_type, input_ids))


DEFAULT_TENSOR_DEBUG_MODE = "NO_TENSOR"


@tf_export("debugging.experimental.enable_dump_debug_info")
def enable_dump_debug_info(dump_root,
                           tensor_debug_mode=DEFAULT_TENSOR_DEBUG_MODE,
                           circular_buffer_size=1000,
                           op_regex=None,
                           tensor_dtypes=None):
  """Enable dumping debugging information from a TensorFlow program.

  The debugging information is dumped to a directory on the file system
  specified as `dump_root`.

  The dumped debugging information can be ingested by debugger UIs.

  The files in the dump directory contain the following information:
    - TensorFlow Function construction (e.g., compilation of Python functions
      decorated with @tf.function), the op types, names (if available), context,
      the input and output tensors, and the associated stack traces.
    - Execution of TensorFlow operations (ops) and Functions and their stack
      traces, op types, names (if available) and contexts. In addition,
      depending on the value of the `tensor_debug_mode` argument (see Args
      section below), the value(s) of the output tensors or more concise
      summaries of the tensor values will be dumped.
    - A snapshot of Python source files involved in the execution of the
      TensorFlow program.

  Once enabled, the dumping can be disabled with the corresponding
  `disable_dump_debug_info()` method under the same Python namespace.
  Calling this method more than once with the same `dump_root` is idempotent.
  Calling this method more than once with different `tensor_debug_mode`s
  leads to a `ValueError`.
  Calling this method more than once with different `circular_buffer_size`s
  leads to a `ValueError`.
  Calling this method with a different `dump_root` abolishes the
  previously-enabled `dump_root`.

  Usage example:

  ```py
  tf.debugging.experimental.enable_dump_debug_info('/tmp/my-tfdbg-dumps')

  # Code to build, train and run your TensorFlow model...
  ```

  Args:
    dump_root: The directory path where the dumping information will be written.
    tensor_debug_mode: Debug mode for tensor values, as a string.
      The currently supported options are:
        - "NO_TENSOR": (Default) Only traces the execution of ops' output
          tensors, while not dumping the value of the ops' output tensors
          or any form of concise summary of them.
    circular_buffer_size: Size of the circular buffers for execution events.
      These circular buffers are designed to reduce the overhead of debugging
      dumping. They hold the most recent debug events concerning eager execution
      of ops and `tf.function`s and traces of tensor values computed inside
      `tf.function`s. They are written to the file system only when the proper
      flushing method is called (see description of return values below).
      Expected to be an integer. If <= 0, the circular-buffer behavior will be
      disabled, i.e., the execution debug events will be written to the file
      writers in the same way as non-execution events such as op creations and
      source-file snapshots.
    op_regex: Dump data from only the tensors from op types that matches to the
      regular expression (through Python's `re.match()`).
      "Op type" refers to the names of the TensorFlow operations (e.g.,
      "MatMul", "LogSoftmax"), which may repeat in a TensorFlow
      function. It does *not* refer to the names of nodes (e.g.,
      "dense/MatMul", "dense_1/MatMul_1") which are unique within a function.
      - Example 1: Dump tensor data from only MatMul and Relu ops
        `op_regex="^(MatMul|Relu)$"`.
      - Example 2: Dump tensors from all ops *except* Relu:
        `op_regex="(?!^Relu$)"`.
      This filter operates in a logical AND relation with `tensor_dtypes`.
    tensor_dtypes: Dump data from only the tensors of which the specified
      dtypes. This optional argument can be in any of the following format:
      - a list or tuple of `DType` objects or strings that can be converted
        to `DType` objects via `tf.as_dtype()`. Examples:
        - `tensor_dtype=[tf.float32, tf.float64]`,
        - `tensor_dtype=["float32", "float64"]`,
        - `tensor_dtypes=(tf.int32, tf.bool)`,
        - `tensor_dtypes=("int32", "bool")`
      - a callable that takes a single `DType` argument and returns a Python
        `boolean` indicating whether the dtype is to be included in the data
        dumping. Examples:
        - `tensor_dtype=lambda dtype: dtype.is_integer`.
      This filter operates in a logical AND relation with `op_regex`.
  Returns:
    A DebugEventsWriter instance used by the dumping callback. The caller
    may use its flushing methods, including `FlushNonExecutionFiles()` and
    `FlushExecutionFiles()`.
  """
  # TODO(cais): Revise the "UIs (currently under construction)" part of the doc
  # string above.
  # TODO(cais): Add Python code example to the doc string above.
  tensor_debug_mode_keys = debug_event_pb2.TensorDebugMode.keys()
  if tensor_debug_mode not in tensor_debug_mode_keys:
    raise ValueError(
        "Invalid value in tensor_debug_mode ('%s'). Valid options are: %s" %
        (tensor_debug_mode, tensor_debug_mode_keys))

  tensor_debug_mode = debug_event_pb2.TensorDebugMode.Value(tensor_debug_mode)
  if tensor_debug_mode not in (debug_event_pb2.TensorDebugMode.NO_TENSOR,
                               debug_event_pb2.TensorDebugMode.FULL_TENSOR):
    raise NotImplementedError(
        "tfdbg dumping: support for tensor debug mode %s is not "
        "implemented yet" % tensor_debug_mode)

  if (hasattr(_state, "config") and
      _state.config.circular_buffer_size != circular_buffer_size):
    raise ValueError(
        "There is already a dumping callback configured with a different "
        "circular-buffer size (%d). Therefore the newly request "
        "circular-buffer size (%d) will not be honored." %
        (_state.config.circular_buffer_size, circular_buffer_size))

  if (hasattr(_state, "config") and
      _state.config.tensor_debug_mode != tensor_debug_mode):
    raise ValueError(
        "There is already a dumping callback configured for dump root "
        "%s with a different "
        "tensor-debug mode (%s). Therefore the newly request "
        "tensor-debug mode (%s) size will not be honored." %
        (_state.config.dump_root,
         tensor_debug_mode_keys[_state.config.tensor_debug_mode],
         tensor_debug_mode_keys[tensor_debug_mode]))

  # Validate the types of tensor_dtypes.
  if tensor_dtypes is not None:
    if (not isinstance(tensor_dtypes, (list, tuple)) and
        not callable(tensor_dtypes)):
      raise ValueError(
          "If specified, tensor_dtypes is expected to be a list, a tuple, or "
          "a callable that takes a DType argument and returns a boolean, "
          "but received %s" % (tensor_dtypes,))
    if isinstance(tensor_dtypes, (list, tuple)):
      tensor_dtypes = [
          dtypes.as_dtype(dtype_item) for dtype_item in tensor_dtypes]

  if not hasattr(_state, "config") or _state.config.dump_root != dump_root:
    _state.config = DumpingConfig(
        dump_root=dump_root,
        tensor_debug_mode=tensor_debug_mode,
        circular_buffer_size=int(circular_buffer_size),
        op_regex=re.compile(op_regex) if op_regex else None,
        tensor_dtypes=tensor_dtypes)
    _state.hostname = socket.gethostname()
    # A list of source-file paths.
    _state.source_file_paths = []
    # A map from stack frame (FileLineCol) to unique ID.
    _state.stack_frame_to_id = dict()
    # Mapping op context to unique ID.
    _state.context_to_id = dict()

  op_callbacks.add_op_callback(_dumping_callback)
  logging.info(
      "Enabled dumping callback in thread %s "
      "(dump root: %s, tensor debug mode: %s)",
      threading.current_thread().name, _state.config.dump_root,
      tensor_debug_mode)

  atexit.register(disable_dump_debug_info)
  return _get_writer()


@tf_export("debugging.experimental.disable_dump_debug_info")
def disable_dump_debug_info():
  """Disable the currently-enabled debugging dumping.

  If the `enable_dump_debug_info()` method under the same Python namespace
  has been invoked before, calling this method disables it. If no call to
  `enable_dump_debug_info()` has been made, calling this method is a no-op.
  Calling this method more than once is idempotent.
  """
  if hasattr(_state, "config"):
    dump_root = _state.config.dump_root
    delattr(_state, "config")
    debug_events_writer.DebugEventsWriter(dump_root).Close()
    op_callbacks.remove_op_callback(_dumping_callback)
    logging.info("Disabled dumping callback in thread %s (dump root: %s)",
                 threading.current_thread().name, dump_root)
