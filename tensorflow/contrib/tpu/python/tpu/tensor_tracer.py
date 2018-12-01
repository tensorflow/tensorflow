# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# ========================================================================
"""A utility to trace tensor values on TPU."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import re

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

_TRACER_LOG_PREFIX = ' [>>>TT>>>]'
_DEVICE_TYPE_TPU = 'tpu'
_DEVICE_TYPE_CPU = 'cpu'
_GLOBAL_STEP_OP_NAME = 'GLOBAL-STEP'
_TRACE_MODE_NAN_INF = 'nan-inf'
_TRACE_MODE_PART_TENSOR = 'part-tensor'
_TRACE_MODE_PART_TENSOR_SIZE = 3
_TRACE_MODE_FULL_TENSOR = 'full-tensor'
_RECORD_OUTSIDE_OP_RANGE = 'not-traced-outside-op-range'
_RECORD_SHOULD_NOT_TRACE = 'not-traced-should-not-trace'
_RECORD_FILTERED_OUT = 'not-traced-filtered-out'
_RECORD_SCALAR = 'not-traced-scalar'
_RECORD_DYNAMIC_SHAPE = 'not-traced-dynamic-shape'
_RECORD_GET_TRACED = 'get-traced'
_MARKER_SECTION_BEGIN = '!!!!!!! section-begin:'
_MARKER_SECTION_END = '!!!!!!! section-end:'
_SECTION_NAME_CONFIG = 'configuration'
_SECTION_NAME_REASON = 'reason'
_SECTION_NAME_OP_LIST = 'op-list'
_SECTION_NAME_GRAPH = 'graph'
_FIELD_NAME_VERSION = 'version:'
_FIELD_NAME_DEVICE = 'device:'
_FIELD_NAME_TRACE_MODE = 'trace-mode:'
_FIELD_NAME_NUM_REPLICAS = 'num-replicas:'
_FIELD_NAME_NUM_OPS = 'number-of-ops:'
_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED = 'topological-sort-succeed:'
_FLAGS_ENV_VAR = 'TENSOR_TRACER_FLAGS'
_FLAG_SINGLE_QUOTE_PAT = re.compile(r"\s*--([^=]+)='([^']*)'")
_FLAG_DOUBLE_QUOTE_PAT = re.compile(r'\s*--([^=]+)="([^"]*)"')
_FLAG_NO_QUOTE_PAT = re.compile(r'\s*--([^=]+)=(\S*)')
_FLAG_NAME_ENABLE = 'enable'
_FLAG_NAME_TRACE_MODE = 'trace_mode'
_FLAG_NAME_INTERESTING_OPS = 'interesting_ops'
_FLAG_NAME_TRACE_FILE = 'trace_file_path'
_FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR = 'use_test_undeclared_outputs_dir'
_FLAG_NAME_OP_RANGE = 'op_range'
_OP_RANGE_PAT = re.compile(r'(\d+):(\d+)')
_OUTPUT_STREAM_ESCAPE = 'file://'
_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR = 'TEST_UNDECLARED_OUTPUTS_DIR'


class TensorTracer(object):
  """A software construct for tracing tensor values in a TF graph on TPU.

  This utility is disabled by default. It can be enabled by setting
  the TENSOR_TRACER_FLAGS env variable as:
    export TENSOR_TRACER_FLAGS="--enable=1"
  If it is enabled, it will trace the output tensor values of
  selected Ops in the graph. It has two outputs: (1) the traces and (2)
  a report. The traces are dumped to a specified local file on the TPU
  host. The report is printed to the log.info of the TPU job.
  By passing options via the env variable, users can change:
     (1) the trace mode (e.g., detecting NaN/Inf, printing partial or
         full tensor values)
     (2) which Ops to be traced (via op.name or op.type)
     (3) output trace file path.
  """

  @staticmethod
  def _match_next_flag(flags, pos):
    """Returns the match for the next TensorTracer flag."""

    match = _FLAG_DOUBLE_QUOTE_PAT.match(flags, pos)
    if match:
      return match
    match = _FLAG_SINGLE_QUOTE_PAT.match(flags, pos)
    if match:
      return match
    match = _FLAG_NO_QUOTE_PAT.match(flags, pos)
    return match

  @staticmethod
  def print_flag_values():
    """Prints all TensorTracer flags passed via environment variables."""

    tensor_tracer_flags = os.environ.get(_FLAGS_ENV_VAR)
    if not tensor_tracer_flags:
      return 'Env variable "%s" is not set'%_FLAGS_ENV_VAR
    result = 'Env variable "%s" is set to "%s"\n'%(_FLAGS_ENV_VAR,
                                                   tensor_tracer_flags)
    result += 'Individual flag value:\n'
    pos = 0
    while True:
      match = TensorTracer._match_next_flag(tensor_tracer_flags, pos)
      if not match:
        break
      flag_name = match.group(1)
      flag_value = match.group(2)
      result += '  %s: %s\n'%(flag_name, flag_value)
      pos = match.end()
    result += '\n'
    return result

  @staticmethod
  def get_flag_value(wanted_flag_name):
    """Returns the value of a TensorTracer flags."""

    tensor_tracer_flags = os.getenv(_FLAGS_ENV_VAR)
    if not tensor_tracer_flags:
      return ''
    pos = 0
    while True:
      match = TensorTracer._match_next_flag(tensor_tracer_flags, pos)
      if not match:
        return ''
      flag_name = match.group(1)
      flag_value = match.group(2)
      if flag_name == wanted_flag_name:
        return flag_value
      pos = match.end()
    return ''

  @staticmethod
  def is_enabled():
    """Returns True if TensorTracer is enabled."""

    flag_value = TensorTracer.get_flag_value(_FLAG_NAME_ENABLE)
    flag_value = flag_value.lower()
    enabled = flag_value in ['1', 't', 'true', 'y', 'yes']
    return enabled

  @staticmethod
  def use_test_undeclared_outputs_dir():
    """Decides the output directory of the trace file.

    Args:
       None.

    Returns:
       True if the output trace file should be written to the
       test-undeclared-outputs-directory defined via an
       env variable.
    """

    flag_value = TensorTracer.get_flag_value(
        _FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR)
    flag_value = flag_value.lower()
    enabled = flag_value in ['1', 't', 'true', 'y', 'yes']
    return enabled

  @staticmethod
  def check_device_type(device_type):
    """Checks if the given device type is valid."""

    if device_type not in [_DEVICE_TYPE_TPU, _DEVICE_TYPE_CPU]:
      raise ValueError('Invalid device_type "%s"'%device_type)

  @staticmethod
  def check_trace_mode(trace_mode):
    """Checks if the given trace mode is valid."""

    valid_trace_modes = [_TRACE_MODE_NAN_INF, _TRACE_MODE_PART_TENSOR,
                         _TRACE_MODE_FULL_TENSOR]
    if trace_mode not in valid_trace_modes:
      raise ValueError('Invalid trace mode "%s" given to the Tensor_Tracer.'
                       'Valid trace modes are: %s'%(trace_mode,
                                                    valid_trace_modes))

  @staticmethod
  def should_trace(device_type, op):
    """Returns True if the given Op should be traced."""

    if device_type != _DEVICE_TYPE_TPU:
      raise ValueError('Non TPU device type is not supported')
    if control_flow_util.IsInCond(op):
      return False
    if op.type in ['Reshape', 'ArgMin', 'ArgMax']:
      return False
    # pylint: disable=protected-access
    return tpu._TPU_REPLICATE_ATTR in op.node_def.attr
    # pylint: enable=protected-access

  @staticmethod
  def reason(op_idx, details):
    """Returns why the Op at op_idx is traced or not."""
    return '%d %s'%(op_idx, details)

  @staticmethod
  def topological_sort(g):
    """Performs topological sort on the given graph.

    Args:
       g: the graph.

    Returns:
       A pair where the first element indicates if the topological
       sort succeeded (True if there is no cycle found; False if a
       cycle is found) and the second element is either the sorted
       list of nodes or the cycle of nodes found.
    """

    def visit(op, cycle, permanently_marked_ops,
              temporarily_marked_ops, sorted_ops):
      """Recursively visits all Ops in a graph.

      Args:
         op: the current Op being visited.
         cycle: a cycle of Ops found.
         permanently_marked_ops: the set of Ops that were already visited.
         temporarily_marked_ops: the set of Ops that we have visited during
                                 the current descent.
         sorted_ops: the list of Ops sorted in topological order.
      """

      if cycle:
        return
      if op in permanently_marked_ops:
        return
      if op in temporarily_marked_ops:
        cycle = temporarily_marked_ops
        return
      temporarily_marked_ops.add(op)
      for i in range(len(op.outputs)):
        out_tensor = op.outputs[i]
        for consumer_op in out_tensor.consumers():
          visit(consumer_op, cycle, permanently_marked_ops,
                temporarily_marked_ops, sorted_ops)
      # pylint: disable=protected-access
      for ctrl_output_op in op._control_outputs:
      # pylint: enable=protected-access
        visit(ctrl_output_op, cycle, permanently_marked_ops,
              temporarily_marked_ops, sorted_ops)
      temporarily_marked_ops.remove(op)
      permanently_marked_ops.add(op)
      sorted_ops.insert(0, op)

    graph_cycle = set([])
    sorted_ops = []
    permanently_marked_ops = set([])
    temporarily_marked_ops = set([])
    unsorted_ops = g.get_operations()
    for op in unsorted_ops:
      visit(op, graph_cycle, permanently_marked_ops,
            temporarily_marked_ops, sorted_ops)
    if graph_cycle:
      return (False, graph_cycle)
    else:
      assert len(unsorted_ops) == len(sorted_ops)
      return (True, sorted_ops)

  def __init__(self):
    """Initializes a TensorTracer.

    Sets the various member fields from the flags (if given) or the defaults.
    """
    self._version = 'use-outside-compilation'
    self._device_type = None
    self._trace_mode = TensorTracer.get_flag_value(_FLAG_NAME_TRACE_MODE)
    if not self._trace_mode:
      self._trace_mode = _TRACE_MODE_NAN_INF
    TensorTracer.check_trace_mode(self._trace_mode)
    self._part_tensor_size = _TRACE_MODE_PART_TENSOR_SIZE
    self._instrument_records = {}
    interesting_ops = TensorTracer.get_flag_value(_FLAG_NAME_INTERESTING_OPS)
    self._selected_ops = interesting_ops.split()
    self._set_trace_file_path()
    self._set_op_range()
    self._num_replicas = None
    self._replica_id = None

  def _add_replica_id_to_graph(self, num_replicas, result_tensor):
    """Adds nodes for computing the replica ID to the graph."""

    if not num_replicas:
      self._replica_id = 'unknown'
      return result_tensor

    self._num_replicas = num_replicas

    with ops.control_dependencies(None):
      # Uses None as dependency to run outside of TPU graph rewrites.
      self._replica_id = tpu_ops.tpu_replicated_input(
          list(range(self._num_replicas)),
          name='tt_replica_id')
    use_replica_id = array_ops.identity(self._replica_id).op
    with ops.control_dependencies([use_replica_id]):
      # Adds a control dependency from the result_tensor to
      # the replica_id to ensure that replica_id will be added to the graph.
      return array_ops.identity(result_tensor)

  def _set_trace_file_path(self):
    """Sets the path of the output trace file."""

    self._trace_file_path = TensorTracer.get_flag_value(_FLAG_NAME_TRACE_FILE)
    if not self._trace_file_path:
      raise ValueError('--%s is not set in the environment variable %s'
                       %(_FLAG_NAME_TRACE_FILE, _FLAGS_ENV_VAR))
    elif TensorTracer.use_test_undeclared_outputs_dir():
      if os.path.isabs(self._trace_file_path):
        raise ValueError('If use_test_undeclared_outputs_dir is set,'
                         'trace_file_path cannot be an absolute path (%s)'
                         %self._trace_file_path)
      outputs_dir = os.environ.get(_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR)
      self._trace_file_path = os.path.join(outputs_dir,
                                           self._trace_file_path)

  def _set_op_range(self):
    """Sets the index range of the Ops that we will consider tracing."""

    op_range = TensorTracer.get_flag_value(_FLAG_NAME_OP_RANGE)
    if not op_range:
      self._op_range = (-1, -1)  # this means including all ops.
      return
    match = _OP_RANGE_PAT.match(op_range)
    if not match:
      self._op_range = (-1, -1)  # this means including all ops.
      return
    self._op_range = (int(match.group(1)), int(match.group(2)))

  def _inside_op_range(self, idx):
    """Return True if the given index is inside the selected range."""

    if idx < self._op_range[0]:
      return False
    return self._op_range[1] < 0 or idx <= self._op_range[1]

  def _write_report(self, content):
    """Writes the given content to the report."""

    logging.info('%s %s'%(_TRACER_LOG_PREFIX, content))

  def _is_selected_op(self, op_name):
    """Returns True if the Op with op_name is selected to be traced."""

    if not self._selected_ops:
      return True
    if op_name in self._selected_ops:
      return True
    return False

  def _write_config_section(self):
    """Writes the config section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_CONFIG))
    self._write_report('%s %s\n'%(_FIELD_NAME_VERSION, self._version))
    self._write_report('%s %s\n'%(_FIELD_NAME_DEVICE, self._device_type))
    self._write_report('%s %s\n'%(_FIELD_NAME_TRACE_MODE, self._trace_mode))
    self._write_report('%s %s\n'%(_FIELD_NAME_NUM_REPLICAS, self._num_replicas))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_CONFIG))

  def _write_reason_section(self):
    """Writes the reason section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_REASON))
    for key in sorted(self._instrument_records):
      self._write_report('"%s" %s\n'%(key, self._instrument_records[key]))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_REASON))

  def _write_op_list_section(self, op_list):
    """Writes the Op-list section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_OP_LIST))
    self._write_report('%s %d\n'%(_FIELD_NAME_NUM_OPS, len(op_list)))
    for i in range(0, len(op_list)):
      self._write_report('%d "%s" %s\n'%(i, op_list[i].name, op_list[i].type))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_OP_LIST))

  def _write_graph_section(self, succeed, sorted_or_cycle):
    """Writes the graph section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_GRAPH))
    self._write_report('%s %s\n'%(_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED,
                                  succeed))
    l = list(sorted_or_cycle)
    for i in range(0, len(l)):
      self._write_report('%d "%s"\n'%(i, l[i].name))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_GRAPH))

  def _make_tensor_trace_fun(self, op_name, output_idx):
    """Makes the tensor tracing function called by outside compilation.

    Args:
      op_name: the name of the Op that outputs the tensor to be traced.
      output_idx: which output of the Op it is (0 means the first output).

    Returns:
      A function to be passed as the first argument to outside compilation.

    Raises:
      RuntimeError: If the trace mode is invalid.
    """

    def _print_tensor(op_name, output_idx, num_elements, tensor, output_tensor):
      """Prints a tensor value to a file.

      Args:
        op_name: the name of the Op that outputs the tensor to be printed.
        output_idx: which output of the Op it is (0 means the first output).
        num_elements: number of elements to print.
        tensor: the tensor needs to be returned.
        output_tensor: the tensor needs to be printed.

      Returns:
        The same tensor passed via the "tensor" argument.
      """
      msg = '"%s:%d" '%(op_name, output_idx)
      output_stream = _OUTPUT_STREAM_ESCAPE + self._trace_file_path
      print_op = logging_ops.print_v2(msg, array_ops.shape(output_tensor),
                                      ' @', self._replica_id,
                                      '\n', output_tensor,
                                      summarize=num_elements,
                                      output_stream=output_stream)
      with ops.control_dependencies([print_op]):
        return array_ops.identity(tensor).op

    def _detect_nan_inf(tensor):
      """Trace function for detecting any NaN/Inf in the tensor."""

      if tensor.dtype.is_floating:
        # Since host can't handle bf16, always convert tensor to f32.
        tensor = math_ops.cast(tensor, dtypes.float32)
        output_tensor = math_ops.reduce_any(
            gen_math_ops.logical_or(gen_math_ops.is_nan(tensor),
                                    gen_math_ops.is_inf(tensor)))
      else:
        output_tensor = constant_op.constant(0)
      return _print_tensor(op_name, output_idx, 1, tensor, output_tensor)

    def _show_global_step(tensor):
      """Trace function for printing the global step count."""

      return _print_tensor(op_name, output_idx, 1, tensor, tensor)

    def _show_part_tensor(tensor):
      """Trace function for printing part of the tensor."""

      return _print_tensor(op_name, output_idx, self._part_tensor_size,
                           tensor, tensor)

    def _show_full_tensor(tensor):
      """Trace function for printing the entire tensor."""

      return _print_tensor(op_name, output_idx, -1, tensor, tensor)

    if op_name == _GLOBAL_STEP_OP_NAME:
      return _show_global_step
    if self._trace_mode == _TRACE_MODE_NAN_INF:
      return _detect_nan_inf
    if self._trace_mode == _TRACE_MODE_PART_TENSOR:
      return _show_part_tensor
    if self._trace_mode == _TRACE_MODE_FULL_TENSOR:
      return _show_full_tensor

    raise RuntimeError('Tensor trace fun for %s is not yet implemented'
                       %self._trace_mode)

  def trace_tpu(self, graph, result_tensor, num_replicas=None):
    """Traces the tensors generated by TPU Ops in a TF graph.

    Args:
      graph: the graph of Ops.
      result_tensor: a result tensor of evaluating the graph.
      num_replicas: number of replicas used on the TPU.

    Returns:
      A tuple (result_tensor_copy, tracing_ops), where:
        result_tensor_copy: an exact copy of result_tensor
        tracing_ops: a list of tracing ops. If this list
                     is non empty, the caller of this function
                     should pose control dependencies upon these
                     Ops so that they will be executed when the
                     graph is evaluated.
    """

    self._device_type = _DEVICE_TYPE_TPU
    TensorTracer.check_device_type(self._device_type)
    result_tensor_copy = self._add_replica_id_to_graph(num_replicas,
                                                       result_tensor)
    self._write_config_section()
    tracing_ops = []
    operations = graph.get_operations()
    self._write_op_list_section(operations)
    # Does the topological sort before adding any nodes to the graph.
    (succeed, sorted_or_cycle) = TensorTracer.topological_sort(graph)
    for op_id, op in enumerate(operations):
      if not self._inside_op_range(op_id):
        self._instrument_records[op.name] = TensorTracer.reason(
            op_id, _RECORD_OUTSIDE_OP_RANGE)
        continue
      if not TensorTracer.should_trace(self._device_type, op):
        self._instrument_records[op.name] = TensorTracer.reason(
            op_id, _RECORD_SHOULD_NOT_TRACE)
        continue
      if not self._is_selected_op(op.name):
        self._instrument_records[op.name] = TensorTracer.reason(
            op_id, _RECORD_FILTERED_OUT)
        continue
      for i in range(len(op.outputs)):
        out_tensor = op.outputs[i]
        if not out_tensor.get_shape().is_fully_defined():
          self._instrument_records[out_tensor.name] = TensorTracer.reason(
              op_id, _RECORD_DYNAMIC_SHAPE)
          continue  # cannot trace tensors with dynamic shape.
        rank = len(out_tensor.shape)
        if rank < 1:
          self._instrument_records[out_tensor.name] = TensorTracer.reason(
              op_id, _RECORD_SCALAR)
          continue  # cannot trace scalar.
        self._instrument_records[out_tensor.name] = TensorTracer.reason(
            op_id, _RECORD_GET_TRACED)
        consumers = out_tensor.consumers()
        trace_op = tpu.outside_compilation(
            self._make_tensor_trace_fun(op.name, i), out_tensor)
        if consumers:
          for consumer_op in consumers:
            # pylint: disable=protected-access
            consumer_op._add_control_input(trace_op)
            # pylint: enable=protected-access
        else:
          # if there is no consumer, we will add the control dependence later
          # when we add the control dependency to the output operations.
          tracing_ops.append(trace_op)

    self._write_reason_section()
    self._write_graph_section(succeed, sorted_or_cycle)

    return (result_tensor_copy, tracing_ops)
