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
import sys

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

_TRACER_LOG_PREFIX = ' [>>>TT>>>]'
_DEVICE_TYPE_TPU = 'tpu'
_DEVICE_TYPE_CPU = 'cpu'
_TRACE_MODE_NAN_INF = 'nan-inf'
_TRACE_MODE_PART_TENSOR = 'part-tensor'
_TRACE_MODE_PART_TENSOR_SIZE = 3
_TRACE_MODE_FULL_TENSOR = 'full-tensor'
_TRACE_MODE_NORM = 'norm'
_TRACE_MODE_MAX_ABS = 'max-abs'
_SUBMODE_BRIEF = 'brief'
_SUBMODE_DETAILED = 'detailed'
_REASON_OUTSIDE_OP_RANGE = 'not-traced-outside-op-range'
_REASON_UNSAFE_OP = 'not-traced-unsafe-op'
_REASON_UNSAFE_SCALAR = 'not-traced-unsafe-scalar'
_REASON_LESS_INTERESTING_OP = 'not-traced-less-interesting-op'
_REASON_DEVICE_MISMATCH = 'not-traced-device-mismatch'
_REASON_DYNAMIC_SHAPE = 'not-traced-dynamic-shape'
_REASON_SCALAR_GET_TRACED = 'traced-scalar'
_REASON_TENSOR_GET_TRACED = 'traced-tensor'
_REASON_USER_INCLUDED = 'traced-user-included'
_REASON_USER_EXCLUDED = 'not-traced-user-excluded'
_REASON_NON_NUMERIC_TENSOR = 'not-traced-non-numeric-tensor'
_MARKER_SECTION_BEGIN = '!!!!!!! section-begin:'
_MARKER_SECTION_END = '!!!!!!! section-end:'
_SECTION_NAME_CONFIG = 'configuration'
_SECTION_NAME_REASON = 'reason'
_SECTION_NAME_OP_LIST = 'op-list'
_SECTION_NAME_TENSOR_LIST = 'tensor-list'
_SECTION_NAME_GRAPH = 'graph'
_FIELD_NAME_VERSION = 'version:'
_FIELD_NAME_DEVICE = 'device:'
_FIELD_NAME_TRACE_MODE = 'trace-mode:'
_FIELD_NAME_SUBMODE = 'submode:'
_FIELD_NAME_NUM_REPLICAS = 'num-replicas:'
_FIELD_NAME_NUM_OPS = 'number-of-ops:'
_FIELD_NAME_NUM_TENSORS = 'number-of-tensors:'
_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED = 'topological-sort-succeed:'
_FLAGS_ENV_VAR = 'TENSOR_TRACER_FLAGS'
_FLAG_SINGLE_QUOTE_PAT = re.compile(r"\s*--([^=]+)='([^']*)'")
_FLAG_DOUBLE_QUOTE_PAT = re.compile(r'\s*--([^=]+)="([^"]*)"')
_FLAG_NO_QUOTE_PAT = re.compile(r'\s*--([^=]+)=(\S*)')
_FLAG_NO_EQUAL_PAT = re.compile(r'\s*--([^=]+)\s*')
_FLAG_NAME_ENABLE = 'enable'
_FLAG_NAME_TRACE_MODE = 'trace_mode'
_FLAG_NAME_SUBMODE = 'submode'
_FLAG_NAME_INCLUDE_LESS_INTERESTING_OPS = 'include_less_interesting_ops'
_FLAG_NAME_EXCLUDED_OPNAMES = 'excluded_opnames'
_FLAG_NAME_EXCLUDED_OPTYPES = 'excluded_optypes'
_FLAG_NAME_INCLUDED_OPNAMES = 'included_opnames'
_FLAG_NAME_INCLUDED_OPTYPES = 'included_optypes'
_FLAG_NAME_TRACE_FILE = 'trace_file_path'
_FLAG_NAME_REPORT_FILE = 'report_file_path'
_FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR = 'use_test_undeclared_outputs_dir'
_FLAG_NAME_OP_RANGE = 'op_range'
_OP_RANGE_PAT = re.compile(r'(\d+):(\d+)')
_OUTPUT_STREAM_ESCAPE = 'file://'
_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR = 'TEST_UNDECLARED_OUTPUTS_DIR'
_TENSOR_TRACER_COLLECTION = 'tensor_tracer_variables'
_TENSOR_TRACER_CHECKPOINT = 'tensor_tracer_checkpoint'


def tensor_tracepoint(tensor, checkpoint_name):
  """Adds a checkpoint with the given checkpoint name for the given tensor.

  The tensor will be added to the list of tensors that will be traced by the
  tensor tracer.

  Args:
     tensor: the tensor object for which the tracing is requested.
     checkpoint_name: a string name for the checkpoint. This name has to be a
     unique name if used within model comparison. The tensors that have the same
     checkpoint identifier is compared in model comparison.
  Returns:
    The provided tensor.
  """

  tensor.graph.get_collection(_TENSOR_TRACER_COLLECTION)
  tensor.graph.add_to_collection(_TENSOR_TRACER_COLLECTION,
                                 (tensor, checkpoint_name))
  return tensor


def keras_layer_tracepoint(layer, checkpoint_name):
  """An interface for adding the tensor outputs of a keras layer.

  Encapsulates tensor_tracepoint.

  Args:
     layer: A keras layer.
     checkpoint_name: a string name for the checkpoint. This name has to be a
     unique name if used within model comparison. The tensors that have the same
     checkpoint identifier is compared in model comparison.

  Returns:
    The provided layer.
  """
  try:
    outputs = layer.output
    if tensor_util.is_tensor(outputs):
      tensor_tracepoint(outputs, '%s' % (checkpoint_name))
    else:
      idx = 0
      for output_tensor in outputs:
        if tensor_util.is_tensor(outputs):
          tensor_tracepoint(output_tensor, '%s_%d' % (checkpoint_name, idx))
        idx += 1
  except AttributeError:
    pass
  except RuntimeError:
    pass
  return layer


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
    """Returns the match for the next TensorTracer flag.

    Args:
       flags: a string that contains the flags.
       pos: where in flags to start the search.

    Returns:
       A pair where the first element is the regular-expression
       match found and the second element indicates if the match
       has a value.
    """

    match = _FLAG_DOUBLE_QUOTE_PAT.match(flags, pos)
    if match:
      return match, True
    match = _FLAG_SINGLE_QUOTE_PAT.match(flags, pos)
    if match:
      return match, True
    match = _FLAG_NO_QUOTE_PAT.match(flags, pos)
    if match:
      return match, True
    match = _FLAG_NO_EQUAL_PAT.match(flags, pos)
    if match:
      # The flag is found but is not given a value.
      return match, False
    # The flag is not found.
    return None, False

  @staticmethod
  def validate_flag_names():
    """Validates if the TensorTrace flags passed are valid."""
    valid_flag_names = [_FLAG_NAME_ENABLE, _FLAG_NAME_TRACE_MODE,
                        _FLAG_NAME_SUBMODE,
                        _FLAG_NAME_EXCLUDED_OPNAMES,
                        _FLAG_NAME_EXCLUDED_OPTYPES,
                        _FLAG_NAME_INCLUDED_OPNAMES,
                        _FLAG_NAME_INCLUDED_OPTYPES,
                        _FLAG_NAME_TRACE_FILE, _FLAG_NAME_REPORT_FILE,
                        _FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR,
                        _FLAG_NAME_INCLUDE_LESS_INTERESTING_OPS,
                        _FLAG_NAME_OP_RANGE]
    tensor_tracer_flags = os.environ.get(_FLAGS_ENV_VAR)
    if not tensor_tracer_flags:
      return
    pos = 0
    while True:
      match, _ = TensorTracer._match_next_flag(tensor_tracer_flags, pos)
      if not match:
        break
      flag_name = match.group(1)
      if flag_name not in valid_flag_names:
        raise ValueError(
            'The flag name "%s" passed via the environment variable "%s" '
            'is invalid. Valid flag names are:'
            '\n%s'%(flag_name, _FLAGS_ENV_VAR, valid_flag_names))
      pos = match.end()

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
      match, has_value = TensorTracer._match_next_flag(
          tensor_tracer_flags, pos)
      if not match:
        break
      flag_name = match.group(1)
      if has_value:
        flag_value = match.group(2)
      else:
        flag_value = None
      result += '  %s: %s\n'%(flag_name, flag_value)
      pos = match.end()
    result += '\n'
    return result

  @staticmethod
  def get_flag_value(wanted_flag_name):
    """Returns the value of a TensorTracer flags.

    Args:
      wanted_flag_name: the name the the flag we are looking for.

    Returns:
      A pair where the first element indicates if the flag is
      found and the second element is the value of the flag.

    Raises:
      RuntimeError: If supposedly deadcode is reached.
    """

    tensor_tracer_flags = os.getenv(_FLAGS_ENV_VAR)
    if not tensor_tracer_flags:
      return False, None
    pos = 0
    while True:
      match, has_value = TensorTracer._match_next_flag(
          tensor_tracer_flags, pos)
      if not match:
        return False, None
      flag_name = match.group(1)
      if has_value:
        flag_value = match.group(2)
      else:
        flag_value = None
      if flag_name == wanted_flag_name:
        return True, flag_value
      pos = match.end()
    raise RuntimeError('Should not reach here.')

  @staticmethod
  def flag_value_to_re_list(flag_name):
    """Converts list of strings to compiled RE."""

    re_list = []
    found, flag_value = TensorTracer.get_flag_value(flag_name)
    if not found or not flag_value:
      return re_list
    list_of_values = flag_value.split()
    for v in list_of_values:
      r = re.compile(v)
      re_list.append(r)
    return re_list

  @staticmethod
  def _is_flag_on(flag_name):
    """Returns True if the given flag is on."""

    found, flag_value = TensorTracer.get_flag_value(flag_name)
    if not found:
      return False
    if flag_value is None:
      return True
    # Depends on the flag value.
    flag_value = flag_value.lower()
    enabled = flag_value in ['1', 't', 'true', 'y', 'yes']
    return enabled

  @staticmethod
  def is_enabled():
    """Returns True if TensorTracer is enabled."""

    return TensorTracer._is_flag_on(_FLAG_NAME_ENABLE)

  @staticmethod
  def use_test_undeclared_outputs_dir():
    """Decides the output directory of the report and trace files.

    Args:
       None.

    Returns:
       True if the output files should be written to the
       test-undeclared-outputs-directory defined via an
       env variable.
    """

    return TensorTracer._is_flag_on(
        _FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR)


  @staticmethod
  def check_device_type(device_type):
    """Checks if the given device type is valid."""

    if device_type not in [_DEVICE_TYPE_TPU, _DEVICE_TYPE_CPU]:
      raise ValueError('Invalid device_type "%s"'%device_type)

  @staticmethod
  def check_trace_mode(trace_mode):
    """Checks if the given trace mode is valid."""

    valid_trace_modes = [_TRACE_MODE_NAN_INF, _TRACE_MODE_PART_TENSOR,
                         _TRACE_MODE_FULL_TENSOR, _TRACE_MODE_NORM,
                         _TRACE_MODE_MAX_ABS]
    if trace_mode not in valid_trace_modes:
      raise ValueError('Invalid trace mode "%s" given to the Tensor_Tracer.'
                       'Valid trace modes are: %s'%(trace_mode,
                                                    valid_trace_modes))

  @staticmethod
  def check_submode(submode):
    """Checks if the given submode is valid."""

    if not submode:
      return
    valid_submodes = [_SUBMODE_DETAILED, _SUBMODE_BRIEF]
    if submode not in valid_submodes:
      raise ValueError('Invalid submode "%s" given to the Tensor_Tracer.'
                       'Valid submodes are: %s'%(submode,
                                                 valid_submodes))

  @staticmethod
  def unsafe_op(op):
    """Returns True if this op is not safe to be traced."""

    if control_flow_util.IsInCond(op):
      return True
    # Reasons for not including following op types:
    #    Assign: cause incorrect result with CPU tracing.
    #    others: compilation problems.
    # TODO(deveci): Check if 'Pack', 'Shape', 'Reshape', 'ArgMin', 'ArgMax'
    #               are still unsafe now that we have handled int64 tensors.
    if op.type in ['Assign', 'Pack', 'Shape', 'Reshape', 'ArgMin', 'ArgMax']:
      return True
    return False

  @staticmethod
  def device_mismatch(device_type, op):
    if device_type == _DEVICE_TYPE_TPU:
      # pylint: disable=protected-access
      return tpu._TPU_REPLICATE_ATTR not in op.node_def.attr
      # pylint: enable=protected-access
    return False

  @staticmethod
  def unsafe_scalar_trace(op):
    """Return true if scalar output tensor from Op is not safe to be traced."""

    # Tracing the following causes cycle in the graph on TPU.
    if op.type in ['LoopCond', 'Enter', 'Merge', 'Const',
                   'Switch', 'Less', 'ReadVariableOp']:
      return True
    # Tracing the following will cause casting-issue
    # with the norm tracing mode or other compilation issues on CPU.
    if op.type in ['VarHandleOp', 'IteratorToStringHandle',
                   'IteratorGetNext', 'OneShotIterator',
                   'IteratorV2', 'MakeIterator',
                   'BatchDatasetV2', 'MapDataset',
                   'FixedLengthRecordDataset', 'TakeDataset', 'ZipDataset',
                   'Placeholder', 'PlaceholderWithDefault', 'StridedSlice']:
      return True
    return False

  @staticmethod
  def less_interesting_op(op):
    """Returns True if the given Op is not an interesting one to be traced."""

    found, _ = TensorTracer.get_flag_value(
        _FLAG_NAME_INCLUDE_LESS_INTERESTING_OPS)
    if found:
      # users force to include all ops.
      return False
    # Following ops are highly unlikey to cause bugs.
    return op.type in ['Const', 'Identity', 'Cast', 'Shape']

  @staticmethod
  def reason(op_idx, details):
    """Returns reason why the Op at op_idx is traced or not."""

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

  @staticmethod
  def _make_op_and_tensor_maps(op_list):
    """Creates various maps and lists from op_list.

    Args:
       op_list: a list of Ops

    Returns:
       opname_idx_map: a map from Op's name to its index in op_list.
       tensor_list: a list of output tensors of the Ops in op_list.
       tensorname_idx_map: a map from output tensor name to its index
                           in tensor_list.
    """

    opname_idx_map = {}
    tensor_list = []
    tensorname_idx_map = {}
    for op_id, op in enumerate(op_list):
      if op.name in opname_idx_map:
        raise ValueError('Duplicated Op name: %s'%op.name)
      opname_idx_map[op.name] = op_id
      for output_tensor in op.outputs:
        if output_tensor.name not in tensorname_idx_map:
          tensor_list.append(output_tensor)
          tensorname_idx_map[output_tensor.name] = len(tensor_list)-1
    return (opname_idx_map, tensor_list, tensorname_idx_map)

  def __init__(self):
    """Initializes a TensorTracer.

    Sets the various member fields from the flags (if given) or the defaults.
    """
    self._version = 'use-outside-compilation'
    self._device_type = None
    TensorTracer.validate_flag_names()
    found, self._trace_mode = TensorTracer.get_flag_value(_FLAG_NAME_TRACE_MODE)
    if not found or not self._trace_mode:
      self._trace_mode = _TRACE_MODE_NAN_INF
    TensorTracer.check_trace_mode(self._trace_mode)
    found, self._submode = TensorTracer.get_flag_value(_FLAG_NAME_SUBMODE)
    if not found or not self._submode:
      self._submode = _SUBMODE_DETAILED
    TensorTracer.check_submode(self._submode)
    self._part_tensor_size = _TRACE_MODE_PART_TENSOR_SIZE
    self._instrument_records = {}
    self._set_trace_file_path()
    self._set_report_file()
    self._set_op_range()
    self._set_excluded_opnames()
    self._set_excluded_optypes()
    self._set_included_opnames()
    self._set_included_optypes()
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

    found, self._trace_file_path = TensorTracer.get_flag_value(
        _FLAG_NAME_TRACE_FILE)
    if found and self._trace_file_path \
       and TensorTracer.use_test_undeclared_outputs_dir():
      if os.path.isabs(self._trace_file_path):
        raise ValueError('If use_test_undeclared_outputs_dir is set,'
                         'trace_file_path cannot be an absolute path (%s)'
                         %self._trace_file_path)
      outputs_dir = os.environ.get(_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR)
      self._trace_file_path = os.path.join(outputs_dir,
                                           self._trace_file_path)

  def _set_report_file(self):
    """Sets the path of the output report file."""

    found, self._report_file_path = TensorTracer.get_flag_value(
        _FLAG_NAME_REPORT_FILE)
    if found and self._report_file_path \
       and TensorTracer.use_test_undeclared_outputs_dir():
      if os.path.isabs(self._report_file_path):
        raise ValueError('If use_test_undeclared_outputs_dir is set,'
                         'report_file_path cannot be an absolute path (%s)'
                         %self._report_file_path)
      outputs_dir = os.environ.get(_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR)
      self._report_file_path = os.path.join(outputs_dir,
                                            self._report_file_path)
    if not self._report_file_path:
      self._report_file = None
      return
    try:
      self._report_file = gfile.Open(self._report_file_path, 'w')
    except IOError as e:
      raise e

  def _close_report_file(self):
    if self._report_file:
      self._report_file.close()

  def _set_op_range(self):
    """Sets the index range of the Ops that we will consider tracing."""

    found, op_range = TensorTracer.get_flag_value(_FLAG_NAME_OP_RANGE)
    if not found or not op_range:
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

  def _set_excluded_opnames(self):
    self._excluded_opname_re_list = TensorTracer.flag_value_to_re_list(
        _FLAG_NAME_EXCLUDED_OPNAMES)

  def _set_excluded_optypes(self):
    self._excluded_optype_re_list = TensorTracer.flag_value_to_re_list(
        _FLAG_NAME_EXCLUDED_OPTYPES)

  def _set_included_opnames(self):
    self._included_opname_re_list = TensorTracer.flag_value_to_re_list(
        _FLAG_NAME_INCLUDED_OPNAMES)

  def _set_included_optypes(self):
    self._included_optype_re_list = TensorTracer.flag_value_to_re_list(
        _FLAG_NAME_INCLUDED_OPTYPES)

  def _is_user_included_op(self, op):
    for opname_re in self._included_opname_re_list:
      if opname_re.match(op.name):
        return True
    for optype_re in self._included_optype_re_list:
      if optype_re.match(op.type):
        return True
    return False

  def _is_user_excluded_op(self, op):
    for opname_re in self._excluded_opname_re_list:
      if opname_re.match(op.name):
        return True
    for optype_re in self._excluded_optype_re_list:
      if optype_re.match(op.type):
        return True
    return False

  def _write_report(self, content):
    """Writes the given content to the report."""

    line = '%s %s'%(_TRACER_LOG_PREFIX, content)
    if self._report_file:
      self._report_file.write(line)
    else:
      logging.info(line)

  def _write_config_section(self):
    """Writes the config section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_CONFIG))
    self._write_report('%s %s\n'%(_FIELD_NAME_VERSION, self._version))
    self._write_report('%s %s\n'%(_FIELD_NAME_DEVICE, self._device_type))
    self._write_report('%s %s\n'%(_FIELD_NAME_TRACE_MODE, self._trace_mode))
    self._write_report('%s %s\n'%(_FIELD_NAME_SUBMODE, self._submode))
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
      op = op_list[i]
      line = '%d "%s" %s'%(i, op.name, op.type)
      for out_tensor in op.outputs:
        if out_tensor.name not in self._tensorname_idx_map:
          raise ValueError(
              'out_tensor %s is not in tensorname_idx_map'%out_tensor.name)
        line += ' %d'%self._tensorname_idx_map[out_tensor.name]
      line += '\n'
      self._write_report(line)
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_OP_LIST))

  def _write_tensor_list_section(self, tensor_list, opname_idx_map):
    """Writes the tensor-list section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN,
                                  _SECTION_NAME_TENSOR_LIST))
    self._write_report('%s %d\n'%(_FIELD_NAME_NUM_TENSORS, len(tensor_list)))
    for i in range(0, len(tensor_list)):
      tensor = tensor_list[i]
      line = '%d "%s"'%(i, tensor.name)
      for consumer_op in tensor.consumers():
        if consumer_op.name not in opname_idx_map:
          raise ValueError(
              'consumer_op %s is not in opname_idx_map'%consumer_op.name)
        line += ' %d'%opname_idx_map[consumer_op.name]
      line += '\n'
      self._write_report(line)
    self._write_report('%s %s\n'%(_MARKER_SECTION_END,
                                  _SECTION_NAME_TENSOR_LIST))

  def _write_graph_section(self, succeed, sorted_or_cycle):
    """Writes the graph section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_GRAPH))
    self._write_report('%s %s\n'%(_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED,
                                  succeed))
    l = list(sorted_or_cycle)
    for i in range(0, len(l)):
      self._write_report('%d "%s"\n'%(i, l[i].name))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_GRAPH))

  def _make_tensor_trace_fun(self, tensor_name):
    """Makes the tensor tracing function called by outside compilation.

    Args:
      tensor_name: name of the tensor being traced.

    Returns:
      A function to be passed as the first argument to outside compilation.

    Raises:
      RuntimeError: If the trace mode is invalid.
    """

    def _print_tensor(tensor_name, num_elements, tensor, output_tensor):
      """Prints a tensor value to a file.

      Args:
        tensor_name: name of the tensor being traced.
        num_elements: number of elements to print (-1 means print all).
        tensor: the tensor needs to be returned.
        output_tensor: the tensor needs to be printed.

      Returns:
        The same tensor passed via the "tensor" argument.

      Raises:
        ValueError: If tensor_name is not already in
                    self._tensorname_idx_map.
      """

      if self._submode == _SUBMODE_BRIEF:
        if tensor_name not in self._tensorname_idx_map:
          raise ValueError(
              'Tensor name %s is not in the tensorname_idx_map'%tensor_name)
        msg = '%d'%self._tensorname_idx_map[tensor_name]
      else:
        msg = '"%s"'%tensor_name

      if self._trace_file_path:
        output_stream = _OUTPUT_STREAM_ESCAPE + self._trace_file_path
      else:
        output_stream = sys.stderr
      print_op = logging_ops.print_v2(msg, array_ops.shape(output_tensor),
                                      '@', self._replica_id,
                                      '\n', output_tensor, '\n',
                                      summarize=num_elements,
                                      output_stream=output_stream)
      with ops.control_dependencies([print_op]):
        return array_ops.identity(tensor).op

    def _detect_nan_inf(tensor):
      """Trace function for detecting any NaN/Inf in the tensor."""

      if tensor.dtype.is_floating:
        output_tensor = math_ops.reduce_any(
            gen_math_ops.logical_or(gen_math_ops.is_nan(tensor),
                                    gen_math_ops.is_inf(tensor)))
      else:
        output_tensor = constant_op.constant(False)

      return _print_tensor(tensor_name, -1, tensor, output_tensor)

    def _show_norm(tensor):
      tensor = math_ops.cast(tensor, dtypes.float64)
      output_tensor = linalg_ops.norm(tensor)
      return _print_tensor(tensor_name, -1, tensor, output_tensor)

    def _show_max_abs(tensor):
      output_tensor = math_ops.cast(math_ops.reduce_max(math_ops.abs(tensor)),
                                    dtypes.float64)
      zero = constant_op.constant(0, dtypes.float64)
      output_tensor = gen_math_ops.maximum(zero, output_tensor)
      return _print_tensor(tensor_name, -1, tensor, output_tensor)

    def _show_part_tensor(tensor):
      """Trace function for printing part of the tensor."""

      return _print_tensor(tensor_name, self._part_tensor_size,
                           tensor, tensor)

    def _show_full_tensor(tensor):
      """Trace function for printing the entire tensor."""

      return _print_tensor(tensor_name, -1, tensor, tensor)

    if self._trace_mode == _TRACE_MODE_NAN_INF:
      return _detect_nan_inf
    if self._trace_mode == _TRACE_MODE_PART_TENSOR:
      return _show_part_tensor
    if self._trace_mode == _TRACE_MODE_FULL_TENSOR:
      return _show_full_tensor
    if self._trace_mode == _TRACE_MODE_NORM:
      return _show_norm
    if self._trace_mode == _TRACE_MODE_MAX_ABS:
      return _show_max_abs

    raise RuntimeError('Tensor trace fun for %s is not yet implemented'
                       %self._trace_mode)

  def _skip_op(self, op_id, op, user_included, user_excluded):
    """Returns True if we should not trace Op."""

    if user_included:
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_USER_INCLUDED)
      return False
    if user_excluded:
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_USER_EXCLUDED)
      return True
    if not self._inside_op_range(op_id):
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_OUTSIDE_OP_RANGE)
      return True
    if TensorTracer.unsafe_op(op):
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_UNSAFE_OP)
      return True
    if TensorTracer.device_mismatch(self._device_type, op):
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_DEVICE_MISMATCH)
      return True
    if TensorTracer.less_interesting_op(op):
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_LESS_INTERESTING_OP)
      return True
    return False

  def _skip_tensor(self, op_id, out_tensor, user_included,
                   user_excluded):
    """Returns True if we should not trace out_tensor."""

    # Skips a tensor if the tensor has a non-numeric type.
    #   Note: we cannot use check_ops.is_numeric_tensor(out_tensor)
    #         because it also excludes tensors with dtypes, bool, and
    #         float32_ref, which we actually want to trace.
    non_numeric_tensor_types = set([dtypes.variant, dtypes.resource,
                                    dtypes.string])
    if out_tensor.dtype in non_numeric_tensor_types:
      self._instrument_records[out_tensor.name] = TensorTracer.reason(
          op_id, _REASON_NON_NUMERIC_TENSOR)
      return True

    if user_included:
      self._instrument_records[out_tensor.name] = TensorTracer.reason(
          op_id, _REASON_USER_INCLUDED)
      return False
    if user_excluded:
      self._instrument_records[out_tensor.name] = TensorTracer.reason(
          op_id, _REASON_USER_EXCLUDED)
      return True
    if not out_tensor.get_shape().is_fully_defined():
      self._instrument_records[out_tensor.name] = TensorTracer.reason(
          op_id, _REASON_DYNAMIC_SHAPE)
      return True
    rank = len(out_tensor.shape)
    if rank < 1:
      # scalar
      if TensorTracer.unsafe_scalar_trace(out_tensor.op):
        self._instrument_records[out_tensor.name] = TensorTracer.reason(
            op_id, _REASON_UNSAFE_SCALAR)
        return True
      else:
        self._instrument_records[out_tensor.name] = TensorTracer.reason(
            op_id, _REASON_SCALAR_GET_TRACED)
        return False
    else:
      # tensor
      self._instrument_records[out_tensor.name] = TensorTracer.reason(
          op_id, _REASON_TENSOR_GET_TRACED)
      return False

  def _pre_tracing(self, graph):
    """Work needs to be done prior to TPU or CPU tracing."""

    operations = graph.get_operations()
    (opname_idx_map, tensor_list, self._tensorname_idx_map) = (
        TensorTracer._make_op_and_tensor_maps(operations))
    self._write_config_section()
    self._write_op_list_section(operations)
    self._write_tensor_list_section(tensor_list, opname_idx_map)
    # Does the topological sort before adding any nodes to the graph.
    (succeed, sorted_or_cycle) = TensorTracer.topological_sort(graph)
    return (operations, succeed, sorted_or_cycle)

  def _post_tracing(self, succeed, sorted_or_cycle):
    """Work needs to be done after TPU or CPU tracing."""

    self._write_reason_section()
    self._write_graph_section(succeed, sorted_or_cycle)
    self._close_report_file()

  def _get_checkpoints(self, graph):
    """Returns the list of Ops that produce the tensors traced with API.

    Args:
      graph: the graph of Ops.

    Returns:
      A set of operation names which should be traced.
    """

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN,
                                  _TENSOR_TRACER_CHECKPOINT))
    checkpoint_operations = set()
    tensor_tracer_variables = graph.get_collection(_TENSOR_TRACER_COLLECTION)
    for (tensor, checkpoint_name) in tensor_tracer_variables:
      self._write_report('%s %s\n'%(tensor.name, checkpoint_name))
      checkpoint_operations.add(tensor.op.name)
    self._write_report('%s %s\n'%(_MARKER_SECTION_END,
                                  _TENSOR_TRACER_CHECKPOINT))
    return checkpoint_operations

  def trace_tpu(self, graph, result_tensor, num_replicas=None):
    """Traces the tensors generated by TPU Ops in a TF graph.

    Args:
      graph: the graph of Ops executed on the TPU.
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

    def _cast_unsupported_dtypes(tensor):
      """Casts tensor to a supported type."""

      if tensor.dtype.__eq__(dtypes.int64):
        # outside-compilation doesn't support int64 input yet.
        return math_ops.cast(tensor, dtypes.int32)
      if tensor.dtype.__eq__(dtypes.bfloat16) or tensor.dtype.__eq__(
          dtypes.float16):
        # Since host can't handle bf16, convert tensor to f32.
        return math_ops.cast(tensor, dtypes.float32)
      return tensor

    self._device_type = _DEVICE_TYPE_TPU
    TensorTracer.check_device_type(self._device_type)
    result_tensor_copy = self._add_replica_id_to_graph(num_replicas,
                                                       result_tensor)
    (operations, succeed, sorted_or_cycle) = self._pre_tracing(graph)
    tracing_ops = []
    checkpoint_operations = self._get_checkpoints(graph)

    for op_id, op in enumerate(operations):
      if checkpoint_operations and op.name not in checkpoint_operations:
        continue
      user_included = self._is_user_included_op(op)
      user_excluded = self._is_user_excluded_op(op)
      if self._skip_op(op_id, op, user_included, user_excluded):
        continue
      for i in range(len(op.outputs)):
        out_tensor = op.outputs[i]
        if self._skip_tensor(op_id, out_tensor, user_included,
                             user_excluded):
          continue
        consumers = out_tensor.consumers()
        tensor_name = out_tensor.name
        out_tensor = _cast_unsupported_dtypes(out_tensor)
        trace_op = tpu.outside_compilation(
            self._make_tensor_trace_fun(tensor_name), out_tensor)
        if consumers:
          for consumer_op in consumers:
            # pylint: disable=protected-access
            consumer_op._add_control_input(trace_op)
            # pylint: enable=protected-access
        else:
          # if there is no consumer, we will add the control dependence later
          # when we add the control dependency to the output operations.
          tracing_ops.append(trace_op)
    self._post_tracing(succeed, sorted_or_cycle)
    return (result_tensor_copy, tracing_ops)

  def trace_cpu(self, graph):
    """Traces the tensors generated by CPU Ops in a TF graph.

    Args:
      graph: the graph of Ops executed on the CPU.

    Returns:
      tracing_calls: a map from keys to trace calls.
                     A key is constructed from an Op's name.
                     A trace call consists of a function and a tensor (
                     the function will be invoked with the tensor).
    """

    self._device_type = _DEVICE_TYPE_CPU
    TensorTracer.check_device_type(self._device_type)
    self._num_replicas = 1
    self._replica_id = 0
    (operations, succeed, sorted_or_cycle) = self._pre_tracing(graph)
    tracing_calls = {}
    checkpoint_operations = self._get_checkpoints(graph)

    for op_id, op in enumerate(operations):
      if checkpoint_operations and op.name not in checkpoint_operations:
        continue
      user_included = self._is_user_included_op(op)
      user_excluded = self._is_user_excluded_op(op)
      if self._skip_op(op_id, op, user_included, user_excluded):
        continue
      for i in range(len(op.outputs)):
        out_tensor = op.outputs[i]
        if self._skip_tensor(op_id, out_tensor, user_included,
                             user_excluded):
          continue
        trace_fun = self._make_tensor_trace_fun(out_tensor.name)
        trace_call = (trace_fun, [out_tensor])
        trace_call_key = 'tensor_tracing_cpu-%s:%d'%(op.name, i)
        tracing_calls[trace_call_key] = trace_call
    self._post_tracing(succeed, sorted_or_cycle)
    return tracing_calls
