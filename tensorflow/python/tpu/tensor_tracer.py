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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu.ops import tpu_ops

_TRACER_LOG_PREFIX = ' [>>>TT>>>]'
_DEVICE_TYPE_TPU = 'tpu'
_DEVICE_TYPE_CPU = 'cpu'
_TRACE_MODE_NAN_INF = 'nan-inf'
_TRACE_MODE_PART_TENSOR = 'part-tensor'
_TRACE_MODE_PART_TENSOR_SIZE = 3
_TRACE_MODE_FULL_TENSOR = 'full-tensor'
_TRACE_MODE_FULL_IF_NAN = 'trace-back-if-nan'
_FLAG_NAME_TRACE_STACK_SIZE = 'trace_stack_size'
_TRACE_MODE_NORM = 'norm'
_TRACE_MODE_MAX_ABS = 'max-abs'
_SUBMODE_BRIEF = 'brief'
_SUBMODE_DETAILED = 'detailed'
_REASON_OUTSIDE_OP_RANGE = 'not-traced-outside-op-range'
_REASON_UNSAFE_OP = 'not-traced-unsafe-op'
_REASON_WHILELOOP_OP = 'not-traced-special-whileloop-op'
_REASON_UNSAFE_SCALAR = 'not-traced-unsafe-scalar'
_REASON_SKIP_SCALAR = 'not-traced-scalar'
_REASON_LESS_INTERESTING_OP = 'not-traced-less-interesting-op'
_REASON_DEVICE_MISMATCH = 'not-traced-device-mismatch'
_REASON_DYNAMIC_SHAPE = 'not-traced-dynamic-shape'
_REASON_SCALAR_GET_TRACED = 'traced-scalar'
_REASON_TENSOR_GET_TRACED = 'traced-tensor'
_REASON_USER_INCLUDED = 'traced-user-included'
_REASON_USER_EXCLUDED = 'not-traced-user-excluded'
_REASON_NOT_EXECUTED = 'not-traced-not-in-exec-path'
_REASON_NON_NUMERIC_TENSOR = 'not-traced-non-numeric-tensor'
_REASON_FEEDS_WHILELOOP_OP = 'not-traced-feeds-special-whileloop-op'
_MARKER_SECTION_BEGIN = '!!!!!!! section-begin:'
_MARKER_SECTION_END = '!!!!!!! section-end:'
_SECTION_NAME_CONFIG = 'configuration'
_SECTION_NAME_REASON = 'reason'
_SECTION_NAME_OP_LIST = 'op-list'
_SECTION_NAME_TENSOR_LIST = 'tensor-list'
_SECTION_NAME_CACHE_INDEX_MAP = 'cache-index-map'
_SECTION_NAME_GRAPH = 'graph'
_FIELD_NAME_VERSION = 'version:'
_FIELD_NAME_DEVICE = 'device:'
_FIELD_NAME_TRACE_MODE = 'trace-mode:'
_FIELD_NAME_SUBMODE = 'submode:'
_FIELD_NAME_NUM_REPLICAS = 'num-replicas:'
_FIELD_NAME_NUM_REPLICAS_PER_HOST = 'num-replicas-per-host:'
_FIELD_NAME_NUM_HOSTS = 'num-hosts:'
_FIELD_NAME_NUM_OPS = 'number-of-ops:'
_FIELD_NAME_NUM_TENSORS = 'number-of-tensors:'
_FIELD_NAME_NUM_CACHE_INDICES = 'number-of-indices:'
_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED = 'topological-sort-succeed:'
_FLAGS_ENV_VAR = 'TENSOR_TRACER_FLAGS'
_FLAG_SINGLE_QUOTE_PAT = re.compile(r"\s*--([^=]+)='([^']*)'")
_FLAG_DOUBLE_QUOTE_PAT = re.compile(r'\s*--([^=]+)="([^"]*)"')
_FLAG_NO_QUOTE_PAT = re.compile(r'\s*--([^=]+)=(\S*)')
_FLAG_NO_EQUAL_PAT = re.compile(r'\s*--([^=]+)\s*')
_FLAG_NAME_ENABLE = 'enable'
_FLAG_NAME_TRACE_MODE = 'trace_mode'
_FLAG_NAME_USE_COMPACT_TRACE = 'compact_trace'
_FLAG_NAME_TRACE_SCALAR_OPS = 'trace_scalar'
_FLAG_NAME_TRACE_BEFORE_OPS = 'trace_before_included_ops'
_FLAG_NAME_TRACE_AFTER_OPS = 'trace_after_included_ops'
_FLAG_NAME_SUBMODE = 'submode'
_FLAG_NAME_INCLUDE_LESS_INTERESTING_OPS = 'include_less_interesting_ops'
_FLAG_NAME_EXCLUDED_OPNAMES = 'excluded_opnames'
_FLAG_NAME_EXCLUDED_OPTYPES = 'excluded_optypes'
_FLAG_NAME_INCLUDED_OPNAMES = 'included_opnames'
_FLAG_NAME_INCLUDED_OPTYPES = 'included_optypes'
_FLAG_NAME_TRACE_DIR = 'trace_dir'
_FLAG_NAME_REPORT_FILE = 'report_file'
_FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR = 'use_test_undeclared_outputs_dir'
_FLAG_NAME_OP_RANGE = 'op_range'
# Folder to dump the pre (before tensor tracer updates) and post graphs (after
# tensor tracer updates).
_FLAG_DUMP_BEFORE_AFTER_GRAPHS = 'dump_graphs'
_OP_RANGE_PAT = re.compile(r'(\d+):(\d+)')
_OUTPUT_STREAM_ESCAPE = 'file://'
_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR = 'TEST_UNDECLARED_OUTPUTS_DIR'
_TENSOR_TRACER_COLLECTION = 'tensor_tracer_variables'
_TENSOR_TRACER_CHECKPOINT = 'tensor_tracer_checkpoint'
_TRACE_FILE_NAME = 'trace.all'
_COMPACT_TRACE_FILE_PREFIX = 'compact_trace.'
_COMPACT_TRACE_ENTRY_INIT_VALUE = -1.0
_TENSOR_TRACER_STORAGE = 'tensor_tracer_storage'
_TENSOR_VALUES_CACHE = 'tensor_values_cache'
_REPLICA_ID_TAG = '#replica-id: '


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


def _trace_files_need_precreated(output_dir):
  """Return True if trace files must be pre-created by users."""

  if not output_dir.startswith('/'):
    return False
  if len(output_dir) < 5:
    return False
  if output_dir[2] != 'n':
    return False
  if output_dir[3] != 's':
    return False
  if output_dir[1] != 'c':
    return False
  if output_dir[4] != '/':
    return False
  return True


def _get_tensor_values_cache(graph=None):
  """Returns the variable that implements tensor-value caching."""

  graph = graph or ops.get_default_graph()
  collection = graph.get_collection(_TENSOR_TRACER_STORAGE)
  if len(collection) == 1:
    return collection[0]
  elif not collection:
    raise RuntimeError('%s has not been created'%_TENSOR_VALUES_CACHE)
  else:
    raise RuntimeError('Multiple %s created'%_TENSOR_VALUES_CACHE)
  return None


def _create_tensor_values_cache(graph, num_tensors):
  """Creates a variable as the cache to store intermediate tensor values."""
  graph = graph or ops.get_default_graph()
  # Create in proper graph and base name_scope.
  with graph.as_default() as g, g.name_scope(None):
    return variable_scope.get_variable(
        _TENSOR_VALUES_CACHE,
        shape=[num_tensors],
        dtype=dtypes.float32,
        initializer=init_ops.constant_initializer(
            _COMPACT_TRACE_ENTRY_INIT_VALUE),
        trainable=False,
        use_resource=True,
        collections=[_TENSOR_TRACER_STORAGE, ops.GraphKeys.LOCAL_VARIABLES])


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
  # The set of graphs that are rewritten by tensor tracer.
  _traced_graphs = set()

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
    valid_flag_names = [
        _FLAG_NAME_ENABLE, _FLAG_NAME_TRACE_MODE, _FLAG_NAME_USE_COMPACT_TRACE,
        _FLAG_NAME_TRACE_SCALAR_OPS, _FLAG_NAME_TRACE_BEFORE_OPS,
        _FLAG_NAME_TRACE_AFTER_OPS, _FLAG_NAME_TRACE_STACK_SIZE,
        _FLAG_NAME_SUBMODE, _FLAG_NAME_EXCLUDED_OPNAMES,
        _FLAG_NAME_EXCLUDED_OPTYPES, _FLAG_NAME_INCLUDED_OPNAMES,
        _FLAG_NAME_INCLUDED_OPTYPES, _FLAG_NAME_TRACE_DIR,
        _FLAG_NAME_REPORT_FILE, _FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR,
        _FLAG_NAME_INCLUDE_LESS_INTERESTING_OPS, _FLAG_NAME_OP_RANGE,
        _FLAG_DUMP_BEFORE_AFTER_GRAPHS
    ]
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
  def get_flag_int_value(wanted_flag_name, default_value):
    """Returns the int value of a TensorTracer flag.

    Args:
      wanted_flag_name: the name of the flag we are looking for.
      default_value: the default value for the flag, if not provided.
    Returns:
      the value of the flag.
    Raises:
      RuntimeError: If supposedly deadcode is reached.
    """
    flag_int_value = default_value
    found, flag_value = TensorTracer.get_flag_value(wanted_flag_name)

    if found:
      try:
        flag_int_value = int(flag_value)
      except ValueError:
        logging.warning('Cannot convert %s to int for flag %s' % (
            flag_int_value, wanted_flag_name))
    return flag_int_value

  @staticmethod
  def get_flag_value(wanted_flag_name):
    """Returns the value of a TensorTracer flags.

    Args:
      wanted_flag_name: the name of the flag we are looking for.

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
  def use_compact_trace():
    return TensorTracer._is_flag_on(
        _FLAG_NAME_USE_COMPACT_TRACE)

  @staticmethod
  def check_device_type(device_type):
    """Checks if the given device type is valid."""

    if device_type not in [_DEVICE_TYPE_TPU, _DEVICE_TYPE_CPU]:
      raise ValueError('Invalid device_type "%s"'%device_type)

  @staticmethod
  def check_trace_mode(trace_mode):
    """Checks if the given trace mode is valid."""

    valid_trace_modes = [
        _TRACE_MODE_NAN_INF, _TRACE_MODE_PART_TENSOR, _TRACE_MODE_FULL_TENSOR,
        _TRACE_MODE_NORM, _TRACE_MODE_MAX_ABS, _TRACE_MODE_FULL_IF_NAN
    ]
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
  def loop_cond_op(op):
    return op.type in ('LoopCond', 'RefLoopCond')

  @staticmethod
  def while_loop_op(op):
    """Returns true if op is one of the special ops of in a while loop.

    Args:
       op: A tf.Operation.

    Returns:
       True if the given op is one of [Switch, Merge, Enter, Exit,
       NextIteration, LoopCond], which are all building blocks for TF while
       loops.
    """
    return  (control_flow_util.IsLoopSwitch(op) or
             control_flow_util.IsLoopMerge(op) or
             control_flow_util.IsLoopEnter(op) or
             control_flow_util.IsLoopExit(op) or
             TensorTracer.loop_cond_op(op) or
             op.type in ('RefNextIteration', 'NextIteration'))

  @staticmethod
  def unsafe_op(op):
    """Returns True if this op is not safe to be traced."""

    if control_flow_util.IsInCond(op):
      return True
    # Reasons for not including following op types:
    #    Assign: cause incorrect result with CPU tracing.
    if op.type in ['Assign']:
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
    def _is_loop_edge(op):
      """Returns true if the op is the end of a while-loop creating a cycle."""
      return op.type in ['NextIteration']

    def _in_op_degree(op):
      """Returns the number of incoming edges to the given op.

      The edge calculation skips the edges that come from 'NextIteration' ops.
      NextIteration creates a cycle in the graph. We break cycles by treating
      this op as 'sink' and ignoring all outgoing edges from it.
      Args:
        op: Tf.Operation
      Returns:
        the number of incoming edges.
      """
      count = 0
      for op in op.control_inputs + [in_tensor.op for in_tensor in op.inputs]:
        if not _is_loop_edge(op):
          count += 1
      return count

    sorted_ops = []
    op_in_degree = {op: _in_op_degree(op) for op in g.get_operations()}

    frontier = [op for (op, degree) in op_in_degree.items() if degree == 0]
    while frontier:
      op = frontier.pop()
      # Remove the op from graph, and remove its outgoing edges.
      sorted_ops.append(op)
      if _is_loop_edge(op):
        continue
      # pylint: disable=protected-access
      consumers = list(op._control_outputs)
      # pylint: enable=protected-access
      for out_tensor in op.outputs:
        consumers += [consumer_op for consumer_op in out_tensor.consumers()]

      for consumer in consumers:
        # For each deleted edge shift the bucket of the vertex.
        op_in_degree[consumer] -= 1
        if op_in_degree[consumer] == 0:
          frontier.append(consumer)
        if op_in_degree[consumer] < 0:
          raise ValueError('consumer:%s degree mismatch'%consumer.name)

    left_ops = set([op for (op, degree) in op_in_degree.items() if degree > 0])
    if left_ops:
      return (False, left_ops)
    else:
      assert len(g.get_operations()) == len(sorted_ops)
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

  @staticmethod
  def is_conditional_trace_mode(trace_mode):
    return trace_mode == _TRACE_MODE_FULL_IF_NAN

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
    self._set_trace_dir()
    self._set_report_file()
    self._set_op_range()
    self._set_excluded_opnames()
    self._set_excluded_optypes()
    self._set_included_opnames()
    self._set_included_optypes()
    self._num_replicas = None
    self._num_replicas_per_host = None
    self._num_hosts = None
    self._replica_id = None
    self._included_op_full_names = set()
    self._is_conditional_trace = TensorTracer.is_conditional_trace_mode(
        self._trace_mode)
    self._trace_scalar_ops = TensorTracer._is_flag_on(
        _FLAG_NAME_TRACE_SCALAR_OPS)

    # _trace_ops_before_included and _trace_ops_after_included denotes to depth
    # of tracing relative to the ops given in --included_opnames or
    # --included_optypes
    # For example, in the below graph
    #                op1 --> op2 --> op3 --> op4 --> op5
    # If --included_opnames=op3 then only op3 will be traced.
    # If also --trace_before_included_ops=2 (_trace_ops_before_included), then
    # op1 and op2 will be traced as they are at most 2 hops apart from an
    # included op. Similarly, if --trace_after_included_ops=2, then op4 and op5
    # will also be traced.
    self._trace_ops_before_included = TensorTracer.get_flag_int_value(
        _FLAG_NAME_TRACE_BEFORE_OPS, 0)
    self._trace_ops_after_included = TensorTracer.get_flag_int_value(
        _FLAG_NAME_TRACE_AFTER_OPS, 0)
    self._trace_stack_size = TensorTracer.get_flag_int_value(
        _FLAG_NAME_TRACE_STACK_SIZE, 1)

    _, self._graph_dump_path = TensorTracer.get_flag_value(
        _FLAG_DUMP_BEFORE_AFTER_GRAPHS)

  def _add_replica_id_to_graph(self):
    """Adds nodes for computing the replica ID to the graph."""

    if self._num_replicas:
      with ops.control_dependencies(None):
        # Uses None as dependency to run outside of TPU graph rewrites.
        self._replica_id = tpu_ops.tpu_replicated_input(
            list(range(self._num_replicas)),
            name='tt_replica_id')
    else:
      self._replica_id = 'unknown'

  def _set_trace_dir(self):
    found, self._trace_dir = TensorTracer.get_flag_value(_FLAG_NAME_TRACE_DIR)
    if found and self._trace_dir \
       and TensorTracer.use_test_undeclared_outputs_dir():
      raise ValueError('Cannot not use --%s and --%s at the same time'
                       %(_FLAG_NAME_TRACE_DIR,
                         _FLAG_NAME_USE_TEST_UNDECLARED_OUTPUTS_DIR))
    if TensorTracer.use_test_undeclared_outputs_dir():
      self._trace_dir = os.environ.get(_TEST_UNDECLARED_OUTPUTS_DIR_ENV_VAR)

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
    """Checks whether the op is included in the tensor tracer flags.

    Args:
      op: tf Operation
    Returns:
      True, if the op is included.
      An op is included if:
      - Its op name is given in included_opnames
      - Its op type is given in included_optypes
      - The op is at most _trace_ops_before_included hops before an included op
      - The op is at most _trace_ops_after_included hops after an included op
    """

    def _is_op_or_any_neighbor_included(op, check_before=0, check_after=0):
      """Helper function to check if op is included or not."""
      if op.name in self._included_op_full_names:
        return True
      for opname_re in self._included_opname_re_list:
        if opname_re.match(op.name):
          self._included_op_full_names.add(op.name)
          return True

      if check_after > 0:
        for out_tensor in op.outputs:
          for consumer in out_tensor.consumers():
            if _is_op_or_any_neighbor_included(consumer, check_after - 1, 0):
              self._included_op_full_names.add(op.name)
              return True
      if check_before > 0:
        for input_tensor in op.inputs:
          if _is_op_or_any_neighbor_included(input_tensor.op,
                                             0,
                                             check_before - 1):
            self._included_op_full_names.add(op.name)
            return True
      return False
    # check_after and check_before are swapped below, as below operation
    # checks the distance from an arbitrary op to included ops.
    return _is_op_or_any_neighbor_included(op,
                                           self._trace_ops_after_included,
                                           self._trace_ops_before_included)

  def _is_user_excluded_op(self, op):
    for opname_re in self._excluded_opname_re_list:
      if opname_re.match(op.name):
        return True
    for optype_re in self._excluded_optype_re_list:
      if optype_re.match(op.type):
        return True
    return False

  def _use_tensor_values_cache(self):
    """Returns True if immediate tensors should be first saved to a cache."""

    if self._trace_mode not in set([_TRACE_MODE_NAN_INF,
                                    _TRACE_MODE_NORM, _TRACE_MODE_MAX_ABS]):
      return False
    if self._trace_dir and _trace_files_need_precreated(self._trace_dir):
      return True
    if TensorTracer.use_compact_trace():
      return True
    return False

  def _save_tensor_value_to_cache_op(self, graph, cache_idx, updates):
    """Returns an Op that will save the given updates to an entry in the cache."""

    cache = _get_tensor_values_cache(graph)
    indices = constant_op.constant([cache_idx])
    return state_ops.scatter_update(cache, indices, updates).op

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
    self._write_report('%s %s\n'%(_FIELD_NAME_NUM_REPLICAS_PER_HOST,
                                  self._num_replicas_per_host))
    self._write_report('%s %s\n'%(_FIELD_NAME_NUM_HOSTS, self._num_hosts))
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

  def _write_cache_index_map_section(self):
    """Writes the mapping from cache index to tensor index to the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN,
                                  _SECTION_NAME_CACHE_INDEX_MAP))
    self._write_report('%s %d\n'%(_FIELD_NAME_NUM_CACHE_INDICES,
                                  len(self._cache_idx_to_tensor_idx)))
    for cache_idx in range(0, len(self._cache_idx_to_tensor_idx)):
      tensor_idx = self._cache_idx_to_tensor_idx[cache_idx]
      line = '%d %d\n'%(cache_idx, tensor_idx)
      self._write_report(line)
    self._write_report('%s %s\n'%(_MARKER_SECTION_END,
                                  _SECTION_NAME_CACHE_INDEX_MAP))

  def _write_graph_section(self, succeed, sorted_or_cycle):
    """Writes the graph section of the report."""

    self._write_report('%s %s\n'%(_MARKER_SECTION_BEGIN, _SECTION_NAME_GRAPH))
    self._write_report('%s %s\n'%(_FIELD_NAME_TOPOLOGICAL_SORT_SUCCEED,
                                  succeed))
    l = list(sorted_or_cycle)
    for i in range(0, len(l)):
      self._write_report('%d "%s"\n'%(i, l[i].name))
    self._write_report('%s %s\n'%(_MARKER_SECTION_END, _SECTION_NAME_GRAPH))

  def _preprocess_traced_tensor(self, tensor):
    """Computes NAN/Norm/Max on TPUs before sending to CPU.

    Args:
      tensor: The tensor to be traced.
    Returns:
      A tensor that should be input to the trace_function.
    Raises:
      RuntimeError: If the trace mode is invalid.
    """

    def _detect_nan_inf(tensor):
      """Trace function for detecting any NaN/Inf in the tensor."""

      if tensor.dtype.is_floating:
        mask = math_ops.reduce_any(
            gen_math_ops.logical_or(
                gen_math_ops.is_nan(tensor), gen_math_ops.is_inf(tensor)))
        output_tensor = control_flow_ops.cond(mask,
                                              lambda: constant_op.constant(1.0),
                                              lambda: constant_op.constant(0.0))
      else:
        output_tensor = constant_op.constant(0.0)
      # The shape has to be 1. Set it if it does not have the information.
      output_tensor = array_ops.reshape(output_tensor, [1])
      return output_tensor

    def _show_norm(tensor):
      tensor = math_ops.cast(tensor, dtypes.float32)
      output_tensor = linalg_ops.norm(tensor)
      # The shape has to be 1. Set it if it does not have the information.
      output_tensor = array_ops.reshape(output_tensor, [1])
      return output_tensor

    def _show_max_abs(tensor):
      tensor = math_ops.cast(tensor, dtypes.float32)
      output_tensor = math_ops.reduce_max(math_ops.abs(tensor))
      zero = constant_op.constant(0, dtypes.float32)
      output_tensor = gen_math_ops.maximum(zero, output_tensor)
      # The shape has to be 1. Set it if it does not have the information.
      output_tensor = array_ops.reshape(output_tensor, [1])
      return output_tensor

    def _detect_inf_nan_producer(tensor):
      """Checks if the tensor is the first NaN/Inf tensor in the computation path."""
      if tensor.op.inputs:
        inp_check = [
            _detect_nan_inf(inp_tensor) for inp_tensor in tensor.op.inputs
        ]
        is_any_input_inf_nan = math_ops.add_n(inp_check)
      else:
        is_any_input_inf_nan = constant_op.constant(0, dtypes.bool)
      is_current_tensor_inf_nan = _detect_nan_inf(tensor)
      # An op is NaN/INF producer only when all inputs are nan/inf free (
      # is_any_input_inf_nan = 0), and its output has nan/inf (
      # is_current_tensor_inf_nan=1). Below will be 1 if op nan/inf is producer.
      is_nan_producer = is_current_tensor_inf_nan - is_any_input_inf_nan
      is_nan_producer = math_ops.reduce_any(is_nan_producer > 0)
      return is_nan_producer

    if self._trace_mode == _TRACE_MODE_FULL_IF_NAN:
      return _detect_inf_nan_producer(tensor)
    if self._trace_mode == _TRACE_MODE_NAN_INF:
      return _detect_nan_inf(tensor)
    if self._trace_mode == _TRACE_MODE_PART_TENSOR:
      return tensor
    if self._trace_mode == _TRACE_MODE_FULL_TENSOR:
      return tensor
    if self._trace_mode == _TRACE_MODE_NORM:
      return _show_norm(tensor)
    if self._trace_mode == _TRACE_MODE_MAX_ABS:
      return _show_max_abs(tensor)
    raise RuntimeError(
        'Tensor trace fun for %s is not yet implemented' % self._trace_mode)

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

      if self._trace_dir:
        output_path = os.path.join(self._trace_dir, _TRACE_FILE_NAME)
        output_stream = _OUTPUT_STREAM_ESCAPE + output_path
      else:
        output_stream = sys.stderr
      return logging_ops.print_v2(msg, array_ops.shape(output_tensor),
                                  '@', self._replica_id,
                                  '\n', output_tensor, '\n',
                                  summarize=num_elements,
                                  output_stream=output_stream)

    def _show_part_tensor(tensor):
      """Trace function for printing part of the tensor."""

      return _print_tensor(tensor_name, self._part_tensor_size,
                           tensor, tensor)

    def _show_full_tensor(tensor):
      """Trace function for printing the entire tensor."""

      return _print_tensor(tensor_name, -1, tensor, tensor)

    def _show_full_tensors(tensor):
      """Prints the full tensor values for the tensors that are _trace_stack_size hops away from a given tensor."""

      def _get_distance_k_tensors(k_before=0):
        """Returns the tensors that are at most k_before hops away from the tensor."""
        if k_before < 0:
          return []
        visited_tensors = {tensor: 0}
        visitor_queue = [tensor]
        head = 0
        while head < len(visitor_queue):
          current_tensor = visitor_queue[head]
          head += 1
          distance = visited_tensors[current_tensor]
          if distance == k_before:
            break
          for input_tensor in current_tensor.op.inputs:
            if input_tensor in visited_tensors:
              continue
            visitor_queue.append(input_tensor)
            visited_tensors[input_tensor] = distance + 1
        return visitor_queue

      tensors_to_print = _get_distance_k_tensors(self._trace_stack_size)
      print_ops = [_print_tensor(t.name, -1, t, t) for t in tensors_to_print]
      with ops.control_dependencies(print_ops):
        return constant_op.constant(0)

    if self._trace_mode == _TRACE_MODE_FULL_IF_NAN:
      return _show_full_tensors
    if self._trace_mode == _TRACE_MODE_PART_TENSOR:
      return _show_part_tensor
    # The input tensor has a shape of "[1]" for _TRACE_MODE_NAN_INF,
    # _TRACE_MODE_NORM, and _TRACE_MODE_MAX_ABS, as related computations are
    # performed within TPUs and only their results are transferred to CPU.
    # Simply, print the full tensor for these trace modes.
    if self._trace_mode in [
        _TRACE_MODE_NAN_INF, _TRACE_MODE_NORM, _TRACE_MODE_FULL_TENSOR,
        _TRACE_MODE_MAX_ABS
    ]:
      return _show_full_tensor

    raise RuntimeError('Tensor trace fun for %s is not yet implemented'
                       %self._trace_mode)

  def _skip_op(self, op_id, op, user_included, user_excluded,
               in_exec_path=True):
    """Returns True if we should not trace Op."""

    if TensorTracer.while_loop_op(op):
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_WHILELOOP_OP)
      return True
    if TensorTracer.unsafe_op(op):
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_UNSAFE_OP)
      return True
    if TensorTracer.device_mismatch(self._device_type, op):
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_DEVICE_MISMATCH)
      return True
    if not in_exec_path:
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_NOT_EXECUTED)
      return True

    if not self._inside_op_range(op_id):
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_OUTSIDE_OP_RANGE)
      return True
    if TensorTracer.less_interesting_op(op):
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_LESS_INTERESTING_OP)
      return True
    if user_included:
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_USER_INCLUDED)
      return False
    if user_excluded:
      self._instrument_records[op.name] = TensorTracer.reason(
          op_id, _REASON_USER_EXCLUDED)
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
    # Skip a tensor if it feeds a special while loop op.
    if [consumer for consumer in out_tensor.consumers() if
        TensorTracer.while_loop_op(consumer)]:
      self._instrument_records[out_tensor.name] = TensorTracer.reason(
          op_id, _REASON_FEEDS_WHILELOOP_OP)
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
      # If trace mode is nan-inf, norm or max, then the tensor will be reduced
      # to a scalar before the outside compilation call.
      if self._trace_mode in [
          _TRACE_MODE_NAN_INF, _TRACE_MODE_NORM, _TRACE_MODE_MAX_ABS
      ]:
        self._instrument_records[out_tensor.name] = TensorTracer.reason(
            op_id, _REASON_TENSOR_GET_TRACED)
        return False
      else:
        self._instrument_records[out_tensor.name] = TensorTracer.reason(
            op_id, _REASON_DYNAMIC_SHAPE)
        return True
    rank = len(out_tensor.shape)
    if rank < 1:
      # scalar
      if self._trace_scalar_ops:
        if TensorTracer.unsafe_scalar_trace(out_tensor.op):
          self._instrument_records[out_tensor.name] = TensorTracer.reason(
              op_id, _REASON_UNSAFE_SCALAR)
          return True
        else:
          self._instrument_records[out_tensor.name] = TensorTracer.reason(
              op_id, _REASON_SCALAR_GET_TRACED)
          return False
      else:
        self._instrument_records[out_tensor.name] = TensorTracer.reason(
            op_id, _REASON_SKIP_SCALAR)
        return True
    else:
      # tensor
      self._instrument_records[out_tensor.name] = TensorTracer.reason(
          op_id, _REASON_TENSOR_GET_TRACED)
      return False

  def _filter_execution_path_operations(self, operations, fetches):
    """Returns the set of ops in the execution path to compute given fetches."""

    # If no fetch provided, then return all operations.
    if fetches is None:
      return set(operations)
    # Convert to list, if a single element is provided.
    if not isinstance(fetches, (list, tuple)):
      fetches = [fetches]
    # If a tensor is given as fetch, convert it to op.
    op_fetches = []
    for fetch in fetches:
      if isinstance(fetch, ops.Operation):
        op_fetches.append(fetch)
      elif isinstance(fetch, ops.Tensor):
        op_fetches.append(fetch.op)
      else:
        raise RuntimeError('Given fetch:%s is neither a tensor nor an op.'
                           %fetch)

    execution_path_operations = set(op_fetches)
    traverse_stack = list(op_fetches)
    while True:
      if not traverse_stack:
        break
      head_op = traverse_stack.pop()
      input_ops = [tensor_input.op for tensor_input in head_op.inputs]
      input_ops.extend(head_op.control_inputs)

      for input_op in input_ops:
        if input_op not in execution_path_operations:
          # Filter out loop condition operations, tracing them causes a cycle.
          # Trace only the loop-body.
          if TensorTracer.loop_cond_op(input_op):
            continue
          execution_path_operations.add(input_op)
          traverse_stack.append(input_op)
    return execution_path_operations

  def _determine_traced_tensors(self, graph, ops_in_exec_path):
    """Determines the tensors that will be traced."""

    self._traced_tensorname_to_cache_idx_map = {}
    self._cache_idx_to_tensor_idx = []
    operations = graph.get_operations()
    checkpoint_operations = self._get_checkpoints(graph)
    for op_id, op in enumerate(operations):
      if checkpoint_operations and op.name not in checkpoint_operations:
        continue
      user_included = self._is_user_included_op(op)
      user_excluded = self._is_user_excluded_op(op)
      in_exec_path = op in ops_in_exec_path
      if self._skip_op(op_id, op, user_included, user_excluded, in_exec_path):
        continue
      for i in range(len(op.outputs)):
        out_tensor = op.outputs[i]
        if self._skip_tensor(op_id, out_tensor, user_included,
                             user_excluded):
          continue
        tensor_name = out_tensor.name
        if tensor_name in self._traced_tensorname_to_cache_idx_map:
          raise ValueError(
              'Tensor name %s should not be already in '
              'traced_tensorname_to_cache_idx_map'%tensor_name)
        if tensor_name not in self._tensorname_idx_map:
          raise ValueError(
              'Tensor name %s is not in the tensorname_idx_map'%tensor_name)
        tensor_idx = self._tensorname_idx_map[tensor_name]
        cache_idx = len(self._traced_tensorname_to_cache_idx_map)
        self._traced_tensorname_to_cache_idx_map[tensor_name] = cache_idx
        self._cache_idx_to_tensor_idx.append(tensor_idx)
        if len(self._traced_tensorname_to_cache_idx_map) != len(
            self._cache_idx_to_tensor_idx):
          raise RuntimeError('len(self._traced_tensorname_to_cache_idx_map) != '
                             'len(self._cache_idx_to_tensor_idx')

  def _check_trace_files(self):
    """Checks if any requirements for trace files are satisfied."""

    if not self._trace_dir:
      # traces will be written to stderr. No need to check trace files.
      return
    if _trace_files_need_precreated(self._trace_dir):
      for replica_id in range(0, self._num_replicas):
        trace_file_path = os.path.join(
            self._trace_dir,
            _COMPACT_TRACE_FILE_PREFIX) + '%d'%replica_id
        if not gfile.Exists(trace_file_path):
          raise RuntimeError(
              '%s must be pre-created with the '
              'appropriate properties.'%trace_file_path)
    else:
      if not gfile.Exists(self._trace_dir):
        gfile.MkDir(self._trace_dir)
        if not gfile.Exists(self._trace_dir):
          raise RuntimeError('Failed to create %s'%self._trace_dir)

  def _pre_tracing(self, graph, fetches):
    """Work needs to be done prior to TPU or CPU tracing."""

    self._check_trace_files()
    operations = graph.get_operations()
    (opname_idx_map, tensor_list, self._tensorname_idx_map) = (
        TensorTracer._make_op_and_tensor_maps(operations))
    self._write_config_section()
    self._write_op_list_section(operations)
    self._write_tensor_list_section(tensor_list, opname_idx_map)
    # Filter out the operations that won't be executed.
    # if fetches=None, then ops_in_exec_path = set(operations)
    ops_in_exec_path = self._filter_execution_path_operations(operations,
                                                              fetches)
    self._determine_traced_tensors(graph, ops_in_exec_path)
    self._write_cache_index_map_section()
    # Does the topological sort before adding any nodes to the graph.
    (succeed, sorted_or_cycle) = TensorTracer.topological_sort(graph)
    if self._use_tensor_values_cache():
      _create_tensor_values_cache(graph,
                                  len(self._cache_idx_to_tensor_idx))
    return (ops_in_exec_path, succeed, sorted_or_cycle)

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

  def _generate_flush_cache_op(self, graph, start_replica, on_tpu):
    """Generates an Op that will flush the cache to file.

    Args:
      graph: the graph of Ops
      start_replica: the ID of the first replica being flushed by this Op.
      on_tpu: if the graph is executed on TPU.

    Returns:
      The Op to flush the cache to file.
    """
    def _make_flush_fun(replica_id):
      """Makes a function for flushing the cache for the given replica."""

      def _fun():
        """A function that flushes the cache to a file."""

        def _flush_fun(cache):
          """Flushes the cache to a file."""

          if isinstance(replica_id, str):
            replica_id_str = replica_id
          else:
            replica_id_str = '%d'%replica_id
          if self._trace_dir:
            output_path = os.path.join(self._trace_dir,
                                       _COMPACT_TRACE_FILE_PREFIX) \
                                       + replica_id_str
            output_stream = _OUTPUT_STREAM_ESCAPE + output_path
          else:
            output_stream = sys.stderr
          new_step_line = _REPLICA_ID_TAG + replica_id_str
          print_op = logging_ops.print_v2(
              new_step_line, '\n',
              cache, '\n',
              summarize=-1,
              output_stream=output_stream)
          with ops.control_dependencies([print_op]):
            return constant_op.constant(0).op

        cache = _get_tensor_values_cache(graph)
        if on_tpu:
          flush_op = tpu.outside_compilation(_flush_fun, cache.value())
        else:
          flush_op = _flush_fun(cache.value())
        with ops.control_dependencies([flush_op]):
          reset_value = constant_op.constant(_COMPACT_TRACE_ENTRY_INIT_VALUE,
                                             dtype=cache.dtype,
                                             shape=cache.shape)
          assign_op = state_ops.assign(cache, reset_value).op
          with ops.control_dependencies([assign_op]):
            return flush_op.outputs[0]

      return _fun

    def _f(replica_id):
      return _make_flush_fun(replica_id)
    def _eq(x):
      return math_ops.equal(x, self._replica_id)
    def _do_nothing():
      return constant_op.constant(0)

    return control_flow_ops.case({\
                                  _eq(start_replica): _f(start_replica), \
                                  _eq(start_replica+1): _f(start_replica+1), \
                                  _eq(start_replica+2): _f(start_replica+2), \
                                  _eq(start_replica+3): _f(start_replica+3), \
                                  _eq(start_replica+4): _f(start_replica+4), \
                                  _eq(start_replica+5): _f(start_replica+5), \
                                  _eq(start_replica+6): _f(start_replica+6), \
                                  _eq(start_replica+7): _f(start_replica+7), \
    },
                                 default=_do_nothing,
                                 exclusive=True).op

  def _flush_tensor_values_cache(self, graph, tensor_fetches, op_fetches,
                                 on_tpu):
    """Flushes the intermediate tensor values in the graph to the cache.

    Args:
      graph: the graph of Ops
      tensor_fetches: list of tensor results returned by the model_fn.
      op_fetches: list of ops that are returned by the model_fn, e.g., train_op.
      on_tpu: if the graph is executed on TPU.

    Returns:
      An identical copy of tensor_fetches.
    """
    # Add a dependency to op and tensor fetches to make sure that all tracing
    # ops are executed before flushing trace results.
    with ops.control_dependencies(op_fetches +
                                  [tensor.op for tensor in tensor_fetches]):
      flush_cache_op_list = []
      for host in range(self._num_hosts):
        start_replica = host * 8
        flush_op = self._generate_flush_cache_op(graph, start_replica, on_tpu)
        flush_cache_op_list.append(flush_op)
      return control_flow_ops.tuple(tensor_fetches,
                                    control_inputs=flush_cache_op_list)

  def _process_tensor_fetches(self, tensor_fetches):
    """Check that tensor_fetches is not empty and have valid tensors."""
    # If none or empty list.
    if tensor_fetches is None:
      raise RuntimeError('tensor_fetches provided to tensor_tracer cannot be '
                         'None.')
    if not isinstance(tensor_fetches, (list, tuple)):
      tensor_fetches = [tensor_fetches]
    elif not tensor_fetches:
      raise RuntimeError('tensor_fetches provided to tensor_tracer cannot be '
                         'empty list.')
    fetches = []
    for fetch in tensor_fetches:
      if isinstance(fetch, ops.Tensor):
        fetches.append(fetch)
      else:
        raise RuntimeError('Given tensor_fetch:%s is not a tensor.' % fetch)
    return fetches

  def _process_op_fetches(self, op_fetches):
    """Check that op_fetches have valid ops."""
    if op_fetches is None:
      return []

    if not isinstance(op_fetches, (list, tuple)):
      op_fetches = [op_fetches]

    fetches = []
    for fetch in op_fetches:
      if isinstance(fetch, ops.Operation):
        fetches.append(fetch)
      else:
        logging.warning('Ignoring the given op_fetch:%s, which is not an op.' %
                        fetch)
    return fetches

  def _convert_fetches_to_input_format(self, input_fetches, current_fetches):
    """Changes current_fetches' format, so that it matches input_fetches."""
    if isinstance(input_fetches, ops.Tensor):
      if len(current_fetches) != 1:
        raise RuntimeError('Tensor tracer input/output fetches do not match.')
      return current_fetches[0]
    else:
      if len(current_fetches) != len(current_fetches):
        raise RuntimeError('Tensor tracer input/output fetches do not match.')
      elif isinstance(input_fetches, tuple):
        return tuple(current_fetches)
      else:
        return current_fetches

  def _get_op_control_flow_context(self, op):
    """Returns the control flow of the given op.

    Args:
      op: tf.Operation for which the control flow context is requested.
    Returns:
      op_control_flow_context: which the is control flow context of the given
      op. If the operation type is LoopExit, returns the outer control flow
      context.
    """
    # pylint: disable=protected-access
    op_control_flow_context = op._control_flow_context
    # pylint: enable=protected-access
    if control_flow_util.IsLoopExit(op):
      op_control_flow_context = op_control_flow_context.outer_context
    return op_control_flow_context

  def _trace_execution(self, graph,
                       tensor_fetches,
                       op_fetches=None,
                       on_tpu=True):
    """Commong tracing function for both CPU and TPUs.

    The caller function should set _device_type, _num_replicas,
    _num_replicas_per_host, _num_hosts and _replica_id before calling
    _trace_execution.


    Args:
      graph: the graph of Ops executed on the TPU.
      tensor_fetches: a (list,tuple,or a single object) of tensor fetches
        returned by model_fn given to session.run. Function must be provided
        with as least one tensor to fetch.
      op_fetches: A list of op fetches returned by model_fn given to
        session.run. op_fetches and tensor_fetches are used to determine the
        nodes that will be executed. Can be None.
      on_tpu: True if executing on TPU.

    Returns:
      tensor_fetches: an exact copy of tensor_fetches that has additional
                      dependencies.
    Raises:
      RuntimeError: If tensor_fetches is None or empty.
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

    TensorTracer.check_device_type(self._device_type)
    # Check in_tensor_fetches, and op_fetches and convert them to lists.
    processed_t_fetches = self._process_tensor_fetches(tensor_fetches)
    op_fetches = self._process_op_fetches(op_fetches)
    all_fetches = op_fetches + [tensor.op for tensor in processed_t_fetches]

    # Filter the set of ops that will be executed, and topological sort.
    (exec_op_set, succeed, sorted_or_cycle) = self._pre_tracing(graph,
                                                                all_fetches)

    tensor_fetch_set = set(processed_t_fetches)
    tracing_ops = []

    # pylint: disable=protected-access
    current_control_flow_context = graph._get_control_flow_context()
    # pylint: enable=protected-access

    sorted_exec_op_list = list(exec_op_set)
    sorted_exec_op_list.sort(key=lambda op: op.name)
    # Trace ops only if they are in the execution path.
    for op in sorted_exec_op_list:
      for i in range(len(op.outputs)):
        out_tensor = op.outputs[i]
        tensor_name = out_tensor.name
        if tensor_name not in self._traced_tensorname_to_cache_idx_map:
          continue
        # Create the list of consumers before calling _preprocess_traced_tensor.
        # Otherwise, adding control input below, will introduce a cycle in the
        # graph.
        consumers = out_tensor.consumers()
        # Not all consumers may be in the exec path. Filter out the consumers
        # to keep the graph simpler.
        consumers = [cop for cop in consumers if cop in exec_op_set]

        # If there is no consumer of the tensor, there is no need to trace it;
        # unless the tensor itself is one of the fetches.
        is_a_fetched_tensor = out_tensor in tensor_fetch_set
        if (not consumers) and (not is_a_fetched_tensor):
          continue

        op_control_flow_context = self._get_op_control_flow_context(op)
        # pylint: disable=protected-access
        graph._set_control_flow_context(op_control_flow_context)
        # pylint: enable=protected-access
        processed_out_tensor = self._preprocess_traced_tensor(out_tensor)

        if on_tpu:
          processed_out_tensor = _cast_unsupported_dtypes(processed_out_tensor)

        if self._use_tensor_values_cache():
          cache_idx = self._traced_tensorname_to_cache_idx_map[tensor_name]
          trace_op = self._save_tensor_value_to_cache_op(graph,
                                                         cache_idx,
                                                         processed_out_tensor)
        else:

          def tpu_wrap_trace_fn(tensor, out_tensor_name):
            """Wraps the trace_fn with outside compilation if on TPUs."""
            tensor_trace_fn = self._make_tensor_trace_fun(out_tensor_name)
            if on_tpu:
              return tpu.outside_compilation(tensor_trace_fn, tensor)
            else:
              return tensor_trace_fn(tensor)

          def conditional_trace_fn(predicate_tensor, out_tensor, trace_fn,
                                   out_tensor_name):
            """Creates a cond op that traces the out_tensor if predicate is satisfied."""
            return control_flow_ops.cond(
                predicate_tensor, lambda: trace_fn(out_tensor, out_tensor_name),
                lambda: constant_op.constant(0)).op

          if self._is_conditional_trace:
            trace_op = conditional_trace_fn(processed_out_tensor, out_tensor,
                                            tpu_wrap_trace_fn, tensor_name)
          else:
            trace_op = tpu_wrap_trace_fn(processed_out_tensor, tensor_name)

        if is_a_fetched_tensor:
          tracing_ops.append(trace_op)
          continue
        # Add it to all consumers, as some consumers may not be executed if they
        # are in a control flow.
        for consumer_op in consumers:
          # pylint: disable=protected-access
          consumer_op._add_control_input(trace_op)
          # pylint: enable=protected-access

    # pylint: disable=protected-access
    graph._set_control_flow_context(current_control_flow_context)
    # pylint: enable=protected-access
    if tracing_ops:
      # If we are tracing a fetched tensor, their dependency is stored in
      # tracing_ops.
      processed_t_fetches = control_flow_ops.tuple(processed_t_fetches,
                                                   control_inputs=tracing_ops)
    if self._use_tensor_values_cache():
      processed_t_fetches = self._flush_tensor_values_cache(graph,
                                                            processed_t_fetches,
                                                            op_fetches,
                                                            on_tpu=on_tpu)
    self._post_tracing(succeed, sorted_or_cycle)
    # processed_t_fetches is a list at this point. Convert it to the same
    # format as given in tensor_fetches.
    return self._convert_fetches_to_input_format(tensor_fetches,
                                                 processed_t_fetches)

  def trace_tpu(self, graph,
                tensor_fetches,
                op_fetches=None,
                num_replicas=None,
                num_replicas_per_host=None,
                num_hosts=None):
    """Traces the tensors generated by TPU Ops in a TF graph.

    Args:
      graph: the graph of Ops executed on the TPU.
      tensor_fetches: a (list,tuple,or a single object) of tensor fetches
        returned by model_fn given to session.run. Function must be provided
        with as least one tensor to fetch.
      op_fetches: A list of op fetches returned by model_fn given to
        session.run. op_fetches and tensor_fetches are used to determine the
        nodes that will be executed. Can be None.
      num_replicas: number of replicas used on the TPU.
      num_replicas_per_host: number of replicas per TPU host.
      num_hosts: total number of TPU hosts.

    Returns:
      tensor_fetches: an exact copy of tensor_fetches that has additional
                      dependencies.
    Raises:
      RuntimeError: If num_replicas_per_host > 8.
      RuntimeError: If tensor_fetches is None or empty.
    """

    if graph in TensorTracer._traced_graphs:
      logging.warning('Graph is already rewritten with tensor tracer, ignoring '
                      'multiple calls.')
      return tensor_fetches
    else:
      TensorTracer._traced_graphs.add(graph)
    self._device_type = _DEVICE_TYPE_TPU
    self._num_replicas = num_replicas
    self._num_replicas_per_host = num_replicas_per_host
    self._num_hosts = num_hosts
    if self._num_replicas is not None:
      if self._num_replicas_per_host is None:
        self._num_replicas_per_host = 8
      if self._num_hosts is None:
        self._num_hosts = num_replicas // self._num_replicas_per_host + \
            (num_replicas % self._num_replicas_per_host > 0)

    if self._num_replicas_per_host > 8:
      # Checks for the assumption in _generate_flush_cache_op().
      raise RuntimeError('num_replicas_per_host (%d) is '
                         'greater than 8'%self._num_replicas_per_host)
    if self._graph_dump_path:
      graph_io.write_graph(graph, self._graph_dump_path,
                           'graph_before_tt.pbtxt')
    with graph.as_default():
      self._add_replica_id_to_graph()
      tensor_fetches = self._trace_execution(graph, tensor_fetches, op_fetches,
                                             on_tpu=True)
    if self._graph_dump_path:
      graph_io.write_graph(graph, self._graph_dump_path,
                           'graph_after_tt.pbtxt')
    return tensor_fetches

  def trace_cpu(self, graph, tensor_fetches, op_fetches=None):
    """Traces the tensors generated by CPU Ops in a TF graph.

    Args:
      graph: the graph of Ops executed on the CPU.
      tensor_fetches: a (list,tuple,or a single object) of tensor fetches
        returned by model_fn given to session.run. Function must be provided
        with as least one tensor to fetch.
      op_fetches: A list of op fetches returned by model_fn given to
        session.run. op_fetches and tensor_fetches are used to determine the
        nodes that will be executed. Can be None.

    Returns:
      tensor_fetches: an exact copy of tensor_fetches that has additional
                      dependencies.
    Raises:
      RuntimeError: If tensor_fetches is None or empty.
    """

    if graph in TensorTracer._traced_graphs:
      logging.warning('Graph is already rewritten with tensor tracer, ignoring '
                      'multiple calls.')
      return tensor_fetches
    else:
      TensorTracer._traced_graphs.add(graph)

    self._device_type = _DEVICE_TYPE_CPU
    self._num_replicas = 1
    self._num_replicas_per_host = 1
    self._num_hosts = 1
    self._replica_id = 0
    if self._graph_dump_path:
      graph_io.write_graph(graph, self._graph_dump_path,
                           'graph_before_tt.pbtxt')
    with graph.as_default():
      tensor_fetches = self._trace_execution(graph, tensor_fetches, op_fetches,
                                             on_tpu=False)
    if self._graph_dump_path:
      graph_io.write_graph(graph, self._graph_dump_path,
                           'graph_after_tt.pbtxt')
    return tensor_fetches
