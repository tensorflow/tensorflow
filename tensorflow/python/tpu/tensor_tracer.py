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

import collections
import hashlib
import operator

import os
import os.path
import sys

import numpy as np
import six

from tensorflow.core.framework import summary_pb2
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_io
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import analytics
from tensorflow.python.platform import gfile
from tensorflow.python.platform import remote_utils
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.summary import summary_iterator
from tensorflow.python.tpu import tensor_tracer_flags
from tensorflow.python.tpu import tensor_tracer_report
from tensorflow.python.tpu import tpu
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.training import training_util

_DEVICE_TYPE_TPU = 'tpu'
_DEVICE_TYPE_CPU = 'cpu'
_TRACE_MODE_PART_TENSOR_SIZE = 3

_REASON_OUTSIDE_OP_RANGE = 'not-traced-outside-op-range'
_REASON_UNSAFE_OP = 'not-traced-unsafe-op'
_REASON_WHILELOOP_OP = 'not-traced-special-whileloop-op'
_REASON_CONTROLFLOW_OP = 'not-traced-control-flow-op'
_REASON_IN_CONTROL_FLOW = 'not-traced-in-control-flow'
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

_OUTPUT_STREAM_ESCAPE = 'file://'
_TENSOR_TRACER_COLLECTION = 'tensor_tracer_variables'
TENSOR_TRACER_SUMMARY_COLLECTION = 'tensor_tracer_summary_writers'
_TRACE_FILE_NAME = 'trace.all'
_COMPACT_TRACE_FILE_PREFIX = 'compact_trace.'
_COMPACT_TRACE_ENTRY_INIT_VALUE = -1.0
_TENSOR_TRACER_STORAGE = 'tensor_tracer_storage'
_TT_SNAPSHOT = 'tensor_tracer_snapshot'
_REPLICA_ID_TAG = '#replica-id: '
_SKIP_REPORT_FILE = 'None'  # Do not write report proto if --report_file=None

_TT_SUMMARY_NORM = tensor_tracer_flags.TT_SUMMARY_NORM
_TT_SUMMARY_MAX = tensor_tracer_flags.TT_SUMMARY_MAX
_TT_SUMMARY_MAX_ABS = tensor_tracer_flags.TT_SUMMARY_MAX_ABS
_TT_SUMMARY_MIN = tensor_tracer_flags.TT_SUMMARY_MIN
_TT_SUMMARY_MEAN = tensor_tracer_flags.TT_SUMMARY_MEAN
_TT_SUMMARY_VAR = tensor_tracer_flags.TT_SUMMARY_VAR
_TT_SUMMARY_SIZE = tensor_tracer_flags.TT_SUMMARY_SIZE
_TT_SUMMARY_SPARSITY = tensor_tracer_flags.TT_SUMMARY_SPARSITY

_TT_SUMMARY_TAG = 'tensor_tracer_summary'
_TT_TENSORBOARD_PLUGIN_NAME = 'tensor_tracer'
_TT_HOSTCALL_KEY = 'tensor_tracer_host_call'
_TT_EVENT_FILE_SUFFIX = '.tensor_tracer'

_TT_SUMMARY_MAX_QUEUE = 10

tt_gauge = monitoring.BoolGauge('/tensorflow/api/tensor_tracer/v1',
                                'tensor tracer usage', 'method')


def _graph_summary_tag(graph):
  """Generates and returns a summary tag name for the given graph."""

  if graph is None:
    raise RuntimeError('graph is None')
  # The chance of collision with md5 is effectively 0.
  hash_id = hashlib.md5()
  hash_id.update(repr(graph).encode('utf-8'))
  # hexdigest() returns a string.
  return hash_id.hexdigest()


def set_parameters(tensor_tracer_params=None):
  """Enables tensor tracer and sets its parameters.

  Example usage:
    tensor_tracer_parameters = {'trace_dir': '/usr/tmp/trace_dir',
                                'trace_mode': 'norm',
                                'report_file': '/usr/tmp/trace_dir/report.all'}
    tensor_tracer.set_parameters(tensor_tracer_parameters)

  This sets up the parameters for tensor tracer. A call to tensor tracer as
  below is necessary to enable debugging on CPUs and GPUs. On TPUs below can be
  skipped as this call is hooked into tpu.rewrite.
    tt = tensor_tracer.TensorTracer()
    loss = tt.trace_cpu(tf.get_default_graph(), tensor_fetches=loss)

  Args:
    tensor_tracer_params: Tensor tracer parameter dictionary. Below gives
    examples of these parameters: See tensor_tracer_report.py for all
      parameters.
        - enable: If set, tensor tracer will be enabled. Calling
          enable_tensor_tracer automatically adds this parameters.
        - trace_mode: The trace_mode to be used by tensor tracer. These include:
          - summary: Collects multiple statistics for traced tensors, and writes
            them a summary file that can be visualized using tensorboard. This
            mode currently only works for TPUEstimator. It can be also be used
            for other models, but outfeed must be handled by the user.
          - norm: Collects norm of each traced tensor and writes them into a
            text file pointed by 'trace_dir' flag. (Default mode).
          - nan-inf: Checks the existince of NaNs and Infs in the tensor, and
            writes a boolean value to a text file pointed by 'trace_dir' flag.
            Note that 'norm' mode can also capture this information with more
            numerical info.
          - max-abs: Collects the absolute max for each traced tensors and
            writes it into a text file pointed by 'trace_dir' flag.
          - full-tensor: Writes the full tensor content of the traced tensors
            into a text file pointed by 'trace_dir' flag.
          - part-tensor: Writes a part of the tensor content of the traced
            tensors into a text file pointed by 'trace_dir' flag.
          - full_tensor_summary: Writes the full tensors as binary event files.
            The outputs can be read using: trace =
              tensor_tracer.read_tensor_tracer_event_file(event_file_path)

        - report_file: Path to the metadata file that is written during graph
          construction. If not set, metadata will be printed to stdout during
          graph construction.
        - trace_dir: Path where the execution traces will be written during the
          graph execution. If not set, trace will be printed to stderr.
        - trace_level: Tensor tracer aims to trace everything it can. This
          introduces some overhead on graph execution and graph compilation
          times. Using trace_level parameter, it is possible to trace operation
          based on their priorities. For example, - trace_level=7 is the highest
          trace_level, in which every op is traced. - trace_level=6 will skip
          constant operations such as tf.constant. - trace_level=5 will skip
          less important ops such as tf.identities. - The default trace_level=3,
          that will skip concat ops, or random number generators. - To reduce
          the graph compile time overhead, trace_level can be set to 0, that
          will skip additions, and substractions, and multiplications as well.
        - excluded_opnames: If set, any matching op name will not be traced.
          excluded_opnames can be set as a regular expression. E.g,
          excluded_opnames=.* will exclude everything.
        - excluded_optypes: If set, any matching op type will not be traced.
          excluded_optypes can be set as a regular expression. E.g,
          excluded_optypes=.* will exclude everything. excluded_optypes=MatMul
          will exclude all MatMul ops from tracing.
        - included_opnames: If set, any matching op name will be forced to be
          traced. included_opnames can be set as a regular expression. E.g,
          '--included_opnames=some_op --excluded_opname=*.' will only trace
          some_op.
        - included_optypes: If set, any matching op type will be forced to be
          traced. included_optypes can be set as a regular expression. E.g,
          '--included_optypes=some_op_type --excluded_optypes=*.' will trace
          only the ops with type 'some_op_type'
        - flush_summaries: If summary mode is used, flush_summaries=1 will
          flush summaries using outside compilation. Note that, if used with
          low level APIs, flush_summaries=1 is necessary to obtain results.
        Advanced Flags:
        - trace_scalar: Scalar values are not traced by default. If this flag is
          set, scalar values will also be traced.
        - op_range: In the form of '%d:%d' that limits the tracing to the ops
          within this limit. --op_range='5:10' will trace only the ops that have
            topological order between 5-10.
        - submode: 'brief' or 'detailed'. If the trace mode is not compact,
          brief mode will print only the id of each traced tensor to save some
          space. 'detailed' mode prints the full tensor name.
        - use_fingerprint_subdirectory: The trace directory will be chosen as
          using the fingerprint of the trace metadata under the provided
          trace_dir.
  """
  flags = '--%s=1' % tensor_tracer_flags.FLAG_NAME_ENABLE
  if tensor_tracer_params:
    for key, value in tensor_tracer_params.items():
      flags += ' --%s=%s' % (key, value)
  os.environ[tensor_tracer_flags.FLAGS_ENV_VAR] = flags


def op_priority(op_type):
  """Returns the priority of the op.

  If the priority of the op is k, it will be traced if trace_level>=k.
  Args:
    op_type: String name of the operation type.
  Returns:
    Integer value corresponding the priority of the op.
  """
  if op_type in ('Const', 'Shape', 'BroadcastGradientArgs', 'Range',
                 'VariableShape', 'Fill', 'OneHot', 'ShapeN'):
    # Lowest priority ops, e.g., constant ops across different steps,
    # They will be traced only if trace_level>=7
    return 7

  if op_type in ('Identity', 'Cast', 'Reshape', 'ExpandDims', 'StopGradient',
                 'PreventGradient', 'Squeeze', 'Gather', 'GatherNd'):
    # Operations without numerical effects.
    # They will be only if trace_level>=6
    return 6
  if op_type in ('ConcatV2', 'Concat', 'StridedSlice', 'Slice', 'Pack', 'Tile',
                 'CollectivePermute', 'SplitV', 'DynamicPartition'):
    # Operations that merge or slice an input, will be traced if trace_level>=5
    return 5
  if op_type in ('Pad', 'RandomUniformInt', 'GreaterEqual'):
    # Operations less likely to provide useful information,
    # will be traced if trace_level>=4
    return 4
  if op_type in ('Sum', 'AddV2', 'Add', 'AddN', 'BiasAdd', 'CrossReplicaSum'):
    # Add operations that are less likely create any issues, will be traced
    # if trace_level>=3 (default=3)
    return 3
  if op_type in ('Neg', 'Sub'):
    # Sub operations that are less likely create any issues, will be traced
    # trace_level>=2
    return 2
  if op_type in ('Mul', 'Square', 'MatMul', 'RandomUniform', 'Select',
                 'Maximum', 'Mean', 'Variance', 'Exp', 'Rsqrt'):
    # Multiplication and some other operations, will be traced if trace_level>=1
    return 1

  # Unclassified op_types default to being traced at level 2 and above.
  return 2


def read_tensor_tracer_event_file(event_file):
  """Reads the event file written by tensor tracer.

  This can be used to read the full tensors written into binary event files by
  by TensorTracer with trace_mode=full_tensor_summary.

  Example usage:
    result_dict_list = tensor_tracer.read_tensor_tracer_event_file(
      event_file_path)
    for result_dict in result_dict_list:
      for step, tensor_dict in result_dict.items():
        for tensor_name, full_tensor_content in tensor_dict.items():
          logging.info(tensor_name, full_tensor_content)

  Args:
    event_file: Path to the event file that contains only tensor tracer events.
  Returns:
    A list of event dictionaries, each of which with the form:
    {step_number: {tensor_name: tensor_content}}. This is a list instead of
    a single event dictionary because it is possible that an event file may
    have multiple event traces, each of them covering the same step ranges.
  Raises:
    ValueError: If an unexpected trace is found.
  """

  # Keeps track of how many times that a step number shows up in these events.
  step_occurrence_count = collections.defaultdict(int)

  # List of step occurrences.
  step_occurrence_list = []

  for trace_event in summary_iterator.summary_iterator(event_file):
    # First event is an event with file_version: "brain.Event:2"
    if not trace_event.HasField('summary'):
      continue
    if len(trace_event.summary.value) != 1:
      raise ValueError('Single step contains %d summary values,'
                       ' expected 1.' % len(trace_event.summary.value))
    step = trace_event.step
    step_occurrence_count[step] += 1  # a new occurrence for this step.

    occurrence_idx = step_occurrence_count[step] - 1
    occurrence_size = len(step_occurrence_list)

    if occurrence_idx == occurrence_size:
      # This particular occurrence isn't yet recorded on step_occurrence_list.
      # So append this new occurrence to the end of step_occurrence_list.
      new_occurrence = collections.defaultdict(dict)
      step_occurrence_list.append(new_occurrence)
    else:
      # This particular occurrence must be already recorded on
      # step_occurrence_list (i.e. occurrence_idx < occurrence_size).
      if occurrence_idx > occurrence_size:
        raise ValueError('Unexpected: occurrence_idx (%d) > '
                         'occurrence_size (%d)' % (occurrence_idx,
                                                   occurrence_size))
    tensor_value = trace_event.summary.value[0]
    tensor_name = tensor_value.tag

    real_shape = [d.size for d in tensor_value.tensor.tensor_shape.dim]
    tensor_content = np.frombuffer(
        tensor_value.tensor.tensor_content,
        dtypes.DType(tensor_value.tensor.dtype).as_numpy_dtype()
        ).reshape(real_shape)
    step_occurrence_list[occurrence_idx][step][tensor_name] = tensor_content
  return step_occurrence_list


def trace_tensor(tensor, tracepoint_name=None):
  """Programmatic interface to trace a tensor with Tensor Tracer.

  Tensor Tracer, by default, traces all tensors in the execution. This function
  can be used to limit traced tensors. If this function is called for a subset
  of the tensors, only those will be traced.

  For example, Tensor Traacer will only trace c below.
    c = tf.MatMul(a, b)
    tensor_tracer.trace_tensor(c)
    d = tf.add(c, 1)
  Args:
     tensor: the tensor object for which the tracing is requested.
     tracepoint_name: an optional tensor tracepoint name string. A tracepoint
       name is an Tensor Tracer internal name for the tensor. It is useful when
       comparing equivalent traces from different models that have different
       tensor namings. Equivalent tensors (with different names) can be mapped
       to each other by assigning a common tracepoint_name.

  Returns:
    The provided tensor.
  """
  if tracepoint_name is None:
    tracepoint_name = tensor.name
  tensor.graph.get_collection(_TENSOR_TRACER_COLLECTION)
  tensor.graph.add_to_collection(_TENSOR_TRACER_COLLECTION,
                                 (tensor, tracepoint_name))
  return tensor


def keras_layer_tracepoint(layer, checkpoint_name):
  """An interface for adding the tensor outputs of a keras layer.

  Encapsulates trace_tensor.

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
    if tensor_util.is_tf_type(outputs):
      trace_tensor(outputs, '%s' % (checkpoint_name))
    else:
      idx = 0
      for output_tensor in outputs:
        if tensor_util.is_tf_type(outputs):
          trace_tensor(output_tensor, '%s_%d' % (checkpoint_name, idx))
        idx += 1
  except AttributeError:
    pass
  except RuntimeError:
    pass
  return layer


class TensorTracer(object):
  """A software construct for tracing tensor values in a TF graph.

  This utility is disabled by default. It is hooked into tpu.rewrite, so it can
  easily be enabled on TPUs by setting the TENSOR_TRACER_FLAGS env variable as
  below without a code change.
    export TENSOR_TRACER_FLAGS="--enable=1"

  Below is the use example to enable it on CPUs or GPUs, or for more advance use
  cases on TPUs.

    a = x + 1
    b = a * 2
    rs = tf.reduce_sum(b)
    tensor_tracer.set_parameters({'trace_dir': 'path/to/trace_dir',
                             'report_file: 'path/to/report/file'})
    tt = tensor_tracer.TensorTracer()
    if on_tpu:
      rs = tt.trace_tpu(tf.get_default_graph(),
                          tensor_fetches=rs)
    else:
      rs = tt.trace_cpu(tf.get_default_graph(),
                          tensor_fetches=rs)
    session.run(rs)

  If it is enabled, it will trace the output tensor values of
  selected Ops in the graph. It has two outputs: (1) the traces and (2)
  a report. The traces are dumped to a specified directory during the graph
  execution, while the report is dumped during the graph construction.
  By passing options via the env variable, users can change:
     (1) the trace mode (e.g., detecting NaN/Inf, printing partial or
         full tensor values)
     (2) which Ops to be traced (via op.name or op.type)
     (3) output trace file path.

  """
  # The set of graphs that are rewritten by tensor tracer.
  _traced_graphs = set()

  @staticmethod
  def is_enabled():
    """Returns True if TensorTracer is enabled."""
    try:
      enable = tensor_tracer_flags.TTParameters().is_enabled()
      # Add metrics to determine API usage.
      if enable: tt_gauge.get_cell('is_enabled').set(True)
      return enable
    except (ValueError, RuntimeError) as e:
      logging.warning(
          'Tensor Tracer V1 flags processing error encountered in is_enabled '
          'check. %s', e)
      # TODO(b/210212559): Find a more robust fix.
      # Should only produce exception if Tensor Tracer is enabled.
      return True

  @staticmethod
  def check_device_type(device_type):
    """Checks if the given device type is valid."""

    if device_type not in (_DEVICE_TYPE_TPU, _DEVICE_TYPE_CPU):
      raise ValueError('Invalid device_type "%s"'%device_type)

  @staticmethod
  def check_trace_mode(device_type, trace_mode):
    """Checks if the given trace mode work on the given device type.

    Args:
      device_type: Device type, TPU, GPU, CPU.
      trace_mode: Tensor tracer trace mode.
    Raises:
      ValueError: If the given trace mode is not supported for the device.
    """
    if trace_mode == tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY:
      if device_type != _DEVICE_TYPE_TPU:
        raise ValueError('Device_type "%s" is not yet supported for '
                         'trace mode "%s"' % (device_type, trace_mode))

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
  def control_flow_op(op):
    """Returns true if op is one of the special ops of in a while loop.

    Args:
       op: A tf.Operation.

    Returns:
       True if the given op is one of [Switch, Merge, Enter, Exit,
       NextIteration, LoopCond], which are all building blocks for TF while
       loops.
    """
    return  (control_flow_util.IsSwitch(op) or
             control_flow_util.IsMerge(op))

  @staticmethod
  def unsafe_op(op):
    """Returns True if this op is not safe to be traced."""

    # Reasons for not including following op types:
    #    Assign: cause incorrect result with CPU tracing.
    if op.type == 'Assign':
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
    if op.type in ('LoopCond', 'Enter', 'Merge', 'Const',
                   'Switch', 'Less', 'ReadVariableOp'):
      return True
    # Tracing the following will cause casting-issue
    # with the norm tracing mode or other compilation issues on CPU.
    if op.type in ('VarHandleOp', 'IteratorToStringHandle',
                   'IteratorGetNext', 'OneShotIterator',
                   'IteratorV2', 'MakeIterator',
                   'BatchDatasetV2', 'MapDataset',
                   'FixedLengthRecordDataset', 'TakeDataset', 'ZipDataset',
                   'Placeholder', 'PlaceholderWithDefault', 'StridedSlice'):
      return True
    return False

  def _is_interesting_op(self, op):
    """Returns True if the given op is not an interesting one to be traced."""
    return op_priority(op.type) <= self._parameters.trace_level

  @staticmethod
  def reason(op_idx, details):
    """Returns reason why the Op at op_idx is traced or not."""

    return '%d %s'%(op_idx, details)

  def __init__(self):
    """Initializes a TensorTracer.

    Sets the various member fields from the flags (if given) or the defaults.
    """
    self._replica_id = None
    self._tt_config = tensor_tracer_report.TensorTracerConfig()
    self._parameters = None
    self._host_call_fn = {}
    # _cache_variables is a dict (key = graph, value = dicts
    # (key = name, value = tensors))
    self._cache_variables = {}
    self._traced_op_names = set()
    self._report_proto = None
    # _temp_cache_var is a dict (key = graph, value = [])
    self._temp_cache_var = {}
    self._report_proto_path = ''
    self._outmost_context = None

  def report_proto(self):
    """Getter for tensor_tracer.proto object for summary and full_tensor_summary modes.

    Returns:
      A tensor_tracer.proto object.
    Raises:
      ValueError if called before tracing happens, or when trace mode is not
      summary or full_tensor_summary.
    """
    if self._report_proto:
      return self._report_proto
    else:
      raise ValueError('Call to report_proto must be done after tracing.'
                       'Report proto only exists for '
                       'trace_mode=[summary|full_tensor_summary]')

  def report_proto_path(self):
    """Getter for path where tensor_tracer.proto object should be written.

    Returns:
      A string path.
    """
    return self._report_proto_path

  def _cache_variable_for_graph(self, graph):
    if graph not in self._cache_variables:
      self._cache_variables[graph] = {}
    return self._cache_variables[graph]

  def _create_or_get_tensor_values_cache(self, cache_name, graph,
                                         shape=None, dtype=dtypes.float32):
    """Creates a variable as the cache to store intermediate tensor values.

    Args:
      cache_name: Name to be given to the cache (an instance of tf.variable).
      graph: Tensorflow graph.
      shape: A list of dimensions.
      dtype: Data type of created cache.
    Returns:
      A ref to newly created or existing cache with the given dimensions.
    Raises:
      ValueError:
        (1) If graph is None, or
        (2) shape is None when a new cache needs to be created.
    """

    def _escape_namescopes(variable_name):
      # TODO(deveci): This might cause name collisions as in "foo/bar/mytensor"
      # and "foo_bar/mytensor".
      return variable_name.replace('/', '_').replace(':', '_')

    if graph is None:
      raise ValueError('Invalid graph.')

    graph_cache_var = self._cache_variable_for_graph(graph)

    if cache_name not in graph_cache_var:
      if shape is None:
        raise ValueError('shape must be provided at cache creation.')
      if dtype.is_integer:
        init_val = int(_COMPACT_TRACE_ENTRY_INIT_VALUE)
      else:
        init_val = _COMPACT_TRACE_ENTRY_INIT_VALUE

      # Create in proper graph and base name_scope.
      with graph.as_default() as g, g.name_scope(None):
        graph_cache_var[cache_name] = variable_scope.get_variable(
            _TT_SNAPSHOT + '_' + _escape_namescopes(cache_name),
            shape=shape, dtype=dtype,
            initializer=init_ops.constant_initializer(init_val),
            trainable=False,
            use_resource=True,
            collections=[_TENSOR_TRACER_STORAGE, ops.GraphKeys.LOCAL_VARIABLES])
    return graph_cache_var[cache_name]

  def _add_replica_id_to_graph(self):
    """Adds nodes for computing the replica ID to the graph."""

    if self._tt_config.num_replicas:
      with ops.control_dependencies(None):
        # Uses None as dependency to run outside of TPU graph rewrites.
        self._replica_id = tpu_ops.tpu_replicated_input(
            list(range(self._tt_config.num_replicas)),
            name='tt_replica_id')
    else:
      self._replica_id = 'unknown'

  def _inside_op_range(self, idx):
    """Return True if the given index is inside the selected range."""

    if idx < self._parameters.op_range[0]:
      return False
    return (self._parameters.op_range[1] < 0 or
            idx <= self._parameters.op_range[1])

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
    for opname_re in self._parameters.included_opname_re_list:
      if opname_re.match(op.name):
        return True

    for optype_re in self._parameters.included_optype_re_list:
      if optype_re.match(op.type):
        return True
    return False

  def _is_user_excluded_op(self, op):
    for opname_re in self._parameters.excluded_opname_re_list:
      if opname_re.match(op.name):
        return True
    for optype_re in self._parameters.excluded_optype_re_list:
      if optype_re.match(op.type):
        return True
    return False

  def _signature_types(self):
    """Returns a dictionary holding the order of signatures in the cache for the selected trace mode."""
    if self._parameters.trace_mode in set([
        tensor_tracer_flags.TRACE_MODE_NAN_INF,
        tensor_tracer_flags.TRACE_MODE_NORM,
        tensor_tracer_flags.TRACE_MODE_MAX_ABS]):
      return {self._parameters.trace_mode: 0}
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_SUMMARY:
      return self._parameters.summary_signatures
    return {}

  def _num_signature_dimensions(self):
    return len(self._signature_types())

  def _use_temp_cache(self):
    """Returns true if the intermediate values should be stacked instead of being stored in a tf.Variable.

    Returns:
      A boolean, denoting whether to use a temporary cache or not.
    """
    # If full tensors need to be stored tf.variables, then do not use temp
    # variables to store them.
    if self._use_tensor_buffer():
      return False
    if self._use_tensor_values_cache():
      return self._parameters.use_temp_cache_var
    else:
      # Temporary caches only replaces tf.Variables caches. If no cache is used
      # return False.
      return False

  def _use_tensor_values_cache(self):
    """Returns True if immediate tensors should be first saved to a cache."""
    return self._parameters.use_compact_trace

  def _use_tensor_buffer(self):
    """Returns true if the whole tensor needs to be cached/buffered in memory."""
    return (self._parameters.trace_mode ==
            tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY)

  def _merge_tensor_signatures(self, signatures):
    """Returns a tensor that merges the given signatures.

    Args:
      signatures: A dictionary of the signature updates from signature name to
      a tensor of dimension [1].
    Returns:
      A tensor that concats the signature values in a predefined order.
    Raises:
      ValueError: Unable to merge signatures.
    """
    sorted_update = []
    if self._num_signature_dimensions() > 1:
      signature_indices = self._signature_types()
      for _, val in sorted(signatures.items(),
                           key=lambda item: signature_indices[item[0]]):
        sorted_update.append(val)
      updates = array_ops.stack(
          sorted_update, axis=0, name='merge_single_op_signatures')
    elif self._num_signature_dimensions() == 1:
      # Avoid stack operation if there is only a single signature.
      (_, val), = signatures.items()
      updates = val
    else:
      raise ValueError('Cannot merge 0 signatures. Check the value passed for '
                       'flag --signatures.')
    return updates

  def _save_tensor_value_to_tmp_cache(self, cache_idx, updates, graph):
    """Returns an op that will save the given updates to an entry in the cache.

    Args:
      cache_idx: The cache index of the tensor within the cache.
      updates: A dictionary of the signature updates from signature name to
      a tensor of dimension [1].
      graph: A TensorFlow graph.
    Raises:
      RuntimeError:
        (1) graph is not already in self._temp_cache_var, or
        (2) cache_idx is out of range.
    """
    updates = self._merge_tensor_signatures(updates)
    updates = array_ops.reshape(updates,
                                [self._num_signature_dimensions()])
    if graph not in self._temp_cache_var:
      raise RuntimeError('graph is not in self._temp_cache_var')
    if cache_idx >= len(self._temp_cache_var[graph]):
      raise RuntimeError('cache_idx (%d) is out of range (%d)' % (
          cache_idx, len(self._temp_cache_var[graph])))
    self._temp_cache_var[graph][cache_idx] = updates

  def _save_tensor_value_to_cache_op(self, cache_idx, updates, graph):
    """Returns an op that will save the given updates to an entry in the cache.

    Args:
      cache_idx: The cache index of the tensor within the cache.
      updates: A dictionary of the signature updates.
      graph: A TensorFlow graph.
    Returns:
      Cache update operation.
    """
    # state_ops.scatter_update allows updates only along the first dimension.
    # Make a compact array by concatenating different signatures, and update
    # them all together.
    updates = self._merge_tensor_signatures(updates)
    updates = array_ops.reshape(updates,
                                [1, self._num_signature_dimensions()])
    indices = constant_op.constant([cache_idx])
    cache = self._create_or_get_tensor_values_cache(_TT_SUMMARY_TAG, graph)
    return state_ops.scatter_update(cache, indices, updates).op

  def _snapshot_tensor(self, tensor):
    """Creates a new tf.Variable and a new tf.Operation that assigns the value of the tensor to this variable.

    Args:
      tensor: tensor whose values will be stored in a new tf.Variable.
    Returns:
      An assignment operation.
    """

    snapshot_variable = self._create_or_get_tensor_values_cache(
        tensor.name, tensor.op.graph,
        tensor.shape.as_list(), tensor.dtype)
    return state_ops.assign(snapshot_variable, tensor).op

  def _preprocess_traced_tensor(self, tensor):
    """Computes NAN/Norm/Max on TPUs before sending to CPU.

    Args:
      tensor: The tensor to be traced.
    Returns:
      A tensor that should be input to the trace_function.
    Raises:
      RuntimeError: If the signature is invalid.
    """

    def _detect_nan_inf(tensor):
      """Trace function for detecting any NaN/Inf in the tensor."""

      if tensor.dtype.is_floating:
        mask = math_ops.reduce_any(
            gen_math_ops.logical_or(
                gen_math_ops.is_nan(tensor), gen_math_ops.is_inf(tensor)))
        output_tensor = control_flow_ops.cond(
            mask,
            lambda: constant_op.constant([1.0]),
            lambda: constant_op.constant([0.0]))
      else:
        output_tensor = constant_op.constant([0.0])
      return output_tensor

    def _compute_signature(tensor, tf_op, cast_to_f32=True):
      if cast_to_f32:
        tensor = math_ops.cast(tensor, dtypes.float32)
      output_tensor = tf_op(tensor)
      # Return type should be scalar. Set it if it does not have the
      # information.
      if not output_tensor.get_shape().is_fully_defined():
        output_tensor = array_ops.reshape(output_tensor, [])
      return output_tensor

    def _show_size(tensor):
      # In order to check the size of a tensor.
      # Not all sizes are known at the compile time, also, different replicas
      # sometimes get different sizes of tensors.
      # Collect it here to be used in merging replica data.
      tsize = _compute_signature(tensor, array_ops.size, cast_to_f32=False)
      # Cast to float32, so that it can be placed into same cache with other
      # signatures.
      return math_ops.cast(tsize, dtypes.float32)

    def _show_max(tensor, cast_to_f32=True):
      # returns -inf for empty tensor
      return _compute_signature(tensor, math_ops.reduce_max, cast_to_f32)

    def _show_min(tensor, cast_to_f32=True):
      # returns inf for empty tensor
      return _compute_signature(tensor, math_ops.reduce_min, cast_to_f32)

    def _show_norm(tensor, cast_to_f32=True):
      # returns 0 for empty tensor
      return _compute_signature(tensor, linalg_ops.norm, cast_to_f32)

    def _show_sparsity(tensor, cast_to_f32=True, tolerance=1e-06):
      # returns nan for empty tensor and treats nans as non-zero numbers
      def sparsity_fn(tensor):
        non_zeros = math_ops.greater_equal(math_ops.abs(tensor), tolerance)
        nans = math_ops.is_nan(tensor)
        return nn_impl.zero_fraction(math_ops.logical_or(non_zeros, nans))

      return _compute_signature(tensor, sparsity_fn, cast_to_f32)

    def _show_mean_and_variance(tensor, cast_to_f32=True):
      """Returns the mean and variance of the given tensor."""
      if cast_to_f32:
        tensor = math_ops.cast(tensor, dtypes.float32)
      # returns nan for empty tensor
      mean, var = nn_impl.moments(array_ops.reshape(tensor, [-1]), axes=[0])
      # The shape has to be 1. Set it if it does not have the information.
      if not mean.get_shape().is_fully_defined():
        mean = array_ops.reshape(mean, [])
      if not var.get_shape().is_fully_defined():
        var = array_ops.reshape(var, [])
      return mean, var

    def _show_max_abs(tensor, cast_to_f32=True):
      return _compute_signature(
          tensor, lambda t: math_ops.reduce_max(math_ops.abs(t)), cast_to_f32)

    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NAN_INF:
      return {self._parameters.trace_mode: _detect_nan_inf(tensor)}
    if (self._parameters.trace_mode ==
        tensor_tracer_flags.TRACE_MODE_PART_TENSOR):
      return {self._parameters.trace_mode: tensor}
    if (self._parameters.trace_mode in (
        tensor_tracer_flags.TRACE_MODE_FULL_TENSOR,
        tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY)):
      return {self._parameters.trace_mode: tensor}
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NORM:
      return {self._parameters.trace_mode: array_ops.reshape(
          _show_norm(tensor), [1])}
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_MAX_ABS:
      return {self._parameters.trace_mode: _show_max_abs(tensor)}

    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_SUMMARY:
      tensor = math_ops.cast(tensor, dtypes.float32)
      result_dict = {}
      # Call mean and variance computation here to avoid adding the same nodes
      # twice.
      if (_TT_SUMMARY_MEAN in self._signature_types() or
          _TT_SUMMARY_VAR in self._signature_types()):
        mean, variance = _show_mean_and_variance(tensor, cast_to_f32=False)

      for signature_name, _ in sorted(self._signature_types().items(),
                                      key=lambda x: x[1]):
        if signature_name == _TT_SUMMARY_NORM:
          signature_result_tensor = _show_norm(tensor, cast_to_f32=False)
        elif signature_name == _TT_SUMMARY_MAX:
          signature_result_tensor = _show_max(tensor, cast_to_f32=False)
        elif signature_name == _TT_SUMMARY_MAX_ABS:
          signature_result_tensor = _show_max_abs(tensor, cast_to_f32=False)
        elif signature_name == _TT_SUMMARY_MIN:
          signature_result_tensor = _show_min(tensor, cast_to_f32=False)
        elif signature_name == _TT_SUMMARY_SPARSITY:
          signature_result_tensor = _show_sparsity(tensor)
        elif signature_name == _TT_SUMMARY_SIZE:
          signature_result_tensor = _show_size(tensor)
        elif signature_name == _TT_SUMMARY_MEAN:
          signature_result_tensor = mean
        elif signature_name == _TT_SUMMARY_VAR:
          signature_result_tensor = variance
        else:
          raise ValueError('Unknown signature type :%s.' % signature_name)

        result_dict[signature_name] = signature_result_tensor
      return result_dict

    raise RuntimeError(
        'Unsupported signature for trace mode %s.'
        % self._parameters.trace_mode)

  def _make_tensor_trace_fun(self, tensor_name, tensor_trace_order):
    """Makes the tensor tracing function called by outside compilation.

    Args:
      tensor_name: name of the tensor being traced.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.
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
                    tensor_trace_order.tensorname_to_cache_idx.
      """

      if self._parameters.is_brief_mode():
        if tensor_name not in tensor_trace_order.tensorname_to_cache_idx:
          raise ValueError(
              'Tensor %s with name %s is not in the tensorname_to_cache_idx' %
              (tensor, tensor_name))
        msg = '%d' % tensor_trace_order.tensorname_to_cache_idx[tensor_name]
      else:
        msg = '"%s"' % tensor_name

      if self._parameters.trace_dir:
        output_path = os.path.join(
            self._parameters.trace_dir,
            _TRACE_FILE_NAME + self._get_outfile_suffix())
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

      return _print_tensor(tensor_name, _TRACE_MODE_PART_TENSOR_SIZE,
                           tensor, tensor)

    def _show_full_tensor(tensor):
      """Trace function for printing the entire tensor."""

      return _print_tensor(tensor_name, -1, tensor, tensor)

    if (self._parameters.trace_mode ==
        tensor_tracer_flags.TRACE_MODE_PART_TENSOR):
      return _show_part_tensor
    # The input tensor has a shape of "[1]" for TRACE_MODE_NAN_INF,
    # TRACE_MODE_NORM, and TRACE_MODE_MAX_ABS, as related computations are
    # performed within TPUs and only their results are transferred to CPU.
    # Simply, print the full tensor for these trace modes.
    if self._parameters.trace_mode in (
        tensor_tracer_flags.TRACE_MODE_NAN_INF,
        tensor_tracer_flags.TRACE_MODE_NORM,
        tensor_tracer_flags.TRACE_MODE_FULL_TENSOR,
        tensor_tracer_flags.TRACE_MODE_MAX_ABS,
        tensor_tracer_flags.TRACE_MODE_SUMMARY
        ):
      return _show_full_tensor

    raise RuntimeError('Full tensor support is not available with trace mode %s'
                       %self._parameters.trace_mode)

  def _is_in_control_flow(self, op):
    """Returns true if the given op is inside a tf.cond or in tf.while_loop.

    Args:
      op: A tensorflow op that should be checked whether in control flow or not.
    Returns:
      A boolean value whether the op is in control flow or not.
    """
    return control_flow_util.IsInCond(op)

  def _is_in_outmost_while_loop(self, op):
    """Returns true if the op is at the same level with the training loop.

    Returns false if the op is in an inner while loop or if it is outside of the
    training loop.
    Args:
      op: tf.Operation

    Returns:
      A boolean.
    """
    ctxt = self._get_op_control_flow_context(op)
    outer_while_context = control_flow_util.GetContainingWhileContext(ctxt)
    return outer_while_context == control_flow_util.GetContainingWhileContext(
        self._outmost_context)

  def _should_trace_in_control_flow(self):
    """Returns false incase it is not safe to trace ops in tf.cond or tf.while_loop."""
    # As different from the other trace modes, TRACE_MODE_OPTIONAL_SUMMARY
    # forces the execution of the traced tensors. We should not trace the ops
    # that may not be executed due to control flow.
    if self._use_temp_cache():
      return False
    elif self._tt_config.device_type == _DEVICE_TYPE_TPU:
      # On TPUs do not trace in control flow unless we use caches to store
      # intermediate values as calling outside compilation within an inner loop
      # causes errors.
      return self._use_tensor_values_cache() or self._use_tensor_buffer()
    return True

  def _skip_op(self, op_id, op, ops_in_exec_path, report_handler):
    """Returns True if we should not trace Op.

    Args:
      op_id: Topological index of the op.
      op: tf.Operation
      ops_in_exec_path: Set of operations that are in the execution path.
      report_handler: An instance of tensor_tracer_report.TTReportHandle.
    Returns:
      True if the op should not be traced, false otherwise.
    """
    if TensorTracer.while_loop_op(op):
      report_handler.instrument_op(
          op, TensorTracer.reason(op_id, _REASON_WHILELOOP_OP))
      return True
    if TensorTracer.control_flow_op(op):
      report_handler.instrument_op(
          op, TensorTracer.reason(op_id, _REASON_CONTROLFLOW_OP))
      return True
    if TensorTracer.unsafe_op(op):
      report_handler.instrument_op(
          op, TensorTracer.reason(op_id, _REASON_UNSAFE_OP))
      return True
    if TensorTracer.device_mismatch(self._tt_config.device_type, op):
      report_handler.instrument_op(
          op, TensorTracer.reason(op_id, _REASON_DEVICE_MISMATCH))
      return True
    if op not in ops_in_exec_path:
      report_handler.instrument_op(
          op, TensorTracer.reason(op_id, _REASON_NOT_EXECUTED))
      return True
    # TensorTracer will not trace the operations that are in an inner while loop
    # or tf.cond when a temporary cache is used. Temporary cache adds direct
    # data dependencies to traced operations, and needs a static number of
    # traced operations. For these cases,
    # - We do not know the number of slots required when there are inner while
    # loops. TensorTracer can only trace the result of a while loop.
    # - We do not know ahead of time which branch of the tf.cond
    # will be taken, so we avoid introducing data dependencies for the
    # operations inside a tf.cond.
    # - We also cannot have a data dependency to an operation in a different
    # while context.
    if self._is_in_control_flow(op) or not self._is_in_outmost_while_loop(op):
      if not self._should_trace_in_control_flow():
        report_handler.instrument_op(
            op, TensorTracer.reason(op_id, _REASON_IN_CONTROL_FLOW))
        return True
    if self._is_user_included_op(op):
      report_handler.instrument_op(
          op, TensorTracer.reason(op_id, _REASON_USER_INCLUDED))
      return False

    if not self._inside_op_range(op_id):
      report_handler.instrument_op(
          op, TensorTracer.reason(op_id, _REASON_OUTSIDE_OP_RANGE))
      return True
    if not self._is_interesting_op(op):
      report_handler.instrument_op(
          op, TensorTracer.reason(op_id, _REASON_LESS_INTERESTING_OP))
      return True
    if self._is_user_excluded_op(op):
      report_handler.instrument_op(
          op, TensorTracer.reason(op_id, _REASON_USER_EXCLUDED))
      return True
    return False

  def _skip_tensor(self, op_id, out_tensor, report_handler):
    """Returns True if we should not trace out_tensor.

    Args:
      op_id: Topological index of the op producing tensor.
      out_tensor: tf.Tensor
      report_handler: An instance of tensor_tracer_report.TTReportHandle.
    Returns:
      True if the tensor should not be traced, false otherwise.
    """

    # Skips a tensor if the tensor has a non-numeric type.
    #   Note: we cannot use check_ops.is_numeric_tensor(out_tensor)
    #         because it also excludes tensors with dtypes, bool, and
    #         float32_ref, which we actually want to trace.
    non_numeric_tensor_types = set([dtypes.variant, dtypes.resource,
                                    dtypes.string])
    if out_tensor.dtype in non_numeric_tensor_types:

      report_handler.instrument_tensor(
          out_tensor, TensorTracer.reason(op_id, _REASON_NON_NUMERIC_TENSOR))
      return True
    # Skip a tensor if it feeds a special while loop op.
    if [consumer for consumer in out_tensor.consumers() if
        TensorTracer.while_loop_op(consumer)]:
      report_handler.instrument_tensor(
          out_tensor, TensorTracer.reason(op_id, _REASON_FEEDS_WHILELOOP_OP))
      return True
    if self._is_user_included_op(out_tensor.op):
      report_handler.instrument_tensor(
          out_tensor, TensorTracer.reason(op_id, _REASON_USER_INCLUDED))
      return False
    if self._is_user_excluded_op(out_tensor.op):
      report_handler.instrument_tensor(
          out_tensor, TensorTracer.reason(op_id, _REASON_USER_EXCLUDED))
      return True
    if not out_tensor.get_shape().is_fully_defined():
      # If trace mode is nan-inf, norm or max, then the tensor will be reduced
      # to a scalar before the outside compilation call.
      if self._parameters.trace_mode in (
          tensor_tracer_flags.TRACE_MODE_NAN_INF,
          tensor_tracer_flags.TRACE_MODE_NORM,
          tensor_tracer_flags.TRACE_MODE_MAX_ABS,
          tensor_tracer_flags.TRACE_MODE_SUMMARY
          ):
        report_handler.instrument_tensor(
            out_tensor, TensorTracer.reason(op_id, _REASON_TENSOR_GET_TRACED))
        return False
      else:
        report_handler.instrument_tensor(
            out_tensor, TensorTracer.reason(op_id, _REASON_DYNAMIC_SHAPE))
        return True
    rank = len(out_tensor.shape)
    if rank < 1:
      # scalar
      if self._parameters.trace_scalar_ops:
        if TensorTracer.unsafe_scalar_trace(out_tensor.op):
          report_handler.instrument_tensor(
              out_tensor, TensorTracer.reason(op_id, _REASON_UNSAFE_SCALAR))
          return True
        else:
          report_handler.instrument_tensor(
              out_tensor, TensorTracer.reason(op_id, _REASON_SCALAR_GET_TRACED))
          return False
      else:
        report_handler.instrument_tensor(
            out_tensor, TensorTracer.reason(op_id, _REASON_SKIP_SCALAR))
        return True
    else:
      # tensor
      report_handler.instrument_tensor(
          out_tensor, TensorTracer.reason(op_id, _REASON_TENSOR_GET_TRACED))
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

  def _determine_and_instrument_traced_tensors(self, graph_order,
                                               ops_in_exec_path,
                                               tensor_trace_points,
                                               report_handler):
    """Determines the tensors to trace and instruments the trace details.

    Args:
      graph_order: graph_order tuple containing graph (tf.graph), operations
        (list of operations), op_to_idx (op id mapping), (tensors) list of
        tensors, tensor_to_idx (tensor id mapping), contains_cycle (whether
        there is a cycle in the graph), topological_order_or_cycle (list of ops
        in topological order or list of ops creating a cycle).
      ops_in_exec_path: Set of ops in the execution path.
      tensor_trace_points: Collection of programatic tensor trace points.
      report_handler: An instance of tensor_tracer_report.TTReportHandle.
    Returns:
      List of tensors to be traced.
    """

    traced_tensors = []
    checkpoint_operations = set([tensor.op
                                 for (tensor, _) in tensor_trace_points])
    for op_id, op in enumerate(graph_order.operations):
      if checkpoint_operations and op not in checkpoint_operations:
        continue
      if self._skip_op(op_id, op, ops_in_exec_path, report_handler):
        continue
      for i in range(len(op.outputs)):
        out_tensor = op.outputs[i]
        if not self._skip_tensor(op_id, out_tensor, report_handler):
          traced_tensors.append(out_tensor)
    return traced_tensors

  def _check_trace_files(self):
    """Checks if any requirements for trace files are satisfied."""

    if not self._parameters.trace_dir:
      # traces will be written to stderr. No need to check trace files.
      return
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_SUMMARY:
      # Output files are handled by tf.summary operations, no need to precreate
      # them.
      return
    if not gfile.Exists(self._parameters.trace_dir):
      file_io.recursive_create_dir(self._parameters.trace_dir)
      if not gfile.Exists(self._parameters.trace_dir):
        raise RuntimeError('Failed to create trace directory at %s' %
                           self._parameters.trace_dir)

  def _create_temp_cache(self, num_traced_tensors, num_signatures, graph):
    """Creates a temporary cache with the given dimensions.

    Fills the self._temp_cache_var with num_traced_tensors tf.constant() ops
    that have shape of [num_signatures].
    Args:
      num_traced_tensors: Int, denoting total number of traced tensors.
      num_signatures: Int, denoting the number of statistics collected per
        tensors.
      graph: TensorFlow graph.
    """
    init_value = constant_op.constant(_COMPACT_TRACE_ENTRY_INIT_VALUE,
                                      dtype=dtypes.float32,
                                      shape=[num_signatures])
    self._temp_cache_var[graph] = [
        init_value for _ in range(num_traced_tensors)]

  def _determine_trace_and_create_report(self, graph, ops_in_exec_path,
                                         graph_summary_tag):
    """Work needs to be done prior to TPU or CPU tracing.

    Args:
      graph: tf.graph
      ops_in_exec_path: Set of operations in the execution path.
      graph_summary_tag: the summary tag name for the given graph.
    Returns:
      An instance of tensor_tracer_report.TensorTraceOrder, containing list of
      tensors to be traced with their topological order information.
    """

    self._check_trace_files()

    graph_order = tensor_tracer_report.sort_tensors_and_ops(graph)
    tensor_trace_points = graph.get_collection(_TENSOR_TRACER_COLLECTION)

    report_handler = tensor_tracer_report.TTReportHandle()
    traced_tensors = self._determine_and_instrument_traced_tensors(
        graph_order, ops_in_exec_path, tensor_trace_points, report_handler)
    logging.info('TensorTracer is tracing %d tensors.', len(traced_tensors))

    tensor_trace_order = tensor_tracer_report.TensorTraceOrder(graph_order,
                                                               traced_tensors)
    num_signatures = self._num_signature_dimensions()
    # Create a cache variable if compact_tracing is used.
    if num_signatures and self._use_tensor_values_cache():
      if self._use_temp_cache():
        self._create_temp_cache(len(traced_tensors), num_signatures, graph)
      else:
        self._create_or_get_tensor_values_cache(_TT_SUMMARY_TAG,
                                                graph,
                                                [len(traced_tensors),
                                                 num_signatures])
    if self._parameters.trace_mode in (
        tensor_tracer_flags.TRACE_MODE_SUMMARY,
        tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY):
      self._report_proto = report_handler.create_report_proto(
          self._tt_config, self._parameters, tensor_trace_order,
          tensor_trace_points, self._signature_types())
      if self._parameters.use_fingerprint_subdir:
        self._parameters.trace_dir = os.path.join(
            self._parameters.trace_dir, self._report_proto.fingerprint)
        logging.info('TensorTracer updating trace_dir to %s',
                     self._parameters.trace_dir)
      self._report_proto_path = report_handler.report_proto_path(
          self._parameters.trace_dir, graph_summary_tag)

      if self._parameters.report_file_path != _SKIP_REPORT_FILE:
        report_handler.write_report_proto(self._report_proto_path,
                                          self._report_proto, self._parameters)
    else:
      report_handler.create_report(self._tt_config, self._parameters,
                                   tensor_trace_order, tensor_trace_points)
    return tensor_trace_order

  def _create_host_call(self):
    return self._parameters.trace_mode in (
        tensor_tracer_flags.TRACE_MODE_SUMMARY,
        tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY)

  def _inspect_summary_cache(self, cache, replica_id, step_num, output_stream,
                             tensor_trace_order):
    """Generates a print operation to print trace inspection.

    Args:
      cache: Tensor storing the trace results for the step.
      replica_id: Tensor storing the replica id of the running core.
      step_num: Step number.
      output_stream: Where to print the outputs, e.g., file path, or sys.stderr.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.

    Returns:
      The Op to flush the cache to file.
    """
    def _inspect_tensor(tensor):
      """Returns the text to be printed for inspection output."""
      if (self._parameters.trace_mode ==
          tensor_tracer_flags.TRACE_MODE_NAN_INF):
        return control_flow_ops.cond(
            math_ops.greater(tensor, 0.0),
            lambda: 'has NaNs/Infs!',
            lambda: 'has no NaNs or Infs.')
      else:
        return tensor

    # Check if there are graph operations being profiled.
    if not tensor_trace_order.traced_tensors:
      logging.warn('Inspect mode has no tensors in the cache to check.')
      return control_flow_ops.no_op

    # Check if the cache includes any nan or inf
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_NAN_INF:
      # Cache has 1s or 0s if the mode is NaN_INF
      step_has_nan_or_inf = math_ops.greater(math_ops.reduce_sum(cache), 0.0)
    else:
      # Cache has the actual numerics for other modes.
      step_has_nan_or_inf = math_ops.reduce_any(
          gen_math_ops.logical_or(
              gen_math_ops.is_nan(cache), gen_math_ops.is_inf(cache)))

    # Summarizing message for each step.
    step_error_message = control_flow_ops.cond(
        step_has_nan_or_inf,
        lambda: 'NaNs or Infs in the step!',
        lambda: 'No numerical issues have been found for the step.')

    # No need to print core numbers if the cache is merged already.
    if self._parameters.collect_summary_per_core:
      stats = ['\n\n', 'core:', replica_id, ',', 'step:', step_num, '-->',
               step_error_message,
               'Printing tensors for mode:%s...' % self._parameters.trace_mode]
    else:
      stats = ['\n\n', 'step:', step_num, '-->', step_error_message,
               'Printing tensors for mode:%s...' % self._parameters.trace_mode]

    for tensor_name, cache_idx in sorted(
        tensor_trace_order.tensorname_to_cache_idx.items(),
        key=lambda item: item[1]):
      if self._parameters.collect_summary_per_core:
        stats.extend([
            '\n', 'core:', replica_id, ',', 'step:', step_num, ',',
            tensor_name, '-->', _inspect_tensor(cache[cache_idx, 0])])
      else:
        stats.extend([
            '\n', 'step:', step_num, ',',
            tensor_name, '-->', _inspect_tensor(cache[cache_idx, 0])])
    return logging_ops.print_v2(*stats, summarize=-1,
                                output_stream=output_stream)

  def _get_outfile_suffix(self):
    if remote_utils.is_remote_path(self._parameters.trace_dir):
      return remote_utils.get_appendable_file_encoding()
    else:
      return ''

  def _generate_flush_cache_op(self, num_replicas, on_tpu,
                               tensor_trace_order, graph):
    """Generates an Op that will flush the cache to file.

    Args:
      num_replicas: total number of replicas.
      on_tpu: if the graph is executed on TPU.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.
      graph: TensorFlow graph.

    Returns:
      The Op to flush the cache to file.
    """

    def _flush_fun(cache, replica_id, step_num):
      """Flushes the cache to a file corresponding to replica_id."""

      def _f(file_index):
        """Generates a func that flushes the cache to a file."""
        def _print_cache():
          """Flushes the cache to a file."""
          replica_str = ('%d' % file_index)
          if self._parameters.trace_dir:
            output_path = (os.path.join(self._parameters.trace_dir,
                                        _COMPACT_TRACE_FILE_PREFIX)
                           + replica_str + self._get_outfile_suffix())
            output_stream = _OUTPUT_STREAM_ESCAPE + output_path
          else:
            output_stream = sys.stderr

          new_step_line = _REPLICA_ID_TAG + replica_str
          print_ops = []
          if self._parameters.inspect_trace:
            if self._num_signature_dimensions() > 1:
              raise ValueError('Inspecting multi signatures are not supported.')
            print_ops.append(self._inspect_summary_cache(
                cache=cache, replica_id=replica_id, step_num=step_num,
                output_stream=output_stream,
                tensor_trace_order=tensor_trace_order))
          else:
            for i in range(self._num_signature_dimensions()):
              print_ops.append(logging_ops.print_v2(
                  new_step_line, '\n',
                  cache[:, i], '\n',
                  summarize=-1,
                  output_stream=output_stream))
          with ops.control_dependencies(print_ops):
            return constant_op.constant(0).op
        return _print_cache

      def _eq(file_index):
        return math_ops.equal(replica_id, file_index)

      flush_op_cases = {}
      flush_op_cases[_eq(0)] = _f(0)
      for i in range(1, num_replicas):
        if on_tpu and not self._parameters.collect_summary_per_core:
          # If this is the case, the cache is already merged for all cores.
          # Only first core flushes the cache.
          flush_op_cases[_eq(i)] = control_flow_ops.no_op
        else:
          flush_op_cases[_eq(i)] = _f(i)
      # Each replica needs to determine where to write their output.
      # To do this, we check if replica_id is 0, then 1, ..., and then
      # num_replicas - 1 statically; and return the corresponding static file
      # name. We cannot simply set the file name in python, as replica_id is
      # only known during tf runtime, and we cannot create dynamic filenames.
      return control_flow_ops.case(flush_op_cases, exclusive=True)

    cache = self._create_or_get_tensor_values_cache(_TT_SUMMARY_TAG, graph)
    if self._use_temp_cache():
      cache_val = cache
    else:
      cache_val = cache.value()

    if on_tpu:
      # If we do not need to collect traces for all cores, merge and aggregate
      # per core trace.
      if not self._parameters.collect_summary_per_core:
        cache_val = self.merge_caches_on_tpu(cache_val)
        cache_val = self.aggregate_global_cache(cache_val)[0]

      flush_op = tpu.outside_compilation(
          _flush_fun, cache_val, self._replica_id,
          array_ops.identity(training_util.get_or_create_global_step()))
    else:
      global_step = training_util.get_or_create_global_step()
      flush_op = _flush_fun(cache_val, self._replica_id, global_step)

    if self._use_temp_cache():
      with ops.control_dependencies([flush_op]):
        return constant_op.constant(0).op
    else:
      # Re-initialize the local cache variable.
      with ops.control_dependencies([flush_op]):
        reset_value = constant_op.constant(_COMPACT_TRACE_ENTRY_INIT_VALUE,
                                           dtype=cache.dtype,
                                           shape=cache.shape)
        assign_op = state_ops.assign(cache, reset_value).op
        with ops.control_dependencies([assign_op]):
          return constant_op.constant(0).op

  def _flush_tensor_values_cache(self, tensor_fetches, op_fetches, on_tpu,
                                 tensor_trace_order, graph):
    """Flushes the intermediate tensor values in the graph to the cache.

    Args:
      tensor_fetches: list of tensor results returned by the model_fn.
      op_fetches: list of ops that are returned by the model_fn, e.g., train_op.
      on_tpu: if the graph is executed on TPU.
      tensor_trace_order: TensorTraceOrder object holding tensorname to id map.
      graph: TensorFlow graph.

    Returns:
      An identical copy of tensor_fetches.
    """
    # Add a dependency to op and tensor fetches to make sure that all tracing
    # ops are executed before flushing trace results.
    if not tensor_trace_order.traced_tensors:
      logging.warn('No tensor values being traced. No flush cache op added.')
      return tensor_fetches
    with ops.control_dependencies(op_fetches +
                                  [tensor.op for tensor in tensor_fetches]):
      flush_cache_op = self._generate_flush_cache_op(
          self._tt_config.num_replicas, on_tpu, tensor_trace_order, graph)
      return control_flow_ops.tuple(tensor_fetches,
                                    control_inputs=[flush_cache_op])

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
      elif isinstance(fetch, ops.Tensor):
        fetches.append(fetch.op)
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

  def merge_caches_on_tpu(self, local_tpu_cache_tensor):
    """Merges the given caches on tpu.

    Args:
      local_tpu_cache_tensor: A local tensor that needs to be merged
        by concanting data from other tpu cores.
    Returns:
      A merged tf.Tensor.
    """
    x = array_ops.broadcast_to(
        local_tpu_cache_tensor,
        shape=[self._tt_config.num_replicas] +
        local_tpu_cache_tensor.shape.as_list())
    return tpu_ops.all_to_all(
        x, concat_dimension=0, split_dimension=0,
        split_count=self._tt_config.num_replicas,
        group_assignment=[list(range(self._tt_config.num_replicas))])

  def aggregate_global_cache(self, global_tt_summary_cache):
    """Merges the given caches on tpu.

    Args:
      global_tt_summary_cache: The global tensor tracer summary cache tensor
        with shape (num_cores, num_traced_tensors, num_traced_signatures). First
        dimension corresponds to core_id, where global_tpu_cache_tensor[i]
        correspond to the local cache from core-i.
    Returns:
      An aggregated tf.Tensor.
    Raises:
      RuntimeError: if there is no aggregate function defined for a signature.
    """

    # Merge only statistics tensor, if it is any other tensor we simply,
    # concatenate them.
    agg_fn_map = self._parameters.get_signature_to_agg_fn_map()
    signature_idx_map = self._signature_types()
    aggregation_result = []
    for signature, idx in sorted(signature_idx_map.items(),
                                 key=operator.itemgetter(1)):
      if signature not in agg_fn_map:
        raise RuntimeError('No aggregation function is defined for '
                           'signature %s.' % signature)
      # The dimensions of the statistics tensor is
      # num_cores x num_traced_tensors x num_signatures
      # value[:,:,idx] will return the portion of the tensor related
      # to signature.
      signature_tensor = global_tt_summary_cache[:, :, idx]
      # Merge it along the first (core) axis.
      agg_fn = agg_fn_map[signature]
      agg_tensor = agg_fn(signature_tensor, axis=0)
      aggregation_result.append(agg_tensor)
    # Merge results corresponding to different signatures

    merged_signatures = array_ops.stack(aggregation_result)
    # merged_signatures has dimensions
    # num_signatures x num_traced_tensors, transpose it so that it
    # will match with the original structure
    # num_traced_tensors x num_signatures.
    transposed_signatures = array_ops.transpose(merged_signatures)
    # Expand 1 more dimension so that it will match with the expected
    # structure num_cores x num_traced_tensors x num_signatures.
    return array_ops.expand_dims(transposed_signatures, axis=0)

  def _prepare_host_call_fn(self, processed_t_fetches,
                            op_fetches, graph, graph_summary_tag):
    """Creates a host call function that will write the cache as tb summary.

    Args:
      processed_t_fetches: List of tensor provided to session.run.
      op_fetches: List of operations provided to session.run.
      graph: TensorFlow graph.
      graph_summary_tag: the summary_tag name for the given graph.
    Raises:
      ValueError if trace_dir is not set.
    """
    if self._parameters.trace_dir is None:
      raise ValueError('Provide a trace_dir for tensor tracer in summary mode. '
                       '--trace_dir=/model/dir')

    def _write_cache(step, event_file_suffix=None, **kwargs):
      """Writes the given caches as tensor summary.

      Args:
        step: Step tensor with dimension [num_cores].
        event_file_suffix: Event filename suffix tensor.
        **kwargs: The dictionary of tensors that needs to be written as
          summaries. Key and value pairs within kwargs correspond to the tag
          name, and tensor content that will be written using summary.write.
          The trace_modes that use this function are:
            - summary: In summary mode, kwargs includes a single (tag, content)
            pair which are, _TT_SUMMARY_TAG and a tf.float32 signature_cache
            variable. The dimension of the signature_cache is:
              num_cores x num_traced_tensors x num_signatures.
            - full_tensor_summary: kwargs will include all traced tensors. Tag
            and content correspond to the name of the tensor, and its actual
            content.
      Returns:
        A tf.Operation that needs to be executed for the host call dependencies.
      """
      file_suffix = _TT_EVENT_FILE_SUFFIX
      if event_file_suffix is not None:
        file_suffix = string_ops.string_join([file_suffix, event_file_suffix],
                                             separator='.')
      # TODO(deveci): Parametrize max_queue, so that flushing op can be called
      # less frequently.
      # Setting max_queue to 100 appears to be safe even when the number of
      # iterations are much lower, as the destructor of the writer flushes it.
      summary_write_ops = []
      summary_writer = summary.create_file_writer_v2(
          self._parameters.trace_dir,
          filename_suffix=file_suffix,
          max_queue=_TT_SUMMARY_MAX_QUEUE)
      graph.add_to_collection(
          TENSOR_TRACER_SUMMARY_COLLECTION, summary_writer)

      step_value = step[0]
      dt = step_value.dtype

      # The step parameter to a summary write call must be 64-bit.
      if dt.__ne__(dtypes.int64) and dt.__ne__(
          dtypes.uint64) and dt.__ne__(dtypes.float64):
        step_value = math_ops.cast(step_value, dtypes.int64)

      with summary_writer.as_default():
        summary_metadata = summary_pb2.SummaryMetadata(
            plugin_data=summary_pb2.SummaryMetadata.PluginData(
                plugin_name=_TT_TENSORBOARD_PLUGIN_NAME))
        for key, value in kwargs.items():
          # Check whether we need to compute aggregated statistics that merge
          # all cores statistics.
          if not self._parameters.collect_summary_per_core:
            # Merge only statistics tensor, if it is any other tensor we simply,
            # concatenate them.
            # Also, if there is only a single core (first dim. is 0), then skip
            # aggregation.
            if key == _TT_SUMMARY_TAG and value.shape.as_list()[0] != 1:
              value = self.aggregate_global_cache(value)
          with ops.control_dependencies([summary_writer.init()]):
            summary_write_ops.append(summary.write(
                _TT_SUMMARY_TAG + '/' + key + '.' + graph_summary_tag,
                value, metadata=summary_metadata,
                step=step_value))
      return control_flow_ops.group(summary_write_ops)

    global_step = training_util.get_or_create_global_step()
    step = array_ops.reshape(global_step, [1])
    self._host_call_fn = {}

    host_call_deps = op_fetches + [tensor.op for tensor in processed_t_fetches]

    caches_to_write = {}
    with ops.control_dependencies(host_call_deps):
      all_caches = self._cache_variable_for_graph(graph)
      for cache_name, cache_variable in all_caches.items():
        # Increase the cache rank by 1, so that when host call concatenates
        # tensors from different replicas, we can identify them with [core_id].
        new_cache_shape = [1]
        new_cache_shape.extend(cache_variable.shape.as_list())
        cache = array_ops.reshape(cache_variable, new_cache_shape)
        caches_to_write[cache_name] = cache
    # Add step to parameter dictionary.
    caches_to_write['step'] = step
    # Other options without adding step to parameter dictionary are
    #  * host_call_fn = (_write_cache(step, caches_to_write)) : fails as it
    #    considers caches_to_write as a single parameter, rather than a keyword
    #    parameters.
    #  * host_call_fn = (_write_cache(step, **caches_to_write)) : fails with
    #    a syntax error.
    self._host_call_fn[_TT_HOSTCALL_KEY] = (_write_cache, caches_to_write)

  def host_call_deps_and_fn(self):
    return self._host_call_fn

  def get_traced_op_names(self):
    """Returns the set of traced op names."""
    return self._traced_op_names

  def _trace_execution(self, graph,
                       tensor_fetches,
                       op_fetches=None,
                       on_tpu=True):
    """Commong tracing function for both CPU and TPUs.

    The caller function should set device_type, num_replicas,
    num_replicas_per_host, num_hosts and replica_id before calling
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

    trace_mode = self._parameters.trace_mode
    device_type = self._tt_config.device_type
    # pylint: disable=protected-access
    self._outmost_context = graph._get_control_flow_context()
    # pylint: enable=protected-access

    analytics.track_usage('tensor_tracer', [trace_mode, device_type])
    TensorTracer.check_device_type(device_type)
    TensorTracer.check_trace_mode(device_type, trace_mode)
    # Check in_tensor_fetches, and op_fetches and convert them to lists.
    processed_t_fetches = self._process_tensor_fetches(tensor_fetches)
    op_fetches = self._process_op_fetches(op_fetches)
    all_fetches = op_fetches + [tensor.op for tensor in processed_t_fetches]

    # Filter out the operations that won't be executed.
    # if fetches=None, then ops_in_exec_path = set(operations)
    exec_op_set = self._filter_execution_path_operations(graph.get_operations(),
                                                         all_fetches)
    graph_summary_tag = _graph_summary_tag(graph)

    # Write report file, and determine the traced tensors.
    tensor_trace_order = self._determine_trace_and_create_report(
        graph, exec_op_set, graph_summary_tag)

    tensor_fetch_set = set(processed_t_fetches)
    tracing_ops = []

    sorted_exec_op_list = list(exec_op_set)
    sorted_exec_op_list.sort(key=lambda op: op.name)
    # Trace ops only if they are in the execution path.
    for op in sorted_exec_op_list:
      for i in range(len(op.outputs)):
        out_tensor = op.outputs[i]
        tensor_name = out_tensor.name
        if tensor_name not in tensor_trace_order.tensorname_to_cache_idx:
          continue
        self._traced_op_names.add(op.name)
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
        if op_control_flow_context:
          # pylint: disable=protected-access
          graph._set_control_flow_context(op_control_flow_context)
          # pylint: enable=protected-access

        processed_tensors = self._preprocess_traced_tensor(out_tensor)

        if on_tpu:
          for signature in processed_tensors.keys():
            processed_tensors[signature] = _cast_unsupported_dtypes(
                processed_tensors[signature])

        if self._use_tensor_values_cache():
          # Use a small cache (either temp cache or tf local variable) to store
          # the characteristics of the tensor.
          if self._use_temp_cache():
            cache_idx = tensor_trace_order.tensorname_to_cache_idx[tensor_name]
            self._save_tensor_value_to_tmp_cache(cache_idx,
                                                 processed_tensors,
                                                 graph)
            trace_op = None
          else:
            cache_idx = tensor_trace_order.tensorname_to_cache_idx[tensor_name]
            trace_op = self._save_tensor_value_to_cache_op(cache_idx,
                                                           processed_tensors,
                                                           graph)
        elif self._use_tensor_buffer():
          if len(processed_tensors) != 1:
            raise RuntimeError('Multiple stats are only allowed in compact '
                               'mode.')
          processed_out_tensor = list(processed_tensors.values())[0]
          # Store the whole tensor in a buffer.
          trace_op = self._snapshot_tensor(processed_out_tensor)
        else:

          def tpu_wrap_trace_fn(tensor, out_tensor_name):
            """Wraps the trace_fn with outside compilation if on TPUs."""
            tensor_trace_fn = self._make_tensor_trace_fun(out_tensor_name,
                                                          tensor_trace_order)
            if on_tpu:
              return tpu.outside_compilation(tensor_trace_fn, tensor)
            else:
              return tensor_trace_fn(tensor)

          if len(processed_tensors) != 1:
            raise RuntimeError('Multiple stats are only allowed in compact '
                               'mode.')
          # Collecting multiple statistics are only supported in the summary
          # mode that uses compact format(self._use_tensor_values_cache = true).
          # Non-compact mode currently allows single stat per tensor.
          processed_out_tensor = six.next(six.itervalues(processed_tensors))
          trace_op = tpu_wrap_trace_fn(processed_out_tensor, tensor_name)

        if op_control_flow_context:
          # pylint: disable=protected-access
          graph._set_control_flow_context(self._outmost_context)
          # pylint: enable=protected-access
        if trace_op:
          if is_a_fetched_tensor:
            tracing_ops.append(trace_op)
            continue
          # Add it to all consumers, as some consumers may not be executed if
          # they are in a control flow.
          for consumer_op in consumers:
            # pylint: disable=protected-access
            consumer_op._add_control_input(trace_op)
            # pylint: enable=protected-access

    # pylint: disable=protected-access
    graph._set_control_flow_context(self._outmost_context)
    # pylint: enable=protected-access
    if tracing_ops:
      # If we are tracing a fetched tensor, their dependency is stored in
      # tracing_ops.
      processed_t_fetches = control_flow_ops.tuple(processed_t_fetches,
                                                   control_inputs=tracing_ops)
    if self._use_tensor_values_cache() or self._use_tensor_buffer():
      if self._use_temp_cache():
        # Create the temporary tf cache variable by concantanating all
        # statistics.
        graph_cache_var = self._cache_variable_for_graph(graph)
        if graph not in self._temp_cache_var:
          raise RuntimeError('graph is not in self._temp_cache_var')
        graph_cache_var[_TT_SUMMARY_TAG] = array_ops.stack(
            self._temp_cache_var[graph], axis=0, name='stack_all_op_signatures')
      if self._create_host_call():
        self._prepare_host_call_fn(processed_t_fetches, op_fetches, graph,
                                   graph_summary_tag)
        if not on_tpu:
          write_cache, caches_to_write = self._host_call_fn[_TT_HOSTCALL_KEY]
          cache_write_op = write_cache(**caches_to_write)
          processed_t_fetches = control_flow_ops.tuple(
              processed_t_fetches, control_inputs=[cache_write_op])
          del self._host_call_fn[_TT_HOSTCALL_KEY]
        elif self._parameters.flush_summaries_with_outside_compile:
          write_cache, caches_to_write = self._host_call_fn[_TT_HOSTCALL_KEY]
          if (_TT_SUMMARY_TAG in caches_to_write and 'step' in caches_to_write):
            step = caches_to_write['step']
            tensor_tracer_summary = caches_to_write[_TT_SUMMARY_TAG]
            tt_core_summary = self.merge_caches_on_tpu(tensor_tracer_summary[0])
            if not self._parameters.collect_summary_per_core:
              tt_core_summary = self.aggregate_global_cache(tt_core_summary)

            def write_if_core_0(step, replica_id, tt_summary):

              return control_flow_ops.cond(
                  math_ops.equal(replica_id, 0),
                  lambda: write_cache(step=step, event_file_suffix=None,  # pylint: disable=g-long-lambda
                                      tensor_tracer_summary=tt_summary),
                  control_flow_ops.no_op)

            write_op = tpu.outside_compilation(write_if_core_0, step=step,
                                               replica_id=self._replica_id,
                                               tt_summary=tt_core_summary)
            processed_t_fetches = control_flow_ops.tuple(
                processed_t_fetches, control_inputs=[write_op])
            del self._host_call_fn[_TT_HOSTCALL_KEY]
          else:
            raise ValueError('Outside compiled flush in only supported for '
                             'summary mode')
      else:
        processed_t_fetches = self._flush_tensor_values_cache(
            processed_t_fetches, op_fetches, on_tpu=on_tpu,
            tensor_trace_order=tensor_trace_order,
            graph=graph)

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
    """
    if isinstance(graph, func_graph.FuncGraph) or isinstance(
        graph, function._FuncGraph):  # pylint: disable=protected-access
      logging.warning('Tensor Tracer is not supported for tracing FuncGraphs. '
                      'Ignoring tracing.')
      return tensor_fetches

    if graph in TensorTracer._traced_graphs:
      logging.warning('Graph is already rewritten with tensor tracer, ignoring '
                      'multiple calls.')
      return tensor_fetches
    else:
      TensorTracer._traced_graphs.add(graph)
    # Reset the parameters in case parameters are changed.
    self._parameters = tensor_tracer_flags.TTParameters()
    self._tt_config.device_type = _DEVICE_TYPE_TPU
    self._tt_config.num_replicas = num_replicas
    self._tt_config.num_replicas_per_host = num_replicas_per_host
    self._tt_config.num_hosts = num_hosts
    if self._tt_config.num_replicas is not None:
      if self._tt_config.num_replicas_per_host is None:
        self._tt_config.num_replicas_per_host = 8
      if self._tt_config.num_hosts is None:
        self._tt_config.num_hosts = (
            num_replicas // self._tt_config.num_replicas_per_host +
            (num_replicas % self._tt_config.num_replicas_per_host > 0))

    if self._parameters.graph_dump_path:
      graph_io.write_graph(graph, self._parameters.graph_dump_path,
                           'graph_before_tt.pbtxt')
    with graph.as_default():
      self._add_replica_id_to_graph()
      tensor_fetches = self._trace_execution(graph, tensor_fetches, op_fetches,
                                             on_tpu=True)
    if self._parameters.graph_dump_path:
      graph_io.write_graph(graph, self._parameters.graph_dump_path,
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
    """
    if isinstance(graph, func_graph.FuncGraph) or isinstance(
        graph, function._FuncGraph):  # pylint: disable=protected-access
      logging.warning('Tensor Tracer is not supported for tracing FuncGraphs. '
                      'Ignoring tracing.')
      return tensor_fetches

    if graph in TensorTracer._traced_graphs:
      logging.warning('Graph is already rewritten with tensor tracer, ignoring '
                      'multiple calls.')
      return tensor_fetches
    else:
      TensorTracer._traced_graphs.add(graph)
    # Reset the parameters in case parameters are changed.
    self._parameters = tensor_tracer_flags.TTParameters()

    self._tt_config.device_type = _DEVICE_TYPE_CPU
    self._tt_config.num_replicas = 1
    self._tt_config.num_replicas_per_host = 1
    self._tt_config.num_hosts = 1
    self._replica_id = 0
    if self._parameters.graph_dump_path:
      graph_io.write_graph(graph, self._parameters.graph_dump_path,
                           'graph_before_tt.pbtxt')
    with graph.as_default():
      tensor_fetches = self._trace_execution(graph, tensor_fetches, op_fetches,
                                             on_tpu=False)
    if self._parameters.graph_dump_path:
      graph_io.write_graph(graph, self._parameters.graph_dump_path,
                           'graph_after_tt.pbtxt')
    return tensor_fetches
