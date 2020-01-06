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

import operator

import os
import os.path
import sys

import numpy as np
import six

from tensorflow.core.framework import summary_pb2
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
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2 as summary
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import analytics
from tensorflow.python.platform import gfile
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
_TRACE_FILE_NAME = 'trace.all'
_COMPACT_TRACE_FILE_PREFIX = 'compact_trace.'
_COMPACT_TRACE_ENTRY_INIT_VALUE = -1.0
_TENSOR_TRACER_STORAGE = 'tensor_tracer_storage'
_TT_SNAPSHOT = 'tensor_tracer_snapshot'
_REPLICA_ID_TAG = '#replica-id: '

_TT_SUMMARY_NORM = tensor_tracer_flags.TT_SUMMARY_NORM
_TT_SUMMARY_MAX = tensor_tracer_flags.TT_SUMMARY_MAX
_TT_SUMMARY_MIN = tensor_tracer_flags.TT_SUMMARY_MIN
_TT_SUMMARY_MEAN = tensor_tracer_flags.TT_SUMMARY_MEAN
_TT_SUMMARY_VAR = tensor_tracer_flags.TT_SUMMARY_VAR
_TT_SUMMARY_SIZE = tensor_tracer_flags.TT_SUMMARY_SIZE

_TT_SUMMARY_TAG = 'tensor_tracer_summary'
_TT_TENSORBOARD_PLUGIN_NAME = 'tensor_tracer'
_TT_HOSTCALL_KEY = 'tensor_tracer_host_call'
_TT_EVENT_FILE_SUFFIX = '.tensor_tracer'

_TT_SUMMARY_MAX_QUEUE = 100


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
    # Lowest priority ops, e.g., constant ops accross different steps,
    # They will be traced only if trace_level>=7
    return 7

  if op_type in ('Identity', 'Cast', 'Reshape', 'ExpandDims', 'StopGradient',
                 'PreventGradient', 'Squeeze'):
    # Operations without numerical effects.
    # They will be only if trace_level>=6
    return 6
  if op_type in ('ConcatV2', 'Concat', 'StridedSlice', 'Slice', 'Pack', 'Tile',
                 'CollectivePermute', 'SplitV'):
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
                 'Maximum', 'Mean', 'Variance'):
    # Multiplication and some other operations, will be traced if trace_level>=1
    return 1
  return 0


def read_tensor_tracer_event_file(event_file):
  """Reads the event file written by tensor tracer.

  Args:
    event_file: Path to the event file that contains only tensor tracer events.
  Returns:
    An event dictionary in the form of
    {step_number: {tensor_name: tensor_content}}
  Raises:
    ValueError: If an unexpected trace is found.
  """
  event_dict = {}
  for trace_event in summary_iterator.summary_iterator(event_file):
    # First event is an event with file_version: "brain.Event:2"
    if not trace_event.HasField('summary'):
      continue
    step = trace_event.step
    if step not in event_dict:
      event_dict[step] = {}

    if len(trace_event.summary.value) != 1:
      raise ValueError('Single step contains %d summary values,'
                       ' expected 1.' % len(trace_event.summary.value))
    tensor_value = trace_event.summary.value[0]
    tensor_name = tensor_value.tag

    real_shape = [d.size for d in tensor_value.tensor.tensor_shape.dim]
    tensor_content = np.frombuffer(
        tensor_value.tensor.tensor_content,
        dtypes.DType(tensor_value.tensor.dtype).as_numpy_dtype()
        ).reshape(real_shape)
    event_dict[step][tensor_name] = tensor_content
  return event_dict


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
  def is_enabled():
    """Returns True if TensorTracer is enabled."""
    return tensor_tracer_flags.TTParameters().is_enabled()

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
  def unsafe_op(op):
    """Returns True if this op is not safe to be traced."""

    if control_flow_util.IsInCond(op):
      return True
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
    # If flag is set to include less interesting ops, then include everything.
    if self._parameters.include_less_interesting_ops:
      return True
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
    self._parameters = tensor_tracer_flags.TTParameters()
    self._included_op_full_names = set()
    self._host_call_fn = {}
    self._cache_variables = {}

  def _get_all_cache_variables(self):
    return self._cache_variables

  def _create_or_get_tensor_values_cache(self, cache_name, graph=None,
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
      ValueError: If missing a parameter to create the cache.
    """
    def _escape_namescopes(variable_name):
      # TODO(deveci): This might cause name collisions as in "foo/bar/mytensor"
      # and "foo_bar/mytensor".
      return variable_name.replace('/', '_').replace(':', '_')

    if cache_name not in self._cache_variables:
      if graph is None:
        raise ValueError('Graph must be provided at cache creation.')
      if shape is None:
        raise ValueError('shape must be provided at cache creation.')
      graph = graph or ops.get_default_graph()
      if dtype.is_integer:
        init_val = int(_COMPACT_TRACE_ENTRY_INIT_VALUE)
      else:
        init_val = _COMPACT_TRACE_ENTRY_INIT_VALUE

      # Create in proper graph and base name_scope.
      with graph.as_default() as g, g.name_scope(None):
        self._cache_variables[cache_name] = variable_scope.get_variable(
            _TT_SNAPSHOT + '_' + _escape_namescopes(cache_name),
            shape=shape, dtype=dtype,
            initializer=init_ops.constant_initializer(init_val),
            trainable=False,
            use_resource=True,
            collections=[_TENSOR_TRACER_STORAGE, ops.GraphKeys.LOCAL_VARIABLES])
    return self._cache_variables[cache_name]

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

    def _is_op_or_any_neighbor_included(op, check_before=0, check_after=0):
      """Helper function to check if op is included or not."""
      if op.name in self._included_op_full_names:
        return True
      for opname_re in self._parameters.included_opname_re_list:
        if opname_re.match(op.name):
          self._included_op_full_names.add(op.name)
          return True

      for optype_re in self._parameters.included_optype_re_list:
        if optype_re.match(op.type):
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
    return _is_op_or_any_neighbor_included(
        op, self._parameters.trace_ops_after_included,
        self._parameters.trace_ops_before_included)

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

  def _use_tensor_values_cache(self):
    """Returns True if immediate tensors should be first saved to a cache."""
    if self._parameters.trace_mode == tensor_tracer_flags.TRACE_MODE_SUMMARY:
      # For summary tace mode only compact format is supported.
      return True

    if self._parameters.trace_mode not in set([
        tensor_tracer_flags.TRACE_MODE_NAN_INF,
        tensor_tracer_flags.TRACE_MODE_NORM,
        tensor_tracer_flags.TRACE_MODE_MAX_ABS,
        tensor_tracer_flags.TRACE_MODE_SUMMARY
    ]):
      return False
    if (self._parameters.trace_dir and
        _trace_files_need_precreated(self._parameters.trace_dir)):
      return True
    return self._parameters.use_compact_trace

  def _use_tensor_buffer(self):
    """Returns true if the whole tensor needs to be cached/buffered in memory."""
    return (self._parameters.trace_mode ==
            tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY)

  def _save_tensor_value_to_cache_op(self, cache_idx, updates):
    """Returns an op that will save the given updates to an entry in the cache.

    Args:
      cache_idx: The cache index of the tensor within the cache.
      updates: A dictionary of the signature updates.
    Returns:
      Cache update operation.
    """
    # state_ops.scatter_update allows updates only along the first dimension.
    # Make a compact array by concantating different signatures, and update
    # them all together.
    sorted_update = []
    if self._num_signature_dimensions() > 1:
      signature_indices = self._signature_types()
      for _, val in sorted(updates.items(),
                           key=lambda item: signature_indices[item[0]]):
        sorted_update.append(val)
      updates = array_ops.stack(sorted_update, axis=0)
      updates = array_ops.reshape(updates, [1,
                                            self._num_signature_dimensions()])
    else:
      (_, val), = updates.items()
      updates = array_ops.reshape(val, [1, self._num_signature_dimensions()])
    indices = constant_op.constant([cache_idx])
    cache = self._create_or_get_tensor_values_cache(_TT_SUMMARY_TAG)
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
      RuntimeError: If the trace mode is invalid.
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

    if (self._parameters.trace_mode ==
        tensor_tracer_flags.TRACE_MODE_FULL_IF_NAN):
      return {self._parameters.trace_mode: _detect_inf_nan_producer(tensor)}
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
        elif signature_name == _TT_SUMMARY_MIN:
          signature_result_tensor = _show_min(tensor, cast_to_f32=False)
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
        'Tensor trace fun for %s is not yet implemented'
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
                    self._tensorname_idx_map.
      """

      if self._parameters.is_brief_mode():
        if tensor_name not in tensor_trace_order.tensorname_idx_map:
          raise ValueError(
              'Tensor name %s is not in the tensorname_idx_map'%tensor_name)
        msg = '%d'%self._tensorname_idx_map[tensor_name]
      else:
        msg = '"%s"'%tensor_name

      if self._parameters.trace_dir:
        output_path = os.path.join(self._parameters.trace_dir, _TRACE_FILE_NAME)
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

      tensors_to_print = _get_distance_k_tensors(
          self._parameters.trace_stack_size)
      print_ops = [_print_tensor(t.name, -1, t, t) for t in tensors_to_print]
      with ops.control_dependencies(print_ops):
        return constant_op.constant(True)

    if (self._parameters.trace_mode ==
        tensor_tracer_flags.TRACE_MODE_FULL_IF_NAN):
      return _show_full_tensors
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

    raise RuntimeError('Tensor trace fun for %s is not yet implemented'
                       %self._parameters.trace_mode)

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
    if _trace_files_need_precreated(self._parameters.trace_dir):
      for replica_id in range(0, self._tt_config.num_replicas):
        trace_file_path = os.path.join(
            self._parameters.trace_dir,
            _COMPACT_TRACE_FILE_PREFIX) + '%d'%replica_id
        if not gfile.Exists(trace_file_path):
          raise RuntimeError(
              '%s must be pre-created with the '
              'appropriate properties.'%trace_file_path)
    else:
      if not gfile.Exists(self._parameters.trace_dir):
        gfile.MkDir(self._parameters.trace_dir)
        if not gfile.Exists(self._parameters.trace_dir):
          raise RuntimeError('Failed to create %s'%self._parameters.trace_dir)

  def _determine_trace_and_create_report(self, graph, ops_in_exec_path):
    """Work needs to be done prior to TPU or CPU tracing.

    Args:
      graph: tf.graph
      ops_in_exec_path: Set of operations in the execution path.
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

    tensor_trace_order = tensor_tracer_report.TensorTraceOrder(graph_order,
                                                               traced_tensors)
    num_signatures = self._num_signature_dimensions()
    if num_signatures:
      self._create_or_get_tensor_values_cache(_TT_SUMMARY_TAG,
                                              graph,
                                              [len(traced_tensors),
                                               num_signatures])
    if self._parameters.trace_mode in (
        tensor_tracer_flags.TRACE_MODE_SUMMARY,
        tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY):
      report_proto = report_handler.create_report_proto(self._tt_config,
                                                        self._parameters,
                                                        tensor_trace_order,
                                                        tensor_trace_points,
                                                        self._signature_types())
      report_handler.write_report_proto(report_proto, self._parameters)
    else:
      report_handler.create_report(self._tt_config, self._parameters,
                                   tensor_trace_order, tensor_trace_points)
    return tensor_trace_order

  def _create_host_call(self):
    return self._parameters.trace_mode in (
        tensor_tracer_flags.TRACE_MODE_SUMMARY,
        tensor_tracer_flags.TRACE_MODE_FULL_TENSOR_SUMMARY)

  def _generate_flush_cache_op(self, num_replicas, on_tpu):
    """Generates an Op that will flush the cache to file.

    Args:
      num_replicas: total number of replicas.
      on_tpu: if the graph is executed on TPU.

    Returns:
      The Op to flush the cache to file.
    """

    def _flush_fun(cache, replica_id):
      """Flushes the cache to a file corresponding to replica_id."""

      def _f(file_index):
        """Generates a func that flushes the cache to a file."""
        def _print_cache():
          """Flushes the cache to a file."""
          replica_str = ('%d' % file_index)
          if self._parameters.trace_dir:
            output_path = (os.path.join(self._parameters.trace_dir,
                                        _COMPACT_TRACE_FILE_PREFIX)
                           + replica_str)
            output_stream = _OUTPUT_STREAM_ESCAPE + output_path
          else:
            output_stream = sys.stderr

          new_step_line = _REPLICA_ID_TAG + replica_str
          print_ops = []
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
      for i in range(num_replicas):
        flush_op_cases[_eq(i)] = _f(i)
      # Each replica needs to determine where to write their output.
      # To do this, we check if replica_id is 0, then 1, ..., and then
      # num_replicas - 1 statically; and return the corresponding static file
      # name. We cannot simply set the file name in python, as replica_id is
      # only known during tf runtime, and we cannot create dynamic filenames.
      return control_flow_ops.case(flush_op_cases, exclusive=True)

    cache = self._create_or_get_tensor_values_cache(_TT_SUMMARY_TAG)
    if on_tpu:
      flush_op = tpu.outside_compilation(_flush_fun,
                                         cache.value(), self._replica_id)
    else:
      flush_op = _flush_fun(cache.value(), self._replica_id)

    with ops.control_dependencies([flush_op]):
      reset_value = constant_op.constant(_COMPACT_TRACE_ENTRY_INIT_VALUE,
                                         dtype=cache.dtype,
                                         shape=cache.shape)
      assign_op = state_ops.assign(cache, reset_value).op
      with ops.control_dependencies([assign_op]):
        return constant_op.constant(0).op

  def _flush_tensor_values_cache(self, tensor_fetches, op_fetches, on_tpu):
    """Flushes the intermediate tensor values in the graph to the cache.

    Args:
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
      flush_cache_op = self._generate_flush_cache_op(
          self._tt_config.num_replicas, on_tpu)
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

  def _prepare_host_call_fn(self, processed_t_fetches, op_fetches):
    """Creates a host call function that will write the cache as tb summary.

    Args:
      processed_t_fetches: List of tensor provided to session.run.
      op_fetches: List of operations provided to session.run.
    Raises:
      ValueError if trace_dir is not set.
    """
    if self._parameters.trace_dir is None:
      raise ValueError('Provide a trace_dir for tensor tracer in summary mode. '
                       '--trace_dir=/model/dir')

    def _write_cache(step, **kwargs):
      """Writes the given caches as tensor summary.

      Args:
        step: Step tensor with dimension [num_cores].
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
      Raises:
        RuntimeError: if there is no aggregate function defined for a signature.
      """

      # TODO(deveci): Parametrize max_queue, so that flushing op can be called
      # less frequently.
      # Setting max_queue to 100 appears to be safe even when the number of
      # iterations are much lower, as the destructor of the writer flushes it.
      summary_write_ops = []
      with summary.create_file_writer_v2(
          self._parameters.trace_dir,
          filename_suffix=_TT_EVENT_FILE_SUFFIX,
          max_queue=_TT_SUMMARY_MAX_QUEUE).as_default():
        summary_metadata = summary_pb2.SummaryMetadata(
            plugin_data=summary_pb2.SummaryMetadata.PluginData(
                plugin_name=_TT_TENSORBOARD_PLUGIN_NAME))
        for key, value in kwargs.items():
          # Check whether we need to compute aggregated statistics that merge
          # all cores statistics.
          if not self._parameters.collect_summary_per_core:
            # Merge only statistics tensor, if it is any other tensor we simply,
            # concatenate them.
            if key == _TT_SUMMARY_TAG:
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
                # value[:,:,idx] will return the portion of the tensor relasted
                # to signature.
                signature_tensor = value[:, :, idx]
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
              value = array_ops.expand_dims(transposed_signatures, axis=0)

          with ops.control_dependencies(
              summary.summary_writer_initializer_op()):
            summary_write_ops.append(summary.write(
                _TT_SUMMARY_TAG + '/' + key, value, metadata=summary_metadata,
                step=step[0]))
      return control_flow_ops.group(summary_write_ops)

    step = array_ops.reshape(training_util.get_or_create_global_step(), [1])
    self._host_call_fn = {}

    host_call_deps = op_fetches + [tensor.op for tensor in processed_t_fetches]

    caches_to_write = {}
    with ops.control_dependencies(host_call_deps):
      all_caches = self._get_all_cache_variables()
      for cache_name, cache_variable in all_caches.items():
        # Increase the cache rank by 1, so that when host call concatenates
        # tensors from different replicas, we can identify them with [core_id].
        new_cache_shape = [1]
        new_cache_shape.extend(cache_variable.shape.as_list())
        cache = array_ops.reshape(cache_variable.value(), new_cache_shape)
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
    # Write report file, and determine the traced tensors.
    tensor_trace_order = self._determine_trace_and_create_report(
        graph, exec_op_set)

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
        if tensor_name not in tensor_trace_order.tensorname_to_cache_idx:
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
        processed_tensors = self._preprocess_traced_tensor(out_tensor)

        if on_tpu:
          for signature in processed_tensors.keys():
            processed_tensors[signature] = _cast_unsupported_dtypes(
                processed_tensors[signature])

        if self._use_tensor_values_cache():
          # Use a small cache to store the characteristics of the tensor.
          cache_idx = tensor_trace_order.tensorname_to_cache_idx[tensor_name]
          trace_op = self._save_tensor_value_to_cache_op(cache_idx,
                                                         processed_tensors)
        elif self._use_tensor_buffer():
          if len(processed_tensors) != 1:
            raise RuntimeError('Multiple stats are only allowed in compact '
                               'mode.')
          processed_out_tensor = processed_tensors.values()[0]
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

          def conditional_trace_fn(predicate_tensor, out_tensor, trace_fn,
                                   out_tensor_name):
            """Creates a cond op that traces the out_tensor if predicate is satisfied."""
            return control_flow_ops.cond(
                predicate_tensor, lambda: trace_fn(out_tensor, out_tensor_name),
                lambda: constant_op.constant(False)).op

          if len(processed_tensors) != 1:
            raise RuntimeError('Multiple stats are only allowed in compact '
                               'mode.')
          # Collecting multiple statistics are only supported in the summary
          # mode that uses compact format(self._use_tensor_values_cache = true).
          # Non-compact mode currently allows single stat per tensor.
          processed_out_tensor = six.next(six.itervalues(processed_tensors))

          if self._parameters.is_conditional_trace:
            trace_op = conditional_trace_fn(processed_out_tensor, out_tensor,
                                            tpu_wrap_trace_fn, tensor_name)
          elif self._parameters.included_cores:
            should_print = constant_op.constant(False)
            for core in self._parameters.included_cores:
              should_print = gen_math_ops.logical_or(
                  should_print, gen_math_ops.equal(self._replica_id, core))
            trace_op = conditional_trace_fn(should_print, processed_out_tensor,
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
    if self._use_tensor_values_cache() or self._use_tensor_buffer():
      if self._create_host_call():
        self._prepare_host_call_fn(processed_t_fetches, op_fetches)
        if not on_tpu:
          write_cache, caches_to_write = self._host_call_fn[_TT_HOSTCALL_KEY]
          cache_write_op = write_cache(**caches_to_write)
          processed_t_fetches = control_flow_ops.tuple(
              processed_t_fetches, control_inputs=[cache_write_op])
          del self._host_call_fn[_TT_HOSTCALL_KEY]
      else:
        processed_t_fetches = self._flush_tensor_values_cache(
            processed_t_fetches, op_fetches, on_tpu=on_tpu)

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
    Raises:
      RuntimeError: If tensor_fetches is None or empty.
    """

    if graph in TensorTracer._traced_graphs:
      logging.warning('Graph is already rewritten with tensor tracer, ignoring '
                      'multiple calls.')
      return tensor_fetches
    else:
      TensorTracer._traced_graphs.add(graph)

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
