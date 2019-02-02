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
# ======================================

"""Library of TPU helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.compiler import xla
from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.contrib.tpu.proto import dynamic_padding_pb2 as dynamic_padding
from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.contrib.tpu.python.tpu import tpu_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.compat import compat as api_compat
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest


# Operations that indicate some error in the users graph, e.g. a placeholder
# that's introduced outside of the infeed.
_BLACKLISTED_OPS = set([
    "Placeholder",
])

# XLA doesn't currently support reading of intermediate tensors, thus some ops
# are not supported.
_UNSUPPORTED_OPS = set([
    "AudioSummary",
    "AudioSummaryV2",
    "HistogramSummary",
    "ImageSummary",
    "MergeSummary",
    "Print",
    "ScalarSummary",
    "TensorSummary",
    "TensorSummaryV2",
    ])

_MAX_WARNING_LINES = 5

_TPU_REPLICATE_ATTR = "_tpu_replicate"
_TPU_COMPILATION_STATUS_ATTR = "_tpu_compilation_status"
_OUTSIDE_COMPILATION_ATTR = "_xla_outside_compilation"


def _tpu_system_device_name(job):
  """Returns the device name for the TPU_SYSTEM device of `job`."""
  if job is None:
    return "/device:TPU_SYSTEM:0"
  else:
    return "/job:%s/device:TPU_SYSTEM:0" % job


def initialize_system(embedding_config=None, job=None):
  """Initializes a distributed TPU system for use with TensorFlow.

  Args:
    embedding_config: If not None, a `TPUEmbeddingConfiguration` proto
      describing the desired configuration of the hardware embedding lookup
      tables. If embedding_config is None, no hardware embeddings can be used.
    job: The job (the XXX in TensorFlow device specification /job:XXX) that
      contains the TPU devices that will be initialized. If job=None it is
      assumed there is only one job in the TensorFlow flock, and an error will
      be returned if this assumption does not hold.
  Returns:
    A serialized `TopologyProto` that describes the TPU system. Note:
      the topology must be evaluated using `Session.run` before it can be used.
  """
  config_string = ("" if embedding_config is None else
                   embedding_config.SerializeToString())
  with ops.device(_tpu_system_device_name(job)):
    return tpu_ops.configure_distributed_tpu(embedding_config=config_string)


def shutdown_system(job=None):
  """Shuts down a running a distributed TPU system."""
  with ops.device(_tpu_system_device_name(job)):
    shutdown_distributed_tpu = tpu_ops.shutdown_distributed_tpu()
  return shutdown_distributed_tpu


def core(num):
  """Returns the device name for a core in a replicated TPU computation.

  Args:
    num: the virtual core number within each replica to which operators should
    be assigned.
  Returns:
    A device name, suitable for passing to `tf.device()`.
  """
  return "device:TPU_REPLICATED_CORE:{}".format(num)


class TPUReplicateContext(control_flow_ops.XLAControlFlowContext):
  """A `ControlFlowContext` for nodes inside a TPU computation.

  The primary role of `TPUReplicateContext` is to mark operators inside a
  tpu.replicate() computation with the attribute "_tpu_replicate=XYZ", where XYZ
  is a unique name.

  We use a `ControlFlowContext` to perform the annotation since it integrates
  with Tensorflow constructs like ResourceVariables. For example, if a
  `ResourceVariable` is constructed inside a tpu.replicate() block, the
  `ResourceVariable` implementation can use
  `with ops.control_dependencies(None)` to build the variable's definition
  outside the replicated computation.
  """

  def __init__(self, name, num_replicas, pivot):
    """Builds a new TPUReplicateContext.

    Args:
      name: a unique name for the context, used to populate the `_tpu_replicate`
        attribute.
      num_replicas: an integer that gives the number of replicas for the
        computation.
      pivot: a pivot node. Nodes in the TPUReplicateContext that do not have any
        inputs will have a control dependency on the pivot node. This ensures
        that nodes are correctly included in any enclosing control flow
        contexts.
    """
    super(TPUReplicateContext, self).__init__()
    self._num_replicas = num_replicas
    self._outer_device_function_stack = None
    self._oc_dev_fn_stack = None
    self._outside_compilation_cluster = None
    self._outside_compilation_counter = 0
    self._in_gradient_colocation = None
    self._gradient_colocation_stack = []
    self._host_compute_core = []
    self._name = name
    self._name_as_bytes = compat.as_bytes(name)
    self._unsupported_ops = []
    self._pivot = pivot
    self._replicated_vars = {}

  def get_replicated_var_handle(self, name, vars_):
    """Returns a variable handle for replicated TPU variable 'var'.

    This is a method used by an experimental replicated variable implementation
    and is not intended as a public API.

    Args:
      name: The common name of the variable.
      vars_: The replicated TPU variables.

    Returns:
      The handle of the TPU replicated input node.
    """
    handle = self._replicated_vars.get(name)
    if handle is not None:
      return handle

    # Builds a TPUReplicatedInput node for the variable, if one does not already
    # exist. The TPUReplicatedInput node must belong to the enclosing
    # control-flow scope of the TPUReplicateContext.
    # TODO(phawkins): consider changing the contract of the TPU encapsulation
    # so the TPUReplicatedInput nodes go inside the TPUReplicateContext scope
    # instead.

    # pylint: disable=protected-access
    graph = ops.get_default_graph()
    saved_context = graph._get_control_flow_context()
    graph._set_control_flow_context(self.outer_context)
    handle = tpu_ops.tpu_replicated_input(
        [v.handle for v in vars_], name=name + "/handle")
    graph._set_control_flow_context(saved_context)
    # pylint: enable=protected-access
    self._replicated_vars[name] = handle
    return handle

  def report_unsupported_operations(self):
    if self._unsupported_ops:
      op_str = "\n".join(["  %s (%s)" % (op.type, op.name)
                          for op in self._unsupported_ops[:_MAX_WARNING_LINES]])
      logging.warning("%d unsupported operations found: \n%s",
                      len(self._unsupported_ops), op_str)
      if len(self._unsupported_ops) > _MAX_WARNING_LINES:
        logging.warning("... and %d more" %
                        (len(self._unsupported_ops) - _MAX_WARNING_LINES))

  def EnterGradientColocation(self, op, gradient_uid):
    if op is not None:
      self._gradient_colocation_stack.append(op)
      if not self._outside_compilation_cluster:
        try:
          outside_attr = op.get_attr(_OUTSIDE_COMPILATION_ATTR)
          if self._in_gradient_colocation:
            raise NotImplementedError(
                "Cannot nest gradient colocation operations outside compilation"
            )
          if gradient_uid == "__unsupported__":
            raise NotImplementedError(
                "No gradient_uid calling gradient within outside_compilation")
          # When we take the gradient of an op X in an outside_compilation
          # cluster C in a forward computation we would like to put the ops
          # corresponding to the gradient of X into a new outside_compilation
          # cluster C'. However, if we take the gradient of X twice, the second
          # one should get yet another new outside_compilation cluster C''.
          #
          # The mechanism we adopt is to use a 'root_cluster' which is the
          # cluster that X was in before we took gradients, and a 'gradient_uid'
          # which is different for every invocation of gradients, and put the
          # gradient of X in cluster 'root_cluster.gradient_uid'.
          #
          # When taking a gradient of a gradient, some ops will be colocated
          # with Op in the forward pass (e.g., cluster root_cluster) and some in
          # the backward pass (e.g., cluster root_cluster.initial_gradient_uid).
          # We need all of the grad-of-grad ops to be in the same cluster to
          # avoid cyclic dependencies between clusters. We adopt a heuristic
          # that puts any op clustered with root_cluster.<xxx> in
          # root_cluster.gradient_uid, even if xxx was initial_gradient_uid.
          self._in_gradient_colocation = op
          parts = outside_attr.split(".")
          cluster = parts[0] + "." + gradient_uid
          self._EnterOutsideCompilationScope(cluster=cluster)
        except ValueError:
          # The attr was not present: do nothing.
          pass

  def ExitGradientColocation(self, op, gradient_uid):
    if op is not None:
      if not self._gradient_colocation_stack:
        raise errors.InternalError(
            op.node_def, op,
            "Badly nested gradient colocation: empty stack when popping Op " +
            op.name)
      last_op = self._gradient_colocation_stack.pop()
      if op is last_op:
        if op is self._in_gradient_colocation:
          self._in_gradient_colocation = None
          self._ExitOutsideCompilationScope()
      else:
        raise errors.InternalError(
            op.node_def, op, "Badly nested gradient colocation, expected " +
            last_op + ", got " + op.name)

  def _EnterOutsideCompilationScope(self, cluster=None):

    class FakeOp(object):
      """A helper class to determine the current device.

      Supports only the type and device set/get methods needed to run the
      graph's _apply_device_function method.
      """

      def __init__(self):
        self._device = ""

      @property
      def type(self):
        return "FakeOp"

      @property
      def device(self):
        return self._device

      def _set_device(self, device):
        if isinstance(device, pydev.DeviceSpec):
          self._device = device.to_string()
        else:
          self._device = device

    if self._outside_compilation_cluster:
      raise NotImplementedError("Cannot nest outside_compilation clusters")
    if cluster:
      self._outside_compilation_cluster = cluster
    else:
      self._outside_compilation_cluster = str(self._outside_compilation_counter)
      self._outside_compilation_counter += 1
    graph = ops.get_default_graph()
    fake_op = FakeOp()
    graph._apply_device_functions(fake_op)  # pylint: disable=protected-access
    device = pydev.DeviceSpec.from_string(fake_op.device)
    if (device.device_type == "TPU_REPLICATED_CORE" and
        device.device_index is not None):
      self._host_compute_core.append(self._outside_compilation_cluster + ":" +
                                     str(device.device_index))
    self._oc_dev_fn_stack = graph._device_function_stack  # pylint: disable=protected-access
    graph._device_function_stack = self._outer_device_function_stack  # pylint: disable=protected-access

  def _ExitOutsideCompilationScope(self):
    if not self._outside_compilation_cluster:
      raise NotImplementedError(
          "Attempted to exit outside_compilation scope when not in scope")
    self._outside_compilation_cluster = None
    graph = ops.get_default_graph()
    graph._device_function_stack = self._oc_dev_fn_stack  # pylint: disable=protected-access

  def Enter(self):
    if not self._outer_device_function_stack:
      # Capture the device function stack at the time of first entry
      # since that is the stack that will be used outside_compilation.
      graph = ops.get_default_graph()
      # pylint: disable=protected-access
      self._outer_device_function_stack = graph._device_function_stack.copy()
      # pylint: enable=protected-access
    super(TPUReplicateContext, self).Enter()

  def HostComputeCore(self):
    return self._host_compute_core

  def _RemoveExternalControlEdges(self, op):
    """Remove any external control dependency on this op."""
    internal_control_inputs = []
    external_control_inputs = []
    for x in op.control_inputs:
      # pylint: disable=protected-access
      is_internal_op = False
      ctxt = x._get_control_flow_context()
      while ctxt is not None:
        if ctxt == self:
          is_internal_op = True
          break
        ctxt = ctxt._outer_context
      if is_internal_op:
        internal_control_inputs.append(x)
      else:
        external_control_inputs.append(x)
      # pylint: enable=protected-access
    # pylint: disable=protected-access
    op._remove_all_control_inputs()
    op._add_control_inputs(internal_control_inputs)
    # pylint: enable=protected-access
    return internal_control_inputs, external_control_inputs

  def AddOp(self, op):
    # pylint: disable=protected-access
    if op.type in _BLACKLISTED_OPS:
      logging.error("Operation of type %s (%s) is not supported on the TPU. "
                    "Execution will fail if this op is used in the graph. " %
                    (op.type, op.name))

    if op.type in _UNSUPPORTED_OPS:
      self._unsupported_ops.append(op)

    if any(x.dtype._is_ref_dtype for x in op.inputs):
      raise NotImplementedError(
          "Non-resource Variables are not supported inside TPU computations "
          "(operator name: %s)" % op.name)
    if _TPU_REPLICATE_ATTR in op.node_def.attr:
      raise ValueError("TPU computations cannot be nested")
    op._set_attr(_TPU_REPLICATE_ATTR,
                 attr_value_pb2.AttrValue(s=self._name_as_bytes))
    if self._outside_compilation_cluster:
      op._set_attr(
          _OUTSIDE_COMPILATION_ATTR,
          attr_value_pb2.AttrValue(
              s=compat.as_bytes(self._outside_compilation_cluster)))
    if self._num_replicas > 1 or not self._outside_compilation_cluster:
      # Prevent feeding or fetching anything that is being compiled,
      # and any replicated outside_compilation Op.
      op.graph.prevent_feeding(op)
      op.graph.prevent_fetching(op)

    # Remove any control edges from outer control flow contexts. These may cause
    # mismatched frame errors.
    (internal_control_inputs,
     external_control_inputs) = self._RemoveExternalControlEdges(op)

    if not op.inputs:
      # Add a control edge from the control pivot to this op.
      if not internal_control_inputs:
        # pylint: disable=protected-access
        op._add_control_input(self.GetControlPivot())
        # pylint: enable=protected-access
    else:
      for index in xrange(len(op.inputs)):
        x = op.inputs[index]
        real_x = self.AddValue(x)
        if real_x != x:
          op._update_input(index, real_x)  # pylint: disable=protected-access

    if external_control_inputs:
      # Use an identity to pull control inputs as data inputs. Note that we
      # ignore ops which don't have outputs. TODO(phawkins): fix that.
      with ops.control_dependencies(None):
        self.Enter()
        external_control_inputs = [
            array_ops.identity(x.outputs[0]).op
            for x in external_control_inputs
            if x.outputs
        ]
        self.Exit()
      # pylint: disable=protected-access
      op._add_control_inputs(external_control_inputs)
      # pylint: enable=protected-access

    # Mark op's outputs as seen by this context and any outer contexts.
    output_names = [x.name for x in op.outputs]
    context = self
    while context is not None:
      # pylint: disable=protected-access
      context._values.update(output_names)
      context = context._outer_context
      # pylint: enable=protected-access

    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  def AddValue(self, val):
    """Add `val` to the current context and its outer context recursively."""
    if val.name in self._values:
      # Use the real value if it comes from outer context.
      result = self._external_values.get(val.name)
      return val if result is None else result

    result = val
    self._values.add(val.name)
    if self._outer_context:
      result = self._outer_context.AddValue(val)
      self._values.add(result.name)

    self._external_values[val.name] = result

    return result

  def AddInnerOp(self, op):
    self.AddOp(op)
    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  @property
  def grad_state(self):
    # Define the gradient loop state associated with the TPUReplicateContext to
    # be None as the TPUReplicateContext does not get nested nor does the
    # grad_state outside the TPUReplicateContext affect the graph inside so the
    # grad_state should be as if this is the top-level gradient state.
    return None

  @property
  def back_prop(self):
    """Forwards to the enclosing while context, if any."""
    if self.GetWhileContext():
      return self.GetWhileContext().back_prop
    return False

  def GetControlPivot(self):
    return self._pivot


def outside_compilation(computation, *args, **kwargs):
  """Builds part of a computation outside any current TPU replicate scope.

  Args:
    computation: A Python function that builds the computation to
      place on the host.
    *args: the positional arguments for the computation.
    **kwargs: the keyword arguments for the computation.

  Returns:
    The Tensors returned by computation.
  """
  args = [] if args is None else args
  graph = ops.get_default_graph()

  # If we are in a TPUReplicateContext, signal that we are now
  # outside_compilation
  initial_context = graph._get_control_flow_context()  # pylint: disable=protected-access
  context = initial_context
  while context:
    if isinstance(context, TPUReplicateContext):
      context._EnterOutsideCompilationScope()  # pylint: disable=protected-access
    context = context.outer_context

  retval = computation(*args, **kwargs)

  # If we are in a TPUReplicateContext, signal that we are no longer
  # outside_compilation
  final_context = graph._get_control_flow_context()  # pylint: disable=protected-access
  if initial_context is not final_context:
    raise NotImplementedError(
        "Control-flow context cannot be different at start and end of an "
        "outside_compilation scope")
  context = initial_context
  while context:
    if isinstance(context, TPUReplicateContext):
      context._ExitOutsideCompilationScope()  # pylint: disable=protected-access
    context = context.outer_context

  return retval


def replicate(computation,
              inputs=None,
              infeed_queue=None,
              device_assignment=None,
              name=None,
              maximum_shapes=None):
  """Builds a graph operator that runs a replicated TPU computation.

  Args:
    computation: A Python function that builds the computation to replicate.
    inputs: A list of lists of input tensors or `None` (equivalent to
      `[[]]`), indexed by `[replica_num][input_num]`. All replicas must
      have the same number of inputs. Each input can be a nested structure
      containing values that are convertible to tensors. Note that passing an
      N-dimension list of compatible values will result in a N-dimention list of
      scalar tensors rather than a single Rank-N tensors. If you need different
      behavior, convert part of inputs to tensors with `tf.convert_to_tensor`.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to computation.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each replica of the computation uses
      only one core, and there is either only one replica, or the number of
      replicas is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
    maximum_shapes: A nested structure of tf.TensorShape representing the shape
      to which the respective component of each input element in each replica
      should be padded. Any unknown dimensions (e.g. tf.Dimension(None) in a
      tf.TensorShape or -1 in a tensor-like object) will be padded to the
      maximum size of that dimension over all replicas. Note that if the input
      dimension is already static, we won't do padding on it and we require the
      maximum_shapes to have the same value or None on that dimension. The
      structure of `maximum_shapes` needs to be the same as `inputs[0]`.
  Returns:
    A list of outputs, indexed by `[replica_num]` each output can be a nested
    structure same as what computation() returns with a few exceptions.

    Exceptions include:
      1) None output: a NoOp would be returned which control-depends on
         computation.
      2) Single value output: A tuple containing the value would be returned.
      3) Operation-only outputs: a NoOp would be returned which
         control-depends on computation.
      TODO(b/121383831): Investigate into removing these special cases.

  Raises:
    ValueError: If all replicas do not have equal numbers of input tensors.
    ValueError: If the number of inputs per replica does not match
      the number of formal parameters to `computation`.
    ValueError: If the static `inputs` dimensions don't match with the values
      given in `maximum_shapes`.
    ValueError: If the structure of inputs per replica does not match
      the structure of `maximum_shapes`.
  """
  return split_compile_and_replicate(
      computation,
      inputs,
      infeed_queue,
      device_assignment,
      name,
      maximum_shapes=maximum_shapes)[1]


def _pad_all_input(inputs, padded_shapes):
  """Pad all input tensors given padded_shapes.

  The real shape tensors will be concatenated with the padded original inputs.

  Args:
    inputs: The original inputs.
    padded_shapes: A list of padded shapes for each input.

  Returns:
    The padded inputs and a PaddingMap list which maps the padded input
    dimension to the real shape argument index.
  """
  input_shape_tensors = []
  for core_idx, inputs_per_core in enumerate(inputs):
    for idx, input_tensor in enumerate(inputs_per_core):
      if core_idx == 0:
        input_shape_tensors.append([])
      input_shape_tensors[idx].append(array_ops.shape(input_tensor))

  maximum_shapes = []
  for shapes_per_input in input_shape_tensors:
    maximum_shapes.append(
        math_ops.reduce_max(array_ops.stack(shapes_per_input), axis=0))

  padded_inputs = []
  real_shapes = []
  padding_maps = []
  for core_idx, inputs_per_core in enumerate(inputs):
    padded_inputs.append([])
    real_shapes.append([])
    real_shape_idx = len(inputs_per_core) - 1
    for idx, input_tensor in enumerate(inputs_per_core):
      input_shape_tensor = input_shape_tensors[idx][core_idx]
      input_shape = input_tensor.get_shape()
      padded_shape = padded_shapes[idx]

      # The static shape of inputs should be compatible with the given padded
      # shapes.
      input_shape.assert_is_compatible_with(padded_shape)

      if input_shape.is_fully_defined():
        # Do nothing if the shape of the whole tensor is already static.
        padded_inputs[core_idx].append(input_tensor)
      else:
        # Only pad the non static shape dimension.
        for i, s in enumerate(input_shape):
          if s.value is None:
            if core_idx == 0:
              real_shape_idx += 1
              padding_map = dynamic_padding.PaddingMap()
              padding_map.arg_index = idx
              padding_map.shape_index = i
              padding_map.padding_arg_index = real_shape_idx
              padding_maps.append(padding_map)
            real_shapes[core_idx].append(
                math_ops.cast(input_shape_tensor[i], dtypes.uint32))

        paddings = []
        for i, s in enumerate(padded_shape):
          if input_shape[i].value:
            # Don't pad if input shape is already static.
            padding = [0, 0]
          else:
            if s.value:
              # Pad to the given maximum value.
              padding = [0, s.value - input_shape_tensor[i]]
            else:
              # If maximum value is not given, then pad to the maximum dimension
              # among all the cores.
              padding = [0, maximum_shapes[idx][i] - input_shape_tensor[i]]
          paddings.append(padding)

        padded_input = array_ops.pad(input_tensor, paddings)
        padded_inputs[core_idx].append(padded_input)

  num_replicas = len(padded_inputs)
  for i in range(num_replicas):
    padded_inputs[i].extend(real_shapes[i])

  return padded_inputs, padding_maps


def split_compile_and_replicate(computation,
                                inputs=None,
                                infeed_queue=None,
                                device_assignment=None,
                                name=None,
                                use_tpu=True,
                                maximum_shapes=None):
  """Builds graph operators that runs compilation and replicated computation.

  This is a lower level interface than replicate that returns a separate compile
  and execute output tensor. In the generated graph the compile op feeds into
  the execute op and no additional compilation is incurred when running the
  compile op before the execute op. The compile op returns additional
  information about the compilation but does not return the compiled program.

  Args:
    computation: A Python function that builds the computation to replicate.
    inputs: A list of lists of input tensors or `None` (equivalent to
      `[[]]`), indexed by `[replica_num][input_num]`. All replicas must
      have the same number of inputs. Each input can be a nested structure
      containing values that are convertible to tensors. Note that passing an
      N-dimension list of compatible values will result in a N-dimention list of
      scalar tensors rather than a single Rank-N tensors. If you need different
      behavior, convert part of inputs to tensors with `tf.convert_to_tensor`.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to computation.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each replica of the computation uses
      only one core, and there is either only one replica, or the number of
      replicas is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
    use_tpu: When false, the input `computation` is executed on the XLA CPU/GPU
      backends. Currently, only supports a default placement (computation is
      placed on GPU if one is available, and on CPU if not).
    maximum_shapes: A nested structure of tf.TensorShape representing the shape
      to which the respective component of each input element in each replica
      should be padded. Any unknown dimensions (e.g. tf.Dimension(None) in a
      tf.TensorShape or -1 in a tensor-like object) will be padded to the
      maximum size of that dimension over all replicas. Note that if the input
      dimension is already static, we won't do padding on it and we require the
      maximum_shapes to have the same value or None on that dimension. The
      structure of `maximum_shapes` needs to be the same as `inputs[0]`.

  Returns:
    A list of lists with the first list corresponding to the compile op and the
    second a list of output tensors, indexed by `[replica_num][output_num]`.
  Raises:
    ValueError: If all replicas do not have equal numbers of input tensors.
    ValueError: If the number of inputs per replica does not match
      the number of formal parameters to `computation`.
    ValueError: If the static `inputs` dimensions don't match with the values
      given in `maximum_shapes`.
    ValueError: If the structure of inputs per replica does not match
      the structure of `maximum_shapes`.
  """
  del name
  inputs = [[]] if inputs is None else inputs

  metadata_kwargs = {}
  if device_assignment is not None:
    # Turn the Numpy array into a flattened list so we can pass it as an
    # operator attribute.
    metadata_kwargs = {
        "topology":
            device_assignment.topology.serialized(),
        "device_assignment":
            device_assignment.core_assignment.flatten().tolist()
    }
    # TODO(phawkins): remove this case after the forward compatibility window
    # expires on 2018-10-5.
    if api_compat.forward_compatible(2018, 10, 5):
      metadata_kwargs["num_cores_per_replica"] = (
          device_assignment.num_cores_per_replica)
    else:
      metadata_kwargs["computation_shape"] = [
          device_assignment.num_cores_per_replica
      ]

  if ((not isinstance(inputs, list)) or
      any(not isinstance(inp, (list, tuple)) for inp in inputs)):
    raise TypeError("tpu.replicate() inputs must be a list of lists/tuples")

  num_replicas = len(inputs)

  # No replicas? Nothing to do.
  if num_replicas == 0:
    return []

  # Checks all replicas have the same structure.
  for i in xrange(1, num_replicas):
    nest.assert_same_structure(inputs[0], inputs[i])

  # Flatten inputs.
  flat_inputs = [
      nest.flatten(per_replica_input) for per_replica_input in inputs
  ]
  # Converts inputs to Tensors.
  flat_inputs = [[ops.convert_to_tensor(x) for x in inp] for inp in flat_inputs]

  # Verifies that all replicas have matching numbers and types of inputs
  flat_input_types = [x.dtype for x in flat_inputs[0]]
  input_arity = len(inputs[0])
  flat_input_arity = len(flat_input_types)
  for i in range(num_replicas):
    if len(inputs[i]) != input_arity:
      raise ValueError("Replicas must have the same number of inputs. "
                       "Replica 0 had {} inputs, replica {} had {} "
                       "inputs.".format(input_arity, i, len(inputs[i])))

    types = [x.dtype for x in flat_inputs[i]]
    if types != flat_input_types:
      raise ValueError("Replicas must have matching input types. Replica 0 had "
                       "input types {}, replica {} had input types {}".format(
                           flat_input_types, i, types))

  arg_error = xla.check_function_argument_count(
      computation, input_arity, infeed_queue)
  if arg_error is not None:
    if infeed_queue is None:
      raise TypeError(
          "Supplied computation cannot be called with the specified inputs. "
          "You specified %d inputs: %s, but the computation needs %s" % (
              input_arity, str([i.name for i in inputs[0]]), arg_error))
    else:
      raise TypeError(
          "Supplied computation cannot be called with the specified inputs. "
          "You specified %d inputs: %s and %d additional inputs from infeed,"
          " but the computation needs %s" % (input_arity, str(
              [i.name
               for i in inputs[0]]), infeed_queue.number_of_tuple_elements,
                                             arg_error))

  if maximum_shapes:
    if infeed_queue:
      raise ValueError(
          "Dynamic input shapes are not supported with infeed queues")

    # Make sure maximum_shapes has the same structure as inputs.
    nest.assert_same_structure(inputs[0], maximum_shapes, check_types=False)

    # Flatten padded shapes.
    flat_maximum_shapes = nest.flatten(maximum_shapes)
    flat_maximum_shapes = [
        tensor_shape.TensorShape(s) for s in flat_maximum_shapes
    ]

    flat_inputs, padding_maps = _pad_all_input(flat_inputs, flat_maximum_shapes)

    serialized_padding_maps = []
    for padding_map in padding_maps:
      serialized_padding_maps.append(padding_map.SerializeToString())
    metadata_kwargs["padding_map"] = serialized_padding_maps

  graph = ops.get_default_graph()

  # Fan-in: Builds a TPUReplicatedInput node for each input.
  flat_replicated_inputs = []
  for i in range(0, len(flat_inputs[0])):
    replicas = [flat_inputs[replica][i] for replica in xrange(num_replicas)]
    flat_replicated_inputs.append(
        tpu_ops.tpu_replicated_input(replicas, name="input{}".format(i)))

  cluster_name = graph.unique_name("cluster")
  pivot = control_flow_ops.no_op(name=cluster_name + "/pivot")
  context = TPUReplicateContext(
      name=cluster_name, num_replicas=num_replicas, pivot=pivot)
  try:
    context.Enter()

    metadata = tpu_ops.tpu_replicate_metadata(
        num_replicas=num_replicas, use_tpu=use_tpu, **metadata_kwargs)

    with tpu_function.tpu_shard_context(
        num_replicas), ops.control_dependencies([metadata]):

      # Add identity ops so even unused inputs are "consumed" by the
      # computation. This is to avoid orphaned TPUReplicatedInput nodes.
      # TODO(phawkins): consider instead pruning unused TPUReplicatedInput
      # and eliding trivial TPUReplicatedInput/TPUReplicatedOutput pairs.
      flat_replicated_inputs = [
          array_ops.identity(x, name="replicated_input_{}".format(i))
          for i, x in enumerate(flat_replicated_inputs)
      ]
      for i in flat_replicated_inputs:
        # pylint: disable=protected-access
        # Add an attribute to the identity node so that they could be removed in
        # encapsulate TPU computation pass if unused. However we don't remove
        # inputs when dynamic padding is enabled.
        # TODO(rxsang): Use other ways except argument index in padding_map so
        # outside compilation can work with dynamic padding correctly.
        if maximum_shapes is None:
          i.op._set_attr("_tpu_input_identity",
                         attr_value_pb2.AttrValue(b=True))
        # pylint: enable=protected-access

      # Unflatten the computation inputs to match original input structure.
      computation_inputs = nest.pack_sequence_as(
          structure=inputs[0],
          flat_sequence=flat_replicated_inputs[:flat_input_arity])

      # If there is an infeed queue, adds the dequeued values to the
      # computation's inputs.
      if infeed_queue is not None:
        infeed_queue.set_number_of_shards(num_replicas)
        for t in infeed_queue.generate_dequeue_op():
          computation_inputs.append(t)

      # Only resource variables work inside a TPU computation, so turn on
      # resource variables for the computation.
      # TODO(phawkins): consider removing this code. It will
      # be less confusing to clients if they knowingly choose to use resource
      # variables.
      # Partitioned variables is not supported (b/112311320).
      vscope = variable_scope.get_variable_scope()
      saved_use_resource = vscope.use_resource
      saved_custom_getter = vscope.custom_getter

      def custom_getter(getter, name, *args, **kwargs):
        """Variables on TPU have a few restrictions."""
        partitioner = kwargs["partitioner"]
        if partitioner is not None:
          kwargs["partitioner"] = None
          logging.warning(
              "Partitioned variables are not supported on TPU. Got "
              "`partitioner` that is {} for variable {}. "
              "Setting `partitioner` to `None`."
              .format(partitioner, name))
        if saved_custom_getter is None:
          return getter(name, *args, **kwargs)
        else:
          return saved_custom_getter(getter, name, *args, **kwargs)

      vscope.set_use_resource(True)
      vscope.set_custom_getter(custom_getter)

      outputs = computation(*computation_inputs)

      vscope.set_use_resource(saved_use_resource)
      vscope.set_custom_getter(saved_custom_getter)

    outputs_is_flat = xla.is_flat(outputs)
    if outputs_is_flat:
      output_tensors, control_deps = _postprocess_flat_outputs(outputs)
    else:
      output_tensors, control_deps = _postprocess_non_flat_outputs(outputs)

    context.ExitResult(output_tensors)
  finally:
    context.report_unsupported_operations()
    context.Exit()
    host_compute_core = context.HostComputeCore()

  if host_compute_core:
    attr_value = attr_value_pb2.AttrValue()
    attr_value.list.s.extend([compat.as_bytes(x) for x in host_compute_core])
    metadata._set_attr("host_compute_core", attr_value)  # pylint: disable=protected-access

  with ops.control_dependencies([metadata]):
    if use_tpu:
      compile_status = tpu_ops.tpu_compilation_result()
      op = compile_status.op
      attr_value = attr_value_pb2.AttrValue(s=compat.as_bytes(cluster_name))
      op._set_attr(_TPU_COMPILATION_STATUS_ATTR, attr_value)  # pylint: disable=protected-access
    else:
      compile_status = control_flow_ops.no_op(name="compilation_status")

  if not output_tensors:
    # Returns a list of NoOps dependent on the replication Op, indexed by
    # [replica_num].
    return [
        compile_status,
        [
            control_flow_ops.group(control_deps, name="shard_%d" % i)
            for i in range(num_replicas)
        ]
    ]

  # Fan-out: Builds a TPUReplicatedOutput node for each output.
  replicated_outputs = [[] for i in xrange(num_replicas)]
  for i, t in enumerate(output_tensors):
    # Fan-out: Builds a TPUReplicatedOutput node for each output.
    ys = tpu_ops.tpu_replicated_output(
        t, num_replicas, name="output{}".format(i))

    # Wraps the outputs in identity operators so the names of any possible
    # `fetch` nodes are preserved by the replication rewrite.
    with ops.control_dependencies(control_deps):
      for replica in xrange(num_replicas):
        replicated_outputs[replica].append(
            array_ops.identity(
                ys[replica], name="output_%d_shard_%d" % (i, replica)))

  if not outputs_is_flat:
    replicated_outputs = [
        nest.pack_sequence_as(outputs, replica_outs)
        for replica_outs in replicated_outputs
    ]

  return [compile_status, replicated_outputs]


def _postprocess_flat_outputs(outputs):
  """Validates non-flat outputs, add backs device assignments and other attrs.

  Args:
    outputs: Output from `computation` inside `tpu.rewrite`.

  Returns:
    Tensors and Operations extracted from outputs.
  """
  # Following code segment is to preserve legacy behavior. Previously we only
  # supported flat outputs and thus for consistency it was nice to convert even
  # single element into a tuple. But now that we support arbitrary output
  # structure, this is no longer necessary.
  # TODO(b/121383831): Migrate all legacy use cases and delete this special
  # case.
  # If the computation returns `None`, make it an empty tuple.
  if outputs is None:
    outputs = tuple()
  # If the computation only returned one value, makes it a tuple.
  if not isinstance(outputs, collections.Sequence):
    outputs = (outputs,)

  # Append `no_op` here so that fetching any return value of this function
  # will trigger TPUExecute node.
  outputs += (control_flow_ops.no_op(),)
  try:
    with ops.device(core(0)):
      outputs = [
          o if isinstance(o, ops.Operation) else ops.convert_to_tensor(o)
          for o in outputs
      ]
  except Exception as e:
    raise ValueError(
        "TPU function return values must all either be Operations or "
        "convertible to Tensors. Got '%s'" % str(e))

  # Separates the returned Operations and Tensors.
  output_operations = [o for o in outputs if isinstance(o, ops.Operation)]
  output_tensors = [o for o in outputs if not isinstance(o, ops.Operation)]

  if outputs != output_tensors + output_operations:
    raise ValueError(
        "TPU functions must return zero-or more Tensor values followed by "
        "zero or more Operations.")

  # Wraps outputs in Identity ops. Otherwise a replicated input copied
  # straight to an output would bypass the replicate(). This would be bad
  # because the TPUReplicatedInput/TPUReplicatedOutput operator would not
  # be rewritten away, leading to a runtime error.
  # TODO(phawkins): extend the rewrite to elide these nodes instead.
  new_output_tensors = []
  for t in output_tensors:
    with ops.device(t.device if t.device else core(0)):
      o = array_ops.identity(t)
      # pylint: disable=protected-access
      o.op._set_attr("_tpu_output_identity", attr_value_pb2.AttrValue(b=True))
      # pylint: enable=protected-access
      new_output_tensors.append(o)
  return new_output_tensors, output_operations


def _postprocess_non_flat_outputs(outputs):
  """Validates non-flat outputs, add backs device assignments and other attrs.

  Args:
    outputs: Output from `computation` inside `tpu.rewrite`.

  Returns:
    Tensors extracted from outputs and an empty list because Operations are not
    allowed in non-flat outputs..
  """

  # Flatten output items.
  flat_outputs = nest.flatten(outputs)

  # Convert all non-Operation outputs to Tensors.
  for i, o in enumerate(flat_outputs):
    if isinstance(o, ops.Operation):
      raise ValueError(
          "tpu.rewrite does not support Operation as return value in non-flat "
          "output structure. You can set returned Operations as control "
          "dependencies of returned Tensors so Operations are triggered when "
          'Tensors are evaluated. Operation found: "%s"' % o.name)

    try:
      o = ops.convert_to_tensor(o)
    except Exception as e:
      raise ValueError(
          "TPU function return values must all either be Operations or "
          'convertible to Tensors. Got error: "%s"' % str(e))

    # Wraps outputs in Identity ops. Otherwise a replicated input copied
    # straight to an output would bypass the replicate(). This would be bad
    # because the TPUReplicatedInput/TPUReplicatedOutput operator would not
    # be rewritten away, leading to a runtime error.
    # TODO(phawkins): extend the rewrite to elide these nodes instead.
    with ops.device(core(0)):
      o = array_ops.identity(o)
      # pylint: disable=protected-access
      o.op._set_attr("_tpu_output_identity", attr_value_pb2.AttrValue(b=True))
      # pylint: enable=protected-access
      flat_outputs[i] = array_ops.identity(o)

  # All flat_outputs are Tensors, and no Operations.
  return flat_outputs, []


def split_compile_and_shard(computation,
                            inputs=None,
                            num_shards=1,
                            input_shard_axes=None,
                            outputs_from_all_shards=True,
                            output_shard_axes=None,
                            infeed_queue=None,
                            device_assignment=None,
                            name=None):
  """Shards `computation` for parallel execution.

  `inputs` must be a list of Tensors or None (equivalent to an empty list), each
  of which has a corresponding split axis (from `input_shard_axes`). Each input
  is split into `num_shards` pieces along the corresponding axis, and
  computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  If `outputs_from_all_shards` is true, the outputs from all shards of
  `computation` are concatenated back together along their `output_shards_axes`.
  Otherwise, each output is taken from an arbitrary shard.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: A Python function that builds a computation to apply to each
      shard of the input.
    inputs: A list of input tensors or None (equivalent to an empty list). Each
      input tensor has a corresponding shard axes, given by `input_shard_axes`,
      which must have size divisible by `num_shards`.
    num_shards: The number of shards.
    input_shard_axes: A list of dimensions along which to shard `inputs`, or
      `None`. `None` means "shard all inputs along dimension 0". If not `None`,
      there must be one dimension per input.
    outputs_from_all_shards: Boolean or list of boolean. For each output, if
      `True`, outputs from all shards are concatenated along the corresponding
      `output_shard_axes` entry. Otherwise, each output is taken
      from an arbitrary shard. If the argument is a boolean, the argument's
      value is used for each output.
    output_shard_axes: A list of dimensions along which to concatenate the
      outputs of `computation`, or `None`. `None` means "concatenate all outputs
      along dimension 0". If not `None`, there must be one dimension per output.
      Ignored if `outputs_from_all_shards` is False.
    infeed_queue: If not `None`, the `InfeedQueue` to use to augment the inputs
      of `computation`.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each shard of the computation uses
      only one core, and there is either only one shard, or the number of shards
      is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
  Returns:
    A tuple of (compile op, [output tensors]).
  Raises:
    ValueError: If num_shards <= 0
    ValueError: If len(input_shard_axes) != len(inputs)
    ValueError: If len(output_shard_axes) != len(outputs from `computation`)
  """
  # TODO(phawkins): consider adding support for broadcasting Tensors passed as
  # inputs.

  if num_shards <= 0:
    raise ValueError("num_shards must be a positive integer.")

  inputs = [] if inputs is None else inputs
  if not isinstance(inputs, list):
    raise TypeError("tpu.shard()'s inputs must be a list of Tensors or None.")

  # Converts inputs to Tensors.
  inputs = [ops.convert_to_tensor(x) for x in inputs]

  if input_shard_axes is None:
    input_shard_axes = [0] * len(inputs)
  if len(inputs) != len(input_shard_axes):
    raise ValueError("Length of input_shard_axes must be equal to the number "
                     "of inputs.")

  if inputs:
    # Splits the `inputs` along the corresponding `input_shard_axes`, giving
    # lists with layout [input][shard]
    split_inputs = [
        array_ops.split(x, num_shards, axis=axis)
        for (axis, x) in zip(input_shard_axes, inputs)]

    # Transposes the input lists to have layout [shard][input]
    transposed_inputs = [list(i) for i in zip(*split_inputs)]
  else:
    transposed_inputs = [[]] * num_shards

  compile_op, outputs = split_compile_and_replicate(
      computation,
      transposed_inputs,
      infeed_queue=infeed_queue,
      device_assignment=device_assignment,
      name=name)

  # There must be at least one shard since num_shards > 0.
  # TODO(b/36647078) remove disable when pylint bug is fixed.
  # pylint: disable=indexing-exception
  if isinstance(outputs[0], ops.Operation):
    # pylint: enable=indexing-exception
    # There were no outputs from the computation and replicate returned a list
    # of NoOps with control dependencies on the computation. Return the first
    # one so it can be used as a control dependency or fetch node.
    # TODO(b/36647078) remove disable when pylint bug is fixed.
    # pylint: disable=indexing-exception
    return compile_op, [outputs[0]]
    # pylint: enable=indexing-exception

  # TODO(b/36647078) remove disable when pylint bug is fixed.
  # pylint: disable=indexing-exception
  num_outputs = len(outputs[0])
  # pylint: enable=indexing-exception

  if output_shard_axes is None:
    output_shard_axes = [0] * num_outputs
  if num_outputs != len(output_shard_axes):
    raise ValueError("Length of output_shard_axes must be equal to the number "
                     "of outputs.")

  if isinstance(outputs_from_all_shards, bool):
    outputs_from_all_shards = [outputs_from_all_shards] * num_outputs

  if num_outputs != len(outputs_from_all_shards):
    raise ValueError("Length of outputs_from_all_shards must be equal to the "
                     "number of outputs.")

  results = []
  for (axis, all_shards, x) in zip(output_shard_axes, outputs_from_all_shards,
                                   zip(*outputs)):
    if all_shards:
      # Concatenate all of the outputs together (use stack for scalars).
      shape = x[0].shape
      is_scalar = shape is not None and (shape.ndims == 0)
      results.append((array_ops.stack(list(x)) if is_scalar
                      else array_ops.concat(list(x), axis=axis)))
    else:
      # TODO(phawkins): use a smarter policy, e.g., round-robin across shards.
      results.append(x[0])

  return compile_op, results


def shard(computation,
          inputs=None,
          num_shards=1,
          input_shard_axes=None,
          outputs_from_all_shards=True,
          output_shard_axes=None,
          infeed_queue=None,
          device_assignment=None,
          name=None):
  """Shards `computation` for parallel execution.

  `inputs` must be a list of Tensors or None (equivalent to an empty list), each
  of which has a corresponding split axis (from `input_shard_axes`). Each input
  is split into `num_shards` pieces along the corresponding axis, and
  computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  TODO(phawkins): consider adding support for broadcasting Tensors passed
  as inputs.

  If `outputs_from_all_shards` is true, the outputs from all shards of
  `computation` are concatenated back together along their `output_shards_axes`.
  Otherwise, each output is taken from an arbitrary shard.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: A Python function that builds a computation to apply to each
      shard of the input.
    inputs: A list of input tensors or None (equivalent to an empty list). Each
      input tensor has a corresponding shard axes, given by `input_shard_axes`,
      which must have size divisible by `num_shards`.
    num_shards: The number of shards.
    input_shard_axes: A list of dimensions along which to shard `inputs`, or
      `None`. `None` means "shard all inputs along dimension 0". If not `None`,
      there must be one dimension per input.
    outputs_from_all_shards: Boolean or list of boolean. For each output, if
      `True`, outputs from all shards are concatenated along the corresponding
      `output_shard_axes` entry. Otherwise, each output is taken
      from an arbitrary shard. If the argument is a boolean, the argument's
      value is used for each output.
    output_shard_axes: A list of dimensions along which to concatenate the
      outputs of `computation`, or `None`. `None` means "concatenate all outputs
      along dimension 0". If not `None`, there must be one dimension per output.
      Ignored if `outputs_from_all_shards` is False.
    infeed_queue: If not `None`, the `InfeedQueue` to use to augment the inputs
      of `computation`.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each shard of the computation uses
      only one core, and there is either only one shard, or the number of shards
      is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
  Returns:
    A list of output tensors.
  Raises:
    ValueError: If num_shards <= 0
    ValueError: If len(input_shard_axes) != len(inputs)
    ValueError: If len(output_shard_axes) != len(outputs from `computation`)
  """
  return split_compile_and_shard(
      computation,
      inputs=inputs,
      num_shards=num_shards,
      input_shard_axes=input_shard_axes,
      outputs_from_all_shards=outputs_from_all_shards,
      output_shard_axes=output_shard_axes,
      infeed_queue=infeed_queue,
      device_assignment=device_assignment,
      name=name)[1]


def batch_parallel(computation,
                   inputs=None,
                   num_shards=1,
                   infeed_queue=None,
                   device_assignment=None,
                   name=None):
  """Shards `computation` along the batch dimension for parallel execution.

  Convenience wrapper around shard().

  `inputs` must be a list of Tensors or None (equivalent to an empty list).
  Each input is split into `num_shards` pieces along the 0-th dimension, and
  computation is applied to each shard in parallel.

  Tensors are broadcast to all shards if they are lexically captured by
  `computation`. e.g.,

  x = tf.constant(7)
  def computation():
    return x + 3
  ... = shard(computation, ...)

  The outputs from all shards are concatenated back together along their 0-th
  dimension.

  Inputs and outputs of the computation must be at least rank-1 Tensors.

  Args:
    computation: A Python function that builds a computation to apply to each
      shard of the input.
    inputs: A list of input tensors or None (equivalent to an empty list). The
      0-th dimension of each Tensor must have size divisible by `num_shards`.
    num_shards: The number of shards.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to `computation`.
    device_assignment: If not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. Uses a default device assignment if `None`. The
      `DeviceAssignment` may be omitted if each shard of the computation uses
      only one core, and there is either only one shard, or the number of shards
      is equal to the number of cores in the TPU system.
    name: (Deprecated) Does nothing.
  Returns:
    A list of output tensors.
  Raises:
    ValueError: If `num_shards <= 0`
  """
  return shard(
      computation,
      inputs,
      num_shards=num_shards,
      infeed_queue=infeed_queue,
      device_assignment=device_assignment,
      name=name)


def rewrite(computation,
            inputs=None,
            infeed_queue=None,
            device_assignment=None,
            name=None):
  """Rewrites `computation` for execution on a TPU system.

  Args:
    computation: A Python function that builds a computation to apply to the
      input. If the function takes n inputs, 'inputs' should be a list of n
      tensors.

      `computation` may return a list of operations and tensors. Tensors must
      come before operations in the returned list.  The return value of
      `rewrite` is a list of tensors corresponding to the tensors from the
      output of `computation`.

      All `Operation`s constructed during `computation` will be executed when
      evaluating any of the returned output tensors, not just the ones returned.
    inputs: A list of input tensors or `None` (equivalent to an empty list).
      Each input can be a nested structure containing values that are
      convertible to tensors. Note that passing an N-dimension list of
      compatible values will result in a N-dimention list of scalar tensors
      rather than a single Rank-N tensors. If you need different behavior,
      convert part of inputs to tensors with `tf.convert_to_tensor`.
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to `computation`.
    device_assignment: if not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. May be omitted for a single-core computation, in which
      case the core attached to task 0, TPU device 0 is used.
    name: (Deprecated) Does nothing.
  Returns:
    Same data structure as if computation(*inputs) is called directly with some
    exceptions for correctness. Exceptions include:
      1) None output: a NoOp would be returned which control-depends on
         computation.
      2) Single value output: A tuple containing the value would be returned.
      3) Operation-only outputs: a NoOp would be returned which
         control-depends on computation.
      TODO(b/121383831): Investigate into removing these special cases.
  """
  # TODO(b/36647078) remove disable when pylint bug is fixed.
  # pylint: disable=indexing-exception
  return replicate(
      computation,
      None if inputs is None else [inputs],
      infeed_queue=infeed_queue,
      device_assignment=device_assignment,
      name=name)[0]
  # pylint: enable=indexing-exception

  # Operations that indicate some error in the user's inference graph.
_BLACKLISTED_INFERENCE_OPS = set([
    "ReadVariableOp",
    "AssignVariableOp",
    "AssignAddVariableOp",
    "AssignSubVariableOp",
    "VarHandleOp",
    "Variable",
    "VariableV2",
])


def under_tpu_inference_context():
  """Check if it is currently under `tpu.rewrite_for_inference()`."""
  graph = ops.get_default_graph()

  context = graph._get_control_flow_context()  # pylint: disable=protected-access
  while context:
    if isinstance(context, _TPUInferenceContext):
      return True
    context = context.outer_context

  return False


class _TPUInferenceContext(control_flow_ops.XLAControlFlowContext):
  """A `ControlFlowContext` for nodes inside a TPU inference computation.

  The primary role of `TPUReplicateContext` is to sanity check operators inside
  a tpu.rewrite_for_inference() computation.
  """

  def __init__(self, name):
    super(_TPUInferenceContext, self).__init__()
    self._name = name

  def AddOp(self, op):
    self._AddOpInternal(op)

  def _AddOpInternal(self, op):
    # pylint: disable=protected-access
    if op.type in _BLACKLISTED_INFERENCE_OPS:
      raise NotImplementedError(
          "Operation of type %s (%s) is not supported on the TPU for inference."
          " Execution will fail if this op is used in the graph. Make sure your"
          " variables are using variable_scope." % (op.type, op.name))
    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  def AddValue(self, val):
    result = val
    if self._outer_context:
      result = self._outer_context.AddValue(val)
    return result

  def AddInnerOp(self, op):
    self._AddOpInternal(op)

  @property
  def grad_state(self):
    return None


@experimental
def validate_inference_rewrite_for_variables(graph):
  """Validates whether rewrite_for_inference() 'worked' for variables.

     The rewrite_for_inference() method is supposed to append GuaranteeConstOps
     after ReadVariableOps, but this mechanism works only if you are using
     tf.get_variable() to create and access variables in your tpu computation.
     This validation method can be called immediately after calling
     tpu.rewrite_for_inference() to check whether GuaranteeConstOps where added
     to the graph.

     Typical usages:
       tpu.validate_inference_rewrite_for_variables(tf.get_default_graph())

       tpu.validate_inference_rewrite_for_variables(sess.graph)

  Args:
    graph: The graph which needs to be validated.
  Raises:
    RuntimeError: if validation failed.
  """
  if not any(x.type == "GuaranteeConst" for x in graph.get_operations()):
    raise RuntimeError(
        "No GuaranteeConst ops found in the graph after running "
        "tpu.rewrite_for_inference(...). Please check that you are using "
        "tf.get_variable() to create and access variables in your tpu "
        "computation.")


@experimental
def rewrite_for_inference(computation,
                          inputs=None,
                          infeed_queue=None,
                          device_assignment=None,
                          name=None):
  """Rewrites `computation` for inference on a TPU system.

     Other than 'rewriting' the computation to run on a TPU, if using variables
     in your computation, it moves the ReadVariableOps outside the TPU
     computation, and adds GuaranteeConst ops just after the ReadVariableOps.
     This mechanism works only if you are using tf.get_variable() to create and
     access variables in your tpu computation. You can validate whether this
     worked, by calling validate_inference_rewrite_for_variables() method
     immediately after this method to check whether GuaranteeConstOps where
     added to the graph.

  Args:
    computation: A Python function that builds a computation to apply to the
      input. If the function takes n inputs, 'inputs' should be a list of n
      tensors. If the function returns m outputs, rewrite will return a list of
      m tensors.
    inputs: A list of input tensors or `None` (equivalent to an empty list).
    infeed_queue: If not `None`, the `InfeedQueue` from which to append a tuple
      of arguments as inputs to `computation`.
    device_assignment: if not `None`, a `DeviceAssignment` describing the
      mapping between logical cores in the computation with physical cores in
      the TPU topology. May be omitted for a single-core computation, in which
      case the core attached to task 0, TPU device 0 is used.
    name: The name of the operator.
  Returns:
    A list of output tensors.
  """

  def guarantee_const_getter(getter, name, *args, **kwargs):
    with ops.control_dependencies(None):
      return array_ops.guarantee_const(
          getter(name, *args, **kwargs), name=name + "/GuaranteeConst")

  def wrapped_computation(*args, **kwargs):
    """Execute computation under `_TPUInferenceContext`."""
    context = _TPUInferenceContext(
        name=ops.get_default_graph().unique_name("rewrite_for_inference"))
    try:
      context.Enter()

      vscope = variable_scope.get_variable_scope()
      prev_custom_getter = vscope.custom_getter
      prev_caching_device = vscope.caching_device
      vscope.set_custom_getter(guarantee_const_getter)
      vscope.set_caching_device(lambda op: op.device)

      result = computation(*args, **kwargs)

      vscope.set_custom_getter(prev_custom_getter)
      vscope.set_caching_device(prev_caching_device)
    finally:
      context.Exit()
    return result

  # pylint: disable=undefined-variable
  return rewrite(
      wrapped_computation,
      inputs=inputs,
      infeed_queue=infeed_queue,
      device_assignment=device_assignment,
      name=name)
  # pylint: enable=undefined-variable
