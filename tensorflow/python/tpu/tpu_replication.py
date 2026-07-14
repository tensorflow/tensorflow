# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file8 except in compliance with the License.
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

"""OutsideCompilation, TPUReplicateContext, and supporting functions."""

from typing import Any, Callable, List, Optional, Text, Tuple, Union
from absl import logging
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables
from tensorflow.python.tpu import device_assignment as device_assignment_lib
from tensorflow.python.tpu.ops import tpu_ops
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

_MAX_WARNING_LINES = 5
_TPU_REPLICATE_ATTR = "_tpu_replicate"
_OUTSIDE_COMPILATION_ATTR = "_xla_outside_compilation"
_MAP_OUTSIDE_COMPILATION_ATTR = "_xla_map_outside_compilation"

# Operations that indicate some error in the users graph, e.g. a placeholder
# that's introduced outside of the infeed.
_DENYLISTED_OPS = frozenset([
    "Placeholder",
])


# XLA doesn't currently support reading of intermediate tensors, thus some ops
# are not supported.
_UNSUPPORTED_OPS = frozenset([
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


def is_tpu_strategy(strategy: Any) -> bool:
  is_tpu_strat = lambda k: k.__name__.startswith("TPUStrategy")
  clz = strategy.__class__
  return is_tpu_strat(clz) or any(map(is_tpu_strat, clz.__bases__))


def _enclosing_tpu_device_assignment(
) -> Optional[device_assignment_lib.DeviceAssignment]:
  if not distribute_lib.has_strategy():
    return None
  strategy = distribute_lib.get_strategy()
  if not is_tpu_strategy(strategy):
    return None
  return strategy.extended._device_assignment  # pylint: disable=protected-access


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

  def __init__(self, name: Text, num_replicas: int, pivot: ops.Operation):
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
    self._is_map_outside_compilation = False
    self._outside_compilation_v2_context = None
    self._outside_compilation_counter = 0
    self._in_gradient_colocation = None
    self._gradient_colocation_stack = []
    self._host_compute_core = []
    self._name = name
    self._tpu_replicate_attr = attr_value_pb2.AttrValue(
        s=compat.as_bytes(self._name)
    )
    self._unsupported_ops = []
    self._pivot = pivot
    self._replicated_vars = {}

  def get_replicated_var_handle(self,
                                name: Text,
                                handle_id: Text,
                                vars_: Union[List[core_types.Tensor],
                                             List[variables.Variable]],
                                is_mirrored: bool = False,
                                is_packed: bool = False) -> core_types.Tensor:
    """Returns a variable handle for replicated TPU variable 'var'.

    This is a method used by an experimental replicated variable implementation
    and is not intended as a public API.

    Args:
      name: The common name of the variable.
      handle_id: Unique ID of the variable handle, used as the cache key.
      vars_: The replicated TPU variables or handles.
      is_mirrored: Whether the variables are mirrored, which guarantees the
        values in each replica are always the same.
      is_packed: Whether the replicated variables are packed into one variable.

    Returns:
      The handle of the TPU replicated input node.
    """
    device_assignment = _enclosing_tpu_device_assignment()
    # We don't need to put device assignment as part of the replicated_vars key
    # because each TPUReplicateContext will only have one device assignment.
    handle = self._replicated_vars.get(handle_id)
    if handle is not None:
      return handle

    if device_assignment is not None and not is_packed:
      # Find a variable copy for each replica in the device assignment.
      # Note that the order of devices for replicas for the variable and the
      # device assignment might not match.
      job_name = pydev.DeviceSpec.from_string(vars_[0].device).job
      devices_to_vars = {device_util.canonicalize(v.device): v for v in vars_}
      replicated_vars = []
      for replica_id in range(device_assignment.num_replicas):
        for logical_core in range(device_assignment.num_cores_per_replica):
          device = device_util.canonicalize(
              device_assignment.tpu_device(
                  replica=replica_id, logical_core=logical_core, job=job_name))
          if device in devices_to_vars:
            replicated_vars.append(devices_to_vars[device])
            break
        else:
          raise ValueError(
              "Failed to find a variable on any device in replica {} for "
              "current device assignment".format(replica_id)
          )
    else:
      replicated_vars = vars_

    # Builds a TPUReplicatedInput node for the variable, if one does not already
    # exist. The TPUReplicatedInput node must belong to the enclosing
    # control-flow scope of the TPUReplicateContext.
    # TODO(phawkins): consider changing the contract of the TPU encapsulation
    # so the TPUReplicatedInput nodes go inside the TPUReplicateContext scope
    # instead.

    _, graph = _enclosing_tpu_context_and_graph()
    with graph.as_default():
      # If replicated_vars are variables, get the handles. Note that this can be
      # done inside TPUReplicateContext because replicated_vars.handle may
      # create new ops.
      if isinstance(replicated_vars[0], variables.Variable):
        replicated_vars = [v.handle for v in replicated_vars]
      # pylint: disable=protected-access
      saved_context = graph._get_control_flow_context()
      graph._set_control_flow_context(self.outer_context)
      handle = tpu_ops.tpu_replicated_input(
          replicated_vars,
          name=name + "/handle",
          is_mirrored_variable=is_mirrored,
          is_packed=is_packed)
      graph._set_control_flow_context(saved_context)
      # pylint: enable=protected-access
    self._replicated_vars[handle_id] = handle
    return handle

  def report_unsupported_operations(self) -> None:
    if self._unsupported_ops:
      op_str = "\n".join(
          "  %s (%s)" % (op.type, op.name) for op in
          self._unsupported_ops[:_MAX_WARNING_LINES])
      logging.warning("%d unsupported operations found: \n%s",
                      len(self._unsupported_ops), op_str)
      if len(self._unsupported_ops
            ) > _MAX_WARNING_LINES:
        logging.warning("... and %d more",
                        (len(self._unsupported_ops) - _MAX_WARNING_LINES))

  def EnterGradientColocation(self, op: ops.Operation, gradient_uid: Text):
    if op is not None:
      if ops.get_default_graph()._control_flow_context is None:  # pylint: disable=protected-access
        # If we are in TF 2 functions (control flow V2 functions, or
        # tf.function()), we need to attach _xla_outside_compilation attribute
        # directly because we are not in TPUReplicateContext.
        try:
          outside_attr = op.get_attr(_OUTSIDE_COMPILATION_ATTR).decode("ascii")
        except ValueError:
          # The attr was not present: do nothing.
          return
        parts = outside_attr.split(".")
        cluster = parts[0] + "." + gradient_uid
        self._outside_compilation_v2_context = OutsideCompilationV2Context(
            cluster)
        self._outside_compilation_v2_context.Enter()
        return
      self._gradient_colocation_stack.append(op)
      if not self._outside_compilation_cluster:
        try:
          outside_attr = op.get_attr(_OUTSIDE_COMPILATION_ATTR).decode("ascii")
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

  def ExitGradientColocation(self, op: ops.Operation, gradient_uid: Text):
    if op is not None:
      if ops.get_default_graph()._control_flow_context is None:  # pylint: disable=protected-access
        # Inside a TF2 tf.function or control flow graph and `op` was not
        # marked to be outside compiled.
        assert self._outside_compilation_v2_context is None
        return
      if self._outside_compilation_v2_context is not None:
        # Inside a TF2 tf.function or control flow graph and `op` was
        # marked to be outside compiled.
        self._outside_compilation_v2_context.Exit()
        self._outside_compilation_v2_context = None
        return
      if not self._gradient_colocation_stack:
        raise errors.InternalError(
            op.node_def, op,
            ("Badly nested gradient colocation: "
             + f"empty stack when popping Op {op.name}")
        )
      last_op = self._gradient_colocation_stack.pop()
      if op is last_op:
        if op is self._in_gradient_colocation:
          self._in_gradient_colocation = None
          self._ExitOutsideCompilationScope()
      else:
        raise errors.InternalError(
            op.node_def, op,
            ("Badly nested gradient colocation, " +
             f"expected {last_op}, got {op.name}")
        )

  def _EnterOutsideCompilationScope(
      self, cluster: Optional[Text] = None, is_map_outside_compilation=False
  ):
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

      def _set_device_from_string(self, device_str):
        self._device = device_str

    if self._outside_compilation_cluster:
      raise NotImplementedError("Cannot nest outside_compilation clusters")
    if cluster:
      self._outside_compilation_cluster = cluster
    else:
      self._outside_compilation_cluster = str(self._outside_compilation_counter)
      self._outside_compilation_counter += 1
    if is_map_outside_compilation:
      self._is_map_outside_compilation = True
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
      raise ValueError(
          "Attempted to exit outside_compilation scope when not in scope")
    self._outside_compilation_cluster = None
    self._is_map_outside_compilation = False
    graph = ops.get_default_graph()
    graph._device_function_stack = self._oc_dev_fn_stack  # pylint: disable=protected-access

  def Enter(self) -> None:
    if not self._outer_device_function_stack:
      # Capture the device function stack at the time of first entry
      # since that is the stack that will be used outside_compilation.
      graph = ops.get_default_graph()
      # pylint: disable=protected-access
      self._outer_device_function_stack = graph._device_function_stack.copy()
      # pylint: enable=protected-access
    super(TPUReplicateContext, self).Enter()

  def HostComputeCore(self) -> List[Text]:
    return self._host_compute_core

  def _RemoveExternalControlEdges(
      self,
      op: ops.Operation) -> Tuple[List[ops.Operation], List[ops.Operation]]:
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

  def AddOp(self, op: ops.Operation) -> None:
    # pylint: disable=protected-access
    if op.type in _DENYLISTED_OPS:
      logging.error(
          "Operation of type %s (%s) is not supported on the TPU. "
          "Execution will fail if this op is used in the graph. ", op.type,
          op.name)

    if op.type in _UNSUPPORTED_OPS:
      self._unsupported_ops.append(op)

    if any(x.dtype._is_ref_dtype for x in op.inputs):
      raise NotImplementedError(
          f"Non-resource Variables are not supported inside TPU computations "
          f"(operator name: {op.name})")

    # TensorFlowOpLayer may clone nodes that are in tpu.rewrite()s. It'll add
    # the "_cloned" attribute and we should continue in that case.
    if (_TPU_REPLICATE_ATTR in op.node_def.attr and
        "_cloned" not in op.node_def.attr):
      raise ValueError(f"TPU computations cannot be nested on op ({op})")
    op._set_attr(_TPU_REPLICATE_ATTR, self._tpu_replicate_attr)
    if self._outside_compilation_cluster:
      op._set_attr(
          _OUTSIDE_COMPILATION_ATTR,
          attr_value_pb2.AttrValue(
              s=compat.as_bytes(self._outside_compilation_cluster)))
    if self._is_map_outside_compilation:
      op._set_attr(
          _MAP_OUTSIDE_COMPILATION_ATTR,
          attr_value_pb2.AttrValue(b=True),
      )
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
      for index in range(len(op.inputs)):
        x = op.inputs[index]
        real_x = self.AddValue(x)
        if real_x is not x:
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

  def AddValue(self, val: core_types.Tensor) -> core_types.Tensor:
    """Add `val` to the current context and its outer context recursively."""
    if not self._outer_context:
      return val

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

  def AddInnerOp(self, op: ops.Operation):
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

  def GetControlPivot(self) -> ops.Operation:
    return self._pivot

  def RequiresUniqueFunctionRetracing(self):
    # More context: b/158152827. TPU stack uses the TPUReplicateContext to
    # create replicated variable handles and cluster TPU computations, thus we
    # always retrace a tf.function when the wrapped TPUReplicateContext changes.
    return True


def _enclosing_tpu_context_and_graph() -> Tuple[Any, Any]:
  """Returns the TPUReplicateContext and its associated graph."""
  graph = ops.get_default_graph()
  while graph is not None:
    # pylint: disable=protected-access
    context_ = graph._get_control_flow_context()
    # pylint: enable=protected-access
    while context_ is not None:
      if isinstance(context_, TPUReplicateContext):
        return context_, graph
      context_ = context_.outer_context
    graph = getattr(graph, "outer_graph", None)
  raise ValueError("get_replicated_var_handle() called without "
                   "TPUReplicateContext. This shouldn't happen. Please file "
                   "a bug.")


class OutsideCompilationV2Context(control_flow_ops.ControlFlowContext):
  """The context for outside compilation in Tensorflow 2.0.

  Every op added in this context will be assigned an _xla_outside_compilation
  attribute.
  """

  def __init__(self, name: Text, is_map_outside_compilation=False):
    control_flow_ops.ControlFlowContext.__init__(self)
    self._name = name
    self._is_map_outside_compilation = is_map_outside_compilation

  def AddOp(self, op: ops.Operation) -> None:
    if self._outer_context:
      self._outer_context.AddOp(op)
    self._set_outside_compilation_attributes(op)

  def AddInnerOp(self, op: ops.Operation) -> None:
    if self._outer_context:
      self._outer_context.AddInnerOp(op)
    self._set_outside_compilation_attributes(op)

  def to_control_flow_context_def(self, context_def, export_scope=None):
    raise NotImplementedError

  def _set_outside_compilation_attributes(self, op: ops.Operation) -> None:
    # pylint: disable=protected-access
    op._set_attr(
        _OUTSIDE_COMPILATION_ATTR,
        attr_value_pb2.AttrValue(s=compat.as_bytes(self._name)),
    )
    if self._is_map_outside_compilation:
      op._set_attr(
          _MAP_OUTSIDE_COMPILATION_ATTR, attr_value_pb2.AttrValue(b=True)
      )
    # pylint: enable=protected-access


def outside_compilation_impl(
    is_map, computation: Callable[..., Any], *args, **kwargs
) -> Any:
  """Tags ops in `computation` with outside compilation attributes for ordinary `outside_compilation` or `map_outside_compilation`."""
  args = [] if args is None else args
  graph = ops.get_default_graph()

  # If we are in TF 2 functions (control flow V2 functions, or tf.function()),
  # we need to attach _xla_outside_compilation attribute directly because we are
  # not in TPUReplicateContext.
  if isinstance(graph, func_graph.FuncGraph):
    try:
      tpu_context, _ = _enclosing_tpu_context_and_graph()
    except ValueError:
      logging.warning(
          "Outside compilation attempted outside TPUReplicateContext "
          "scope. As no enclosing TPUReplicateContext can be found, "
          "returning the result of `computation` as is."
      )
      return computation(*args, **kwargs)

    # pylint: disable=protected-access
    outside_compilation_name = str(tpu_context._outside_compilation_counter)
    tpu_context._outside_compilation_counter = (
        tpu_context._outside_compilation_counter + 1
    )
    # pylint: enable=protected-access

    outside_compilation_context = OutsideCompilationV2Context(
        outside_compilation_name, is_map_outside_compilation=is_map
    )
    outside_compilation_context.Enter()
    args = [] if args is None else args
    retval = computation(*args, **kwargs)
    outside_compilation_context.Exit()
    return retval

  # If we are in a TPUReplicateContext, signal that we are now
  # outside_compilation
  initial_context = graph._get_control_flow_context()  # pylint: disable=protected-access
  context = initial_context
  while context:
    if isinstance(context, TPUReplicateContext):
      context._EnterOutsideCompilationScope(is_map_outside_compilation=is_map)  # pylint: disable=protected-access
    context = context.outer_context

  retval = computation(*args, **kwargs)

  # If we are in a TPUReplicateContext, signal that we are no longer
  # outside_compilation
  final_context = graph._get_control_flow_context()  # pylint: disable=protected-access
  if initial_context is not final_context:
    raise NotImplementedError(
        "Control-flow context cannot be different at start and end of an "
        "outside_compilation scope"
    )
  context = initial_context
  while context:
    if isinstance(context, TPUReplicateContext):
      context._ExitOutsideCompilationScope()  # pylint: disable=protected-access
    context = context.outer_context

  return retval


@tf_export(v1=["tpu.outside_compilation"])
def outside_compilation(
    computation: Callable[..., Any], *args, **kwargs
) -> Any:
  """Builds part of a computation outside any current TPU replicate scope.

  `tf.tpu.outside_compilation()` is used to run ops in `computation` on CPU
  instead of running on TPU. For example, users can run ops that are not
  supported on TPU's (e.g. tf.summary.write()) by explicitly placing those
  ops on CPU's. Below usage of outside compilation will place ops in
  `computation_with_string_ops` on CPU.

  Example usage:

  ```python
  def computation_with_string_ops(x):
    # strings types are not supported on TPU's and below ops must
    # run on CPU instead.
    output = tf.strings.format('1{}', x)
    return tf.strings.to_number(output)

  def tpu_computation():
    # Expected output is 11.
    output = tf.tpu.outside_compilation(computation_with_string_ops, 1)
  ```

  Outside compilation should be called inside TPUReplicateContext. That is,
  `tf.tpu.outside_compilation()` should be called inside a function that is
  passed to `tpu.split_compile_and_replicate()` -- this is implied when
  outside compilation is invoked inside a function passed to TPUStrategy
  `run()`. If invoked outside of TPUReplicateContext,
  then this simply returns the result of `computation`, and therefore,
  would be a no-op. Note that outside compilation is different from
  `tf.distribute.experimental.TPUStrategy.merge_call()` as logic in
  outside compilation is replicated and executed separately for each
  replica. On the other hand, `merge_call()` requires a `merge_fn`
  to aggregate the inputs from different replicas and is executed only
  once.

  For variables placed in TPU device, which includes variables created inside
  TPUStrategy scope, outside compilation logic must not include variable
  read/write. For variables placed on host, variable read/write is only allowed
  if the variable is not accessed by any other ops in the TPU computation.
  Variable read/write from outside compilation cluster is not visible from TPU
  computation and vice versa. Therefore, if outside compilation logic contains
  such host variables read/write ops and if the variables are accessed by TPU
  computation as well, then this may lead to deadlock.

  Internally, `tf.tpu.outside_compilation()` adds outside compilation
  attributes to all ops in `computation`. During a later passes ops with outside
  compilation attributes are moved to a host-side graph. Inputs to this extract
  host-side graph are sent from TPU computation graph to host graph via a pair
  of XlaSendToHost and XlaRecvFromHost ops. Note that using
  `tf.tpu.outside_compilation()` may result in tensor transfer between TPU and
  CPU, leading to non-trivial performance impact.

  Args:
    computation: A Python function that builds the computation to place on the
      host.
    *args: the positional arguments for the computation.
    **kwargs: the keyword arguments for the computation.

  Returns:
    The Tensors returned by computation.
  """
  return outside_compilation_impl(False, computation, *args, **kwargs)


def experimental_map_outside_compilation(
    computation: Callable[..., Any], *args, **kwargs
) -> Any:
  """Maps `computation` onto shards and puts it outside any current TPU replicate scope.

  `experimental_map_outside_compilation(f, x)` maps `f` onto the shards
  of `x`, where `x` is split-sharded. Each invocation of `f` on a split occurs
  on the CPU that's associated with the TPU that owns the split.

  Example usage:

  ```python
  def normalize_each_split(split):
    return split - tf.math.reduce_mean(split)

  def tpu_computation(x):
    x_split = strategy.experimental_split_to_logical_devices(
                x, [num_cores_per_replica, 1])
    y = experimental_map_outside_compilation(
          normalize_each_split, x_split)
    y_split = strategy.experimental_split_to_logical_devices(
                x, [num_cores_per_replica, 1])
    return y_split
  ```

  `experimental_map_outside_compilation` should be called inside
  TPUReplicateContext. That is, `outside_compilation()` should be called
  inside a function that is passed to `tpu.split_compile_and_replicate()` --
  this is implied when outside compilation is invoked inside a function passed
  to TPUStrategy `run()`. It is invalid to invoke outside of
  TPUReplicateContext.

  `experimental_map_outside_compilation` should input and output tensors that
  are located on the TPU.

  Internally, `experimental_map_outside_compilation()` adds outside
  compilation attributes to all ops in `computation` and moves outside-compiled
  ops to a host-side graph. This is similar to `tf.tpu.outside_compilation()`.
  Send/recv ops from/to the TPU send each split directly to the TPU's host.

  Args:
    computation: A Python function that builds the computation to place on the
      host.
    *args: the positional arguments for the computation.
    **kwargs: the keyword arguments for the computation.

  Returns:
    The Tensors returned by computation.
  """
  return outside_compilation_impl(True, computation, *args, **kwargs)
