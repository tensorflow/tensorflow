# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Implements the graph generation for computation of gradients."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib

from six.moves import xrange, zip  # pylint: disable=redefined-builtin

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function as framework_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_state
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


def _MarkReachedOps(from_ops, reached_ops, func_graphs):
  """Mark all ops reached from "from_ops".

  Args:
    from_ops: list of Operations.
    reached_ops: set of Operations.
    func_graphs: list of FuncGraphs. This method will traverse through
      these functions if they capture from_ops or any reachable ops.
  """
  queue = collections.deque()
  queue.extend(from_ops)
  while queue:
    op = queue.popleft()
    if op not in reached_ops:
      reached_ops.add(op)
      for output in op.outputs:
        if _IsBackpropagatable(output):
          queue.extend(_Consumers(output, func_graphs))


def _PendingCount(to_ops, from_ops, colocate_gradients_with_ops, func_graphs,
                  xs_set):
  """Initialize the pending count for ops between two lists of Operations.

  'pending_count[op]' indicates the number of backprop inputs
  to this operation.

  Args:
    to_ops: list of Operations.
    from_ops: list of Operations.
    colocate_gradients_with_ops: Python bool.  See docstring of gradients().
    func_graphs: list of FuncGraphs. This method will traverse through
      these functions if they capture from_ops or any reachable ops. This is
      useful if to_ops occur in a function and from_ops are in an outer function
      or graph.
    xs_set: ObjectIdentitySet of Tensors.

  Returns:
    A tuple containing: (1) the subset of to_ops reachable from from_ops by a
    path of zero or more backpropagatable tensors, (2) a mapping from operation
    to the number of backprop inputs to that op, and (3) a ControlFlowState
    object which is not None if the ops between from_ops and to_ops contain
    control flow loops.
  """
  # Mark reachable ops from from_ops.
  reached_ops = set()
  _MarkReachedOps(from_ops, reached_ops, func_graphs)
  # X in reached_ops iff X is reachable from from_ops by a path of zero or more
  # backpropagatable tensors.

  reachable_to_ops = set(op for op in to_ops if op in reached_ops)

  # Mark between ops.
  between_ops = set()
  between_op_list = []
  queue = collections.deque()
  queue.extend(to_ops)
  while queue:
    op = queue.popleft()
    # We are interested in this op.
    if op in reached_ops:
      between_ops.add(op)
      between_op_list.append(op)
      # Clear the boolean so we won't add the inputs again.
      reached_ops.remove(op)
      for inp in _NonEagerInputs(op, xs_set):
        queue.append(inp.op)
  # X in between_ops iff X is on a path of zero or more backpropagatable tensors
  # between from_ops and to_ops

  # 'loop_state' is None if there are no while loops.
  loop_state = control_flow_state.MaybeCreateControlFlowState(
      between_op_list, between_ops, colocate_gradients_with_ops)

  # Initialize pending count for between ops.
  pending_count = collections.defaultdict(int)
  for op in between_op_list:
    for x in _NonEagerInputs(op, xs_set):
      if x.op in between_ops:
        pending_count[x.op] += 1

  return reachable_to_ops, pending_count, loop_state


def _AsList(x):
  return x if isinstance(x, (list, tuple)) else [x]


def _DefaultGradYs(grad_ys,
                   ys,
                   colocate_gradients_with_ops,
                   gradient_uid="__unsupported__"):
  """Fill in default values for grad_ys.

  Args:
    grad_ys: List of gradients, can contain None.
    ys: List of tensors.
    colocate_gradients_with_ops: If True, try colocating gradients with
      the corresponding op.
    gradient_uid: A unique identifier within the graph indicating
      which invocation of gradients is being executed. Used to cluster
      ops for compilation.

  Returns:
    A list of gradients to use, without None.

  Raises:
    ValueError: If sizes of gradients and inputs don't match
    TypeError: If type of any gradient is not valid for its input.
  """
  if len(grad_ys) != len(ys):
    raise ValueError("Passed %d grad_ys for %d ys" % (len(grad_ys), len(ys)))
  grad_ys = ops.convert_n_to_tensor_or_indexed_slices(grad_ys, name="grad_y")
  new_grad_ys = []
  for i, (y, grad_y) in enumerate(zip(ys, grad_ys)):
    with _maybe_colocate_with(y.op, gradient_uid, colocate_gradients_with_ops):
      if grad_y is None:
        if y.dtype.is_complex:
          raise TypeError(
              "Gradients of complex tensors must set grad_ys (y.dtype = %r)" %
              y.dtype)
        new_grad_ys.append(
            array_ops.fill(
                array_ops.shape(y),
                constant_op.constant(1, dtype=y.dtype, name="grad_ys_%d" % i)))
        continue
      if y.dtype.is_floating or y.dtype.is_integer:
        if not grad_y.dtype.is_floating and not grad_y.dtype.is_integer:
          raise TypeError(
              "Gradient type %s generated for real or "
              "integer-valued tensor %s with type %s must be "
              "real or integer" % (dtypes.as_dtype(grad_y.dtype).name, y,
                                   dtypes.as_dtype(y.dtype).name))
      elif y.dtype.is_complex:
        if not grad_y.dtype.is_complex:
          raise TypeError(
              "Gradient type %s generated for complex-valued "
              "tensor %s with type %s must be real" % (dtypes.as_dtype(
                  grad_y.dtype).name, y, dtypes.as_dtype(y.dtype).name))
      elif y.dtype == dtypes.variant:
        if grad_y.dtype != dtypes.variant:
          raise TypeError(
              "Gradient type %s generated for variant "
              "tensor %s with type %s must be variant" % (dtypes.as_dtype(
                  grad_y.dtype).name, y, dtypes.as_dtype(y.dtype).name))
      elif y.dtype == dtypes.resource:
        # We assume y is the handle of a ResourceVariable. The gradient of a
        # ResourceVariable should be a numeric value, not another resource.
        if grad_y.dtype == dtypes.resource:
          raise TypeError("Input gradient %s for resource tensor %s should not "
                          "be a resource" % (grad_y, y))
      else:
        raise TypeError(
            "Tensor %s with type %s must be numeric "
            "to obtain a default gradient" % (y, dtypes.as_dtype(y.dtype).name))
      # Create a grad_y tensor in the name scope of the gradient.
      # Required for TensorArrays to identify which gradient call a
      # grad_y value is coming from.
      if isinstance(grad_y, ops.IndexedSlices):
        new_grad_ys.append(
            ops.IndexedSlices(
                indices=(array_ops.identity(
                    grad_y.indices, name="grad_ys_%d_indices" % i)
                         if isinstance(grad_y.indices, ops.Tensor) else
                         grad_y.indices),
                values=(array_ops.identity(
                    grad_y.values, name="grad_ys_%d_values" % i) if isinstance(
                        grad_y.values, ops.Tensor) else grad_y.values),
                dense_shape=(array_ops.identity(
                    grad_y.dense_shape, name="grad_ys_%d_shape" % i)
                             if isinstance(grad_y.dense_shape, ops.Tensor) else
                             grad_y.dense_shape)))
      else:
        new_grad_ys.append(array_ops.identity(grad_y, name="grad_ys_%d" % i))

  return new_grad_ys


def _IsBackpropagatable(tensor):
  if backprop_util.IsTrainable(tensor):
    return True
  dtype = dtypes.as_dtype(tensor.dtype)
  return dtype.base_dtype == dtypes.bfloat16


def _VerifyGeneratedGradients(grads, op):
  """Verify that gradients are valid in number and type.

  Args:
    grads: List of generated gradients.
    op: Operation for which the gradients where generated.

  Raises:
    ValueError: if sizes of gradients and inputs don't match.
    TypeError: if type of any gradient is not valid for its input.
  """
  # While ops have inputs added to them during the gradient computation, so we
  # skip the below check. See while_v2 for details.
  if op.type == "While" or op.type == "StatelessWhile":
    return

  if len(grads) != len(op.inputs):
    raise ValueError("Num gradients %d generated for op %s do not match num "
                     "inputs %d" % (len(grads), op.node_def, len(op.inputs)))


def _StopOps(from_ops, stop_gradient_ops, pending_count, xs_set):
  """The set of ops that terminate the gradient computation.

  This computes the frontier of the forward graph *before* which backprop
  should stop. Operations in the returned set will not be differentiated.
  This set is defined as the subset of `from_ops` containing ops that have
  no predecessor in `from_ops`. `pending_count` is the result of
  `_PendingCount(xs, from_ops)`. An 'op' has predecessors in `from_ops`
  iff pending_count[op] > 0.

  In addition, none of `stop_gradient_ops` will be differentiated.

  Args:
    from_ops: list of Operations.
    stop_gradient_ops: list of Operations never to backprop through.
    pending_count: mapping from operation to number of backprop inputs.
    xs_set: ObjectIdentitySet of Tensors.

  Returns:
    The set of operations.
  """
  stop_ops = set()
  for op in from_ops:
    is_stop_op = True
    for inp in _NonEagerInputs(op, xs_set):
      if pending_count[inp.op] > 0:
        is_stop_op = False
        break
    if is_stop_op:
      stop_ops.add(op)
  stop_ops.update(op for op in stop_gradient_ops)
  return stop_ops


@contextlib.contextmanager
def _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops):  # pylint: disable=invalid-name
  """Context to colocate with `op` if `colocate_gradients_with_ops`."""
  if colocate_gradients_with_ops:
    with ops._colocate_with_for_gradient(op, gradient_uid):  # pylint: disable=protected-access
      yield
  else:
    yield


def _IsPartitionedCall(op):
  return op.type == "PartitionedCall" or op.type == "StatefulPartitionedCall"


def _SymGrad(op, out_grads):
  """Backprop through a function call node op given its outputs' gradients."""
  f_in = [x for x in op.inputs] + out_grads
  f_types = [default_gradient.get_zeros_dtype(x) for x in op.inputs]
  f = attr_value_pb2.NameAttrList()
  if _IsPartitionedCall(op):
    f.name = op.get_attr("f").name
  else:
    f.name = op.type
  for k in op.node_def.attr:
    f.attr[k].CopyFrom(op.node_def.attr[k])
  in_grads = functional_ops.symbolic_gradient(input=f_in, Tout=f_types, f=f)
  return in_grads


def _MaybeCompile(scope, op, func, grad_fn):
  """Compile the calculation in grad_fn if op was marked as compiled."""
  scope = scope.rstrip("/").replace("/", "_")
  if func is not None:
    xla_compile = func.definition.attr["_XlaCompile"].b
    xla_separate_compiled_gradients = func.definition.attr[
        "_XlaSeparateCompiledGradients"].b
    xla_scope = func.definition.attr["_XlaScope"].s.decode()
  else:
    try:
      xla_compile = op.get_attr("_XlaCompile")
      xla_separate_compiled_gradients = op.get_attr(
          "_XlaSeparateCompiledGradients")
      xla_scope = op.get_attr("_XlaScope").decode()
    except ValueError:
      xla_compile = False

  if not xla_compile:
    return grad_fn()  # Exit early

  # If the gradients are supposed to be compiled separately, we give them a
  # _XlaScope name that is based on the name_scope of the gradients.  Otherwise
  # they just inherit the existing _XlaScope name, which lets them be merged
  # together with the non-gradient computation.
  if xla_separate_compiled_gradients:
    xla_grad_scope = "%s_grad_%s" % (xla_scope, scope)
  else:
    xla_grad_scope = xla_scope

  attrs = {
      "_XlaCompile": attr_value_pb2.AttrValue(b=xla_compile),
      "_XlaScope": attr_value_pb2.AttrValue(s=xla_grad_scope.encode())
  }
  with ops.get_default_graph()._attr_scope(attrs):  # pylint: disable=protected-access
    return grad_fn()


def _RaiseNoGradWrtInitialLoopValError(op, from_ops, xs_set):
  """Raises an error if we backprop through a loop var."""
  # Find the nearest 'to_op' reachable from 'op' to provide a more helpful error
  # message.
  target_op = None
  queue = collections.deque([op])
  visited = set()
  while queue:
    curr_op = queue.popleft()
    if curr_op in visited: continue
    visited.add(curr_op)
    if curr_op in from_ops:
      target_op = curr_op
      break
    queue.extend(t.op for t in _NonEagerInputs(curr_op, xs_set))
  assert target_op
  raise ValueError(
      "Cannot compute gradient inside while loop with respect to op '%s'. "
      "We do not support taking the gradient wrt or through the initial value "
      "of a loop variable. Gradients can be computed through loop invariants "
      "or wrt the input parameters to the loop body."
      % target_op.name)


def _IsFunction(graph):
  return (isinstance(graph, FuncGraph) or
          isinstance(graph, framework_function._FuncGraph))  # pylint: disable=protected-access


def _Captures(func_graph):
  if isinstance(func_graph, FuncGraph):
    return func_graph.captures
  else:
    assert isinstance(func_graph, framework_function._FuncGraph)  # pylint: disable=protected-access
    return func_graph.captures


def _MaybeCaptured(t):
  """If t is a captured value placeholder, returns the original captured value.

  Args:
    t: Tensor

  Returns:
    A tensor, potentially from a different Graph/FuncGraph.
  """
  # pylint: disable=protected-access
  if (not isinstance(t, ops.EagerTensor) and
      _IsFunction(t.op.graph) and t.op.type == "Placeholder"):
    for input_t, placeholder_t in _Captures(t.op.graph):
      if t is placeholder_t:
        return _MaybeCaptured(input_t)
  # pylint: enable=protected-access
  return t


def _NonEagerInputs(op, xs_set):
  """Returns the inputs of op, crossing closure boundaries where necessary.

  Does not return any captured EagerTensors, i.e., the number of tensors
  returned may be less than than the actual number of inputs.

  Args:
    op: Operation
    xs_set: ObjectIdentitySet of Tensors we are differentiating w.r.t.

  Returns:
    A list of tensors. The tensors may be from multiple Graph/FuncGraphs if op
    is in a FuncGraph and has captured inputs.
  """
  return [t for t in _Inputs(op, xs_set) if not isinstance(t, ops.EagerTensor)]


# TODO(skyewm): plumbing xs through everywhere is ugly, consider making
# _GradientsHelper a class with xs as a member variable.
def _Inputs(op, xs_set):
  """Returns the inputs of op, crossing closure boundaries where necessary.

  Args:
    op: Operation
    xs_set: ObjectIdentitySet of Tensors we are differentiating w.r.t.

  Returns:
    A list of tensors. The tensors may be from multiple Graph/FuncGraphs if op
    is in a FuncGraph and has captured inputs.
  """
  if _IsFunction(op.graph):  # pylint: disable=protected-access
    inputs = []
    for t in op.inputs:
      # If we're differentiating w.r.t. `t`, do not attempt to traverse through
      # it to a captured value. The algorithm needs to "see" `t` in this case,
      # even if it's a function input for a captured value, whereas usually we'd
      # like to traverse through these closures as if the captured value was the
      # direct input to op.
      if t not in xs_set:
        t = _MaybeCaptured(t)
      inputs.append(t)
    return inputs
  else:
    return op.inputs


def _Consumers(t, func_graphs):
  """Returns the consumers of t, crossing closure boundaries where necessary.

  Args:
    t: Tensor
    func_graphs: a list of FuncGraphs that may have captured t.

  Returns:
    A list of tensors. The tensors will be from the current graph and/or
    func_graphs.
  """
  consumers = t.consumers()
  for func in func_graphs:
    for input_t, placeholder in _Captures(func):
      if input_t is t:
        consumers.extend(_Consumers(placeholder, func_graphs))
  return consumers


def _GradientsHelper(ys,
                     xs,
                     grad_ys=None,
                     name="gradients",
                     colocate_gradients_with_ops=False,
                     gate_gradients=False,
                     aggregation_method=None,
                     stop_gradients=None,
                     unconnected_gradients=UnconnectedGradients.NONE,
                     src_graph=None):
  """Implementation of gradients()."""
  if context.executing_eagerly():
    raise RuntimeError("tf.gradients is not supported when eager execution "
                       "is enabled. Use tf.GradientTape instead.")
  if src_graph is None:
    src_graph = ops.get_default_graph()
  try:
    unconnected_gradients = UnconnectedGradients(unconnected_gradients)
  except ValueError:
    raise ValueError(
        "Unknown value for unconnected_gradients: %r" % unconnected_gradients)

  # If src_graph is a _FuncGraph (i.e. a function body), gather it and all
  # ancestor graphs. This is necessary for correctly handling captured values.
  func_graphs = []
  curr_graph = src_graph
  while _IsFunction(curr_graph):
    func_graphs.append(curr_graph)
    if isinstance(curr_graph, FuncGraph):
      curr_graph = curr_graph.outer_graph
    else:
      assert isinstance(curr_graph, framework_function._FuncGraph)  # pylint: disable=protected-access
      curr_graph = curr_graph._outer_graph  # pylint: disable=protected-access

  ys = _AsList(ys)
  xs = _AsList(xs)
  stop_gradients = [] if stop_gradients is None else _AsList(stop_gradients)
  if grad_ys is None:
    grad_ys = [None] * len(ys)
  else:
    grad_ys = _AsList(grad_ys)

  with ops.name_scope(
      name, "gradients",
      list(ys) + list(xs) + list(stop_gradients) + list(grad_ys)) as grad_scope:
    # Get a uid for this call to gradients that can be used to help
    # cluster ops for compilation.
    gradient_uid = ops.get_default_graph().unique_name("uid")
    ys = ops.convert_n_to_tensor_or_indexed_slices(ys, name="y")
    xs = [
        x.handle if resource_variable_ops.is_resource_variable(x) else x
        for x in xs
    ]
    xs = ops.internal_convert_n_to_tensor_or_indexed_slices(
        xs, name="x", as_ref=True)
    xs_set = object_identity.ObjectIdentitySet(xs)
    grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops,
                             gradient_uid)

    # The approach we take here is as follows: Create a list of all ops in the
    # subgraph between the ys and xs.  Visit these ops in reverse order of ids
    # to ensure that when we visit an op the gradients w.r.t its outputs have
    # been collected.  Then aggregate these gradients if needed, call the op's
    # gradient function, and add the generated gradients to the gradients for
    # its input.

    # Initialize the pending count for ops in the connected subgraph from ys
    # to the xs.
    to_ops = [t.op for t in ys]
    from_ops = [t.op for t in xs]
    stop_gradient_ops = [t.op for t in stop_gradients]
    reachable_to_ops, pending_count, loop_state = _PendingCount(
        to_ops, from_ops, colocate_gradients_with_ops, func_graphs, xs_set)

    # Iterate over the collected ops.
    #
    # grads: op => list of gradients received on each output endpoint of the
    # op.  The gradients for each endpoint are initially collected as a list.
    # When it is time to call the op's gradient function, for each endpoint we
    # aggregate the list of received gradients into a Add() Operation if there
    # is more than one.
    grads = {}

    # Add the initial gradients for the ys.
    for y, grad_y in zip(ys, grad_ys):
      _SetGrad(grads, y, grad_y)

    # Initialize queue with to_ops.
    queue = collections.deque()
    # Add the ops in 'to_ops' into the queue.
    to_ops_set = set()
    for op in to_ops:
      # 'ready' handles the case where one output gradient relies on
      # another output's gradient.
      ready = (pending_count[op] == 0)
      if ready and op not in to_ops_set and op in reachable_to_ops:
        to_ops_set.add(op)
        queue.append(op)

    if loop_state:
      loop_exits = loop_state.ProcessUnusedLoopExits(pending_count, to_ops_set)
      for y in loop_exits:
        if backprop_util.IsTrainable(y):
          _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
          queue.append(y.op)

    stop_ops = _StopOps(from_ops, stop_gradient_ops, pending_count, xs_set)
    while queue:
      # generate gradient subgraph for op.
      op = queue.popleft()
      with _maybe_colocate_with(op, gradient_uid, colocate_gradients_with_ops):
        if loop_state:
          loop_state.EnterGradWhileContext(op, before=True)
        out_grads = _AggregatedGrads(grads, op, gradient_uid, loop_state,
                                     aggregation_method)
        if loop_state:
          loop_state.ExitGradWhileContext(op, before=True)

        grad_fn = None
        func_call = None
        is_partitioned_call = _IsPartitionedCall(op)
        # pylint: disable=protected-access
        is_func_call = (
            src_graph._is_function(op.type) or is_partitioned_call)
        # pylint: enable=protected-access
        has_out_grads = any(isinstance(g, ops.Tensor) or g for g in out_grads)
        if has_out_grads and (op not in stop_ops):
          try:
            grad_fn = ops.get_gradient_function(op)
          except LookupError:
            if is_func_call:
              if is_partitioned_call:
                func_name = compat.as_bytes(op.get_attr("f").name)
                func_call = src_graph._get_function(  # pylint: disable=protected-access
                    func_name)
                # When a graph is imported, the FunctionDefs are not copied over
                # to each sub-graph so we recursively search the outer graphs
                # for the FunctionDef.
                if not func_call and hasattr(src_graph, "outer_graph"):
                  graph = src_graph.outer_graph
                  while graph is not None:
                    func_call = graph._get_function(func_name)  # pylint: disable=protected-access
                    if func_call  is not None:
                      break
                    if hasattr(graph, "outer_graph"):
                      graph = graph.outer_graph
                    else:
                      break
              else:
                func_call = src_graph._get_function(op.type)  # pylint: disable=protected-access
              # Note that __defun is not set if the graph is
              # imported. If it's set, we prefer to access the original
              # defun.
              func_call = getattr(op, "__defun", func_call)
              grad_fn = func_call.python_grad_func
            else:
              raise LookupError(
                  "No gradient defined for operation '%s' (op type: %s)" %
                  (op.name, op.type))
        if loop_state:
          loop_state.EnterGradWhileContext(op, before=False)

        # NOTE(skyewm): We don't support computing gradients wrt a loop variable
        # unless it's within the context of a single iteration (i.e. the
        # gradient is wrt to the loop parameter in the body function, not wrt or
        # through the initial value). This means if we're in a while loop
        # context, we should never see a switch node from this context.
        # pylint: disable=protected-access
        if (control_flow_util.IsSwitch(op) and
            op._control_flow_context is not None and
            op._control_flow_context.IsWhileContext() and
            op._control_flow_context ==
            ops.get_default_graph()._get_control_flow_context()):
          _RaiseNoGradWrtInitialLoopValError(op, from_ops, xs_set)
        # pylint: enable=protected-access

        if (grad_fn or is_func_call) and has_out_grads:
          # NOTE: If _AggregatedGrads didn't compute a value for the i'th
          # output, it means that the cost does not depend on output[i],
          # therefore dC/doutput[i] is 0.
          for i, out_grad in enumerate(out_grads):
            if (not isinstance(out_grad, ops.Tensor) and not out_grad) and (
                (not grad_fn and is_func_call)
                or backprop_util.IsTrainable(op.outputs[i])):
              # Only trainable outputs or outputs for a function call that
              # will use SymbolicGradient get a zero gradient. Gradient
              # functions should ignore the gradient for other outputs.
              # TODO(apassos) gradients of resource handles might be an
              # issue here because of zeros.
              if loop_state:
                out_grads[i] = loop_state.ZerosLikeV1WhileLoop(op, i)
              elif default_gradient.supports_default_grad(op.outputs[i]):
                # TODO(b/143286622): The supports_default_grad check is needed
                # because While op emits non-differentiable resource tensors
                # as outputs. Remove this check when that is not the case.
                out_grads[i] = control_flow_state.ZerosLike(op, i)
          with ops.name_scope(op.name + "_grad"):
            # pylint: disable=protected-access
            with src_graph._original_op(op):
              # pylint: enable=protected-access
              if grad_fn:
                # If grad_fn was found, do not use SymbolicGradient even for
                # functions.
                in_grads = _MaybeCompile(grad_scope, op, func_call,
                                         lambda: grad_fn(op, *out_grads))
              else:
                # For function call ops, we add a 'SymbolicGradient'
                # node to the graph to compute gradients.
                in_grads = _MaybeCompile(grad_scope, op, func_call,
                                         lambda: _SymGrad(op, out_grads))
              in_grads = _AsList(in_grads)
              _VerifyGeneratedGradients(in_grads, op)
              if gate_gradients and len([x for x in in_grads
                                         if x is not None]) > 1:
                with ops.device(None):
                  with ops._colocate_with_for_gradient(  # pylint: disable=protected-access
                      None,
                      gradient_uid,
                      ignore_existing=True):
                    in_grads = control_flow_ops.tuple(in_grads)
          _LogOpGradients(op, out_grads, in_grads)
        else:
          # If no grad_fn is defined or none of out_grads is available,
          # just propagate a list of None backwards.
          in_grads = [None] * len(_Inputs(op, xs_set))
        # Note: we don't filter out eager inputs here because the inputs need to
        # line up with in_grads.
        for i, (t_in, in_grad) in enumerate(zip(_Inputs(op, xs_set), in_grads)):
          if in_grad is not None:
            if (isinstance(in_grad, ops.Tensor) and
                t_in.dtype != dtypes.resource):
              try:
                in_grad.set_shape(t_in.get_shape())
              except ValueError:
                raise ValueError(
                    "Incompatible shapes between op input and calculated "
                    "input gradient.  Forward operation: %s.  Input index: %d. "
                    "Original input shape: %s.  "
                    "Calculated input gradient shape: %s" %
                    (op.name, i, t_in.shape, in_grad.shape))
            if not isinstance(t_in, ops.EagerTensor):
              _SetGrad(grads, t_in, in_grad)
        if loop_state:
          loop_state.ExitGradWhileContext(op, before=False)

      # Update pending count for the inputs of op and enqueue ready ops.
      _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state,
                                    xs_set)

  if loop_state:
    loop_state.PostProcessing()
  return [_GetGrad(grads, x, unconnected_gradients) for x in xs]


def _HasAnyNotNoneGrads(grads, op):
  """Return true iff op has real gradient."""
  out_grads = _GetGrads(grads, op)
  for out_grad in out_grads:
    if isinstance(out_grad, (ops.Tensor, ops.IndexedSlices)):
      return True
    if out_grad and isinstance(out_grad, collections_abc.Sequence):
      if any(g is not None for g in out_grad):
        return True
  return False


def _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state,
                                  xs_set):
  """Update pending count for the inputs of op and enqueue ready ops."""
  for x in _NonEagerInputs(op, xs_set):
    pending_count[x.op] -= 1
    ready = (pending_count[x.op] == 0)
    if loop_state and not ready:
      ready = pending_count[x.op] > 0 and control_flow_util.IsLoopSwitch(x.op)
    if ready:
      if control_flow_util.IsLoopExit(x.op):
        # if x is an exit without real gradient, defer processing them.
        grad_state = loop_state.GetGradState(x.op, before=False)
        grad_state.deferred_exits.append(x)
        grad_state.pending_exits_count -= 1
        if grad_state.pending_exits_count == 0:
          # We now have all the exits so process them.
          has_not_none_grad = False
          for y in grad_state.deferred_exits:
            if _HasAnyNotNoneGrads(grads, y.op):
              has_not_none_grad = True
              queue.append(y.op)
            else:
              grad_state.unused_exits.append(y)
          if has_not_none_grad:
            # For an unused exit, if it has trainable outputs, backprop
            # a zero gradient. Otherwise, just ignore it.
            for y in grad_state.unused_exits:
              if backprop_util.IsTrainable(y):
                _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
              queue.append(y.op)
          else:
            # All exits are "unused" so use None as gradient.
            for y in grad_state.unused_exits:
              queue.append(y.op)
      else:
        queue.append(x.op)


def _SetGrad(grads, t, grad):
  """Sets gradient "grad" in "grads" for tensor "t"."""
  op = t.op
  op_grads = grads.get(op)
  if not op_grads:
    op_grads = [[] for _ in xrange(len(op.outputs))]
    grads[op] = op_grads
  t_grads = op_grads[t.value_index]
  if isinstance(t_grads, list):
    t_grads.append(grad)
  else:
    assert control_flow_util.IsLoopSwitch(op)
    op_grads[t.value_index] = grad


def _ZerosLike(t):
  t_dtype = default_gradient.get_zeros_dtype(t)
  if t.dtype == dtypes.resource:
    return array_ops.zeros(
        resource_variable_ops.variable_shape(t), dtype=t_dtype)
  else:
    return array_ops.zeros_like(t, dtype=t_dtype)


def _GetGrad(grads, t, unconnected_gradients):
  """Gets gradient for tensor "t"."""
  op = t.op
  op_grads = grads.get(op)
  if not op_grads:
    if unconnected_gradients == UnconnectedGradients.ZERO:
      return _ZerosLike(t)
    elif unconnected_gradients == UnconnectedGradients.NONE:
      return None
    else:
      raise ValueError(
          "Unknown value for unconnected_gradients: %r" % unconnected_gradients)

  t_grad = op_grads[t.value_index]
  # This can happen if some other output of `t.op` has non-None grad.
  if unconnected_gradients == UnconnectedGradients.ZERO and t_grad is None:
    return _ZerosLike(t)

  assert not isinstance(
      t_grad, list), ("gradients list should have been aggregated by now.")
  return t_grad


def _GetGrads(grads, op):
  """Gets all gradients for op."""
  if op in grads:
    return grads[op]
  else:
    return [[] for _ in xrange(len(op.outputs))]


def _AccumulatorShape(inputs):
  shape = tensor_shape.unknown_shape()
  for i in inputs:
    if isinstance(i, ops.Tensor):
      shape = shape.merge_with(i.get_shape())
  return shape


def _LogOpGradients(op, out_grads, in_grads):
  """Log the in and out grads of an op."""
  logging.vlog(1, "Gradient for '" + op.name + "'")

  def _FilterGrad(x):
    if x is None:
      return False
    if isinstance(x, (list, tuple)):
      return bool(x)
    else:
      return True

  logging.vlog(1, "  in  --> %s",
               ", ".join(x.name for x in out_grads if _FilterGrad(x)))
  logging.vlog(1, "  out --> %s",
               ", ".join(x.name for x in in_grads if _FilterGrad(x)))


def _MultiDeviceAddN(tensor_list, gradient_uid):
  """Adds tensors from potentially multiple devices."""
  # Basic function structure comes from control_flow_ops.group().
  # Sort tensors according to their devices.
  tensors_on_device = collections.defaultdict(lambda: [])
  for tensor in tensor_list:
    tensors_on_device[tensor.device].append(tensor)

  # For each device, add the tensors on that device first.
  # Then gather the partial sums from multiple devices.
  # TODO(sjhwang): Create hierarchical aggregation tree as pbar's suggestion.
  # E.g., aggregate per GPU, then per task, and so on.
  summands = []

  def DeviceKey(dev):
    return "" if dev is None else dev

  for dev in sorted(tensors_on_device, key=DeviceKey):
    tensors = tensors_on_device[dev]
    with ops._colocate_with_for_gradient(  # pylint: disable=protected-access
        tensors[0].op,
        gradient_uid,
        ignore_existing=True):
      summands.append(math_ops.add_n(tensors))

  return math_ops.add_n(summands)


@tf_export("AggregationMethod")
class AggregationMethod(object):
  """A class listing aggregation methods used to combine gradients.

  Computing partial derivatives can require aggregating gradient
  contributions. This class lists the various methods that can
  be used to combine gradients in the graph.

  The following aggregation methods are part of the stable API for
  aggregating gradients:

  *  `ADD_N`: All of the gradient terms are summed as part of one
     operation using the "AddN" op (see `tf.add_n`). This
     method has the property that all gradients must be ready and
     buffered separately in memory before any aggregation is performed.
  *  `DEFAULT`: The system-chosen default aggregation method.

  The following aggregation methods are experimental and may not
  be supported in future releases:

  * `EXPERIMENTAL_TREE`: Gradient terms are summed in pairs using
    using the "AddN" op. This method of summing gradients may reduce
    performance, but it can improve memory utilization because the
    gradients can be released earlier.

  """
  ADD_N = 0
  DEFAULT = ADD_N
  # The following are experimental and may not be supported in future releases.
  EXPERIMENTAL_TREE = 1
  EXPERIMENTAL_ACCUMULATE_N = 2  # An alias for EXPERIMENTAL_ADD_N = 1


def _AggregatedGrads(grads,
                     op,
                     gradient_uid,
                     loop_state,
                     aggregation_method=None):
  """Get the aggregated gradients for op.

  Args:
    grads: The map of memoized gradients.
    op: The op to get gradients for.
    gradient_uid: A unique identifier within the graph indicating
      which invocation of gradients is being executed. Used to cluster
      ops for compilation.
    loop_state: An object for maintaining the state of the while loops in the
                graph. It is of type ControlFlowState. None if the graph
                contains no while loops.
    aggregation_method: Specifies the method used to combine gradient terms.
      Accepted values are constants defined in the class `AggregationMethod`.

  Returns:
    A list of gradients, one per each output of `op`. If the gradients
      for a particular output is a list, this function aggregates it
      before returning.

  Raises:
    TypeError: if the incoming grads are not Tensors or IndexedSlices.
    ValueError: if the arguments are invalid.

  """
  if aggregation_method is None:
    aggregation_method = AggregationMethod.DEFAULT
  if aggregation_method not in [
      AggregationMethod.ADD_N, AggregationMethod.EXPERIMENTAL_TREE,
      AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
  ]:
    raise ValueError(
        "Invalid aggregation_method specified %s." % aggregation_method)
  out_grads = _GetGrads(grads, op)
  for i, out_grad in enumerate(out_grads):
    if loop_state:
      if isinstance(out_grad, (ops.Tensor, ops.IndexedSlices)):
        assert control_flow_util.IsLoopSwitch(op)
        continue
    # Grads have to be Tensors or IndexedSlices
    if (isinstance(out_grad, collections_abc.Sequence) and not all(
        isinstance(g, (ops.Tensor, ops.IndexedSlices))
        for g in out_grad
        if g is not None)):
      raise TypeError("gradients have to be either all Tensors "
                      "or all IndexedSlices")
    # Aggregate multiple gradients, and convert [] to None.
    if out_grad:
      if len(out_grad) < 2:
        used = "nop"
        out_grads[i] = out_grad[0]
      elif all(isinstance(g, ops.Tensor) for g in out_grad if g is not None):
        tensor_shape = _AccumulatorShape(out_grad)
        if aggregation_method in [
            AggregationMethod.EXPERIMENTAL_TREE,
            AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
        ]:
          # Aggregate all gradients by doing pairwise sums: this may
          # reduce performance, but it can improve memory because the
          # gradients can be released earlier.
          #
          # TODO(vrv): Consider replacing this with a version of
          # tf.AddN() that eagerly frees its inputs as soon as they are
          # ready, so the order of this tree does not become a problem.
          used = "tree"
          with ops.name_scope(op.name + "_gradient_sum"):
            running_sum = out_grad[0]
            for grad in out_grad[1:]:
              running_sum = math_ops.add_n([running_sum, grad])
            out_grads[i] = running_sum
        else:
          used = "add_n"
          out_grads[i] = _MultiDeviceAddN(out_grad, gradient_uid)
        logging.vlog(2, "  _AggregatedGrads %d x %s using %s", len(out_grad),
                     tensor_shape, used)
      else:
        out_grads[i] = backprop.aggregate_indexed_slices_gradients(out_grad)  # pylint: disable=protected-access
    else:  # not out_grad
      # out_grads[i] is [], thus its aggregation is simply None.
      out_grads[i] = None
  return out_grads


# Represents the output of TFE_Py_TapeSetPossibleGradientTypes. Real enums are
# unfortunately too slow to use here.
POSSIBLE_GRADIENT_TYPES_NONE = 0
POSSIBLE_GRADIENT_TYPES_FIRST_ORDER = 1
POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER = 2


def PossibleTapeGradientTypes(tensors):
  """Determines whether and how `args` may require tape gradients."""
  return pywrap_tfe.TFE_Py_TapeSetPossibleGradientTypes(tensors)
