# Copyright 2015 Google Inc. All Rights Reserved.
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
import warnings

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import image_grad  # pylint: disable=unused-import
from tensorflow.python.ops import logging_ops  # pylint: disable=unused-import
from tensorflow.python.ops import linalg_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import functional_ops

from tensorflow.python.platform import logging

# Warn the user if we convert a sparse representation to dense with at
# least this number of elements.
_LARGE_SPARSE_NUM_ELEMENTS = 100000000


def _IndexedSlicesToTensor(value, dtype=None, name=None, as_ref=False):
  """Converts an IndexedSlices object `value` to a Tensor.

  NOTE(mrry): This function is potentially expensive.

  Args:
    value: An ops.IndexedSlices object.
    dtype: The dtype of the Tensor to be returned.
    name: Optional name to use for the returned Tensor.
    as_ref: True if a ref is requested.

  Returns:
    A dense Tensor representing the values in the given IndexedSlices.

  Raises:
    ValueError: If the IndexedSlices does not have the same dtype.
  """
  _ = as_ref
  if dtype and not dtype.is_compatible_with(value.dtype):
    raise ValueError(
        "Tensor conversion requested dtype %s for IndexedSlices with dtype %s" %
        (dtype.name, value.dtype.name))
  if value.dense_shape is None:
    raise ValueError(
        "Tensor conversion requested for IndexedSlices without dense_shape: %s"
        % str(value))
  # TODO(mrry): Consider adding static shape information to
  # IndexedSlices, to avoid using numpy here.
  dense_shape_value = tensor_util.constant_value(value.dense_shape)
  if dense_shape_value is not None:
    num_elements = np.prod(dense_shape_value)
    if num_elements >= _LARGE_SPARSE_NUM_ELEMENTS:
      warnings.warn(
          "Converting sparse IndexedSlices to a dense Tensor with %d elements. "
          "This may consume a large amount of memory." % num_elements)
  else:
    warnings.warn(
        "Converting sparse IndexedSlices to a dense Tensor of unknown shape. "
        "This may consume a large amount of memory.")
  return math_ops.unsorted_segment_sum(value.values,
                                       value.indices,
                                       value.dense_shape[0],
                                       name=name)


ops.register_tensor_conversion_function(ops.IndexedSlices,
                                        _IndexedSlicesToTensor)


def _MarkReachedOps(from_ops, reached_ops):
  """Mark all ops reached from "from_ops".

  Args:
    from_ops: list of Operations.
    reached_ops: list of booleans, indexed by operation id.
  """
  queue = collections.deque()
  queue.extend(from_ops)
  while queue:
    op = queue.popleft()
    if not reached_ops[op._id]:
      reached_ops[op._id] = True
      for output in op.outputs:
        queue.extend(output.consumers())


def _GatherInputs(to_ops, reached_ops):
  """List all inputs of to_ops that are in reached_ops.

  Args:
    to_ops: list of Operations.
    reached_ops: list of booleans, indexed by operation id.

  Returns:
    The list of all inputs of to_ops that are in reached_ops.
    That list includes all elements of to_ops.
  """
  inputs = []
  queue = collections.deque()
  queue.extend(to_ops)
  while queue:
    op = queue.popleft()
    # We are interested in this op.
    if reached_ops[op._id]:
      inputs.append(op)
      # Clear the boolean so we won't add the inputs again.
      reached_ops[op._id] = False
      for inp in op.inputs:
        queue.append(inp.op)
  return inputs


def _GetGradsDevice(op, colocate_gradients_with_ops):
  """Gets the device to which to assign gradients of "op".

  Args:
    op: an Operation.
    colocate_gradients_with_ops: If True, try colocating gradients with the
      corresponding op.

  Returns:
    A device string.
  """
  if colocate_gradients_with_ops and op.device:
    return op.device
  else:
    return ""


def _PendingCount(graph, to_ops, from_ops):
  """Initialize the pending count for ops between two lists of Operations.

  'pending_count[op._id]' indicates the number of backprop inputs
  to this operation.

  Args:
    graph: a Graph.
    to_ops: list of Operations.
    from_ops: list of Operations.

  Returns:
    A tuple containing: (1) a list of integers indexed by operation id,
    indicating the number of backprop inputs to this operation, and (2)
    a boolean which is True if any of the ops in between from_ops and to_ops
    contain control flow loops.
  """
  # Mark reachable ops from from_ops.
  reached_ops = [False] * (graph._last_id + 1)
  for op in to_ops:
    reached_ops[op._id] = True
  _MarkReachedOps(from_ops, reached_ops)

  # Mark between ops.
  between_ops = [False] * (graph._last_id + 1)
  between_op_list = []
  queue = collections.deque()
  queue.extend(to_ops)
  while queue:
    op = queue.popleft()
    # We are interested in this op.
    if reached_ops[op._id]:
      between_ops[op._id] = True
      between_op_list.append(op)
      # Clear the boolean so we won't add the inputs again.
      reached_ops[op._id] = False
      for inp in op.inputs:
        queue.append(inp.op)

  # 'loop_state' is None if there are no while loops.
  loop_state = control_flow_ops.MaybeCreateControlFlowState(between_op_list,
                                                            between_ops)

  # Initialize pending count for between ops.
  pending_count = [0] * (graph._last_id + 1)
  for op in between_op_list:
    for x in op.inputs:
      if between_ops[x.op._id]:
        pending_count[x.op._id] += 1
    for x in op.control_inputs:
      if between_ops[x._id]:
        pending_count[x._id] += 1

  return pending_count, loop_state


def _AsList(x):
  return x if isinstance(x, (list, tuple)) else [x]


def _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops):
  """Fill in default values for grad_ys.

  Args:
    grad_ys: List of gradients, can contain None.
    ys: List of tensors.
    colocate_gradients_with_ops: If True, try colocating gradients with
      the corresponding op.

  Returns:
    A list of gradients to use, without None.

  Raises:
    ValueError: If one of the grad_ys is invalid.
  """
  if len(grad_ys) != len(ys):
    raise ValueError("Passed %d grad_ys for %d ys" % (len(grad_ys), len(ys)))
  grad_ys = ops.convert_n_to_tensor_or_indexed_slices(grad_ys, name="grad_y")
  for i in xrange(len(grad_ys)):
    grad_y = grad_ys[i]
    y = ys[i]
    if grad_y is None:
      with ops.device(_GetGradsDevice(y.op, colocate_gradients_with_ops)):
        grad_ys[i] = array_ops.fill(
            array_ops.shape(y),
            constant_op.constant(1, dtype=y.dtype))
    else:
      if grad_y.dtype != y.dtype:
        raise ValueError("Y and ys_grad must be of the same type, "
                         "not y: %s, ys_grad: %s " %
                         (dtypes.as_dtype(y.dtype).name,
                          dtypes.as_dtype(grad_y.dtype).name))
  return grad_ys


def _IsFloat(tensor):
  dtype = dtypes.as_dtype(tensor.dtype)
  return dtype.base_dtype in (dtypes.float32, dtypes.float64)


def _VerifyGeneratedGradients(grads, op):
  """Verify that gradients are valid in number and type.

  Args:
    grads: List of generated gradients.
    op: Operation for which the gradients where generated.

  Raises:
    ValueError: if the gradients are invalid.
  """
  if len(grads) != len(op.inputs):
    raise ValueError("Num gradients %d generated for op %s do not match num "
                     "inputs %d" % (len(grads), op.node_def, len(op.inputs)))
  for i in xrange(len(grads)):
    grad = grads[i]
    inp = op.inputs[i]
    if grad is not None:
      if not grad.dtype.is_compatible_with(inp.dtype):
        raise ValueError("Gradient type %s generated for op %s does "
                         "not match input type %s" %
                         (dtypes.as_dtype(grad.dtype).name, op.node_def,
                          dtypes.as_dtype(inp.dtype).name))


def _StopOps(from_ops, pending_count):
  """The set of ops that terminate the gradient computation.

  This computes the frontier of the forward graph *before* which backprop
  should stop. Operations in the returned set will not be differentiated.
  This set is defined as the subset of `from_ops` containing ops that have
  no predecessor in `from_ops`. `pending_count` is the result of
  `_PendingCount(g, xs, from_ops)`. An 'op' has predecessors in `from_ops`
  iff pending_count[op._id] > 0.

  Args:
    from_ops: list of Operations.
    pending_count: List of integers, indexed by operation id.

  Returns:
    The set of operations.
  """
  stop_ops = set()
  for op in from_ops:
    is_stop_op = True
    for inp in op.inputs:
      if pending_count[inp.op._id] > 0:
        is_stop_op = False
        break
    if is_stop_op:
      stop_ops.add(op._id)
  return stop_ops


def gradients(ys,
              xs,
              grad_ys=None,
              name="gradients",
              colocate_gradients_with_ops=False,
              gate_gradients=False,
              aggregation_method=None):
  """Constructs symbolic partial derivatives of `ys` w.r.t. x in `xs`.

  `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
  is a list of `Tensor`, holding the gradients received by the
  `ys`. The list must be the same length as `ys`.

  `gradients()` adds ops to the graph to output the partial
  derivatives of `ys` with respect to `xs`.  It returns a list of
  `Tensor` of length `len(xs)` where each tensor is the `sum(dy/dx)`
  for y in `ys`.

  `grad_ys` is a list of tensors of the same length as `ys` that holds
  the initial gradients for each y in `ys`.  When `grad_ys` is None,
  we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
  user can provide their own initial `grad_ys` to compute the
  derivatives using a different initial gradient for each y (e.g., if
  one wanted to weight the gradient differently for each value in
  each y).

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    grad_ys: Optional. A `Tensor` or list of tensors the same size as
      `ys` and holding the gradients computed for each y in `ys`.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'gradients'.
    colocate_gradients_with_ops: If True, try colocating gradients with
      the corresponding op.
    gate_gradients: If True, add a tuple around the gradients returned
      for an operations.  This avoids some race conditions.
    aggregation_method: Specifies the method used to combine gradient terms.
      Accepted values are constants defined in the class `AggregationMethod`.

  Returns:
    A list of `sum(dy/dx)` for each x in `xs`.

  Raises:
    LookupError: if one of the operations between `x` and `y` does not
      have a registered gradient function.
    ValueError: if the arguments are invalid.

  """
  ys = _AsList(ys)
  xs = _AsList(xs)
  if grad_ys is None:
    grad_ys = [None] * len(ys)
  else:
    grad_ys = _AsList(grad_ys)
  with ops.op_scope(ys + xs + grad_ys, name, "gradients"):
    ys = ops.convert_n_to_tensor_or_indexed_slices(ys, name="y")
    xs = ops.convert_n_to_tensor_or_indexed_slices(xs, name="x")
    grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops)

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
    pending_count, loop_state = _PendingCount(ops.get_default_graph(),
                                              to_ops, from_ops)

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
      # pylint: disable=protected-access
      ready = (pending_count[op._id] == 0)
      if ready and op._id not in to_ops_set:
        to_ops_set.add(op._id)
        queue.append(op)

    if loop_state:
      # The "unused" exits of the loops are added to ys. As an example,
      # people often write:
      #         v1, _ = While(p, b, [x1, x2])
      #         result = gradients(v1, x1)
      # The exit node of x2 is not included by the betweenness analysis.
      # But we need it if x2 is involved in computing v1. So we add it
      # back in backprop with a zeros_like gradient.
      loop_exits = loop_state.GetAllLoopExits()
      for y in loop_exits:
        if pending_count[y.op._id] == 0 and y.op._id not in to_ops_set:
          if _IsFloat(y):
            # Floating-point outputs get a zero gradient.
            _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
          queue.append(y.op)

    # The set of 'from_ops'.
    stop_ops = _StopOps(from_ops, pending_count)
    while queue:
      # generate gradient subgraph for op.
      op = queue.popleft()
      with ops.device(_GetGradsDevice(op, colocate_gradients_with_ops)):
        if loop_state:
          loop_state.EnterGradWhileContext(op)
        out_grads = _AggregatedGrads(grads, op, loop_state, aggregation_method)
        grad_fn = None

        # pylint: disable=protected-access
        is_func_call = ops.get_default_graph()._is_function(op.type)
        # pylint: enable=protected-access

        if not is_func_call and any(out_grads) and op._id not in stop_ops:
        # pylint: enable=protected-access
          # A grad_fn must be defined, either as a function or as None
          # for ops that do not have gradients.
          try:
            grad_fn = ops.get_gradient_function(op)
          except LookupError:
            raise LookupError(
                "No gradient defined for operation '%s' (op type: %s)" %
                (op.name, op.type))
        if (grad_fn or is_func_call) and any(out_grads):
          # NOTE: If _AggregatedGrads didn't compute a value for the i'th
          # output, it means that the cost does not depend on output[i],
          # therefore dC/doutput[i] is 0.
          for i, out_grad in enumerate(out_grads):
            if not out_grad and _IsFloat(op.outputs[i]):
              # Only floating-point outputs get a zero gradient. Gradient
              # functions should ignore the gradient for other outputs.
              if loop_state:
                out_grads[i] = loop_state.ZerosLike(op, i)
              else:
                out_grads[i] = array_ops.zeros_like(op.outputs[i])
          with ops.name_scope(op.name + "_grad"):
            # pylint: disable=protected-access
            with ops.get_default_graph()._original_op(op):
              # pylint: enable=protected-access
              wrapped_op = op
              if loop_state:
                wrapped_op = loop_state.MakeWrapper(op)
              if is_func_call:
                # For function call ops, we add a 'SymbolicGradient'
                # node to the graph to compute gradients.
                f_in = [x for x in op.inputs] + out_grads
                f_types = [x.dtype for x in op.inputs]
                # pylint: disable=protected-access
                in_grads = _AsList(functional_ops._symbolic_gradient(
                    f_in, f_types, op.type))
                # pylint: enable=protected-access
              else:
                in_grads = _AsList(grad_fn(wrapped_op, *out_grads))
              _VerifyGeneratedGradients(in_grads, op)
              if gate_gradients and len(tuple(filter(None, in_grads))) > 1:
                in_grads = control_flow_ops.tuple(in_grads)
          logging.vlog(1, "Gradient for '" + op.name + "'")
          logging.vlog(1, "  in  --> %s",
                       ", ".join([x.name for x in out_grads if x]))
          logging.vlog(1, "  out --> %s",
                       ", ".join([x.name for x in in_grads if x]))
        else:
          # If no grad_fn is defined or none of out_grads is available,
          # just propagates a list of None backwards.
          in_grads = [None] * len(op.inputs)
        for t_in, in_grad in zip(op.inputs, in_grads):
          if in_grad:
            _SetGrad(grads, t_in, in_grad)
        if loop_state:
          loop_state.ExitGradWhileContext(op)

      # update pending count for the inputs of op.
      # pylint: disable=protected-access
      for x in op.inputs:
        pending_count[x.op._id] -= 1
        ready = (pending_count[x.op._id] == 0)
        if loop_state and not ready:
          ready = (pending_count[x.op._id] > 0 and
                   control_flow_ops.IsLoopSwitch(x.op))
        if ready:
          queue.append(x.op)
      for x in op.control_inputs:
        pending_count[x._id] -= 1
        if pending_count[x._id] is 0:
          queue.append(x)
      # pylint: enable=protected-access
  return [_GetGrad(grads, x) for x in xs]


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
    assert control_flow_ops.IsLoopSwitch(op)
    op_grads[t.value_index] = grad


def _GetGrad(grads, t):
  """Gets gradient for tensor "t"."""
  op = t.op
  op_grads = grads.get(op)
  if not op_grads:
    return None
  t_grad = op_grads[t.value_index]
  assert not isinstance(t_grad, list), (
      "gradients list should have been aggregated by now.")
  return t_grad


def _GetGrads(grads, op):
  """Gets all gradients for op."""
  if op in grads:
    return grads[op]
  else:
    return [[] for _ in xrange(len(op.outputs))]


def _HandleNestedIndexedSlices(grad):
  assert isinstance(grad, ops.IndexedSlices)
  if isinstance(grad.values, ops.Tensor):
    return grad
  else:
    assert isinstance(grad.values, ops.IndexedSlices)
    g = _HandleNestedIndexedSlices(grad.values)
    return ops.IndexedSlices(
        g.values, array_ops.gather(grad.indices, g.indices), g.dense_shape)


def _AccumulatorShape(inputs):
  shape = tensor_shape.unknown_shape()
  for i in inputs:
    if isinstance(i, ops.Tensor):
      shape = shape.merge_with(i.get_shape())
  return shape


class AggregationMethod(object):
  """A class listing aggregation methods used to combine gradients.

  Computing partial derivatives can require aggregating gradient
  contributions. This class lists the various methods that can
  be used to combine gradients in the graph:

  *  `ADD_N`: All of the gradient terms are summed as part of one
     operation using the "AddN" op. It has the property that all
     gradients must be ready before any aggregation is performed.
  *  `DEFAULT`: The system-chosen default aggregation method.
  """
  ADD_N = 0
  DEFAULT = ADD_N
  # The following are experimental and may not be supported in future releases.
  EXPERIMENTAL_TREE = 1
  EXPERIMENTAL_ACCUMULATE_N = 2


def _AggregatedGrads(grads, op, loop_state, aggregation_method=None):
  """Get the aggregated gradients for op.

  Args:
    grads: The map of memoized gradients.
    op: The op to get gradients for.
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
  if aggregation_method not in [AggregationMethod.ADD_N,
                                AggregationMethod.EXPERIMENTAL_TREE,
                                AggregationMethod.EXPERIMENTAL_ACCUMULATE_N]:
    raise ValueError(
        "Invalid aggregation_method specified %s." % aggregation_method)
  out_grads = _GetGrads(grads, op)
  for i, out_grad in enumerate(out_grads):
    if loop_state:
      if isinstance(out_grad, (ops.Tensor, ops.IndexedSlices)):
        assert control_flow_ops.IsLoopSwitch(op)
        continue
    # Grads have to be Tensors or IndexedSlices
    if not all([isinstance(g, (ops.Tensor, ops.IndexedSlices))
                for g in out_grad if g]):
      raise TypeError("gradients have to be either all Tensors "
                      "or all IndexedSlices")
    # Aggregate multiple gradients, and convert [] to None.
    if out_grad:
      if all([isinstance(g, ops.Tensor) for g in out_grad if g]):
        tensor_shape = _AccumulatorShape(out_grad)
        if len(out_grad) < 2:
          used = "nop"
          out_grads[i] = out_grad[0]
        elif (aggregation_method == AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
              and len(out_grad) > 2 and tensor_shape.is_fully_defined()):
          # The benefit of using AccumulateN is that its inputs can be combined
          # in any order and this can allow the expression to be evaluated with
          # a smaller memory footprint.  When used with gpu_allocator_retry,
          # it is possible to compute a sum of terms which are much larger than
          # total GPU memory.
          # AccumulateN can currently only be used if we know the shape for
          # an accumulator variable.  If this is not known, or if we only have
          # 2 grads then we fall through to the "tree" case below.
          used = "accumulate_n"
          out_grads[i] = math_ops.accumulate_n(out_grad)
        elif aggregation_method in [AggregationMethod.EXPERIMENTAL_TREE,
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
          out_grads[i] = math_ops.add_n(out_grad)
        logging.vlog(2, "  _AggregatedGrads %d x %s using %s", len(out_grad),
                     tensor_shape, used)
      else:
        out_grad = math_ops._as_indexed_slices_list([g for g in out_grad if g])
        out_grad = [_HandleNestedIndexedSlices(x) for x in out_grad]
        # Form IndexedSlices out of the concatenated values and
        # indices.
        out_grads[i] = ops.IndexedSlices(
            array_ops.concat(0, [x.values for x in out_grad]),
            array_ops.concat(0, [x.indices
                                 for x in out_grad]), out_grad[0].dense_shape)
    else:
      out_grads[i] = []
  return out_grads


# TODO(vrv): Make this available when we want to make it public.
def _hessian_vector_product(ys, xs, v):
  """Multiply the Hessian of `ys` wrt `xs` by `v`.

  This is an efficient construction that uses a backprop-like approach
  to compute the product between the Hessian and another vector. The
  Hessian is usually too large to be explicitly computed or even
  represented, but this method allows us to at least multiply by it
  for the same big-O cost as backprop.

  Implicit Hessian-vector products are the main practical, scalable way
  of using second derivatives with neural networks. They allow us to
  do things like construct Krylov subspaces and approximate conjugate
  gradient descent.

  Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
  x, v)` will return an expression that evaluates to the same values
  as (A + A.T) `v`.

  Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    v: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.

  Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `v`.

  Raises:
    ValueError: `xs` and `v` have different length.

  """

  # Validate the input
  length = len(xs)
  if len(v) != length:
    raise ValueError("xs and v must have the same length.")

  # First backprop
  grads = gradients(ys, xs)

  assert len(grads) == length
  elemwise_products = [math_ops.mul(grad_elem, array_ops.stop_gradient(v_elem))
                       for grad_elem, v_elem in zip(grads, v)
                       if grad_elem is not None]

  # Second backprop
  return gradients(elemwise_products, xs)
