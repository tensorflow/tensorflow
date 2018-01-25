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
import warnings

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import image_grad  # pylint: disable=unused-import
from tensorflow.python.ops import linalg_grad  # pylint: disable=unused-import
from tensorflow.python.ops import linalg_ops  # pylint: disable=unused-import
from tensorflow.python.ops import logging_ops  # pylint: disable=unused-import
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import spectral_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export

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
  return math_ops.unsorted_segment_sum(
      value.values, value.indices, value.dense_shape[0], name=name)


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


def _PendingCount(graph, to_ops, from_ops, colocate_gradients_with_ops):
  """Initialize the pending count for ops between two lists of Operations.

  'pending_count[op._id]' indicates the number of backprop inputs
  to this operation.

  Args:
    graph: a Graph.
    to_ops: list of Operations.
    from_ops: list of Operations.
    colocate_gradients_with_ops: Python bool.  See docstring of gradients().

  Returns:
    A tuple containing: (1) a list of integers indexed by operation id,
    indicating the number of backprop inputs to this operation, and (2)
    a ControlFlowState object which is not None if the ops between from_ops
    and to_ops contain control flow loops.
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
  loop_state = control_flow_ops.MaybeCreateControlFlowState(
      between_op_list, between_ops, colocate_gradients_with_ops)

  # Initialize pending count for between ops.
  pending_count = [0] * (graph._last_id + 1)
  for op in between_op_list:
    for x in op.inputs:
      if between_ops[x.op._id]:
        pending_count[x.op._id] += 1

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
    ValueError: If sizes of gradients and inputs don't match
    TypeError: If type of any gradient is not valid for its input.
  """
  if len(grad_ys) != len(ys):
    raise ValueError("Passed %d grad_ys for %d ys" % (len(grad_ys), len(ys)))
  grad_ys = ops.convert_n_to_tensor_or_indexed_slices(grad_ys, name="grad_y")
  new_grad_ys = []
  for i in xrange(len(grad_ys)):
    grad_y = grad_ys[i]
    y = ys[i]
    with _maybe_colocate_with(y.op, colocate_gradients_with_ops):
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
          raise TypeError("Gradient type %s generated for real or "
                          "integer-valued tensor %s with type %s must be "
                          "real or integer" %
                          (dtypes.as_dtype(grad_y.dtype).name, y,
                           dtypes.as_dtype(y.dtype).name))
      elif y.dtype.is_complex:
        if not grad_y.dtype.is_complex:
          raise TypeError("Gradient type %s generated for complex-valued "
                          "tensor %s with type %s must be real" %
                          (dtypes.as_dtype(grad_y.dtype).name, y,
                           dtypes.as_dtype(y.dtype).name))
      else:
        raise TypeError("Tensor %s with type %s must be numeric "
                        "to obtain a default gradient" %
                        (y, dtypes.as_dtype(y.dtype).name))
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


def _IsTrainable(tensor):
  dtype = dtypes.as_dtype(tensor.dtype)
  return dtype.base_dtype in (dtypes.float16, dtypes.float32, dtypes.float64,
                              dtypes.complex64, dtypes.complex128)


def _VerifyGeneratedGradients(grads, op):
  """Verify that gradients are valid in number and type.

  Args:
    grads: List of generated gradients.
    op: Operation for which the gradients where generated.

  Raises:
    ValueError: if sizes of gradients and inputs don't match.
    TypeError: if type of any gradient is not valid for its input.
  """
  if len(grads) != len(op.inputs):
    raise ValueError("Num gradients %d generated for op %s do not match num "
                     "inputs %d" % (len(grads), op.node_def, len(op.inputs)))


def _StopOps(from_ops, stop_gradient_ops, pending_count):
  """The set of ops that terminate the gradient computation.

  This computes the frontier of the forward graph *before* which backprop
  should stop. Operations in the returned set will not be differentiated.
  This set is defined as the subset of `from_ops` containing ops that have
  no predecessor in `from_ops`. `pending_count` is the result of
  `_PendingCount(g, xs, from_ops)`. An 'op' has predecessors in `from_ops`
  iff pending_count[op._id] > 0.

  In addition, none of `stop_gradient_ops` will be differentiated.

  Args:
    from_ops: list of Operations.
    stop_gradient_ops: list of Operations never to backprop through.
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
  stop_ops.update(op._id for op in stop_gradient_ops)  # pylint: disable=protected-access
  return stop_ops


@contextlib.contextmanager
def _maybe_colocate_with(op, colocate_gradients_with_ops):
  """Context to colocate with `op` if `colocate_gradients_with_ops`."""
  if colocate_gradients_with_ops:
    with ops.colocate_with(op):
      yield
  else:
    yield


def _SymGrad(op, out_grads):
  """Backprop through a function call node op given its outputs' gradients."""
  f_in = [x for x in op.inputs] + out_grads
  f_types = [x.dtype for x in op.inputs]
  f = attr_value_pb2.NameAttrList()
  f.name = op.type
  for k in op.node_def.attr:
    f.attr[k].CopyFrom(op.node_def.attr[k])
  # pylint: disable=protected-access
  in_grads = functional_ops._symbolic_gradient(input=f_in, Tout=f_types, f=f)
  # pylint: enable=protected-access
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
      return grad_fn()  # Exit early

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


@tf_export("gradients")
def gradients(ys,
              xs,
              grad_ys=None,
              name="gradients",
              colocate_gradients_with_ops=False,
              gate_gradients=False,
              aggregation_method=None,
              stop_gradients=None):
  """Constructs symbolic derivatives of sum of `ys` w.r.t. x in `xs`.

  `ys` and `xs` are each a `Tensor` or a list of tensors.  `grad_ys`
  is a list of `Tensor`, holding the gradients received by the
  `ys`. The list must be the same length as `ys`.

  `gradients()` adds ops to the graph to output the derivatives of `ys` with
  respect to `xs`.  It returns a list of `Tensor` of length `len(xs)` where
  each tensor is the `sum(dy/dx)` for y in `ys`.

  `grad_ys` is a list of tensors of the same length as `ys` that holds
  the initial gradients for each y in `ys`.  When `grad_ys` is None,
  we fill in a tensor of '1's of the shape of y for each y in `ys`.  A
  user can provide their own initial `grad_ys` to compute the
  derivatives using a different initial gradient for each y (e.g., if
  one wanted to weight the gradient differently for each value in
  each y).

  `stop_gradients` is a `Tensor` or a list of tensors to be considered constant
  with respect to all `xs`. These tensors will not be backpropagated through,
  as though they had been explicitly disconnected using `stop_gradient`.  Among
  other things, this allows computation of partial derivatives as opposed to
  total derivatives. For example:

  ```python
  a = tf.constant(0.)
  b = 2 * a
  g = tf.gradients(a + b, [a, b], stop_gradients=[a, b])
  ```

  Here the partial derivatives `g` evaluate to `[1.0, 1.0]`, compared to the
  total derivatives `tf.gradients(a + b, [a, b])`, which take into account the
  influence of `a` on `b` and evaluate to `[3.0, 1.0]`.  Note that the above is
  equivalent to:

  ```python
  a = tf.stop_gradient(tf.constant(0.))
  b = tf.stop_gradient(2 * a)
  g = tf.gradients(a + b, [a, b])
  ```

  `stop_gradients` provides a way of stopping gradient after the graph has
  already been constructed, as compared to `tf.stop_gradient` which is used
  during graph construction.  When the two approaches are combined,
  backpropagation stops at both `tf.stop_gradient` nodes and nodes in
  `stop_gradients`, whichever is encountered first.

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
    stop_gradients: Optional. A `Tensor` or list of tensors not to differentiate
      through.

  Returns:
    A list of `sum(dy/dx)` for each x in `xs`.

  Raises:
    LookupError: if one of the operations between `x` and `y` does not
      have a registered gradient function.
    ValueError: if the arguments are invalid.
    RuntimeError: if called in Eager mode.

  """
  if context.in_eager_mode():
    raise RuntimeError("tf.gradients not supported in EAGER mode. Use "
                       "functions in tf.contrib.eager.backprop instead.")
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
    ys = ops.convert_n_to_tensor_or_indexed_slices(ys, name="y")
    xs = [
        x.handle if isinstance(x, resource_variable_ops.ResourceVariable) else x
        for x in xs
    ]
    xs = ops.internal_convert_n_to_tensor_or_indexed_slices(
        xs, name="x", as_ref=True)
    grad_ys = _DefaultGradYs(grad_ys, ys, colocate_gradients_with_ops)

    # The approach we take here is as follows: Create a list of all ops in the
    # subgraph between the ys and xs.  Visit these ops in reverse order of ids
    # to ensure that when we visit an op the gradients w.r.t its outputs have
    # been collected.  Then aggregate these gradients if needed, call the op's
    # gradient function, and add the generated gradients to the gradients for
    # its input.

    # Initialize the pending count for ops in the connected subgraph from ys
    # to the xs.
    if len(ys) > 1:
      ys = [array_ops.identity(y) if y.consumers() else y for y in ys]
    to_ops = [t.op for t in ys]
    from_ops = [t.op for t in xs]
    stop_gradient_ops = [t.op for t in stop_gradients]
    pending_count, loop_state = _PendingCount(
        ops.get_default_graph(), to_ops, from_ops, colocate_gradients_with_ops)

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
      # pylint: enable=protected-access

    if loop_state:
      loop_exits = loop_state.ProcessUnusedLoopExits(pending_count, to_ops_set)
      for y in loop_exits:
        if _IsTrainable(y):
          _SetGrad(grads, y, loop_state.ZerosLikeForExit(y))
          queue.append(y.op)

    stop_ops = _StopOps(from_ops, stop_gradient_ops, pending_count)
    while queue:
      # generate gradient subgraph for op.
      op = queue.popleft()
      with _maybe_colocate_with(op, colocate_gradients_with_ops):
        if loop_state:
          loop_state.EnterGradWhileContext(op, before=True)
        out_grads = _AggregatedGrads(grads, op, loop_state, aggregation_method)
        if loop_state:
          loop_state.ExitGradWhileContext(op, before=True)

        grad_fn = None
        # pylint: disable=protected-access
        func_call = None
        is_func_call = ops.get_default_graph()._is_function(op.type)
        has_out_grads = any(isinstance(g, ops.Tensor) or g for g in out_grads)
        if has_out_grads and (op._id not in stop_ops):
          if is_func_call:
            func_call = ops.get_default_graph()._get_function(op.type)
            grad_fn = func_call.python_grad_func
            # pylint: enable=protected-access
          else:
            # A grad_fn must be defined, either as a function or as None
            # for ops that do not have gradients.
            try:
              grad_fn = ops.get_gradient_function(op)
            except LookupError:
              raise LookupError(
                  "No gradient defined for operation '%s' (op type: %s)" %
                  (op.name, op.type))
        if loop_state:
          loop_state.EnterGradWhileContext(op, before=False)
        if (grad_fn or is_func_call) and has_out_grads:
          # NOTE: If _AggregatedGrads didn't compute a value for the i'th
          # output, it means that the cost does not depend on output[i],
          # therefore dC/doutput[i] is 0.
          for i, out_grad in enumerate(out_grads):
            if (not isinstance(out_grad, ops.Tensor) and not out_grad) and (
                (not grad_fn and is_func_call) or _IsTrainable(op.outputs[i])):
              # Only trainable outputs or outputs for a function call that
              # will use SymbolicGradient get a zero gradient. Gradient
              # functions should ignore the gradient for other outputs.
              # TODO(apassos) gradients of resource handles might be an
              # issue here because of zeros.
              if loop_state:
                out_grads[i] = loop_state.ZerosLike(op, i)
              else:
                out_grads[i] = control_flow_ops.ZerosLikeOutsideLoop(op, i)
          with ops.name_scope(op.name + "_grad"):
            # pylint: disable=protected-access
            with ops.get_default_graph()._original_op(op):
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
                  with ops.colocate_with(None, ignore_existing=True):
                    in_grads = control_flow_ops.tuple(in_grads)
          _LogOpGradients(op, out_grads, in_grads)
        else:
          # If no grad_fn is defined or none of out_grads is available,
          # just propagate a list of None backwards.
          in_grads = [None] * len(op.inputs)
        for i, (t_in, in_grad) in enumerate(zip(op.inputs, in_grads)):
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
            _SetGrad(grads, t_in, in_grad)
        if loop_state:
          loop_state.ExitGradWhileContext(op, before=False)

      # Update pending count for the inputs of op and enqueue ready ops.
      _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state)

  if loop_state:
    loop_state.PostProcessing()
  return [_GetGrad(grads, x) for x in xs]


def _HasAnyNotNoneGrads(grads, op):
  """Return true iff op has real gradient."""
  out_grads = _GetGrads(grads, op)
  for out_grad in out_grads:
    if isinstance(out_grad, (ops.Tensor, ops.IndexedSlices)):
      return True
    if out_grad and isinstance(out_grad, collections.Sequence):
      if any([g is not None for g in out_grad]):
        return True
  return False


def _UpdatePendingAndEnqueueReady(grads, op, queue, pending_count, loop_state):
  """Update pending count for the inputs of op and enqueue ready ops."""
  for x in op.inputs:
    # pylint: disable=protected-access
    pending_count[x.op._id] -= 1
    ready = (pending_count[x.op._id] == 0)
    if loop_state and not ready:
      ready = (
          pending_count[x.op._id] > 0 and control_flow_util.IsLoopSwitch(x.op))
    # pylint: enable=protected-access
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
              if _IsTrainable(y):
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


def _GetGrad(grads, t):
  """Gets gradient for tensor "t"."""
  op = t.op
  op_grads = grads.get(op)
  if not op_grads:
    return None
  t_grad = op_grads[t.value_index]
  assert not isinstance(
      t_grad, list), ("gradients list should have been aggregated by now.")
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
    return ops.IndexedSlices(g.values, array_ops.gather(
        grad.indices, g.indices), g.dense_shape)


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
               ", ".join([x.name for x in out_grads if _FilterGrad(x)]))
  logging.vlog(1, "  out --> %s",
               ", ".join([x.name for x in in_grads if _FilterGrad(x)]))


def _MultiDeviceAddN(tensor_list):
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

  for dev in sorted(six.iterkeys(tensors_on_device), key=DeviceKey):
    tensors = tensors_on_device[dev]
    with ops.colocate_with(tensors[0].op, ignore_existing=True):
      summands.append(math_ops.add_n(tensors))

  return math_ops.add_n(summands)


@tf_export("AggregationMethod")
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
    if (isinstance(out_grad, collections.Sequence) and not all([
        isinstance(g, (ops.Tensor, ops.IndexedSlices))
        for g in out_grad
        if g is not None
    ])):
      raise TypeError("gradients have to be either all Tensors "
                      "or all IndexedSlices")
    # Aggregate multiple gradients, and convert [] to None.
    if out_grad:
      if len(out_grad) < 2:
        used = "nop"
        out_grads[i] = out_grad[0]
      elif all([isinstance(g, ops.Tensor) for g in out_grad if g is not None]):
        tensor_shape = _AccumulatorShape(out_grad)
        if (aggregation_method == AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
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
        elif aggregation_method in [
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
          out_grads[i] = _MultiDeviceAddN(out_grad)
        logging.vlog(2, "  _AggregatedGrads %d x %s using %s", len(out_grad),
                     tensor_shape, used)
      else:
        out_grad = math_ops._as_indexed_slices_list(
            [g for g in out_grad if g is not None])
        out_grad = [_HandleNestedIndexedSlices(x) for x in out_grad]
        # Form IndexedSlices out of the concatenated values and
        # indices.
        out_grads[i] = ops.IndexedSlices(
            array_ops.concat([x.values for x in out_grad], 0),
            array_ops.concat([x.indices for x in out_grad], 0),
            out_grad[0].dense_shape)
    else:  # not out_grad
      # out_grads[i] is [], thus its aggregation is simply None.
      out_grads[i] = None
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
  elemwise_products = [
      math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
      for grad_elem, v_elem in zip(grads, v)
      if grad_elem is not None
  ]

  # Second backprop
  return gradients(elemwise_products, xs)


@tf_export("hessians")
def hessians(ys,
             xs,
             name="hessians",
             colocate_gradients_with_ops=False,
             gate_gradients=False,
             aggregation_method=None):
  """Constructs the Hessian of sum of `ys` with respect to `x` in `xs`.

  `hessians()` adds ops to the graph to output the Hessian matrix of `ys`
  with respect to `xs`.  It returns a list of `Tensor` of length `len(xs)`
  where each tensor is the Hessian of `sum(ys)`.

  The Hessian is a matrix of second-order partial derivatives of a scalar
  tensor (see https://en.wikipedia.org/wiki/Hessian_matrix for more details).

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    name: Optional name to use for grouping all the gradient ops together.
      defaults to 'hessians'.
    colocate_gradients_with_ops: See `gradients()` documentation for details.
    gate_gradients: See `gradients()` documentation for details.
    aggregation_method: See `gradients()` documentation for details.

  Returns:
    A list of Hessian matrices of `sum(ys)` for each `x` in `xs`.

  Raises:
    LookupError: if one of the operations between `xs` and `ys` does not
      have a registered gradient function.
  """
  xs = _AsList(xs)
  kwargs = {
      "colocate_gradients_with_ops": colocate_gradients_with_ops,
      "gate_gradients": gate_gradients,
      "aggregation_method": aggregation_method
  }
  # Compute first-order derivatives and iterate for each x in xs.
  hessians = []
  _gradients = gradients(ys, xs, **kwargs)
  for gradient, x in zip(_gradients, xs):
    # change shape to one-dimension without graph branching
    gradient = array_ops.reshape(gradient, [-1])

    # Declare an iterator and tensor array loop variables for the gradients.
    n = array_ops.size(x)
    loop_vars = [
        array_ops.constant(0, dtypes.int32),
        tensor_array_ops.TensorArray(x.dtype, n)
    ]
    # Iterate over all elements of the gradient and compute second order
    # derivatives.
    _, hessian = control_flow_ops.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j + 1,
                           result.write(j, gradients(gradient[j], x)[0])),
        loop_vars
    )

    _shape = array_ops.shape(x)
    _reshaped_hessian = array_ops.reshape(hessian.stack(),
                                          array_ops.concat((_shape, _shape), 0))
    hessians.append(_reshaped_hessian)
  return hessians
