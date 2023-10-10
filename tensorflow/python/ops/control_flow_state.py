# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for managing state of v1 control flow for computing gradients."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops

# pylint: disable=protected-access


def _GetMaxSizeFromNestedMaximumIterations(value, while_ctxt):
  """Calculate a max_size for use by stack ops inside an XLA while_loop.

  Args:
    value: The value inside the while_loop forward context.  Used for printing
      error messages.
    while_ctxt: The forward context inside which value resides.  This does not
      always match the value's immediate context, as `value` may be inside e.g.
      a cond context inside the while_loop.

  Returns:
    A tensor containing the `max_size` to feed to a Stack initializer.

  Raises:
    ValueError: If `value` is nested inside a `while_loop` that either
      lacks a `maximum_iterations` parameter, or the `maximum_iterations`
      parameter:

        - is inside a `while_loop` that is a parent of the calling context, and
        - cannot be evaluated at graph build time to a constant.
  """
  value_name = value.name
  # curr_ctxt is the context that tf.gradients was called in.
  curr_ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access

  curr_ctxt_name = curr_ctxt.name if curr_ctxt is not None else ""
  max_size = constant_op.constant(1)

  # Loop through all containing while contexts between value and the
  # current context, multiplying together each context's
  # max_iterations to get the maximum stack size.
  while while_ctxt not in (None, curr_ctxt):
    max_iter = while_ctxt.maximum_iterations
    if max_iter is None:
      raise ValueError(
          "Cannot create a gradient accumulator for tensor '%s' inside "
          "XLA while_loop because maximum_iterations was not passed to "
          "the tf.while_loop call ('%s')." % (value_name, while_ctxt.name))

    # pylint: disable=protected-access
    max_iter_ctxt = max_iter.op._get_control_flow_context()
    # pylint: enable=protected-access

    # If max_iter_ctxt (non-strictly) contains curr_ctxt, then it's OK to use.
    if util.IsContainingContext(curr_ctxt, max_iter_ctxt):
      max_size *= max_iter
    else:
      # We cannot use max_iter because it's defined in a nested while
      # or cond context, so will fail if we try to use it as input to
      # any ops in curr_ctxt (e.g. max_size or the final accumulator
      # stack). Attempt to get a constant value out to use instead.
      const_max_iter = tensor_util.constant_value(max_iter)
      if const_max_iter is None:
        raise ValueError(
            "Cannot create a gradient accumulator for tensor '%s' inside XLA "
            "while_loop. maximum_iterations tensor '%s' for while_loop context "
            "'%s' must be statically known (e.g. a constant value or known "
            "shape dimension), or be defined at or outside the while loop "
            "context '%s' (currently defined in '%s')." %
            (value_name, max_iter.name, while_ctxt.name, curr_ctxt_name,
             max_iter_ctxt.name))
      max_size *= const_max_iter

    # Find the next outer WhileContext (or stop if we reach the
    # tf.gradient's context).
    while_ctxt = util.GetContainingWhileContext(
        while_ctxt.outer_context, stop_ctxt=curr_ctxt)

  return max_size


class _GradLoopState:
  """The state used for constructing the gradient graph for a while loop.

  We create a _GradLoopState for each while loop in forward and its
  corresponding while loop in backprop. This gives us access to both
  the forward and the backprop WhileContexts.

  During the construction of gradient graph, any time when we detect
  a forward value that is needed for backprop, we create a history
  accumulator and add it to `history_map`. Any time when we backprop
  a loop switch op (in _SwitchGrad), we add the grad merge op in
  `switch_map`.
  """

  def __init__(self, forward_ctxt, outer_grad_state):
    # The grad loop state for the outer while loop.
    self._outer_grad_state = None

    # The while loop context for forward.
    self._forward_context = None

    # The loop counter added by AddForwardLoopCounter. It is the value
    # of the loop counter for the next iteration.
    self._forward_index = None

    # A sync op for forward.
    self._forward_sync = None

    # The while loop context for backprop.
    self._grad_context = None

    # The loop counter added by AddBackpropLoopCounter. It is the value
    # of the loop counter for the current iteration.
    self._grad_index = None

    # A sync op for backprop.
    self._grad_sync = None

    # Information needed by backprop.
    self._history_map = {}
    self._switch_map = {}
    self._unused_exits = []
    self._deferred_exits = []
    self._forward_loop_exits = list(forward_ctxt.loop_exits)
    self._pending_exits_count = len(forward_ctxt.loop_exits)

    self._outer_grad_state = outer_grad_state
    if outer_grad_state:
      outer_forward_ctxt = outer_grad_state.forward_context
    else:
      if not hasattr(forward_ctxt, "outer_context"):
        raise ValueError("Failed to call gradients on a while loop without"
                         "properly serializing graph via MetaGraphDef")
      outer_forward_ctxt = forward_ctxt.outer_context

    # Add the forward loop counter.
    with forward_ctxt._graph.as_default():  # pylint: disable=protected-access
      if outer_forward_ctxt:
        outer_forward_ctxt.Enter()
      cnt, forward_index = forward_ctxt.AddForwardLoopCounter(outer_grad_state)
      if outer_forward_ctxt:
        outer_forward_ctxt.Exit()
    self._forward_context = forward_ctxt
    self._forward_index = forward_index

    # Add the backprop WhileContext, and the backprop loop counter.
    if outer_grad_state:
      # This is a nested loop. Remember the iteration counts for each
      # execution of this inner loop.
      outer_forward_ctxt.AddName(cnt.name)
      history_cnt = outer_grad_state.AddForwardAccumulator(cnt)

      outer_grad_ctxt = outer_grad_state.grad_context
      outer_grad_ctxt.Enter()
      self._grad_context = control_flow_ops.WhileContext(
          maximum_iterations=forward_ctxt.maximum_iterations,
          parallel_iterations=forward_ctxt.parallel_iterations,
          back_prop=forward_ctxt.back_prop,
          swap_memory=forward_ctxt.swap_memory,
          name=forward_ctxt.name,
          grad_state=self)
      real_cnt = outer_grad_state.AddBackpropAccumulatedValue(history_cnt, cnt)
      self._grad_index = self._grad_context.AddBackpropLoopCounter(
          real_cnt, outer_grad_state)
      outer_grad_ctxt.Exit()
    else:
      if outer_forward_ctxt:
        outer_forward_ctxt.Enter()
      self._grad_context = control_flow_ops.WhileContext(
          maximum_iterations=forward_ctxt.maximum_iterations,
          parallel_iterations=forward_ctxt.parallel_iterations,
          back_prop=forward_ctxt.back_prop,
          swap_memory=forward_ctxt.swap_memory,
          name=forward_ctxt.name,
          grad_state=self)
      self._grad_index = self._grad_context.AddBackpropLoopCounter(
          cnt, outer_grad_state)
      if outer_forward_ctxt:
        outer_forward_ctxt.Exit()

  @property
  def outer_grad_state(self):
    """The grad loop state for outer loop."""
    return self._outer_grad_state

  @property
  def forward_context(self):
    """The while loop context for forward."""
    return self._forward_context

  @property
  def forward_index(self):
    """The loop index of forward loop."""
    return self._forward_index

  @property
  def forward_sync(self):
    """A control trigger node for synchronization in the forward loop.

    One main use is to keep the push ops of a stack executed in the
    iteration order.
    """
    if self._forward_sync is None:
      with ops.control_dependencies(None):
        self._forward_sync = control_flow_ops.control_trigger(name="f_sync")
      self._forward_sync._set_control_flow_context(self._forward_context)
      self._forward_index.op._add_control_input(self._forward_sync)
    return self._forward_sync

  @property
  def grad_context(self):
    """The corresponding WhileContext for gradient."""
    return self._grad_context

  @property
  def grad_index(self):
    """The loop index of backprop loop."""
    return self._grad_index

  @property
  def grad_sync(self):
    """A control trigger node for synchronization in the grad loop.

    One main use is to keep the pop ops of a stack executed in the
    iteration order.
    """
    if self._grad_sync is None:
      with ops.control_dependencies(None):
        self._grad_sync = control_flow_ops.control_trigger(name="b_sync")
      self._grad_sync._set_control_flow_context(self._grad_context)
      self._grad_index.op._add_control_input(self._grad_sync)
      if self._grad_context.outer_context:
        self._grad_context.outer_context.AddInnerOp(self._grad_sync)
    return self._grad_sync

  @property
  def history_map(self):
    """The map that records all the tensors needed for backprop."""
    return self._history_map

  @property
  def switch_map(self):
    """The map that records all the Switch ops for the while loop."""
    return self._switch_map

  @property
  def unused_exits(self):
    """The list of "unused" exits."""
    return self._unused_exits

  @property
  def deferred_exits(self):
    """The list of "deferred" exits."""
    return self._deferred_exits

  @property
  def forward_loop_exits(self):
    """The list of exits of the forward loop."""
    return self._forward_loop_exits

  @property
  def pending_exits_count(self):
    """The number of exits we expect to see but haven't."""
    return self._pending_exits_count

  @pending_exits_count.setter
  def pending_exits_count(self, cnt):
    """Set the pending count to cnt."""
    self._pending_exits_count = cnt

  def AddForwardAccumulator(self, value, dead_branch=False):
    """Add an accumulator for each forward tensor that is needed in backprop.

    This is added to the forward loop at the first time when a tensor
    in the forward loop is used by backprop gradient computation loop.
    We create an accumulator that accumulates the value of tensor at each
    iteration. Called in the control flow context where gradients() is called.

    The pseudocode is:
    ```
      acc = stack();
      while (_pivot) {
        acc = stack_push(acc, value);
      }
    ```

    We make sure that the stack push op in one iteration is executed before
    next iteration. This is achieved by adding a control edge from
    `forward_index.op.inputs[0].op` to the push op, and another control
    edge from the push op to either `forward_index.op` or `forward_sync`.

    Args:
      value: The source tensor in forward that is to be accumulated.
      dead_branch: True iff the tensor is on a dead branch of a cond.

    Returns:
      The stack that contains the accumulated history of the tensor.

    Raises:
      TypeError: For internal errors involving the value condition context.
      ValueError: If `value` is inside a XLA scope and a valid max size
        for the stack can't be found.
    """
    # curr_ctxt is the context that tf.gradients was called in.
    with self._forward_index.graph.as_default():
      curr_ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
      with ops.control_dependencies(None):
        if curr_ctxt:
          curr_ctxt.Enter()
        with ops.colocate_with(value):
          # We only need to pass maximum_iterations to the stack if
          # we're inside an XLA context.
          if not util.IsInXLAContext(value.op):
            max_size = constant_op.constant(-1, dtypes.int32)
          else:
            max_size = _GetMaxSizeFromNestedMaximumIterations(
                value, self.forward_context)
          acc = gen_data_flow_ops.stack_v2(
              max_size=max_size, elem_type=value.dtype.base_dtype, name="f_acc")
        if curr_ctxt:
          curr_ctxt.Exit()

        # Make acc available in the forward context.
        enter_acc = self.forward_context.AddValue(acc)

        # Add the stack_push op in the context of value.op.
        swap_enabled = self.forward_context.swap_memory
        value_ctxt = util.GetOutputContext(value.op)
        if value_ctxt == self.forward_context:
          # value is not nested in the forward context.
          self.forward_context.Enter()
          push = gen_data_flow_ops.stack_push_v2(
              enter_acc, value, swap_memory=swap_enabled)
          self.forward_context.Exit()
          # Protect stack push and order it before forward_index.
          self.forward_index.op._add_control_input(push.op)
        else:
          # value is in a cond context within the forward context.
          if not isinstance(value_ctxt, control_flow_ops.CondContext):
            raise TypeError("value_ctxt is not a CondContext: %s" % value_ctxt)
          if dead_branch:
            # The special case for creating a zero tensor for a dead
            # branch of a switch. See _ControlFlowState.ZerosLikeV1WhileLoop().
            value_ctxt.outer_context.Enter()
            push = gen_data_flow_ops.stack_push_v2(
                enter_acc, value, swap_memory=swap_enabled)
            value_ctxt.outer_context.Exit()
            push.op._set_control_flow_context(value_ctxt)
          else:
            value_ctxt.Enter()
            push = gen_data_flow_ops.stack_push_v2(
                enter_acc, value, swap_memory=swap_enabled)
            value_ctxt.Exit()
          # Protect stack push and order it before forward_sync.
          self.forward_sync._add_control_input(push.op)
        # Order stack push after the successor of forward_index
        add_op = self.forward_index.op.inputs[0].op
        push.op._add_control_input(add_op)
        return acc

  def AddBackpropAccumulatedValue(self, history_value, value,
                                  dead_branch=False):
    """Add the getter for an accumulated value in the grad context.

    This is added to the backprop loop. Called in the grad context to
    get the value of an accumulated value. The stack pop op must be guarded
    by the pred of the controlling cond.

    Args:
      history_value: The history (a stack) of a value.
      value: The value that is pushed onto the stack.
      dead_branch: True iff the tensor is on a dead branch of a cond.

    Returns:
      The current value (the top of the stack).
    """
    history_ctxt = history_value.op._get_control_flow_context()
    # Find the cond context that controls history_value if any.
    cond_ctxt = None
    value_ctxt = value.op._get_control_flow_context()
    while value_ctxt and value_ctxt != history_ctxt:
      if isinstance(value_ctxt, control_flow_ops.CondContext):
        cond_ctxt = value_ctxt
        break
      value_ctxt = value_ctxt.outer_context
    with ops.control_dependencies(None):
      self.grad_context.Enter()
      if cond_ctxt:
        # Guard stack pop with a switch if it is controlled by a cond.
        grad_state = self
        pred = None
        while pred is None and grad_state:
          pred = grad_state.history_map.get(cond_ctxt.pred.name)
          grad_state = grad_state.outer_grad_state
        if pred is None:
          pred = cond_ctxt.pred
        branch = (1 - cond_ctxt.branch) if dead_branch else cond_ctxt.branch
        history_value = control_flow_ops._SwitchRefOrTensor(
            history_value, pred)[branch]
      pop = gen_data_flow_ops.stack_pop_v2(history_value,
                                           value.dtype.base_dtype)
      pop.set_shape(value.get_shape())
      self.grad_context.Exit()
    parallel_iterations = self.grad_context.parallel_iterations
    if parallel_iterations > 1:
      # All pops are ordered after pivot_for_body and before grad_sync.
      self.grad_sync._add_control_input(pop.op)
    return pop

  def GetRealValue(self, value):
    """Get the real value of `value`.

    If backprop "uses" a value produced by forward inference, an accumulator
    is added in the forward loop to accumulate its values.  We use the
    accumulated value. This method must be called in the grad loop context.
    `value` must be in forward and needed for backprop.

    Args:
      value: A tensor to be captured.

    Returns:
      The same tensor obtained from the saved history.
    """
    assert value.op.type not in ["Variable", "VariableV2"]
    real_value = self._history_map.get(value.name)
    if real_value is None:
      cur_value = value
      cur_grad_state = self
      while True:
        enter_op = util.GetLoopConstantEnter(cur_value)
        if enter_op:
          # Special case: cur_value comes from a constant Enter node.
          cur_value = enter_op.inputs[0]
          cur_grad_state = cur_grad_state.outer_grad_state
          if cur_grad_state is None:
            # We are now outside all nested loops for this gradient(),
            # so `value` is a loop invariant and there is no need to
            # save the history of value. Just make cur_value to enter
            # the right control flow context.
            real_value = self._grad_context.AddValue(cur_value)
            break
        elif constant_op.is_constant(cur_value):
          # If the value to be forwarded is a constant, clone the constant in
          # the gradient loop rather than using a stack.
          # TODO(phawkins): consider hoisting the constant out of the loop
          # instead.
          real_value = constant_op.constant(
              tensor_util.constant_value(cur_value), dtype=cur_value.dtype)
          break
        else:
          # Record the history of this value in forward_ctxt.
          self._grad_context.Exit()
          history_value = cur_grad_state.AddForwardAccumulator(cur_value)
          self._grad_context.Enter()
          break

      if real_value is None:
        # Add the stack pop op in the grad context.
        real_value = cur_grad_state.AddBackpropAccumulatedValue(
            history_value, cur_value)
        if cur_grad_state != self:
          real_value = self._grad_context.AddValue(real_value)
      self._history_map[value.name] = real_value
    return real_value


class _ControlFlowState:
  """Maintain the mapping from the loops to their grad states."""

  def __init__(self):
    self._map = {}  # maps forward loop context to _GradLoopState

  def GetGradState(self, op: ops.Operation, before):
    """Return the grad state for this op if it's in a forward loop context."""
    if before and util.IsLoopExit(op):
      forward_ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
      forward_ctxt = forward_ctxt.outer_context
      if forward_ctxt:
        forward_ctxt = forward_ctxt.GetWhileContext()
    else:
      forward_ctxt = util.GetWhileContext(op)
    if forward_ctxt:
      return self._map.get(forward_ctxt)
    return None

  def ProcessUnusedLoopExits(self, pending_count, to_ops_set):
    """Process all the "unused" loop exits.

    The "unused" exits of the loops are added to `unused_exits`. An exit is
    unused if its pending_count is 0. If there is an exit with real gradient,
    all these deferred exits will enter the backprop loop with zero gradient.
    Otherwise, they will enter the backprop loop with None. As an example,
    people often write:

    ```python
    v1, _ = tf.while_loop(p, b, [x1, x2])
    result = gradients(v1, x1)
    ```

    The exit node for x2 is not included by the betweenness analysis. But we
    need to backprop x2 if x2 is involved in computing v1.

    Args:
      pending_count: The number of backprop inputs for every op.
      to_ops_set: The set of ops for ys in gradients(ys, xs)

    Returns:
      The set of unused loop exits that we know at this point we need
      to backprop.
    """
    loop_exits = []
    for grad_state in self._map.values():
      for y in grad_state.forward_loop_exits:
        if pending_count[y.op] == 0:
          grad_state.pending_exits_count -= 1
          if y.op not in to_ops_set:
            grad_state.unused_exits.append(y)
          if grad_state.pending_exits_count == 0:
            loop_exits.extend(grad_state.unused_exits)
      # Need to include Enters in backprop for higher-order gradients.
      for y in grad_state.forward_context.loop_enters:
        if pending_count[y.op] == 0:
          pending_count[y.op] = 1
    return loop_exits

  def EnterGradWhileContext(self, op, before):
    """Enter the WhileContext for gradient computation."""
    grad_state = self.GetGradState(op, before)
    if grad_state:
      grad_state.grad_context.Enter()

  def ExitGradWhileContext(self, op, before):
    """Exit the WhileContext for gradient computation."""
    grad_state = self.GetGradState(op, before)
    if grad_state:
      grad_state.grad_context.Exit()

  def AddWhileContext(self, op, between_op_list, between_ops):
    """Add the grad state for the while loop that op belongs to.

    Note that op is an Exit, and this method must be called in
    the control flow context where gradients() is called.

    Note that this method modifies `between_op_list` and `between_ops`.
    """
    forward_ctxt = util.GetWhileContext(op)
    grad_state = self._map.get(forward_ctxt)
    if grad_state is None:
      # This is a new while loop so create a grad state for it.
      outer_forward_ctxt = forward_ctxt.outer_context
      if outer_forward_ctxt:
        outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
      outer_grad_state = None
      if outer_forward_ctxt:
        outer_grad_state = self._map.get(outer_forward_ctxt)
      grad_state = _GradLoopState(forward_ctxt, outer_grad_state)
      self._map[forward_ctxt] = grad_state

      # We need to include all exits of a loop for backprop.
      for loop_exit in grad_state.forward_loop_exits:
        if loop_exit.op not in between_ops:
          between_ops.add(loop_exit.op)
          between_op_list.append(loop_exit.op)

  def ZerosLikeForExit(self, val):
    """Create zeros_like gradient for a loop exit.

    If the result of a loop variable is not used but is involved in
    computing the result of some needed loop variable, we create a
    zero-valued tensor that is fed as gradient for the Exit node of that
    loop variable. Note that val.op is an Exit, and this method must be
    called in the control flow context where gradients() is called.

    Args:
      val: The output tensor of an Exit op.

    Returns:
      A zero tensor of the same shape of val.
    """
    val_shape = val.get_shape()
    forward_ctxt = val.op._get_control_flow_context()
    outer_forward_ctxt = forward_ctxt.outer_context
    if outer_forward_ctxt:
      outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
    outer_grad_state = None
    if outer_forward_ctxt:
      outer_grad_state = self._map.get(outer_forward_ctxt)
    if outer_grad_state:
      # This is a nested loop.
      if val_shape.is_fully_defined():
        # If the shape is known statically, just create a zero tensor
        # with the right shape in the right context.
        outer_grad_state.grad_context.Enter()
        result = array_ops.zeros(val_shape.dims, val.dtype)
        outer_grad_state.grad_context.Exit()
      else:
        # Only the shape of value is needed for backprop.
        forward_ctxt.outer_context.Enter()
        shape = array_ops.shape_internal(val, optimize=False)
        forward_ctxt.outer_context.Exit()
        # Save the shape to a stack.
        history_shape = outer_grad_state.AddForwardAccumulator(shape)
        # Get the shape back from the stack.
        outer_grad_ctxt = outer_grad_state.grad_context
        outer_grad_ctxt.Enter()
        real_shape = outer_grad_state.AddBackpropAccumulatedValue(
            history_shape, shape)
        result = array_ops.zeros(real_shape, val.dtype)
        outer_grad_ctxt.Exit()
    else:
      # This is not a nested loop.
      if val_shape.is_fully_defined():
        # If the shape is known statically, just create a zero tensor
        # with the right shape.
        result = array_ops.zeros(val_shape.dims, val.dtype)
      else:
        result = array_ops.zeros_like(val, optimize=False)
    return result

  def ZerosLikeV1WhileLoop(self, op, index):
    """Create zeros_like for the specified output of an op.

    If op is in a while loop that is part of gradients(), this method
    must be called in its grad loop context.

    Args:
      op: A tensorflow operation.
      index: the index for a specific output of the op.

    Returns:
      A zero tensor of the same shape of op.outputs[index].
    """
    if util.IsLoopSwitch(op):
      return None
    if op.graph.building_function:
      # The optimization here is tricky to apply to functions
      return array_ops.zeros_like(op.outputs[index])
    dead_branch = util.IsSwitch(op)
    forward_ctxt = util.GetWhileContext(op)
    grad_state = self._map.get(forward_ctxt)
    if grad_state is None:
      # op is not in a while loop that is part of gradients().
      return ZerosLike(op, index)
    op_ctxt = op._get_control_flow_context()
    val = ops.convert_to_tensor(op.outputs[index], name="tensor")
    shape = val.get_shape()
    if shape.is_fully_defined():
      # If the shape is known statically, just create a zero tensor with
      # the right shape in the grad loop context.
      if val.dtype == dtypes.resource:
        result = array_ops.zeros(
            resource_variable_ops.variable_shape(val),
            dtype=default_gradient.get_zeros_dtype(val))
      else:
        result = constant_op.constant(0, shape=shape.dims, dtype=val.dtype)
      if dead_branch:
        # op is a cond switch. Guard the zero tensor with a switch.
        pred = grad_state.history_map.get(op_ctxt.pred.name)
        branch = op_ctxt.branch
        result = control_flow_ops._SwitchRefOrTensor(result, pred)[1 - branch]
    else:
      # Unknown shape so keep a history of the shape at runtime.
      if dead_branch:
        # Need to add a special switch to guard the value.
        pred = op_ctxt.pred
        branch = op_ctxt.branch
        op_ctxt.outer_context.Enter()
        val = control_flow_ops._SwitchRefOrTensor(op.inputs[0],
                                                  pred)[1 - branch]
        zeros_shape = array_ops.shape_internal(val, optimize=False)
        op_ctxt.outer_context.Exit()
        val.op._set_control_flow_context(op_ctxt)
        zeros_shape.op._set_control_flow_context(op_ctxt)
      else:
        op_ctxt.Enter()
        zeros_shape = array_ops.shape_internal(val, optimize=False)
        op_ctxt.Exit()

      # Add forward accumulator for shape.
      grad_state.grad_context.Exit()
      history_zeros_shape = grad_state.AddForwardAccumulator(
          zeros_shape, dead_branch=dead_branch)
      grad_state.grad_context.Enter()

      # Create a zero tensor with the right shape.
      shape = grad_state.AddBackpropAccumulatedValue(history_zeros_shape,
                                                     zeros_shape, dead_branch)
      result = array_ops.zeros(shape, val.dtype)
    return result

  def PostProcessing(self):
    """Perform postprocessing at the end of gradients().

    We have created the gradient graph at this point. So this function
    can be used to perform any postprocessing on the gradient graph.
    We currently perform the following postprocessing:
      1. Patch the gradient graph if the output of a loop variable
         doesn't depend on its input.
    """
    for _, grad_state in self._map.items():
      for _, b_merge in grad_state.switch_map.items():
        if b_merge.op.inputs[0] == b_merge.op.inputs[1]:
          # The value of this loop variable at iteration i+1 doesn't
          # depend on its value at iteration i. So use zeros as the
          # gradients for all iterations > 0.
          dtype = b_merge.op.inputs[0].dtype
          shape = b_merge.op.inputs[0].get_shape()
          # pylint: disable=protected-access
          if shape.is_fully_defined():
            grad_state.grad_context.Enter()
            # Create a zeros and use it for iterations > 0.
            grad_val = constant_op.constant(0, dtype=dtype, shape=shape)
            next_grad_val = control_flow_ops._NextIteration(grad_val)
            grad_state.grad_context.Exit()
          else:
            # Create a zeros in the outer grad context.
            outer_grad_ctxt = grad_state.grad_context.outer_context
            if outer_grad_ctxt:
              outer_grad_ctxt.Enter()
            enter_grad_op = b_merge.op.inputs[0].op
            enter_grad = enter_grad_op.inputs[0]
            grad_shape = array_ops.shape_internal(enter_grad, optimize=False)
            grad_val = array_ops.zeros(grad_shape)
            if outer_grad_ctxt:
              outer_grad_ctxt.Exit()
            # Use the zeros for iterations > 0.
            grad_state.grad_context.Enter()
            next_grad_val = control_flow_ops._NextIteration(grad_val)
            grad_state.grad_context.Exit()
          b_merge.op._update_input(1, next_grad_val)
          # pylint: enable=protected-access


def MaybeCreateControlFlowState(between_op_list, between_ops,
                                colocate_gradients_with_ops):
  """Create the state for all the while loops involved in one gradients().

  We create a _ControlFlowState when there are while loops involved in
  gradients(). In gradients(), control flow logic is only invoked when
  the _ControlFlowState is not None.

  Note that this method modifies `between_op_list` and `between_ops`.
  """
  loop_state = None
  for op in between_op_list:
    if util.IsLoopExit(op):
      if loop_state is None:
        loop_state = _ControlFlowState()
      if colocate_gradients_with_ops:
        with ops.colocate_with(op):
          loop_state.AddWhileContext(op, between_op_list, between_ops)
      else:
        loop_state.AddWhileContext(op, between_op_list, between_ops)
  return loop_state


def _ZerosLikeV1(op, index):
  """Branch of ZerosLike for TF1."""
  val = op.outputs[index]
  op_ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  if op_ctxt:
    # We are in a cond context. Use a switch to create zeros only when needed.
    pred = op_ctxt.pred
    branch = op_ctxt.branch
    switch_val = control_flow_ops.switch(op.inputs[0], pred)[1 - branch]
    # A op is created along the branch taken as control dependencies are on
    # the whole op and not on the tensor output.
    pivot = array_ops.identity(switch_val)
    if val.dtype == dtypes.resource:
      with ops.control_dependencies([pivot]):
        return array_ops.zeros(
            gen_resource_variable_ops.variable_shape(switch_val),
            dtype=default_gradient.get_zeros_dtype(val))
    zeros_shape = array_ops.shape_internal(switch_val, optimize=False)
    # Ensure ops created within array_ops.zeros are dominated by switch in
    # cond context.
    with ops.control_dependencies([pivot]):
      return array_ops.zeros(zeros_shape, dtype=val.dtype)
  else:
    return array_ops.zeros_like(val, optimize=False)


def _ZerosLikeV2(op, index):
  """Branch of ZerosLike for TF2."""
  val = op.outputs[index]
  if val.dtype == dtypes.resource:
    return array_ops.zeros(
        gen_resource_variable_ops.variable_shape(val),
        dtype=default_gradient.get_zeros_dtype(val))
  if (isinstance(val.op.graph, control_flow_v2_func_graphs.WhileBodyFuncGraph)
      and val.dtype != dtypes.variant):
    # In while_v2 we do not want to add a `ZerosLike` op because that will
    # trigger accumulation of `val`. Normally `ZerosLike` is preferred because
    # it helps avoid creating extra nodes(possibly Consts) for the shape.
    # For variants, we must use ZerosLike.
    if val.shape.is_fully_defined():
      return constant_op.constant(0, shape=val.shape.dims, dtype=val.dtype)
    else:
      # Note: Even though we add `Shape` in the default graph, while_v2 is smart
      # enough to place it in the forward graph i.e. `val.graph`.
      zeros_shape = array_ops.shape_internal(val, optimize=False)
      return array_ops.zeros(zeros_shape, val.dtype)
  else:
    return array_ops.zeros_like(val, optimize=False)


def ZerosLike(op, index):
  """Create zeros_like for the specified output of an op."""
  if not util.IsSwitch(op):
    return _ZerosLikeV2(op, index)
  else:
    return _ZerosLikeV1(op, index)
