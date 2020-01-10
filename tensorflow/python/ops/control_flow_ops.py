"""## Control Flow Operations

TensorFlow provides several operations and classes that you can use to control
the execution of operations and add conditional dependencies to your graph.

@@identity
@@tuple
@@group
@@no_op
@@count_up_to

## Logical Operators

TensorFlow provides several operations that you can use to add logical operators
to your graph.

@@logical_and
@@logical_not
@@logical_or
@@logical_xor

## Comparison Operators

TensorFlow provides several operations that you can use to add comparison
operators to your graph.

@@equal
@@not_equal
@@less
@@less_equal
@@greater
@@greater_equal
@@select
@@where

## Debugging Operations

TensorFlow provides several operations that you can use to validate values and
debug your graph.

@@is_finite
@@is_inf
@@is_nan
@@verify_tensor_all_finite
@@check_numerics
@@add_check_numerics_ops
@@Assert
@@Print
"""
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import types
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.gen_control_flow_ops import *


# We override the 'tuple' for a control flow op, so we keep python's
# existing 'tuple' for later use in this module.
_basetuple = tuple


# pylint: disable=protected-access
def _Identity(data, name=None):
  """Return a tensor with the same shape and contents as the input tensor.

  Args:
    data: A Tensor.
    name: A name for this operation (optional).

  Returns:
    A Tensor with the same type and value as the input Tensor.
  """
  if not data.dtype.is_ref_dtype:
    return array_ops.identity(data, name=name)
  else:
    return gen_array_ops._ref_identity(data, name=name)


def _Enter(data, frame_name, is_constant=False, parallel_iterations=10,
           name=None):
  """Creates or finds a child frame, and makes 'data' available to it.

  The unique `frame_name` is used by the `Executor` to identify frames. If
  `is_constant` is true, `output` is a constant in the child frame; otherwise
  it may be changed in the child frame. At most `parallel_iterations` iterations
  are run in parallel in the child frame.

  Args:
    data: The tensor to be made available to the child frame.
    frame_name: The name of the child frame.
    is_constant: If true, the output is constant within the child frame.
    parallel_iterations: The number of iterations allowed to run in parallel.
    name: A name for this operation (optional).

  Returns:
    The same tensor as 'data'.
  """
  if not data.dtype.is_ref_dtype:
    return enter(data, frame_name, is_constant, parallel_iterations,
                 name=name)
  else:
    return ref_enter(data, frame_name, is_constant, parallel_iterations,
                     name=name)


def exit(data, name=None):
  """Exits the current frame to its parent frame.

  Exit makes its input `data` available to the parent frame.

  Args:
    data: The tensor to be made available to the parent frame.
    name: A name for this operation (optional).

  Returns:
    The same tensor as `data`.
  """
  return gen_control_flow_ops._exit(data, name)


def switch(data, pred, name=None):
  """Forwards `data` to an output determined by `pred`.

  If `pred` is true, the `data` input is forwared to the first output.
  Otherwise, the data goes to the second output.

  This op handles `Tensor`s and `IndexedSlices`.

  Args:
    data: The tensor to be forwarded to the appropriate output.
    pred: A scalar that specifies which output port will receive data.
    name: A name for this operation (optional).

  Returns:
    `(output_true, output_false)`: If `pred` is true, data will be forwarded to
    `output_true`, otherwise it goes to `output_false`.
  """
  with ops.op_scope([data, pred], name, "Switch") as name:
    data = ops.convert_to_tensor_or_indexed_slices(data, name="data")
    pred = ops.convert_to_tensor(pred, name="pred")
    if isinstance(data, ops.Tensor):
      return gen_control_flow_ops._switch(data, pred, name=name)
    else:
      val, ind, dense_shape = data.values, data.indices, data.dense_shape
      val_f, val_t = gen_control_flow_ops._switch(val, pred, name=name)
      ind_f, ind_t = gen_control_flow_ops._switch(ind, pred, name="indices")
      if dense_shape:
        dense_shape_f, dense_shape_t = gen_control_flow_ops._switch(
            dense_shape, pred, name="dense_shape")
      else:
        dense_shape_f, dense_shape_t = None, None
      return (ops.IndexedSlices(val_f, ind_f, dense_shape_f),
              ops.IndexedSlices(val_t, ind_t, dense_shape_t))


def merge(inputs, name=None):
  """Returns the value of an available element of `inputs`.

  This op tests each of the tensors in `inputs` in turn to determine if any of
  them is available. If it finds an available tensor, it returns it and its
  index in `inputs`.

  It is an error if more than one tensor in `inputs` is available. If no tensor
  in `inputs` is available, the returned tensor and index are not set.

  This op handles both `Tensor`s and `IndexedSlices`. If inputs has a mix of
  `Tensor`s and `IndexedSlices`, all inputs are converted to IndexedSlices
  before merging.

  Args:
    inputs: The input tensors, at most one of which is available.
    name: A name for this operation (optional).

  Returns:
    A tuple containing the chosen input tensor and its index in `inputs`.

  Raises:
    ValueError: If inputs are IndexedSlices and some but not all have a
      dense_shape property.
  """
  with ops.op_scope(inputs, name, "Merge") as name:
    inputs = [ops.convert_to_tensor_or_indexed_slices(inp) for inp in inputs]
    if all([isinstance(inp, ops.Tensor) for inp in inputs]):
      return gen_control_flow_ops._merge(inputs, name=name)
    else:
      inputs = math_ops._as_indexed_slices_list(inputs)
      values, _ = gen_control_flow_ops._merge([inp.values for inp in inputs],
                                              name=name)
      indices, chosen_index = gen_control_flow_ops._merge(
          [inp.indices for inp in inputs], name="indices")
      if any(inp.dense_shape for inp in inputs):
        if not all(inp.dense_shape for inp in inputs):
          raise ValueError("Either all merged IndexedSlices must have a "
                           "dense_shape, or none must have a dense_shape.")
        dense_shape, _ = gen_control_flow_ops._merge(
            [inp.dense_shape for inp in inputs], name="dense_shape")
      else:
        dense_shape = None
      return ops.IndexedSlices(values, indices, dense_shape), chosen_index


def _SwitchRefOrTensor(data, pred, name="Switch"):
  """Forwards `data` to an output determined by `pred`.

  If `pred` is true, the `data` input is forwared to the first output.
  Otherwise, the data goes to the second output.

  This op handles `Tensor`s and `IndexedSlices`.

  Args:
    data: The tensor to be forwarded to the appropriate output.
    pred: A scalar that specifies which output port will receive data.
    name: A name for this operation (optional).

  Returns:
    `(output_false, output_false)`: If `pred` is true, data will be forwarded to
    `output_true`, otherwise it goes to `output_false`.

  Raises:
    TypeError: if data is not a Tensor or IndexedSlices
  """
  data = ops.convert_to_tensor_or_indexed_slices(data, name="data")
  if isinstance(data, ops.Tensor):
    if not data.dtype.is_ref_dtype:
      return switch(data, pred, name=name)
    else:
      return ref_switch(data, pred, name=name)
  else:
    return switch(data, pred, name=name)


class ControlFlowOpInputs(object):
  """An indirection to capture the input tensors needed in backprop."""

  def __init__(self, op):
    self._op = op
    self._inputs = None

  def __len__(self):
    return len(self._op._inputs)

  def __getitem__(self, index):
    if self._inputs is None:
      self._inputs = [None for _ in self._op.inputs]
    if isinstance(index, int):
      val = self._inputs[index]
      if val is None:
        f_val = self._op.inputs[index]
        val = _GetRealValue(f_val)
        self._inputs[index] = val
      return val
    elif isinstance(index, slice):
      start, stop, step = index.indices(len(self))
      vals = [self[i] for i in xrange(start, stop, step)]
      return vals
    else:
      raise TypeError("index must be an integer or slice")


class ControlFlowOpOutputs(object):
  """An indirection to capture the output tensors needed in backprop."""

  def __init__(self, op):
    self._op = op
    self._outputs = None

  def __len__(self):
    return len(self._op._outputs)

  def __getitem__(self, index):
    if self._outputs is None:
      self._outputs = [None for _ in self._op.outputs]
    if isinstance(index, int):
      val = self._outputs[index]
      if val is None:
        f_val = self._op.outputs[index]
        val = _GetRealValue(f_val)
        self._outputs[index] = val
      return val
    elif isinstance(index, slice):
      start, stop, step = index.indices(len(self))
      vals = [self[i] for i in xrange(start, stop, step)]
      return vals
    else:
      raise TypeError("index must be an integer or slice")


class ControlFlowOpWrapper(object):
  """A wrapper class for Operation."""

  def __init__(self, op):
    self._op = op
    self._inputs = None
    self._outputs = None

  @property
  def inputs(self):
    if self._inputs is None:
      self._inputs = ControlFlowOpInputs(self._op)
    return self._inputs

  @property
  def outputs(self):
    if self._outputs is None:
      self._outputs = ControlFlowOpOutputs(self._op)
    return self._outputs

  @property
  def op(self):
    return self._op

  @property
  def name(self):
    """Returns the name of this instance of op."""
    return self._op.name

  @property
  def _id(self):
    """Returns the unique id of this operation."""
    return self._op._id

  @property
  def device(self):
    """Returns the device of this operation.

    Returns:
      a string or None if the device was not set.
    """
    return self._op.device

  @property
  def output_types(self):
    return self._op.output_types

  @property
  def input_types(self):
    return self._op._input_types

  @property
  def type(self):
    """Returns the type of the op."""
    return self._op.type

  @property
  def graph(self):
    """Returns the parent graph."""
    return self._op.graph

  def GetAttr(self, attr_name):
    """Returns the value of attribute 'attr_name' of NodeDef."""
    return self._op.get_attr(attr_name)

  def _get_control_flow_context(self):
    return self._op._get_control_flow_context()


def GetRealOp(op):
  while isinstance(op, ControlFlowOpWrapper):
    op = op.op
  return op


def MakeWrapper(op):
  """Make a wrapper for op if it is in a WhileContext."""
  forward_ctxt = op._get_control_flow_context()
  if forward_ctxt and isinstance(forward_ctxt, WhileContext):
    return ControlFlowOpWrapper(op)
  return op


def EnterGradWhileContext(op):
  """Enter the WhileContext for gradient computation."""
  forward_ctxt = op._get_control_flow_context()
  if forward_ctxt and isinstance(forward_ctxt, WhileContext):
    grad_ctxt = forward_ctxt.CreateGradWhileContext()
    grad_ctxt.Enter()


def ExitGradWhileContext(op):
  """Exit the WhileContext for gradient computation."""
  forward_ctxt = op._get_control_flow_context()
  if forward_ctxt and isinstance(forward_ctxt, WhileContext):
    assert forward_ctxt.grad_context
    forward_ctxt.grad_context.Exit()


def _GetRealValue(value):
  """Get the real value.

  If backprop "uses" a value produced by forward inference, an
  accumulator is added in the forward loop to accumulate its values,
  so we use the accumulated value, indexed by the backprop counter.

  Args:
    value: A tensor to be captured.

  Returns:
    The same tensor value from the saved history.
  """
  real_value = value
  forward_ctxt = value.op._get_control_flow_context()
  real_value = forward_ctxt.history_map.get(value.name)
  assert value.op.type != "Variable"
  if real_value is None:
    if value.op.type == "Enter" and value.op.get_attr("is_constant"):
      # Use the input of this Enter node
      real_value = GetRealOp(value.op).inputs[0]
    else:
      # Accumulate the history of this value.
      # NOTE(yuanbyu): Don't accumulate for constants. One approach is
      # to deepcopy the constants for the grad while context.
      history_value = forward_ctxt.AddForwardAccumulateLoop(value)

      # The shapes of the whole history and a single event element.
      forward_ctxt.grad_context.Exit()
      elem_rank = array_ops.rank(history_value) - 1
      elem_rank_vec = array_ops.expand_dims(elem_rank, 0)
      elem_shape = array_ops.slice(array_ops.shape(history_value), [1],
                                   elem_rank_vec)
      slice_shape = array_ops.concat(0, [[1], elem_shape])
      forward_ctxt.grad_context.Enter()

      # The begin position of the slice at slice_index.
      slice_index = forward_ctxt.grad_context.index
      b1 = array_ops.zeros(elem_rank_vec, dtype=types.int32)
      b = array_ops.concat(0, [array_ops.expand_dims(slice_index, 0), b1])

      # The slice at slice_index.
      # TODO(irving): Replace with gather once that's GPU accelerated
      real_value = array_ops.squeeze(
          array_ops.slice(history_value,
                          b,
                          slice_shape,
                          name="real"),
          squeeze_dims=[0])
  forward_ctxt.history_map[value.name] = real_value
  return real_value


def IsLoopSwitch(op):
  """Returns true if `op` is the Switch for a While loop."""
  if op.type == "Switch":
    ctxt = op._get_control_flow_context()
    return ctxt and isinstance(ctxt, WhileContext)
  return False


class ControlFlowContext(object):
  """The base class for control flow context.

  The usage pattern is a sequence of (Enter, Exit) followed by a final
  ExitResult.
  """

  def AddName(self, name):
    self._values.add(name)

  # pylint: disable=protected-access
  def Enter(self):
    """Enter the current context."""
    self._outer_context = ops.get_default_graph()._get_control_flow_context()
    ops.get_default_graph()._set_control_flow_context(self)

  def Exit(self):
    """Exit the current context."""
    ops.get_default_graph()._set_control_flow_context(self._outer_context)
  # pylint: enable=protected-access

  def ExitResult(self, result):
    """Make a list of tensors available in the outer context."""
    if self._outer_context is not None:
      for x in result:
        self._outer_context.AddName(x.name)

  def GetWhileContext(self):
    """Get the current while context."""
    if self._outer_context is not None:
      return self._outer_context.GetWhileContext()
    return None

  def AddToWhileContext(self, op):
    """Add a control dependency to the containing WhileContext.

    The added control dependency ensures that the outputs of this op
    belong to the WhileContext.

    Args:
      op: An operation.
    """
    while_ctxt = self.GetWhileContext()
    if while_ctxt is not None:
      # pylint: disable=protected-access
      op._add_control_input(while_ctxt.GetControlPivot().op)
      # pylint: enable=protected-access


class CondContext(ControlFlowContext):
  """The context for the conditional construct."""

  def __init__(self, pred, pivot, branch):
    self._pred = pred
    self._outer_context = None
    self._pivot = pivot
    self._branch = branch
    self._values = set()
    self._values.add(pred.name)
    self._values.add(pivot.name)
    self._external_values = {}

  @property
  def pred(self):
    return self._pred

  @property
  def pivot(self):
    return self._pivot

  @property
  def branch(self):
    return self._branch

  def AddValue(self, val):
    """Add 'val' to the current context and its outer context recursively."""
    result = val
    if val.name not in self._values:
      self._values.add(val.name)
      if self._outer_context is not None:
        result = self._outer_context.AddValue(val)
      result = with_dependencies([self._pivot], result)
      self._external_values[val.name] = result
    return result

  def AddOp(self, op):
    """Add 'op' to the current context."""
    if not op.inputs:
      # Add this op to the enclosing while context
      self.AddToWhileContext(op)
      # pylint: disable=protected-access
      op._add_control_input(self._pivot.op)
      # pylint: enable=protected-access
      for x in op.outputs:
        self._values.add(x.name)
    else:
      for index in range(len(op.inputs)):
        x = op.inputs[index]
        if x.name not in self._values:
          self._values.add(x.name)
          # Add this value to the parent contexts up to the context that
          # creates this value.
          real_x = x
          if self._outer_context is not None:
            real_x = self._outer_context.AddValue(x)
          real_x = _SwitchRefOrTensor(real_x, self._pred)[self._branch]
          self._external_values[x.name] = real_x
        x = self._external_values.get(x.name)
        if x is not None:
          op._update_input(index, x)
      for x in op.outputs:
        self._values.add(x.name)

  def BuildCondBranch(self, fn):
    """Add the subgraph defined by fn() to the graph."""
    r = fn()
    result = []
    if r is not None:
      if not isinstance(r, list) and not isinstance(r, _basetuple):
        r = [r]
      for v in r:
        if isinstance(v, ops.Operation):
          v = with_dependencies([v], self._pivot)
        elif v.name not in self._values:
          self._values.add(v.name)
          if self._outer_context is not None:
            v = self._outer_context.AddValue(v)
          v = _SwitchRefOrTensor(v, self._pred)[self._branch]
        else:
          external_v = self._external_values.get(v.name)
          if external_v is not None:
            v = external_v
        result.append(v)
    return result


def cond(pred, fn1, fn2, name=None):
  """Return either 'fn1()' or 'fn2()' based on the boolean predicate 'pred'.

  `fn1` and `fn2` both return lists of output tensors. `fn1` and `fn2` must have
  the same number and type of outputs.

  Args:
    pred: A scalar determining whether to return the result of `fn1` or `fn2`.
    fn1: The function to be performed if pred is true.
    fn2: The function to be performed if pref is false.
    name: Optional name prefix for the returned tensors.

  Returns:
    Tensors returned by the call to either `fn1` or `fn2`. If the functions
    return a singleton list, the element is extracted from the list.

  Raises:
    TypeError: if `fn1` or `fn2` is not callable.
    ValueError: if `fn1` and `fn2` do not return the same number of tensors, or
                return tensors of different types.

  Example:
  ```python
    x = constant(2)
    y = constant(5)
    def f1(): return constant(17)
    def f2(): return constant(23)
    r = cond(math_ops.less(x, y), f1, f2)
    # r is set to f1()
  ```
  """
  with ops.op_scope([pred], name, "Cond") as name:
    if not callable(fn1):
      raise TypeError("fn1 must be callable.")
    if not callable(fn2):
      raise TypeError("fn2 must be callable.")

    # Add the Switch to the graph.
    p_2, p_1 = switch(pred, pred)
    pivot_1 = array_ops.identity(p_1, name="switch_t")
    pivot_2 = array_ops.identity(p_2, name="switch_f")
    pred = array_ops.identity(pred, name="pred_id")

    # Build the graph for the true branch in a new context.
    context_t = CondContext(pred, pivot_1, 1)
    context_t.Enter()
    res_t = context_t.BuildCondBranch(fn1)
    context_t.ExitResult(res_t)
    context_t.Exit()

    # Build the graph for the false branch in a new context.
    context_f = CondContext(pred, pivot_2, 0)
    context_f.Enter()
    res_f = context_f.BuildCondBranch(fn2)
    context_t.ExitResult(res_f)
    context_f.Exit()

    # Add the final merge to the graph.
    if len(res_t) != len(res_f):
      raise ValueError("fn1 and fn2 must return the same number of tensors.")
    for x, y in zip(res_f, res_t):
      assert ((isinstance(x, ops.IndexedSlices) and
               isinstance(y, ops.IndexedSlices)) or
              (isinstance(x, ops.Tensor) and isinstance(y, ops.Tensor)))
      val_x = x if isinstance(x, ops.Tensor) else x.values
      val_y = y if isinstance(y, ops.Tensor) else y.values
      if val_x.dtype.base_dtype != val_y.dtype.base_dtype:
        raise ValueError("Outputs of fn1 and fn2 must have the same type: "
                         "%s, %s" % (val_x.dtype.name, val_y.dtype.name))
    merges = [merge([x[0], x[1]])[0] for x in zip(res_f, res_t)]
    return merges[0] if len(merges) == 1 else merges


# TODO(yuanbyu): We should probably separate the notion of context so it
# could be used not only for conditionals and loops but also subgraphs.
class WhileContext(ControlFlowContext):
  """The context for the loop construct."""

  def __init__(self, parallel_iterations, back_prop, name):
    self._name = ops.get_default_graph().unique_name(name)
    self._parallel_iterations = parallel_iterations
    self._back_prop = back_prop
    self._outer_context = None
    # We use this node to control constants created by the pred lambda.
    self._pivot_for_pred = None
    # We use this node to control constants created by the body lambda.
    self._pivot_for_body = None
    # The boolean tensor for loop termination condition. Used in code
    # generation for gradient computation
    self._pivot = None

    # The tensors for the counters added by AddForwardCounterLoop or
    # AddBackPropCounterLoop
    self._index = None

    # Information needed by backprop
    self._grad_context = None
    self._total_iterations = None
    self._history_map = {}
    self._switch_map = {}

    # values considered to have been already seen in this context
    self._values = set()

    # values referenced by but external to this context
    self._external_values = {}

  @property
  def name(self):
    return self._name

  @property
  def parallel_iterations(self):
    """The number of iterations allowed to run in parallel."""
    return self._parallel_iterations

  @property
  def back_prop(self):
    """True iff backprop is enabled for this While loop."""
    return self._back_prop

  @property
  def pivot(self):
    """The boolean tensor representing the loop termination condition."""
    return self._pivot

  @property
  def index(self):
    """The loop index representing the current iteration."""
    return self._index

  @property
  def grad_context(self):
    """The corresponding WhileContext for gradient."""
    return self._grad_context

  @property
  def history_map(self):
    """The map that records all the tensors needed for backprop."""
    return self._history_map

  @property
  def switch_map(self):
    """The map that records all the Switch ops in the While loop."""
    return self._switch_map

  @property
  def total_iterations(self):
    """The total number of iterations of the while loop."""
    return self._total_iterations

  def GetWhileContext(self):
    return self

  def GetControlPivot(self):
    if self._pivot_for_body:
      return self._pivot_for_body
    return self._pivot_for_pred

  def AddValue(self, val):
    """Add 'val' to the current context and its outer context recursively."""
    result = val
    if val.name not in self._values:
      self._values.add(val.name)
      if self._outer_context is not None:
        result = self._outer_context.AddValue(val)
      # Create an Enter that makes 'result' known to this context.
      enter = _Enter(result, self._name, is_constant=True,
                     parallel_iterations=self._parallel_iterations)
      self._values.add(enter.name)
      self._external_values[val.name] = enter
      result = enter
    else:
      actual_val = self._external_values.get(val.name)
      if actual_val is not None:
        result = actual_val
    return result

  def AddOp(self, op):
    """Adds 'op' to the current context."""
    if not op.inputs:
      if not op.control_inputs:
        # Add a control edge from the control pivot to this op.
        # pylint: disable=protected-access
        op._add_control_input(self.GetControlPivot().op)
        # pylint: enable=protected-access
      else:
        # Control edges must be in the same context.
        for x in op.control_inputs:
          assert x._get_control_flow_context() == self, (
              "Control inputs must come from Operations in the same while "
              "loop context (not an outer context).")
      for x in op.outputs:
        self._values.add(x.name)
    else:
      for index in range(len(op.inputs)):
        x = op.inputs[index]
        self.AddValue(x)
        real_x = self._external_values.get(x.name)
        if real_x is not None:
          op._update_input(index, real_x)
          # Add a control dependency to prevent loop invariants from
          # enabling ops that should not be executed.
          if real_x.op.type == "RefEnter" and real_x.op.get_attr("is_constant"):
            # pylint: disable=protected-access
            op._add_control_input(self.GetControlPivot().op)
            # pylint: enable=protected-access
      for x in op.outputs:
        self._values.add(x.name)

  def CreateGradWhileContext(self):
    """Creates the WhileContext for backprop gradient computation."""
    if self._grad_context is None:
      cnt = self.AddForwardCounterLoop()
      self._grad_context = WhileContext(self._parallel_iterations,
                                        self._back_prop, self._name)
      self._grad_context.AddBackPropCounterLoop(cnt)
    return self._grad_context

  def AddForwardCounterLoop(self):
    """Adds a loop that counts the number of iterations.

    This is added to the forward loop at the time when we start to
    create the loop for backprop gradient computation.

    The pseudocode is:
      `n = 0; while (_pivot) { n++; }`

    Returns:
      The number of iterations taken by the forward loop.
    """
    n = constant_op.constant(0, name="f_count")
    self.Enter()
    self.AddName(n.name)
    enter_n = _Enter(n, self._name, is_constant=False,
                     parallel_iterations=self._parallel_iterations,
                     name="f_count")
    merge_n = merge([enter_n, enter_n])[0]
    switch_n = switch(merge_n, self._pivot)
    self._index = switch_n[1]

    add_n = math_ops.add(self._index, 1)
    next_n = next_iteration(add_n)
    merge_n.op._update_input(1, next_n)

    self._total_iterations = exit(switch_n[0], name="f_count")
    self.Exit()
    return self._total_iterations

  def AddForwardAccumulateLoop(self, value):
    """Add an accumulation loop for each value needed in backprop.

    This is added to the forward loop at the first time when a value
    in the forward loop is used by backprop gradient computation loop.

    The pseudocode is:
    ```
      acc;
      while (_pivot) {
        if (index == 0) [value] else Concat(acc, [value]);
      }
    ```

    Args:
      value: The tensor that is accumulated.

    Returns:
      The accumulated history of value.

    Raises:
      ValueError: If the shape of "value" is not known statically.
    """
    if not value.get_shape().is_fully_defined():
      raise ValueError("Must have known shape: %s" % value)
    self._grad_context.Exit()
    # TODO(irving): Now that acc starts out empty, most of the
    # conditional logic can go away.
    acc = constant_op.constant([],
                               value.dtype,
                               shape=[0] + value.get_shape().as_list(),
                               name="f_acc")
    self.Enter()
    self.AddName(acc.name)
    enter_acc = _Enter(acc, self._name, is_constant=False,
                       parallel_iterations=self._parallel_iterations,
                       name="f_acc")
    merge_acc = merge([enter_acc, enter_acc])[0]
    switch_acc = switch(merge_acc, self._pivot)

    # If index = 0 then [value] else Concat(acc, [value]).
    cond = math_ops.greater(self._index, 0)
    switch_add_acc = switch(switch_acc[1], cond)
    expand_value = array_ops.expand_dims(value, 0)
    true_branch = array_ops.concat(0, [switch_add_acc[1], expand_value])
    false_branch = array_ops.identity(switch_add_acc[0])
    false_branch = with_dependencies([false_branch], expand_value)
    add_acc = merge([false_branch, true_branch])[0]

    next_acc = next_iteration(add_acc)
    merge_acc.op._update_input(1, next_acc)

    exit_acc = exit(switch_acc[0], name="f_acc")
    self.Exit()
    self._grad_context.Enter()
    return exit_acc

  def AddForwardAccumulateCondLoop(self, value):
    """Add an accumulation loop for each conditional switch.

    This is added to the forward loop at the first time when a conditional
    switch in the forward loop is used by backprop gradient computation loop.

    The pseudocode is:
      ```
      acc;
      while (_pivot) {
        Concat(acc, value);
      }
      ```

    Args:
      value: The boolean tensor that is accumulated.

    Returns:
      The accumulated history of value.
    """
    self._grad_context.Exit()
    acc = constant_op.constant(False, name="f_acc")
    self.Enter()
    self.AddName(acc.name)
    enter_acc = _Enter(acc, self._name, is_constant=False,
                       parallel_iterations=self._parallel_iterations,
                       name="f_acc")
    merge_acc = merge([enter_acc, enter_acc])[0]
    switch_acc = switch(merge_acc, self._pivot)
    acc = array_ops.concat(0, [switch_add_acc[1], value])
    next_acc = next_iteration(acc)
    merge_acc.op._update_input(1, next_acc)

    exit_acc = exit(switch_acc[0], name="f_acc")
    self.Exit()
    self._grad_context.Enter()
    return exit_acc

  def AddBackPropCounterLoop(self, count):
    """Add the backprop loop that controls the iterations.

    This is added to the backprop loop. It is used to control the loop
    termination and the slice index.

    The pseudocode is:
      `n = count; while (n >= 1) { n--; }`

    Args:
      count: The number of iterations for backprop.

    Returns:
      always 0.
    """
    one = constant_op.constant(1, name="b_count")
    self.Enter()
    self.AddName(count.name)
    enter_count = _Enter(count, self._name, is_constant=False,
                         parallel_iterations=self._parallel_iterations,
                         name="b_count")
    merge_count = merge([enter_count, enter_count])[0]
    self._pivot_for_pred = merge_count

    cond = math_ops.greater_equal(merge_count, one)
    self._pivot = loop_cond(cond, name="b_count")
    switch_count = switch(merge_count, self._pivot)

    # Add next_iteration right after Switch to match the gradient function.
    next_count = next_iteration(switch_count[1])
    self._pivot_for_body = next_count
    self._index = math_ops.sub(next_count, one)
    merge_count.op._update_input(1, self._index)

    exit_count = exit(switch_count[0], name="b_count")
    self.Exit()
    return exit_count

  def AddBackPropAccumulateLoop(self, value):
    """Add an accumulation loop for every loop invariant.

    This is added to the backprop loop. It is used to accumulate partial
    gradients for each loop iteration. Called when in the while context
    for gradient.

    The pseudocode is:
      ```
      acc = 0;
      while (_pivot) {
        acc += value;
      }
      ```

    Args:
      value: The partial gradient of an iteration for a loop invariant.

    Returns:
      The gradient for a loop invariant.
    """
    self.Exit()
    acc = constant_op.constant(0, value.dtype, name="b_acc")
    self.Enter()
    self.AddName(acc.name)
    enter_acc = _Enter(acc, self._name, is_constant=False,
                       parallel_iterations=self._parallel_iterations,
                       name="b_acc")
    merge_acc = merge([enter_acc, enter_acc], name="b_acc")[0]
    switch_acc = switch(merge_acc, self._pivot)

    next_acc = next_iteration(switch_acc[1])
    add_acc = math_ops.add(next_acc, value)
    merge_acc.op._update_input(1, add_acc)

    exit_acc = exit(switch_acc[0], name="b_acc")
    return exit_acc

  def BuildLoop(self, pred, body, loop_vars):
    """Add the loop termination condition and body to the graph."""

    loop_vars = ops.convert_n_to_tensor_or_indexed_slices(loop_vars)
    # Let the context know the loop variabes so the _Enter nodes below
    # would be added into the context correctly.
    self._values = set([x.name for x in loop_vars])
    if self._outer_context is not None:
      real_vars = [self._outer_context.AddValue(x) for x in loop_vars]
    else:
      real_vars = loop_vars
    enter_vars = [_Enter(x, self._name, is_constant=False,
                         parallel_iterations=self._parallel_iterations)
                  for x in real_vars]
    self._values = set([x.name for x in enter_vars])

    merge_vars = [merge([x, x])[0] for x in enter_vars]
    self._pivot_for_pred = merge_vars[0]

    # Build the graph for pred.
    c = ops.convert_to_tensor(pred(*merge_vars))
    self._pivot = loop_cond(c, name="LoopCond")
    switch_vars = [_SwitchRefOrTensor(x, self._pivot) for x in merge_vars]

    # Build the graph for body.
    vars_for_body = [_Identity(x[1]) for x in switch_vars]
    self._pivot_for_body = vars_for_body[0]

    body_result = body(*vars_for_body)
    if not isinstance(body_result, (list, _basetuple)):
      body_result = [body_result]
    result = ops.convert_n_to_tensor_or_indexed_slices(body_result)
    next_vars = [next_iteration(x) for x in result]

    # Add the back edges to complete the loop.
    assert len(merge_vars) == len(next_vars)
    for x in zip(merge_vars, next_vars):
      x[0].op._update_input(1, x[1])

    # Add the exit ops.
    exit_vars = [exit(x[0]) for x in switch_vars]

    for m_var, n_var, e_var in zip(merge_vars, next_vars, exit_vars):
      if m_var.get_shape().is_compatible_with(n_var.get_shape()):
        e_var.set_shape(m_var.get_shape().merge_with(n_var.get_shape()))

    # Exit the loop.
    self.ExitResult(exit_vars)
    self.Exit()
    return exit_vars[0] if len(exit_vars) == 1 else exit_vars


def While(cond, body, loop_vars, parallel_iterations=10, back_prop=True,
          name=None):
  """Repeat `body` while the condition `cond` is true.

  `cond` is a function taking a list of tensors and returning a boolean scalar
  tensor. `body` is a function taking a list of tensors and returning a list of
  tensors of the same length and with the same types as the input. `loop_vars`
  is a list of tensors that is passed to both `cond` and `body`.

  While `cond` evaluates to true, `body` is executed.

  Args:
    cond: The termination condition of the loop.
    body: A function that represents the loop body.
    loop_vars: The list of variable input tensors.
    parallel_iterations: The number of iterations allowed to run in parallel.
    back_prop: Whether backprop is enabled for this while loop.
    name: Optional name prefix for the returned tensors.

  Returns:
    The output tensors for the loop variables after the loop.

  Raises:
    TypeError: if `cond` or `body` is not callable.
    ValueError: if `loop_var` is empty.

  Example:
    ```python
    i =  Constant(0)
    c = lambda i: math_ops.less(i, 10)
    b = lambda i: math_ops.add(i, 1)
    r = While(c, b, [i])
    ```
  """
  with ops.op_scope(loop_vars, name, "While") as name:
    if not loop_vars:
      raise ValueError("No loop variables provided")
    if not callable(cond):
      raise TypeError("cond must be callable.")
    if not callable(body):
      raise TypeError("body must be callable.")

    context = WhileContext(parallel_iterations, back_prop, name)
    context.Enter()
    return context.BuildLoop(cond, body, loop_vars)


def _AsTensorList(x, p):
  """Return x as a list of Tensors or IndexedSlices.

  For entries of `x` that are Operations, this returns an Identity of `p`
  with a dependency on the operation.

  Args:
    x: A Tensor/IndexedSlices/Operation or a list or tuple of them.
    p: A Tensor to return for entries in `x` that are Operations.

  Returns:
    A list of Tensors or IndexedSlices.
  """
  if not isinstance(x, list) and not isinstance(x, _basetuple):
    x = [x]

  l = []
  for v in x:
    if isinstance(v, ops.Operation):
      v = with_dependencies([v], p)
    v = ops.convert_to_tensor_or_indexed_slices(v)
    if isinstance(v, ops.Tensor):
      l.append(array_ops.identity(v))
    else:
      l.append(ops.IndexedSlices(array_ops.identity(v.values),
                                 array_ops.identity(v.indices)))
  return l


def _CheckResults(a, b):
  assert len(a) == len(b), (
      "Values returned by a() and b() must have the same length.")
  for x, y in zip(a, b):
    assert x.dtype == y.dtype, (
        "Values returned by a() [%s] and b() [%s] must have "
        "the same type: %s, %s." %
        (x.name, y.name, x.dtype.name, y.dtype.name))


def with_dependencies(dependencies, output_tensor, name=None):
  """Produces the content of `output_tensor` only after `dependencies`.

  In some cases, a user may want the output of an operation to be
  consumed externally only after some other dependencies have run
  first. This function ensures returns `output_tensor`, but only after all
  operations in `dependencies` have run. Note that this means that there is
  no guarantee that `output_tensor` will be evaluated after any `dependencies`
  have run.

  See also `tuple` and `group`.

  Args:
    dependencies: A list of operations to run before this op finishes.
    output_tensor: A `Tensor` or `IndexedSlices` that will be returned.
    name: (Optional) A name for this operation.

  Returns:
    Same as `output_tensor`.

  Raises:
    TypeError: if `output_tensor` is not a `Tensor` or `IndexedSlices`.
  """
  with ops.op_scope(dependencies + [output_tensor], name,
                   "control_dependency") as name:
    with ops.device(output_tensor.device
                    or ops.get_default_graph().get_default_device()):
      with ops.control_dependencies(dependencies):
        output_tensor = ops.convert_to_tensor_or_indexed_slices(output_tensor)
        if isinstance(output_tensor, ops.Tensor):
          return _Identity(output_tensor, name=name)
        else:
          return ops.IndexedSlices(_Identity(output_tensor.values, name=name),
                                   output_tensor.indices,
                                   output_tensor.dense_shape)


def _GroupControlDeps(dev, deps, name=None):
  with ops.control_dependencies(deps):
    if dev is None:
      return no_op(name=name)
    else:
      with ops.device(dev):
        return no_op(name=name)


# TODO(touts): Accept "inputs" as a list.
def group(*inputs, **kwargs):
  """Create an op that groups multiple operations.

  When this op finishes, all ops in `input` have finished. This op has no
  output.

  See also `tuple` and `with_dependencies`.

  Args:
    *inputs: One or more tensors to group.
    **kwargs: Optional parameters to pass when constructing the NodeDef.
    name: A name for this operation (optional).

  Returns:
    An Operation that executes all its inputs.

  Raises:
    ValueError: If an unknown keyword argument is provided, or if there are
                no inputs.
  """
  name = kwargs.pop("name", None)
  if kwargs:
    raise ValueError("Unknown keyword arguments: " + ", ".join(kwargs.keys()))
  if not inputs:
    # TODO(touts): Would make sense to return a NoOp.
    raise ValueError("No inputs provided")
  with ops.op_scope(inputs, name, "group_deps") as name:
    # Sorts *inputs according to their devices.
    ops_on_device = {}  # device -> operations specified on the device.
    for inp in inputs:
      dev = inp.device
      if dev in ops_on_device:
        ops_on_device[dev].append(inp)
      else:
        ops_on_device[dev] = [inp]
    if len(ops_on_device) == 1:
      # 1-level tree. The root node is the returned NoOp node.
      dev, deps = ops_on_device.items()[0]
      return _GroupControlDeps(dev, deps, name=name)
    # 2-level tree. The root node is the returned NoOp node.
    # deps contains 1 NoOp node for each device.
    deps = []
    for dev in sorted(ops_on_device.iterkeys()):
      deps.append(_GroupControlDeps(dev, ops_on_device[dev]))
    return _GroupControlDeps(None, deps, name=name)

def tuple(tensors, name=None, control_inputs=None):
  """Group tensors together.

  This creates a tuple of tensors with the same values as the `tensors`
  argument, except that the value of each tensor is only returned after the
  values of all tensors have been computed.

  `control_inputs` contains additional ops that have to finish before this op
  finishes, but whose outputs are not returned.

  This can be used as a "join" mechanism for parallel computations: all the
  argument tensors can be computed in parallel, but the values of any tensor
  returned by `tuple` are only available after all the parallel computations
  are done.

  See also `group` and `with_dependencies`.

  Args:
    tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
    name: (optional) A name to use as a `name_scope` for the operation.
    control_inputs: List of additional ops to finish before returning.

  Returns:
    Same as `tensors`.

  Raises:
    ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.

  """
  with ops.op_scope(tensors, name, "tuple") as name:
    gating_ops = [t.op for t in tensors if t]
    if control_inputs:
      gating_ops += control_inputs
    # Note that in order to ensure ordering in the pbtxt, we must take care to
    # ensure the order here.
    gating_ops = sorted(set(gating_ops), key=lambda op: op._id)  # Uniquify ops.
    if not gating_ops:
      raise ValueError("Must have at least one Tensor: %s" % tensors)
    gate = group(*gating_ops)
    tpl = []
    for t in tensors:
      if t:
        tpl.append(with_dependencies([gate], t))
      else:
        tpl.append(None)
    return tpl


# TODO(yuanbyu): It would be nicer if we could have the distributed list
# support that Derek has been proposing.
# TODO(yuanbyu, mrry): Handle stride to support sliding windows.
def fold(fn, elems, elem_shape, name=None):
  """The fold operator on slices of a tensor.

  This fold operator applies the function `fn` to slices of `elems` on
  dimension 0. The shape of the slices is specified by `elem_shape`. `elems`
  must contain at least one slice (`shape(elems)[0] / elem_shape[0] > 0`).

  Args:
    fn: The function to be performed on each slice of the tensor.
    elems: The tensor to whose slices we want to apply `fn`.
    elem_shape: The shape definition for the slices.
    name: Optional name prefix for the returned tensors.

  Returns:
    A tensor resulting from applying `fn` consecutively on each slice of
    `elems`.

  Raises:
    TypeError: if `fn` is not callable.
  """
  with ops.op_scope([elems], name, "Fold") as name:
    if not callable(fn):
      raise TypeError("fn must be callable.")

    s0 = array_ops.shape(elems)[0]
    d0 = elem_shape[0]
    n = math_ops.div(s0, d0)
    b1 = array_ops.zeros(array_ops.expand_dims(array_ops.rank(elems) - 1, 0),
                         dtype=types.int32)
    # Initialize the output with slice 0
    b = array_ops.concat(0, [[0], b1])
    o = array_ops.slice(elems, b, elem_shape)
    i = ops.convert_to_tensor(d0)

    def Compute(i, o):
      b = array_ops.concat(0, [array_ops.expand_dims(i, 0), b1])
      x = array_ops.slice(elems, b, elem_shape)
      o = fn(o, x)
      i = math_ops.add(i, d0)
      return [i, o]
    r = While(lambda i, o: math_ops.less(i, n), Compute, [i, o])
    return r[1]


def case(pred_fn_pairs, default, exclusive=False, name="Case"):
  """Create a Case operation.

  The `pred_fn_pairs` parameter is a dict or list of pairs of size N.
  Each pair contains a boolean scalar tensor and a python callable that
  creates the tensors to be returned if the boolean evaluates to True. `default`
  is a callable generating a list of tensors. All the callables in
  `pred_fn_pairs` as well as `default` should return the same number and types
  of tensors.

  If `exclusive==True`, all predicates are evaluated, and a logging operation
  with an error is returned if more than one of the predicates evaluates to
  True. If `exclusive==False`, execution stops are the first predicate which
  evaluates to True, and the tensors generated by the corresponding function
  are returned immediately. If none of the predicates evaluate to True, this
  operation returns the tensors generated by `default`.

  Example 1:
    Pseudocode:
    ```
      if (x < y) return 17;
      else return 23;
    ```

    Expressions:
    ```
      f1 = lambda: Constant(17)
      f2 = lambda: Constant(23)
      r = Case([(math_ops.less(x, y), f1)], default=f2)
    ```

  Example 2:
    Pseudocode:
    ```
      if (x < y && x > z) raise OpError("Only one predicate may evaluate true");
      if (x < y) return 17;
      else if (x > z) return 23;
      else return -1;
    ```

    Expressions:
    ```
      def f1(): return Constant(17)
      def f2(): return Constant(23)
      def f3(): return Constant(-1)
      r = Case({math_ops.less(x, y): f1, math_ops.greater(x, z): f2},
               default=f3, exclusive=True)
    ```

  Args:
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor and a
                   callable which returns a list of tensors.
    default: A callable that returns a list of tensors.
    exclusive: True iff more than one predicate is allowed to evaluate to True.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the first pair whose predicate evaluated to True, or
    those returned by `default` if none does.

  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  """
  pfp = pred_fn_pairs  # For readability
  if not (isinstance(pfp, list) or isinstance(pfp, _basetuple)
          or isinstance(pfp, dict)):
    raise TypeError("fns must be a list, tuple, or dict")
  if isinstance(pfp, dict):
    pfp = pfp.items()
    if not exclusive:
      logging.warn("%s: Provided dictionary of predicate/fn pairs, but "
                   "exclusive=False.  Order of conditional tests is "
                   "not guaranteed." % name)
  for tup in pfp:
    if not isinstance(tup, _basetuple) or len(tup) != 2:
      raise TypeError("Each entry in pred_fn_pairs must be a 2-tuple")
    pred, fn = tup
    if pred.dtype != types.bool:
      raise TypeError("pred must be of type bool: %s", pred.name)
    if not callable(fn):
      raise TypeError("fn for pred %s must be callable." % pred.name)
  if not callable(default):
    raise TypeError("default must be callable.")

  preds, fns = map(list, zip(*pfp))
  with ops.op_scope([[f() for f in fns] + preds + [default()]], name, "Case"):
    if not preds:
      return default()
    not_preds = []
    for i, p in enumerate(preds):
      with ops.name_scope("not_%d" % i):
        not_preds.append(math_ops.logical_not(p))
    and_not_preds = [constant_op.constant(True, name="and_not_true")]
    for i, notp in enumerate(not_preds[:-1]):
      with ops.name_scope("and_not_%d" % i):
        and_not_preds.append(math_ops.logical_and(and_not_preds[-1], notp))

    # preds = [p1, p2, p3]
    # fns = [f1, f2, f3]
    # not_preds = [~p1, ~p2, ~p3]
    # case_preds = [p1 & True,
    #               p2 & ~p1,
    #               p3 & ~p1 & ~ p2]
    case_preds = []
    for i, (p, and_not_p_prev) in enumerate(zip(preds, and_not_preds)):
      with ops.name_scope("case_%d" % i):
        case_preds.append(math_ops.logical_and(p, and_not_p_prev))

    # case_sequence = [Cond(p3 & ..., f3, default),
    #                  Cond(p2 & ..., f2, lambda: case_sequence[0]),
    #                  ...
    #                  Cond(p1 & True, f1, lambda: case_sequence[i-1])]
    # and prev_case_seq will loop from case_sequence[0] to case_sequence[-1]
    if exclusive:
      # TODO(ebrevdo): Add Where() for DT_BOOL, replace with Size(Where(preds))
      preds_c = array_ops.concat(0, preds, name="preds_c")
      num_true_conditions = math_ops.reduce_sum(
          math_ops.cast(preds_c, types.int32), name="num_true_conds")
      at_most_one_true_condition = math_ops.less(
          num_true_conditions, constant_op.constant(2, name="two_true_conds"))

      error_msg = [
          ("More than one condition evaluated as True but "
           "exclusive=True.  Conditions: (%s), Values:"
           % ", ".join([p.name for p in preds])),
          preds_c]
      with ops.control_dependencies([
          logging_ops.Assert(condition=at_most_one_true_condition,
                             data=error_msg, summarize=len(preds))]):
        prev_case_seq = default()
        for i, (cp, fn) in enumerate(zip(case_preds, fns)[::-1]):
          prev_case_seq = cond(cp, fn, lambda: prev_case_seq, name="If_%d" % i)
    else:
      prev_case_seq = default()
      for i, (cp, fn) in enumerate(zip(case_preds, fns)[::-1]):
        prev_case_seq = cond(cp, fn, lambda: prev_case_seq, name="If_%d" % i)

    return prev_case_seq


ops.RegisterShape("Enter")(common_shapes.unchanged_shape)
ops.RegisterShape("Exit")(common_shapes.unknown_shape)
ops.RegisterShape("NextIteration")(common_shapes.unchanged_shape)
ops.RegisterShape("RefEnter")(common_shapes.unchanged_shape)
ops.RegisterShape("ControlTrigger")(common_shapes.no_outputs)
ops.RegisterShape("NoOp")(common_shapes.no_outputs)


@ops.RegisterShape("LoopCond")
def _LoopCondShape(op):
  """Shape function for the LoopCond op."""
  return [op.inputs[0].get_shape().merge_with(tensor_shape.scalar())]


@ops.RegisterShape("Merge")
def _MergeShape(op):
  """Shape function for the Merge op.

  The Merge op takes many inputs of arbitrary shapes, and produces a
  first output that is one of those inputs, and a second scalar
  output.

  This function conservatively assumes that if any of its inputs is
  not fully defined, the output shape is unknown. If all of the inputs
  have the exact same known shape, the output must have that shape.

  Args:
    op: A Merge Operation.

  Returns:
    A single-element list containing the Shape of the Merge op.

  """
  first_input_shape = op.inputs[0].get_shape()
  if first_input_shape.is_fully_defined():
    for input_ in op.inputs[1:]:
      input_shape = input_.get_shape()
      if (not input_shape.is_fully_defined()
          or not input_shape.is_compatible_with(first_input_shape)):
        return [tensor_shape.unknown_shape(), tensor_shape.scalar()]
    return [first_input_shape, tensor_shape.scalar()]
  else:
    return [tensor_shape.unknown_shape(), tensor_shape.scalar()]


@ops.RegisterShape("RefSelect")
def _RefSelectShape(op):
  """Shape function for the RefSelect op.

  The RefSelect takes one scalar input and N inputs of arbitrary
  shapes, and produces one output, which is one of those N inputs.

  This function conservatively assumes that if any of the N inputs is
  not fully defined, the output shape is unknown. If all of the N
  inputs have the exact same known shape, the output must have that
  shape.

  Args:
    op: A RefSelect Operation.

  Returns:
    A single-element list containing the Shape of the RefSelect op.
  """
  unused_shape = op.inputs[0].get_shape().merge_with(tensor_shape.scalar())
  first_input_shape = op.inputs[1].get_shape()
  if first_input_shape.is_fully_defined():
    for input_ in op.inputs[2:]:
      input_shape = input_.get_shape()
      if (not input_shape.is_fully_defined()
          or not input_shape.is_compatible_with(first_input_shape)):
        return [tensor_shape.unknown_shape()]
    return [first_input_shape]
  else:
    return [tensor_shape.unknown_shape()]


@ops.RegisterShape("RefSwitch")
@ops.RegisterShape("Switch")
def _SwitchShape(op):
  input_shape = op.inputs[0].get_shape()
  unused_pred_shape = op.inputs[1].get_shape().merge_with(tensor_shape.scalar())
  return [input_shape] * 2
