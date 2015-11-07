"""Gradients for operators defined in control_flow_ops.py."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.control_flow_ops import *
from tensorflow.python.ops.gen_control_flow_ops import *


@ops.RegisterGradient("Switch")
def _SwitchGrad(op, *grad):
  op = GetRealOp(op)
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  if isinstance(ctxt, WhileContext):
    merge_op = ctxt.switch_map.get(op)
    if merge_op:
      merge_op._update_input(1, grad[1])
      return None, None
    else:
      merge_op = merge(grad, name="b_switch")[0]
      ctxt.switch_map[op] = merge_op.op
      return merge_op, None
  elif isinstance(ctxt, CondContext):
    good_grad = grad[ctxt.branch]
    zero_grad = grad[1 - ctxt.branch]
    zero_grad = switch(zero_grad, ctxt.pred, name="grad_0")[1 - ctxt.branch]
    return merge([good_grad, zero_grad], name="switch_grad")[0], None
  else:
    false_grad = switch(grad[0], op.inputs[1])[0]
    true_grad = switch(grad[1], op.inputs[1])[1]
    return merge([false_grad, true_grad])[0], None


@ops.RegisterGradient("RefSwitch")
def _RefSwitchGrad(op, *grad):
  return _SwitchGrad(op, *grad)


@ops.RegisterGradient("Merge")
def _MergeGrad(op, grad, _):
  op = GetRealOp(op)
  input_op = op.inputs[0].op
  # pylint: disable=protected-access
  ctxt = input_op._get_control_flow_context()
  # pylint: enable=protected-access
  if isinstance(ctxt, WhileContext):
    grad_ctxt = ctxt.grad_context
    return switch(grad, grad_ctxt.pivot)
  elif isinstance(ctxt, CondContext):
    return switch(grad, ctxt.pred, name="merge_grad")
  else:
    num_inputs = len(op.inputs)
    cond = [math_ops.equal(op.outputs[1], i) for i in xrange(num_inputs)]
    return [Switch(grad, cond[i])[1] for i in xrange(num_inputs)]


@ops.RegisterGradient("Exit")
def _ExitGrad(op, grad):
  # pylint: disable=protected-access
  forward_ctxt = op._get_control_flow_context()
  # pylint: enable=protected-access
  if not forward_ctxt.back_prop:
    return None
  grad_ctxt = forward_ctxt.grad_context
  grad_ctxt.AddName(grad.name)
  return enter(grad, grad_ctxt.name, is_constant=False,
               parallel_iterations=forward_ctxt.parallel_iterations,
               name="b_exit")


@ops.RegisterGradient("NextIteration")
def _NextIterationGrad(_, grad):
  return next_iteration(grad)


@ops.RegisterGradient("Enter")
def _EnterGrad(op, grad):
  op = GetRealOp(op)
  # pylint: disable=protected-access
  forward_ctxt = op._get_control_flow_context()
  # pylint: enable=protected-access
  grad_ctxt = forward_ctxt.grad_context
  if grad_ctxt:
    if op.get_attr("is_constant"):
      # Add a gradient accumulator for every loop invariant.
      result = grad_ctxt.AddBackPropAccumulateLoop(grad)
    else:
      result = exit(grad)
    return result
  else:
    return grad


@ops.RegisterGradient("RefEnter")
def _RefEnterGrad(op, grad):
  return _EnterGrad(op, grad)


@ops.RegisterGradient("LoopCond")
def _LoopCondGrad(_):
  return None
