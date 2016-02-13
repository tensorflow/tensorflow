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

"""Gradients for operators defined in control_flow_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.control_flow_ops import *
from tensorflow.python.ops.gen_control_flow_ops import *
# pylint: enable=wildcard-import


def _SwitchGrad(op, *grad):
  """Gradients for a Switch op is calculated using a Merge op.

  If the switch is a loop switch, it will be visited twice. We create
  the merge on the first visit, and update the other input of the merge
  on the second visit. A next_iteration is also added on second visit.
  """
  real_op = GetRealOp(op)
  # pylint: disable=protected-access
  ctxt = real_op._get_control_flow_context()
  # pylint: enable=protected-access
  if isinstance(ctxt, WhileContext):
    merge_op = op.grad_state.switch_map.get(real_op)
    if merge_op:
      # This is the second time this Switch is visited. It comes from
      # the non-exit branch of the Switch, so update the second input
      # to the Merge.
      # TODO: Need to perform shape inference with this new input.
      # pylint: disable=protected-access
      merge_op._update_input(1, control_flow_ops._NextIteration(grad[1]))
      # pylint: enable=protected-access
      return None, None
    else:
      # This is the first time this Switch is visited. It always comes
      # from the Exit branch, which is grad[0]. grad[1] is empty at this point.
      # Use grad[0] for both inputs to merge for now, but update the second
      # input of merge when we see this Switch the second time.
      merge_fn = control_flow_ops._Merge  # pylint: disable=protected-access
      merge_op = merge_fn([grad[0], grad[0]], name="b_switch")[0]
      op.grad_state.switch_map[real_op] = merge_op.op
      return merge_op, None
  elif isinstance(ctxt, CondContext):
    good_grad = grad[ctxt.branch]
    zero_grad = grad[1 - ctxt.branch]
    # If this Switch is wrapped, it is part of a cond within a loop. In
    # this case, we have called ControlFlowState.ZeroLike() so grad is
    # ready for merge. Otherwise, we need a switch to control zero_grad.
    if not isinstance(op, ControlFlowOpWrapper):
      dtype = good_grad.dtype
      zero_grad = switch(zero_grad, ctxt.pred, dtype=dtype)[1 - ctxt.branch]
    return merge([good_grad, zero_grad], name="cond_grad")[0], None
  else:
    false_grad = switch(grad[0], real_op.inputs[1])[0]
    true_grad = switch(grad[1], real_op.inputs[1])[1]
    return merge([false_grad, true_grad])[0], None


ops.RegisterGradient("Switch")(_SwitchGrad)
ops.RegisterGradient("RefSwitch")(_SwitchGrad)


@ops.RegisterGradient("Merge")
def _MergeGrad(op, grad, _):
  """Gradients for a Merge op are calculated using a Switch op."""
  real_op = GetRealOp(op)
  input_op = real_op.inputs[0].op
  # pylint: disable=protected-access
  ctxt = input_op._get_control_flow_context()
  # pylint: enable=protected-access
  if isinstance(ctxt, WhileContext):
    grad_ctxt = op.grad_state.grad_context
    # pylint: disable=protected-access
    return control_flow_ops._SwitchRefOrTensor(grad, grad_ctxt.pivot)
    # pylint: enable=protected-access
  elif isinstance(ctxt, CondContext):
    pred = ctxt.pred
    if isinstance(op, ControlFlowOpWrapper):
      # This Merge node is part of a cond within a loop.
      # The backprop needs to have the value of this predicate for every
      # iteration. So we must have its values accumulated in the forward, and
      # use the accumulated values as the predicate for this backprop switch.
      grad_state = op.grad_state
      real_pred = grad_state.history_map.get(pred.name)
      if not real_pred:
        # Remember the value of pred for every iteration.
        grad_ctxt = grad_state.grad_context
        grad_ctxt.Exit()
        history_pred = grad_state.AddForwardAccumulator(pred)
        grad_ctxt.Enter()

        # Add the stack pop op. If pred.op is in a (outer) CondContext,
        # the stack pop will be guarded with a switch.
        real_pred = grad_state.AddBackPropAccumulatedValue(history_pred, pred)
        grad_state.history_map[pred.name] = real_pred
      pred = real_pred
    # pylint: disable=protected-access
    return control_flow_ops._SwitchRefOrTensor(grad, pred, name="cond_grad")
    # pylint: enable=protected-access
  else:
    num_inputs = len(real_op.inputs)
    cond = [math_ops.equal(real_op.outputs[1], i) for i in xrange(num_inputs)]
    # pylint: disable=protected-access
    return [control_flow_ops._SwitchRefOrTensor(grad, cond[i])[1]
            for i in xrange(num_inputs)]
    # pylint: enable=protected-access


@ops.RegisterGradient("RefMerge")
def _RefMergeGrad(op, grad, _):
  return _MergeGrad(op, grad, _)


@ops.RegisterGradient("Exit")
def _ExitGrad(op, grad):
  """Gradients for an exit op are calculated using an Enter op."""
  real_op = GetRealOp(op)
  # pylint: disable=protected-access
  forward_ctxt = real_op._get_control_flow_context()
  # pylint: enable=protected-access
  if not forward_ctxt.back_prop:
    # No gradient computation for this loop.
    return None
  grad_ctxt = op.grad_state.grad_context
  grad_ctxt.AddName(grad.name)
  enter_fn = control_flow_ops._Enter  # pylint: disable=protected-access
  return enter_fn(grad, grad_ctxt.name, is_constant=False,
                  parallel_iterations=grad_ctxt.parallel_iterations,
                  name="b_exit")


ops.RegisterGradient("RefExit")(_ExitGrad)


@ops.RegisterGradient("NextIteration")
def _NextIterationGrad(_, grad):
  """A forward next_iteration is translated into a backprop identity.

  Note that the backprop next_iteration is added in switch grad.
  """
  return grad


@ops.RegisterGradient("RefNextIteration")
def _RefNextIterationGrad(_, grad):
  return _NextIterationGrad(_, grad)


@ops.RegisterGradient("Enter")
def _EnterGrad(op, grad):
  """Gradients for an Enter are calculated using an Exit op.

  For loop variables, grad is the gradient so just add an exit.
  For loop invariants, we need to add an accumulator loop.
  """
  real_op = GetRealOp(op)
  # pylint: disable=protected-access
  forward_ctxt = real_op._get_control_flow_context()
  # pylint: enable=protected-access
  if not forward_ctxt.back_prop:
    # The flag `back_prop` is set by users to suppress gradient
    # computation for this loop. If the flag `back_prop` is true,
    # no gradient computation.
    return grad
  grad_ctxt = op.grad_state.grad_context
  if real_op.get_attr("is_constant"):
    # Add a gradient accumulator for each loop invariant.
    result = grad_ctxt.AddBackPropAccumulator(grad)
  else:
    result = exit(grad)
    grad_ctxt.ExitResult([result])
  return result


@ops.RegisterGradient("RefEnter")
def _RefEnterGrad(op, grad):
  return _EnterGrad(op, grad)


@ops.RegisterGradient("LoopCond")
def _LoopCondGrad(_):
  """Stop backprop for the predicate of a while loop."""
  return None
