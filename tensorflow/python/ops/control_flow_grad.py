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

"""Gradients for operators defined in control_flow_ops.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
# go/tf-wildcard-import
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
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  op_ctxt = op._get_control_flow_context()
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if isinstance(op_ctxt, WhileContext):
    merge_grad = grad_ctxt.grad_state.switch_map.get(op)
    if merge_grad is not None:
      # This is the second time this Switch is visited. It comes from
      # the non-exit branch of the Switch, so update the second input
      # to the Merge.
      # TODO: Perform shape inference with this new input.
      if grad[1] is not None:
        # pylint: disable=protected-access
        control_flow_ops._AddNextAndBackEdge(merge_grad, grad[1])
        # pylint: enable=protected-access
      return None, None
    else:
      # This is the first time this Switch is visited. It always comes from
      # the Exit branch, which is grad[0]. grad[1] is empty at this point.
      # Use grad[0] for both inputs to merge for now, but update the second
      # input of merge when we see this Switch the second time.
      merge_grad = merge([grad[0], grad[0]], name="b_switch")[0]
      grad_ctxt.grad_state.switch_map[op] = merge_grad
      return merge_grad, None
  elif isinstance(op_ctxt, CondContext):
    good_grad = grad[op_ctxt.branch]
    zero_grad = grad[1 - op_ctxt.branch]
    # At this point, we have created zero_grad guarded by the right switch.
    return merge([good_grad, zero_grad], name="cond_grad")[0], None
  else:
    false_grad = switch(grad[0], op.inputs[1])[0]
    true_grad = switch(grad[1], op.inputs[1])[1]
    return merge([false_grad, true_grad])[0], None


ops.RegisterGradient("Switch")(_SwitchGrad)
ops.RegisterGradient("RefSwitch")(_SwitchGrad)


@ops.RegisterGradient("Merge")
def _MergeGrad(op, grad, _):
  """Gradients for a Merge op are calculated using a Switch op."""
  input_op = op.inputs[0].op
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  op_ctxt = input_op._get_control_flow_context()
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if isinstance(op_ctxt, WhileContext):
    # pylint: disable=protected-access
    return control_flow_ops._SwitchRefOrTensor(grad, grad_ctxt.pivot)
    # pylint: enable=protected-access
  elif isinstance(op_ctxt, CondContext):
    pred = op_ctxt.pred
    if grad_ctxt and grad_ctxt.grad_state:
      # This Merge node is part of a cond within a loop.
      # The backprop needs to have the value of this predicate for every
      # iteration. So we must have its values accumulated in the forward, and
      # use the accumulated values as the predicate for this backprop switch.
      grad_state = grad_ctxt.grad_state
      real_pred = grad_state.history_map.get(pred.name)
      if real_pred is None:
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
    num_inputs = len(op.inputs)
    cond = [math_ops.equal(op.outputs[1], i) for i in xrange(num_inputs)]
    # pylint: disable=protected-access
    return [control_flow_ops._SwitchRefOrTensor(grad, cond[i])[1]
            for i in xrange(num_inputs)]
    # pylint: enable=protected-access


@ops.RegisterGradient("RefMerge")
def _RefMergeGrad(op, grad, _):
  return _MergeGrad(op, grad, _)


@ops.RegisterGradient("Exit")
def _ExitGrad(_, grad):
  """Gradients for an exit op are calculated using an Enter op."""
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if not grad_ctxt.back_prop:
    # The flag `back_prop` is set by users to suppress gradient
    # computation for this loop. If the attribute `back_prop` is false,
    # no gradient computation.
    return None
  if isinstance(grad, ops.Tensor):
    grad_ctxt.AddName(grad.name)
  else:
    if not isinstance(grad, (ops.IndexedSlices, ops.SparseTensor)):
      raise TypeError("Type %s not supported" % type(grad))
    grad_ctxt.AddName(grad.values.name)
    grad_ctxt.AddName(grad.indices.name)
    if isinstance(grad, ops.IndexedSlices):
      dense_shape = grad.dense_shape
    else:
      dense_shape = grad.shape
    if dense_shape is not None:
      grad_ctxt.AddName(dense_shape.name)
  enter_fn = control_flow_ops._Enter  # pylint: disable=protected-access
  grad_ctxt.Enter()
  result = enter_fn(grad, grad_ctxt.name, is_constant=False,
                    parallel_iterations=grad_ctxt.parallel_iterations,
                    name="b_exit")
  grad_ctxt.Exit()
  return result


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
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if not grad_ctxt.back_prop:
    # Skip gradient computation, if the attribute `back_prop` is false.
    return grad
  if grad_ctxt.grad_state is None:
    # Pass the gradient grough if we are not in a gradient while context.
    return grad
  if op.get_attr("is_constant"):
    # Add a gradient accumulator for each loop invariant.
    if isinstance(grad, ops.Tensor):
      result = grad_ctxt.AddBackPropAccumulator(op, grad)
    elif isinstance(grad, ops.IndexedSlices):
      result = grad_ctxt.AddBackPropIndexedSlicesAccumulator(op, grad)
    else:
      # TODO(yuanbyu, lukasr): Add support for SparseTensor.
      raise TypeError("Type %s not supported" % type(grad))
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
