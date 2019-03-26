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
# ==============================================================================

"""Utilty functions for control flow.

This file is necessary to avoid cyclic dependencies between ops.py and
control_flow_ops.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import traceback

from tensorflow.python.platform import tf_logging as logging

ENABLE_CONTROL_FLOW_V2 = (os.getenv("TF_ENABLE_CONTROL_FLOW_V2", "0") != "0" or
                          os.getenv("TF_ENABLE_COND_V2", "0") != "0" or
                          os.getenv("TF_ENABLE_WHILE_V2", "0") != "0" or
                          os.getenv("TF_ENABLE_TENSOR_ARRAY_V2", "0") != "0")


def EnableControlFlowV2(graph):
  """Returns whether control flow v2 should be used in `graph`."""
  # Enable new control flow in FuncGraphs (but not legacy _FuncGraphs).
  # TODO(skyewm): do something better than hasattr without messing up imports.
  return ENABLE_CONTROL_FLOW_V2 or (
      graph.building_function and not hasattr(graph, "_captured"))


def IsInXLAContext(op):
  try:
    xla_compile = op.get_attr("_XlaCompile")
    if xla_compile: return True
  except ValueError:
    pass
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  return GetContainingXLAContext(ctxt) is not None


def InXlaContext(graph):
  ctxt = graph._get_control_flow_context()  # pylint: disable=protected-access
  return GetContainingXLAContext(ctxt) is not None


def GraphOrParentsInXlaContext(graph):
  while True:
    if InXlaContext(graph): return True
    try:
      graph = graph.outer_graph
    except AttributeError:
      return False


def IsInWhileLoop(op):
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  return GetContainingWhileContext(ctxt) is not None


def IsInCond(op):
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  return GetContainingCondContext(ctxt) is not None


def IsSwitch(op):
  """Return true if `op` is a Switch."""
  return op.type == "Switch" or op.type == "RefSwitch"


def IsMerge(op):
  """Return true if `op` is a Merge."""
  return op.type == "Merge" or op.type == "RefMerge"


def IsLoopEnter(op):
  """Returns true if `op` is an Enter."""
  return op.type == "Enter" or op.type == "RefEnter"


def IsLoopExit(op):
  """Return true if `op` is an Exit."""
  return op.type == "Exit" or op.type == "RefExit"


def IsCondSwitch(op):
  """Return true if `op` is the Switch for a conditional."""
  if not IsSwitch(op):
    return False
  if not op.outputs:
    return False
  # Switch nodes are not part of the cond control flow context that they
  # represent, so consider the consumers of its outputs to determine if it is
  # cond switch or not. A switch is a cond switch iff all its consumers are in
  # cond contexts.
  is_cond_switch = True
  for o in op.outputs:
    for c in o.consumers():
      ctxt = c._get_control_flow_context()  # pylint: disable=protected-access
      if IsLoopEnter(c):
        ctxt = ctxt.outer_context
      is_cond_switch = is_cond_switch and (ctxt is not None and
                                           ctxt.IsCondContext())
  return is_cond_switch


def IsCondMerge(op):
  """Return true if `op` is the Merge for a conditional."""
  if not IsMerge(op):
    return False
  if not op.inputs:
    return False
  # Merge nodes are not part of the cond control flow context that they
  # represent, so consider the inputs to the merge of to determine if it is
  # cond merge or not: A merge is a cond merge iff all its inputs are in
  # cond contexts.
  is_cond_merge = True
  for i in op.inputs:
    ctxt = GetOutputContext(i.op)
    is_cond_merge = is_cond_merge and ctxt is not None and ctxt.IsCondContext()
  return is_cond_merge


def IsLoopSwitch(op):
  """Return true if `op` is the Switch for a while loop."""
  if IsSwitch(op):
    ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
    return ctxt is not None and ctxt.IsWhileContext() and not IsCondSwitch(op)
  return False


def IsLoopMerge(op):
  """Return true if `op` is the Merge for a while loop."""
  if IsMerge(op):
    ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
    return ctxt is not None and ctxt.IsWhileContext() and not IsCondMerge(op)
  return False


def IsLoopConstantEnter(op):
  """Return true iff op is a loop invariant."""
  return IsLoopEnter(op) and op.get_attr("is_constant")


def GetLoopConstantEnter(value):
  """Return the enter op if we can infer `value` to be a loop invariant."""
  id_ops = {"Switch", "RefSwitch", "Identity", "RefIdentity"}
  op = value.op
  while op.type in id_ops:
    op = op.inputs[0].op
  return op if IsLoopConstantEnter(op) else None


def GetOutputContext(op):
  """Return the control flow context for the output of an op."""
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  # Exit nodes usually have a control flow context, except in the case where the
  # exit node was imported via import_graph_def (in which case no nodes have
  # control flow contexts).
  if ctxt is not None and IsLoopExit(op):
    ctxt = ctxt.outer_context
  return ctxt


def GetContainingWhileContext(ctxt, stop_ctxt=None):
  """Returns the first ancestor WhileContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a WhileContext, or None if `ctxt` is not in a
  while loop.

  Args:
    ctxt: ControlFlowContext
    stop_ctxt: ControlFlowContext, optional. If provided, the search will end
      if it sees stop_ctxt.

  Returns:
    `ctxt` if `ctxt` is a WhileContext, the most nested WhileContext containing
    `ctxt`, or None if `ctxt` is not in a while loop.  If `stop_ctxt` is not
    `None`, this returns `ctxt` if it matches `stop_ctxt` in its traversal.
  """
  while ctxt:
    if ctxt.IsWhileContext() or ctxt == stop_ctxt: return ctxt
    ctxt = ctxt.outer_context
  return None


def GetContainingXLAContext(ctxt):
  """Returns the first ancestor XLAContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a XLAContext, or None if `ctxt` is not in a
  while loop.

  Args:
    ctxt: ControlFlowContext

  Returns:
    `ctxt` if `ctxt` is a XLAContext, the most nested XLAContext containing
    `ctxt`, or None if `ctxt` is not in a while loop.
  """
  while ctxt:
    if ctxt.IsXLAContext(): return ctxt
    ctxt = ctxt.outer_context
  return None


def GetContainingCondContext(ctxt):
  """Returns the first ancestor CondContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a CondContext, or None if `ctxt` is not in a cond.

  Args:
    ctxt: ControlFlowContext

  Returns:
    `ctxt` if `ctxt` is a CondContext, the most nested CondContext containing
    `ctxt`, or None if `ctxt` is not in a cond.
  """
  while ctxt:
    if ctxt.IsCondContext(): return ctxt
    ctxt = ctxt.outer_context
  return None


def IsContainingContext(ctxt, maybe_containing_ctxt):
  """Returns true if `maybe_containing_ctxt` is or contains `ctxt`."""
  while ctxt is not maybe_containing_ctxt:
    if ctxt is None: return False
    ctxt = ctxt.outer_context
  return True


def OpInContext(op, ctxt):
  return IsContainingContext(op._get_control_flow_context(), ctxt)  # pylint: disable=protected-access


def TensorInContext(tensor, ctxt):
  return OpInContext(tensor.op, ctxt)


def CheckInputFromValidContext(op, input_op):
  """Returns whether `input_op` can be used from `op`s context.

  Conceptually, only inputs from op's while context or any ancestor while
  context (including outside of any context) are valid. In practice, there are
  many other edge cases as well.

  Args:
    op: Operation
    input_op: Operation

  Raises:
    ValueError: if input_op is from an invalid context.
  """
  op_ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  input_ctxt = GetOutputContext(input_op)
  valid = False

  if not input_ctxt:
    # input_op isn't in a control flow context.
    valid = True
  elif op_ctxt is input_ctxt:
    # input_op is in the same context as op.
    valid = True
  else:
    while_ctxt = GetContainingWhileContext(op_ctxt)
    input_while_ctxt = GetContainingWhileContext(input_ctxt)

    if while_ctxt is None:
      if input_while_ctxt is None:
        # Neither op nor input_op is in a while loop, but one or both are in
        # conds. We allow this, although execution will fail if the branch
        # corresponding to input_op's cond context isn't taken.
        valid = True
      # Invalid if op isn't in a while loop and input_op is. Unless...
      if IsLoopEnter(op):
        # WhileContext._BuildLoop clears context for Enter nodes.
        valid = True
      if IsSwitch(op):
        # CondContext.AddValue clears context for Switch nodes.
        valid = True
    elif IsContainingContext(while_ctxt, input_while_ctxt):
      # input_op is in a while loop which contains op's while loop (or not in a
      # while loop at all).
      valid = True
    elif (while_ctxt.grad_state and
          IsContainingContext(while_ctxt.grad_state.forward_context,
                              input_while_ctxt)):
      # op is in a gradient context and input_op is in the associated forward
      # pass context or an ancestor thereof. This case is need to build while
      # loop gradients.
      # NOTE(skyewm): we theoretically also need this case for custom gradient
      # functions that close over tensors from ancestor contexts, but I haven't
      # verified this.
      valid = True
    elif (while_ctxt.grad_state and
          while_ctxt.grad_state.forward_context is
          input_while_ctxt._outer_context):  # pylint: disable=protected-access
      # op is in a gradient context and input_op is in a child of the associated
      # forward pass context. This case is needed for the gradients of while
      # loops with conds.
      valid = True
    elif (input_while_ctxt.grad_state and
          input_while_ctxt.grad_state.forward_context is while_ctxt):
      # input_op is in the gradient context of op's context. This case is needed
      # when the gradient of a while loop gradient is requested (this will
      # eventually fail unless there is a stop_gradient() or similar).
      valid = True
    elif (input_while_ctxt.grad_state and
          input_ctxt.grad_state.forward_context.grad_state and
          input_ctxt.grad_state.forward_context.grad_state.forward_context is
          while_ctxt):
      # input_op is in the grad grad context of op's context. This case is
      # needed when the gradient of a while loop gradient is requested (this
      # will eventually fail unless there is a stop_gradient() or similar).
      valid = True

  if not valid:
    if while_ctxt:
      error_msg = (
          "Cannot use '%s' as input to '%s' because they are in different while"
          " loops." % (input_op.name, op.name))
    else:
      error_msg = (
          "Cannot use '%s' as input to '%s' because '%s' is in a while loop."
          % (input_op.name, op.name, input_op.name))

    # Log the error message plus the relevant stack traces. The stacks may be
    # useful for debugging this error, but we don't want to raise an
    # unreadable exception.
    log_msg = error_msg
    log_msg += "\n\n%s while context: %s" % (op.name, while_ctxt)
    log_msg += "\n%s while context: %s" % (input_op.name, input_while_ctxt)
    log_msg += "\n\nTraceback for %s:\n%s\nTraceback for %s:\n%s\n" % (
        op.name, "".join(traceback.format_list(op.traceback)),
        input_op.name, "".join(traceback.format_list(input_op.traceback)))
    logging.info(log_msg)
    raise ValueError(error_msg + " See info log for more details.")
