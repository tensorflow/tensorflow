# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""while_v2 and gradient.

This is a version of while_loop that emits a single While op, as well as the
gradient function for While ops produced by while_loop. This will eventually
replace the current tf.while_loop implementation once it reaches feature and
performance parity.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import sys

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2_impl as cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import list_ops
from tensorflow.python.util import nest

# pylint: disable=protected-access

control_flow_ops._while_v2 = sys.modules[__name__]

# TODO(b/79881896): Handle external control dependencies. tf.while_loop allows
# control dependencies on external nodes with at least 1 output.
# Another idea is to create const nodes outside the loop and add control edges
# to them and then pass those in as data inputs. This should probably be
# handled in the CapturingGraph itself.


def while_loop(cond, body, loop_vars, name=None):
  """Like tf.while_loop, except emits a single While op."""
  if not name:
    name = "while"

  with ops.name_scope(name) as scope:
    with ops.name_scope(None):
      cond_name = _get_unique_name(("%scond" % scope).replace("/", "_"))
      body_name = _get_unique_name(("%sbody" % scope).replace("/", "_"))

    flattened_loop_vars = nest.flatten(loop_vars)
    num_outputs = len(flattened_loop_vars)

    # Add loop counter needed for computing gradients.
    flattened_loop_vars = [constant_op.constant(0., name="loop_counter")
                          ] + flattened_loop_vars

    # Build a `cond` wrapper that can handle the extra counter loop_var.
    def wrapped_cond(unused_loop_counter, *loop_vars):
      return cond(*loop_vars)

    cond_graph = function.func_graph_from_py_func(cond_name, wrapped_cond,
                                                  flattened_loop_vars, {})

    # Add external_captures of cond to the list of loop vars.
    # Note that external tensors will be treated as loop invariants, i.e.,
    # the value of that tensor in each iteration is the same as it was at the
    # beginning of the loop execution.
    flattened_loop_vars = flattened_loop_vars + cond_graph.external_captures

    def wrapped_body(loop_counter, *args):
      """Loop body augmented with counter update.

      Args:
        loop_counter: Loop counter which needs to be incremented in the body.
        *args: List of args
          args[:num_outputs] - Args for the original loop body.
          args[num_outputs:] - External captures of cond. These get passed
            through as is.

      Returns:
        A list of tensors the same length as args.
      """
      outputs = body(*args[:num_outputs])
      if not isinstance(outputs, collections.Sequence):
        outputs = [outputs]

      # Return the external_captures of cond_graph as is, i.e., treat them as
      # loop invariants.
      # TODO(srbs): Update lowering code to create _Enter nodes with
      # is_constant=True for inputs that are directly passed to outputs.
      return [loop_counter + 1] + list(outputs) + list(args[num_outputs:])

    body_graph = function.func_graph_from_py_func(body_name, wrapped_body,
                                                  flattened_loop_vars, {})
    # Add external captures of body to the list of loop vars.
    # Note that external tensors will be treated as loop invariants, i.e.,
    # the value of that tensor in each iteration is the same as it was at the
    # beginning of the loop execution.
    flattened_loop_vars = flattened_loop_vars + body_graph.external_captures
    # TODO(srbs): Update lowering code to create _Enter nodes with
    # is_constant=True for inputs that are directly passed to outputs.
    body_graph.outputs.extend(body_graph.internal_captures)

    # Capture `external_captures` of `body_graph` in `cond_graph` so that it
    # expects to receive those as arguments.
    # TODO(srbs): Dedup tensors that are captured in both the cond and body.
    # This logic already exists in cond_v2.
    with cond_graph.as_default():
      for external_capture in body_graph.external_captures:
        cond_graph.capture(external_capture)

    # Export all tensors in the loop body that may be needed for gradient
    # computation. We do this by accumulating the intermediate values in
    # TensorLists.
    intermediate_tensors = _get_intermediates(body_graph)

    for intermediate_tensor in intermediate_tensors:
      # TODO(srbs): Cache and re-use empty tensor lists.
      tensor_list = list_ops.empty_tensor_list(
          element_dtype=intermediate_tensor.dtype,
          element_shape=_get_tensor_convertible_shape(
              intermediate_tensor.shape))
      flattened_loop_vars.append(tensor_list)
      with cond_graph.as_default():
        # Add a placeholder to cond_graph's inputs corresponding to the
        # tensor_list.
        cond_graph.capture(tensor_list)
      with body_graph.as_default():
        # Push the intermediate tensor to the tensor list. This captures the
        # `tensor_list` as well.
        appended_tensor_list = list_ops.tensor_list_push_back(
            tensor_list,
            intermediate_tensor)
        # Add this modified tensor list to the list of outputs.
        body_graph.outputs.append(appended_tensor_list)

    outputs = gen_functional_ops._while(
        flattened_loop_vars,
        cond_v2._create_new_tf_function(cond_graph),
        cond_v2._create_new_tf_function(body_graph),
        name=scope)

    _copy_handle_data(body_graph.outputs, outputs)
    _maybe_set_lowering_attr(outputs[0].op)

  # First var is loop counter.
  if num_outputs == 1:
    return outputs[1]
  else:
    return nest.pack_sequence_as(loop_vars, outputs[1:1 + num_outputs])


@ops.RegisterGradient("While")
def _WhileGrad(op, *grads):  # pylint: disable=invalid-name
  """The gradient of a While op produced by while_loop."""
  body_graph = _get_body_graph(op)

  # Replace None gradients with zeros. This is needed because `grads` could have
  # None incoming gradients for the TensorLists. If we pass None's through, the
  # custom gradient of TensorListPopBack will create an EmptyTensorList inside
  # the FuncGraph which is undesirable.
  # TODO(b/80444525): There might be an issue with treating no gradient as zero
  # gradient in certain cases. Consider replacing None gradients with Zeros
  # for accumulators only.
  grads = [
      g if g is not None else array_ops.zeros_like(output)
      for g, output in zip(grads, op.outputs)
  ]

  body_grad_graph, args = _create_grad_func(
      body_graph, grads,
      _get_unique_name("%s_grad" % body_graph.name), op)

  intermediate_tensors = _get_intermediates(body_grad_graph)

  for intermediate_tensor in intermediate_tensors:
    tensor_list = list_ops.empty_tensor_list(
        element_dtype=intermediate_tensor.dtype,
        element_shape=_get_tensor_convertible_shape(intermediate_tensor.shape))
    with body_grad_graph.as_default():
      tensor_list_ph = body_grad_graph.capture(tensor_list, whitelisted=True)
      # Push the intermediate tensor to the tensor list.
      appended_tensor_list = list_ops.tensor_list_push_back(tensor_list_ph,
                                                            intermediate_tensor)
      # Add this modified tensor list to the list of outputs.
      body_grad_graph.outputs.append(appended_tensor_list)

  def grad_cond(counter, max_iters, *unused_args):
    return counter < max_iters

  loop_vars = args + body_grad_graph.external_captures
  cond_grad_graph = function.func_graph_from_py_func(
      _get_unique_name("%s_grad_cond" % op.name),
      grad_cond, loop_vars, {})

  assert len(loop_vars) == len(body_grad_graph.inputs)
  assert len(loop_vars) == len(body_grad_graph.outputs)
  assert len(loop_vars) == len(cond_grad_graph.inputs)

  outputs = gen_functional_ops._while(
      loop_vars,
      cond_v2._create_new_tf_function(cond_grad_graph),
      cond_v2._create_new_tf_function(body_grad_graph),
      name=_get_unique_name("%s_grad" % op.name))

  _copy_handle_data(body_grad_graph.outputs, outputs)
  _maybe_set_lowering_attr(outputs[0].op)

  # outputs[0] is the loop counter.
  # outputs[1] is the total number of loop iterations.
  return outputs[2:2 + len(op.inputs)]


# TODO(srbs): Pull this into common utils for cond_v2 and while_v2.
def _get_body_graph(while_op):
  """Returns `FuncGraph` for the while body.

  Args:
    while_op: The While Operation.

  Returns:
    `FuncGraph` for the while body.
  """
  extra_inputs = list(while_op.inputs)
  input_shapes = [t.shape for t in extra_inputs]
  func_name = while_op.get_attr("body").name
  fdef = while_op.graph._get_function(func_name).definition
  func_graph = function_def_to_graph.function_def_to_graph(fdef, input_shapes)
  func_graph._while = while_op
  return func_graph


def _create_grad_func(func_graph, grads, name, while_op):
  """Builds and returns the gradient FuncGraph of `func_graph` and its args.

  The returned grad_func_graph must be called with the returned
  args + grad_func_graph.captures.

  Args:
    func_graph: FuncGraph for the forward body function.
    grads: The incoming grads for `func_graph`'s outputs.
    name: Name of the returned gradient function.
    while_op: The forward While op.

  Returns:
    2-tuple of (grad_func_graph, args).
  """
  assert len(func_graph.outputs) == len(grads)

  loop_counter = constant_op.constant(0.)
  # TODO(srbs): For nested while loops will need to lookup this value from
  # the accumulator of the enclosing while loop. For now use as is assuming
  # there is no nesting.
  num_iters_t = while_op.outputs[0]

  args = [loop_counter, num_iters_t] + grads

  # Note: The returned function does not have `args` in the list of
  # `external_captures`.
  grad_func_graph = function.func_graph_from_py_func(
      name,
      lambda *args: _grad_fn(func_graph, args),
      args, {},
      func_graph=_WhileBodyGradFuncGraph(name, func_graph))

  # Add the popped accumulators to the list of outputs.
  for internal_capture in grad_func_graph.internal_captures:
    grad_func_graph.outputs.append(
        grad_func_graph.popped_tensor_lists[internal_capture])

  return grad_func_graph, args


def _grad_fn(func_graph, args):
  """Computes the gradient of `func_graph` in the current graph.

  This function builds the gradient graph of the corresponding forward-pass
  `func_graph` by differentiating `func_graph`'s outputs w.r.t. its inputs.

  Args:
    func_graph: function.FuncGraph. The corresponding forward-pass function.
    args: The input arguments. args[0] - Loop counter args[1] - Total number of
      iterations.
      args[2:] - Incoming gradients for `func_graph.outputs`.

  Returns:
    The output gradient Tensors.
  """
  xs = func_graph.inputs
  ys = func_graph.outputs
  grad_ys = args[2:]

  # Build the gradient graph. Note that this builds the gradient computation of
  # func_graph in the current graph, which requires capturing tensors from
  # func_graph. The captured func_graph tensors are resolved to external tensors
  # in _resolve_grad_inputs.
  # TODO(srbs): Mark GradientsHelper as public?
  grad_outs = gradients_impl._GradientsHelper(
      ys, xs, grad_ys=grad_ys, src_graph=func_graph)

  assert all([g is not None for g in grad_outs])
  counter = args[0]
  total_iters = args[1]
  return [counter + 1, total_iters] + grad_outs


def _get_intermediates(func_graph):
  """Returns all tensors in `func_graph` that should be accumulated."""
  # We currently accumulate output tensors of most ops in the function and rely
  # on the pruning pass to get rid of the unused accumulators at runtime.
  # However, this can bloat the GraphDef and make debugging harder so we perform
  # some optimizations.
  #
  # Optimization we currently perform:
  # 1. We do not accumulate tensors which already have an accumulator
  #    in the loop body.
  # 2. We do not accumulate outputs of Identity nodes. When building the
  #    FuncGraph, we add an Identity node for each output (see
  #    `AutomaticControlDependencies.mark_as_return`). Accumulating outputs
  #    of all these nodes bloats the GraphDef quite a bit so we remove those.
  #    Since the gradient of an Identity node does not rely on its forward op's
  #    input this is safe to do.
  #
  # Other possible optimizations:
  # 1. Only accumulate tensors that will be required by the backward pass.
  #    This will require running the gradient pass and hence would increase the
  #    graph building time for the forward pass.
  # 2. Do not accumulate Const nodes created inside the loop body.
  # 3. Do not accumulate inputs that are passed as-is, e.g. loop invariants.
  # TODO(srbs): 2 and 3 may be hard optimizations for the runtime optimizer
  # since it requires knowledge of the while loop semantics. If so, consider
  # doing those here.
  intermediates = []

  for op in func_graph.get_operations():
    if op.type == "Identity":
      continue
    for o in op.outputs:
      if (o != func_graph.inputs[0] and  # Loop counter.
          _get_accumulator(o) is None):  # Has existing accumulator.
        intermediates.append(o)
  return intermediates


def _get_accumulator(tensor):
  r"""Returns TensorList if any containing accumulated values of tensor.

  We try to find a pattern of the form:

     input_tl   tensor
        \        /
    (TensorListPushBack)
            |
        output_tl

  which satisfies the following conditions:

  1. input_tl must be in tensor.graph.inputs.
  2. output_tl or Identity(output_tl) must be in tensor.graph.outputs.
  3. tensor.graph.input_index(input_tl) == tensor.graph.output_index(output_t).

  output_tl or Identity(output_tl) (whichever is in tensor.graph.outputs) is
  returned if such a pattern is found else None is returned.

  Args:
    tensor: The Tensor to be accumulated.

  Returns:
    A variant tensor in the same graph as `tensor` or None if no accumulator is
    found.
  """
  assert isinstance(tensor.graph, function.FuncGraph)

  def get_func_graph_output(t):
    """Returns t or Identity(t) whichever exists in graph outputs else None."""
    if t in tensor.graph.outputs:
      return t
    # tf.defun adds an Identity for each output, check whether that is the case.
    identity_op = t.consumers()[0]
    if (identity_op.type == "Identity" and
        identity_op.outputs[0] in tensor.graph.outputs):
      return identity_op.outputs[0]
    return None

  for consumer in tensor.consumers():
    # Find the consumer that is a TensorListPushBack node whose TensorList input
    # is in the list of function inputs.
    if (consumer.type != "TensorListPushBack" or
        consumer.inputs[0] not in tensor.graph.inputs):
      continue

    output = get_func_graph_output(consumer.outputs[0])
    if output is None:
      # The TensorList output of `consumer` is not in the list of function
      # outputs.
      continue

    accum_input_idx = tensor.graph.inputs.index(consumer.inputs[0])
    accum_output_idx = tensor.graph.outputs.index(output)
    if accum_input_idx == accum_output_idx:
      return output
  return None


# TODO(srbs): Add to common utils for cond_v2 and while_v2.
def _get_unique_name(name):
  """Returns a name that is unique in the root graph of `func_graph`.

  Args:
    name: String to uniquify.

  Returns:
    A string.
  """
  with ops.init_scope():
    return ops.get_default_graph().unique_name(name)


class _WhileBodyGradFuncGraph(function.FuncGraph):
  """FuncGraph for the gradient function of the body of a While op.

  Contains the logic for capturing the tensors from the body of the forward
  While op which is as follows:
  1. Find the accumulator for that tensor.
  2. Capture the forward While op output tensor corresponding to the
     accumulator in this FuncGraph.
  3. Pop a value from the captured placeholder and use it as the captured value
     for the forward pass tensor.

  This only allows capturing tensors in the forward graph. A ValueError is
  raised if an attempt is made to capture a tensor not in the forward graph.
  To manually capture capture a tensor that is not in the forward graph, call
  `capture` with `whitelisted=True`.

  Note: The `captures` dict does not contain the forward tensor since it is not
  directly captured. It contains the accumulator corresponding to this forward
  tensor.

  Attributes:
    popped_tensor_lists: Dict from the captured accumulator placeholder to the
      TensorList obtained after popping the intermediate tensor from it. The
      values of this dict need to be added to the list of outputs.
  """

  def __init__(self, name, forward_graph):
    super(_WhileBodyGradFuncGraph, self).__init__(name)
    self.popped_tensor_lists = {}
    # FuncGraph for the body of the forward While op.
    self._forward_graph = forward_graph
    # Dict from forward intermediate tensor to the corresponding "popped" tensor
    # in this graph.
    self._indirect_captures = {}
    # Dict from forward graph tensor to the While op output corresponding to its
    # accumulator.
    self._tensor_to_accumulator = {}

  def capture(self, tensor, name=None, whitelisted=False):
    """Selectively captures external tensors.

    If `whitelisted` is False only allows capturing tensors in the
    `_forward_graph`.

    Args:
      tensor: Tensor. May be from this FuncGraph or a different graph.
      name: Optional name if a placeholder is created.
      whitelisted: If False (default), only allows capturing tensors from the
        forward graph.

    Returns:
      The placeholder in this graph for the tensor.

    Raises:
      ValueError: If attempting to capture an external tensor not in the forward
        graph with `whitelisted` set to False.
    """
    if (not whitelisted and tensor.graph is not self and
        tensor.graph != self._forward_graph):
      raise ValueError("Attempting to capture tensor", str(tensor),
                       " which is not in the forward graph but in ",
                       _graph_name(tensor.graph), ".")
    return super(_WhileBodyGradFuncGraph, self).capture(tensor, name)

  def _capture_helper(self, tensor, name):
    if tensor.graph is not self._forward_graph:
      return super(_WhileBodyGradFuncGraph, self)._capture_helper(tensor, name)

    captured_tensor = self._indirect_captures.get(tensor)
    if captured_tensor is not None:
      # For GradientTape housekeeping.
      assert self._tensor_to_accumulator[tensor] in self.captures
      super(_WhileBodyGradFuncGraph, self)._capture_helper(
          self._tensor_to_accumulator[tensor], name)
      return captured_tensor

    assert tensor not in self._tensor_to_accumulator

    accumulator = None

    # Find the TensorList that was used to accumulate the tensors of this
    # intermediate tensor.
    accumulator = _get_accumulator(tensor)
    if accumulator is None:
      raise ValueError("Reference to un-accumulated intermediate tensor: ",
                       tensor.name)
    assert accumulator.graph == self._forward_graph
    # Get the While op output corresponding to the accumulator.
    accumulator = self._forward_graph._while.outputs[self._forward_graph.outputs
                                                     .index(accumulator)]

    assert accumulator.graph == self._forward_graph.outer_graph
    self._tensor_to_accumulator[tensor] = accumulator

    # Capture the `accumulator`.
    accumulator_ph = super(_WhileBodyGradFuncGraph, self)._capture_helper(
        accumulator, name)
    new_tensor_list, captured_tensor = list_ops.tensor_list_pop_back(
        accumulator_ph, element_dtype=tensor.dtype)
    self._indirect_captures[tensor] = captured_tensor
    self.popped_tensor_lists[accumulator_ph] = new_tensor_list
    return captured_tensor


def _copy_handle_data(src_tensors, tgt_tensors):
  for src_t, tgt_t in zip(src_tensors, tgt_tensors):
    function._copy_handle_data(src_t, tgt_t)


# TODO(srbs): Move to common utils for cond_v2 and while_v2.
def _maybe_set_lowering_attr(op):
  """Sets the flag to enable lowering on the `While` op if necessary.

  Lowering allows while_v2 to avoid some of the limitations of Functions,
  allowing users to specify devices & colocation inside of while_v2
  branches, and enabling non-strict evaluation & partial pruning of while_v2
  branches. This brings while_v2 closer to feature parity with
  tf.while_loop.

  However, we do not lower `While` in the XLA context because it is easier
  for XLA to apply its own optimizations when dealing with un-lowered
  `While` operators than with low-level control flow primitives.

  Args:
    op: The While op.
  """
  if not control_flow_util.IsInXLAContext(op):
    # pylint: disable=protected-access
    op._set_attr("_lower_using_switch_merge", attr_value_pb2.AttrValue(b=True))
    # pylint: enable=protected-access


def _get_tensor_convertible_shape(shape):
  assert isinstance(shape, tensor_shape.TensorShape)
  if shape.is_fully_defined():
    return shape
  if not shape:  # Unknown shape.
    return -1
  # Partially defined shape.
  shape_list = shape.as_list()
  shape_list = [s if s is not None else -1 for s in shape_list]
  return ops.convert_to_tensor(shape_list)


def _graph_name(graph):
  if isinstance(graph, function.FuncGraph):
    return graph.name
  return "Base"


# pylint: enable=protected-access
