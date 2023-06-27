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
# =============================================================================
"""Methods for rewriting while_v2 grad functions with IndexedSlices output."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.util import nest


def rewrite_grad_indexed_slices(grads, body_grad_graph, loop_vars,
                                forward_inputs):
  """Handles special case of IndexedSlices returned from while gradient.

  Some gradient functions return IndexedSlices instead of a Tensor (e.g. the
  gradient of Gather ops). When this happens in the gradient of a while body,
  the resulting gradient body function will have mismatched inputs and outputs,
  since the input is a single Tensor, but the IndexedSlices gets unnested into
  three output Tensors.

  This function fixes this by rewriting the gradient body to have three inputs
  to match the three outputs, i.e., it effectively converts the input Tensor
  into an input IndexedSlices. It also returns new `loop_vars` to reflect the
  new inputs.

  Args:
    grads: the input gradient Tensors to the while gradient computation.
    body_grad_graph: _WhileBodyGradFuncGraph.
    loop_vars: list of Tensors. The inputs to body_grad_graph.
    forward_inputs: list of Tensors. The (flat) inputs to the forward-pass While
      op.

  Returns:
    The new loop_vars to pass to body_grad_graph.
  """
  # Match up body_grad_graph.structured_outputs with the corresponding
  # forward_inputs.
  #
  # Note that we don't expect a gradient computation to have structured output
  # (e.g. no nested lists), so no need to flatten
  # body_grad_graph.structured_outputs. However, structured_outputs may still
  # contain composite tensors such as IndexedSlices, unlike
  # body_grad_graph.outputs, which contains flattened composite tensors.
  inputs_with_grads = [
      t for g, t in zip(grads, forward_inputs) if g is not None
  ]
  # Skip loop counter, maximum_iterations and total number of loop iterations.
  structured_outputs = body_grad_graph.structured_outputs[3:]

  for forward_input, output in zip(inputs_with_grads, structured_outputs):
    if not isinstance(output, indexed_slices.IndexedSlices):
      continue

    if forward_input.dtype == dtypes.resource:
      # TODO(skyewm): In theory we should use this for all captured inputs, not
      # just resource handles (which can only be captured). We can do this by
      # checking that forward_input is passed straight through to its output.
      loop_vars = _rewrite_input_as_indexed_slices(body_grad_graph, output,
                                                   forward_input, loop_vars)
    else:
      _rewrite_output_as_tensor(body_grad_graph, output)

  return loop_vars


def _get_tensor_index_in_iterable(iterable, t):
  """Returns index of first occurence of `t`, raises ValueError if not found."""
  for i, elem in enumerate(iterable):
    if t is elem:
      return i
  raise ValueError(f"Element `{t!r}` is not found in iterable `{iterable!r}`.")


def _rewrite_output_as_tensor(body_grad_graph, grad_output_slices):
  """Rewrites grad_output_slices to be a Tensor output.

  Args:
    body_grad_graph: _WhileBodyGradFuncGraph.
    grad_output_slices: IndexedSlices output of body_grad_graph.
  """
  with body_grad_graph.as_default():
    new_output = tensor_conversion.convert_to_tensor_v2(grad_output_slices)

  idx = _get_tensor_index_in_iterable(body_grad_graph.structured_outputs,
                                      grad_output_slices)
  body_grad_graph.structured_outputs[idx] = new_output
  body_grad_graph.outputs = func_graph.flatten(
      body_grad_graph.structured_outputs)


def _rewrite_input_as_indexed_slices(body_grad_graph, grad_output_slices,
                                     forward_input, loop_vars):
  """Rewrites grad_output_slices's corresponding input to be an IndexedSlices.

  This rewrite requires that forward_input was captured in the forward loop,
  i.e. is not a user-specified loop variable. This is important because the
  rewrite assumes that forward_input is passed through to its corresponding
  output unchanged. This assumption is used in _rewrite_input_as_indexed_slices,
  which depends on the exact gradient structure produced by the input's fanout.

  This can yield a more efficient computation than using
  _rewrite_output_as_tensor, since it preserves the IndexedSlices structure
  instead of converting the IndexedSlices to a dense Tensor.

  Args:
    body_grad_graph: _WhileBodyGradFuncGraph.
    grad_output_slices: IndexedSlices output of body_grad_graph.
    forward_input: the corresponding Tensor input to the forward loop.
    loop_vars: list of Tensors. The inputs to body_grad_graph.

  Returns:
    The new loop_vars to pass to body_grad_graph.
  """
  # Create initial IndexedSlices that will be the input to the grad While
  # op. This will start as zeros, and accumulate the IndexedSlices grad output.
  # Note that because forward_input is captured and not a loop var, its incoming
  # gradient should always be zero.
  init_slices = _create_grad_indexed_slices_init(grad_output_slices,
                                                 forward_input)

  # Create a new version of grad_output_slices's gradient computation that uses
  # the new IndexedSlices input instead of the original Tensor input. We'll
  # return the new computation and leave the old computation as dead code.
  # TODO(skyewm): considering pruning body_grad_graph to remove the old
  # computation.
  with body_grad_graph.as_default():
    input_slices = indexed_slices.IndexedSlices(
        values=body_grad_graph.capture(init_slices.values, allowlisted=True),
        indices=body_grad_graph.capture(init_slices.indices, allowlisted=True),
        dense_shape=body_grad_graph.capture(
            init_slices.dense_shape, allowlisted=True))

    # Remove the captured tensors from the function inputs. We'll add them back
    # at the correct index in _update_indexed_slices_param.
    for t in _flatten(init_slices):
      captured_t = body_grad_graph.captures.pop(t)
      body_grad_graph.inputs.remove(captured_t)

    new_output_slices = _rewrite_grad_indexed_slices_output(
        grad_output_slices, input_slices)

  # Update body_grad_graph's inputs and outputs to reflect the new
  # IndexedSlices computation.
  return _update_indexed_slices_param(body_grad_graph, loop_vars, init_slices,
                                      input_slices, new_output_slices,
                                      grad_output_slices)


def _create_grad_indexed_slices_init(grad_output_slices, forward_input):
  """Creates an IndexedSlices to pass as input to the while grad function.

  Args:
    grad_output_slices: IndexedSlices. The corresponding while grad function
      output.
    forward_input: Tensor. The corresponding input to the forward while op.

  Returns:
    Zeros IndexedSlices, created in current Graph.
  """
  assert isinstance(grad_output_slices, indexed_slices.IndexedSlices)
  assert isinstance(forward_input, ops.Tensor)
  values_out = grad_output_slices.values
  indices_out = grad_output_slices.indices

  # Create the initial values tensor.
  if values_out.shape.is_fully_defined():
    values_shape = tensor_shape.TensorShape([0] +
                                            values_out.shape.as_list()[1:])
    values = array_ops.zeros(
        values_shape, dtype=values_out.dtype, name="values_init")
  else:
    if forward_input.dtype == dtypes.resource:
      forward_shape = gen_resource_variable_ops.variable_shape(forward_input)
    else:
      forward_shape = array_ops.shape(forward_input)
    values_shape = array_ops.concat([[0], forward_shape[1:]], 0)
    values = array_ops.zeros(
        values_shape, dtype=values_out.dtype, name="values_init")

  # Create the initial indices tensor.
  indices = constant_op.constant([], indices_out.dtype, name="indices_init")

  # Create the initial dense_shape tensor. We assume is the same shape as
  # forward_input, since captured tensors don't change shape across loop
  # iterations.
  if forward_input.dtype == dtypes.resource:
    shape = gen_resource_variable_ops.variable_shape(
        forward_input, name="shape_init")
  else:
    shape = array_ops.shape(forward_input, name="shape_init")

  return indexed_slices.IndexedSlices(
      values=values, indices=indices, dense_shape=shape)


def _rewrite_grad_indexed_slices_output(old_output_slices, new_input_slices):
  """Creates a new version of old_output_slices with new_input_slices as input.

  This method assumes that old_output_slices.{values,indices} are produced by
  concatenating the incoming gradient Tensor input with the IndexedSlices
  produced by the gradient computation of the while body. See
  backprop.aggregate_indexed_slices_gradients for where these concats are
  constructed. We build new concats that use new_input_slices instead of the
  original Tensor input.

  Args:
    old_output_slices: original IndexedSlices output of while gradient.
    new_input_slices: new IndexedSlices to use as input to while gradient.

  Returns:
    A new IndexedSlices to replace old_output_slices.
  """

  def rewrite(old_output, new_input):
    assert old_output.type == "Identity"
    concat_op = old_output.inputs[0].op
    assert concat_op.type == "ConcatV2"
    # Don't include axis arg
    old_concat_args = concat_op.inputs[:-1]
    # We assume that the original gradient input was the first argument to the
    # concat op.
    # TODO(skyewm): do this in a more robust way.
    return array_ops.concat([new_input] + old_concat_args[1:], 0)

  values = rewrite(old_output_slices.values.op, new_input_slices.values)
  indices = rewrite(old_output_slices.indices.op, new_input_slices.indices)
  return indexed_slices.IndexedSlices(
      values=values, indices=indices, dense_shape=new_input_slices.dense_shape)


def _update_indexed_slices_param(graph, loop_vars, init_slices, input_slices,
                                 output_slices, old_output_slices):
  """Updates graph with new IndexedSlices input/output.

  Updates graph's metadata to output the gradient computation defined by
  init_slices, input_slices, and output_slices, instead of outputting
  old_output_slices. Also returns a new version of loop_vars with init_slices
  replacing the old input.

  Args:
    graph: _WhileBodyGradFuncGraph.
    loop_vars: the inputs to graph.
    init_slices: the new IndexedSlices to use as input to graph.
    input_slices: the new IndexedSlices in graph that should be fed by
      init_slices.
    output_slices: the new IndexedSlices in graph that should be the
      corresponding output to input_slices.
    old_output_slices: the IndexedSlices in graph that are currently being
      output.

  Returns:
    New loop_vars to pass to graph.
  """
  structured_idx = _get_tensor_index_in_iterable(graph.structured_outputs,
                                                 old_output_slices)
  # We assume that the component tensors of old_output_slices appear
  # sequentially in graph.outputs. We use the first of these tensors
  # as the reference index.
  flat_idx = _get_tensor_index_in_iterable(
      graph.outputs,
      func_graph.flatten(old_output_slices)[0])

  graph.structured_outputs[structured_idx] = output_slices
  graph.outputs = func_graph.flatten(graph.structured_outputs)

  graph.inputs = (
      graph.inputs[:flat_idx] + _flatten(input_slices) +
      graph.inputs[flat_idx + 1:])

  return loop_vars[:flat_idx] + _flatten(init_slices) + loop_vars[flat_idx + 1:]


def _flatten(arg):
  return nest.flatten(arg, expand_composites=True)
