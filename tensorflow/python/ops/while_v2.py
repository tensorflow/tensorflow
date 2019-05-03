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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import nest

# pylint: disable=protected-access

# TODO(b/79881896): Handle external control dependencies. tf.while_loop allows
# control dependencies on external nodes with at least 1 output.
# Another idea is to create const nodes outside the loop and add control edges
# to them and then pass those in as data inputs. This should probably be
# handled in the CapturingGraph itself.


def while_loop(cond,
               body,
               loop_vars,
               shape_invariants=None,
               parallel_iterations=10,
               maximum_iterations=None,
               name=None,
               return_same_structure=True):
  """Like tf.while_loop, except emits a single While op."""
  # Keep the original loop_vars around to know which args were TensorArrays.
  orig_loop_vars = loop_vars
  # Cache its length since we use it at multiple places below.
  len_orig_loop_vars = len(orig_loop_vars)

  # Convert TensorArrays to their flow variables. These get converted back to
  # TensorArrays before calling `cond` and `body`. See `wrapped_cond` and
  # `wrapped_body` below.
  loop_vars = list(_tensor_array_to_flow(orig_loop_vars))
  loop_vars = nest.map_structure(
      ops.internal_convert_to_tensor_or_indexed_slices, loop_vars,
      expand_composites=True)
  if shape_invariants is not None:
    nest.assert_same_structure(orig_loop_vars, shape_invariants,
                               expand_composites=False)
    shape_invariants = nest.map_structure(
        control_flow_ops._get_shape_invariant, loop_vars,
        list(shape_invariants), expand_composites=False)
  else:
    shape_invariants = nest.map_structure(
        control_flow_ops._get_shape_invariant, loop_vars,
        expand_composites=False)
  if not name:
    name = "while"

  with ops.name_scope(name) as scope:
    with ops.name_scope(None):
      cond_name = util.unique_fn_name(scope, "cond")
      body_name = util.unique_fn_name(scope, "body")
    maximum_iterations_loop_var = _build_maximum_iterations_loop_var(
        maximum_iterations)
    loop_counter = constant_op.constant(
        0,
        dtype=maximum_iterations_loop_var.dtype
        if maximum_iterations is not None else None,
        name="loop_counter")
    # Add loop counter needed for computing gradients.
    loop_vars = [loop_counter, maximum_iterations_loop_var] + loop_vars

    shape_invariants = type(shape_invariants)(
        [tensor_shape.scalar(), tensor_shape.scalar()]) + shape_invariants

    # Automatic control dependencies are added in defuns, but not in v1
    # graphs. Propagate that behavior here.
    add_control_dependencies = ops.get_default_graph()._add_control_dependencies

    # Build a `cond` wrapper that can handle the extra counter loop_var.
    def wrapped_cond(loop_counter, maximum_iterations_arg, *args):
      # Convert the flow variables in `args` to TensorArrays. `args` should
      # already have the same structure as `orig_loop_vars` but currently there
      # is no nest.zip so we call `_pack_sequence_as` which flattens both
      # `orig_loop_vars` and `args`, converts flows in `args` to TensorArrays
      # and packs it into the structure of `orig_loop_vars`.
      if maximum_iterations is None:
        return cond(*_pack_sequence_as(orig_loop_vars, args))
      else:
        return math_ops.logical_and(
            loop_counter < maximum_iterations_arg,
            cond(*_pack_sequence_as(orig_loop_vars, args)))

    # NOTE(skyewm): we set collections to the outer graph's collections for
    # compatibility with TPUEstimator.
    cond_graph = func_graph_module.func_graph_from_py_func(
        cond_name,
        wrapped_cond,
        [],  # We provide signature instead of args.
        {},
        signature=_build_signature(loop_vars, shape_invariants),
        func_graph=util.WhileCondFuncGraph(
            cond_name, collections=ops.get_default_graph()._collections),  # pylint: disable=protected-access
        add_control_dependencies=add_control_dependencies)

    def wrapped_body(loop_counter, maximum_iterations_arg, *args):
      """Loop body augmented with counter update.

      Args:
        loop_counter: Loop counter which needs to be incremented in the body.
        maximum_iterations_arg: Maximum iterations of the loop.
        *args: List of args

      Returns:
        A list of tensors the same length as args.
      """
      # Capture the tensors already captured in cond_graph so that they appear
      # in the same order in body_graph.external_captures.
      for t in cond_graph.external_captures:
        ops.get_default_graph().capture(t)

      # Convert the flow variables in `args` to TensorArrays. `args` should
      # already have the same structure as `orig_loop_vars` but currently there
      # is no nest.zip so we call `_pack_sequence_as` which flattens both
      # `orig_loop_vars` and `args`, converts flows in `args` to TensorArrays
      # and packs it into the structure of `orig_loop_vars`.
      outputs = body(*_pack_sequence_as(orig_loop_vars, args))
      if not nest.is_sequence_or_composite(outputs):
        outputs = [outputs]
      # Compare the structure of input and output of body converting the
      # top-level tuples to list to be compatible with legacy while_loop.
      nest.assert_same_structure(list(outputs), list(orig_loop_vars),
                                 expand_composites=True)

      outputs = _tensor_array_to_flow(outputs)

      # TODO(srbs): Update lowering code to create _Enter nodes with
      # is_constant=True for inputs that are directly passed to outputs.
      return [loop_counter + 1, maximum_iterations_arg] + list(outputs)

    body_graph = func_graph_module.func_graph_from_py_func(
        body_name,
        wrapped_body,
        [],  # We provide signature instead of args.
        {},
        signature=_build_signature(loop_vars, shape_invariants),
        func_graph=util.WhileBodyFuncGraph(
            body_name, collections=ops.get_default_graph()._collections),  # pylint: disable=protected-access
        add_control_dependencies=add_control_dependencies)
    # Add external captures of body to the list of loop vars.
    # Note that external tensors will be treated as loop invariants, i.e.,
    # the value of that tensor in each iteration is the same as it was at the
    # beginning of the loop execution.
    loop_vars = loop_vars + body_graph.external_captures
    # TODO(srbs): Update lowering code to create _Enter nodes with
    # is_constant=True for inputs that are directly passed to outputs.
    body_graph.outputs.extend(body_graph.internal_captures)

    # Capture the extra `external_captures` of `body_graph` in `cond_graph` so
    # that it expects to receive those as arguments.
    with cond_graph.as_default():
      num_cond_captures = len(cond_graph.external_captures)
      assert (cond_graph.external_captures ==
              body_graph.external_captures[:num_cond_captures])
      for body_capture in body_graph.external_captures[num_cond_captures:]:
        assert body_capture not in cond_graph.captures
        cond_graph.capture(body_capture)

    # Make sure that the shapes of the loop outputs are compatible with the
    # shape invariants, or the shapes of the loop vars if the invariants are not
    # specified.
    num_flattened_outputs = len(nest.flatten(orig_loop_vars,
                                             expand_composites=True))
    # First var is loop counter and second var is maximum_iterations.
    first_loop_var_index = 2
    _check_shapes_compat(
        body_graph.outputs[first_loop_var_index:first_loop_var_index +
                           num_flattened_outputs],
        nest.flatten(
            shape_invariants[first_loop_var_index:first_loop_var_index +
                             len_orig_loop_vars], expand_composites=True),
        nest.flatten(loop_vars[first_loop_var_index:first_loop_var_index +
                               len_orig_loop_vars], expand_composites=True))
    flattened_loop_vars = nest.flatten(loop_vars, expand_composites=True)
    _check_num_inputs_outputs(cond_graph, body_graph,
                              len(flattened_loop_vars))

    with ops.control_dependencies(
        list(cond_graph.control_captures) + list(body_graph.control_captures)):
      outputs = gen_functional_ops._while(
          flattened_loop_vars,
          util.create_new_tf_function(cond_graph),
          util.create_new_tf_function(body_graph),
          output_shapes=[t.shape for t in body_graph.outputs],
          parallel_iterations=parallel_iterations,
          name=scope)

    _copy_handle_data(body_graph.outputs, outputs)
    util.maybe_set_lowering_attr(outputs[0].op)
    util.maybe_propagate_compile_time_consts_in_xla(outputs[0].op)

    # Return identities for each output of the While op, rather than the output
    # of the While op directly. This makes pruning work if the output of
    # while_loop() is fetched: the lowering pass converts the While outputs into
    # IdentityN outputs, which if fetched will cause all ops in the body to be
    # run (since it takes all exit ops as input). After lowering, each output
    # identity op will end up with only the appropriate exit op as input.
    outputs = tuple(array_ops.identity(t) for t in outputs)

  outputs = _pack_sequence_as(
      orig_loop_vars, outputs[first_loop_var_index:first_loop_var_index +
                              num_flattened_outputs])

  if return_same_structure:
    return outputs

  flattened_outputs = nest.flatten(outputs, expand_composites=True)
  if len(flattened_outputs) == 1:
    return flattened_outputs[0]
  else:
    return outputs


@ops.RegisterGradient("While")
def _WhileGrad(op, *grads):  # pylint: disable=invalid-name
  """The gradient of a While op produced by while_loop."""
  # Note that op is not always the same as while_op because the gradient tape,
  # for eager mode compatibility, forgets information about the proper op. Since
  # the loop cannot run in eager mode, however, we can safely introspect into
  # the graph here.
  while_op = op.outputs[0].op
  cond_graph = _get_graph(while_op, "cond")
  body_graph = _get_graph(while_op, "body")
  orig_num_params = len(body_graph.outputs)

  maximum_iterations = op.inputs[1]
  parallel_iterations = op.get_attr("parallel_iterations")

  grads = [_preprocess_grad(grad, body_out, while_out)
           for grad, body_out, while_out
           in zip(grads, body_graph.outputs, while_op.outputs)]

  # We compute the gradient for the sub-graph between trainable ys and xs
  # with non-None incoming gradients. We later pad the None's to the list of
  # outputs.
  ys, xs, non_none_grads = zip(*[(y, x, grad) for (y, x, grad) in zip(
      body_graph.outputs, body_graph.inputs, grads) if grad is not None])

  body_grad_graph, args = _create_grad_func(
      ys, xs, non_none_grads, cond_graph, body_graph,
      util.unique_grad_fn_name(body_graph.name), op, maximum_iterations)

  if body_grad_graph.while_op_needs_rewrite:
    # Modify 'op' to output the intermediate accumulators needed by the grad
    # function.
    # NOTE(skyewm): if there are any active sessions, this modification to `op`
    # may make them unrunnable!

    cond_graph.name += "_rewritten"
    body_graph.name += "_rewritten"

    new_inputs = body_grad_graph.empty_tensor_lists
    new_outputs = body_graph.outputs[orig_num_params:]

    while_op._set_func_attr("cond", util.create_new_tf_function(cond_graph))
    while_op._set_func_attr("body", util.create_new_tf_function(body_graph))
    while_op._set_type_list_attr("T", body_graph.output_types)
    while_op._set_shape_list_attr("output_shapes", body_graph.output_shapes)
    while_op._add_while_inputs(new_inputs)
    while_op._add_outputs([t.dtype for t in new_outputs],
                          [t.shape for t in new_outputs])
    _copy_handle_data(new_outputs, op.outputs[orig_num_params:])

  captured_inputs = _resolve_grad_captures(body_graph, body_grad_graph,
                                           while_op)
  loop_vars = args + captured_inputs

  # This modifies body_grad_graph.
  loop_vars = while_v2_indexed_slices_rewriter.rewrite_grad_indexed_slices(
      grads, body_grad_graph, loop_vars, while_op.inputs)

  def grad_cond(counter, unused_maximum_iterations_arg, forward_loop_iters,
                *unused_args):
    return counter < forward_loop_iters

  grad_cond_name = util.unique_grad_fn_name(op.get_attr("cond").name)
  cond_grad_graph = func_graph_module.func_graph_from_py_func(
      grad_cond_name, grad_cond, loop_vars, {},
      func_graph=util.WhileCondFuncGraph(grad_cond_name))

  _check_num_inputs_outputs(cond_grad_graph, body_grad_graph, len(loop_vars))

  outputs = gen_functional_ops._while(
      loop_vars,
      util.create_new_tf_function(cond_grad_graph),
      util.create_new_tf_function(body_grad_graph),
      output_shapes=[t.shape for t in body_grad_graph.outputs],
      parallel_iterations=parallel_iterations,
      name="%s_grad" % while_op.name)
  grad_op = outputs[0].op

  _copy_handle_data(body_grad_graph.outputs, outputs)
  util.maybe_set_lowering_attr(grad_op)
  util.maybe_propagate_compile_time_consts_in_xla(grad_op)

  # See comment in while_loop.
  outputs = [array_ops.identity(t) for t in outputs]
  return _get_structured_grad_output(outputs, grads, body_grad_graph)


def _preprocess_grad(grad, body_graph_output, while_op_output):
  """Returns the initial gradient to be used for a given output tensor.

  Args:
    grad: the original gradient Tensor passed to the gradient function.
    body_graph_output: the corresponding Tensor in the body graph.
    while_op_output: the corresponding Tensor output of the While op.

  Returns:
    A Tensor or None.
  """
  # Set the incoming gradient of non-trainable inputs to None. It is possible
  # that we receive non-None gradients for non-trainable types in nested while
  # loops because we accumulate outputs of the inner while as variant tensors
  # which are trainable and hence receive zeros_like tensors in the gradient
  # pass. The non-trainable tensors then receive the popped zeros tensor from
  # this zeros variant. The gradient for the loop vars corresponding to these
  # tensors is None or zeros (this happens only if the loop var is accumulated
  # as well) in _grad_fn so we reset these.
  # TODO(b/118712257): Remove once we can handle None output grads in _grad_fn.
  if not _is_trainable(body_graph_output):
    return None

  # GradientTape initializes resource and variant grads as None instead of
  # zeros. Set to zeros so _GradientsHelper computes the gradients instead of
  # returning None.
  if (while_op_output.dtype in (dtypes.resource, dtypes.variant)
      and grad is None):
    return _zeros_like(while_op_output)

  return grad


# TODO(skyewm): make this return constants if op_output's shape is fully
# defined (this can be done by checking the "shape" attr of resource vars).
def _zeros_like(op_output):
  """Like array_ops.zeros_like() but also accepts resource var handles."""
  if op_output.dtype == dtypes.resource:
    return array_ops.zeros(
        gen_resource_variable_ops.variable_shape(op_output))
  return array_ops.zeros_like(op_output)


def _is_trainable(tensor):
  """Returns whether the given tensor is trainable."""
  if not gradients_util.IsTrainable(tensor):
    return False

  # Special case: untrainable accumulator output. The gradients algorithm
  # doesn't know about tensor lists of untrainable elements. In theory the
  # tensor list gradient functions should return None as appropriate, but
  # because we can't return None from the gradient function we filter out
  # untrainable accumulator output here to avoid computing the gradient at all.
  if tensor.op.type == "TensorListPopBack" and tensor.value_index == 0:
    assert tensor.dtype == dtypes.variant
    element_type = tensor.op.get_attr("element_dtype")
    return gradients_util.IsTrainable(element_type)

  return True


# TODO(srbs): Pull this into common utils for cond_v2 and while_v2.
def _get_graph(while_op, func_attr_name):
  """Returns `FuncGraph` for the given function attribute.

  Args:
    while_op: The While Operation.
    func_attr_name: string

  Returns:
    `FuncGraph`
  """
  # TODO(srbs): Handle TensorShapeProto in function_def_to_graph.input_shapes.
  input_shapes = [
      tensor_shape.TensorShape(s) for s in while_op.get_attr("output_shapes")
  ]
  func_name = while_op.get_attr(func_attr_name).name
  fdef = while_op.graph._get_function(func_name).definition
  # `while_op.graph` may not be the same as `ops.get_default_graph()` e.g.
  # if the `while_op` is in the body of another if/while/defun. We build the
  # `func_graph` with `while_op.graph` as its `outer_graph`. This resembles how
  # the `FuncGraph` was built in the forward pass. We need this so that we can
  # appropriately capture references to outer tensors in the nested grad graphs.
  with while_op.graph.as_default():
    func_graph = function_def_to_graph.function_def_to_graph(fdef, input_shapes)
  func_graph._while = while_op
  return func_graph


def _create_grad_func(ys, xs, grads, cond_graph, body_graph, name, while_op,
                      maximum_iterations):
  """Builds and returns the gradient FuncGraph of `func_graph` and its args.

  The returned grad_func_graph must be called with the returned
  args + grad_func_graph.captures.

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    grads: The incoming grads for `ys`.
    cond_graph: FuncGraph for the forward cond function.
    body_graph: FuncGraph for the forward body function.
    name: Name of the returned gradient function.
    while_op: The forward While op.
    maximum_iterations: Tensor. The maximum number of iterations.

  Returns:
    2-tuple of (grad_func_graph, args).
  """
  assert len(ys) == len(grads)

  total_iters = while_op.outputs[0]
  counter = constant_op.constant(
      0, dtype=total_iters.dtype, name="grad_counter")

  args = [counter, maximum_iterations, total_iters] + list(grads)
  # Note: The returned function does not have `args` in the list of
  # `external_captures`.
  grad_func_graph = func_graph_module.func_graph_from_py_func(
      name,
      lambda *args: _grad_fn(ys, xs, args, body_graph),
      args, {},
      func_graph=_WhileBodyGradFuncGraph(name, cond_graph, body_graph,
                                         maximum_iterations, while_op))

  # Add the popped accumulators to the list of outputs.
  for internal_capture in grad_func_graph.internal_captures:
    if internal_capture in grad_func_graph.popped_tensor_lists:
      new_output = grad_func_graph.popped_tensor_lists[internal_capture]
    elif internal_capture.dtype == dtypes.resource:
      new_output = internal_capture
    else:
      raise ValueError("Tensor %s is in list of internal_captures but is"
                       " neither a resource nor is in popped_tensor_lists." %
                       str(internal_capture))
    grad_func_graph.outputs.append(new_output)
    grad_func_graph.structured_outputs.append(new_output)

  return grad_func_graph, args


def _grad_fn(ys, xs, args, func_graph):
  """Computes the gradient of `func_graph` in the current graph.

  This function builds the gradient graph of the corresponding forward-pass
  `func_graph` by differentiating `func_graph`'s outputs w.r.t. its inputs.

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    args: The input arguments.
      args[0] - Loop counter
      args[1] - Total number of iterations.
      args[2] - maximum_iterations.
      args[3:] - Incoming gradients for `ys`.
    func_graph: function.FuncGraph. The corresponding forward-pass function.

  Returns:
    The output gradient Tensors.
  """
  grad_ys = args[3:]

  # Build the gradient graph. Note that this builds the gradient computation of
  # func_graph in the current graph, which requires capturing tensors from
  # func_graph. The captured func_graph tensors are resolved to external tensors
  # after the forward While op has been rewritten in _resolve_grad_captures.
  # TODO(srbs): Mark GradientsHelper as public?
  grad_outs = gradients_util._GradientsHelper(
      ys, xs, grad_ys=grad_ys, src_graph=func_graph,
      unconnected_gradients="zero")

  # TODO(b/118712257): Handle the case when grad_outs has None's e.g. when there
  # is a tf.StopGradient in the loop body.
  assert all(g is not None for g in grad_outs)
  counter = args[0]
  maximum_iterations = args[1]
  total_iters = args[2]
  return [counter + 1, maximum_iterations, total_iters] + grad_outs


def _resolve_grad_captures(body_graph, body_grad_graph, while_op):
  """Returns the tensors to pass as captured inputs to `body_grad_graph`.

  `body_grad_graph` may have external references to:
  1. Its outer graph containing the input gradients. These are left as-is.
  2. Accumulators captured from the forward-pass graph. These should have been
     added as `while_op` outputs after the gradient graph was built. We replace
     these with the corresponding output of `while_op`, i.e. a tensor in
     `body_graph.outer_graph`. In the case of nested control flow or functions,
     the gradient logic handling `body_grad_graph.outer_graph` will make sure
     the tensor from `body_graph.outer_graph` is also correctly captured.

  Args:
    body_graph: FuncGraph. The forward-pass body function.
    body_grad_graph: FuncGraph. The body gradients function.
    while_op: The forward-pass While Operation calling `body_graph`.

  Returns:
    A list of input tensors to be passed as the captured inputs to
      `body_grad_graph`.
  """
  new_capture_inputs = []
  for t in body_grad_graph.external_captures:
    # All values captured by gradient computation should be from the forward
    # graph or a captured resource variable (note that input gradients are
    # regular non-captured inputs).
    if t.graph == body_graph:
      # Captured accumulator
      t = while_op.outputs[t.graph.outputs.index(t)]
      # Note: We rely on the capturing logic of the gradient While op graph to
      # correctly capture the tensors in `body_graph.outer_graph`. Both cond_v2
      # and while_v2 handle this while building their gradient functions.
      assert t.graph == body_graph.outer_graph
    else:
      # Captured resource variable
      assert t.dtype == dtypes.resource

    new_capture_inputs.append(t)
  return new_capture_inputs


def _get_structured_grad_output(outputs, grads, body_grad_graph):
  """Returns the values that should be returned from the while grad function.

  Args:
    outputs: the raw Tensor outputs of the grad While op.
    grads: the input gradients to the gradient function.
    body_grad_graph: _WhileBodyGradFuncGraph.

  Returns:
    A list of gradient values. May include Nones.
  """
  result = []
  # outputs[0] is the loop counter.
  # outputs[1] is maximum_iterations.
  # outputs[2] is the total number of loop iterations.
  outputs_idx = 3
  structured_outputs_idx = 3
  for g in grads:
    # Set None as the output gradient for tensors with None input gradient.
    if g is None:
      result.append(None)
      continue
    output = body_grad_graph.structured_outputs[structured_outputs_idx]
    structured_outputs_idx += 1
    if isinstance(output, ops.IndexedSlices):
      # TODO(skyewm): is there a more robust way to determine the order of
      # flattened IndexedSlices components?
      result.append(ops.IndexedSlices(
          values=outputs[outputs_idx],
          indices=outputs[outputs_idx + 1],
          dense_shape=outputs[outputs_idx + 2]))
      outputs_idx += 3
    else:
      assert isinstance(output, ops.Tensor)
      result.append(outputs[outputs_idx])
      outputs_idx += 1

  return result


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
  assert isinstance(tensor.graph, func_graph_module.FuncGraph)

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


class _WhileBodyGradFuncGraph(util.WhileBodyFuncGraph):
  """FuncGraph for the gradient function of the body of a While op.

  Contains the logic for capturing the tensors from the body of the forward
  While op which is as follows:
  1. If the tensor is of resource type (these are not accumulated):
     a. Ensure that the tensor is a loop invariant, i.e., it exists in both loop
        inputs and outputs at the same index.
     b. Lookup the corresponding resource tensor in the forward outer graph and
        try to capture that.
  2. If the tensor is not of resource type:
     a. Create an accumulator for that tensor and output it from the forward
        pass. Note this also requires adding it as an input to the forward pass.
     b. Capture the accumulator from the forward pass in this FuncGraph. This
        will later be resolved to the correct output of the forward While op.
     c. Pop a value from the captured placeholder and use it as the captured
        value for the forward pass tensor.

  This only allows capturing tensors in the forward graph. A ValueError is
  raised if an attempt is made to capture a tensor not in the forward graph.
  To manually capture capture a tensor that is not in the forward graph, call
  `capture` with `whitelisted=True`.

  Note: The `captures` dict does not contain the forward tensor since it is not
  directly captured. It contains the accumulator corresponding to this forward
  tensor.

  Attributes:
    while_op_needs_rewrite: True if any non-resource intermediates were
      captured, meaning the forward While op needs to be rewritten to output the
      corresponding accumulators.
    empty_tensor_lists: list of EmptyTensorList tensors to be used as initial
      input to the new accumulators in the forward graph.
    popped_tensor_lists: dict from the captured accumulator placeholder to the
      TensorList obtained after popping the intermediate tensor from it. The
      values of this dict need to be added to the list of outputs.
  """

  def __init__(self, name, forward_cond_graph, forward_body_graph,
               maximum_iterations, forward_while_op):
    super(_WhileBodyGradFuncGraph, self).__init__(name)
    self.empty_tensor_lists = []
    self.popped_tensor_lists = {}
    # FuncGraph for the body of the forward While op.
    self._forward_graph = forward_body_graph
    # FuncGraph for the cond of the forward While op.
    self._forward_cond_graph = forward_cond_graph
    self._maximum_iterations = maximum_iterations
    self._forward_while_op = forward_while_op
    # Dict from forward intermediate tensor to its indirectly captured tensor
    # in this graph. Indirect capturing happens in two ways:
    # 1. For non-resource tensors we capture their accumulators from the forward
    #    outer graph and pop values from that accumulator inside this graph
    #    using TensorListPopBack.
    # 2. For resource tensors we directly capture their corresponding tensor
    #    in the forward outer graph.
    self._indirect_captures = {}

  @property
  def while_op_needs_rewrite(self):
    return self.empty_tensor_lists

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
      raise ValueError("Attempting to capture tensor %s which is not in the "
                       "forward graph but in %s." %
                       (str(tensor), _graph_name(tensor.graph)))
    return super(_WhileBodyGradFuncGraph, self).capture(tensor, name)

  def _capture_helper(self, tensor, name):
    if tensor.graph is not self._forward_graph:
      return super(_WhileBodyGradFuncGraph, self)._capture_helper(tensor, name)

    while tensor.op.type == "Identity":
      # We do not accumulate the output of identity nodes so we try to capture
      # the input of the Identity node instead.
      tensor = tensor.op.inputs[0]

    captured_tensor = self._indirect_captures.get(tensor)
    if captured_tensor is not None:
      return captured_tensor

    # Resource tensors are not accumulated and handled specially.
    if tensor.dtype == dtypes.resource:
      return self._resource_capture_helper(tensor)

    # Create or find an existing accumulator output for `tensor` in the forward
    # graph, and fetch from this accumulator in the gradient graph to get the
    # raw intermediate value.
    accumulator = _get_accumulator(tensor)
    if accumulator is None:
      # Create the initial empty tensor list.
      #
      # Note: We clear the control dependencies to avoid a cycle in case a
      # control tensor has an input path to an output of the  forward While.
      #
      # E.g.:
      # x = tf.while_loop(...)
      # y = f(x)
      # with tf.control_dependencies([y]):
      #   tf.gradients(y, x)
      #
      # Since the EmptyTensorList is fed back into the forward While, not
      # removing the control edge would cause a cycle.
      with self._forward_graph.outer_graph.as_default():
        with util.clear_control_inputs():
          tensor_list = list_ops.empty_tensor_list(
              element_dtype=tensor.dtype,
              element_shape=tensor.shape,
              max_num_elements=self._maximum_iterations,
              name=_build_accumulator_name(tensor))
      self.empty_tensor_lists.append(tensor_list)

      # Push the intermediate tensor to the tensor list. This captures
      # `tensor_list`.
      with self._forward_graph.as_default():
        accumulator = list_ops.tensor_list_push_back(tensor_list, tensor)
      # Add the modified tensor list to the list of outputs. This output will be
      # all the accumulated values.
      self._forward_graph.outputs.append(accumulator)

      # Capture in the cond graph as well so the forward cond and body inputs
      # match.
      with self._forward_cond_graph.as_default():
        self._forward_cond_graph.capture(tensor_list)

    # Capture the accumulator tensor list in the gradient graph directly from
    # the forward graph -- we'll later modify this to capture the final list
    # output by the forward While op instead.
    captured_accumulator = super(_WhileBodyGradFuncGraph, self)._capture_helper(
        accumulator, name)

    # Pop the intermediate value from the tensor list in the gradient graph.
    new_tensor_list, captured_tensor = list_ops.tensor_list_pop_back(
        captured_accumulator, element_dtype=tensor.dtype)

    self._indirect_captures[tensor] = captured_tensor
    self.popped_tensor_lists[captured_accumulator] = new_tensor_list
    return captured_tensor

  def _resource_capture_helper(self, tensor):
    """Returns the captured resource tensor.

    Resource-type tensors are not accumulated. If a resource tensor exists in
    the loop body it must either be a loop input or an output of a nested While
    op inside the loop body which had captured the external resource.

    Args:
      tensor: the external resource Tensor to be captured.

    Returns:
      Tensor in this graph.
    """
    assert tensor.dtype == dtypes.resource

    index = util.resource_input_index(
        tensor.name, [t.name for t in self._forward_graph.inputs],
        {op.name: op.node_def for op in self._forward_graph.get_operations()},
        self._forward_graph._functions)

    input_placeholder = self._forward_graph.inputs[index]
    tensor_in_outer_graph = self._forward_graph._while.inputs[index]

    assert input_placeholder.dtype == dtypes.resource
    assert tensor_in_outer_graph.dtype == dtypes.resource
    # This must be a loop invariant.
    assert input_placeholder == self._forward_graph.outputs[index], (
        "Resource tensors must be loop invariants %s." %
        tensor_in_outer_graph)

    self._indirect_captures[tensor] = self.capture(
        tensor_in_outer_graph, whitelisted=True)
    return self._indirect_captures[tensor]


def _check_shapes_compat(output_tensors, shape_invariants, input_tensors):
  for (t, shape, input_t) in zip(output_tensors, shape_invariants,
                                 input_tensors):
    if not control_flow_ops._ShapeLessThanOrEqual(t.shape, shape):
      raise ValueError(
          "Input tensor '%s' enters the loop with shape %s, but has "
          "shape %s after one iteration. To allow the shape to vary across "
          "iterations, use the `shape_invariants` argument of tf.while_loop to "
          "specify a less-specific shape." % (input_t.name, shape, t.shape))


def _check_num_inputs_outputs(cond_graph, body_graph, num_flattened_loop_vars):
  """Checks the number of inputs/outputs of `cond_graph` and `body_graph`."""
  assert len(cond_graph.inputs) == num_flattened_loop_vars, (
      "cond_graph takes %d inputs; Expected: %d" % (len(cond_graph.inputs),
                                                    num_flattened_loop_vars))
  assert len(cond_graph.outputs) == 1, (
      "cond_graph has %d outputs; Expected: 1" % len(cond_graph.outputs))
  assert len(body_graph.inputs) == num_flattened_loop_vars, (
      "body_graph takes %d inputs; Expected: %d" % (len(body_graph.inputs),
                                                    num_flattened_loop_vars))
  assert len(body_graph.outputs) == num_flattened_loop_vars, (
      "body_graph has %d outputs; Expected: %d" % (len(body_graph.outputs),
                                                   num_flattened_loop_vars))


def _copy_handle_data(src_tensors, tgt_tensors):
  for src_t, tgt_t in zip(src_tensors, tgt_tensors):
    custom_gradient.copy_handle_data(src_t, tgt_t)


def _graph_name(graph):
  if isinstance(graph, func_graph_module.FuncGraph):
    return graph.name
  return "Base"


def _pack_sequence_as(structure_with_tas, loop_vars):
  """Like `nest.pack_sequence_as` but also replaces flows with TensorArrays."""

  def flow_to_tensor_array(flow, ta):  # pylint: disable=missing-docstring
    return (tensor_array_ops.build_ta_with_new_flow(ta, flow) if isinstance(  # pylint: disable=g-long-ternary
        ta, tensor_array_ops.TensorArray) else flow)

  flattened_loop_vars = [
      flow_to_tensor_array(*z)
      for z in zip(nest.flatten(loop_vars, expand_composites=True),
                   nest.flatten(structure_with_tas, expand_composites=True))
  ]
  return nest.pack_sequence_as(structure_with_tas, flattened_loop_vars,
                               expand_composites=True)


def _tensor_array_to_flow(loop_vars):

  def f(maybe_ta):
    if isinstance(maybe_ta, tensor_array_ops.TensorArray):
      return maybe_ta.flow
    return maybe_ta

  return nest.map_structure(f, loop_vars, expand_composites=True)


def _build_signature(loop_vars, shape_invariants):
  return nest.pack_sequence_as(loop_vars, [
      tensor_spec.TensorSpec(s, t.dtype, name=t.op.name)
      for s, t in zip(nest.flatten(shape_invariants, expand_composites=True),
                      nest.flatten(loop_vars, expand_composites=True))
  ], expand_composites=True)


def _build_maximum_iterations_loop_var(maximum_iterations):
  if maximum_iterations is None:
    # Default value for max_num_elements to EmptyTensorList meaning that the
    # list size is unbounded.
    maximum_iterations = -1
  # EmptyTensorList expects `max_num_elements` to be of type int32.
  return ops.convert_to_tensor(
      maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")


def _build_accumulator_name(tensor):
  # Tensor name may be of the form "pow/y:0". Name scope does not allow ":".
  return "{}/accumulator".format(tensor.name).replace(":", "_")

# pylint: enable=protected-access
