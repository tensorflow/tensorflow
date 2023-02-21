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
import collections

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.capture import capture_container
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils

# pylint: disable=protected-access


def while_loop(cond,
               body,
               loop_vars,
               shape_invariants=None,
               parallel_iterations=10,
               maximum_iterations=None,
               name=None,
               return_same_structure=True,
               back_prop=True):
  """Like tf.while_loop, except emits a single While op."""
  loop_vars = variable_utils.convert_variables_to_tensors(loop_vars)
  # Keep the original loop_vars around to know which args were TensorArrays.
  orig_loop_vars = loop_vars
  flat_orig_loop_vars = nest.flatten(orig_loop_vars, expand_composites=True)
  # Cache its length since we use it at multiple places below.
  len_orig_loop_vars = len(orig_loop_vars)

  # Convert TensorArrays to their flow variables. These get converted back to
  # TensorArrays before calling `cond` and `body`. See `wrapped_cond` and
  # `wrapped_body` below.
  loop_vars = _tensor_array_to_flow(loop_vars)
  loop_vars = nest.map_structure(
      indexed_slices.internal_convert_to_tensor_or_indexed_slices,
      loop_vars,
      expand_composites=True)

  # `loop_vars_signature` is a structure of TypeSpecs and has the same
  # structure with the `orig_loop_vars`. If `shape_invariants` is not None, its
  # shape information comes from `shape_invariants` instead of `orig_loop_vars`.
  # It is used to pack flattened vars into structured vars.
  if shape_invariants is not None:
    loop_vars_signature = nest.map_structure(
        control_flow_ops._shape_invariant_to_type_spec,
        loop_vars, shape_invariants)
  else:
    loop_vars_signature = nest.map_structure(
        control_flow_ops._shape_invariant_to_type_spec, loop_vars)

  flat_shape_invariants = nest.map_structure(
      lambda spec: spec.shape,
      nest.flatten(loop_vars_signature, expand_composites=True))

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
    loop_vars = [loop_counter, maximum_iterations_loop_var] + list(loop_vars)

    func_graph_signature = (
        [tensor_spec.TensorSpec.from_tensor(loop_counter),
         tensor_spec.TensorSpec.from_tensor(maximum_iterations_loop_var)] +
        list(loop_vars_signature))

    # Automatic control dependencies are added in defuns, but not in v1
    # graphs. Propagate that behavior here.
    add_control_dependencies = ops.get_default_graph()._add_control_dependencies

    def wrapped_cond(loop_counter, maximum_iterations_arg, *args):
      """Extra `cond` wrapper that can handle the extra counter loop_var."""
      # Convert the flow variables in `args` to TensorArrays. `args` should
      # already have the same structure as `orig_loop_vars` but currently there
      # is no nest.zip so we call `_pack_sequence_as` which flattens `args`,
      # converts flows in `args` to TensorArrays and packs it into the
      # structure of `loop_vars_signature`.
      pred = cond(
          *_pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, args))
      if (tensor_util.is_tf_type(pred) and
          (pred.shape.dims is None or pred.shape.dims)):
        pred = array_ops.squeeze_v2(pred)

      if maximum_iterations is None:
        return pred
      else:
        return math_ops.logical_and(
            loop_counter < maximum_iterations_arg, pred)

    # NOTE(skyewm): we set collections to the outer graph's collections for
    # compatibility with TPUEstimator.
    cond_graph = func_graph_module.func_graph_from_py_func(
        cond_name,
        wrapped_cond,
        [],  # We provide signature instead of args.
        {},
        signature=func_graph_signature,
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
      # The function was created with a signature rather than tensors, so
      # internal placeholders were created without handle data.
      _copy_handle_data(nest.flatten(loop_vars[2:], expand_composites=True),
                        nest.flatten(args, expand_composites=True))
      # Capture the tensors already captured in cond_graph so that they appear
      # in the same order in body_graph.external_captures.
      for t in cond_graph.external_captures:
        ops.get_default_graph().capture(t)

      # Convert the flow variables in `args` to TensorArrays. `args` should
      # already have the same structure as `orig_loop_vars` but currently there
      # is no nest.zip so we call `_pack_sequence_as` which flattens `args`,
      # converts flows in `args` to TensorArrays and packs it into the
      # structure of `loop_vars_signature`.
      outputs = body(
          *_pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, args))
      if not nest.is_nested(outputs):
        outputs = [outputs]
      try:
        # The legacy while_loop considers list and tuple to be the same
        # structure.
        nest.assert_same_structure(outputs, orig_loop_vars, check_types=False,
                                   expand_composites=True)
      except ValueError:
        # Traditionally we consider variables and tensors to be the same
        # structure.
        vars1 = variable_utils.convert_variables_to_tensors(outputs)
        vars2 = variable_utils.convert_variables_to_tensors(orig_loop_vars)
        nest.assert_same_structure(vars1, vars2, check_types=False,
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
        signature=func_graph_signature,
        func_graph=util.WhileBodyFuncGraph(
            body_name, collections=ops.get_default_graph()._collections),  # pylint: disable=protected-access
        add_control_dependencies=add_control_dependencies)
    # Add external captures of body to the list of loop vars.
    # Note that external tensors will be treated as loop invariants, i.e.,
    # the value of that tensor in each iteration is the same as it was at the
    # beginning of the loop execution.
    deferred_external_captures = nest.flatten(
        [c() for c in body_graph.deferred_external_captures],
        expand_composites=True)
    loop_vars = (
        loop_vars + body_graph.external_captures + deferred_external_captures)
    # TODO(srbs): Update lowering code to create _Enter nodes with
    # is_constant=True for inputs that are directly passed to outputs.
    body_graph.outputs.extend(body_graph.internal_captures)
    body_graph.outputs.extend(body_graph.deferred_internal_captures)

    # Capture the extra `external_captures` of `body_graph` in `cond_graph` so
    # that it expects to receive those as arguments.
    with cond_graph.as_default():
      num_cond_captures = len(cond_graph.external_captures)
      assert (cond_graph.external_captures ==
              body_graph.external_captures[:num_cond_captures])
      _duplicate_body_captures_in_cond(
          cond_graph, body_graph.external_captures[num_cond_captures:] +
          deferred_external_captures)

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
        flat_shape_invariants,
        nest.flatten(loop_vars[first_loop_var_index:first_loop_var_index +
                               len_orig_loop_vars], expand_composites=True))

    num_original_outputs = len(body_graph.outputs)
    if back_prop and util.output_all_intermediates():
      # Export all tensors in the loop body that may be needed for gradient
      # computation. We do this by accumulating the intermediate values in
      # TensorLists.
      intermediate_tensors = _get_intermediates(body_graph)

      for intermediate_tensor in intermediate_tensors:
        tensor_list = list_ops.empty_tensor_list(
            element_dtype=intermediate_tensor.dtype,
            element_shape=intermediate_tensor.shape,
            max_num_elements=maximum_iterations)
        loop_vars.append(tensor_list)
        with cond_graph.as_default():
          # Add a placeholder to cond_graph's inputs corresponding to the
          # tensor_list.
          cond_graph.capture(tensor_list)
        with body_graph.as_default():
          # Push the intermediate tensor to the tensor list. This captures the
          # `tensor_list` as well.
          appended_tensor_list = list_ops.tensor_list_push_back(
              tensor_list, intermediate_tensor)
          # Add this modified tensor list to the list of outputs.
          body_graph.outputs.append(appended_tensor_list)

    flattened_loop_vars = nest.flatten(loop_vars, expand_composites=True)
    _check_num_inputs_outputs(cond_graph, body_graph,
                              len(flattened_loop_vars))
    _check_inputs_outputs_types_match(body_graph, flattened_loop_vars)

    with ops.control_dependencies(
        list(cond_graph._function_captures.control) + list(  # pylint: disable=protected-access
            body_graph._function_captures.control)):  # pylint: disable=protected-access
      output_shapes = [t.shape for t in body_graph.outputs]
      orig_loop_vars_range = slice(first_loop_var_index,
                                   first_loop_var_index + num_flattened_outputs)
      output_shapes[orig_loop_vars_range] = flat_shape_invariants

      outputs = _build_while_op(
          flattened_loop_vars,
          cond_graph,
          body_graph,
          output_shapes=output_shapes,
          parallel_iterations=parallel_iterations,
          name=scope,
          num_original_outputs=num_original_outputs)
    if not ops.get_default_graph().building_function:
      # In V1 graph mode, return identities for each output of the While op,
      # rather than the output of the While op directly. This makes pruning work
      # if the output of while_loop() is fetched: the lowering pass converts the
      # While outputs into IdentityN outputs, which if fetched will cause all
      # ops in the body to be run (since it takes all exit ops as input). After
      # lowering, each output identity op will end up with only the appropriate
      # exit op as input.
      outputs = tuple(array_ops.identity(t) for t in outputs)

  output_loop_vars = outputs[first_loop_var_index:first_loop_var_index +
                             num_flattened_outputs]
  if not back_prop:
    output_loop_vars = [array_ops.stop_gradient(t) for t in output_loop_vars]
  outputs = _pack_sequence_as(
      loop_vars_signature, flat_orig_loop_vars, output_loop_vars)

  if return_same_structure:
    return outputs

  flattened_outputs = nest.flatten(outputs, expand_composites=True)
  if len(flattened_outputs) == 1:
    return flattened_outputs[0]
  else:
    return outputs


@ops.RegisterGradient("StatelessWhile")
@ops.RegisterGradient("While")
def _WhileGrad(op, *grads):  # pylint: disable=invalid-name
  """The gradient of a While op produced by while_loop."""
  # Note that op is not always the same as while_op because the gradient tape,
  # for eager mode compatibility, forgets information about the proper op. Since
  # the loop cannot run in eager mode, however, we can safely introspect into
  # the graph here.
  while_op = op.outputs[0].op
  cond_graph = _get_graph(while_op, "cond", "_cond_graph")
  body_graph = _get_graph(while_op, "body", "_body_graph")
  orig_num_params = len(body_graph.outputs)

  maximum_iterations = op.inputs[1]
  parallel_iterations = op.get_attr("parallel_iterations")

  try:
    num_original_outputs = while_op.get_attr("_num_original_outputs")
  except:  # pylint: disable=bare-except
    num_original_outputs = len(while_op.outputs)

  num_intermediates = len(while_op.outputs) - num_original_outputs
  grads = [
      _preprocess_grad(grad, body_out, while_in, while_out)  # pylint: disable=g-complex-comprehension
      for grad, body_out, while_in, while_out in zip(
          grads[:num_original_outputs],
          body_graph.outputs[:num_original_outputs],
          while_op.inputs[:num_original_outputs],
          while_op.outputs[:num_original_outputs])
  ] + [None] * num_intermediates

  # Skip gradients with respect to the captures whenever possible.
  if "skip_input_indices" in op.__dict__ and op.skip_input_indices is not None:
    captures_start_index = (
        len(body_graph.inputs) - len(body_graph.internal_captures))
    for i in op.skip_input_indices:
      if i >= captures_start_index:
        grads[i] = None

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

    # `body_grad_graph.extra_inputs` here is equivalent to skimming off the new
    # `body_graph.external_captures` added during `_create_grad_func`.
    new_inputs = body_grad_graph.extra_inputs
    new_outputs = body_graph.outputs[orig_num_params:]

    while_op._set_func_attr("cond", util.create_new_tf_function(cond_graph))
    while_op._set_func_attr("body", util.create_new_tf_function(body_graph))
    if len(body_graph.output_types) != len(while_op.inputs) + len(new_inputs):
      # Continuing leads to an invalid graph with disconnected inputs.
      raise AssertionError(
          "Inputs and outputs constructed for the forward op of a While "
          "gradient don't match with 'output_types' at  "
          f"{len(body_graph.output_types)},'inputs' at length "
          f"{len(while_op.inputs)}, and 'new_inputs' at length "
          f"{len(new_inputs)}. This doesn't make sense, please file a bug.")
    while_op._set_type_list_attr("T", body_graph.output_types)
    while_op._set_shape_list_attr("output_shapes", body_graph.output_shapes)
    while_op._add_while_inputs(new_inputs)
    while_op._add_outputs([t.dtype for t in new_outputs],
                          [t.shape for t in new_outputs])
    _copy_handle_data(new_outputs, while_op.outputs[orig_num_params:])

  # Do not ignore grads wrt extra outputs when computing higher order
  # derivatives.
  while_op._set_attr("_num_original_outputs",
                     attr_value_pb2.AttrValue(i=len(while_op.outputs)))

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

  outputs = _build_while_op(
      loop_vars,
      cond_grad_graph,
      body_grad_graph,
      output_shapes=[t.shape for t in body_grad_graph.outputs],
      parallel_iterations=parallel_iterations,
      name="%s_grad" % while_op.name,
      num_original_outputs=len(body_grad_graph.outputs))

  # See comment in while_loop.
  outputs = [array_ops.identity(t) for t in outputs]
  return _get_structured_grad_output(outputs, grads, body_grad_graph)


def _build_while_op(loop_vars, cond_graph, body_graph, output_shapes,
                    parallel_iterations, name, num_original_outputs):
  """Builds the functional StatelessWhile/While op."""
  cond_stateful_ops = [
      op for op in cond_graph.get_operations() if op._is_stateful
  ]
  body_stateful_ops = [
      op for op in body_graph.get_operations() if op._is_stateful
  ]
  if (cond_stateful_ops or body_stateful_ops):
    op_fn = gen_functional_ops._while
  else:
    op_fn = gen_functional_ops.stateless_while

  def _make_op(inputs):
    while_op, tensors = util.get_op_and_outputs(op_fn(
        inputs,
        util.create_new_tf_function(cond_graph),
        util.create_new_tf_function(body_graph),
        output_shapes=output_shapes,
        parallel_iterations=parallel_iterations,
        name=name))
    _copy_handle_data(body_graph.outputs, tensors)
    util.maybe_set_lowering_attr(while_op)
    util.maybe_propagate_compile_time_consts_in_xla(while_op)
    _set_read_only_resource_inputs_attr(while_op, [cond_graph, body_graph])
    # This is needed so we do not compute derivative wrt these extra outputs.
    while_op._set_attr("_num_original_outputs",
                       attr_value_pb2.AttrValue(i=num_original_outputs))
    # The while op may be created inside a tf.function, in which case ops
    # needs to capture "through" it when taking gradients; outer_graph is used
    # as a sanity check that capturing only happens from parent to child.
    cond_graph.outer_graph = ops.get_default_graph()
    body_graph.outer_graph = ops.get_default_graph()
    while_op._cond_graph = cond_graph
    while_op._body_graph = body_graph
    return tensors
  return util.run_as_function_for_tape_gradients(_make_op, loop_vars)


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
  # 3. Do not accumulate loop vars that are returned as-is just like captured
  #    tensors.
  intermediates = []
  reverse_captures = dict((v.ref(), k) for k, v in func_graph.captures)

  for op in func_graph.get_operations():
    if op.type == "Identity":
      continue
    # Accumulating mutexes can cause deadlock.
    if op.type == "MutexLock":
      continue
    for o in op.outputs:
      if (o is not func_graph.inputs[0] and  # Loop counter.
          o.dtype != dtypes.resource and  # Do not accumulate resource tensors.
          _get_accumulator(o) is None and  # Has existing accumulator.
          o.ref() not in reverse_captures
         ):  # Captured value, hence loop invariant.
        intermediates.append(o)
  return intermediates


def _preprocess_grad(grad, body_graph_output, while_op_input, while_op_output):
  """Returns the initial gradient to be used for a given output tensor.

  Args:
    grad: the original gradient Tensor passed to the gradient function.
    body_graph_output: the corresponding Tensor in the body graph.
    while_op_input: the corresponding Tensor input of the While op.
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
  # TODO(b/143286622): The supports_default_grad check is needed
  # because While op emits non-differentiable resource tensors
  # as outputs. Remove this check when that is not the case.
  # Note: We use `while_op_input` instead of `while_op_output` for the call
  # to `supports_default_grad` because `while_op_output` may be missing
  # handle_data if the While is in a restored saved model.
  if (while_op_output.dtype in (dtypes.resource, dtypes.variant) and
      default_gradient.supports_default_grad(while_op_input) and grad is None):
    return _zeros_like(while_op_input, while_op_output)

  # Convert IndexedSlices to dense tensors since it is unlikely that downstream
  # gradient functions with properly handle indexed slices. This is similar to
  # what we do in tf.function gradients.
  if isinstance(grad, indexed_slices.IndexedSlices):
    return ops.convert_to_tensor(grad)

  return grad


# TODO(skyewm): make this return constants if op_output's shape is fully
# defined (this can be done by checking the "shape" attr of resource vars).
def _zeros_like(op_input, op_output):
  """Like array_ops.zeros_like() but also accepts resource var handles."""
  if op_output.dtype == dtypes.resource:
    # Note: We use `op_input` instead of `op_output` to get the zeros dtype
    # because `op_output` may be missing handle_data if the While is in a
    # restored saved model.
    return array_ops.zeros(
        gen_resource_variable_ops.variable_shape(op_output),
        dtype=default_gradient.get_zeros_dtype(op_input))
  return array_ops.zeros_like(op_output)


def _is_trainable(tensor):
  """Returns whether the given tensor is trainable."""
  if not backprop_util.IsTrainable(tensor):
    return False

  # Special case: untrainable accumulator output. The gradients algorithm
  # doesn't know about tensor lists of untrainable elements. In theory the
  # tensor list gradient functions should return None as appropriate, but
  # because we can't return None from the gradient function we filter out
  # untrainable accumulator output here to avoid computing the gradient at all.
  if tensor.op.type == "TensorListPopBack" and tensor.value_index == 0:
    assert tensor.dtype == dtypes.variant
    element_type = tensor.op.get_attr("element_dtype")
    return backprop_util.IsTrainable(element_type)

  return True


def _get_graph(while_op, func_attr_name, attr_graph_name):
  """Returns `FuncGraph` for the given function attribute.

  Args:
    while_op: The While Operation.
    func_attr_name: string
    attr_graph_name: cached forward graph name

  Returns:
    `FuncGraph`
  """
  func_graph = getattr(while_op, attr_graph_name, None)
  if func_graph is None:
    # TODO(srbs): Handle TensorShapeProto in function_def_to_graph.input_shapes.
    input_shapes = [
        tensor_shape.TensorShape(s) for s in while_op.get_attr("output_shapes")
    ]
    func_name = while_op.get_attr(func_attr_name).name
    func_graph = util.get_func_graph(while_op, input_shapes, func_name)
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

  # Build frozen sets so that we do not have linear time lookups in
  # `_is_loop_invariant`. Note: `body_graph.inputs` and `body_graph.outputs`
  # may get updated during gradient computation because we add accumulators to
  # the forward op. However, those are not loop invariants so wouldn't affect
  # the output of `_is_loop_invariant`. Also we would never attempt to capture
  # those accumulators so `_is_loop_invariant` should never receive those new
  # tensors as args.
  body_graph_inputs = object_identity.ObjectIdentitySet(body_graph.inputs)
  body_graph_outputs = object_identity.ObjectIdentitySet(body_graph.outputs)

  args = [counter, maximum_iterations, total_iters] + list(grads)
  # Note: The returned function does not have `args` in the list of
  # `external_captures`.
  grad_func_graph = func_graph_module.func_graph_from_py_func(
      name,
      lambda *args: _grad_fn(ys, xs, args, body_graph),
      args, {},
      func_graph=_WhileBodyGradFuncGraph(name, cond_graph, body_graph,
                                         maximum_iterations, while_op,
                                         body_graph_inputs, body_graph_outputs))

  # Update the list of outputs with tensors corresponding to the captured
  # tensors. We capture 3 types of tensors when building the grad fn:
  # 1. Accumulators for forward graph intermediates which are not loop
  #    invariants. The outputs corresponding to these are populated in
  #    `internal_capture_to_output` by `_WhileBodyGradFuncGraph`.
  # 2. Resources, which are output as is.
  # 3. Forward graph loop invariants, which are output as is.
  for external_capture, internal_capture in grad_func_graph.captures:
    if (ops.tensor_id(internal_capture)
        in grad_func_graph.internal_capture_to_output):
      new_output = grad_func_graph.internal_capture_to_output[ops.tensor_id(
          internal_capture)]
    else:
      raise ValueError(
          f"Tensor {str(internal_capture)} which captures "
          f"{str(external_capture)} is in list of "
          f"internal_captures but not in internal_capture_to_output.")
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
    # Resolve tensors captured from the forward graph to the outputs of the
    # forward while_op.
    if t.graph == body_graph:
      # Captured accumulator or loop invariant.
      for i, output in enumerate(t.graph.outputs):
        if output is t:
          t = while_op.outputs[i]
          break

      # Note: We rely on the capturing logic of the gradient While op graph to
      # correctly capture the tensors in `body_graph.outer_graph`. Both cond_v2
      # and while_v2 handle this while building their gradient functions.
      assert t.graph == body_graph.outer_graph

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
    if isinstance(output, indexed_slices.IndexedSlices):
      # TODO(skyewm): is there a more robust way to determine the order of
      # flattened IndexedSlices components?
      result.append(indexed_slices.IndexedSlices(
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
    for output in tensor.graph.outputs:
      if output is t:
        return t
    # tf.defun adds an Identity for each output, check whether that is the case.
    identity_op = t.consumers()[0]
    if (identity_op.type == "Identity" and
        any(identity_op.outputs[0] is t for t in tensor.graph.outputs)):
      return identity_op.outputs[0]
    return None

  for consumer in tensor.consumers():
    # Find the consumer that is a TensorListPushBack node whose TensorList input
    # is in the list of function inputs.
    if consumer.type != "TensorListPushBack":
      continue

    accum_input_idx = -1
    for accum_input_idx, inp in enumerate(tensor.graph.inputs):
      if inp is consumer.inputs[0]:
        break
    else:
      continue

    output = get_func_graph_output(consumer.outputs[0])
    if output is None:
      # The TensorList output of `consumer` is not in the list of function
      # outputs.
      continue

    for accum_output_idx, out in enumerate(tensor.graph.outputs):
      if out is output:
        if accum_input_idx == accum_output_idx:
          return output
        break

  return None


OptimizedReductionOpsCacheKey = collections.namedtuple(
    "OptimizedReductionOpsCacheKey", [
        "op_type",
        "inputs",
        "dtypes",
        "input_types",
        "name",
        "attrs",
        "op_def",
        "compute_device",
    ])


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
  To manually capture a tensor that is not in the forward graph, call `capture`
  with `allowlisted=True`.

  Note: The `captures` dict does not contain the forward tensor since it is not
  directly captured. It contains the accumulator corresponding to this forward
  tensor.

  Attributes:
    while_op_needs_rewrite: True if any non-resource intermediates were
      captured, meaning the forward While op needs to be rewritten to output the
      corresponding accumulators.
    extra_inputs: list of EmptyTensorList tensors to be used as initial input to
    the new accumulators in the forward graph. It may also contain external
    captures of the custom gradient function.
    internal_capture_to_output: dict from a tensor_id(captured placeholder) to
      the corresponding tensor that needs to be added to the list of outputs.
      For instance, when capturing an accumulator TensorList this contains the
      TensorList obtained after popping a tensor from the list. Other entries
      in this dict are expected, though not enforced, to be identities.
      This dict is needed because these output tensors need to be added to
      FuncGraph.outputs "after" the tensors returned from the gradient function.
  """

  def __init__(self, name, forward_cond_graph, forward_body_graph,
               maximum_iterations, forward_while_op, body_graph_inputs,
               body_graph_outputs):
    super(_WhileBodyGradFuncGraph, self).__init__(name)
    self.extra_inputs = []
    self.internal_capture_to_output = {}
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
    return self.extra_inputs

  def _create_op_internal(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    # For a reduction op, if op is in the gradient body graph and its input is
    # from the forward graph, moving op to the forward graph means we would
    # store the tensor after the reduction as opposed to the tensor before
    # reduction, and therefore could significantly reduce memory consumption.
    # For now, we do this only for a few ops.
    #
    # We don't do this if any input tensor has already been accumulated. This
    # can happen if we output all intermediates in the forward pass.
    #
    # If in XLA context, do not move constant ops to forward pass as pushing to
    # and popping from a TensorList removes the constant property of an op and
    # breaks XLA compilation, which requires certain inputs to be compile-time
    # constant for certain ops.
    #
    # This optimization is currently also disabled when under a persistent tape,
    # since it leads to an unbounded number of side outputs. With caching it may
    # be possible to re-enable it.
    optimized_reduction_ops = {
        "Shape", "Size", "Rank", "TensorListElementShape", "TensorListLength"
    }
    if (op_type in optimized_reduction_ops and
        not util.output_all_intermediates() and
        all(input.graph is self._forward_graph for input in inputs) and
        all(_get_accumulator(input) is None for input in inputs) and
        not util_v1.GraphOrParentsInXlaContext(self._forward_graph) and
        not util.graph_wrapped_for_higher_order_tape_gradients(
            self._forward_graph)):
      return self._move_op_to_forward_graph(
          op_type,
          inputs,
          dtypes=dtypes,
          input_types=input_types,
          name=name,
          attrs=attrs,
          op_def=op_def,
          compute_device=compute_device)

    return super(_WhileBodyGradFuncGraph, self)._create_op_internal(
        op_type,
        inputs,
        dtypes=dtypes,
        input_types=input_types,
        name=name,
        attrs=attrs,
        op_def=op_def,
        compute_device=compute_device)

  def _move_op_to_forward_graph(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    # We have a cache of reduction ops that have already been moved to the
    # forward graph, and we will check it first to avoid moving an op twice.
    if not hasattr(self._forward_graph, "_optimized_reduction_ops_cache"):
      self._forward_graph._optimized_reduction_ops_cache = {}
    cache_key = self._get_optimized_reduction_ops_cache_key(
        op_type, inputs, dtypes, input_types, name, attrs, op_def,
        compute_device)
    cached_op = self._forward_graph._optimized_reduction_ops_cache.get(
        cache_key)
    if cached_op is not None:
      # This op has already been moved to the forward graph and we have it in
      # the cache.
      return cached_op

    with self._forward_graph.as_default():
      # `name` was built using name_scope stack of gradient graph and may not
      # be unique in the forward graph. `Graph.create_op` does not uniquify
      # names which are name scopes i.e. end in `/`. To ensure that the op
      # created gets a unique name in the forward graph we get rid of the
      # trailing slash.
      name = ops.name_from_scope_name(name)
      result = self._forward_graph._create_op_internal(
          op_type,
          inputs,
          dtypes=dtypes,
          input_types=input_types,
          name=name,
          attrs=attrs,
          op_def=op_def,
          compute_device=compute_device)

      # Store the op we just moved to the forward graph so that it does
      # not need to be added there again.
      self._forward_graph._optimized_reduction_ops_cache[cache_key] = result
      return result

  def _get_optimized_reduction_ops_cache_key(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    # We need all elements of CacheKey to be hashable.
    inputs = tuple(map(lambda t: t.ref(), inputs))

    if dtypes is not None:
      dtypes = tuple(dtypes)

    if input_types is not None:
      input_types = tuple(input_types)

    if attrs is not None:
      hashable_attrs = []
      for attr_name, attr_value in sorted(attrs.items()):
        hashable_attrs.append((attr_name, attr_value.SerializeToString()))
      attrs = tuple(hashable_attrs)

    if op_def is not None:
      op_def = op_def.SerializeToString()

    return OptimizedReductionOpsCacheKey(op_type, inputs, dtypes, input_types,
                                         name, attrs, op_def, compute_device)

  def _capture_helper(self, tensor, name):
    """Implements the capturing described in the class docstring."""
    captured_tensor = self._indirect_captures.get(ops.tensor_id(tensor))
    if captured_tensor is not None:
      return captured_tensor

    if tensor.graph is not self._forward_graph:
      already_captured = id(tensor) in self._function_captures.by_val_captures  # pylint: disable=protected-access
      captured_tensor = super(_WhileBodyGradFuncGraph, self)._capture_helper(
          tensor, name)
      if not already_captured:
        # Adds the captured tensor to the list of outputs so that the input
        # and output signatures match.
        self.internal_capture_to_output[ops.tensor_id(
            captured_tensor)] = captured_tensor
        self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
      return captured_tensor

    while tensor.op.type == "Identity":
      # We do not accumulate the output of identity nodes so we try to capture
      # the input of the Identity node instead.
      tensor = tensor.op.inputs[0]

    captured_tensor = self._indirect_captures.get(ops.tensor_id(tensor))
    if captured_tensor is not None:
      return captured_tensor

    # No need to accumulate loop invariants. Capture them directly.
    # The captured tensor gets resolved to the corresponding while output in
    # `_resolve_grad_captures`.
    if _is_loop_invariant(tensor, self._forward_graph.inputs,
                          self._forward_graph.outputs):
      captured_tensor = super(_WhileBodyGradFuncGraph,
                              self)._capture_helper(tensor, name)
      # Add to `internal_capture_to_output` so that this gets added to the list
      # of outputs.
      self.internal_capture_to_output[ops.tensor_id(
          captured_tensor)] = captured_tensor
      self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
      return captured_tensor

    # Do not accumulate Const nodes. Instead copy them directly in the backward
    # graph.
    # TODO(srbs): This just checks for `Const` nodes. Consider checking for
    # graph compile time consts in general.
    # TODO(srbs): Consider making this a loop input.
    if constant_op.is_constant(tensor):
      real_value = constant_op.constant(
          tensor_util.constant_value(tensor), dtype=tensor.dtype)
      self._indirect_captures[ops.tensor_id(tensor)] = real_value
      return real_value

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
      self.extra_inputs.append(tensor_list)

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

    self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
    self.internal_capture_to_output[ops.tensor_id(
        captured_accumulator)] = new_tensor_list
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

    forward_graph_input_names = [t.name for t in self._forward_graph.inputs]
    forward_graph_name_to_opdef = {
        op.name: op.node_def for op in self._forward_graph.get_operations()}
    index = util.resource_input_index(
        tensor.name, forward_graph_input_names,
        forward_graph_name_to_opdef,
        self._forward_graph._functions)

    input_placeholder = self._forward_graph.inputs[index]
    tensor_in_outer_graph = self._forward_graph._while.inputs[index]

    assert input_placeholder.dtype == dtypes.resource
    assert tensor_in_outer_graph.dtype == dtypes.resource
    # This must be a loop invariant. However, infrastructure
    # (e.g. tf.vectorized_map) may insert identity nodes, function calls, conds,
    # etc. which take and return the resource tensor unmodified; this means that
    # the Python objects may differ.
    if index != util.resource_input_index(
        self._forward_graph.outputs[index].name, forward_graph_input_names,
        forward_graph_name_to_opdef,
        self._forward_graph._functions):
      raise AssertionError(
          f"Resource tensors must be loop invariants {tensor_in_outer_graph}")

    self._indirect_captures[ops.tensor_id(tensor)] = self.capture(
        tensor_in_outer_graph)
    return self._indirect_captures[ops.tensor_id(tensor)]


def _check_shapes_compat(flat_output_tensors, flat_shape_invariants,
                         flat_input_tensors):
  for (t, shape, input_t) in zip(flat_output_tensors, flat_shape_invariants,
                                 flat_input_tensors):
    if not control_flow_ops._ShapeLessThanOrEqual(t.shape, shape):
      raise ValueError(
          f"Input tensor `{input_t.name}` enters the loop with shape {shape}, "
          f"but has shape {t.shape} after one iteration. To allow the shape to "
          "vary across iterations, use the `shape_invariants` argument of "
          "tf.while_loop to specify a less-specific shape.")


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


def _check_inputs_outputs_types_match(body_graph, flattened_loop_vars):
  for inp, out, loop_var in zip(body_graph.inputs, body_graph.outputs,
                                flattened_loop_vars):
    if inp.dtype != out.dtype:
      raise TypeError(
          f"Loop var {loop_var.name} enters the loop with type {inp.dtype} "
          f"but has type {out.dtype} after 1 iteration. {loop_var.name} type "
          "should remain constant.")


def _build_cond_placeholders_name_prefix(cond_graph):
  return cond_graph.unique_name(cond_graph.name + "___redundant_placeholder")


def _duplicate_body_captures_in_cond(cond_graph, body_graph_captures):
  """Creates placeholders for body captures in cond_graph.

  This is needed to match signatures of cond and body graphs.

  Args:
    cond_graph: cond branch graph
    body_graph_captures: Tensors which were captured when building the
      `body_graph`.
  """
  types = [t.dtype.as_datatype_enum for t in body_graph_captures]
  # TODO(srbs): Providing a unique prefix does not ensure that there is no
  # conflict between the placeholder names and existing nodes in the graph.
  # However passing a list of strings may not be performant.
  # Ideally we should move `Graph.unique_name` to C++ or make
  # `Graph._names_in_use` a trie so that we can find a unique prefix.
  # TODO(b/143286622): This should not be required once captures are separated
  # from regular loop vars.
  with cond_graph._c_graph.get() as c_graph:
    placeholders = c_api.TF_CreatePlaceholders(
        c_graph, types,
        compat.as_str(_build_cond_placeholders_name_prefix(cond_graph)))
  placeholder_ops = [
      ops.Operation._from_c_op(ph.oper, cond_graph) for ph in placeholders
  ]

  tensors = []
  for op, ph, dtype in zip(placeholder_ops, placeholders, types):
    tensor = ops.Tensor._create_with_tf_output(op, 0, dtype, ph)
    op._outputs = [tensor]
    tensors.append(tensor)

  # Update `cond_graph._captures` and `cond_graph.inputs` to contain the
  # newly created placeholders.
  tuples = zip(body_graph_captures, tensors)
  keys = [id(t) for t in body_graph_captures]
  for k, v in zip(keys, tuples):
    capture = capture_container.CaptureContainer(v[0], v[1], k, False)
    cond_graph._function_captures._by_val[k] = capture  # pylint: disable=protected-access
  cond_graph.inputs.extend(tensors)


def _copy_handle_data(src_tensors, tgt_tensors):
  for src_t, tgt_t in zip(src_tensors, tgt_tensors):
    handle_data_util.copy_handle_data(src_t, tgt_t)


def _pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, loop_vars):
  """Like `nest.pack_sequence_as` but also replaces flows with TensorArrays."""

  def flow_to_tensor_array(flow, ta):  # pylint: disable=missing-docstring
    return (tensor_array_ops.build_ta_with_new_flow(ta, flow) if isinstance(  # pylint: disable=g-long-ternary
        ta, tensor_array_ops.TensorArray) else flow)

  flattened_loop_vars = [
      flow_to_tensor_array(*z)
      for z in zip(nest.flatten(loop_vars, expand_composites=True),
                   flat_orig_loop_vars)
  ]
  return nest.pack_sequence_as(loop_vars_signature, flattened_loop_vars,
                               expand_composites=True)


def _tensor_array_to_flow(loop_vars):

  def f(maybe_ta):
    if isinstance(maybe_ta, tensor_array_ops.TensorArray):
      return maybe_ta.flow
    return maybe_ta

  return nest.map_structure(f, loop_vars, expand_composites=True)


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


def _is_loop_invariant(tensor, inputs, outputs):
  return (any(tensor is t for t in inputs) and
          any(tensor is t for t in outputs))


def _set_read_only_resource_inputs_attr(op, branch_graphs):
  """Sets the list of resource inputs which are read-only.

  This is used by AutomaticControlDependencies.

  Args:
    op: While Operation.
    branch_graphs: List of branch FuncGraphs.
  """
  read_only_indices = set(range(len(op.inputs)))
  for branch_graph in branch_graphs:
    if not read_only_indices:
      break
    branch_read_only_indices = acd.get_read_only_resource_input_indices_graph(
        branch_graph)
    read_only_indices = read_only_indices.intersection(branch_read_only_indices)

  ops.set_int_list_attr(op, acd.READ_ONLY_RESOURCE_INPUTS_ATTR,
                        sorted(read_only_indices))

# pylint: enable=protected-access
