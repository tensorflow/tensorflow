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
"""cond_v2 and gradient.

This is a version of cond that emits a single If op, as well as the gradient
function for If ops produced by cond_v2. This will eventually replace the
current tf.cond implementation once it reaches feature and performance parity.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


# NOTE(skyewm): TensorFlow uses protected class methods and fields to signify
# that they aren't part of the official public API. These protected members
# often need to be used by implementation code however. Rather than litter the
# code with pylint comments, we ignore protected access violations for
# readability.
# pylint: disable=protected-access


def cond_v2(pred, true_fn, false_fn, name="cond"):
  """Like tf.cond, except emits a single If op."""
  if isinstance(pred, bool):
    raise TypeError("pred must not be a Python bool", pred)

  if not name:
    name = "cond"

  with ops.name_scope(name) as scope:
    true_name = util.unique_fn_name(scope, "true")
    false_name = util.unique_fn_name(scope, "false")

    # Automatic control dependencies are added in defuns, but not in v1
    # graphs. Propagate that behavior here.
    add_control_dependencies = ops.get_default_graph()._add_control_dependencies
    pred = ops.convert_to_tensor(pred)

    true_graph = func_graph_module.func_graph_from_py_func(
        true_name,
        true_fn, [], {},
        func_graph=util.CondBranchFuncGraph(
            true_name, collections=ops.get_default_graph()._collections),  # pylint: disable=protected-access
        add_control_dependencies=add_control_dependencies,
        op_return_value=pred)
    false_graph = func_graph_module.func_graph_from_py_func(
        false_name,
        false_fn, [], {},
        func_graph=util.CondBranchFuncGraph(
            false_name, collections=ops.get_default_graph()._collections),  # pylint: disable=protected-access
        add_control_dependencies=add_control_dependencies,
        op_return_value=pred)

    return _build_cond(pred, true_graph, false_graph,
                       true_graph.external_captures,
                       false_graph.external_captures,
                       name=scope)


@ops.RegisterGradient("If")
def _IfGrad(op, *grads):  # pylint: disable=invalid-name
  """The gradient of an If op produced by cond_v2."""
  # Get the if operator (this logic handles the case where op is a MockOp)
  if_op = op.outputs[0].op
  true_graph, false_graph = _get_func_graphs(if_op)
  # Note: op.graph != ops.get_default_graph() when we are computing the gradient
  # of a nested cond.
  assert true_graph.outer_graph == if_op.graph
  assert false_graph.outer_graph == if_op.graph

  # Create grad functions that compute the gradient of the true/false forward
  # graphs. These functions will capture tensors from the forward pass
  # functions.
  true_grad_graph = _create_grad_func(
      true_graph, grads, util.unique_grad_fn_name(true_graph.name))
  false_grad_graph = _create_grad_func(
      false_graph, grads, util.unique_grad_fn_name(false_graph.name))

  if (true_grad_graph.if_op_needs_rewrite or
      false_grad_graph.if_op_needs_rewrite):
    # Modify 'op' to output the intermediates needed by the grad functions. Note
    # that all needed intermediates are wrapped in optionals. Each optional
    # intermediate output will have a value iff its corresponding branch is
    # taken.
    # NOTE(skyewm): if there are any active sessions, this modification to `op`
    # may make them unrunnable!

    if control_flow_util.InXlaContext(ops.get_default_graph()):
      # XLA does not yet support optionals, so output intermediates directly and
      # make them match via FakeParams, which can be converted to zeros in XLA.
      # TODO(skyewm,jpienaar): can XLA support optionals?
      true_intermediates = true_grad_graph.xla_intermediates
      false_intermediates = false_grad_graph.xla_intermediates
      extra_true_outputs, extra_false_outputs = _make_intermediates_match_xla(
          true_graph, false_graph, true_intermediates, false_intermediates)
    else:
      true_intermediates = true_grad_graph.wrapped_intermediates
      false_intermediates = false_grad_graph.wrapped_intermediates
      # Make outputs match by adding none optionals.
      extra_true_outputs, extra_false_outputs = _make_intermediates_match(
          true_graph, false_graph, true_intermediates, false_intermediates)

    true_graph.outputs.extend(extra_true_outputs)
    false_graph.outputs.extend(extra_false_outputs)
    # TODO(skyewm): indicate it's an internal bug if this fails.
    _check_same_outputs(true_graph, false_graph)

    true_graph.name += "_rewritten"
    false_graph.name += "_rewritten"

    if_op._set_func_attr("then_branch", util.create_new_tf_function(true_graph))
    if_op._set_func_attr("else_branch",
                         util.create_new_tf_function(false_graph))
    if_op._set_type_list_attr("Tout", true_graph.output_types)
    if_op._set_shape_list_attr("output_shapes", true_graph.output_shapes)
    if_op._add_outputs(
        [t.dtype for t in extra_true_outputs],
        [t.shape for t in extra_true_outputs])

  # Resolve references to forward graph tensors in grad graphs and ensure
  # they are in-scope, i.e., belong to one of outer graphs of the grad graph.
  true_grad_inputs = _resolve_grad_inputs(true_graph, true_grad_graph)
  false_grad_inputs = _resolve_grad_inputs(false_graph, false_grad_graph)

  # This modifies true_grad_graph and false_grad_graph.
  _make_output_composite_tensors_match(true_grad_graph, false_grad_graph)

  outputs = _build_cond(if_op.inputs[0], true_grad_graph, false_grad_graph,
                        true_grad_inputs, false_grad_inputs)

  # The predicate has no gradient.
  return [None] + outputs


def _build_cond(pred, true_graph, false_graph, true_inputs, false_inputs,
                name=None):
  """Creates an If op from the specified predicate, branch functions and inputs.

  Note that this modifies true_graph and false_graph to make the inputs match,
  and to output all intermediates values so they're available for the gradient
  computation.

  true_graph and false_graph need not have the same input types, but they must
  have the same outpute types.

  Args:
    pred: boolean Tensor
    true_graph: FuncGraph
    false_graph: FuncGraph
    true_inputs: a list of Tensors to be passed to true_graph as input.
    false_inputs: a list of Tensors to be passed to false_graph as input.
    name: the name for the If op.

  Returns:
    A list of Tensors which are the outputs of the If op. Does not include added
    intermediate outputs.
  """
  _check_same_outputs(true_graph, false_graph)

  # Add inputs to true_graph and false_graph to make them match. Note that
  # this modifies true_graph and false_graph.
  cond_inputs = _make_inputs_match(true_graph, false_graph,
                                   true_inputs, false_inputs)

  # Create the If op.
  tensors = gen_functional_ops._if(  # pylint: disable=protected-access
      pred,
      cond_inputs, [t.dtype for t in true_graph.outputs],
      util.create_new_tf_function(true_graph),
      util.create_new_tf_function(false_graph),
      output_shapes=_get_output_shapes(true_graph.outputs,
                                       false_graph.outputs),
      name=name)

  # TODO(b/110167197) this approach requires cond_v2 to have at least 1 output
  if_op = tensors[0].op
  util.maybe_set_lowering_attr(if_op)

  # Return identities for each output of the If op, rather than the output of
  # the If op directly. This makes pruning work if the output of cond() is
  # fetched: the lowering pass converts the If outputs into IdentityN outputs,
  # which if fetched will cause all ops in the taken branch to be run (since
  # it takes all merge ops as input). After lowering, each output identity op
  # will end up with only the appropriate merge op as input.
  # TODO(b/79984175): this doesn't have to be a tuple once we covert to the
  # correct output structure
  tensors = [array_ops.identity(t) for t in tensors]

  # Prevent fetching since the variant outputs can't be fetched directly.
  if_op.graph.prevent_fetching(if_op)
  return func_graph_module.pack_sequence_as(true_graph.structured_outputs,
                                            tensors)


def _get_func_graphs(if_op):
  """Returns `FuncGraph`s for the input op branches.

  Args:
    if_op: The _If Operation.

  Returns:
    A 2-tuple of the `FuncGraph`s of the then_branch and else_branch.
  """
  def _get_func_graph_for_branch(branch_name):
    """Generates and returns a FuncGraph for the given branch."""
    inputs = if_op.inputs[1:]  # First input is pred.
    input_shapes = [t.shape for t in inputs]
    func_name = if_op.get_attr(branch_name).name
    fdef = if_op.graph._get_function(func_name).definition
    # `if_op.graph` may not be the same as `ops.get_default_graph()` e.g.
    # in the case of nested if ops or when the gradient is being computed
    # from inside a Defun. We build the `func_graph` with `if_op.graph` as its
    # `outer_graph`. This resembles how the `FuncGraph` was built in the
    # forward pass. We need this so that we can resolve references to tensors
    # in `func_graph` from its gradient graph in `_resolve_grad_inputs`.
    with if_op.graph.as_default():
      func_graph = function_def_to_graph.function_def_to_graph(
          fdef, input_shapes)
    func_graph.captures = collections.OrderedDict(zip(inputs,
                                                      func_graph.inputs))
    # Set the if op so that the gradient code can use it.
    func_graph._if = if_op
    return func_graph

  return (_get_func_graph_for_branch("then_branch"),
          _get_func_graph_for_branch("else_branch"))


def _grad_fn(func_graph, grads):
  """The gradient function for each conditional branch.

  This function builds the gradient graph of the corresponding forward-pass
  conditional branch in `func_graph`. This is done by differentiating
  func_graph's outputs w.r.t. its inputs.

  Args:
    func_graph: FuncGraph. The corresponding forward-pass function.
    grads: The list of input gradient Tensors.

  Returns:
    The output gradient Tensors.
  """
  # Filter out untrainable function outputs.
  # NOTE(skyewm): If we don't do this, the untrainable tensors can sometimes
  # cause _GradientsHelper to raise an exception (e.g. the implementation
  # doesn't expect 'ys' to contain boolean tensors).
  assert len(func_graph.outputs) == len(grads)
  ys = []
  grad_ys = []
  for y, grad_y in zip(func_graph.outputs, grads):
    if not gradients_util.IsTrainable(y):
      continue
    ys.append(y)
    grad_ys.append(grad_y)

  # Build the gradient graph. Note that this builds the gradient computation of
  # func_graph in the current graph, which requires capturing tensors from
  # func_graph. The captured func_graph tensors are resolved to external tensors
  # in _resolve_grad_inputs.
  result = gradients_util._GradientsHelper(
      ys, func_graph.inputs, grad_ys=grad_ys,
      src_graph=func_graph)

  # Functions can't return None; replace Nones with zero tensors.
  # TODO(b/80444525): don't return anything here and make _IfGrad return None if
  # both branches have zero gradient.
  for i in range(len(result)):
    if result[i] is None:
      if func_graph.inputs[i].dtype == dtypes.resource:
        result[i] = array_ops.zeros(
            gen_resource_variable_ops.variable_shape(func_graph.inputs[i]))
      else:
        result[i] = array_ops.zeros_like(func_graph.inputs[i])

  return result


def _create_grad_func(func_graph, grads, name):
  """Returns the FuncGraph representation of _grad_fn."""
  return func_graph_module.func_graph_from_py_func(
      name,
      lambda: _grad_fn(func_graph, grads), [], {},
      func_graph=_CondGradFuncGraph(name, func_graph))


def _resolve_grad_inputs(cond_graph, grad_graph):
  """Returns the tensors to pass as inputs to `grad_graph`.

  The `grad_graph` may have external references to
  1. Its outer graph containing the input gradients. These references are kept
     as is.
  2. Tensors in the forward pass graph. These tensors may not be "live"
     when the gradient is being computed. We replace such references by their
     corresponding tensor in `cond_graph.outer_graph`. In the case of nested
     control flow or functions, the gradient logic handling
     `grad_graph.outer_graph` will make sure the tensor from
     `cond_graph.outer_graph` is also correctly captured.

  Args:
    cond_graph: FuncGraph. The forward-pass function.
    grad_graph: FuncGraph. The gradients function.

  Returns:
    A list of inputs tensors to be passed to grad_graph.
  """
  new_inputs = []

  for t in grad_graph.external_captures:
    # `t` must either be in `grad_graph.outer_graph` or in the forward
    # `cond_graph`.
    if t.graph != grad_graph.outer_graph:
      assert t.graph == cond_graph
      # `internal_captures` are not treated as intermediates and hence not added
      # to If op outputs. So we get the outer tensor corresponding to those
      # from the list of `external_captures`.
      try:
        t = t.graph._if.outputs[t.graph.outputs.index(t)]
      except ValueError:
        index = t.graph.internal_captures.index(t)
        t = t.graph.external_captures[index]

      # Note: We rely on the capturing logic of the gradient If op graph to
      # correctly capture the tensors in `cond_graph.outer_graph`. Both cond_v2
      # and while_v2 handle this while building their gradient functions.
      assert t.graph == cond_graph.outer_graph
    new_inputs.append(t)

  return new_inputs


def _get_intermediates(func_graph):
  """Returns all tensors in `func_graph` that aren't inputs or outputs."""
  intermediates = []
  for op in func_graph.get_operations():
    for t in op.outputs:
      if t in func_graph.inputs: continue
      if t in func_graph.outputs: continue
      intermediates.append(t)
  return intermediates


def _separate_unique_inputs(true_inputs, false_inputs):
  """Separates tensors appearing only in true_inputs or false_inputs, or both.

  Args:
    true_inputs: list of Tensors
    false_inputs: list of Tensors

  Returns:
    Three lists of Tensors:
      1. The tensors that appear in both true_inputs and false_inputs
      2. The tensors that only appear in true_inputs
      3. The tensors that only appear in false_inputs
  """
  true_inputs = set(true_inputs)
  false_inputs = set(false_inputs)

  shared_inputs = true_inputs.intersection(false_inputs)
  true_only_inputs = true_inputs - false_inputs
  false_only_inputs = false_inputs - true_inputs

  return list(shared_inputs), list(true_only_inputs), list(false_only_inputs)


def _make_intermediates_match(true_graph, false_graph,
                              true_optionals, false_optionals):
  """Returns new optionals lists that have matching signatures.

  This is done by mirroring each list in the other using none optionals.
  There is no merging of like optionals.

  Args:
    true_graph: FuncGraph
    false_graph: FuncGraph
    true_optionals: a list of optional Tensors from true_graph
    false_optionals: a list of optional Tensors from false_graph

  Returns:
    A new list of Tensors in true_graph and a new list of Tensors in
    false_graph. The two lists have the same number of Tensors, all of which
    will be optionals of the same shape/type.
  """
  new_true_optionals = (true_optionals +
                        _create_none_optionals(true_graph, false_optionals))
  new_false_optionals = (_create_none_optionals(false_graph, true_optionals)
                         + false_optionals)
  return new_true_optionals, new_false_optionals


def _make_intermediates_match_xla(true_graph, false_graph, true_intermediates,
                                  false_intermediates):
  """Like _make_intermediates_match but for the XLA case."""
  new_true_intermediates = (true_intermediates +
                            _create_fakeparams(true_graph, false_intermediates))
  new_false_intermediates = (_create_fakeparams(false_graph, true_intermediates)
                             + false_intermediates)
  return new_true_intermediates, new_false_intermediates


def _make_inputs_match(true_graph, false_graph, true_inputs, false_inputs):
  """Modifies true_graph and false_graph so they have the same input signature.

  This method reorders and/or adds parameters to true_graph and false_graph so
  they have the same input signature, and updates the 'inputs' and 'captured'
  fields of both graphs accordingly. It uses the input tensors from the outer
  graph to avoid duplicating shared arguments.

  Args:
    true_graph: FuncGraph
    false_graph: FuncGraph
    true_inputs: a list of Tensors in the outer graph. The inputs for
      true_graph.
    false_inputs: a list of Tensors in the outer graph. The inputs for
      false_graph.

  Returns:
    A new list of Tensors from the outer graph that are the new inputs for both
    true_graph and false_graph. This is a deduped version of true_inputs +
    false_inputs.
  """
  shared_inputs, true_only_inputs, false_only_inputs = _separate_unique_inputs(
      true_inputs, false_inputs)

  new_inputs = shared_inputs + true_only_inputs + false_only_inputs

  true_input_to_param = dict(zip(true_inputs, true_graph.inputs))
  false_input_to_param = dict(zip(false_inputs, false_graph.inputs))

  true_graph.inputs = (
      [true_input_to_param[t] for t in shared_inputs] +
      [true_input_to_param[t] for t in true_only_inputs] +
      _create_dummy_inputs(true_graph, false_only_inputs))

  false_graph.inputs = (
      [false_input_to_param[t] for t in shared_inputs] +
      _create_dummy_inputs(false_graph, true_only_inputs) +
      [false_input_to_param[t] for t in false_only_inputs])

  # Rewrite the FuncGraphs' state to reflect the new inputs.
  true_graph.captures = collections.OrderedDict(zip(new_inputs,
                                                    true_graph.inputs))
  false_graph.captures = collections.OrderedDict(zip(new_inputs,
                                                     false_graph.inputs))

  return new_inputs


def _make_output_composite_tensors_match(true_graph, false_graph):
  """Rewrites {true,false}_graph's outputs to use the same _TensorLike classes.

  Currently the only transformation implemented is turning a Tensor into an
  equivalent IndexedSlices if the other branch returns an IndexedSlices.
  Updates {true,false}_graph.{outputs,structured_outputs}.

  Args:
    true_graph: FuncGraph
    false_graph: FuncGraph

  Raises:
    TypeError: if a pair of outputs cannot be rewritten.
  """
  # Note: since this is only used for gradient graphs, we do not expect the
  # outputs to be structured (e.g. nested lists), and thus do not need to use
  # nest.flatten, etc.
  true_outputs = list(true_graph.structured_outputs)
  false_outputs = list(false_graph.structured_outputs)
  assert len(true_outputs) == len(false_outputs)

  for idx, (true_out, false_out) in enumerate(zip(true_outputs, false_outputs)):
    if type(true_out) == type(false_out):  # pylint: disable=unidiomatic-typecheck
      continue
    if (isinstance(true_out, ops.IndexedSlices) and
        isinstance(false_out, ops.Tensor)):
      with false_graph.as_default():
        false_outputs[idx] = math_ops._as_indexed_slices(false_out)
    elif (isinstance(true_out, ops.Tensor) and
          isinstance(false_out, ops.IndexedSlices)):
      with true_graph.as_default():
        true_outputs[idx] = math_ops._as_indexed_slices(true_out)
    else:
      raise TypeError(
          "Cannot reconcile tf.cond %i-th outputs:\n"
          "  true_fn returned:  %s\n"
          "  false_fn returned: %s" % (idx, true_out, false_out))

  true_graph.structured_outputs = true_outputs
  true_graph.outputs = func_graph_module.flatten(true_outputs)
  false_graph.structured_outputs = false_outputs
  false_graph.outputs = func_graph_module.flatten(false_outputs)


def _wrap_intermediates(func_graph, intermediates):
  with func_graph.as_default():
    return [gen_dataset_ops.optional_from_value([t]) for t in intermediates]


def _create_dummy_inputs(func_graph, template_tensors):
  """Creates tensors in func_graph to represent template_tensors.

  Args:
    func_graph: FuncGraph.
    template_tensors: a list of tensors in the outer graph.

  Returns:
    A list of tensors in func_graph.
  """
  with func_graph.as_default():
    return [array_ops.placeholder(t.dtype, shape=t.shape)
            for t in template_tensors]


def _create_none_optionals(func_graph, template_tensors):
  """Creates none optionals in func_graph to represent template_tensors.

  Args:
    func_graph: FuncGraph.
    template_tensors: a list of tensors in func_graph.

  Returns:
    A list of tensors in func_graph.
  """
  with func_graph.as_default():
    return [gen_dataset_ops.optional_none() for _ in template_tensors]


def _create_fakeparams(func_graph, template_tensors):
  """Create FakeParams for the XLA case."""
  with func_graph.as_default():
    return [gen_functional_ops.fake_param(dtype=t.dtype, shape=t.shape)
            for t in template_tensors]


def _check_same_outputs(true_graph, false_graph):
  """Raises an error if true_graph and false_graph have different outputs."""

  def error(error_detail):
    raise TypeError(
        "true_fn and false_fn arguments to tf.cond must have the same number, "
        "type, and overall structure of return values.\n"
        "\n"
        "true_fn output:  %s\n"
        "false_fn output: %s\n"
        "\n"
        "Error details:\n"
        "%s" % (true_graph.structured_outputs, false_graph.structured_outputs,
                error_detail))

  try:
    nest.assert_same_structure(true_graph.structured_outputs,
                               false_graph.structured_outputs,
                               expand_composites=True)
  except (ValueError, TypeError) as e:
    error(str(e))

  assert len(true_graph.outputs) == len(false_graph.outputs)
  for true_out, false_out in zip(true_graph.outputs, false_graph.outputs):
    if true_out.dtype != false_out.dtype:
      error("%s and %s have different types" % (true_out, false_out))


def _get_output_shapes(true_graph_outputs, false_graph_outputs):
  output_shapes = [
      t_out.shape.most_specific_compatible_shape(f_out.shape)
      for t_out, f_out in zip(true_graph_outputs, false_graph_outputs)
  ]
  return output_shapes


class _CondGradFuncGraph(util.CondBranchFuncGraph):
  """FuncGraph for the gradient function of the branch of an If op.

  Handles wrapping and unwrapping intermediate values that are captured by the
  gradient computation in optionals.

  Attributes:
    if_op_needs_rewrite: True if any intermediates were captured, meaning the
      forward If op needs to be written to output the wrapped intermediates.
  """

  def __init__(self, name, forward_graph):
    super(_CondGradFuncGraph, self).__init__(
        name, collections=ops.get_default_graph()._collections)  # pylint: disable=protected-access
    self.if_op_needs_rewrite = False
    self._forward_graph = forward_graph
    # Maps from forward intermediate tensor -> the unwrapped captured
    # intermediate.
    self._indirect_captures = {}
    # Maps unwrapped intermediate -> optional-wrapped intermediate in the
    # forward graph.
    self._wrapped_intermediates = collections.OrderedDict()
    # Raw intermediates captured from the forward graph. Populated iff we're in
    # an XLA context.
    self._xla_intermediates = []

  @property
  def wrapped_intermediates(self):
    """The optional-wrapped intermediates captured from the forward graph."""
    return list(self._wrapped_intermediates.values())

  @property
  def xla_intermediates(self):
    """Raw intermediates captured from the forward graph if XLA is enabled."""
    return self._xla_intermediates

  def _capture_helper(self, tensor, name):
    if (tensor.graph is not self._forward_graph or
        tensor in self._forward_graph.inputs or
        tensor in self._forward_graph.outputs):
      return super(_CondGradFuncGraph, self)._capture_helper(tensor, name)

    if control_flow_util.InXlaContext(ops.get_default_graph()):
      # XLA does not yet support optionals, so capture intermediates directly.
      # TODO(skyewm,jpienaar): can XLA support optionals?
      if tensor not in self.captures:
        self.xla_intermediates.append(tensor)
        self.if_op_needs_rewrite = True
      return super(_CondGradFuncGraph, self)._capture_helper(tensor, name)

    captured_tensor = self._indirect_captures.get(tensor)
    if captured_tensor is not None:
      return captured_tensor

    # 'tensor' is an uncaptured intermediate in the forward graph. We wrap it in
    # an optional in the forward graph and capture the optional normally. We
    # then unwrap the captured optional value in the gradient graph to get the
    # raw intermediate value.

    if tensor not in self._wrapped_intermediates:
      # If the gradient has already been computed for this If op, 'tensor' may
      # already be wrapped.
      for consumer in tensor.consumers():
        if (consumer.type == "OptionalFromValue"
            and consumer.outputs[0] in self._forward_graph.outputs):
          optional = consumer.outputs[0]
          break
      else:
        # 'tensor' hasn't been wrapped, do it now.
        with self._forward_graph.as_default():
          optional = gen_dataset_ops.optional_from_value([tensor])
        self.if_op_needs_rewrite = True

      self._wrapped_intermediates[tensor] = optional

    optional = self._wrapped_intermediates[tensor]
    captured_optional = super(_CondGradFuncGraph, self)._capture_helper(
        optional, name)
    captured_tensor = gen_dataset_ops.optional_get_value(
        captured_optional, [tensor.dtype], [tensor.shape])[0]
    self._indirect_captures[tensor] = captured_tensor
    return captured_tensor
