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
# ==============================================================================

"""Utilities for V2 control flow."""

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import gradients_util
from tensorflow.python.util import keras_deps
from tensorflow.python.util import tf_contextlib


_EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE = None
_DISABLE_LOWER_USING_SWITCH_MERGE = False


CondBranchFuncGraph = control_flow_v2_func_graphs.CondBranchFuncGraph
WhileCondFuncGraph = control_flow_v2_func_graphs.WhileCondFuncGraph
WhileBodyFuncGraph = control_flow_v2_func_graphs.WhileBodyFuncGraph


def in_defun():
  """Returns if the current graph is, or is nested in, a defun."""
  if context.executing_eagerly(): return False

  graph = ops.get_default_graph()
  while (isinstance(graph, CondBranchFuncGraph) or
         isinstance(graph, WhileBodyFuncGraph) or
         isinstance(graph, WhileCondFuncGraph)):
    graph = graph.outer_graph
  return isinstance(graph, FuncGraph)


def in_while_loop_defun(graph):
  """Returns if the graph is a while loop FuncGraph."""
  if context.executing_eagerly(): return False
  return (isinstance(graph, WhileCondFuncGraph) or
          isinstance(graph, WhileBodyFuncGraph))


def create_new_tf_function(func_graph):
  """Converts func_graph to a TF_Function and adds it to the current graph.

  Args:
    func_graph: FuncGraph

  Returns:
    The name of the new TF_Function.
  """
  func = function._EagerDefinedFunction(  # pylint: disable=protected-access
      func_graph.name, func_graph, func_graph.inputs, func_graph.outputs, {})
  func.add_to_graph(func_graph.outer_graph)
  return func_graph.name


def unique_fn_name(scope, name):
  """Returns a unique name to use for a control flow function.

  Args:
    scope: A name scope string.
    name: An identifier for this function (e.g. "true", "body").

  Returns:
    A string, the name to use for the function.
  """
  return ("%s%s_%s" % (scope, name, ops.uid())).replace("/", "_")


def unique_grad_fn_name(forward_name):
  return "%s_grad_%s" % (forward_name, ops.uid())


def maybe_set_lowering_attr(op, lower_using_switch_merge=None):
  """Sets the flag to enable lowering on `op` if necessary.

  Lowering allows cond_v2 and while_v2 to avoid some of the limitations of
  Functions, allowing users to specify devices & colocation inside of cond_v2
  and while_v2 input functions, and enabling non-strict evaluation & partial
  pruning. This brings v2 control flow closer to feature parity with v1 control
  flow.

  However, we do not lower in the following cases:
    - When the `If` or `While` ops are in the XLA context. Because it is easier
      for XLA to apply its own optimizations when dealing with un-lowered
      control flow operators than with low-level control flow primitives.
    - When the eager execution context specifies the executor of functions to
      be the single threaded executor (see context.function_executor_type()).
      Because the single threaded executor does not support v1 control flow ops.
    - When 'lower_using_switch_merge' is explicitly set to False.

  Args:
    op: An `If` or `While` Operation.
    lower_using_switch_merge: Explicit value to lower or not (optional).
  """
  if lower_using_switch_merge is not None:
    # pylint: disable=protected-access
    op._set_attr("_lower_using_switch_merge",
                 attr_value_pb2.AttrValue(b=lower_using_switch_merge))
    # pylint: enable=protected-access
  elif (not _DISABLE_LOWER_USING_SWITCH_MERGE and
        not control_flow_util.GraphOrParentsInXlaContext(op.graph) and
        context.context().function_call_options.executor_type !=
        "SINGLE_THREADED_EXECUTOR"):
    # pylint: disable=protected-access
    op._set_attr("_lower_using_switch_merge", attr_value_pb2.AttrValue(b=True))
    # pylint: enable=protected-access


def maybe_propagate_compile_time_consts_in_xla(op):
  """Tells XLA whether to propagate compile-time consts in the loop body.

  This is needed to make compile time constants available to ops, for example
  `max_num_elements` in `EmptyTensorList`, inside the loop body. Ideally this
  would always be turned on, but that doesn't work with legacy functionalized
  while_loops.

  Args:
    op: A `While` Operation.
  """
  if control_flow_util.GraphOrParentsInXlaContext(op.graph):
    # pylint: disable=protected-access
    op._set_attr("_xla_propagate_compile_time_consts",
                 attr_value_pb2.AttrValue(b=True))
    # pylint: enable=protected-access


def resource_input_index(tensor_name, input_names, node_defs, functions):
  """Returns the index of the input corresponding to `tensor_name`.

  This method is used to find the corresponding index of an arbitrary resource
  tensor in a function (the function could be a loop body). We assume that
  resource handles are never created in functions, so that every resource
  tensor can be traced back to a function input.

  The awkward signature of this method is to make it work with both FuncGraphs
  and FunctionDefs. This is so we can recurse on function call ops without
  building the corresponding FuncGraph (note that even if a FuncGraph for a
  FunctionDef already exists, the input/output/node names may have been
  changed when the FuncGraph was serialized to the FunctionDef, which makes it
  unusable with this algorithm).

  Args:
    tensor_name: the name of the resource tensor to be resolved to an input.
    input_names: a list of the names of all inputs to the function.
    node_defs: a dict mapping op name -> NodeDef for every op in the function.
    functions: a dict mapping function name -> _EagerDefinedFunction.

  Returns:
    The index into input_names corresponding to `tensor_name`.
  """
  while tensor_name not in input_names:
    # FunctionDefs and graphs use different tensor naming conventions.
    parts = tensor_name.split(":")
    if len(parts) == 3:
      op_name, _, output_idx = parts
    elif len(parts) == 2:
      op_name, output_idx = parts
    else:
      assert len(parts) == 1
      op_name = parts[0]
      output_idx = 0
      tensor_name = "%s:%d" % (tensor_name, output_idx)
      # Check again for cases where the tensor suffix (":0") is stripped out.
      if tensor_name in input_names:
        break
    output_idx = int(output_idx)
    node_def = node_defs[op_name]

    def _extract_input_index(function_attribute_name):
      func_name = node_def.attr[function_attribute_name].func.name
      fdef = functions[func_name].definition
      output_arg_name = fdef.signature.output_arg[output_idx].name
      output_tensor_name = fdef.ret[output_arg_name]
      return resource_input_index(
          output_tensor_name, [arg.name for arg in fdef.signature.input_arg],
          {ndef.name: ndef for ndef in fdef.node_def}, functions)

    if node_def.op in ("Identity", "While"):
      # Captured resources occur at the same index in the lists of inputs and
      # outputs of a while or identity op. So we lookup the input of `tensor.op`
      # at the same index as the index of `tensor` in the `tensor.op.outputs`.
      tensor_name = node_def.input[output_idx]
    elif node_def.op in ("PartitionedCall", "StatefulPartitionedCall"):
      # Functions output any captured resource tensors used by their
      # gradients.  `tensor_name` is one of these outputs from a nested
      # function call, so recursively find the corresponding input in the
      # nested FunctionDef.
      tensor_name = node_def.input[_extract_input_index("f")]
    elif node_def.op in ("If", "StatelessIf"):
      input_index = _extract_input_index("then_branch")
      if input_index != _extract_input_index("else_branch"):
        raise AssertionError(
            ("Expected cond branches ({} op) to each have the same "
             "input->output mapping of resources.").format(node_def.op))
      tensor_name = node_def.input[
          # Ignore the `cond` input; the function inputs come after.
          input_index + 1]
    else:
      # We assume there are no other ops types that will "forward" resource
      # handles like this, so all other handles must have been created by the
      # op. (Note that cond_v2 wraps resource handle outputs in optionals,
      # which we'll end up accumulating).
      raise ValueError("Taking gradient of a while loop which creates "
                       "a resource in its body is not supported: %s (%s)"
                       % (op_name, node_def.op))

  return input_names.index(tensor_name)


@tf_contextlib.contextmanager
def clear_control_inputs():
  """Clears the control inputs but preserves the ControlFlowContext.

  This is needed to preserve the XLAControlFlowControl when clearing
  control inputs for the gradient accumulators in while_v2.
  `ops.control_dependencies` does not allow that.

  Yields:
    A context manager in which the ops created will not have any control inputs
    by default but the control flow context is the same.
  """
  # pylint: disable=protected-access
  control_flow_context = ops.get_default_graph()._get_control_flow_context()
  with ops.control_dependencies(None):
    ops.get_default_graph()._set_control_flow_context(control_flow_context)
    yield
  # pylint: enable=protected-access


def _is_tpu_strategy(strategy):
  return (strategy is not None and
          strategy.__class__.__name__.startswith("TPUStrategy"))


def _is_building_keras_layer():
  # TODO(srbs): Remove this function when we no long support session with Keras.
  keras_call_context_function = keras_deps.get_call_context_function()
  if keras_call_context_function:
    return keras_call_context_function().layer is not None
  else:
    return False


def output_all_intermediates():
  """Whether to output all intermediates of a functional control flow op.

  The default behavior is to output intermediates only when building a Keras
  Layer in graph mode and that too when certain other conditions are met:
  1. We do not output intermediates if the functional control flow op
     is being built inside a FuncGraph which is not a If/While graph. This
     guards against outputting intermediates in eager mode since keras adds
     tensors to a FuncGraph named "keras_graph" in that case. Also because we
     do not output intermediates of tf.function (since this feature is only for
     backwards compatibility) outputting intermediates of functional control
     flow ops built inside tf.function is of no value.
  2. We do not output intermediates when the compilation is using XLA or for a
     TPU.
  3. We do not output intermediates when a single threaded executor is used
     since that does not perform inlining and pruning.

  Returns:
    A bool telling whether to output all intermediates.
  """
  if _EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE is not None:
    return _EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
  if in_defun():
    return False
  if (control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()) or
      _is_tpu_strategy(distribution_strategy_context.get_strategy())):
    return False
  if (context.context().function_call_options.executor_type ==
      "SINGLE_THREADED_EXECUTOR"):
    return False
  return _is_building_keras_layer()


def get_func_graph(op, input_shapes, func_name):
  """Generates and returns a FuncGraph for the given op and input_shapes."""
  fdef = None
  graph = op.graph
  # Recursively search the func in graphs.
  while graph is not None:
    func = graph._get_function(func_name)  # pylint: disable=protected-access
    if func is not None:
      fdef = func.definition
      break
    if hasattr(graph, "outer_graph"):
      graph = graph.outer_graph
    else:
      break

  if fdef is None:
    raise KeyError("%s cannot be found in the graph" % func_name)

  # `op.graph` may not be the same as `ops.get_default_graph()` e.g.
  # in the case of nested if ops or when the gradient is being computed
  # from inside a Defun. We build the `func_graph` with `op.graph` as its
  # `outer_graph`. This resembles how the `FuncGraph` was built in the
  # forward pass. We need this so that we can resolve references to tensors
  # in `func_graph` from its gradient graph in `_resolve_grad_inputs`.
  with op.graph.as_default():
    func_graph = function_def_to_graph.function_def_to_graph(
        fdef, input_shapes)
  return func_graph


def get_op_and_outputs(op_or_outputs):
  if isinstance(op_or_outputs, ops.Operation):
    return op_or_outputs, []
  elif not op_or_outputs:  # Empty list.
    return None, []
  else:
    return op_or_outputs[0].op, op_or_outputs


def graph_wrapped_for_higher_order_tape_gradients(graph):
  """Check if `graph` is wrapped by `run_as_function_for_tape_gradients`."""
  while graph is not None:
    if "cflow_gradient_wrapper" in getattr(graph, "name", ""):
      return True
    graph = getattr(graph, "outer_graph", None)
  return False


def run_as_function_for_tape_gradients(make_op, inputs):
  """Fix higher-order tape gradients by wrapping `make_op` in a function.

  Args:
    make_op: A function that takes a list of inputs and returns a list of output
      tensors. This function should set any handle data relevant to its outputs
      before returning.
    inputs: A list of tensors to check for tape gradients and pass to
      `make_op`. These should include all tensors used in `make_op`.

  Returns:
    Tensors corresponding to `make_op`'s output.
  """
  # GradientTapes created inside a function currently don't work well with
  # un-wrapped control flow ops in that same function. Wrapping in an extra
  # layer of intermediate function means we run extra logic in the function
  # gradient code to record the correct intermediates on the tape.
  #
  # The function attribute inputs to control flow ops are not hashable, so we
  # pass everything as a capture to bypass defun's caching.
  if (gradients_util.PossibleTapeGradientTypes(inputs)
      == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER
      # We only need one function between the tape and the op; if we've already
      # wrapped once, we stop wrapping to avoid infinite recursion.
      and not (ops.get_default_graph().building_function
               and "cflow_gradient_wrapper" in ops.get_default_graph().name)):
    results = function.defun_with_attributes(
        make_op,
        autograph=False,
        attributes=dict(func_name="cflow_gradient_wrapper"))(inputs)
    return results
  else:
    return make_op(inputs)
