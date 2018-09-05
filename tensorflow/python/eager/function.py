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
# pylint: disable=unidiomatic-typecheck
"""Defun decorator for defining graph-mode functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import sys
import threading

import numpy as np
import six

from tensorflow.core.framework import function_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2_impl
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

# This is to avoid a circular dependency with cond_v2_impl
# (function -> gradients_impl -> control_flow_ops -> cond_v2_impl).
cond_v2_impl._function = sys.modules[__name__]  # pylint: disable=protected-access


def create_substitute_placeholder(value, name, dtype=None):
  """Creates a placeholder for `value` and propagates shape info to it."""
  # Note: setting ops.control_dependencies(None) ensures we always put
  # capturing placeholders outside of any control flow context.
  with ops.control_dependencies(None):
    placeholder = graph_placeholder(
        dtype=dtype or value.dtype, shape=value.shape, name=name)
  if placeholder.dtype == dtypes_module.resource:
    if isinstance(value, ops.EagerTensor):
      handle_data = value._handle_data  # pylint: disable=protected-access
    else:
      handle_data = resource_variable_ops.get_resource_handle_data(value)
    if handle_data is not None and handle_data.is_set:
      # pylint: disable=protected-access
      pywrap_tensorflow.SetResourceHandleShapeAndType(
          placeholder.graph._c_graph, placeholder._as_tf_output(),
          handle_data.SerializeToString())
      # pylint: enable=protected-access
      # Ensure that shapes and dtypes are propagated.
      shapes, types = zip(*[(pair.shape, pair.dtype)
                            for pair in handle_data.shape_and_type])
      ranks = [len(s.dim) if not s.unknown_rank else -1 for s in shapes]
      shapes = [[d.size for d in s.dim]
                if not s.unknown_rank else None for s in shapes]
      pywrap_tensorflow.TF_GraphSetOutputHandleShapesAndTypes_wrapper(
          placeholder._op._graph._c_graph,  # pylint: disable=protected-access
          placeholder._as_tf_output(),  # pylint: disable=protected-access
          shapes, ranks, types)

  return placeholder


def capture_value(tensor_map, value, dtype, name):
  """Capture a value from outside the function, to pass in as an extra arg."""
  captured_value = tensor_map.get(value, None)
  if captured_value is None:
    captured_value = create_substitute_placeholder(value, name=name,
                                                   dtype=dtype)
    tensor_map[value] = captured_value
  tape.record_operation("captured_value", [captured_value], [value],
                        lambda x: [x])
  return captured_value


class CapturingGraph(ops.Graph):
  """Graph that can capture tensors from other graphs.

  Attributes:
    captures: Maps external tensor -> internal tensor (e.g. input placeholder).
      The entries are in the order they were captured.
  """

  def __init__(self):
    super(CapturingGraph, self).__init__()

    self.captures = collections.OrderedDict()
    self._building_function = True

    # Map from resource tensor name to last op (in program order) which uses
    # this tensor. Used to enforce that execution order matches program order
    # for resource tensors.
    self._last_op_using_resource_tensor = {}

  def clear_resource_control_flow_state(self):
    self._last_op_using_resource_tensor = {}

  # TODO(skyewm): get rid of name and use the name of `tensor`.
  def capture(self, tensor, name=None):
    """Capture `tensor` if it's external to this graph.

    If `tensor` is from a different graph, returns a placeholder for it.
    `tensor` and the placeholder will also appears in self.captures. Multiple
    calls to this method with the same `tensor` argument will return the same
    placeholder. If `tensor` is from this graph, returns `tensor`.

    Args:
      tensor: Tensor. May be from this FuncGraph or a different graph.
      name: Optional name if a placeholder is created.

    Returns:
      Tensor from this FuncGraph.
    """
    if isinstance(tensor, ops.EagerTensor):
      if name is None:
        name = str(ops.uid())
      return capture_value(self.captures, tensor, tensor.dtype, name)
    if tensor.graph is not self:
      if name is None:
        name = tensor.op.name
      return capture_value(self.captures, tensor, tensor.dtype, name)
    return tensor

  def create_op(
      self,
      op_type,
      inputs,
      dtypes,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_shapes=True,
      compute_device=True):
    """Captures an external inputs before calling Graph.capture_op."""
    # This capturing logic interacts poorly with control flow contexts which
    # want to replace inputs of ops far too late in the process. This can lead
    # the context to get confused and try to create an Enter for an Enter. We
    # can detect this here and skip the additional Enter which can confuse loop
    # validation logic.
    if op_type == "Enter" and inputs[0].op.type == "Enter":
      if inputs[0].op.get_attr("frame_name") == attrs["frame_name"].s:
        return inputs[0].op
    # Calling AddValue on the control flow contexts to force creation of the
    # backward accumulators in the original graph before we create placeholders
    # to capture the inputs.
    ctxt = ops.get_default_graph()._control_flow_context  # pylint: disable=protected-access
    for i, inp in enumerate(inputs):
      if ctxt is not None and hasattr(ctxt, "AddValue"):
        inp = ctxt.AddValue(inp)
      inp = self.capture(inp)
      inputs[i] = inp
    return super(CapturingGraph, self).create_op(
        op_type, inputs, dtypes, input_types, name, attrs, op_def,
        compute_device=compute_device)


def _get_device_functions(ctx, graph):
  """Returns a tuple of device functions representing the device stack."""
  if ctx.executing_eagerly():
    return (pydev.merge_device(ctx.device_name),)
  else:
    return tuple(graph._device_functions_outer_to_inner)  # pylint: disable=protected-access


class FuncGraph(CapturingGraph):
  """Graph representing a function body.

  Attributes:
    name: The name of the function.
    inputs: Placeholder tensors representing the inputs to this function. The
      tensors are in this FuncGraph. This represents "regular" inputs as well as
      captured inputs (i.e. the values of self.captures), with the regular
      inputs coming first.
    outputs: Tensors that will be returned by this function. The tensors are in
      this FuncGraph.
    structured_outputs: A possibly-nested python object which will be returned
      by this function. The Tensors in this structure are the same as those of
      self.outputs. Note that this structure might contain Python `None`s.
    variables: Variables that should be watched during function execution.
    outer_graph: The graph this function is defined in. May be another FuncGraph
      or the global default Graph.
    seed: The graph-level random seed.
  """

  def __init__(self, name):
    """Construct a new FuncGraph.

    The graph will inherit its graph key, collections, seed, device stack, and
    distribution strategy stack from the current context or graph.

    Args:
      name: the name of the function.
    """
    super(FuncGraph, self).__init__()

    self.name = name
    self.inputs = []
    self.outputs = []
    self.structured_outputs = None
    self.variables = []
    self.outer_graph = ops.get_default_graph()

    graph = self.outer_graph

    if context.executing_eagerly():
      self.seed = context.global_seed()
      self._xla_compile = (context.context().device_spec.device_type == "TPU")
      self._add_device_to_stack(context.context().device_name)
    else:
      self.seed = graph.seed
      self._xla_compile = getattr(graph, "_xla_compile", False)
      self._device_function_stack = graph._device_function_stack.copy()  # pylint: disable=protected-access
      self._colocation_stack = graph._colocation_stack.copy()  # pylint: disable=protected-access

    # TODO(b/112165328, b/112906995): summaries depend on inheriting collections
    # from the default graph even in eager mode. It'd be nice to not have a
    # default graph with eager execution, so hopefully this will go away when we
    # remove collections.
    # pylint: disable=protected-access
    self._collections = graph._collections
    # TODO(b/112906995): distribution strategy depends on inheriting this stack
    # from the default graph even in eager mode. Maybe it should be part of the
    # eager context?
    self._distribution_strategy_stack = graph._distribution_strategy_stack
    # Inherit the graph key, since this is used for matching variables in
    # optimizers.
    self._graph_key = graph._graph_key
    # pylint: enable=protected-access

  def capture(self, tensor, name=None):
    """Calls CapturingGraph.capture and updates self.inputs if necessary."""
    new_capture = tensor not in self.captures
    internal_tensor = super(FuncGraph, self).capture(tensor, name)

    if new_capture and tensor is not internal_tensor:
      self.inputs.append(internal_tensor)

    return internal_tensor

  @property
  def external_captures(self):
    """External tensors captured by this function."""
    return list(self.captures.keys())

  @property
  def internal_captures(self):
    """Placeholders in this function corresponding captured tensors."""
    return list(self.captures.values())


def _forward_name(n):
  """The name of a generated forward defun named n."""
  return "__forward_%s_%s" % (n, ops.uid())


def _backward_name(n):
  """The name of a generated backward defun named n."""
  return "__backward_%s_%s" % (n, ops.uid())


def _inference_name(n):
  """The name of a forward-but-no-gradient defun named n."""
  return "__inference_%s_%s" % (n, ops.uid())


def _register(fn):
  """Registers the function `fn`."""
  context.context().add_function(fn)


# TODO(apassos) get rid of this by splitting framework.function._DefinedFunction
# so it doesn't have the definition-generating logic and is just a container for
# an already-defined function.
class _EagerDefinedFunction(object):
  """Callable with the interface of `framework.function._DefinedFunction.`

  `_EagerDefinedFunction` encapsulates a function definition and its properties,
  and it provides a method for calling the encapsulated function. Some Ops
  take functions as attributes, which have type `func`; an instance of this
  class may be provided as the value of these `func` attributes.
  """

  def __init__(self, name, graph, inputs, outputs, attrs):
    """Initializes an eager defined function.

    Args:
      name: str, the name for the created function.
      graph: Graph, the graph containing the operations in the function
      inputs: the tensors in the graph to be used as inputs to the function
      outputs: the tensors in the graph which will be outputs to the function
      attrs: dict mapping names of attributes to their AttrValue values
    """
    operations = [
        op for op in graph.get_operations()
        if op not in set(arg.op for arg in inputs)
    ]
    fn = pywrap_tensorflow.TF_GraphToFunction_wrapper(
        graph._c_graph,  # pylint: disable=protected-access
        compat.as_str(name),
        False,
        [o._c_op for o in operations],  # pylint: disable=protected-access
        [t._as_tf_output() for t in inputs],  # pylint: disable=protected-access
        [t._as_tf_output() for t in outputs],  # pylint: disable=protected-access
        [],
        None,
        compat.as_str(""))

    for name, attr_value in attrs.items():
      serialized = attr_value.SerializeToString()
      # TODO(iga): this creates and deletes a new TF_Status for every attr.
      # It might be worth creating a convenient way to re-use status.
      pywrap_tensorflow.TF_FunctionSetAttrValueProto(
          fn, compat.as_str(name), serialized)

    # TODO(apassos) avoid creating a FunctionDef (specially to grab the
    # signature, but also in general it's nice not to depend on it.
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tensorflow.TF_FunctionToFunctionDef(fn, buffer_)
      proto_data = pywrap_tensorflow.TF_GetBuffer(buffer_)
    function_def = function_pb2.FunctionDef()
    function_def.ParseFromString(compat.as_bytes(proto_data))
    if context.executing_eagerly():
      _register(fn)
    self.definition = function_def
    self.name = compat.as_bytes(function_def.signature.name)
    self.signature = function_def.signature
    self._num_outputs = len(self.signature.output_arg)
    self._output_types = [o.type for o in self.signature.output_arg]
    self._output_shapes = [o.shape for o in outputs]
    self.grad_func_name = None
    self.python_grad_func = None
    self._c_func = c_api_util.ScopedTFFunction(fn)
    self._grad_func = None
    self._graph = graph
    self._stateful_ops = tuple(op for op in operations if op.op_def.is_stateful)

  def add_to_graph(self, g):
    # pylint: disable=protected-access
    if self.name not in g._functions:
      g._add_function(self)
    for f in self._graph._functions.values():
      if f.name not in g._functions:
        g._add_function(f)
    # pylint: enable=protected-access

  @property
  def stateful_ops(self):
    return self._stateful_ops

  def call(self, ctx, args):
    """Calls this function with `args` as inputs.

    Function execution respects device annotations only if the function won't
    be compiled with xla.

    Args:
      ctx: a Context object
      args: a list of arguments to supply this function with.

    Returns:
      The outputs of the function call.
    """

    executing_eagerly = ctx.executing_eagerly()

    if self._graph._xla_compile:  # pylint: disable=protected-access
      # XLA compilation relies upon a custom kernel creator to run functions.
      signature = self.signature
      if executing_eagerly:
        outputs = execute.execute(
            str(signature.name),
            num_outputs=self._num_outputs,
            inputs=args,
            attrs=None,
            ctx=ctx)
      else:
        g = ops.get_default_graph()
        self.add_to_graph(g)
        op = g.create_op(
            signature.name,
            [ops.internal_convert_to_tensor(x, ctx=ctx) for x in args],
            tuple(dtypes_module.DType(x.type) for x in signature.output_arg),
            op_def=signature,
            name="FunctionCall",
            compute_shapes=False)
        outputs = op.outputs
        if not outputs:
          return op
        outputs = [outputs] if isinstance(
            outputs, (ops.Tensor, type(None))) else list(outputs)
    else:
      # TODO(akshayka): Either remove this if the FunctionLibraryRuntime
      # creates `PartitionedCallOp` kernels by default, or remove the previous
      # branch if a TPU kernel is registered for `PartitionedCall`.
      outputs = functional_ops.partitioned_call(
          args=args,
          f=self,
          tout=self._output_types,
          executing_eagerly=executing_eagerly)

    if executing_eagerly:
      return outputs
    else:
      for i, shape in enumerate(self._output_shapes):
        outputs[i].set_shape(shape)
      return outputs


def _flatten(sequence):
  """A wrapper around `nest.flatten` that also unpacks `IndexedSlices`."""
  # TODO(akshayka): Support `SparseTensor` in a similar fashion.
  flat_sequence = nest.flatten(sequence)
  outputs = []
  for item in flat_sequence:
    if isinstance(item, ops.IndexedSlices):
      if item.dense_shape is not None:
        outputs.extend([item.values, item.indices, item.dense_shape])
      else:
        outputs.extend([item.values, item.indices])
    else:
      outputs.append(item)
  return outputs


class Function(object):
  """Callable object encapsulating a function definition and its gradient.

  `Function` is a callable that encapsulates a function definition and
  is differentiable under `tf.GradientTape` objects.
  """

  def __init__(self, func_graph, attrs=None):
    """Initialize a Function.

    Args:
      func_graph: An instance of FuncGraph: the function body to wrap.
      attrs: (optional) dict mapping names of attributes to their AttrValue
        values. Attributes in `attrs` will be included in this function's
        definition.

    Raises:
      ValueError: If number of input_placeholders is not equal to the number
        of function inputs.
    """
    self._func_graph = func_graph
    self._captured_inputs = list(self._func_graph.captures.keys())
    self._num_outputs = len(self._func_graph.outputs)
    self._output_shapes = tuple(
        output.shape for output in self._func_graph.outputs)
    self._attrs = attrs or {}
    self._device_functions = tuple(
        self._func_graph._device_functions_outer_to_inner)  # pylint: disable=protected-access

    self._inference_function = _EagerDefinedFunction(
        _inference_name(self._func_graph.name), self._func_graph,
        self._func_graph.inputs, self._func_graph.outputs, self._attrs)
    self._backward_graph_function = None

    # Map holding distributed variables, keyed by resource handle tensors.
    self._distributed_variables = {}
    strategy = distribution_strategy_context.get_distribution_strategy()
    for variable in self._func_graph.variables:
      # If variable is not distributed, unwrap returns [variable].
      component_variables = strategy.unwrap(variable)
      # Only update the dictionary when the variable is actually distributed.
      if (len(component_variables) > 1 or component_variables[0] != variable):
        for component_variable in component_variables:
          self._distributed_variables[component_variable.handle] = variable

  def __call__(self, *args):
    """Executes the wrapped function."""
    ctx = context.context()
    device_functions = _get_device_functions(ctx, ops.get_default_graph())
    if device_functions != self._device_functions:
      raise ValueError(
          "The current device stack does not match the device stack under "
          "which the TensorFlow function '%s' was created.\n"
          "Current device stack: %s\n%s device stack: %s" %
          (self._inference_function.name, device_functions,
           self._inference_function.name, self._device_functions))

    for v in self._func_graph.variables:
      if v.trainable:
        tape.watch_variable(v)

    captures = self._resolve_captured_inputs()
    tensor_inputs = [x for x in nest.flatten(args) if isinstance(x, ops.Tensor)]
    args = tensor_inputs + captures

    if tape.should_record(tensor_inputs) or tape.should_record(captures):
      return self._backprop_call(args)

    outputs = self._inference_function.call(ctx, args)
    return self._build_call_outputs(outputs)

  @property
  def graph(self):
    """Returns the graph from which this function was constructed."""
    return self._func_graph

  @property
  def variables(self):
    """Returns all variables touched by this function."""
    return self._func_graph.variables

  @property
  def inputs(self):
    """Returns tensors in `self.graph` corresponding to arguments."""
    return self._func_graph.inputs

  @property
  def outputs(self):
    """Returns tensors in `self.graph` corresponding to return values."""
    return self._func_graph.outputs

  @property
  def captured_inputs(self):
    """Returns external Tensors captured by this function.

    self.__call__(*args) passes `args + self.captured_inputs` to the function.
    """
    return self._captured_inputs

  @property
  def function_def(self):
    """Returns a `FunctionDef` object representing this function."""
    return self._inference_function.definition

  @property
  def output_shapes(self):
    """The function's output shapes."""
    # TODO(ebrevdo): Should we only keep the output shapes associated
    # with len(self._python_returns) outputs?
    # TODO(akshayka): Consider removing this.
    outputs_list = nest.flatten(self._func_graph.structured_outputs)
    j = 0
    for i, o in enumerate(outputs_list):
      if o is not None:
        if isinstance(o, ops.IndexedSlices):
          # Extract the shape of the `IndexedSlices` object's `values` field.
          outputs_list[i] = self._output_shapes[j]  # the `values` shape
          if o.dense_shape is not None:
            j += 3  # skip over shapes for `values`, `indices`, `dense_shape`
          else:
            j += 2  # skip over shapes for `values`, `indices`
        else:
          outputs_list[i] = self._output_shapes[j]
          j += 1
    return nest.pack_sequence_as(self._func_graph.structured_outputs,
                                 outputs_list)

  @property
  def output_dtypes(self):
    # TODO(akshayka): Consider removing this.
    return nest.map_structure(lambda x: x.dtype if x is not None else None,
                              self._func_graph.structured_outputs)

  def _construct_backprop_function(self):
    """Constructs the backprop function object for this function."""
    backwards_graph = FuncGraph(_backward_name(self._func_graph.name))
    with backwards_graph.as_default():
      gradients_wrt_outputs = [
          graph_placeholder(x.dtype, x.shape) for x in self._func_graph.outputs
      ]
      gradients_wrt_inputs = gradients_impl._GradientsHelper(  # pylint: disable=protected-access
          self._func_graph.outputs,
          self._func_graph.inputs,
          grad_ys=gradients_wrt_outputs,
          src_graph=self._func_graph)

    self._forward_function = _EagerDefinedFunction(
        _forward_name(
            self._func_graph.name), self._func_graph, self._func_graph.inputs,
        self._func_graph.outputs + list(backwards_graph.captures.keys()),
        self._attrs)

    # The ordering of `backwards_graph.inputs` is important: inputs of
    # `self._backward_graph_function` correspond to outputs of
    # `self._forward_function`.
    backwards_graph.inputs = gradients_wrt_outputs + list(
        backwards_graph.captures.values())
    # Clear captures, since we pass them in as inputs.
    backwards_graph.captures = {}
    backwards_graph.outputs.extend(
        grad for grad in _flatten(gradients_wrt_inputs) if grad is not None)
    backwards_graph.structured_outputs = gradients_wrt_inputs
    self._backward_graph_function = Function(
        backwards_graph, attrs=self._attrs)

  def _backprop_call(self, args):
    """Calls the forward function and records the result on a tape.

    (Only records results on a tape if the function has outputs)

    Args:
      args: All inputs to the function, including resolved captured inputs

    Returns:
      The call output.
    """
    if self._backward_graph_function is None:
      self._construct_backprop_function()

    ctx = context.context()
    outputs = self._forward_function.call(ctx, args)
    if isinstance(outputs, ops.Operation) or outputs is None:
      return outputs

    # `real_outputs` are the actual outputs of the inference graph function;
    # `side_outputs` are the intermediate Tensors that were added as outputs to
    # the forward graph function so that we can compute its gradient.
    real_outputs = outputs[:self._num_outputs]
    side_outputs = outputs[self._num_outputs:]

    def backward_function(*args):
      return self._backward_graph_function(*(list(args) + side_outputs))  # pylint: disable=not-callable

    tape.record_operation(self._forward_function.signature.name, real_outputs,
                          args, backward_function)
    return self._build_call_outputs(real_outputs)

  def _resolve_captured_inputs(self):
    """Resolve captured distributed variables to their current values.

    Some inputs can be distributed variables. Such variables yield a different
    component (i.e. actual tf.Variable) variables depending on the context of
    execution.

    Returns:
      a list of resolved captured input tensors.
    """
    if self._distributed_variables:
      # Loop over each captured input and check if it corresponds to something
      # distributed. If so, get its _distributed_container and fetch the
      # component appropriate for the current execution context.
      resolved_captured_inputs = self._captured_inputs[:]
      for i, captured_input in enumerate(self._captured_inputs):
        distributed_var = self._distributed_variables.get(captured_input, None)
        if distributed_var is not None:
          # distributed variables override __getattr__ and substitute the
          # right component variable. In here, `distributed_var.handle`
          # actually does the equivalent of
          # distributed_var.get_current_component_var().handle.
          resolved_captured_inputs[i] = distributed_var.handle
      return resolved_captured_inputs
    return self._captured_inputs

  def _build_call_outputs(self, result):
    """Maps the fdef output list to actual output structure.

    Args:
      result: Output lists defined by FunctionDef.
    Returns:
      The actual call output.
    """
    if self._func_graph.structured_outputs is None:
      return result

    # Use `nest.flatten` instead of `_flatten` in order to preserve any
    # IndexedSlices in `self._func_graph.structured_outputs`.
    outputs_list = nest.flatten(self._func_graph.structured_outputs)
    j = 0
    for i, o in enumerate(outputs_list):
      if o is not None:
        if isinstance(o, ops.IndexedSlices):
          # Repack Tensors for IndexedSlices.
          if o.dense_shape is not None:
            outputs_list[i] = ops.IndexedSlices(
                values=result[j],
                indices=result[j + 1],
                dense_shape=result[j + 2])
            j += 3
          else:
            outputs_list[i] = ops.IndexedSlices(
                values=result[j], indices=result[j + 1])
            j += 2
        else:
          outputs_list[i] = result[j]
          j += 1
    ret = nest.pack_sequence_as(self._func_graph.structured_outputs,
                                outputs_list)
    return ret


def _get_defun_inputs_from_signature(signature):
  """Maps a signature to graph-construction inputs."""
  function_inputs = [
      graph_placeholder(spec.dtype, spec.shape)
      for spec in nest.flatten(signature)
  ]
  return nest.pack_sequence_as(signature, function_inputs)


def _get_defun_inputs_from_args(args):
  """Maps python function args to graph-construction inputs."""
  function_inputs = [
      graph_placeholder(arg.dtype, arg.shape)
      if isinstance(arg, ops.Tensor) else arg for arg in nest.flatten(args)
  ]
  return nest.pack_sequence_as(args, function_inputs)


def func_graph_from_py_func(name, python_func, args, kwds, signature=None):
  """Returns a `FuncGraph` generated from `python_func`.

  Args:
    name: an identifier for the function.
    python_func: the Python function to trace.
    args: the positional args with which the Python function should be called;
      ignored if a signature is provided.
    kwds: the keyword args with which the Python function should be called;
      ignored if a signature is provided.
    signature: a possibly nested sequence of `TensorSpecs` specifying the shapes
      and dtypes of the arguments. When a signature is provided, `args` and
      `kwds` are ignored, and `python_func` is traced with Tensors conforming
      to `signature`. If `None`, the shapes and dtypes are inferred from the
      inputs.

  Returns:
    A FuncGraph.

  Raises:
    TypeError: If any of `python_func`'s return values is neither `None` nor a
      `Tensor`.
  """
  func_graph = FuncGraph(name)
  with func_graph.as_default(), AutomaticControlDependencies() as a:
    variable_scope.get_variable_scope().set_use_resource(True)

    if signature is None:
      func_args = _get_defun_inputs_from_args(args)
      func_kwds = _get_defun_inputs_from_args(kwds)
    else:
      func_args = _get_defun_inputs_from_signature(signature)
      func_kwds = {}

    # Note: `nest.flatten` sorts by keys, as does `_deterministic_dict_values`.
    func_graph.inputs.extend(
        x for x in nest.flatten(func_args) + nest.flatten(func_kwds)
        if isinstance(x, ops.Tensor))

    # Variables to help check whether mutation happens in calling the function
    # Copy the recursive list, tuple and map structure, but not base objects
    func_args_before = nest.pack_sequence_as(func_args, nest.flatten(func_args))
    func_kwds_before = nest.pack_sequence_as(func_kwds, nest.flatten(func_kwds))

    def convert(x):
      """Converts an argument to a Tensor."""
      if x is None:
        return None
      try:
        x = ops.convert_to_tensor_or_indexed_slices(x)
      except (ValueError, TypeError):
        raise TypeError(
            "To be compatible with tf.contrib.eager.defun, Python functions "
            "must return zero or more Tensors; in compilation of %s, found "
            "return value of type %s, which is not a Tensor." %
            (str(python_func), type(x)))
      x = a.mark_as_return(x)
      return x

    this_tape = tape.push_new_tape()
    try:
      func_outputs = python_func(*func_args, **func_kwds)
      # invariant: `func_outputs` contains only Tensors and `None`s.
      func_outputs = nest.map_structure(convert, func_outputs)

      def check_mutation(n1, n2):
        """Check if two list of arguments are exactly the same."""
        errmsg = ("Function to be traced should not modify structure of input "
                  "arguments. Check if your function has list and dictionary "
                  "operations that alter input arguments, "
                  "such as `list.pop`, `list.append`")
        try:
          nest.assert_same_structure(n1, n2)
        except ValueError:
          raise ValueError(errmsg)

        for arg1, arg2 in zip(nest.flatten(n1), nest.flatten(n2)):
          if arg1 is not arg2:
            raise ValueError(errmsg)

      check_mutation(func_args_before, func_args)
      check_mutation(func_kwds_before, func_kwds)
    finally:
      tape.pop_tape(this_tape)

    func_graph.structured_outputs = func_outputs
    # Returning a closed-over tensor does not trigger convert_to_tensor.
    func_graph.outputs.extend(
        func_graph.capture(x)
        for x in _flatten(func_graph.structured_outputs)
        if x is not None)

    # Some captured variables might be components of DistributedValues.
    # Instead of storing non-distributed component variables, we
    # store their distributed containers so we can retrieve the correct
    # component variables at call-time.
    variables = list(this_tape.watched_variables())
    strategy = distribution_strategy_context.get_distribution_strategy()
    for i, variable in enumerate(variables):
      # If variable is not distributed value_container returns itself.
      variables[i] = strategy.value_container(variable)
    func_graph.variables = variables

  # Register any other functions defined in the graph.
  if context.executing_eagerly():
    for f in func_graph._functions.values():  # pylint: disable=protected-access
      # TODO(ashankar): What about the gradient registry?
      _register(f._c_func.func)  # pylint: disable=protected-access

  return func_graph


_TensorType = collections.namedtuple("_TensorType", ["dtype", "shape"])


def _encode_arg(arg):
  """A canonical representation for this argument, for use in a cache key."""

  # `defun` uses dtypes and shapes instead of `Tensors` as cache keys. Dtypes
  # are used because TensorFlow graphs are not parametric w.r.t. dtypes. Shapes
  # are used for both performance reasons, as much TensorFlow code specializes
  # on known shapes to produce slimmer graphs, and correctness, as some
  # high-level APIs require shapes to be fully-known.
  #
  # TODO(akshayka): Add support for sparse tensors.
  #
  # pylint: disable=protected-access
  if isinstance(arg, ops.Tensor):
    return _TensorType(arg.dtype, arg._shape_tuple())
  elif isinstance(arg, ops.IndexedSlices):
    if arg.dense_shape is not None:
      return tuple([
          _TensorType(arg.values.dtype, arg.values._shape_tuple()),
          _TensorType(arg.indices.dtype, arg.indices._shape_tuple()),
          _TensorType(arg.dense_shape.dtype, arg.dense_shape._shape_tuple()),
      ])
    else:
      return tuple([
          _TensorType(arg.values.dtype, arg.values._shape_tuple()),
          _TensorType(arg.indices.dtype, arg.indices._shape_tuple()),
      ])
  elif isinstance(arg, np.ndarray):
    tensor = ops.convert_to_tensor(arg)
    return _TensorType(tensor.dtype, tensor._shape_tuple())
  # pylint: enable=protected-access
  elif isinstance(arg, (list, tuple)):
    return tuple([_encode_arg(elem) for elem in arg])
  elif isinstance(arg, dict):
    return tuple(
        (_encode_arg(key), _encode_arg(arg[key])) for key in sorted(arg))
  else:
    return arg


def _deterministic_dict_values(dictionary):
  return tuple(dictionary[key] for key in sorted(dictionary))


class PolymorphicFunction(object):
  """Wrapper class for the graph functions defined for a Python function.

  See the documentation for `defun` for more information on the semantics of
  defined functions.

  PolymorphicFunction class is thread-compatible meaning that minimal
  usage of defuns (defining and calling) is thread-safe, but if users call other
  methods or invoke the base `python_function` themselves, external
  synchronization is necessary.
  """

  def __init__(self,
               python_function,
               name,
               input_signature=None):
    """Initializes a polymorphic function.

    Args:
      python_function: the function to be wrapped.
      name: the name given to it.
      input_signature: a possibly nested sequence of `TensorSpec` objects
        specifying the input signature of this function. If `None`, a separate
        function is instantiated for each inferred input signature.

    Raises:
      ValueError: if `input_signature` is not None and the `python_function`'s
        argspec has keyword arguments.
    """

    if isinstance(python_function, functools.partial):
      self._python_function = python_function.func
      self._args_to_prepend = python_function.args or tuple()
      self._kwds_to_include = python_function.keywords or {}
    else:
      self._python_function = python_function
      self._args_to_prepend = tuple()
      self._kwds_to_include = {}
    self._name = name
    self._function_cache = collections.OrderedDict()
    self._variables = []

    self._lock = threading.Lock()

    fullargspec = tf_inspect.getfullargspec(self._python_function)
    if tf_inspect.ismethod(self._python_function):
      # Remove `self`: default arguments shouldn't be matched to it.
      args = fullargspec.args[1:]
    else:
      args = fullargspec.args

    # A cache mapping from argument name to index, for canonicalizing
    # arguments that are called in a keyword-like fashion.
    self._args_to_indices = {arg: i for i, arg in enumerate(args)}
    # A cache mapping from arg index to default value, for canonicalization.
    offset = len(args) - len(fullargspec.defaults or [])
    self._arg_indices_to_default_values = {
        offset + index: default
        for index, default in enumerate(fullargspec.defaults or [])
    }
    if input_signature is None:
      self._input_signature = None
    else:
      if fullargspec.varkw is not None or fullargspec.kwonlyargs:
        raise ValueError("Cannot define a TensorFlow function from a Python "
                         "function with keyword arguments when "
                         "input_signature is provided.")

      if not isinstance(input_signature, (tuple, list)):
        raise TypeError("input_signature must be either a tuple or a "
                        "list, received " + str(type(input_signature)))

      self._input_signature = tuple(input_signature)
      self._flat_input_signature = tuple(nest.flatten(input_signature))

  def __call__(self, *args, **kwds):
    """Calls a graph function specialized to the inputs."""
    graph_function, inputs = self._maybe_define_function(*args, **kwds)
    return graph_function(*inputs)

  @property
  def python_function(self):
    """Returns the wrapped Python function."""
    return self._python_function

  # TODO(akshayka): Remove this property.
  @property
  def variables(self):
    """Returns the union of all variables referenced by cached `Function`s`."""
    return self._variables

  def get_concrete_function(self, *args, **kwargs):
    """Returns a `Function` object specialized to inputs and execution context.

    `args` and `kwargs` are ignored if this `PolymorphicFunction` was created
    with an `input_signature`.

    Args:
      *args: inputs to specialize on.
      **kwargs: inputs to specialize on.
    """
    graph_function, _ = self._maybe_define_function(*args, **kwargs)
    return graph_function

  def __get__(self, instance, owner):
    """Makes it possible to defun instance methods."""
    del owner
    # `instance` here is the instance that this `PolymorphicFunction` was
    # accessed through; e.g., for
    #
    #   class Foo(object):
    #
    #     @function.defun
    #     def bar(self):
    #       ...
    #
    #   foo = Foo()
    #   foo.bar()  # `foo.bar` is a `PolymorphicFunction` instance
    #
    # then `instance` will be `foo` (and `owner` will be `Foo`).
    return functools.partial(self.__call__, instance)

  def _cache_key(self, args, kwds, ctx, graph):
    """Computes the cache key given inputs and execution context."""
    if self._input_signature is None:
      inputs = (args, kwds) if kwds else args
      cache_key = tuple(_encode_arg(arg) for arg in inputs)
    else:
      del args, kwds
      cache_key = self._flat_input_signature

    # The graph, or whether we're executing eagerly, should be a part of the
    # cache key so we don't improperly capture tensors such as variables.
    executing_eagerly = ctx.executing_eagerly()
    execution_context = executing_eagerly or graph

    # Putting the device in the cache key ensures that call-site device
    # annotations are respected.
    device_functions = _get_device_functions(ctx, graph)

    # `ops.colocate_with` directives translate into `ops.device` directives when
    # eager execution is enabled.
    colocation_stack = (None if executing_eagerly else
                        tuple(graph._colocation_stack.peek_objs()))  # pylint: disable=protected-access

    return cache_key + (execution_context, device_functions, colocation_stack)

  def _canonicalize_function_inputs(self, *args, **kwds):
    """Canonicalizes `args` and `kwds`.

    Canonicalize the inputs to the Python function using its fullargspec. In
    particular, we parse the varags and kwargs that this
    `PolymorphicFunction` was called with into a tuple corresponding to the
    Python function's positional (named) arguments and a dictionary
    corresponding to its kwargs.

    Args:
      *args: The varargs this object was called with.
      **kwds: The keyword args this function was called with.

    Returns:
      A canonicalized ordering of the inputs.

    Raises:
      ValueError: If a keyword in `kwds` cannot be matched with a positional
        argument when an input signature is specified, or when the inputs
        do not conform to the input signature.
    """
    args = self._args_to_prepend + args
    kwds = dict(kwds, **self._kwds_to_include)
    # Maps from index of arg to its corresponding value, according to `args`
    # and `kwds`; seeded with the default values for the named args that aren't
    # in `args`.
    arg_indices_to_values = {
        index: default
        for index, default in six.iteritems(self._arg_indices_to_default_values)
        if index >= len(args)
    }
    consumed_args = []
    for arg, value in six.iteritems(kwds):
      index = self._args_to_indices.get(arg, None)
      if index is not None:
        arg_indices_to_values[index] = value
        consumed_args.append(arg)
      elif self._input_signature is not None:
        raise ValueError("Cannot define a TensorFlow function from a Python "
                         "function with keyword arguments when "
                         "input_signature is provided.")
    for arg in consumed_args:
      # After this loop, `kwds` will only contain true keyword arguments, as
      # opposed to named arguments called in a keyword-like fashion.
      kwds.pop(arg)
    inputs = args + _deterministic_dict_values(arg_indices_to_values)
    if self._input_signature is None:
      return inputs, kwds
    else:
      assert not kwds
      try:
        nest.assert_same_structure(self._input_signature, inputs)
      except (ValueError, TypeError):
        raise ValueError("Structure of Python function inputs does not match "
                         "input_signature.")
      flat_inputs = nest.flatten(inputs)
      if any(not isinstance(arg, ops.Tensor) for arg in flat_inputs):
        raise ValueError("When input_signature is provided, all inputs to "
                         "the Python function must be Tensors.")
      tensor_specs = [
          tensor_spec.TensorSpec.from_tensor(tensor) for tensor in flat_inputs
      ]
      if any(not spec.is_compatible_with(other)
             for spec, other in zip(self._flat_input_signature, tensor_specs)):
        raise ValueError("Python inputs incompatible with input_signature: "
                         "inputs (%s), input_signature (%s)" %
                         (str(inputs), str(self._input_signature)))
      return inputs, {}

  def _maybe_define_function(self, *args, **kwds):
    """Gets a function for these inputs, defining it if necessary.

    Args:
      *args: args for the Python function.
      **kwds: keywords for the Python function.

    Returns:
      A graph function corresponding to the input signature implied by args and
      kwds, as well as the inputs that the object should be called with.

    Raises:
      ValueError: If inputs are incompatible with the input signature.
      TypeError: If the function inputs include non-hashable objects
    """

    args, kwds = self._canonicalize_function_inputs(*args, **kwds)
    cache_key = self._cache_key(args, kwds, context.context(),
                                ops.get_default_graph())
    with self._lock:
      try:
        graph_function = self._function_cache.get(cache_key, None)
      except TypeError:
        raise TypeError("Arguments supplied to `defun`-generated functions "
                        "must be hashable.")

      if graph_function is None:
        graph_function = Function(
            func_graph_from_py_func(self._name, self._python_function, args,
                                    kwds, self._input_signature))
        self._variables.extend(
            [v for v in graph_function.variables if v not in self._variables])
        self._function_cache[cache_key] = graph_function
      return graph_function, (args, kwds)


def _validate_signature(signature):
  if any(not isinstance(arg, tensor_spec.TensorSpec)
         for arg in nest.flatten(signature)):
    raise TypeError("Invalid input_signature %s; input_signature must be "
                    "a possibly nested sequence of TensorSpec objects.")


def defun(func=None, input_signature=None):
  """Compiles a Python function into a callable TensorFlow graph.

  `defun` (short for "define function") trace-compiles a Python function
  composed of TensorFlow operations into a callable that executes a `tf.Graph`
  containing those operations. The callable produced by `defun` contains only
  the subgraph of TensorFlow operations that were executed when the Python
  function was called with a particular input signature, defined as a list
  of the shapes and dtypes of the Python function's Tensor-valued arguments and
  the values of its non-Tensor Python objects. In particular, `defun` is _not_ a
  compiler for arbitrary Python code.

  When eager execution is enabled, the ability to create graphs from Python
  functions makes it possible to incrementally trade off debugability and
  interactivity for performance.  Functions compiled with `defun` cannot be
  inspected with `pdb` and `print` statements; however, executing a graph
  generated by `defun` sometimes takes less time and memory than eagerly
  executing the corresponding Python function, since specifying computations as
  graphs allows for optimizations like automatic buffer reuse and
  parallelization among ops. Note that executing a `defun`-compiled function
  incurs a small constant overhead, so eagerly executing sufficiently small
  Python functions might take less time than executing their corresponding
  `defun`-generated graphs.

  For a Python function to be compatible with `defun`, all of its arguments must
  be hashable Python objects or lists thereof. The function itself may not
  modify the list/map structure of its arguments. Additionally, it must return
  zero or more `tf.Tensor` objects. If the Python function returns
  a `tf.Variable`, its compiled version will return the value of that variable
  as a `tf.Tensor`.

  Executing a graph generated by `defun` respects device annotations (i.e.,
  all `with tf.device` directives present in a Python function will also be
  present in its corresponding graph), but it is not yet possible to execute the
  generated graphs across multiple machines.

  _Example Usage_

  ```python
  import tensorflow as tf

  tf.enable_eager_execution()

  # A simple example.
  def f(x, y):
    return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)

  g = tf.contrib.eager.defun(f)

  x = tf.constant([[2.0, 3.0]])
  y = tf.constant([[3.0, -2.0]])

  # `f` and `g` will return the same value, but `g` will be executed as a
  # TensorFlow graph.
  assert f(x, y).numpy() == g(x, y).numpy()

  # `defun` is capable of compiling Python functions that close over Python
  # objects, including Tensors and Variables.
  @tf.contrib.eager.defun
  def h():
    return f(x, y)

  assert (h().numpy() == f(x, y).numpy()).all()

  # `defun` automatically lifts variables out of the graphs it creates,
  # allowing you to compile the `call` methods of `tf.keras.layers.Layer` and
  # `tf.keras.Model` objects.
  class MyModel(tf.keras.Model):

    def __init__(self, keep_probability=0.2):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
      self.keep_probability = keep_probability

    @tf.contrib.eager.defun
    def call(self, inputs, training=True):
      x = self.dense2(self.dense1(inputs))
      if training:
        return tf.nn.dropout(x, self.keep_probability)
      else:
        return x

  model = MyModel()
  model(x, training=True)  # executes a graph, with dropout
  model(x, training=False) # executes a graph, without dropout

  # `defun`-compiled functions are differentiable.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
  with tf.GradientTape() as tape:
    outputs = model(x)
  gradient = tape.gradient(outputs, model.trainable_variables)
  optimizer.apply_gradients((grad, var) for grad, var in zip(gradient,
                            model.trainable_variables))
  ```

  When using `defun`, there are subtleties regarding inputs, Python control
  flow, and variable creation that one should be aware of. For concreteness, let
  `f` be a Python function that returns zero or more `tf.Tensor` objects and
  let `F = defun(f)`. `F` builds a graph for each unique input signature it
  sees, Python control flow is baked into graphs, and operations related to
  variable initialization are automatically lifted out of the graphs that `F`
  generates and placed in the eager context if executing eagerly or into an
  outer graph otherwise.

  _Input Signatures_
  By default, `F = tf.contrib.eager.defun(f)` instantiates a separate graph
  for every unique sequence of the shapes and dtypes of Tensor arguments and
  the values of Python objects it is invoked with. For example, calling
  `F(tf.random_uniform([2])` will execute a different graph than
  `F(tf.random_uniform([3])` because the two inputs have different shapes.
  The first time that `F(*args, **kwargs)` is called with a particular sequence
  of Tensor shapes and dtypes and Python values, it constructs a graph by
  tracing the execution of `f(*args, **kwargs)`; this graph is bound to an
  input signature inferred from `(*args, **kwargs)` and cached for future reuse.

  `tf.contrib.eager.defun` caches graphs for your convenience, letting you
  define TensorFlow functions without explicitly specifying their signatures.
  However, this policy is conservative and potentially expensive; for example,
  when different invocations of your function have differently-shaped Tensor
  inputs, this policy might generate more graph functions than necessary. To
  eliminate such costs, `tf.contrib.eager.defun` allows you to supply an
  optional `input_signature` argument specifying the shapes and dtypes of the
  inputs. In particular, the shapes may be partially unspecified, with `None`s
  in the unknown dimensions.  When an input signature is provided,
  `tf.contrib.eager.defun` will only instantiate a single graph for the
  decorated Python function. The following is an example:

  ```python
  import tensorflow as tf

  # The first `TensorSpec` below describes the shape and dtype of `words`,
  # and the second describes the shape and dtype of `another_tensor`. Note that
  # the last dimension of the `words` `TensorSpec` is left unspecified.
  @tf.contrib.eager.defun(input_signature=[
    tf.contrib.eager.TensorSpec(shape=[50, 300, None], dtype=tf.float32),
    tf.contrib.eager.TensorSpec(shape=[300, 100], dtype=tf.float32)
  ])
  def my_sequence_model(words, another_tensor):
    ...

  # Note how the third dimension of the first input can vary freely.
  words = tf.random_uniform(([50, 300, 10])
  second_input = tf.random_uniform([300, 100])
  my_sequence_model(words, second_input)

  words = tf.random_uniform(([50, 300, 20])
  my_sequence_model(words, second_input)

  # Passing an input with an incompatible shape will raise an error.
  words = tf.random_uniform(([50, 100, 20])
  my_sequence_model(words, second_input)  # <---- This will raise an error.

  ```

  Python functions that are compiled with an `input_signature` must only accept
  Tensors as arguments and must not take unnamed keyword arguments (**kwargs).

  _Tracing_
  Be aware that because `F` only logs TensorFlow operations, all the other
  Python code that `f` executes will only shape the _construction_ of the graphs
  that `F` executes: the Python code won't be executed when the graphs
  themselves are executed, though it will be executed every time the Python
  function is traced (and a given Python function might be traced multiple
  times, once for each input signature it is invoked with). For example, whereas
  the Python function

  ```python
  import tensorflow as tf
  import numpy as np

  tf.enable_eager_execution()

  def add_noise():
    return tf.eye(5) + np.random.randn(5, 5)
  ```

  will return a different output everytime it is invoked, the compiled function
  `compiled = tf.contrib.eager.defun(add_noise)` will return the same value
  every time it is called, since a particular random offset generated by NumPy
  will be inserted into the graph as a TensorFlow constant. The solution is to
  replace the call to `np.random.randn` with `tf.random_normal((5, 5))`.

  _Python Side-Effects_
  A corollary of the previous discussion on tracing is the following: If a
  Python function `f` has Python side-effects, then executing `f` multiple times
  will not necessarily be semantically equivalent to executing `F =
  tf.contrib.eager.defun(f)` multiple times; this difference is due to the fact
  that `defun` only captures the subgraph of TensorFlow operations that is
  constructed when `f` is called in a graph-building context.

  _Python Control Flow_.
  The structure of many machine learning computations depend upon whether one is
  training or validating, and it is common to nest specialized logic under `if
  training:` blocks. By mapping each input signature to a unique graph, `defun`
  lets users transparently compile such code, as the following code snippet
  demonstrates:

  ```python
  import tensorflow as tf

  tf.enable_eager_execution()

  @tf.contrib.eager.defun
  def lossy_matmul(W, x, training=True):
    outputs = tf.matmul(W, x)
    if training:
      outputs = tf.nn.dropout(outputs, keep_probability=0.2)
    return outputs

  W = tf.random_normal((3, 5))
  x = tf.random_normal((5, 1))

  # Executes a graph that applies dropout.
  lossy_outputs = lossy_matmul(W, x, training=True)

  # Executes a graph that does not apply dropout.
  exact_outputs = lossy_matmul(W, x, training=False)
  ```

  On the other hand, because `defun` generates graphs by tracing and not by
  source code analysis, it fully unrolls Python `for` and `while` loops,
  potentially creating large graphs. If your Python function has native loops
  that run for many iterations, consider replacing them with `tf.while_loop`
  operations.

  When constructing graphs, `tf.Tensor` objects cannot be used as Python
  `bool` objects. This means, for example, that you should replace code in `f`
  resembling

  ```python

  if tensor < 10:
    true_fn()
  else:
    false_fn()
  ```

  with `tf.cond(tensor < 10, true_fn, false_fn)`.

  _Variables_
  TensorFlow operations related to variable creation and initialization are
  automatically lifted out of the graphs generated by `defun`. In practice, this
  implies that variable creation and initialization only happen the first time
  `F` is called, and that variables are reused every time thereafter. Many
  TensorFlow APIs, like `tf.keras.layers.Layer` objects, create variables the
  first time they are called and reuse them thereafter. Automatic variable
  lifting makes it possible to compile these APIs without extra effort, at the
  cost of introducing a discrepancy between the semantics of executing Python
  functions and their corresponding compiled functions. For example:

  ```python
  import tensorflow as tf

  tf.enable_eager_execution()

  def fn():
    x = tf.Variable(0.0)
    x.assign_add(1.0)
    return x.read_value()

  # `fn` is a Python function, so x is created, initialized, and destroyed upon
  # every invocation
  assert fn().numpy() == fn().numpy() == 1.0

  compiled = tf.contrib.eager.defun(fn)

  # Compiling `fn` with `defun` hoists all variables outside of the generated
  # graph, so initialization happens exactly once.
  assert compiled().numpy() == 1.0
  assert compiled().numpy() == 2.0
  ```

  Finally, because each input signature is bound to a unique graph, if your
  Python function constructs `tf.Variable` objects, then each graph constructed
  for that Python function will reference a unique set of variables. To
  circumvent this problem, we recommend against compiling Python functions that
  create `tf.Variable` objects. Instead, Python functions should either
  lexically close over `tf.Variable` objects or accept them as arguments,
  preferably encapsulated in an object-oriented container. If you must create
  variables inside your Python function and you want each graph generated for it
  to reference the same set of variables, add logic to your Python function that
  ensures that variables are only created the first time it is called and are
  reused for every subsequent invocation; note that this is precisely what
  `tf.keras.layers.Layer` objects do, so we recommend using them to represent
  variable-bearing computations whenever possible.

  Args:
    func: function to be compiled. If `func` is None, returns a
      decorator that can be invoked with a single argument - `func`. The
      end result is equivalent to providing all the arguments up front.
      In other words, defun(input_signature=...)(func) is equivalent to
      defun(func, input_signature=...). The former allows
      the following use case:
        @tf.contrib.eager.defun(input_signature=...)
        def foo(...):
          ...

    input_signature: A possibly nested sequence of
      `tf.contrib.eager.TensorSpec` objects specifying the shapes and dtypes of
      the Tensors that will be supplied to this function. If `None`, a separate
      function is instantiated for each inferred input signature.  If a
      signature is specified, every input to `func` must be a `Tensor`, and
      `func` cannot accept `**kwargs`.

  Returns:
     If `func` is not None, returns a callable that will execute the compiled
     function (and return zero or more `tf.Tensor` objects).
     If `func` is None, returns a decorator that, when invoked with a single
     `func` argument, returns a callable equivalent to the case above.

  Raises:
    TypeError: If `input_signature` is neither `None` nor a sequence of
      `tf.contrib.eager.TensorSpec` objects.
  """

  if input_signature is not None:
    _validate_signature(input_signature)

  # TODO(apassos): deal with captured global state. Deal with control flow.
  def decorated(function):
    try:
      name = function.__name__
    except AttributeError:
      name = "function"
    return tf_decorator.make_decorator(
        function,
        PolymorphicFunction(function, name, input_signature=input_signature))

  # This code path is for the `foo = tfe.defun(foo, ...)` use case
  if func is not None:
    return decorated(func)

  # This code path is for the
  #
  # @tfe.defun(...)
  # def foo(...):
  #    ...
  #
  # use case, which is equivalent to `foo = tfe.defun(...)(foo)`
  return decorated


class AutomaticControlDependencies(object):
  """Context manager to automatically add control dependencies.

  Code under this context manager will act as if a sensible set of control
  dependencies were present. More specifically:
    1. All stateful ops in the scope will execute
    2. Stateful ops which modify the same resource will execute in program order

  Note: creating variables in an automatic control dependencies context is not
  supported (the value of the variables will never change as they will keep
  getting reinitialized).

  NOT THREAD SAFE
  """

  def __init__(self):
    self._returned_tensors = set()

  def mark_as_return(self, tensor):
    """Acts like identity but marks the `Tensor` as a return value.

    This will possibly return a copy of the `Tensor`. Usage:

    ```
      with AutomaticControlDependencies() as a:
       ...
       t = a.mark_as_return(t)
      _ = ...(t...)  # i.e. it's safe to use t here
    ```

    Args:
      tensor: the `Tensor` to be marked

    Returns:
      a copy of the `Tensor`.
    """
    if isinstance(tensor, ops.IndexedSlices):
      values = array_ops.identity(tensor.values)
      indices = array_ops.identity(tensor.indices)
      self._returned_tensors.add(indices)
      self._returned_tensors.add(values)
      return ops.IndexedSlices(values, indices, dense_shape=tensor.dense_shape)
    # We want to make the return values depend on the stateful operations, but
    # we don't want to introduce a cycle, so we make the return value the result
    # of a new identity operation that the stateful operations definitely don't
    # depend on.
    tensor = array_ops.identity(tensor)
    self._returned_tensors.add(tensor)
    return tensor

  def __enter__(self):
    if context.executing_eagerly():
      return self
    # This code assumes no other thread is adding ops to the graph while
    # we're adding ops to the graph.
    # TODO(apassos): Fix this by locking the graph or using a temporary
    # graph (but that would mess up devices and collections at least,
    # probably other things as well).
    self._graph = ops.get_default_graph()
    self._n_operations = len(self._graph.get_operations())
    return self

  def _process_switch(self, switch_op, ops_which_must_run,
                      last_op_using_resource_tensor, merge_for_resource):
    """Processes a switch node for a resource input.

    When tensorflow creates a cond, it creates a control flow context for each
    branch of the cond. Each external tensor accessed by that branch is routed
    through a switch op, which gets created in the graph _after_ the op which
    uses that tensor get created.

    If the resource comes from another switch op we process that one first.

    _process_switch creates a corresponding merge node for the switch node. This
    merge node is added to the outer control flow context of the switch
    node. We also ensure that:

      1. The switch node executes after the previous op which used the resource
         tensor

      2. Any op which uses a resource output of the switch node executes before
         the merge for the switch node.

      3. The next op which uses the input resource to the switch node (which
         might be another switch node for the other branch of the conditional)
         will execute after the merge node is done.

      4. The merge node is marked as must_run so it will run even if no
         subsequent operation uses the resource.

    Args:
      switch_op: the switch op to be processed
      ops_which_must_run: the set of ops which must run
      last_op_using_resource_tensor: map from resource tensor to last op using
        it
      merge_for_resource: map from resource tensor to merge which must follow
        all usages of it.
    """
    inp = switch_op.inputs[0]
    if inp.dtype == dtypes_module.resource and inp.op.type == "Switch":
      self._process_switch(inp.op, ops_which_must_run,
                           last_op_using_resource_tensor, merge_for_resource)
    if switch_op.outputs[0] in merge_for_resource:
      return
    new_merge = control_flow_ops.merge(switch_op.outputs,
                                       name="artificial_merge")
    new_merge[0].op._control_flow_context = (  # pylint: disable=protected-access
        switch_op._control_flow_context.outer_context)  # pylint: disable=protected-access
    # Ensures the merge always runs
    ops_which_must_run.add(new_merge[0].op)
    if inp in last_op_using_resource_tensor:
      # Ensures the switch executes after the previous op using the resource.
      switch_op._add_control_input(last_op_using_resource_tensor[inp])  # pylint: disable=protected-access
    # Ensure the next op outside the cond happens after the merge.
    last_op_using_resource_tensor[inp] = new_merge[0].op
    if inp in merge_for_resource:
      merge_for_resource[inp]._add_control_input(new_merge[0].op)  # pylint: disable=protected-access
    for o in switch_op.outputs:
      # Ensures the merge will execute after all ops inside the cond
      merge_for_resource[o] = new_merge[0].op

  def __exit__(self, unused_type, unused_value, unused_traceback):
    if context.executing_eagerly():
      return

    if self._graph is not ops.get_default_graph():
      raise RuntimeError(
          "Graph changed while trying to add control dependencies.")

    # map from resource tensor to the last op which used it
    last_op_using_resource_tensor = {}
    # set of conditional and loop exits
    ops_which_must_run = set()
    # merge which must depend on ops which use this resource
    merge_for_resource = {}

    new_operations = self._graph.get_operations()[self._n_operations:]

    # Ensures that uses of resource tensors get serialized properly and all
    # execute. This is done by keeping a map from resource tensor to the last op
    # in graph-construction order which used it (last_op_using_resource_tensor).
    #
    # Conditionals are written in TensorFlow such that every external tensor
    # accessed in the conditional goes through a switch op and every return
    # tensor (it's guaranteed that there will be at least one) goes through a
    # merge op.
    #
    # To handle conditionals, switches are handled in a special way (see
    # comments for _process_switch). Merge nodes created by TF's conditional
    # logic (as opposed to by _process_switch) are forced to run and also get a
    # control dependency added to them to ensure all stateful ops inside their
    # control flow context run.
    #
    # We also ensure that if an op is using a resource output by a switch node
    # (that is, a resource tensor for which there's a value in
    # merge_for_resource) this op will run before the merge for that resource.
    #
    # We try to add control inputs to nodes respecting their control flow
    # contexts to avoid dead nodes propagating everywhere and leading to
    # "retval[0] doesn't have value" errors. If a node gets a control dependency
    # on a dead node (i.e. a note from an untaken control flow branch) that node
    # will be marked as dead unless it's a merge node.
    #
    # TODO(apassos): serialize non-resource-taking stateful ops as well, and
    # test that it works. Support while loops. Support init_scope escaping from
    # this.
    for op in new_operations:
      # TODO(apassos) make this code safely support while loops.
      if isinstance(op._control_flow_context, control_flow_ops.WhileContext):  # pylint: disable=protected-access
        continue
      control_inputs = set()
      # Ensure stateful ops run
      if (op.type not in self._graph._registered_ops  # pylint: disable=protected-access
          or self._graph._registered_ops[op.type].is_stateful):  # pylint: disable=protected-access
        ops_which_must_run.add(op)
      # Ignore switches (they're handled separately)
      if op.type == "Switch" and op.inputs[0].dtype == dtypes_module.resource:
        continue
      # Make merges trigger all other computation which must run
      if op.type == "Merge":
        for o in ops_which_must_run:
          op._add_control_input(o)  # pylint: disable=protected-access
          for inp in o.inputs:
            if inp in last_op_using_resource_tensor:
              last_op_using_resource_tensor[inp] = op
        ops_which_must_run = set([op])
        continue
      for inp in op.inputs:
        if inp.dtype == dtypes_module.resource:
          # Deal with switches, finally.
          if inp.op.type == "Switch":
            self._process_switch(inp.op, ops_which_must_run,
                                 last_op_using_resource_tensor,
                                 merge_for_resource)
          # Ensure uses of resources are serialized
          if inp in last_op_using_resource_tensor:
            if (last_op_using_resource_tensor[inp]._control_flow_context  # pylint: disable=protected-access
                is op._control_flow_context):  # pylint: disable=protected-access
              control_inputs.add(last_op_using_resource_tensor[inp])
          # Ensure merges happen after the closing of a cond block
          if inp in merge_for_resource:
            merge_for_resource[inp]._add_control_input(op)  # pylint: disable=protected-access
          last_op_using_resource_tensor[inp] = op
      control_inputs = [c for c in control_inputs
                        if c._control_flow_context is op._control_flow_context]  # pylint: disable=protected-access
      op._add_control_inputs(control_inputs)  # pylint: disable=protected-access

    # Ensure all ops which must run do run
    for r in self._returned_tensors:
      if ops_which_must_run:
        r.op._add_control_inputs(  # pylint: disable=protected-access
            [o for o in ops_which_must_run
             if o._control_flow_context is r.op._control_flow_context])  # pylint: disable=protected-access


def automatic_control_dependencies(f):
  """Wraps f to automatically insert control dependencies.

  The inserted dependencies ensure that:
    1. All stateful ops in f run when the result of f runs
    2. Updates to the same resources happen in order.

  Args:
    f: the function to be wrapped.

  Returns:
    The wrapped function.
  """

  def wrapper(*args, **kwds):
    with AutomaticControlDependencies() as a:
      result = f(*args, **kwds)
      result_flat = [a.mark_as_return(t) for t in nest.flatten(result)]
      return nest.pack_sequence_as(result, result_flat)

  return tf_decorator.make_decorator(f, wrapper)
