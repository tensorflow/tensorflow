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
import threading

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator


def capture_value(tensor_map, value, dtype, name):
  """Capture a value from outside the function, to pass in as an extra arg."""
  captured_value = tensor_map.get(ops.tensor_id(value), None)
  if captured_value is None:
    # Note: setting ops.control_dependencies(None) ensures we always put
    # capturing placeholders outside of any control flow context.
    with ops.control_dependencies(None):
      captured_value = graph_placeholder(
          dtype=dtype or value.dtype, shape=value.shape, name=name)
    if captured_value.dtype == dtypes_module.resource:
      if ops._USE_C_SHAPES:  # pylint: disable=protected-access
        if isinstance(value, ops.EagerTensor):
          handle_data = value._handle_data  # pylint: disable=protected-access
        else:
          handle_data = resource_variable_ops.get_resource_handle_data(value)
      else:
        handle_data = value._handle_data  # pylint: disable=protected-access
      if handle_data is not None and handle_data.is_set:
        # pylint: disable=protected-access
        if ops._USE_C_SHAPES:
          pywrap_tensorflow.SetResourceHandleShapeAndType(
              captured_value.graph._c_graph, captured_value._as_tf_output(),
              handle_data.SerializeToString())
        else:
          captured_value._handle_data = handle_data
        # pylint: enable=protected-access
        # Ensure that shapes and dtypes are propagated.
        shapes, types = zip(*[(pair.shape, pair.dtype)
                              for pair in handle_data.shape_and_type])
        ranks = [len(s.dim) if not s.unknown_rank else -1 for s in shapes]
        shapes = [[d.size for d in s.dim]
                  if not s.unknown_rank else None for s in shapes]
        pywrap_tensorflow.TF_GraphSetOutputHandleShapesAndTypes_wrapper(
            captured_value._op._graph._c_graph,  # pylint: disable=protected-access
            captured_value._as_tf_output(),  # pylint: disable=protected-access
            shapes, ranks, types)

    tensor_map[ops.tensor_id(value)] = (value, captured_value)
  else:
    captured_value = captured_value[1]
  tape.record_operation("captured_value", [captured_value], [value],
                        lambda x: [x])
  return captured_value


class CapturingGraph(ops.Graph):
  """Graph used when constructing eager functions."""

  def __init__(self, captures):
    super(CapturingGraph, self).__init__()
    self._building_function = True
    self.captures = captures
    # Map from resource tensor name to last op (in program order) which uses
    # this tensor. Used to enforce that execution order matches program order
    # for resource tensors.
    self._last_op_using_resource_tensor = {}

  # TODO(apassos) remove once the C API is used by default.
  def _use_c_api_hack(self):
    return True

  def clear_resource_control_flow_state(self):
    self._last_op_using_resource_tensor = {}

  def capture(self, tensor, name=None):
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
    # TODO(apassos) this should do some form of alias analysis as ops which
    # forward the resources such as Identity and Switch can cause serialization
    # to fail.
    for i, inp in enumerate(inputs):
      inputs[i] = self.capture(inp)
    return super(CapturingGraph, self).create_op(
        op_type, inputs, dtypes, input_types, name, attrs, op_def,
        compute_device=compute_device)


# pylint: disable=invalid-name
class HelperContext(object):
  """ControlFlowContext with a customizable AddOp method."""

  def __init__(self, add_op_internal):
    self._add_op_internal = add_op_internal
    self._values = set()  # control flow code sometimes updates this.

  def _AddOpInternal(self, op):
    self._add_op_internal(op)

  @property
  def outer_context(self):
    return self._outer_context

  def GetWhileContext(self):
    if self._outer_context:
      return self._outer_context.GetWhileContext()

  def IsWhileContext(self):
    return False

  def IsCondContext(self):
    return False

  def IsXLAContext(self):
    return False

  def AddOp(self, op):  # pylint: disable=invalid-name
    self._AddOpInternal(op)
    if self._outer_context:
      self._outer_context.AddOp(op)

  def AddName(self, _):
    pass

  def AddInnerOp(self, op):
    self._AddOpInternal(op)
    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  def AddValue(self, val):
    if self._outer_context:
      return self._outer_context.AddValue(val)
    else:
      return val

  def EnterGradientColocation(self, op, gradient_uid):
    """Start building a gradient colocated with an op."""
    if self._outer_context:
      self._outer_context.EnterGradientColocation(op, gradient_uid)

  def ExitGradientColocation(self, op, gradient_uid):
    """Start building a gradient colocated with an op."""
    if self._outer_context:
      self._outer_context.ExitGradientColocation(op, gradient_uid)

  def __enter__(self):
    # pylint: disable=protected-access
    self._g = ops.get_default_graph()
    self._outer_context = self._g._get_control_flow_context()
    self._g._set_control_flow_context(self)
    self._nested_contexts = (
        self._outer_context._nested_contexts
        if self._outer_context is not None else None)
    # pylint: enable=protected-access

  def __exit__(self, *_):
    self._g._set_control_flow_context(self._outer_context)  # pylint: disable=protected-access
# pylint: enable=invalid-name


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


_xla_compile_attr = "_XlaCompile"


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

  def __init__(self, name, graph, operations, inputs, outputs, attrs):
    """Initializes an eager defined function.

    Args:
      name: str, the name for the created function.
      graph: Graph, the graph containing the operations in the function
      operations: list of Operation; the subset of operations in the graph
        which will be in the function
      inputs: the tensors in the graph to be used as inputs to the function
      outputs: the tensors in the graph which will be outputs to the function
      attrs: dict mapping names of attributes to their AttrValue values
    """
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
    self._xla_compile = _xla_compile_attr in attrs

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

  def call(self, ctx, args, output_shapes):
    """Calls this function with `args` as inputs.

    Function execution respects device annotations only if the function won't
    be compiled with xla.

    Args:
      ctx: a Context object
      args: a list of arguments to supply this function with.
      output_shapes: shapes to which outputs should be set; ignored when
        executing eagerly.

    Returns:
      The outputs of the function call.
    """

    executing_eagerly = ctx.executing_eagerly()

    xla_compile = self._xla_compile or (executing_eagerly and
                                        ctx.device_spec.device_type == "TPU")

    if xla_compile:
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
      for i, shape in enumerate(output_shapes):
        outputs[i].set_shape(shape)
      return outputs


def _map_sequence_obj_to_idx(sequence):
  """Maps objs in the sequence from id(obj) to sequence index."""
  return {id(x): i for i, x in enumerate(sequence)}


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


# TODO(akshayka): Perhaps rename to something more appropriate.
class GraphModeFunction(object):
  """Callable object encapsulating a function definition and its gradient.

  `GraphModeFunction` is a callable that encapsulates a function definition and
  is differentiable under `tf.GradientTape` objects.
  """

  def __init__(self,
               name,
               input_placeholders,
               extra_inputs,
               graph,
               operations,
               outputs,
               python_func_outputs,
               output_shapes,
               variables=None,
               attrs=None):
    """Initialize a GraphModeFunction.

    Args:
      name: str the name of the created function
      input_placeholders: list of placeholder values (tensors) to feed when
        calling the wrapped function.
      extra_inputs: Tensor inputs this function definition closed over which
        are passed as arguments. Need to track so gradients are supported
        correctly.
      graph: the Graph from which the operations will be pulled. Used as
        a context when computing gradients.
      operations: the subset of Operations in the graph used in the function
        definition.
      outputs: a flat list of the Tensors in the graph used as outputs to the
        function
      python_func_outputs: a possibly nested python object which will be
        returned by this function. The Tensors in this structure will be
        replaced by their corresponding values in outputs. Note that this
        structure might contain Python `None`s.
      output_shapes: List of shapes of all tensors in outputs
      variables: (optional) List of variables to watch during function
        execution.
      attrs: (optional) dict mapping names of attributes to their AttrValue
        values. Attributes in `attrs` will be included in this function's
        definition.
    """
    self._attrs = attrs or {}
    defined_function = _EagerDefinedFunction(
        name, graph, operations, input_placeholders, outputs, self._attrs)
    if len(input_placeholders) != len(defined_function.signature.input_arg):
      raise ValueError("Internal error: invalid lengths. %s %s" % (
          len(input_placeholders), len(defined_function.signature.input_arg)))
    self._input_placeholders = input_placeholders
    self._extra_inputs = list(extra_inputs)
    self._graph = graph
    self._backward_function = None
    self._func_name = name
    self._function_def = defined_function
    self._num_outputs = len(defined_function.signature.output_arg)
    self._ops = operations
    self._python_func_outputs = python_func_outputs
    self._python_returns = [python_func_outputs] if isinstance(
        python_func_outputs,
        (ops.Tensor, type(None))) else _flatten(python_func_outputs)
    self._output_shapes = output_shapes
    self._variables = variables if variables is not None else []

  @property
  def variables(self):
    return self._variables

  def _construct_backprop_function(self):
    """Constructs the backprop function object for this function."""
    with self._graph.as_default():
      c_known_ops = set()
      c_captured_tensors = set()

      existing_op_len = len(self._graph.get_operations())
      filtered_outputs = [x for x in self._python_returns if x is not None]
      self._out_grad_placeholders = [
          graph_placeholder(x.dtype, x.shape) for x in filtered_outputs]
      in_gradients = gradients_impl.gradients(
          filtered_outputs,
          self._input_placeholders,
          grad_ys=self._out_grad_placeholders)
      for op in self._graph.get_operations()[existing_op_len:]:
        if op.type in ["Variable", "VariableV2", "VarHandleOp"]:
          raise ValueError("defun cannot capture variables created without "
                           "using tf.get_variable. Op: %s" % op)
        c_known_ops.add(op)
        for i in op.inputs:
          if i.op not in c_known_ops:
            c_captured_tensors.add(i)

    backward_outputs = tuple(
        grad for grad in _flatten(in_gradients) if grad is not None)
    output_shapes = tuple(grad.shape for grad in backward_outputs)

    captures = list(sorted(c_captured_tensors, key=lambda x: x.name))
    forward_name = _forward_name(self._func_name)
    self._forward_fdef = _EagerDefinedFunction(
        forward_name, self._graph, self._ops, self._input_placeholders,
        filtered_outputs + captures, self._attrs)
    all_inputs = self._out_grad_placeholders + captures
    # Excluding input ops from the body as we do not intend to execute these
    # operations when the function is executed.
    all_ignored_ops = frozenset(x.op for x in all_inputs)
    # Enforce a deterministic order of operations in the generated graph. This
    # means rerunning the function-defining code will always define the same
    # function, which is useful if we serialize this etc.
    function_def_ops = tuple(x
                             for x in sorted(c_known_ops, key=lambda x: x.name)
                             if x not in all_ignored_ops)
    bname = _backward_name(self._func_name)
    self._backward_function = GraphModeFunction(
        bname, all_inputs, [], self._graph, function_def_ops,
        backward_outputs, in_gradients, output_shapes, attrs=self._attrs)

  def _backprop_call(self, args):
    """Calls the wrapped function and records the result on a tape.

    (Only records results on a tape if the function has outputs)

    Args:
      args: The tensor inputs to the function.
    Returns:
      The call output.
    """
    all_args = args + self._extra_inputs
    ctx = context.context()
    outputs = self._forward_fdef.call(ctx, all_args, self._output_shapes)
    if isinstance(outputs, ops.Operation) or outputs is None:
      return outputs

    # `real_outputs` are the actual outputs of the inference graph function;
    # `side_outputs` are the intermediate Tensors that were added as outputs to
    # the forward graph function so that we can compute its gradient.
    real_outputs = outputs[:self._num_outputs]
    side_outputs = outputs[self._num_outputs:]

    def backward_function(*args):
      return self._backward_function(*(list(args) + side_outputs))  # pylint: disable=not-callable

    tape.record_operation(
        self._forward_fdef.signature.name,
        real_outputs,
        (args + self._extra_inputs),
        backward_function)

    return self._build_call_outputs(real_outputs)

  @property
  def output_shapes(self):
    """The function's output shapes."""
    # TODO(ebrevdo): Should we only keep the output shapes associated
    # with len(self._python_returns) outputs?
    outputs_list = nest.flatten(self._python_func_outputs)
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
    return nest.pack_sequence_as(self._python_func_outputs, outputs_list)

  @property
  def output_dtypes(self):
    return nest.map_structure(
        lambda x: x.dtype if x is not None else None, self._python_func_outputs)

  @property
  def captured_inputs(self):
    return self._extra_inputs

  @property
  def name(self):
    """Returns the name of the function in Eager-compatible format."""
    return self._function_def.name.encode("utf-8")

  def __call__(self, *args):
    """Executes the passed function in eager mode."""
    for v in self._variables:
      if v.trainable:
        tape.watch_variable(v)

    tensor_inputs = [x for x in nest.flatten(args) if isinstance(x, ops.Tensor)]
    if tape.should_record(tensor_inputs) or tape.should_record(
        self._extra_inputs):
      if self._backward_function is None:
        self._construct_backprop_function()
      return self._backprop_call(tensor_inputs)

    ctx = context.context()
    args = tensor_inputs + self._extra_inputs
    outputs = self._function_def.call(ctx, args, self._output_shapes)
    return self._build_call_outputs(outputs)

  def _build_call_outputs(self, result):
    """Maps the fdef output list to actual output structure.

    Args:
      result: Output lists defined by FunctionDef.
    Returns:
      The actual call output.
    """
    if self._python_func_outputs is None:
      return result

    # Use `nest.flatten` instead of `_flatten` in order to preserve any
    # IndexedSlices in `self._python_func_outputs`.
    outputs_list = nest.flatten(self._python_func_outputs)
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
                values=result[j],
                indices=result[j + 1])
            j += 2
        else:
          outputs_list[i] = result[j]
          j += 1
    ret = nest.pack_sequence_as(self._python_func_outputs, outputs_list)
    return ret


def _get_defun_inputs(args):
  """Maps the inputs args to graph inputs."""
  ret = []
  flat_args = nest.flatten(args)
  for a in flat_args:
    if isinstance(a, ops.Tensor):
      ret.append(graph_placeholder(a.dtype, a.shape))
    else:
      ret.append(a)
  return nest.pack_sequence_as(args, ret)


def _deterministic_dict_values(kwds):
  return tuple(kwds[key] for key in sorted(kwds))


def _trace_and_define_function(name, func, compiled, args, kwds):
  """Defines and returns graph-mode version of func."""
  graph_key = ops.get_default_graph()._graph_key  # pylint: disable=protected-access
  captures = {}
  tmp_graph = CapturingGraph(captures)
  # Inherit the graph key, since this is used for matching variables in
  # optimizers.
  tmp_graph._graph_key = graph_key  # pylint: disable=protected-access
  # Copy the graph collections to ensure summaries and other things work. This
  # lets the function access (but not mutate) collections of the containing
  # graph, such as the global step and the summary writer collections.
  curr_graph = ops.get_default_graph()
  for collection in curr_graph.collections:
    tmp_graph.get_collection_ref(collection)[:] = curr_graph.get_collection(
        collection)
  with tmp_graph.as_default(), AutomaticControlDependencies() as a:
    func_args = _get_defun_inputs(args)
    func_kwds = _get_defun_inputs(kwds)

    def convert(x):
      if x is None:
        return None
      x = ops.convert_to_tensor_or_indexed_slices(x)
      x = a.mark_as_return(x)
      return x

    this_tape = tape.push_new_tape()
    try:
      func_outputs = func(*func_args, **func_kwds)
      func_outputs = nest.map_structure(convert, func_outputs)
    finally:
      tape.pop_tape(this_tape)
    variables = this_tape.watched_variables()

    # Returning a closed-over tensor as an output does not trigger a
    # call to convert_to_tensor, so we manually capture all such tensors.
    outputs_list = _flatten(func_outputs)
    func_def_outputs = [
        tmp_graph.capture(x) for x in outputs_list
        if x is not None
    ]

    ids = list(sorted(captures.keys()))
    if ids:
      extra_inputs, extra_placeholders = zip(* [captures[x] for x in ids])
    else:
      extra_inputs = []
      extra_placeholders = []
    output_shapes = tuple(
        x.shape if isinstance(x, ops.Tensor) else None
        for x in func_def_outputs)

  func_kwds_values = _deterministic_dict_values(func_kwds)
  flat_inputs = [
      x for x in nest.flatten(func_args) + nest.flatten(func_kwds_values)
      if isinstance(x, ops.Tensor)
  ]
  all_inputs = flat_inputs + list(extra_placeholders)
  all_ignored_ops = frozenset(x.op for x in all_inputs)
  fname = _inference_name(name)
  operations = tuple(x for x in tmp_graph.get_operations()
                     if x not in all_ignored_ops)
  # Register any other functions defined in the graph
  # TODO(ashankar): Oh lord, forgive me for this lint travesty.
  if context.executing_eagerly():
    for f in tmp_graph._functions.values():  # pylint: disable=protected-access
      # TODO(ashankar): What about the gradient registry?
      _register(f._c_func.func)  # pylint: disable=protected-access

  attrs = {}
  if compiled:
    attrs[_xla_compile_attr] = attr_value_pb2.AttrValue(b=True)

  return GraphModeFunction(
      fname, all_inputs, extra_inputs, tmp_graph, operations, func_def_outputs,
      func_outputs, output_shapes, variables, attrs)


# Defun uses this instead of Tensor as a cache key. Using dtype because
# TensorFlow graphs are not parametric wrt dtypes, and using shapes for
# performance reasons, as much TensorFlow code specializes on known shapes to
# produce slimmer graphs.
_TensorDtype = collections.namedtuple("_TensorDtype", ["dtype", "shape"])
_ZeroDtype = collections.namedtuple("_ZeroDtype", ["dtype", "shape"])


def _cache_key(x):
  """Cache key for tfe functions."""
  if isinstance(x, ops.Tensor):
    return _TensorDtype(x.dtype, x._shape_tuple())  # pylint: disable=protected-access
  if isinstance(x, ops.IndexedSlices):
    if x.dense_shape is not None:
      return tuple([
          _TensorDtype(x.values.dtype, x.values._shape_tuple()),  # pylint: disable=protected-access
          _TensorDtype(x.indices.dtype, x.indices._shape_tuple()),  # pylint: disable=protected-access
          _TensorDtype(x.dense_shape.dtype, x.dense_shape._shape_tuple())  # pylint: disable=protected-access
      ])
    else:
      return tuple([
          _TensorDtype(x.values.dtype, x.values._shape_tuple()),  # pylint: disable=protected-access
          _TensorDtype(x.indices.dtype, x.indices._shape_tuple())  # pylint: disable=protected-access
      ])
  if isinstance(x, np.ndarray):
    return ("array", x.shape, tuple(x.reshape(-1)))
  if isinstance(x, (list, tuple)):
    return tuple([_cache_key(a) for a in x])
  if isinstance(x, dict):
    return tuple(tuple([_cache_key(k), _cache_key(v)]) for k, v in x.items())
  return x


class _PolymorphicFunction(object):
  """Wrapper class for the graph functions defined for a Python function.

  See the documentation for `defun` for more information on the semantics of
  defined functions.

  _PolymorphicFunction class is thread-compatible meaning that minimal
  usage of defuns (defining and calling) is thread-safe, but if users call other
  methods or invoke the base `python_function` themselves, external
  synchronization is necessary.
  """

  def __init__(self, python_function, name, compiled=False):
    """Initializes a polymorphic function.

    Args:
      python_function: the function to be wrapped.
      name: the name given to it.
      compiled: if True, the framework will attempt to compile func with XLA.
    """

    self._python_function = python_function
    self._name = name
    self._compiled = compiled
    self._arguments_to_functions = {}
    self._variables = []

    self._lock = threading.Lock()

  def __get__(self, instance, owner):
    """Makes it possible to defun instance methods."""
    del owner
    # `instance` here is the instance that this `_PolymorphicFunction` was
    # accessed through; e.g., for
    #
    #   class Foo(object):
    #
    #     @function.defun
    #     def bar(self):
    #       ...
    #
    #   foo = Foo()
    #   foo.bar()  # `foo.bar` is a `_PolymorphicFunction` instance
    #
    # then `instance` will be `foo` (and `owner` will be `Foo`).
    return functools.partial(self.__call__, instance)

  def _maybe_define_function(self, *args, **kwds):
    """Gets a function for these inputs, defining it if necessary.

    Args:
      *args: args for the Python function; used to compute the signature
      **kwds: kwds for the Python function; used to compute the signature

    Returns:
      A graph function corresponding to the input signature implied by args and
      kwds, as well as the inputs that the object should be called with.
    """

    # TODO(apassos): Better error messages for non-hashable arguments.
    kwd_values = _deterministic_dict_values(kwds)
    inputs = args + kwd_values
    signature = tuple(_cache_key(x) for x in inputs)
    # The graph, or whether we're executing eagerly, should be a part of the
    # signature so we don't improperly capture tensors such as variables.
    signature += tuple([context.executing_eagerly() or ops.get_default_graph()])

    with self._lock:
      if signature not in self._arguments_to_functions:
        graph_function = _trace_and_define_function(
            self._name, self._python_function, self._compiled, args, kwds)
        self._arguments_to_functions[signature] = graph_function
        self._variables.extend(
            [v for v in graph_function.variables if v not in self._variables])
        return graph_function, inputs
      else:
        return self._arguments_to_functions[signature], inputs

  def __call__(self, *args, **kwds):
    """Calls a graph function specialized for this input signature."""
    graph_function, inputs = self._maybe_define_function(*args, **kwds)
    return graph_function(*inputs)

  def call_python_function(self, *args, **kwargs):
    """Directly calls the wrapped python function."""
    return self._python_function(*args, **kwargs)

  @property
  def variables(self):
    """Returns a list of variables used in any of the defined functions."""
    return self._variables


# TODO(akshayka): Remove the `compiled` flag and create a separate
# API for xla compilation (`defun` is already complicated enough
# as it is, and the keyword argument makes 'compiled' an overloaded concept)
def defun(func=None, compiled=False):
  """Compiles a Python function into a callable TensorFlow graph.

  `defun` (short for "define function") trace-compiles a Python function
  composed of TensorFlow operations into a callable that executes a @{tf.Graph}
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
  be hashable Python objects or lists thereof. Additionally, it must return zero
  or more @{tf.Tensor} objects.

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

    def call(self, inputs, training=True):
      x = self.dense2(self.dense1(inputs))
      if training:
        return tf.nn.dropout(x, self.keep_probability)
      else:
        return x

  model = MyModel()
  model.call = tf.contrib.eager.defun(model.call)
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
  `f` be a Python function that returns zero or more @{tf.Tensor} objects and
  let `F = defun(f)`. `F` builds a graph for each unique input signature it
  sees, Python control flow is baked into graphs, and operations related to
  variable initialization are automatically lifted out of the graphs that `F`
  generates and placed in the eager context if executing eagerly or into an
  outer graph otherwise.

  _Tracing and Input Signatures_.
  The signature of inputs supplied to `F` is defined to be a tuple of the shapes
  and dtypes of Tensor-typed arguments and the values of non-Tensor arguments,
  where "arguments" includes both args and kwargs. Every time `F` is invoked,
  the signature of its inputs are inferred. The first time `F(*args, **kwargs)`
  is invoked with a particular signature, `f(*args, **kwargs)` is executed and
  all the TensorFlow operations that `f` executes, along with the Tensors that
  flow between them, are recorded in a TensorFlow graph. `F` caches this graph
  and binds it to the inputs' signature; every subsequent invocation of `F` with
  inputs conforming to this signature will immediately retrieve the cached graph
  and pass it to the TensorFlow runtime for execution.

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
  that run for many iterations, consider replacing them with @{tf.while_loop}
  operations.

  When constructing graphs, @{tf.Tensor} objects cannot be used as Python
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
  TensorFlow APIs, like @{tf.keras.layers.Layer} objects, create variables the
  first time they are called and reuse them thereafter. Automatic variable
  lifting makes it possible to compile these APIs without extra effort, at the
  cost of introducing a discrepancy between the semantics of executing Python
  functions and their corresponding compiled functions. For example:

  ```python
  import tensorflow as tf

  tf.enable_eager_execution()

  def fn():
    x = tf.contrib.eager.Variable(0.0)
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
  Python function constructs `tf.contrib.eager.Variable` objects, then each
  graph constructed for that Python function will reference a unique set of
  variables. To circumvent this problem, we recommend against compiling Python
  functions that create `tf.contrib.eager.Variable` objects. Instead, Python
  functions should either lexically close over `tf.contrib.eager.Variable`
  objects or accept them as arguments, preferably encapsulated in an
  object-oriented container. If you must create variables inside your Python
  function and you want each graph generated for it to reference the same set of
  variables, add logic to your Python function that ensures that variables are
  only created the first time it is called and are reused for every subsequent
  invocation; note that this is precisely what @{tf.keras.layers.Layer} objects
  do, so we recommend using them to represent variable-bearing computations
  whenever possible.

  Args:
    func: function to be compiled. If `func` is None, returns a
      decorator that can be invoked with a single argument - `func`. The
      end result is equivalent to providing all the arguments up front.
      In other words, defun(compiled=True)(func) is equivalent to
      defun(func, compiled=True). The former allows the following use case:
        @tf.contrib.eager.defun(compiled=True)
        def foo(...):
          ...

    compiled: If True, an attempt to compile `func` with XLA will be made.
      If it fails, function will be run normally. Experimental.  Currently
      supported only for execution on TPUs. For the vast majority of users,
      this argument should be False.

  Returns:
     If `func` is not None, returns a callable that will execute the compiled
     function (and return zero or more `tf.Tensor` objects).
     If `func` is None, returns a decorator that, when invoked with a single
     `func` argument, returns a callable equivalent to the case above.
  """
  # TODO(apassos): deal with captured global state. Deal with control flow.
  def decorated(function):
    try:
      name = function.__name__
    except AttributeError:
      name = "function"
    return tf_decorator.make_decorator(
        function, _PolymorphicFunction(function, name, compiled=compiled))

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


def make_defun_op(func, *args, **kwds):
  """Compile func into graph_mode, assuming func arguments are *args, **kwargs.

  `make_defun_op` converts a function that constructs a TensorFlow graph into
  a function object and attaches it to the graph.  The resulting function
  object can be queried for its properties, and called directly with different
  inputs to execute.

  More details on use cases and limitations are available in the
  documentation for `defun`.

  Example:
  ```python
  def f(x, y):
    return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)

  def g(x, y):
    return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)

  z = tf.constant([[0.0, 0.0]])
  g_op = make_defun_op(g, z, z)

  assert g_op.output_shapes == tf.TensorShape([])
  assert g_op.output_types == tf.float32

  x = tf.constant([[2.0, 3.0]])
  y = tf.constant([[3.0, -2.0]])

  # The plain function and defun-compiled function should return the same value.
  assert f(x, y).numpy() == g_op(x, y).numpy()
  ```

  Args:
    func: function to be compiled.
    *args: List arguments to pass to `func` when attaching to the graph.
    **kwds: Keyword arguments to pass to `func` when attaching to the graph.

  Returns:
     A wrapper object which can be queried for its output properties,
     and which can be called directly the way a `@defun` wrapped function
     can.
  """
  return _trace_and_define_function(func.__name__, func, False, args, kwds)


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
