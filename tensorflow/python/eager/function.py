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
import contextlib
import threading

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_to_function_def
from tensorflow.python.framework import ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator

# Thread-local storage for tfe Tensors which are referenced while evaluating a
# graph-mode function.
_scoped_captures = threading.local()
# _scoped_captures.tensors is either None or a map from Tensor id to a pair
# of a tfe tensor and its corresponding placeholder to pass as a function
# argument. The value should be None unless we're in function definition
# context.
_scoped_captures.tensors = None


def make_function_def(graph, operations, inputs, outputs):
  """Makes function def where accesses to resources are serialized."""
  last_op_using_resource_tensor = {}

  # TODO(apassos) probably control flow has to be handled delicately here as in
  # if a resource is accessed inside a control flow context we need the control
  # dependency to point to something outside the context which is guaranteed to
  # happen after the access.
  #
  # TODO(apassos) this should do some form of alias analysis as ops which
  # forward the resources such as Identity and Switch can cause serialization to
  # fail.
  for op in operations:
    for t in op.inputs:
      if t.dtype == dtypes.resource:
        if t.name in last_op_using_resource_tensor:
          op._add_control_input(last_op_using_resource_tensor[t.name])  # pylint: disable=protected-access
        last_op_using_resource_tensor[t.name] = op
  return graph_to_function_def.graph_to_function_def(
      graph, operations, inputs, outputs)


@contextlib.contextmanager
def capture_tensors(captures):
  old = _scoped_captures.__dict__.get("tensors", None)
  try:
    _scoped_captures.tensors = captures
    yield
  finally:
    _scoped_captures.tensors = old


def capture_value(tensor_map, value, dtype, name):
  """Capture a value from outside the function, to pass in as an extra arg."""
  captured_value = tensor_map.get(ops.tensor_id(value), None)
  if captured_value is None:
    captured_value = graph_placeholder(
        dtype=dtype or value.dtype, shape=value.shape, name=name)
    if captured_value.dtype == dtypes.resource:
      captured_value._handle_data = value._handle_data  # pylint: disable=protected-access
    tensor_map[ops.tensor_id(value)] = (value, captured_value)
  else:
    captured_value = captured_value[1]
  tape.record_operation("captured_value", [captured_value], [value],
                        lambda x: [x])
  return captured_value


def _convert_to_graph_tensor(value, dtype=None, name=None, as_ref=False):
  """Captures a Tensor while building a graph mode function.

  Arguments:
    value: A Tensor object.
    dtype: The datatype of the value produced by the node in the graph.
    name:  Name of the node in the graph.
    as_ref: Ignored (required by register_tensor_conversion_function).

  Returns:
    Returns a constant (the current value of the tensor) if capturing
    is not enabled. A placeholder which will have the value of the
    tensor at runtime otherwise.
  """
  if context.in_eager_mode():
    return value
  _ = as_ref
  tensor_map = _scoped_captures.tensors
  if tensor_map is None:
    # Capturing is not enabled.
    return constant_op.constant(value.numpy())
  return capture_value(tensor_map, value, dtype, name)


class CapturingGraph(ops.Graph):

  def __init__(self, captures):
    super(CapturingGraph, self).__init__()
    self._building_function = True
    self.captures = captures

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
    for i, inp in enumerate(inputs):
      if inp.graph is not self:
        inputs[i] = capture_value(self.captures, inp, inp.dtype, inp.op.name)
    return super(CapturingGraph, self).create_op(
        op_type, inputs, dtypes, input_types, name, attrs, op_def,
        compute_shapes, compute_device)


# TODO(apassos): it'd be really nice if we could scope this registration.
# Note that we register this at a higher priority than ops.Tensor since we want
# to handle subclass specific conversion before a superclass conversion.
ops.register_tensor_conversion_function(
    ops.EagerTensor, _convert_to_graph_tensor, priority=-1)


class _CapturingContext(object):
  """Tracks references to Tensors outside this context while it is active."""

  def __init__(self):
    # known_ops are ops which are created while this context is active
    self.known_ops = set()

    # captured_tensors are all tensors referenced to by ops in this context but
    # not produced in it
    self.captured_tensors = set()

  def AddOp(self, op):  # pylint: disable=invalid-name
    if op.type in ["Variable", "VariableV2", "VarHandleOp"]:
      raise ValueError("tfe.defun cannot capture variables created without "
                       "using tf.get_variable. Op: %s" % op)
    self.known_ops.add(op)
    for i in op.inputs:
      if i.op not in self.known_ops:
        self.captured_tensors.add(i)

  def __enter__(self):
    self._g = ops.get_default_graph()
    self._old = self._g._get_control_flow_context()  # pylint: disable=protected-access
    self._g._set_control_flow_context(self)  # pylint: disable=protected-access

  def __exit__(self, _, __, ___):  # pylint: disable=invalid-name
    self._g._set_control_flow_context(self._old)  # pylint: disable=protected-access


def _forward_name(n):
  """The name of a generated forward defun named n."""
  return "__forward_%s_%s" % (n, ops.uid())


def _backward_name(n):
  """The name of a generated backward defun named n."""
  return "__backward_%s_%s" % (n, ops.uid())


def _inference_name(n):
  """The name of a forward-but-no-gradient defun named n."""
  return "__inference_%s_%s" % (n, ops.uid())


class _DefinedFunction(object):
  """Mocks the interface of tf _DefinedFunction."""

  def __init__(self, fdef):
    self.definition = fdef
    self.name = fdef.signature.name
    self.grad_func_name = None
    self.python_grad_func = None


def _map_sequence_obj_to_idx(sequence):
  """Maps objs in the sequence from id(obj) to sequence index."""
  return {id(x): i for i, x in enumerate(sequence)}


class _GraphModeFunction(object):
  """Callable object representing a graph-mode function.

  Args:
    input_placeholders: list of placeholder values to feed when calling
      the wrapped function.
    extra_inputs: Tensor inputs this function definition closed over which
      are passed as arguments. Need to track so gradients are supported
      correctly.
    fdef: the function definition we want to call.
    graph: the graph from which the fdef operations were pulled. Used as
      a context when computing gradients.
    operations: the subset of operations in the graph used in the function
      definition.
    func_outputs: the python outputs of the graph-mode function, with
      tensorflow.Tensor objects to be replaced by tfe values when called.
    func_outputs_to_fdef_outputs: Maps id(obj) in func_outputs to index of
      fdef's outputs. It allows mapping fdef output tensors to nested
      func_outputs structure.
    output_shapes: List of shapes of all tensors which are output by the
      internal function.
  """

  def __init__(self, input_placeholders, extra_inputs, fdef, graph, operations,
               func_outputs, func_outputs_to_fdef_outputs, output_shapes):
    assert len(input_placeholders) == len(fdef.signature.input_arg), "%s %s" % (
        len(input_placeholders), len(fdef.signature.input_arg))
    self._input_placeholders = input_placeholders
    self._extra_inputs = list(extra_inputs)
    self._graph = graph
    self._has_backprop = False
    self._func_name = fdef.signature.name
    self._fdef = _DefinedFunction(fdef)
    self._num_outputs = len(fdef.signature.output_arg)
    self._ops = operations
    self._func_outputs = func_outputs
    self._returns = [func_outputs] if isinstance(
        func_outputs, (ops.Tensor, type(None))) else list(func_outputs)
    self._returns_to_fedf_outputs = func_outputs_to_fdef_outputs
    self._output_shapes = output_shapes

  def _compute_backprop(self):
    """Computes the backprop function object for this function."""
    self._has_backprop = True
    with self._graph.as_default(), context.graph_mode():
      c = _CapturingContext()
      with c:
        filtered_outputs = [
            x for x in self._returns if x is not None
        ]
        self._out_grad_placeholders = [
            graph_placeholder(x.dtype, x.shape) for x in filtered_outputs
        ]
        in_gradients = gradients_impl.gradients(
            filtered_outputs,
            self._input_placeholders,
            grad_ys=self._out_grad_placeholders)
        shapes = [x.shape for x in in_gradients if x is not None]
    captures = list(sorted(c.captured_tensors, key=lambda x: x.name))
    forward_function_def = make_function_def(
        self._graph, self._ops, self._input_placeholders,
        filtered_outputs + captures)
    self._forward_fdef = _DefinedFunction(forward_function_def)
    _register_with_name(_forward_name(self._func_name), forward_function_def)
    backward_outputs = [x for x in in_gradients if x is not None]
    all_inputs = self._out_grad_placeholders + captures
    backward_function_def = make_function_def(
        self._graph, [x.op for x in self._out_grad_placeholders
                     ] + list(sorted(c.known_ops, key=lambda x: x.name)),
        all_inputs, backward_outputs)
    _register_with_name(_backward_name(self._func_name), backward_function_def)
    self._backward_function = _GraphModeFunction(
        all_inputs, [], backward_function_def, self._graph, c.known_ops,
        in_gradients, _map_sequence_obj_to_idx(backward_outputs), shapes)

  def _backprop_call(self, args):
    """Calls the wrapped function and records the result on a tape."""
    all_args = args + self._extra_inputs
    signature = self._forward_fdef.definition.signature
    ctx = context.context()
    if ctx.in_graph_mode():
      g = ops.get_default_graph()
      g._add_function(self._forward_fdef)  # pylint: disable=protected-access
      def make_tensor(x):
        if isinstance(x, ops.Tensor):
          return x
        return ops.internal_convert_to_tensor(x, ctx=ctx)
      op = g.create_op(
          signature.name, [make_tensor(x) for x in all_args],
          [dtypes.DType(x.type) for x in signature.output_arg],
          op_def=signature,
          name="FunctionCall",
          compute_shapes=False)
      outputs = op.outputs
      outputs = [outputs] if isinstance(
          outputs, (ops.Tensor, type(None))) else list(outputs)
      for i, s in enumerate(self._output_shapes):
        outputs[i].set_shape(s)
    else:
      outputs = execute.execute(
          str(signature.name),
          num_outputs=len(signature.output_arg),
          inputs=all_args,
          attrs=None,
          ctx=ctx)
    real_outputs = outputs[:len(self._returns)]
    side_outputs = outputs[len(self._returns):]

    def backward_function(*args):
      return self._backward_function(*(list(args) + side_outputs))

    tape.record_operation(
        signature.name,
        real_outputs,
        (args + self._extra_inputs),
        backward_function)

    return self._build_call_outputs(real_outputs)

  def __call__(self, *args):
    """Executes the passed function in eager mode."""
    tensor_inputs = [
        x for x in nest.flatten(args)
        if isinstance(x, ops.Tensor)
    ]
    if tape.should_record(tensor_inputs) or tape.should_record(
        self._extra_inputs):
      if not self._has_backprop:
        self._compute_backprop()
      return self._backprop_call(tensor_inputs)

    ctx = context.context()
    if ctx.in_graph_mode():
      g = ops.get_default_graph()
      if self._fdef.name not in g._functions:  # pylint: disable=protected-access
        g._add_function(self._fdef)  # pylint: disable=protected-access
      for f in self._graph._functions.values():  # pylint: disable=protected-access
        if f.name not in g._functions:  # pylint: disable=protected-access
          g._add_function(f)  # pylint: disable=protected-access
      signature = self._fdef.definition.signature
      args = list(tensor_inputs) + self._extra_inputs
      op = g.create_op(
          signature.name, [ops.convert_to_tensor(x) for x in args],
          [dtypes.DType(x.type) for x in signature.output_arg],
          op_def=signature,
          name="FunctionCall",
          compute_shapes=False)
      result = op.outputs
      if not result:
        return op
      for i, s in enumerate(self._output_shapes):
        result[i].set_shape(s)
    else:
      result = execute.execute(
          str(self._func_name),
          num_outputs=self._num_outputs,
          inputs=tensor_inputs + self._extra_inputs,
          attrs=None,
          ctx=ctx)

    return self._build_call_outputs(result)

  def _build_call_outputs(self, result):
    """Maps the fdef output list to actual output structure.

    Args:
      result: Output lists defined by FunctionDef.
    Returns:
      The actual call output.
    """
    if self._func_outputs is None:
      return None
    outputs_list = nest.flatten(self._func_outputs)
    j = 0
    for i, o in enumerate(outputs_list):
      if o is not None:
        outputs_list[i] = result[j]
        j += 1
    return nest.pack_sequence_as(self._func_outputs, outputs_list)


def _get_defun_inputs(args):
  """Maps the inputs args to graph inputs."""
  ret = []
  for a in args:
    if isinstance(a, ops.Tensor):
      ret.append(graph_placeholder(a.dtype, a.shape))
    elif type(a) in (tuple, list):
      ret.append(_get_defun_inputs(a))
    else:
      ret.append(a)
  return tuple(ret) if type(args) is tuple else ret


def _defun_internal(name, func, args, kwds):
  """Defines and returns graph-mode version of func."""
  with context.graph_mode():
    captures = {}
    tmp_graph = CapturingGraph(captures)
    # Copy the graph collections to ensure summaries and other things work. This
    # lets the function access (but not mutate) collections of the containing
    # graph, such as the global step and the summary writer collections.
    curr_graph = ops.get_default_graph()
    for collection in curr_graph.collections:
      tmp_graph.get_collection_ref(collection)[:] = curr_graph.get_collection(
          collection)
    with tmp_graph.as_default():
      func_inputs = _get_defun_inputs(args)

      with capture_tensors(captures):
        func_outputs = func(*func_inputs, **kwds)
      ids = list(sorted(captures.keys()))
      if ids:
        extra_inputs, extra_placeholders = zip(* [captures[x] for x in ids])
      else:
        extra_inputs = []
        extra_placeholders = []
      outputs_list = nest.flatten(func_outputs)
      output_shapes = [x.shape for x in outputs_list if x is not None]

  flat_inputs = [
      x for x in nest.flatten(func_inputs) if isinstance(x, ops.Tensor)
  ]
  all_inputs = flat_inputs + list(extra_placeholders)

  func_def_outputs = [x for x in outputs_list if x is not None]
  inference_function_def = make_function_def(
      tmp_graph, tmp_graph.get_operations(), all_inputs, func_def_outputs)
  # Register any other functions defined in the graph
  # TODO(ashankar): Oh lord, forgive me for this lint travesty.
  for f in tmp_graph._functions.values():  # pylint: disable=protected-access
    # TODO(ashankar): What about the gradient registry?
    _register_with_name(f.name, f.definition)
  _register_with_name(_inference_name(name), inference_function_def)

  return _GraphModeFunction(
      all_inputs, extra_inputs, inference_function_def, tmp_graph,
      tmp_graph.get_operations(), func_outputs,
      _map_sequence_obj_to_idx(func_def_outputs), output_shapes)


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
  if isinstance(x, np.ndarray):
    return ("array", x.shape, tuple(x.reshape(-1)))
  if type(x) in (list, tuple):
    return tuple([_cache_key(a) for a in x])
  return x


def _register_with_name(name, fdef):
  """Registers the function `fdef` with the name `name`."""
  fdef.signature.name = name
  context.context().add_function_def(fdef)


# TODO(apassos): better error messages for non-hashable arguments.
def named_defun(func, name):
  """Defines a function with a given name.

  See the documentation for `defun` for more information on the semantics of the
  function.

  Args:
    func: the function to be wrapped.
    name: the name given to it.

  Returns:
    the wrapped function.
  """
  arguments_to_functions = {}

  def decorated(*args, **kwds):
    """Decorated version of func."""
    # Macroexpand on non-Tensor arguments
    cache_key = tuple(_cache_key(x) for x in args)
    assert all(not isinstance(x, ops.EagerTensor) for x in kwds.values())
    cache_key = (cache_key, tuple(kwds.items()))

    if cache_key not in arguments_to_functions:
      arguments_to_functions[cache_key] = _defun_internal(
          name, func, args, kwds)
    return arguments_to_functions[cache_key](*args)

  return decorated


def defun(func):
  """Decorator to compile func into graph_mode.

  `defun` converts a function that constructs a TensorFlow graph into a function
  that executes the graph. TensorFlow graphs typically execute faster and with a
  lower memory-footprint than executing each of the operations that make up the
  function individually as the TensorFlow runtime can optimize the graph and
  execute sub-operations in parallel.

  func must be a Python function that constructs a TensorFlow graph,
  typically using functions in the tensorflow module.

  Arguments to func can be either Tensor objects or Python
  objects. Non-Tensor python objects are treated as constants, and new function
  definitions are created internally based on their values.

  func must return a tf.Tensor (NOT a Tensor) or a list of tf.Tensor (NOT a
  Tensor).

  Control flow constructs (e.g., `if`, `while`) are not yet compatible with
  `defun`.

  Example:
  ```python
  def f(x, y):
    return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)

  @tfe.defun
  def g(x, y):
    return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)

  x = tf.constant([[2.0, 3.0]])
  y = tf.constant([[3.0, -2.0]])
  # The plain function and defun-compiled function should return the same value.
  assert f(x, y).numpy() == g(x, y).numpy()

  # After the first invocation, the defun-compiled (graph) function runs faster
  # than the plain function because the defun-compiled function does not involve
  # Python interpreter overhead during the execution.
  %time print(f(x, y))
  %time print(g(x, y))
  ```

  Args:
    func: function to be compiled.

  Returns:
     A callable that will execute the compiled function (and return zero
     or more Tensor objects).
  """
  # TODO(apassos): deal with captured global state. Deal with control flow.
  return tf_decorator.make_decorator(func, named_defun(func, func.__name__))
