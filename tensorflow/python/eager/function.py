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

from tensorflow.core.framework import function_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.util import compat
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
    if captured_value.dtype == dtypes_module.resource:
      handle_data = value._handle_data  # pylint: disable=protected-access
      captured_value._handle_data = handle_data  # pylint: disable=protected-access
      if handle_data is not None and handle_data.is_set:
        # Ensure that shapes and dtypes are propagated.
        shapes, types = zip(*[(pair.shape, pair.dtype)
                              for pair in handle_data.shape_and_type])
        ranks = [len(s.dim) if not s.unknown_rank else -1 for s in shapes]
        shapes = [[d.size for d in s.dim]
                  if not s.unknown_rank else None for s in shapes]
        with errors.raise_exception_on_not_ok_status() as status:
          pywrap_tensorflow.TF_GraphSetOutputHandleShapesAndTypes_wrapper(
              captured_value._op._graph._c_graph,  # pylint: disable=protected-access
              captured_value._as_tf_output(),  # pylint: disable=protected-access
              shapes,
              ranks,
              types,
              status)

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
    name:  str, Name of the node in the graph.
    as_ref: Ignored (required by register_tensor_conversion_function).

  Returns:
    Returns a constant (the current value of the tensor) if capturing
    is not enabled. A placeholder which will have the value of the
    tensor at runtime otherwise.
  """
  del as_ref  # Unused.

  if context.in_eager_mode():
    return value

  default_graph = ops.get_default_graph()
  if not default_graph.building_function:
    return value

  tensor_map = _scoped_captures.tensors
  if tensor_map is None:
    # Capturing is not enabled.
    return constant_op.constant(value.numpy())
  if type(value) == ops.Tensor and value.graph is default_graph:
    # The tensor has already been converted and captured. The type check
    # is intentional: we are checking that value is a Tensor and not an
    # EagerTensor.
    return value
  return capture_value(tensor_map, value, dtype, name)


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
    # TODO(apassos) probably control flow has to be handled delicately here as
    # in if a resource is accessed inside a control flow context we need the
    # control dependency to point to something outside the context which is
    # guaranteed to happen after the access.
    #
    # TODO(apassos) this should do some form of alias analysis as ops which
    # forward the resources such as Identity and Switch can cause serialization
    # to fail.
    resource_inputs = set()
    control_inputs = set()
    for i, inp in enumerate(inputs):
      if inp.graph is not self:
        inputs[i] = capture_value(self.captures, inp, inp.dtype, inp.op.name)
      inp = inputs[i]
      if inp.dtype == dtypes_module.resource:
        if inp.name in self._last_op_using_resource_tensor:
          control_inputs.add(self._last_op_using_resource_tensor[inp.name])
        resource_inputs.add(inp.name)
    with self.control_dependencies(list(control_inputs)):
      op = super(CapturingGraph, self).create_op(
          op_type, inputs, dtypes, input_types, name, attrs, op_def,
          compute_shapes, compute_device)
    for name in resource_inputs:
      self._last_op_using_resource_tensor[name] = op
    return op


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


# TODO(apassos) get rid of this by splitting framework.function._DefinedFunction
# so it doesn't have the definition-generating logic and is just a container for
# an already-defined function.
class _EagerDefinedFunction(object):
  """Function object with the interface of tf _DefinedFunction."""

  def __init__(self, name, graph, operations, inputs, outputs):
    """Initializes an eager defined function.

    Args:
      name: str, the name for the created function.
      graph: Graph, the graph containing the operations in the function
      operations: list of Operation; the subset of operations in the graph
        which will be in the function
      inputs: the tensors in the graph to be used as inputs to the function
      outputs: the tensors in the graph which will be outputs to the function
    """
    with errors.raise_exception_on_not_ok_status() as status:
      fn = pywrap_tensorflow.TF_GraphToFunction_wrapper(
          graph._c_graph,  # pylint: disable=protected-access
          compat.as_str(name),
          False,
          [o._c_op for o in operations],  # pylint: disable=protected-access
          [t._as_tf_output() for t in inputs],  # pylint: disable=protected-access
          [t._as_tf_output() for t in outputs],  # pylint: disable=protected-access
          [],
          None,
          compat.as_str(""),
          status)
    # TODO(apassos) avoid creating a FunctionDef (specially to grab the
    # signature, but also in general it's nice not to depend on it.
    with c_api_util.tf_buffer() as buffer_:
      with errors.raise_exception_on_not_ok_status() as status:
        pywrap_tensorflow.TF_FunctionToFunctionDef(fn, buffer_, status)
      proto_data = pywrap_tensorflow.TF_GetBuffer(buffer_)
    function_def = function_pb2.FunctionDef()
    function_def.ParseFromString(compat.as_bytes(proto_data))
    if context.in_eager_mode():
      _register(fn)
    self.definition = function_def
    self.name = function_def.signature.name
    self.signature = function_def.signature
    self.grad_func_name = None
    self.python_grad_func = None
    self._c_func = fn
    self._grad_func = None


def _map_sequence_obj_to_idx(sequence):
  """Maps objs in the sequence from id(obj) to sequence index."""
  return {id(x): i for i, x in enumerate(sequence)}


class GraphModeFunction(object):
  """Callable object representing a graph-mode function.

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
    func_outputs: a possibly nested python object which will be returned by
      this function. The Tensors in this structure will be replaced by their
      corresponding values in outputs.
    output_shapes: List of shapes of all tensors in outputs
    variables: (optional) List of variables to watch during function execution.
  """

  def __init__(self,
               name,
               input_placeholders,
               extra_inputs,
               graph,
               operations,
               outputs,
               func_outputs,
               output_shapes,
               variables=None):
    defined_function = _EagerDefinedFunction(
        name, graph, operations, input_placeholders, outputs)
    if len(input_placeholders) != len(defined_function.signature.input_arg):
      raise ValueError("Internal error: invalid lengths. %s %s" % (
          len(input_placeholders), len(defined_function.signature.input_arg)))
    self._input_placeholders = input_placeholders
    self._extra_inputs = list(extra_inputs)
    self._graph = graph
    self._has_backprop = False
    self._func_name = name
    self._function_def = defined_function
    self._num_outputs = len(defined_function.signature.output_arg)
    self._ops = operations
    self._func_outputs = func_outputs
    self._returns = [func_outputs] if isinstance(
        func_outputs, (ops.Tensor, type(None))) else list(func_outputs)
    self._output_shapes = output_shapes
    self._variables = variables if variables is not None else []

  @property
  def variables(self):
    return self._variables

  def _compute_backprop(self):
    """Computes the backprop function object for this function."""
    self._has_backprop = True
    with self._graph.as_default(), context.graph_mode():
      c = _CapturingContext()
      with c:
        filtered_outputs = [x for x in self._returns if x is not None]
        self._out_grad_placeholders = [
            graph_placeholder(x.dtype, x.shape) for x in filtered_outputs]
        in_gradients = gradients_impl.gradients(
            filtered_outputs,
            self._input_placeholders,
            grad_ys=self._out_grad_placeholders)
        shapes = tuple(x.shape for x in in_gradients if x is not None)
    captures = list(sorted(c.captured_tensors, key=lambda x: x.name))
    forward_name = _forward_name(self._func_name)
    self._forward_fdef = _EagerDefinedFunction(
        forward_name, self._graph, self._ops, self._input_placeholders,
        filtered_outputs + captures)
    backward_outputs = tuple(x for x in in_gradients if x is not None)
    all_inputs = self._out_grad_placeholders + captures
    # Excluding input ops from the body as we do not intend to execute these
    # operations when the function is executed.
    all_ignored_ops = frozenset(x.op for x in all_inputs)
    # Enforce a deterministic order of operations in the generated graph. This
    # means rerunning the function-defining code will always define the same
    # function, which is useful if we serialize this etc.
    function_def_ops = tuple(x
                             for x in sorted(c.known_ops, key=lambda x: x.name)
                             if x not in all_ignored_ops)
    bname = _backward_name(self._func_name)
    self._backward_function = GraphModeFunction(
        bname, all_inputs, [], self._graph, function_def_ops,
        backward_outputs, in_gradients, shapes)

  def _backprop_call(self, args):
    """Calls the wrapped function and records the result on a tape."""
    all_args = args + self._extra_inputs
    signature = self._forward_fdef.signature
    ctx = context.context()
    if ctx.in_graph_mode():
      g = ops.get_default_graph()
      g._add_function(self._forward_fdef)  # pylint: disable=protected-access
      op = g.create_op(
          signature.name,
          [ops.internal_convert_to_tensor(x, ctx=ctx) for x in all_args],
          tuple(dtypes_module.DType(x.type) for x in signature.output_arg),
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
      return self._backward_function(*(list(args) + side_outputs))  # pylint: disable=not-callable

    tape.record_operation(
        signature.name,
        real_outputs,
        (args + self._extra_inputs),
        backward_function)

    return self._build_call_outputs(real_outputs)

  @property
  def output_shapes(self):
    # TODO(ebrevdo): Should we only keep the output shapes associated
    # with len(self._returns) outputs?
    return nest.pack_sequence_as(self._func_outputs, self._output_shapes)

  @property
  def output_dtypes(self):
    return nest.map_structure(
        lambda x: x.dtype if x is not None else None, self._func_outputs)

  @property
  def captured_inputs(self):
    return self._extra_inputs

  @property
  def name(self):
    """Returns the name of the function in Eager-compatible format."""
    return self._function_def.name.encode("utf-8")

  def add_to_graph(self, g):
    if self._function_def.name not in g._functions:  # pylint: disable=protected-access
      g._add_function(self._function_def)  # pylint: disable=protected-access
    for f in self._graph._functions.values():  # pylint: disable=protected-access
      if f.name not in g._functions:  # pylint: disable=protected-access
        g._add_function(f)  # pylint: disable=protected-access

  def __call__(self, *args):
    """Executes the passed function in eager mode."""
    for v in self._variables:
      if v._trainable:  # pylint: disable=protected-access
        tape.watch_variable(v)

    tensor_inputs = [x for x in nest.flatten(args)
                     if isinstance(x, ops.Tensor)]
    if tape.should_record(tensor_inputs) or tape.should_record(
        self._extra_inputs):
      if not self._has_backprop:
        self._compute_backprop()
      return self._backprop_call(tensor_inputs)

    ctx = context.context()
    if ctx.in_graph_mode():
      g = ops.get_default_graph()
      self.add_to_graph(g)
      signature = self._function_def.definition.signature
      args = list(tensor_inputs) + self._extra_inputs
      op = g.create_op(
          signature.name,
          [ops.internal_convert_to_tensor(x, ctx=ctx) for x in args],
          tuple(dtypes_module.DType(x.type) for x in signature.output_arg),
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
  flat_args = nest.flatten(args)
  for a in flat_args:
    if isinstance(a, ops.Tensor):
      ret.append(graph_placeholder(a.dtype, a.shape))
    else:
      ret.append(a)
  return nest.pack_sequence_as(args, ret)


def _defun_internal(name, func, args, kwds):
  """Defines and returns graph-mode version of func."""
  container_prefix = ops.get_default_graph()._container_prefix  # pylint: disable=protected-access
  with context.graph_mode():
    captures = {}
    tmp_graph = CapturingGraph(captures)
    # Inherit the container prefix, since this is used for error checking when
    # isolating eager execution (the container prefix at creation must match the
    # container prefix when used, and variables accessed in the defun will be
    # used in the outside context).
    tmp_graph._container_prefix = container_prefix  # pylint: disable=protected-access
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
        this_tape = tape.push_new_tape()
        try:
          func_outputs = func(*func_inputs, **kwds)
        finally:
          tape.pop_tape(this_tape)
        variables = this_tape.watched_variables()

        # Returning a closed-over tensor as an output does not trigger a
        # call to convert_to_tensor, so we manually capture all such tensors.
        outputs_list = nest.flatten(func_outputs)
        func_def_outputs = [
            _convert_to_graph_tensor(x) for x in outputs_list if x is not None
        ]

      ids = list(sorted(captures.keys()))
      if ids:
        extra_inputs, extra_placeholders = zip(* [captures[x] for x in ids])
      else:
        extra_inputs = []
        extra_placeholders = []
      output_shapes = tuple(
          x.shape if isinstance(x, ops.Tensor) else None
          for x in outputs_list)

  flat_inputs = [x for x in nest.flatten(func_inputs)
                 if isinstance(x, ops.Tensor)]
  all_inputs = flat_inputs + list(extra_placeholders)
  all_ignored_ops = frozenset(x.op for x in all_inputs)
  fname = _inference_name(name)
  operations = tuple(x for x in tmp_graph.get_operations()
                     if x not in all_ignored_ops)
  # Register any other functions defined in the graph
  # TODO(ashankar): Oh lord, forgive me for this lint travesty.
  if context.in_eager_mode():
    for f in tmp_graph._functions.values():  # pylint: disable=protected-access
      # TODO(ashankar): What about the gradient registry?
      _register(f._c_func)  # pylint: disable=protected-access
  return GraphModeFunction(
      fname, all_inputs, extra_inputs, tmp_graph, operations, func_def_outputs,
      func_outputs, output_shapes, variables)


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
  if isinstance(x, (list, tuple)):
    return tuple([_cache_key(a) for a in x])
  if isinstance(x, dict):
    return tuple(tuple([_cache_key(k), _cache_key(v)]) for k, v in x.items())
  return x


def _register(fn):
  """Registers the function `fn`."""
  context.context().add_function(fn)


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
    if any(isinstance(x, ops.EagerTensor) for x in kwds.values()):
      raise ValueError("Tensor keyword arguments are not supported.")
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

  Raises:
    ValueError: if any of the keyword arguments to `func` are `EagerTensor`
      objects (not yet supported).
  """
  name = func.__name__
  if any(isinstance(x, ops.EagerTensor) for x in kwds.values()):
    raise ValueError("Tensor keyword arguments are not supported.")
  return _defun_internal(name, func, args, kwds)
