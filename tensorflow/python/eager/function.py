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

import collections
import functools
import itertools
import pprint
import threading
import types as types_lib
from typing import List
import weakref

import numpy as np
import six
from six.moves import map

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import function_cache
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import tape
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import memory
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

# Loaded lazily due to a circular dependency (roughly
# tf.function->autograph->->dataset->tf.function).
# TODO(b/133251390): Use a regular import.
ag_ctx = lazy_loader.LazyLoader(
    "ag_ctx", globals(),
    "tensorflow.python.autograph.core.ag_ctx")
np_arrays = lazy_loader.LazyLoader(
    "np_arrays", globals(),
    "tensorflow.python.ops.numpy_ops.np_arrays")


FORWARD_FUNCTION_ATTRIBUTE_NAME = "forward_function_name"
BACKWARD_FUNCTION_ATTRIBUTE_NAME = "backward_function_name"
IMPLEMENTS_ATTRIBUTE_NAME = "_implements"
SHARED_RENDEZVOUS_ATTRIBUTE_NAME = "shared_rendezvous"
# TODO(b/202429845): Remove this flag and related args:
USE_FUNCTION_SUBTYPING = True

_graph_building_time_counter = monitoring.Counter(
    "/tensorflow/core/tf_function/graph_building_time_usecs",
    "Time for tf.function to build a graph (us).")


def _type_spec_for(x):
  """Returns a TypeSpec for `x`, or `None` if `x` doesn't have a TensorSpec."""
  if isinstance(x, ops.Tensor):
    return tensor_spec.TensorSpec.from_tensor(x)
  elif isinstance(x, type_spec.TypeSpec):
    return x
  elif isinstance(x, composite_tensor.CompositeTensor):
    return x._type_spec  # pylint: disable=protected-access
  else:
    return None


def _is_type_subset(a, b):
  """Returns true if TypeSpec `b` is a subset of type `a` (or if a is None.)"""
  if a is None:
    return True
  else:
    return a.most_specific_compatible_type(b) == a


def _shape_relaxed_type_for_composite_tensor(x):
  """Returns a shape-relaxed TypeSpec for x (if composite) or x (if not)."""
  if isinstance(x, composite_tensor.CompositeTensor):
    # pylint: disable=protected-access
    return x._type_spec._with_tensor_ranks_only()
  else:
    return x


def common_shape(x, y):
  """Find a `TensorShape` that is compatible with both `x` and `y`."""
  if x is None != y is None:
    raise RuntimeError(
        "Cannot find a common shape when LHS shape is None but RHS shape "
        f"is not (or vice versa): {x} vs. {y}.")
  if x is None:
    return None  # The associated input was not a Tensor, no shape generated.
  if not isinstance(x, tensor_shape.TensorShape):
    raise TypeError(f"`x` must be a TensorShape, got type {type(x)}.")
  if not isinstance(y, tensor_shape.TensorShape):
    raise TypeError(f"`y` must be a TensorShape, got type {type(y)}.")
  if x.rank != y.rank or x.rank is None:
    return tensor_shape.TensorShape(None)
  dims = []
  for dim_x, dim_y in zip(x.dims, y.dims):
    if (dim_x != dim_y
        or tensor_shape.dimension_value(dim_x) is None
        or tensor_shape.dimension_value(dim_y) is None):
      dims.append(None)
    else:
      dims.append(tensor_shape.dimension_value(dim_x))
  return tensor_shape.TensorShape(dims)


def is_same_structure(structure1,
                      structure2,
                      check_values=False):
  """Check two structures for equality, optionally of types and of values."""
  try:
    nest.assert_same_structure(structure1, structure2, expand_composites=True)
  except (ValueError, TypeError):
    return False
  if check_values:
    flattened1 = nest.flatten(structure1, expand_composites=True)
    flattened2 = nest.flatten(structure2, expand_composites=True)
    # First check the types to avoid AttributeErrors.
    if any(type(f1) != type(f2) for f1, f2 in zip(flattened1, flattened2)):
      return False
    return flattened1 == flattened2
  return True


def _parse_func_attrs(attributes):
  """Convert the keyword arguments into function_def attributes.

  Currently only support primitive types: bool, int, float and string.

  Args:
    attributes: the dictionary of attributes.
  Returns:
    A dict of attributes where the key is the name of attribute and the value
      is the AttrValue proto.
  Raises:
    ValueError: If the kwargs contains unallowlisted name or unsupported value
      types.
  """
  attrs = {}
  for key, value in attributes.items():
    if isinstance(value, attr_value_pb2.AttrValue):
      attrs[key] = value
    # bool type check has to happen before int since bool is a subclass of int.
    elif isinstance(value, bool):
      attrs[key] = attr_value_pb2.AttrValue(b=value)
    elif isinstance(value, int):
      attrs[key] = attr_value_pb2.AttrValue(i=value)
    elif isinstance(value, float):
      attrs[key] = attr_value_pb2.AttrValue(f=value)
    elif isinstance(value, (str, bytes, six.text_type)):
      attrs[key] = attr_value_pb2.AttrValue(s=compat.as_bytes(value))
    else:
      raise ValueError(f"Attribute {key} must be bool, int, float, string, or "
                       f"AttrValue. Got {type(value)}.")
  return attrs


class _InterpolateFunctionError(object):
  """Context Manager that interpolates the exception from 'top_level_func'."""

  __slots__ = ["_func"]

  def __init__(self, top_level_func):
    self._func = top_level_func

  def __enter__(self):
    pass

  def __exit__(self, typ, exc, tb):
    if not exc or not isinstance(exc, errors.OpError):
      return False
    message = compat.as_text(exc.message)
    _, func_tags, _ = error_interpolation.parse_message(message)
    g = None
    for func_tag in func_tags:
      # TODO(mdan): Tests should cover this.
      if func_tag.name == compat.as_str(self._func.name):
        g = self._func.graph
      elif g:
        next_func = g._get_function(func_tag.name)  # pylint: disable=protected-access
        if next_func is not None and isinstance(next_func,
                                                _EagerDefinedFunction):
          g = next_func.graph
    if g:
      exc._message = error_interpolation.interpolate(message, g)  # pylint: disable=protected-access
    return False


_function_callbacks = set()


def add_function_callback(function_callback):
  """Add a callback function for Function creation.

  The callback function has the signature:

    `def function_callback(function, name, graph, inputs, outputs):`

  where:
  - `function`: _EagerDefinedFunction being created before finalizing the graph.
      Do not modify the function directly but instead modify the graph.
  - `name`: name of the function.
  - `graph`: Graph of the function.
  - `inputs`: `tuple` of tensors used as inputs to the function.
  - `outputs`: `tuple` of tensors used as outputs from the function.

  The callback is at the top of the `_EagerDefinedFunction` construction, giving
  callback an opportunity to make the last edits to the graph. Do not make
  changes to `graph, inputs`, and `outputs` manually, but, instead, set the
  `graph` as the default then define ops.

  Repeated registration of the same callback function is idempotent.
  After a callback is added, it can be removed with the
  `remove_function_callback()` method.

  Args:
    function_callback: The callback to add.
  """
  _function_callbacks.add(function_callback)


def remove_function_callback(function_callback):
  """Remove an already-added function callback.

  See the doc string of `add_function_callback()` for more information.

  Args:
    function_callback: The callback to remove.
  """
  _function_callbacks.remove(function_callback)


def clear_function_callbacks():
  """Clear all function callbacks, if any have been regisered."""
  _function_callbacks.clear()


_FORWARD_PREFIX = "__forward_"
_BACKWARD_PREFIX = "__backward_"
_INFERENCE_PREFIX = "__inference_"


def _forward_name(n):
  """The name of a generated forward defun named n."""
  return "%s%s_%s" % (_FORWARD_PREFIX, n, ops.uid())


def _backward_name(n):
  """The name of a generated backward defun named n."""
  return "%s%s_%s" % (_BACKWARD_PREFIX, n, ops.uid())


def _inference_name(n):
  """The name of a forward-but-no-gradient defun named n."""
  return "%s%s_%s" % (_INFERENCE_PREFIX, n, ops.uid())


class _EagerDefinedFunctionDeleter(object):
  """Unregister function from eager context."""

  __slots__ = ["name"]

  def __init__(self, name):
    self.name = name

  def __del__(self):
    try:
      context.remove_function(self.name)
    except TypeError:
      # Suppress some exceptions, mainly for the case when we're running on
      # module deletion. Things that can go wrong include the context module
      # already being unloaded, self._handle._handle_data no longer being
      # valid, and so on. Printing warnings in these cases is silly
      # (exceptions raised from __del__ are printed as warnings to stderr).
      pass  # 'NoneType' object is not callable when the handle has been
      # partially unloaded.
    except AttributeError:
      pass  # 'NoneType' object has no attribute 'eager_mode' when context has
      # been unloaded. Will catch other module unloads as well.


class FunctionAlreadyGarbageCollectedError(Exception):

  def __init__(self, function_name):
    super(FunctionAlreadyGarbageCollectedError, self).__init__(
        "{} has already been garbage collected and cannot be called.".format(
            function_name))


# TODO(apassos) get rid of this by splitting framework.function._DefinedFunction
# so it doesn't have the definition-generating logic and is just a container for
# an already-defined function.
class _EagerDefinedFunction(object):
  """Callable with the interface of `framework.function._DefinedFunction`.

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
      outputs: the tensors in the graph which will be outputs from the function
      attrs: dict mapping names of attributes to their AttrValue values
    """
    for function_callback in _function_callbacks:
      function_callback(self, name, graph, tuple(inputs), tuple(outputs))

    input_ops = set(arg.op for arg in inputs)
    operations = [op for op in graph.get_operations() if op not in input_ops]

    graph_output_names = graph._output_names  # pylint: disable=protected-access
    if (graph_output_names is not None and
        all(ops.tensor_id(t) in graph_output_names for t in outputs)):
      output_names = [
          compat.as_bytes(graph_output_names[ops.tensor_id(t)]) for t in outputs
      ]
      if len(set(output_names)) != len(output_names):
        # There are duplicate names for some reason, probably an invalid
        # signature. Revert to auto-naming.
        output_names = []
    else:
      output_names = []
    fn = pywrap_tf_session.TF_GraphToFunction_wrapper(
        graph._c_graph,  # pylint: disable=protected-access
        compat.as_str(name),
        False,
        [o._c_op for o in operations],  # pylint: disable=protected-access
        [t._as_tf_output() for t in inputs],  # pylint: disable=protected-access
        [t._as_tf_output() for t in outputs],  # pylint: disable=protected-access
        output_names,
        [o._c_op for o in graph.control_outputs],  # pylint: disable=protected-access
        [],  # control_output_names
        None,
        compat.as_str(""))

    for name, attr_value in attrs.items():
      serialized = attr_value.SerializeToString()
      # TODO(iga): this creates and deletes a new TF_Status for every attr.
      # It might be worth creating a convenient way to re-use status.
      pywrap_tf_session.TF_FunctionSetAttrValueProto(fn, compat.as_str(name),
                                                     serialized)

    # TODO(apassos) avoid creating a FunctionDef (specially to grab the
    # signature, but also in general it's nice not to depend on it.
    with c_api_util.tf_buffer() as buffer_:
      pywrap_tf_session.TF_FunctionToFunctionDef(fn, buffer_)
      proto_data = pywrap_tf_session.TF_GetBuffer(buffer_)
    function_def = function_pb2.FunctionDef()
    function_def.ParseFromString(compat.as_bytes(proto_data))
    self._name = compat.as_bytes(function_def.signature.name)
    with ops.init_scope():
      if context.executing_eagerly():
        context.ensure_initialized()
        context.add_function(fn)
        self._function_deleter = _EagerDefinedFunctionDeleter(self.name)
        self._registered_on_context = True
    self.definition = function_def
    self.signature = function_def.signature
    self._num_outputs = len(self.signature.output_arg)
    self._output_types = [o.type for o in self.signature.output_arg]
    self._output_shapes = [o.shape for o in outputs]
    self._control_captures = graph.control_captures
    # Shallow copy outputs since ConcreteFunction may mutate it.
    self._func_graph_outputs = list(outputs)
    self.grad_func_name = None
    self.python_grad_func = None
    self._c_func = c_api_util.ScopedTFFunction(fn)
    self._grad_func = None
    self.graph = graph
    self._stateful_ops = tuple(op for op in operations if op._is_stateful)  # pylint: disable=protected-access

  def add_to_graph(self, g=None):
    """Add the function to the current context or a graph, if supplied.

    Args:
      g: the graph to add the function to. If not supplied, the function will
        be added to the current context.
    """
    # pylint: disable=protected-access
    if not g and context.executing_eagerly():
      ctx = context.context()
      if not ctx.has_function(self.name):
        ctx.add_function_def(self.definition)
    else:
      if not g._is_function(self.name):
        g._add_function(self)
      for f in self.graph._functions.values():
        if not g._is_function(f.name):
          g._add_function(f)
    # pylint: enable=protected-access

  @property
  def name(self):
    return self._name

  @property
  def stateful_ops(self):
    return self._stateful_ops

  def call(self, ctx, args, cancellation_manager=None):
    """Calls this function with `args` as inputs.

    `ConcreteFunction` execution respects device annotations only if the
    function won't be compiled with xla.

    Args:
      ctx: a Context object
      args: a list of arguments to supply this function with.
      cancellation_manager: a `CancellationManager` object that can be used to
        cancel function execution.

    Returns:
      The outputs of the function call.

    Raises:
      ValueError: if the number of arguments is incorrect.
      FunctionAlreadyGarbageCollectedError: if the function is no longer
        available to be called because it has been garbage collected.
    """
    if len(args) != len(self.signature.input_arg):
      raise ValueError(
          f"Signature specifies {len(list(self.signature.input_arg))} "
          f"arguments, got: {len(args)}.")

    # If the `ScopedTFFunction` (accessed via `_c_func`) has already been
    # cleaned up as a part of garbage collection, this `_EagerDefinedFunction`
    # should also be garbage and is likely being called as part of a `__del__`
    # elsewhere. In that case, there's nothing we can do, so we raise an
    # exception for the caller to handle.
    if self._c_func.has_been_garbage_collected:
      raise FunctionAlreadyGarbageCollectedError(self.name)

    function_call_options = ctx.function_call_options
    if function_call_options.config_proto_serialized is None:
      config = function_utils.get_disabled_rewriter_config()
    else:
      config = function_call_options.config_proto_serialized
    executor_type = function_call_options.executor_type or ""

    executing_eagerly = ctx.executing_eagerly()
    attrs = ("executor_type", executor_type, "config_proto", config)
    if executing_eagerly:
      with _InterpolateFunctionError(self):
        if cancellation_manager is None:
          outputs = execute.execute(
              str(self.signature.name),
              num_outputs=self._num_outputs,
              inputs=args,
              attrs=attrs,
              ctx=ctx)
        else:
          outputs = execute.execute_with_cancellation(
              str(self.signature.name),
              num_outputs=self._num_outputs,
              inputs=args,
              attrs=attrs,
              ctx=ctx,
              cancellation_manager=cancellation_manager)
      # Replace empty list with None
      outputs = outputs or None
    else:
      # TODO(akshayka): Either remove this if the FunctionLibraryRuntime
      # creates `PartitionedCallOp` kernels by default, or remove the previous
      # branch if a TPU kernel is registered for `PartitionedCall`.
      with _InterpolateFunctionError(self):
        with ops.control_dependencies(self._control_captures):
          # The caller must use record_operation to record this operation in the
          # eager case, so we enforce the same requirement for the non-eager
          # case by explicitly pausing recording. We don't have a gradient
          # registered for PartitionedCall, so recording this operation confuses
          # forwardprop code (GradientTape manages to ignore it).
          with tape.stop_recording():
            outputs = functional_ops.partitioned_call(
                args=args,
                f=self,
                tout=self._output_types,
                executing_eagerly=executing_eagerly,
                config=config,
                executor_type=executor_type)

    for i, func_graph_output in enumerate(self._func_graph_outputs):
      handle_data_util.copy_handle_data(func_graph_output, outputs[i])
    if executing_eagerly:
      return outputs
    else:
      # TODO(b/128924522): This additional set_shape should not be
      # necessary. ShapeRefiner likely needs to inspect handle_data. Remove this
      # once that's done.
      for i, shape in enumerate(self._output_shapes):
        outputs[i].set_shape(shape)
      return outputs


def _create_forward_backward_with_graph(attrs, forward_graph, backwards_graph):
  """Creates forward and backward functions from the function graphs."""
  forward_function_name = _forward_name(forward_graph.name)
  common_attributes = dict(attrs)
  # NB: forward and backward function need to drop "_implements".
  # attribute, because their signature contains all the intermediate tensors
  # that they compute. Thus they don't have a stable signature which can
  # be directly optimized downstream.
  # See for more details:
  # https://github.com/tensorflow/community/blob/master/rfcs/20190610-standardizing-composite_ops.md#appendix-future-support-for-optimizing-gradient-functions
  common_attributes.pop(IMPLEMENTS_ATTRIBUTE_NAME, None)
  backward_function_attr = _parse_func_attrs(
      {FORWARD_FUNCTION_ATTRIBUTE_NAME: forward_function_name})
  backward_function_attr.update(common_attributes)
  backward_function = ConcreteFunction(
      backwards_graph, attrs=backward_function_attr)
  forward_function_attr = _parse_func_attrs({
      BACKWARD_FUNCTION_ATTRIBUTE_NAME:
      backward_function.name})
  forward_function_attr.update(common_attributes)
  forward_function = _EagerDefinedFunction(
      forward_function_name, forward_graph, forward_graph.inputs,
      forward_graph.outputs, forward_function_attr)
  return forward_function, backward_function


class _DelayedRewriteGradientFunctions(object):
  """Caches forward/backward functions with a delayed forward rewrite."""

  def __init__(self, func_graph, attrs, func_graph_deleter):
    """Construct an inference function and initialize caches."""
    # A map from the number of forward function outputs with accepted gradients
    # to forward and backward functions, used to cache non-tape backward
    # function generation.
    self._cached_function_pairs = {}
    self._func_graph = func_graph
    self._inference_function = _EagerDefinedFunction(
        _inference_name(self._func_graph.name), self._func_graph,
        self._func_graph.inputs, self._func_graph.outputs, attrs)
    self._attrs = attrs
    self._gradient_name = None
    # Note that the FuncGraph is mutated later, so we need to inspect it now to
    # figure out the user-specified outputs of the inference function.
    self._num_inference_outputs = len(self._func_graph.outputs)
    self._func_graph_deleter = func_graph_deleter

  def forward_backward(self, num_doutputs=None):
    """A possibly-cached pair of forward and backward functions."""
    if num_doutputs is None:
      num_doutputs = self._num_inference_outputs
    forward_backward = self._cached_function_pairs.get(num_doutputs)
    if forward_backward is not None:
      return forward_backward
    forward, backward = self._construct_forward_backward(num_doutputs)
    self._cached_function_pairs[num_doutputs] = (forward, backward)
    return forward, backward

  def _construct_forward_backward(self, num_doutputs):
    """Constructs a pair of forward and backward functions.

    Args:
      num_doutputs: The constructed backprop function will take output gradients
        for the first `num_doutputs` outputs of the forward function. Defaults
        to the number of outputs for the inference function, but when
        higher-order gradients are computed this will increase to include side
        outputs.

    Returns:
      A pair of (forward_function, backward_function):
        forward_function: A re-generated inference function (an
          _EagerDefinedFunction) to account for new side outputs, if any extra
          were required when building the backward pass.
        backward_function: A ConcreteFunction that Takes `num_doutputs`
          arguments and returns gradients with respect to inputs of the forward
          function.
    """
    trainable_outputs = [
        output for output in self._func_graph.outputs[:num_doutputs]
        if backprop_util.IsTrainable(output)]

    signature = []
    for t in trainable_outputs:
      signature.append(
          tensor_spec.TensorSpec(*default_gradient.shape_and_dtype(t)))

    def _backprop_function(*grad_ys):
      with ops.device(None):
        return gradients_util._GradientsHelper(  # pylint: disable=protected-access
            trainable_outputs,
            self._func_graph.inputs,
            grad_ys=grad_ys,
            src_graph=self._func_graph)

    with self._func_graph.as_default():
      backwards_graph = func_graph_module.FuncGraph(
          _backward_name(self._func_graph.name))
      func_graph_module.func_graph_from_py_func(
          name=backwards_graph.name,
          python_func=_backprop_function,
          args=[], kwargs={},
          signature=signature,
          func_graph=backwards_graph)
      backwards_graph_captures = backwards_graph.external_captures
      captures_from_forward = [
          c for c in backwards_graph_captures if
          not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph]

      existing_outputs = object_identity.ObjectIdentitySet(
          self._func_graph.outputs)
      for capture in captures_from_forward:
        if capture not in existing_outputs:
          existing_outputs.add(capture)
          self._func_graph.outputs.append(capture)

      forward_function, backward_function = _create_forward_backward_with_graph(
          self._attrs, self._func_graph, backwards_graph)
      return forward_function, backward_function

  def _rewrite_forward_and_call_backward(self, op, *doutputs):
    """Add outputs to the forward call and feed them to the grad function."""
    forward_function, backwards_function = self.forward_backward(len(doutputs))
    if not backwards_function.outputs:
      return backwards_function.structured_outputs
    forward_function.add_to_graph(op.graph)

    # pylint: disable=protected-access
    # Rewrite an inference call op to be a forward call op
    op._set_func_attr("f", forward_function.name)
    op._set_type_list_attr("Tout", forward_function._output_types)
    op._add_outputs(
        forward_function._output_types[len(op.outputs):],
        forward_function._output_shapes[len(op.outputs):])
    for i in range(len(op.outputs)):
      func_graph_output = forward_function._func_graph_outputs[i]
      handle_data_util.copy_handle_data(func_graph_output, op.outputs[i])
    # pylint: enable=protected-access

    capture_mapping = dict(
        zip((ops.tensor_id(t) for t in self._func_graph.outputs), op.outputs))
    remapped_captures = [
        capture_mapping.get(ops.tensor_id(capture), capture)
        for capture in backwards_function.captured_inputs
    ]

    # Replace Nones with zeros since we're calling a graph function which
    # expects numeric inputs.
    cleaned_doutputs = []
    for doutput, placeholder in zip(doutputs, self._func_graph.outputs):
      if backprop_util.IsTrainable(placeholder):
        if isinstance(doutput, indexed_slices.IndexedSlices):
          # Gradient passed to a backward ConcreteFunction must be tf.Tensor,
          # so we convert tf.IndexedSlices to tf.Tensor.
          cleaned_doutputs.append(ops.convert_to_tensor(doutput))
        elif doutput is not None:
          cleaned_doutputs.append(doutput)
        else:
          cleaned_doutputs.append(default_gradient.zeros_like(placeholder))

    # Compute the gradients using the side outputs
    return backwards_function._call_flat(  # pylint: disable=protected-access
        cleaned_doutputs, remapped_captures)

  def get_gradient_function(self):
    """Returns gradient function.

    The gradient rewrites an inference call op to a forward call op, but does
    not modify a pre-existing forward call op. It then computes the gradient
    from the output's gradients and the side outputs of the forward op.
    """
    return self._rewrite_forward_and_call_backward

  def forward(self, inference_args=None, input_tangents=None):
    """A forward function with only user-specified outputs.

    The call operation for the returned inference function can be rewritten into
    a forward function. This only happens if the backward function (from the
    `backward` method) ends up being used to compute gradients.

    This approach avoids constructing unnecessary graphs, but it only works if
    we are calling this function when not executing eagerly.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function. Unused, but taken for compatibility with
        _TapeGradientFunctions.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`. Unused; if required, tape functions must be used
        instead.

    Returns:
      An _EagerDefinedFunction.
    """
    del inference_args  # unused
    if input_tangents:
      # This class does not support special-cased forwardprop. The arguments are
      # here for compatibility with _TapeGradientFunctions.
      raise errors.InternalError("unexpectedly got forwardprop information in "
                                 "a class that does not support forwardprop.")
    return self._inference_function

  def _backward(self, outputs):
    """Fetch a backward function for `outputs` from the forward function."""
    def _backward_function(*args):
      call_op = outputs[0].op
      return self._rewrite_forward_and_call_backward(call_op, *args)
    return _backward_function, outputs

  def record(self, flat_outputs, inference_args, input_tangents):
    """Record the function call operation.

    _DelayedRewriteGradientFunctions supports only first-order backprop tape
    gradients (and then only when graph building). It does not work with
    higher-order tape gradients or forward autodiff, but does work with
    higher-order symbolic gradients (tf.gradients).

    Args:
      flat_outputs: The result of running `forward`.
      inference_args: A flat list of Tensors with inference inputs to the
        operation.
      input_tangents: A flat list of Tensors with input tangents consumed by the
        operation.
    """
    backward_function, to_record = self._backward(flat_outputs)
    tape.record_operation(self._inference_function.signature.name,
                          to_record, inference_args + input_tangents,
                          backward_function)


# Contains information about a forward function wrapped to compute jvps.
_ForwardWrapper = collections.namedtuple(
    "_ForwardWrapper", (
        # The wrapper Graph.
        "graph",
        # A flat list of non-tangent Tensor outputs from the wrapped forward
        # function.
        "outputs",
        # Indices for output tangents, same format as
        # forwardprop_util.pack_tangents.
        "output_indices",
        # A flat list of tangents for `outputs`.
        "output_tangents"))


class _TapeGradientFunctions(object):
  """Caches forward and backward functions compatible with eager gradients.

  In contrast to the delayed-rewrite approach in
  `_DelayedRewriteGradientFunctions` which only works with delayed execution,
  the forward function generated by this class has a fixed set of outputs which
  may be preserved by a tape in order to compute gradients later.

  This class is abstract; its child classes differ in how many side outputs of
  the forward function their backward function accepts gradients for, which
  determines whether higher-order tape gradients are possible.
  """

  def __init__(self, func_graph, attrs, func_graph_deleter,
               forwardprop_input_indices, delayed_rewrite_functions,
               need_gradients_for_jvps):
    self._func_graph = func_graph
    self._forward_graph = None
    self._attrs = attrs
    self._forward = None
    self._backward = None
    self._num_outputs = len(func_graph.outputs)
    self._func_graph_deleter = func_graph_deleter
    self._forwardprop_input_indices = forwardprop_input_indices
    self._forwardprop_output_indices = None
    self._num_forwardprop_outputs = 0
    self._num_inference_outputs = len(func_graph.outputs)
    self._num_trainable_inference_outputs = len(
        [t for t in func_graph.outputs if backprop_util.IsTrainable(t)])
    self._delayed_rewrite_functions = delayed_rewrite_functions
    self._need_gradients_for_jvps = need_gradients_for_jvps

  def _build_functions_for_outputs(
      self, outputs, inference_args, input_tangents):
    """Forward+backward functions where the backward function sees `outputs`."""
    # First figure out which of `outputs` are trainable. We'll accept gradients
    # for each of these in the backward function.
    handles_to_variables = self._func_graph.variable_captures
    trainable_outputs = []
    trainable_indices = []
    for index, output in enumerate(outputs):

      if backprop_util.IsTrainable(output):
        # Swap in the Variable object for resource handles if we can so
        # sparse gradients work.
        output = handles_to_variables.get(id(output), output)
        trainable_outputs.append(output)
        trainable_indices.append(index)

    backwards_graph = func_graph_module.FuncGraph(
        _backward_name(self._func_graph.name))
    with backwards_graph.as_default():
      gradients_wrt_outputs = []
      for output in trainable_outputs:
        gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(
            output)
        gradient_placeholder = graph_placeholder(gradient_dtype, gradient_shape)
        handle_data_util.copy_handle_data(output, gradient_placeholder)
        gradients_wrt_outputs.append(gradient_placeholder)
      with ops.device(None):
        gradients_wrt_inputs = gradients_util._GradientsHelper(  # pylint: disable=protected-access
            trainable_outputs,
            self._func_graph.inputs,
            grad_ys=gradients_wrt_outputs,
            src_graph=self._func_graph)

      if input_tangents:
        # Convert IndexedSlices to dense tensors (as we do elsewhere for
        # function gradients). Our C++ bindings don't know how to handle them
        # currently.
        gradients_wrt_inputs = nest.map_structure(
            lambda x: ops.convert_to_tensor(x) if x is not None else None,
            gradients_wrt_inputs)
      captures_from_forward = [
          c for c in backwards_graph.external_captures
          if not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph
      ]
      existing_outputs = object_identity.ObjectIdentitySet(
          self._func_graph.outputs)
      for capture in captures_from_forward:
        if capture not in existing_outputs:
          existing_outputs.add(capture)
          self._func_graph.outputs.append(capture)

    # The ordering of `backwards_graph.inputs` is important: inputs of
    # `backward_function` correspond to outputs (including
    # side outputs) of `self._tape_forward_function`.
    backwards_graph.inputs = (
        gradients_wrt_outputs + backwards_graph.internal_captures)
    backwards_graph.outputs.extend(
        grad
        for grad in nest.flatten(gradients_wrt_inputs, expand_composites=True)
        if grad is not None)
    backwards_graph.structured_outputs = gradients_wrt_inputs

    forward_function, backward_function = _create_forward_backward_with_graph(
        self._attrs, self._func_graph, backwards_graph)

    if not input_tangents:
      # There is no need to special-case forwardprop, so we can return the
      # forward+backward pair we've created without further wrapping.
      return (forward_function, self._func_graph, backward_function,
              # No forwardprop outputs.
              None, 0)
    forward_wrapper = self._wrap_forward_function_with_jvps(
        forward_function, backward_function, inference_args, input_tangents)
    (wrapped_backwards_graph,
     forward_wrapper) = self._wrap_backward_function_with_jvp_backprop(
         backward_function, gradients_wrt_outputs, forward_wrapper)
    # Now that we've added new captures, we need to make sure forward outputs
    # are in the same order the backward function expects them to be in:
    # [inference outputs] + [jvps] + [side outputs] + [captures].
    forward_wrapper = self._shuffle_forward_outputs(forward_wrapper)
    (wrapped_forward_function,
     wrapped_backward_function) = _create_forward_backward_with_graph(
         self._attrs, forward_wrapper.graph, wrapped_backwards_graph)
    if (len(inference_args) + len(input_tangents)
        != len(forward_wrapper.graph.inputs)):
      raise errors.InternalError(
          f"The forward graph had {len(forward_wrapper.graph.inputs)} inputs, "
          f"but we expected {len(inference_args) + len(input_tangents)} "
          f"({len(inference_args)} inference inputs and "
          f"{len(input_tangents)} input tangents).")
    return (wrapped_forward_function, forward_wrapper.graph,
            wrapped_backward_function, forward_wrapper.output_indices,
            len(forward_wrapper.output_tangents))

  def _wrap_forward_function_with_jvps(
      self, forward_function, backward_function,
      inference_args, input_tangents):
    """Adds inline JVP computation to a forward function."""
    forward_wrapper_graph = func_graph_module.FuncGraph(
        _forward_name(self._func_graph.name))
    with forward_wrapper_graph.as_default():
      # Tell forward accumulators to free up space for new JVP computations,
      # since one may be in the process of computing a JVP (if that computation
      # triggered this function building).
      #
      # We'll make symbolic versions of input JVPs, run the forward function
      # under forward accumulators to get symbolic output JVPs, then set those
      # as outputs of the new wrapped forward function.
      with forwardprop_util.push_forwardprop_state():
        forward_captures = {
            ops.tensor_id(internal): external
            for external, internal in self._func_graph.captures}
        for input_index, real_input in enumerate(self._func_graph.inputs):
          # This loop is more or less equivalent to running tf.identity on each
          # of self._func_graph.inputs. However, doing that also captures jvps
          # for resource handles, which confuses the jvp capturing code below
          # (since primal inputs are interwoven with jvp inputs).
          input_placeholder = array_ops.placeholder(
              dtype=real_input.dtype,
              shape=real_input.shape)
          capture = forward_captures.get(ops.tensor_id(real_input))
          if capture is not None:
            forward_wrapper_graph.add_capture(capture, input_placeholder)
            if capture.dtype == dtypes.resource:
              handle_data_util.copy_handle_data(capture, input_placeholder)
          else:
            forward_wrapper_graph.inputs.append(input_placeholder)
        for inp, arg in zip(forward_wrapper_graph.inputs, inference_args):
          tape.record_operation(
              "captured_value", [inp], [arg],
              backward_function=lambda x: [x],
              forward_function=lambda x: [x])
        num_inference_inputs = len(inference_args)
        for tape_indices in self._forwardprop_input_indices:
          for input_index, jvp_index in tape_indices:
            input_placeholder = forward_wrapper_graph.inputs[input_index]
            if len(forward_wrapper_graph.inputs) != jvp_index:
              raise errors.InternalError(
                  f"Expected {jvp_index} forward graph inputs, "
                  f"got {len(forward_wrapper_graph.inputs)}.")
            gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(
                input_placeholder)
            jvp_placeholder = graph_placeholder(gradient_dtype, gradient_shape)
            external_jvp = input_tangents[jvp_index - num_inference_inputs]
            forward_wrapper_graph.add_capture(external_jvp, jvp_placeholder)
            tensor_shape.TensorShape(
                external_jvp.shape).assert_is_compatible_with(
                    jvp_placeholder.shape)
            tape.record_operation(
                "captured_value",
                [jvp_placeholder],
                [external_jvp],
                backward_function=lambda x: [x],
                forward_function=lambda x: [x])
        forward_inputs = forward_wrapper_graph.inputs[:num_inference_inputs]
        gradient_function = (
            self._delayed_rewrite_functions._rewrite_forward_and_call_backward)  # pylint: disable=protected-access
        with ops.get_default_graph()._override_gradient_function(  # pylint: disable=protected-access
            {"PartitionedCall": gradient_function,
             "StatefulPartitionedCall": gradient_function}):
          forward_outputs = forward_function.call(context.context(),
                                                  forward_inputs)
          if isinstance(forward_outputs, ops.Operation):
            # _wrapped_backward_function expects a list, but if the function has
            # no outputs its call() returns an Operation. We need to undo that
            # so we don't cause problems later.
            forward_outputs = []
        py_backward, _ = self._wrap_backward_function(
            self._func_graph, backward_function, forward_outputs)
      # We will never request backward tape gradients for this operation
      # directly since we're wrapping the call; forwardprop will call the
      # backward function (and nested forward accumulators may build
      # higher-order gradients), but any watching GradientTapes should ignore
      # it.
      #
      # TODO(allenl): It might be better to explicitly stop backward recording
      # so we don't use the second-order tape cases unnecessarily.
      tape.record_operation_forwardprop_only(
          forward_function.signature.name,
          forward_outputs, forward_inputs, py_backward, None)
      output_indices, output_tangents = (
          pywrap_tfe.TFE_Py_PackJVPs(forward_outputs))
      output_tangents = [forward_wrapper_graph.capture(t)
                         for t in output_tangents]
    return _ForwardWrapper(
        graph=forward_wrapper_graph, outputs=forward_outputs,
        output_indices=output_indices, output_tangents=output_tangents)

  def _wrap_backward_function_with_jvp_backprop(
      self, backward_function, gradients_wrt_outputs, forward_wrapper):
    """Wraps `backward_function` to include gradients for JVPs."""
    wrapped_backwards_graph = func_graph_module.FuncGraph(
        _backward_name(self._func_graph.name))
    with wrapped_backwards_graph.as_default():
      py_backward, recorded_outputs = self._wrap_backward_function(
          self._func_graph, backward_function, forward_wrapper.outputs)
      trainable_index = 0
      forward_doutputs = []
      doutput_args = []
      for output in recorded_outputs:
        if backprop_util.IsTrainable(output):
          doutput = gradients_wrt_outputs[trainable_index]
          doutput_placeholder = graph_placeholder(doutput.dtype, doutput.shape)
          doutput_args.append(doutput_placeholder)
          forward_doutputs.append(doutput_placeholder)
          trainable_index += 1
        else:
          doutput_args.append(None)

      dinputs = py_backward(*doutput_args)
      existing_outputs = object_identity.ObjectIdentitySet(
          forward_wrapper.outputs + forward_wrapper.output_tangents)
      num_processed_output_tangents = 0
      gradients_wrt_output_tangents = []
      tangent_doutputs = []
      output_tangents = forward_wrapper.output_tangents
      output_indices = forward_wrapper.output_indices
      if self._need_gradients_for_jvps:
        # TODO(allenl): Consider using a throwaway graph to avoid extra gradient
        # evaluations; gradients for jvps may have common subgraphs.
        while num_processed_output_tangents != len(output_tangents):
          for output in output_tangents[num_processed_output_tangents:]:
            gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(
                output)
            placeholder = graph_placeholder(gradient_dtype, gradient_shape)
            gradients_wrt_output_tangents.append(placeholder)
            tangent_doutputs.append(placeholder)
          num_processed_output_tangents = len(output_tangents)
          with ops.device(None):
            gradients_wrt_inputs = gradients_util._GradientsHelper(  # pylint: disable=protected-access
                output_tangents,
                forward_wrapper.graph.inputs,
                grad_ys=gradients_wrt_output_tangents,
                src_graph=forward_wrapper.graph)
          dinputs = [
              backprop.aggregate_indexed_slices_gradients((existing, new))
              for existing, new in zip(dinputs, gradients_wrt_inputs)
              if existing is not None or new is not None]
          dinputs.extend(gradients_wrt_inputs[len(dinputs):])
          captures_from_forward = [
              c for c in wrapped_backwards_graph.external_captures
              if (not isinstance(c, ops.EagerTensor)
                  and c.graph is forward_wrapper.graph)]
          for capture in captures_from_forward:
            if capture not in existing_outputs:
              existing_outputs.add(capture)
              forward_wrapper.outputs.append(capture)
          output_indices, output_tangents = (
              forwardprop_util.pack_tangents(forward_wrapper.outputs))
          output_tangents = [forward_wrapper.graph.capture(t)
                             for t in output_tangents]
          for t in output_tangents:
            existing_outputs.add(t)
    wrapped_backwards_graph.inputs = (
        forward_doutputs[:self._num_trainable_inference_outputs]
        + tangent_doutputs
        + forward_doutputs[self._num_trainable_inference_outputs:]
        + wrapped_backwards_graph.internal_captures)
    wrapped_backwards_graph.structured_outputs = dinputs
    wrapped_backwards_graph.outputs = [t for t in dinputs if t is not None]
    return (wrapped_backwards_graph,
            forward_wrapper._replace(output_indices=output_indices,
                                     output_tangents=output_tangents))

  def _shuffle_forward_outputs(self, forward_wrapper):
    """Reorders function outputs so captures are last."""
    def _index_map(original):
      if original < self._num_inference_outputs:
        return original
      if original >= len(forward_wrapper.outputs):
        return (original - len(forward_wrapper.outputs)
                + self._num_inference_outputs)
      return original + len(forward_wrapper.output_tangents)
    output_indices = nest.map_structure(
        _index_map, forward_wrapper.output_indices)
    forward_wrapper.graph.outputs = (
        forward_wrapper.outputs[:self._num_inference_outputs]
        + forward_wrapper.output_tangents
        + forward_wrapper.outputs[self._num_inference_outputs:])
    return forward_wrapper._replace(output_indices=output_indices)

  def forward(self, inference_args, input_tangents):
    """Construct or fetch a forward function with side-outputs.

    When graph building without a tape active, symbolic gradients rely on
    regenerating the backward function for higher-order gradients (to account
    for new side outputs of the rewritten forward function call). Thus there is
    no fixed backward function for this case. However, when a tape is active
    (eager or graph building), we generate fixed backward and forward functions
    at forward function call time.

    This difference between the tape and non-tape cases is to avoid building
    unneeded backward functions while graph building (where we may or may not
    eventually need gradients).

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.

    Returns:
      A forward _EagerDefinedFunction.
    """
    if self._forward is None:
      (self._forward, self._forward_graph, self._backward,
       self._forwardprop_output_indices, self._num_forwardprop_outputs) = (
           self._forward_and_backward_functions(inference_args, input_tangents))
    return self._forward

  def _wrap_backward_function(self, forward_graph, backward, outputs):
    """Create a backward function given `outputs` from the forward function."""
    capture_mapping = dict(
        zip((ops.tensor_id(t) for t in forward_graph.outputs), outputs))
    captured_inputs = backward.captured_inputs
    remapped_captures = [
        capture_mapping.get(ops.tensor_id(capture), capture)
        for capture in captured_inputs
    ]
    if any(t.graph is forward_graph for t in remapped_captures
           if not isinstance(t, ops.EagerTensor)):
      incorrect_mapping = [t for t in remapped_captures
                           if (not isinstance(t, ops.EagerTensor) and
                               t.graph is not forward_graph)]
      raise errors.InternalError("Failed to map all backward graph captures to "
                                 "the forward graph. Incorrectly mapped: "
                                 f"{incorrect_mapping}.")
    # We may need to use zeros_like to get a zero for variant Tensors with
    # unconnected gradients. We do that in advance so we don't have to hold on
    # to the outputs themselves, which may not be needed otherwise.
    variant_zeros_like = {}
    backward_function_inputs = (len(backward.inputs) - len(captured_inputs))
    recorded_outputs = []
    trainable_recorded_outputs = 0
    skip_positions = []
    if self._num_forwardprop_outputs and not self._need_gradients_for_jvps:
      relevant_outputs = (
          outputs[:self._num_inference_outputs]
          + outputs[self._num_inference_outputs
                    + self._num_forwardprop_outputs:])
    else:
      relevant_outputs = outputs
    for output_index, output in enumerate(relevant_outputs):
      if trainable_recorded_outputs < backward_function_inputs:
        recorded_outputs.append(output)
      if backprop_util.IsTrainable(output):
        trainable_recorded_outputs += 1
      else:
        skip_positions.append(output_index)
      if output.dtype == dtypes.variant:
        variant_zeros_like[output_index] = default_gradient.zeros_like(output)

    def _backward_function_wrapper(*args):
      """Process output gradients and call the backward function."""
      if not backward.outputs:
        return backward.structured_outputs

      processed_args = []
      input_index = 0
      for output_index, arg in enumerate(args):
        # Convert IndexedSlices to dense tensors. The IndexedSlices optimization
        # is only really effective when doing tf.gather(variable) as the
        # adjoint functions for most operations are unlikely to preserve the
        # sparsity in IndexedSlices.
        if isinstance(arg, indexed_slices.IndexedSlices):
          arg = ops.convert_to_tensor(arg)
        if output_index in skip_positions:
          continue
        if arg is None:
          # We're calling a (non-polymorphic) ConcreteFunction, so we need to
          # have a Tensor value for each Tensor we thought would be trainable
          # based on its dtype, even if it ended up being unconnected.
          input_placeholder = backward.inputs[
              input_index]
          if input_placeholder.dtype == dtypes.variant:
            arg = variant_zeros_like[output_index]
          else:
            arg = array_ops.zeros(
                *default_gradient.shape_and_dtype(input_placeholder))
        processed_args.append(arg)
        input_index += 1
        if input_index >= backward_function_inputs:
          break
      return backward._call_flat(  # pylint: disable=protected-access
          processed_args, remapped_captures)

    return _backward_function_wrapper, recorded_outputs

  def record(self, flat_outputs, inference_args, input_tangents):
    """Record the function call operation.

    For backprop, indicates the backward function to use and which new Tensors
    must be watched. For forwardprop from eager, the function call itself will
    have produced tangents which need to be recorded.

    Args:
      flat_outputs: The result of running `forward`.
      inference_args: A flat list of Tensors with inference inputs to the
        operation.
      input_tangents: A flat list of Tensors with input tangents consumed by the
        operation.
    """
    backward_function, to_record = self._wrap_backward_function(
        self._forward_graph, self._backward, flat_outputs)
    if self._forwardprop_output_indices:
      tape.record_operation_backprop_only(
          self._forward.signature.name,
          to_record, inference_args,
          backward_function)
      tape.record_operation_forwardprop_only(
          self._forward.signature.name,
          flat_outputs, inference_args + input_tangents,
          backward_function,
          self._forwardprop_output_indices)
    else:
      tape.record_operation(self._forward.signature.name,
                            to_record, inference_args + input_tangents,
                            backward_function)


class _FirstOrderTapeGradientFunctions(_TapeGradientFunctions):
  """Caches tape-friendly functions for first-order gradients."""

  def __init__(self, func_graph, attrs, func_graph_deleter,
               forwardprop_input_indices, delayed_rewrite_functions,
               need_gradients_for_jvps):
    super(_FirstOrderTapeGradientFunctions, self).__init__(
        func_graph, attrs, func_graph_deleter, forwardprop_input_indices,
        delayed_rewrite_functions, need_gradients_for_jvps)
    self._func_graph_deleter = func_graph_deleter
    self._forwardprop_input_indices = forwardprop_input_indices

  def _forward_and_backward_functions(self, inference_args, input_tangents):
    """Shortcut for when only first-order gradients are required.

    The returned backward function does not accept gradients with respect to
    side output of forward_function. This is fine as long as the user can't
    possibly request second order tape gradients, as when they've used a single
    non-persistent GradientTape. Since we don't need the backward function to
    take gradients with respect to side outputs, we can skip some potentially
    slow graph building.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.

    Returns:
      A tuple of (forward_function, backward_function):
        forward_function: Takes the same inputs as the inference function, but
          returns side outputs used by backward_function in addition to the
          inference function's outputs.
        backward_function: Takes side outputs from forward_function and
          gradients with respect to the "real" outputs of forward_function and
          returns gradients with respect to the inputs.
    """
    outputs = self._func_graph.outputs[:self._num_inference_outputs]
    return self._build_functions_for_outputs(
        outputs, inference_args, input_tangents)


class _HigherOrderTapeGradientFunctions(_TapeGradientFunctions):
  """Caches tape-friendly functions for higher-order gradients."""

  # TODO(b/136189779): Cond/while under a tape may need similar logic. Consider
  # generalizing if so.
  def _forward_and_backward_functions(self, inference_args, input_tangents):
    """Forward and backward functions suitable for higher-order gradients.

    Unlike in `_FirstOrderTapeGradientFunctions`, the backward function built by
    this method accepts gradients for all of the outputs of the returned forward
    function, including side outputs.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.

    Returns:
      A tuple of (forward_function, backward_function):
        forward_function: Takes the same inputs as the inference function, but
          returns side outputs used by backward_function in addition to the
          inference function's outputs.
        backward_function: Takes side outputs from forward_function and
          gradients with respect to all of its outputs, real and side. Returns
          gradients with respect to the inputs.
    """
    outputs = []
    iteration_count = 0
    # First we need to figure out how many side outputs from the forward pass
    # will be required. We do this in a temporary graph to avoid actually
    # running multiple copies of the backward pass (one per _GradientsHelper
    # call).
    #
    # While computing gradients, the backward function captures Tensors from
    # the forward function. We add these as side outputs of the original
    # function. However, we then need to accept output gradients with respect
    # to these side outputs for higher order gradients to work. Thus we loop
    # until the number of outputs of the function stabilizes. Note that this
    # is only required for tape gradients, where we need to declare in advance
    # all of the forward op's outputs: symbolic gradients with tf.gradients
    # instead rely on regenerating backward functions when higher-order
    # gradients are requested.
    while (len(outputs) < len(self._func_graph.outputs)
           # It's possible for gradient generation to add new ops to the forward
           # pass. If all of the new outputs are non-trainable, there's no
           # reason to continue.
           and any(backprop_util.IsTrainable(output)
                   for output in self._func_graph.outputs[len(outputs):])):
      iteration_count += 1
      if iteration_count >= 20 and iteration_count % 5 == 0:
        new_op_with_trainable_output = None
        num_new_trainable_outputs = 0
        for output in self._func_graph.outputs[len(outputs):]:
          if backprop_util.IsTrainable(output):
            num_new_trainable_outputs += 1
            new_op_with_trainable_output = output.op
        logging.warning(
            ("Determining side outputs for the function '{}' is taking longer "
             "than expected ({} iterations, typically this converges in 5 or "
             "so). This could indicate that a gradient registration is adding "
             "new ops to the forward pass every time gradients are generated. "
             "{} new trainable output(s) were added this iteration, one from "
             "the following op:\n {}\nThis may indicate a TensorFlow bug, or "
             "an issue in a tf.custom_gradient.")
            .format(
                self._func_graph.name, iteration_count,
                num_new_trainable_outputs, new_op_with_trainable_output))
      outputs = list(self._func_graph.outputs)
      self._build_functions_for_outputs(
          outputs, inference_args, input_tangents)

    (forward_function, forward_graph,
     backward_function, output_indices, num_output_tangents) = (
         self._build_functions_for_outputs(
             outputs, inference_args, input_tangents))
    if (len(self._func_graph.outputs) > len(outputs)
        and any(backprop_util.IsTrainable(output)
                for output in self._func_graph.outputs[len(outputs):])):
      raise errors.InternalError(
          "Unexpectedly added new outputs to the forward function when "
          "building the backward function: "
          f"{self._func_graph.outputs[len(outputs):]}.")
    return (forward_function, forward_graph, backward_function, output_indices,
            num_output_tangents)


class _ForwardBackwardCall(object):
  """Holds the state of a function call between execution and recording."""

  __slots__ = [
      "_functions", "_inference_args", "_input_tangents", "_tape_watching"
  ]

  def __init__(self, functions, inference_args, input_tangents, tape_watching):
    """Collects information about the function call.

    Args:
      functions: An object which produces forward and backward functions, either
        a _DelayedRewriteGradientFunctions or a _TapeGradientFunctions object.
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.
      tape_watching: Boolean, with True indicating that recording is necessary.
    """
    self._functions = functions
    self._inference_args = inference_args
    self._input_tangents = input_tangents
    self._tape_watching = tape_watching

  def forward(self):
    """Builds or retrieves a forward function for this call."""
    forward_function = self._functions.forward(
        self._inference_args, self._input_tangents)
    return forward_function, self._inference_args + self._input_tangents

  def record(self, flat_outputs):
    """Given outputs from the execution of `forward`, records the operation."""
    if (self._tape_watching
        and not isinstance(flat_outputs, ops.Operation)
        and flat_outputs is not None):
      # We only record function calls which have outputs, and then only when a
      # tape is watching.
      self._functions.record(
          flat_outputs, self._inference_args, self._input_tangents)


# Sentinel value used by with ConcreteFunction's structured signature to
# indicate that a non-tensor parameter should use the value that was
# specified when the concrete function was created.
_BOUND_VALUE = object()


class ConcreteFunction(core.ConcreteFunction, trackable.Trackable):
  """A `tf.types.experimental.ConcreteFunction` created from `tf.function`."""

  def __init__(self,
               func_graph,
               attrs=None,
               shared_func_graph=True,
               function_spec=None):
    """Initialize a `ConcreteFunction`.

    Args:
      func_graph: An instance of FuncGraph: the function body to wrap.
      attrs: (optional) dict mapping names of attributes to their AttrValue
        values. Attributes in `attrs` will be included in this function's
        definition.
     shared_func_graph: If False, the ConcreteFunction takes ownership of
       `func_graph` and will break reference cycles when it is deleted. This
       makes the FuncGraph inoperable.
     function_spec: FunctionSpec for the original function.  If not specified,
       then this ConcreteFunction may only be called using the flat signature.

    Raises:
      ValueError: If number of input_placeholders is not equal to the number
        of function inputs.
    """
    # _arg_keywords and _num_positional_args define the flat signature.  They
    # are assigned after construction.
    self._arg_keywords = None
    self._num_positional_args = None

    self._func_graph = func_graph
    self._captured_inputs = self._func_graph.external_captures + self._func_graph.deferred_external_captures

    # function_spec defines the structured signature.
    self._set_function_spec(function_spec)

    if attrs and IMPLEMENTS_ATTRIBUTE_NAME in attrs:
      # The alternative is to silently drop "implements" tag
      # but it seems likely it would lead to hard to catch bugs.
      # Another alternative is to make func_body to preserve the order
      # of arguments if variables are present. Yet another option
      # is to automatically replace variables as arguments to functions
      # to v.read_value() whenever "implements" tag is present
      # Anytime we annotate existing function we probably want to wrap
      # it with safe read_value for backward compatibility.
      has_resource_vars = any(inp.dtype == dtypes.resource
                              for inp in self.inputs)

      assert not any((has_resource_vars, self._captured_inputs)), (
          'Function {name} has "{attr}={value}" attribute and thus can not '
          "depend on any tensors outside of its signature or modify variables. "
          "\n\nNote: variables are always captured and cause function "
          "re-tracing for every variable called.\n"
          "  inputs: {inputs}\n  captures: {captured}\n\n"
          "To pass a variable to such function use  "
          "use variable.read_value().".format(
              name=func_graph.name,
              attr=IMPLEMENTS_ATTRIBUTE_NAME,
              value=attrs[IMPLEMENTS_ATTRIBUTE_NAME],
              inputs=self.inputs,
              captured=self._captured_inputs))
    self._output_shapes = tuple(
        output.shape for output in self._func_graph.outputs)
    self._attrs = _parse_func_attrs(attrs or {})

    if shared_func_graph:
      self._garbage_collector = None
    else:
      self._garbage_collector = ConcreteFunctionGarbageCollector(func_graph)

    # Pairs of forward and backward functions used for computing gradients.
    #
    # These each get a reference to the FuncGraph deleter since they use the
    # FuncGraph directly.
    self._delayed_rewrite_functions = _DelayedRewriteGradientFunctions(
        func_graph, self._attrs, self._garbage_collector)
    self._first_order_tape_functions = {}
    self._higher_order_tape_functions = {}
    # Cache the inference function to avoid a (Python) function call when not
    # building gradients.
    self._inference_function = self._delayed_rewrite_functions.forward()

  def _set_function_spec(self, function_spec):
    """Enables the structured signature by supplying a function_spec."""
    self._function_spec = None
    self._pre_initialized_function_spec = function_spec
    self._initialize_function_spec()

  def _initialize_function_spec(self):
    """Updates `self._function_spec` to include varargs and bound variables.

    Adds new positional arguments for any varargs (i.e., for args that are
    in `structured_input_signature`, but not in the original fullargspec.args).

    Replaces `defaults` and `kwonlydefaults` with the `_BOUND_VALUE`, for
    all args and kwargs in `structured_input_signature`.

    Sets `varkw` and `varargs` to None.
    """
    if self._pre_initialized_function_spec is None:
      return  # e.g., SavedBareConcreteFunction doesn't have function_spec yet.
    assert not self._function_spec, "already initialized"
    function_spec = self._pre_initialized_function_spec
    args = function_spec.fullargspec.args
    arg_specs, kwarg_specs = self.structured_input_signature
    vararg_indices = range(len(function_spec.arg_names), len(arg_specs))
    fullargspec = tf_inspect.FullArgSpec(
        args=list(args) + ["<arg{}>".format(i + 1) for i in vararg_indices],
        varargs=None,
        varkw=None,
        defaults=[_BOUND_VALUE] * len(arg_specs),
        kwonlyargs=list(sorted(kwarg_specs)),
        kwonlydefaults=dict((k, _BOUND_VALUE) for k in kwarg_specs),
        annotations=function_spec.fullargspec.annotations)
    self._function_spec = FunctionSpec(
        fullargspec,
        function_spec.is_method,
        function_spec.input_signature,
        function_spec.is_pure,
        name=self._func_graph.name)

  @property
  def variables(self):
    """Sequence of variables for this function."""
    return tuple(self._func_graph.variables)

  @property
  def trainable_variables(self):
    """Sequence of trainable variables for this function."""
    return tuple(self._func_graph.trainable_variables)

  def __call__(self, *args, **kwargs):
    """Executes the wrapped function.

    ConcreteFunctions have two signatures:

    * The signature of the original function wrapped by this ConcreteFunction.
    * A flat signature, where each argument accepts a single Tensor.

    The original function signature is generally preferred, but the flat input
    signature is supported for backward compatibility.

    ### Original Function Signature

    When calling a ConcreteFunction with the signature of the original function,
    each argument must match the type or value that was used when the
    ConcreteFunction's graph was traced.  In particular:

    * Tensor arguments (including CompositeTensors, such as RaggedTensor) must
      have matching `TypeSpec`s.
    * Non-Tensor arguments (such as booleans or ints) must have equal values.
    * Nested arguments (such as lists, tuples, or dictionaries) must have the
      same nesting structure; and each nested value must have a matching type
      or value.

    The default value for any arguments that were traced with non-Tensor values
    is the value that was used in the trace.  Arguments that were traced with
    tensor arguments do not have a default value (even if the original function
    had a default value for that argument).

    ### Flat Signature

    When calling a ConcreteFunction with the flat signature, the arguments
    correspond to the flattened component tensors of the arguments that were
    used to construct the ConcreteFunction.  Parameter names are assigned based
    on `TensorSpec.name` (when specified) or the original argument names (with
    suffixes automatically added for nested arguments or composite tensors with
    multiple components).

    Args:
      *args: Positional arguments to the concrete function.
      **kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the TF function on the given Tensors.

    Raises:
      AssertionError: If this `ConcreteFunction` was not created through
        `get_concrete_function`.
      TypeError: If the arguments do not match the function's signature.
    """
    return self._call_impl(args, kwargs)

  def _call_impl(self, args, kwargs, cancellation_manager=None):
    """See `__call__` for details."""
    with trace.Trace(self._func_graph.name, tf_function_call="concrete"):
      # Construct the list of input tensors: check if the structured signature
      # applies first; and if not, then use the flat signature.
      if self._function_spec is not None:
        try:
          return self._call_with_structured_signature(args, kwargs,
                                                      cancellation_manager)
        except TypeError as structured_err:
          try:
            return self._call_with_flat_signature(args, kwargs,
                                                  cancellation_manager)
          except TypeError:
            raise structured_err

      return self._call_with_flat_signature(args, kwargs, cancellation_manager)

  def _call_with_flat_signature(self, args, kwargs, cancellation_manager):
    """Executes the wrapped function with the flat signature.

    Args:
      args: Positional arguments to the concrete function.
      kwargs: Keyword arguments to the concrete function.
      cancellation_manager: A `CancellationManager` that can be used to cancel
        function invocation.

    Returns:
      The result of applying the function on the Tensors/Variables contained in
      `args` and `kwargs`.
    Raises:
      TypeError: if `args` and `kwargs` do not match the flat signature of this
        `ConcreteFunction`.
    """
    if len(args) > self._num_positional_args:
      raise TypeError(
          f"{self._flat_signature_summary()} takes {self._num_positional_args} "
          f"positional arguments, got {len(args)}.")
    args = list(args)
    kwargs = dict(kwargs)
    for keyword in self._arg_keywords[len(args):]:
      try:
        args.append(kwargs.pop(compat.as_str(keyword)))
      except KeyError:
        specified_keywords = (
            list(self._arg_keywords[:len(args)]) + list(kwargs.keys()))
        missing_required_args = sorted(
            set(self._arg_keywords) - set(specified_keywords))
        raise TypeError(f"{self._flat_signature_summary()} missing required "
                        f"arguments: {', '.join(missing_required_args)}.")
    if kwargs:
      positional_arg_keywords = set(self._arg_keywords[:len(args)])
      for unused_key in kwargs:
        if unused_key in positional_arg_keywords:
          raise TypeError(f"{self._flat_signature_summary()} got two values "
                          f"for '{unused_key}'.")
      raise TypeError(f"{self._flat_signature_summary()} got unexpected "
                      f"keyword arguments: {', '.join(sorted(kwargs))}.")

    for i, arg in enumerate(args):
      if not isinstance(
          arg, (ops.Tensor, resource_variable_ops.BaseResourceVariable)):
        raise TypeError(f"{self._flat_signature_summary()}: expected argument "
                        f"#{i}(zero-based) to be a Tensor; "
                        f"got {type(arg).__name__} ({arg}).")
    return self._call_flat(args, self.captured_inputs, cancellation_manager)

  def _call_with_structured_signature(self, args, kwargs, cancellation_manager):
    """Executes the wrapped function with the structured signature.

    Args:
      args: Positional arguments to the concrete function.
      kwargs: Keyword arguments to the concrete function.
      cancellation_manager: A `CancellationManager` that can be used to cancel
        function invocation.

    Returns:
      The result of applying the function on the Tensors/Variables contained in
      `args` and `kwargs`.
    Raises:
      TypeError: if `args` and `kwargs` do not match the structured signature
        of this `ConcreteFunction`.
    """
    args, kwargs, _, filtered_flat_args = \
        self._function_spec.canonicalize_function_inputs(*args, **kwargs)
    self._structured_signature_check_missing_args(args, kwargs)
    self._structured_signature_check_unexpected_args(args, kwargs)
    self._structured_signature_check_arg_types(args, kwargs)
    return self._call_flat(
        filtered_flat_args,
        captured_inputs=self.captured_inputs,
        cancellation_manager=cancellation_manager)

  def _structured_signature_check_missing_args(self, args, kwargs):
    """Raises a TypeError if any args are missing."""
    arg_specs, kwarg_specs = self.structured_input_signature
    missing_arguments = []
    for i, (arg, spec) in enumerate(zip(args, arg_specs)):
      if arg is _BOUND_VALUE and _contains_type_spec(spec):
        missing_arguments.append(self._function_spec.arg_names[i])
    for (name, arg) in kwargs.items():
      if arg is _BOUND_VALUE and _contains_type_spec(kwarg_specs[name]):
        missing_arguments.append(name)
    if missing_arguments:
      raise TypeError(f"{self._structured_signature_summary()} missing "
                      "required arguments: "
                      f"{', '.join(sorted(missing_arguments))}.")

  def _structured_signature_check_unexpected_args(self, args, kwargs):
    """Raises a TypeError if there are any extra args."""
    arg_specs, kwarg_specs = self.structured_input_signature
    if len(args) > len(arg_specs):
      raise TypeError(
          f"{self._structured_signature_summary()} takes "
          f"{len(self._function_spec.arg_names)} positional arguments but got "
          f"{len(args)}.")
    if len(kwargs) > len(kwarg_specs):
      extra_args = set(kwargs) - set(kwarg_specs)
      raise TypeError(f"{self._structured_signature_summary()} got unexpected "
                      f"keyword arguments: {', '.join(extra_args)}.")

  def _structured_signature_check_arg_types(self, args, kwargs):
    """Raises a TypeError if any args have the wrong type."""
    # Check argument types
    arg_specs, kwarg_specs = self.structured_input_signature
    for i, (arg, spec) in enumerate(zip(args, arg_specs)):
      name = self._function_spec.arg_names[i]
      self._structured_signature_check_arg_type(arg, spec, name)
    for (name, arg) in kwargs.items():
      self._structured_signature_check_arg_type(arg, kwarg_specs[name], name)

  def _structured_signature_check_arg_type(self, arg, spec, name):
    """Raise TypeError if `arg`'s type doesn't match `spec`."""
    if arg is _BOUND_VALUE:
      return

    # Check the overall nested structure of the argument.
    try:
      nest.assert_same_structure(arg, spec, expand_composites=True)
    except (ValueError, TypeError):
      try:
        nest.assert_same_structure(arg, spec, expand_composites=False)
        expected, got = spec, arg
      except (ValueError, TypeError):
        expected, got = _structure_summary(spec), _structure_summary(arg)
      raise TypeError(f"{self._structured_signature_summary()}: argument "
                      f"{name} had incorrect type\n"
                      f"  expected: {expected}\n"
                      f"       got: {got}")

    # Check the type for each leaf in the nested structure.
    arg_pieces = nest.flatten(arg, expand_composites=True)
    spec_pieces = nest.flatten(spec, expand_composites=True)
    for (arg_piece, spec_piece) in zip(arg_pieces, spec_pieces):
      # TODO(mdan): Use consistent error messages.
      if isinstance(spec_piece, tensor_spec.DenseSpec):
        # TODO(edloper): Consider calling convert_to_tensor on non-tensor
        # values here.  That would match the behavior of
        # _call_concrete_function() in function_deserialization.py.  If
        # we do, then we need to change the nest assert_same_structure and
        # flatten calls above to use shallow variants.
        tensor_types = (ops.Tensor, resource_variable_ops.BaseResourceVariable)
        if not isinstance(arg_piece, tensor_types):
          raise TypeError(f"{self._structured_signature_summary()} expected a "
                          f"Tensor in {name}, but got "
                          f"{type(arg_piece).__name__} value {arg_piece}.")
      elif arg_piece is not _BOUND_VALUE:
        try:
          arg_matches_spec = bool(arg_piece == spec_piece)
        except (ValueError, TypeError):
          logging.vlog(1, "Error matching value with spec", exc_info=True)
          arg_matches_spec = False
        if not arg_matches_spec:
          raise TypeError(
              f"ConcreteFunction {self._structured_signature_summary()} was "
              f"constructed with {type(spec_piece).__name__} value "
              f"{spec_piece} in {name}, but was called with "
              f"{type(arg_piece).__name__} value {arg_piece}.")

  def _call_flat(self, args, captured_inputs, cancellation_manager=None):
    """Executes the wrapped function.

    Args:
      args: a list of Tensors or Variables. Arguments from the Python function
        should be filtered before calling this method: objects aside from
        Tensors, CompositeTensors, and Variables are ignored. Any
        CompositeTensors should be expanded before calling this method.
      captured_inputs: the captured inputs that are also part of the input args
        to the actual execution. By default, it should be self._captured_inputs.
      cancellation_manager: (Optional.) A `CancellationManager` that can be
        used to cancel function invocation.

    Returns:
      The result of applying the TF function to `args`.

    Raises:
      ValueError: If `args` contains anything other than Tensors or Variables.
    """
    ctx = context.context()
    executing_eagerly = ctx.executing_eagerly()

    # Copy saveable status of function's graph to current FuncGraph.
    default_graph = ops.get_default_graph()
    if default_graph.building_function and not self._func_graph.saveable:
      default_graph.mark_as_unsaveable(self._func_graph.saving_errors)

    if (tape.could_possibly_record() or
        hasattr(default_graph, "watch_variable")):
      for v in self._func_graph.variables:
        resource_variable_ops.variable_accessed(v)

    tensor_inputs = []
    variables_used = set([])
    for i, arg in enumerate(args):
      if isinstance(arg, resource_variable_ops.BaseResourceVariable):
        # We can pass a variable more than once, and in this case we need to
        # pass its handle only once.
        if id(arg.handle) in variables_used:
          continue
        resource_variable_ops.variable_accessed(arg)
        tensor_inputs.append(arg.handle)
        variables_used.add(id(arg.handle))
      elif isinstance(arg, ops.Tensor):
        tensor_inputs.append(arg)
        if not executing_eagerly:
          # If we're graph building, shape inference is on. We check for input
          # compatibility up front to avoid hard to debug incompatibilities
          # later.
          graph_input_shape = tensor_shape.TensorShape(
              self._func_graph.inputs[i].shape)
          if not graph_input_shape.is_compatible_with(arg.shape):
            if self._arg_keywords:
              arg_name = "'{}'".format(self._arg_keywords[i])
            else:
              arg_name = "with index {}".format(i)
            raise ValueError(
                f"The argument {arg_name} (value {arg}) is not compatible with "
                "the shape this function was traced with. Expected shape "
                f"{self._func_graph.inputs[i].shape}, but got shape "
                f"{arg.shape}.\n\nIf you called get_concrete_function, you may "
                "need to pass a tf.TensorSpec(..., shape=...) with a less "
                "specific shape, having None on axes which can vary.")
      else:
        raise ValueError(f"{i:d}-th input {arg} must be a Tensor, got "
                         f"{type(arg)} when calling {self._func_graph.name}.")
    args = tensor_inputs + captured_inputs
    possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
    if (possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE
        and executing_eagerly):
      # No tape is watching; skip to running the function.
      return self._build_call_outputs(self._inference_function.call(
          ctx, args, cancellation_manager=cancellation_manager))
    forward_backward = self._select_forward_and_backward_functions(
        args,
        possible_gradient_type,
        executing_eagerly)
    forward_function, args_with_tangents = forward_backward.forward()
    if executing_eagerly:
      flat_outputs = forward_function.call(
          ctx, args_with_tangents, cancellation_manager=cancellation_manager)
    else:
      with default_graph._override_gradient_function(  # pylint: disable=protected-access
          {"PartitionedCall": self._get_gradient_function(),
           "StatefulPartitionedCall": self._get_gradient_function()}):
        flat_outputs = forward_function.call(ctx, args_with_tangents)
    forward_backward.record(flat_outputs)
    return self._build_call_outputs(flat_outputs)

  def _experimental_with_cancellation_manager(self, cancellation_manager):
    """Returns a callable that invokes a cancellable version of this function.

    Args:
      cancellation_manager: A `CancellationManager` object that can be used to
        cancel function invocation.

    Returns:
      A callable with the same signature as this concrete function.
    """

    def cancellable_call(*args, **kwargs):
      return self._call_impl(
          args, kwargs, cancellation_manager=cancellation_manager)

    return cancellable_call

  @property
  def name(self):
    """`ConcreteFunction` name."""
    return self._delayed_rewrite_functions.forward().name

  @property
  def graph(self):
    """Returns the graph from which this function was constructed."""
    return self._func_graph

  @property
  def inputs(self):
    """Returns tensors in `self.graph` corresponding to arguments."""
    return self._func_graph.inputs

  @property
  def structured_input_signature(self):
    """Returns structured signature for this concrete function.

    Returns:
      A tuple `(args, kwargs)`, where:

        * `args` is a tuple that specifies the expected type or value each for
          positional argument.
        * `kwargs` is a dictionary that specifies the expected type or value
          for each keyword-only argument.

      The type or value for each argument is specified using one of the
      following:

        * A `tf.TypeSpec`, indicating that a Tensor or other TensorFlow-native
          value is expected.
        * A Python value, such as an integer, indicating that an equal value
          is expected.
        * A nested structure of `tf.TypeSpec`s and Python values, indicating
          that a corresponding nested structure is expected.
    """
    return self._func_graph.structured_input_signature

  @property
  def outputs(self):
    """Returns tensors in `self.graph` corresponding to returned tensors."""
    return self._func_graph.outputs

  @property
  def structured_outputs(self):
    """Returns outputs in `self.graph` as returned by the original function."""
    return self._func_graph.structured_outputs

  def set_external_captures(self, captures):
    """Updates the function capture values.

    The new values must have tensor types and shapes consistent with the
    original captures of the concrete function, but it is allowed to change a
    value captured with a deferred one and vice-versa.

    Args:
      captures: A list of tensors or closures. Tensors are value captures, and
        closures are call-time (deferred captures).
    """
    # TODO(wxinyi): 1. verify that the new captures' type spec is compatible
    # with the original's. However, doing so requires MirroredVariable captures
    # initialized. 2. replace the original/new captures/deferred
    # captures in the wrapped graph. Doing such for a capture-to-deferred
    # capture replacement requires more arguments than the deferred capture
    # itself, e.g. default value, spec.
    self._captured_inputs = captures

  def replace_capture_with_deferred_capture(self,
                                            tensor,
                                            closure,
                                            spec,
                                            placeholder=None,
                                            default_value=None):
    """Replaces existing capture `tensor` with a deferred capture `closure`.

    This API replaces the capture `tensor` from the concrete function's captured
    inputs list, and places the deferred capture `closure` in
    its spot so the order of captured inputs is preserved. This is important
    because the old `tensor` and the new `closure` will have the same internal
    placeholder, which can be passed through the `placeholder` argument, or
    skipped, in which case we find the placeholder from internal inputs by
    indexing `tensor` in the external captured inputs list. Thus, it is
    important that the new deferred capture has output spec (specified by the
    `spec` argument) compatible with the internal placeholder (`placeholder`)
    and the original capture (`tensor`).

    For example,

    ```python
    bool_captured_tensor = tf.constant(True)
    float_captured_tensor = tf.constant([3.], dtype=tf.float32)
    value = tf.constant([2.], dtype=tf.float32)

    @tf.function
    def fn():
      deferred_tensor = ops.get_default_graph().capture_call_time_value(
          lambda: value,
          tf.TensorSpec(shape=(1,), dtype=tf.float32))
      if bool_captured_tensor:
        return deferred_tensor
      else:
        return deferred_tensor + float_captured_tensor

    concrete_fn = fn.get_concrete_function()
    print(concrete_fn())  # tf.Tensor([2.], shape=(1,), dtype=float32)

    new_bool_captured_tensor = constant_op.constant(False)
    def bool_closure():
      return new_bool_captured_tensor

    concrete_fn.replace_capture_with_deferred_capture(
        bool_captured_tensor,
        bool_closure,
        spec=tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool))

    print(concrete_fn())  # tf.Tensor([5.], shape=(1,), dtype=float32)
    ```

    Args:
      tensor: Tensor already captured. This `tensor` should be listed in
        concrete_function.captured_inputs except when it's empty such as when
        the concrete function is restored from SavedModel.
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      placeholder: optional. The internal placeholder corresponding to the
        captured `tensor` and the new `closure`.
      default_value: optional value to use in environments that cannot safely
        evaluate closure.
    """
    capture_index = None
    for i, capture in enumerate(self._captured_inputs):
      if id(tensor) == id(capture):
        capture_index = i
        break

    if placeholder is None:
      if capture_index is None:
        raise ValueError(
            f"Did not find `tensor` argument {tensor} in the ConcreteFunction's"
            " captured inputs list, and did not receive a placeholder argument."
            " Thus we're unable to infer the internal placeholder. ")

      placeholder = self.inputs[-len(self._captured_inputs) + capture_index]

    if not (spec.is_compatible_with(tensor) or
            spec.is_compatible_with(placeholder)):
      raise ValueError(
          f"Attempting to substitute closure with spec {spec} that's "
          f"incompatible with the original capture {tensor} or the internal "
          f"placeholder {placeholder}.")

    self._func_graph.replace_capture_with_deferred_capture(
        tensor=tensor,
        closure=closure,
        spec=spec,
        placeholder=placeholder,
        default_value=default_value)

    if capture_index is not None:
      self._captured_inputs[capture_index] = closure

  @property
  def captured_inputs(self):
    """Returns external Tensors captured by this function.

    self.__call__(*args) passes `args + self.captured_inputs` to the function.
    """
    return nest.flatten(
        [x() if callable(x) else x for x in self._captured_inputs],
        expand_composites=True)

  @property
  def function_def(self):
    """Returns a `FunctionDef` object representing this function."""
    return self._delayed_rewrite_functions.forward().definition

  @property
  def output_shapes(self):
    """The function's output shapes."""
    return nest.map_structure(
        lambda x: getattr(x, "shape", tensor_shape.TensorShape(None)),
        composite_tensor.replace_composites_with_components(
            self._func_graph.structured_outputs),
        expand_composites=False)

  @property
  def output_dtypes(self):
    # TODO(akshayka): Consider removing this.
    return nest.map_structure(
        lambda x: x.dtype if x is not None else None,
        composite_tensor.replace_composites_with_components(
            self._func_graph.structured_outputs),
        expand_composites=False)

  def add_to_graph(self, g=None):
    """Registers the function, adds it to the graph g or default graph.

    Args:
      g: If specified, registers the function with this graph. Defaults to the
        current context (either the default graph or the eager context).
    """
    # If we are not executing eagerly, adds the function to default graph if no
    # graph is specified.
    # In case of eager execution, function definition gets added to context
    # during construction itself.

    if not context.executing_eagerly() and not g:
      g = ops.get_default_graph()
    self._delayed_rewrite_functions.forward().add_to_graph(g)

  def add_gradient_functions_to_graph(self, g=None):
    """Add forward/backward functions to graph `g` or the current context."""
    if not context.executing_eagerly() and not g:
      g = ops.get_default_graph()
    self._delayed_rewrite_functions.forward().add_to_graph(g)
    forward_function, backward_function = (
        self._delayed_rewrite_functions.forward_backward())
    forward_function.add_to_graph(g)
    backward_function.add_to_graph(g)

  def _get_gradient_function(self):
    """Returns gradient function. It will be lazily created at first call."""
    return self._delayed_rewrite_functions._rewrite_forward_and_call_backward  # pylint: disable=protected-access

  def _select_forward_and_backward_functions(
      self, args, possible_gradient_type, executing_eagerly):
    """Selects forward and backward functions based on the calling context.

    The forward function computes the "real" function outputs, `self._outputs`,
    and any extra values needed by the corresponding backward function.

    Args:
      args: A flat list of Tensors with all of the inputs to the forward
        function (including user-specified and captured inputs).
      possible_gradient_type: One of gradients_util.POSSIBLE_GRADIENT_TYPES_*.
      executing_eagerly: Boolean, the value of context.executing_eagerly().

    Returns:
      An object with a `forward` method returning a tuple of (forward_function :
      _EagerDefinedFunction, augmented_arguments : List), and a corresponding
      `record` method which takes outputs from the forward function and records
      the operation. forward_function should be called with augmented_arguments.
    """
    if executing_eagerly:
      input_tangents = forwardprop_util.pack_tangents(args)
    else:
      input_tangents = forwardprop_util.TangentInfo()
    need_gradients_for_jvps = tape.should_record_backprop(
        input_tangents.tangents)
    # Allows re-use of forward and backward function pairs depending on the
    # tapes and forward accumulators watching its inputs.
    cache_key = (need_gradients_for_jvps, input_tangents.indices)
    if (possible_gradient_type
        == gradients_util.POSSIBLE_GRADIENT_TYPES_FIRST_ORDER):
      if input_tangents.indices or executing_eagerly:
        # There is a single non-persistent tape active, so the user can only
        # request first-order gradients from a tape. We can spend less time
        # graph building since we know this.
        #
        # We may still end up computing higher-order gradients, but that'd be
        # through `tf.gradients`, which can re-write the forward pass and so
        # needs no preparation here.
        functions = self._first_order_tape_functions.get(cache_key, None)
        if functions is None:
          functions = _FirstOrderTapeGradientFunctions(
              self._func_graph, self._attrs, self._garbage_collector,
              forwardprop_input_indices=input_tangents.indices,
              delayed_rewrite_functions=self._delayed_rewrite_functions,
              need_gradients_for_jvps=need_gradients_for_jvps)
          self._first_order_tape_functions[cache_key] = functions
        return _ForwardBackwardCall(
            functions, args, input_tangents.tangents, tape_watching=True)
      else:
        # We can avoid computing second-order gradients in some cases by doing a
        # delayed rewrite when graph building. Since we know we'll only compute
        # first-order tape gradients, the delayed rewrite is safe: we won't need
        # to tell the tape about side outputs.
        #
        # TODO(allenl): This case is really dirty. It would be better if we
        # could temporarily pop all of the current tapes to avoid
        # accidentally taking second-order gradients.
        return _ForwardBackwardCall(
            self._delayed_rewrite_functions, args, input_tangents.tangents,
            tape_watching=True)
    elif (possible_gradient_type
          == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER):
      # Either there's a persistent tape watching, or there are multiple nested
      # tapes. Either way, the user may request higher-order gradients. We'll
      # spend a bit more time and make sure higher-order gradients are correct.
      functions = self._higher_order_tape_functions.get(
          cache_key, None)
      if functions is None:
        functions = _HigherOrderTapeGradientFunctions(
            self._func_graph, self._attrs, self._garbage_collector,
            forwardprop_input_indices=input_tangents.indices,
            delayed_rewrite_functions=self._delayed_rewrite_functions,
            need_gradients_for_jvps=need_gradients_for_jvps)
        self._higher_order_tape_functions[cache_key] = functions
      return _ForwardBackwardCall(functions, args, input_tangents.tangents,
                                  tape_watching=True)
    # else possible_gradient_type == POSSIBLE_GRADIENT_TYPES_NONE, meaning no
    # tape is recording.
    return _ForwardBackwardCall(
        self._delayed_rewrite_functions, args, input_tangents.tangents,
        tape_watching=False)

  def _build_call_outputs(self, result):
    """Maps the fdef output list to actual output structure.

    Args:
      result: Output lists defined by FunctionDef.
    Returns:
      The actual call output.
    """
    # TODO(jlchu): call C++ version in function.cc when speed is improved
    if self._func_graph.structured_outputs is None:
      return result

    # Replace outputs with results, skipping over any 'None' values.
    outputs_list = nest.flatten(
        self._func_graph.structured_outputs, expand_composites=True)
    j = 0
    for i, o in enumerate(outputs_list):
      if o is not None:
        handle_data_util.copy_handle_data(self.outputs[j], result[j])
        outputs_list[i] = result[j]
        j += 1
    ret = nest.pack_sequence_as(self._func_graph.structured_outputs,
                                outputs_list, expand_composites=True)
    return ret

  @property
  def _as_name_attr_list(self):
    """Returns a `NameAttrList` representing this function."""
    ret = attr_value_pb2.NameAttrList(name=self.name)
    for name, value in self._attrs.items():
      ret.attr[name].CopyFrom(value)
    return ret

  def _structured_signature_summary(self, default_values=False):
    """Returns a string summarizing this function's structured signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
    # Note: we can't just use self._funcion_spec.signature_summary(), because
    # that would show "_BOUND_VALUE" as the default value for all arguments.
    assert self._function_spec is not None
    arg_specs, kwarg_specs = self.structured_input_signature
    arg_names = list(self._function_spec.arg_names)

    # If an explicit input_signature is provided to @tf.function, then any
    # arguments with defaults that are not covered by that explicit signature
    # are simply dropped from the signature.
    # TODO(b/159639913) Look into whether dropping arguments with default values
    # from the signature is the right thing to do.
    arg_names = arg_names[:len(arg_specs)]

    if default_values:
      for i in range(len(arg_names)):
        if not _contains_type_spec(arg_specs[i]):
          arg_names[i] += "={}".format(arg_specs[i])
    if kwarg_specs:
      arg_names.append("*")
      for name, spec in kwarg_specs.items():
        arg_names.append(name)
        if default_values and not _contains_type_spec(spec):
          arg_names[-1] += "={}".format(spec)
    signature = f"{self._func_graph.name}({', '.join(arg_names)})"

    return signature

  def _flat_signature_summary(self):
    """Returns a string summarizing this function's flat signature."""
    assert self._arg_keywords is not None
    assert self._num_positional_args is not None
    arg_names = self._arg_keywords
    if self._num_positional_args > len(arg_names):
      arg_names.extend(
          "<arg{}>".format(i + 1)
          for i in range(len(arg_names), self._num_positional_args))
    return f"{self._func_graph.name}({', '.join(arg_names)})"

  def pretty_printed_signature(self, verbose=True):
    """Returns a string summarizing the signature of this concrete function."""
    if not verbose:
      return self._structured_signature_summary(default_values=True)

    def pretty_print_spec(spec):
      """Returns a string describing the spec for a single argument."""
      if isinstance(spec, tensor_spec.TensorSpec):
        return "{} Tensor, shape={}".format(spec.dtype.name, spec.shape)
      elif nest.is_nested(spec):
        pieces = nest.flatten(spec, expand_composites=False)
        markers = [_Marker("<{}>".format(i + 1)) for i in range(len(pieces))]
        structure = nest.pack_sequence_as(spec, markers)
        # Ensure dictionaries are sorted by key (for determinism)
        result = pprint.pformat(structure, width=10000)
        for (marker, piece) in zip(markers, pieces):
          result += "\n      {}: {}".format(marker, pretty_print_spec(piece))
        return result
      else:
        return repr(spec)

    lines = [self._structured_signature_summary(default_values=True)]
    arg_specs, kwarg_specs = self.structured_input_signature
    names = list(self._function_spec.arg_names)

    # If an explicit input_signature is provided to @tf.function, then any
    # arguments with defaults that are not covered by that explicit signature
    # are simply dropped from the signature.
    # TODO(b/159639913) Look into whether dropping arguments with default values
    # from the signature is the right thing to do.

    # Note: we can skip bound args, since we already displayed their bound
    # value in the signature summary.
    arg_details = []
    for (name, spec) in zip(names[:len(arg_specs)], list(arg_specs)):
      if _contains_type_spec(spec):
        arg_details.append("    {}: {}".format(name, pretty_print_spec(spec)))

    if kwarg_specs:
      for kwarg in sorted(kwarg_specs):
        spec = kwarg_specs[kwarg]
        if _contains_type_spec(spec):
          arg_details.append("    {}: {}".format(
              kwarg, pretty_print_spec(spec)))

    if arg_details:
      lines.append("  Args:")
      lines.extend(arg_details)
    lines.append("  Returns:")

    def spec_from_value(value):
      # For loaded function, structured_outputs are already specs.
      if isinstance(value, type_spec.TypeSpec):
        return value
      return type_spec.type_spec_from_value(value)

    lines.append("    {}".format(
        pretty_print_spec(
            nest.map_structure(spec_from_value, self.structured_outputs))))

    return "\n".join(lines)

  def __repr__(self):
    if self._function_spec is not None:
      return "<ConcreteFunction {} at 0x{:X}>".format(
          self.pretty_printed_signature(verbose=False), id(self))
    elif not (self._num_positional_args is None or self._arg_keywords is None):
      return "<ConcreteFunction {} at 0x{:X}>".format(
          self._flat_signature_summary(), id(self))
    else:
      return object.__repr__(self)

  def __str__(self):
    if self._function_spec is not None:
      return "ConcreteFunction {}".format(self.pretty_printed_signature())
    else:
      return self.__repr__()


_pywrap_utils.RegisterType("Tensor", ops.Tensor)
_pywrap_utils.RegisterType("EagerTensor", ops.EagerTensor)
_pywrap_utils.RegisterType("IndexedSlices", indexed_slices.IndexedSlices)


def _deterministic_dict_values(dictionary):
  return tuple(dictionary[key] for key in sorted(dictionary))


class FunctionSpec(object):
  """Specification of how to bind arguments to a function."""

  @staticmethod
  def from_function_and_signature(python_function,
                                  input_signature,
                                  is_pure=False,
                                  experimental_follow_type_hints=False,
                                  jit_compile=None):
    """Create a FunctionSpec instance given a python function and signature.

    Args:
      python_function: a function to inspect
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
      will be converted to tensors and no variable changes allowed.
      experimental_follow_type_hints: see `tf.function`
      jit_compile: see `tf.function`

    Returns:
      instance of FunctionSpec
    """
    fullargspec = tf_inspect.getfullargspec(python_function)
    if (input_signature is not None and
        set(fullargspec.kwonlyargs) - set(fullargspec.kwonlydefaults or ())):
      nodefault_kwonlyargs = set(fullargspec.kwonlyargs)
      if fullargspec.kwonlydefaults is not None:
        nodefault_kwonlyargs -= set(fullargspec.kwonlydefaults)
      raise ValueError("Cannot build TF function from "
                       f"{python_function.__name__}: keyword-only arguments "
                       "must have default values when input_signature is "
                       "provided. Got keyword-only arguments without default "
                       f"values: {sorted(nodefault_kwonlyargs)}.")

    # Checks if the `fullargspec` contains self or cls as its first argument.
    is_method = tf_inspect.isanytargetmethod(python_function)

    # Treat a wrapped partial function as a special case. For all arguments that
    # were overridden with keywords in the partial:
    #   - remove the corresponding arguments,
    #   - remove the corresponding keywords.
    _, unwrapped = tf_decorator.unwrap(python_function)
    if isinstance(unwrapped, functools.partial):
      # Also consider the Python3 case with kwonlydefaults.
      if fullargspec.defaults or fullargspec.kwonlydefaults:
        new_defaults = fullargspec.defaults
        new_args = fullargspec.args
        if fullargspec.defaults:
          # To be able to canonicalize the function properly, we want to ignore
          # default values that are overridden via a partial kwarg. For example:
          #
          #   def func(a, b, c, d=5, e=7):
          #     return a, b, c, d, e
          #   p_func = tf.function(functools.partial(func, 10, e=9))
          #
          # Here we want to drop from the defaults the parameter `e`. If we
          # forwarded the call to the partial function with a default for `e`
          # we would get an error for passing two values for one parameter.
          #
          # Note that this has a limitation: we can only override parameters at
          # the end of the parameter list.
          #
          # In this case we want to end up with 3 arguments (b, c, d) and 1
          # default value (5). We do this by constructing a mask where 0 stands
          # for a value that was overridden by a partial kwarg. The seemingly
          # complicated logic below does just that - for arguments (b, c, d, e)
          # we would get a mask (1, 1, 1, 0).
          old_args = fullargspec.args
          old_defaults = fullargspec.defaults

          no_default = object()
          num_args_without_defaults = len(old_args) - len(old_defaults)
          left_padding = tuple([no_default] * num_args_without_defaults)

          args_with_defaults = zip(old_args, left_padding + old_defaults)

          # Create a mask where 0 stands for args that had a partial kwarg
          # defined.
          non_keyword_defaults_mask = [
              0 if key in unwrapped.keywords else 1 for key in old_args
          ]
          # Keep only arguments and defaults that were not kwargs of partial.
          new_args_with_defaults = list(
              itertools.compress(args_with_defaults, non_keyword_defaults_mask))
          # Keep all args.
          new_args = [arg for arg, _ in new_args_with_defaults]
          # Keep only real default values.
          new_defaults = [
              default for _, default in new_args_with_defaults
              if default is not no_default
          ]
        fullargspec = tf_inspect.FullArgSpec(
            args=new_args,
            varargs=fullargspec.varargs,
            varkw=fullargspec.varkw,
            defaults=new_defaults,
            kwonlyargs=[],
            kwonlydefaults={},
            annotations=fullargspec.annotations)

    # Get the function's name.  Remove functools.partial wrappers if necessary.
    while isinstance(python_function, functools.partial):
      python_function = python_function.func
    name = getattr(python_function, "__name__", "f")

    return FunctionSpec(
        fullargspec,
        is_method,
        input_signature,
        is_pure=is_pure,
        jit_compile=jit_compile,
        experimental_follow_type_hints=experimental_follow_type_hints,
        name=name)

  def __init__(self,
               fullargspec,
               is_method,
               input_signature,
               is_pure=False,
               experimental_follow_type_hints=False,
               name=None,
               jit_compile=None):
    """Constructs a FunctionSpec describing a python function.

    Args:
      fullargspec: `tf_inspect.FullArgSpec` object describing the function.
      is_method: True if the function is a method.
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      experimental_follow_type_hints: see `tf.function`.
      name: Name of the function
      jit_compile: see `tf.function`.
    """
    self._fullargspec = fullargspec
    self._is_method = is_method
    self._is_pure = is_pure
    self._jit_compile = jit_compile
    self._experimental_follow_type_hints = experimental_follow_type_hints

    # TODO(edloper): Include name when serializing for SavedModel?
    self._name = name or "f"

    if self._is_method:
      # Remove `self`: default arguments shouldn't be matched to it.
      # TODO(b/127938157): Should this error out if there is no arg to
      # be removed?
      args = fullargspec.args[1:]
    else:
      args = fullargspec.args

    # A cache mapping from argument name to index, for canonicalizing
    # arguments that are called in a keyword-like fashion.
    self._args_to_indices = {arg: i for i, arg in enumerate(args)}
    self._arg_names = args

    # A cache mapping from arg index to default value, for canonicalization.
    default_values = fullargspec.defaults
    offset = len(args) - len(default_values or [])
    self._arg_indices_to_default_values = {
        offset + index: default
        for index, default in enumerate(default_values or [])
    }
    self._arg_indices_no_default_values = set(range(len(args))) - set(
        self._arg_indices_to_default_values)
    if input_signature is None:
      self._input_signature = None
    else:
      self._input_signature = tuple(input_signature)
      self._flat_input_signature = tuple(nest.flatten(input_signature,
                                                      expand_composites=True))

  @property
  def fullargspec(self):
    return self._fullargspec

  @property
  def is_method(self):
    return self._is_method

  @property
  def args_to_indices(self):
    return self._args_to_indices

  @property
  def kwargs_to_include(self):
    return self._kwargs_to_include

  @property
  def input_signature(self):
    return self._input_signature

  @property
  def flat_input_signature(self):
    return self._flat_input_signature

  @property
  def is_pure(self):
    return self._is_pure

  @property
  def jit_compile(self):
    return self._jit_compile

  @property
  def arg_names(self):
    return self._arg_names

  @property
  def vararg_name(self):
    return self._fullargspec.varargs

  @property
  def varkw_name(self):
    return self._fullargspec.varkw

  def signature_summary(self, default_values=False):
    """Returns a string summarizing this function's signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
    args = list(self._arg_names)
    if default_values:
      for (i, default) in self._arg_indices_to_default_values.items():
        args[i] += "={}".format(default)
    if self._fullargspec.kwonlyargs:
      args.append("*")
      for arg_name in self._fullargspec.kwonlyargs:
        args.append(arg_name)
        if default_values and arg_name in self._fullargspec.kwonlydefaults:
          args[-1] += "={}".format(self._fullargspec.kwonlydefaults[arg_name])
    return f"{self._name}({', '.join(args)})"

  def _to_tensor_or_tensor_spec(self, x):
    return (x if isinstance(x, (ops.Tensor, tensor_spec.TensorSpec))
            else ops.convert_to_tensor(x))

  def _convert_variables_to_tensors(self, args, kwargs):
    args = [self._to_tensor_or_tensor_spec(x) for x in args]
    kwargs = {kw: self._to_tensor_or_tensor_spec(x)
              for kw, x in kwargs.items()}
    return tuple(args), kwargs

  def _convert_annotated_args_to_tensors(self, args, kwargs):
    """Attempts to autobox arguments annotated as tf.Tensor."""
    if self.input_signature is not None:
      return

    args = list(args)
    for i, arg in enumerate(args):
      # See
      # https://docs.python.org/3/library/inspect.html#inspect.getfullargspec
      if i < len(self._fullargspec.args):
        annotation_key = self._fullargspec.args[i]
      else:
        annotation_key = self._fullargspec.varargs
      arg_annotation = self._fullargspec.annotations.get(annotation_key, None)

      # TODO(rahulkamat): Change to TensorLike (here ans below)
      if arg_annotation == ops.Tensor:
        args[i] = self._to_tensor_or_tensor_spec(arg)

    for kw, v in kwargs.items():
      if kw in self._fullargspec.kwonlyargs or kw in self._fullargspec.args:
        annotation_key = kw
      else:
        annotation_key = self._fullargspec.varkw
      kwarg_annotation = self._fullargspec.annotations.get(annotation_key, None)
      if kwarg_annotation == ops.Tensor:
        kwargs[kw] = self._to_tensor_or_tensor_spec(v)
    return tuple(args), kwargs

  def _validate_inputs(self, flat_inputs):
    """Raises an error if inputs contain illegal values."""
    for inp in flat_inputs:
      # TODO(b/183107079): Allow these once they're handled properly.
      if isinstance(inp, weakref.ref):
        raise ValueError(
            f"weakref input {inp} not supported for function {self._name}")

  def canonicalize_function_inputs(self, *args, **kwargs):
    """Canonicalizes `args` and `kwargs`.

    Canonicalize the inputs to the Python function using a `FunctionSpec`
    instance. In particular, we parse the varargs and kwargs that the
    original function was called with into a tuple corresponding to the
    Python function's positional (named) arguments and a dictionary
    corresponding to its kwargs.  Missing default arguments are added.

    If this `FunctionSpec` has an input signature, then it is used to convert
    arguments to tensors; otherwise, any inputs containing numpy arrays are
    converted to tensors.

    Additionally, any inputs containing numpy arrays are converted to Tensors.

    Args:
      *args: The varargs this object was called with.
      **kwargs: The keyword args this function was called with.

    Returns:
      A canonicalized ordering of the inputs, as well as full and filtered
      (Tensors and Variables only) versions of their concatenated flattened
      representations, represented by a tuple in the form (args, kwargs,
      flat_args, filtered_flat_args). Here: `args` is a full list of bound
      arguments, and `kwargs` contains only true keyword arguments, as opposed
      to named arguments called in a keyword-like fashion.

    Raises:
      ValueError: If a keyword in `kwargs` cannot be matched with a positional
        argument when an input signature is specified, or when the inputs
        do not conform to the input signature.
    """
    if self._is_pure:
      args, kwargs = self._convert_variables_to_tensors(args, kwargs)
    if self._experimental_follow_type_hints:
      args, kwargs = self._convert_annotated_args_to_tensors(args, kwargs)
    # Pre-calculate to reduce overhead
    arglen = len(args)
    if self._input_signature is not None:
      if arglen > len(self._input_signature):
        raise TypeError(f"{self.signature_summary()} specifies "
                        f"{len(self._input_signature)} positional arguments, "
                        f"but got {arglen}.")
      for arg in six.iterkeys(kwargs):
        index = self._args_to_indices.get(arg, None)
        if index is None:
          raise TypeError(f"{self.signature_summary()} got unexpected keyword "
                          f"argument `{arg}`.")
        if index >= len(self._input_signature):
          raise TypeError(
              f"{self.signature_summary()} got keyword argument `{arg}` that "
              "was not included in input_signature.")

    if not kwargs:
      inputs = args
      if self._arg_indices_to_default_values:
        try:
          inputs += tuple(self._arg_indices_to_default_values[i]
                          for i in range(arglen, len(self._arg_names)))
        except KeyError:
          missing_args = [
              self._arg_names[i]
              for i in range(arglen, len(self._arg_names))
              if i not in self._arg_indices_to_default_values
          ]
          raise TypeError(f"{self.signature_summary()} missing required "
                          f"arguments: {', '.join(missing_args)}.")

      if self._fullargspec.kwonlydefaults:
        kwargs.update(self._fullargspec.kwonlydefaults)
    else:
      # Maps from index of arg to its corresponding value, according to `args`
      # and `kwargs`; seeded with the default values for the named args that
      # aren't in `args`.
      arg_indices_to_values = {
          index: default for index, default in six.iteritems(
              self._arg_indices_to_default_values) if index >= arglen
      }
      consumed_args = []
      missing_arg_indices = self._arg_indices_no_default_values - set(
          range(arglen))
      for arg, value in six.iteritems(kwargs):
        index = self._args_to_indices.get(arg, None)
        if index is not None:
          if index < arglen:
            raise TypeError(f"{self.signature_summary()} got two values for "
                            f"{arg!r}.")
          arg_indices_to_values[index] = value
          # These arguments in 'kwargs' might also belong to
          # positional arguments
          missing_arg_indices.discard(index)
          consumed_args.append(arg)
      for arg in consumed_args:
        # After this loop, `kwargs` will only contain keyword_only arguments,
        # and all positional_or_keyword arguments have been moved to `inputs`.
        kwargs.pop(arg)
      inputs = args + _deterministic_dict_values(arg_indices_to_values)
      # Exclude positional args with values
      if missing_arg_indices:
        missing_args = [self._arg_names[i] for i in sorted(missing_arg_indices)]
        if len(missing_args) == 1:
          raise TypeError(f"{self.signature_summary()} missing 1 required "
                          f"argument: {missing_args[0]}.")
        else:
          raise TypeError(f"{self.signature_summary()} missing required "
                          f"arguments: {', '.join(missing_args)}.")

      if kwargs and self._input_signature is not None:
        raise TypeError("Keyword arguments are not supported when "
                        "input_signature is provided. Signature: "
                        f"{self.signature_summary()}. Keyword arguments: "
                        f"{kwargs}.")

      if self._fullargspec.kwonlydefaults:
        for (kwarg, default) in self._fullargspec.kwonlydefaults.items():
          kwargs.setdefault(kwarg, default)

    if self._input_signature is None:
      inputs, flat_inputs, filtered_flat_inputs = _convert_numpy_inputs(inputs)
      kwargs, flat_kwargs, filtered_flat_kwargs = _convert_numpy_inputs(kwargs)
      flat_inputs += flat_kwargs
      filtered_flat_inputs += filtered_flat_kwargs
    else:
      inputs, flat_inputs, filtered_flat_inputs = _convert_inputs_to_signature(
          inputs, self._input_signature, self._flat_input_signature)

    self._validate_inputs(flat_inputs)

    return inputs, kwargs, flat_inputs, filtered_flat_inputs


def _convert_numpy_inputs(inputs):
  """Convert numpy array inputs to tensors."""
  # We assume that any CompositeTensors have already converted their components
  # from numpy arrays to Tensors, so we don't need to expand composites here for
  # the numpy array conversion. Instead, we do so because the flattened inputs
  # are eventually passed to ConcreteFunction()._call_flat, which requires
  # expanded composites.
  flat_inputs = nest.flatten(inputs, expand_composites=True)

  # Check for NumPy arrays in arguments and convert them to Tensors.
  # TODO(nareshmodi): Skip ndarray conversion to tensor altogether, perhaps
  # finding a way to store them directly in the cache key (currently not
  # possible since ndarrays are not hashable).
  need_packing = False
  filtered_flat_inputs = []
  for index, value in enumerate(flat_inputs):
    if isinstance(value,
                  (ops.Tensor, resource_variable_ops.BaseResourceVariable)):
      filtered_flat_inputs.append(value)
    elif hasattr(value, "__array__") and not (
        hasattr(value, "_should_act_as_resource_variable") or
        isinstance(value, (np.str_, type, composite_tensor.CompositeTensor))):
      # This case is equivalent to _is_ndarray(value) == True
      a = value.__array__()
      if not isinstance(a, np.ndarray):
        raise TypeError(f"The output of __array__ must be an np.ndarray, "
                        f"got {type(a)} from {value}.")
      flat_inputs[index] = constant_op.constant(a)
      filtered_flat_inputs.append(flat_inputs[index])
      need_packing = True
  if need_packing:
    return (nest.pack_sequence_as(
        structure=inputs, flat_sequence=flat_inputs,
        expand_composites=True), flat_inputs, filtered_flat_inputs)
  else:
    return inputs, flat_inputs, filtered_flat_inputs


def _convert_inputs_to_signature(inputs, input_signature, flat_input_signature):
  """Convert inputs to pass into a function with an explicit signature."""

  def format_error_message(inputs, input_signature):
    return ("  inputs: (\n" + "    " + ",\n    ".join(str(i) for i in inputs) +
            ")\n" + "  input_signature: (\n" + "    " +
            ",\n    ".join(str(i) for i in input_signature) + ")")

  try:
    flatten_inputs = nest.flatten_up_to(
        input_signature,
        inputs[:len(input_signature)],
        expand_composites=True,
        check_types=False)  # lists are convert to tuples for `tf.data`.
  except ValueError:
    raise ValueError("Structure of Python function inputs does not match "
                     "input_signature:\n"
                     f"{format_error_message(inputs, input_signature)}.")

  need_packing = False
  for index, (value, spec) in enumerate(zip(flatten_inputs,
                                            flat_input_signature)):
    if (isinstance(spec, tensor_spec.TensorSpec) and
        not _pywrap_utils.IsTensor(value)):
      try:
        flatten_inputs[index] = ops.convert_to_tensor(
            value, dtype_hint=spec.dtype)
        need_packing = True
      except ValueError:
        raise ValueError("When input_signature is provided, all inputs to "
                         "the Python function must be convertible to "
                         "tensors:\n"
                         f"{format_error_message(inputs, input_signature)}.")

  if any(not spec.is_compatible_with(other) for spec, other in zip(
      flat_input_signature,
      flatten_inputs)):
    raise ValueError("Python inputs incompatible with input_signature:\n"
                     f"{format_error_message(inputs, input_signature)}.")

  if need_packing:
    inputs = nest.pack_sequence_as(
        structure=input_signature,
        flat_sequence=flatten_inputs,
        expand_composites=True)

  flat_inputs = nest.flatten(inputs, expand_composites=True)

  return (inputs, flat_inputs, [
      t for t in flat_inputs
      if isinstance(t, (ops.Tensor, resource_variable_ops.BaseResourceVariable))
  ])


# TODO(mdan): Refactor this and clarify relationship with def_function.Function.
# Right now, def_function.Function is the higher level implementation.
class Function(object):
  """Wrapper class for the graph functions defined for a Python function.

  See the documentation for `defun` for more information on the semantics of
  defined functions.

  `Function` class is thread-compatible meaning that minimal usage of defuns
  (defining and calling) is thread-safe, but if users call other methods or
  invoke the base `python_function` themselves, external synchronization is
  necessary.
  In addition, Function is not reentrant, so recursive functions need to call
  the wrapped function, not the wrapper.
  """

  def __init__(self,
               python_function,
               name,
               input_signature=None,
               attributes=None,
               autograph=True,
               autograph_options=None,
               experimental_relax_shapes=False,
               capture_by_value=None,
               jit_compile=None,
               experimental_follow_type_hints=False):
    """Initializes a `Function`.

    Args:
      python_function: the function to be wrapped.
      name: the name given to it.
      input_signature: a possibly nested sequence of `TensorSpec` objects
        specifying the input signature of this function. If `None`, a separate
        function is instantiated for each inferred input signature.
      attributes: dict, extra keyword arguments that will be added as attribute
        of the function.
      autograph: whether to use autograph to compile
        `python_function`. See https://www.tensorflow.org/guide/autograph for
        more information.
      autograph_options: Experimental knobs to control behavior
        `when autograph=True`. See https://www.tensorflow.org/guide/autograph
        for more information.
      experimental_relax_shapes: When true, argument shapes may be relaxed to
        avoid unnecessary retracing.
      capture_by_value: Experimental. Whether to capture resource variables by
        value or reference. If None, will inherit from a parent context or
        default to False.
      jit_compile: Force-compile the function with XLA, cf.
        def_function.Function doc on jit_compile.
      experimental_follow_type_hints: See the documentation for `tf.function`.

    Raises:
      ValueError: if `input_signature` is not None and the `python_function`'s
        argspec has keyword arguments.
    """
    self._python_function = python_function
    pure_function = attributes and IMPLEMENTS_ATTRIBUTE_NAME in attributes
    self._function_spec = FunctionSpec.from_function_and_signature(
        python_function,
        input_signature,
        is_pure=pure_function,
        experimental_follow_type_hints=experimental_follow_type_hints)
    self._name = name
    self._autograph = autograph
    self._autograph_options = autograph_options
    self._experimental_relax_shapes = experimental_relax_shapes
    self._function_cache = function_cache.FunctionCache()
    self._function_attributes = attributes or {}
    self._capture_by_value = capture_by_value
    self.tracing_count = 0

    self._lock = threading.RLock()
    # _descriptor_cache is a of instance of a class to an instance-specific
    # `Function`, used to make sure defun-decorated methods create different
    # functions for each instance.
    self._descriptor_cache = weakref.WeakKeyDictionary()
    self._jit_compile = jit_compile
    self._experimental_follow_type_hints = experimental_follow_type_hints

  def __call__(self, *args, **kwargs):
    """Calls a graph function specialized to the inputs."""
    with self._lock:
      (graph_function,
       filtered_flat_args) = self._maybe_define_function(args, kwargs)
    return graph_function._call_flat(
        filtered_flat_args, captured_inputs=graph_function.captured_inputs)  # pylint: disable=protected-access

  @property
  def python_function(self):
    """Returns the wrapped Python function."""
    return self._python_function  # pylint: disable=protected-access

  @property
  def function_spec(self):
    return self._function_spec

  @property
  def input_signature(self):
    """Returns the input signature."""
    return self._function_spec.input_signature

  @property
  def flat_input_signature(self):
    """Returns the flattened input signature."""
    return self._function_spec.flat_input_signature

  def _get_concrete_function_internal_garbage_collected(self, *args, **kwargs):
    """Returns a concrete function which cleans up its graph function."""
    if self.input_signature:
      args, kwargs = None, None
    with self._lock:
      graph_function, _ = self._maybe_define_function(args, kwargs)
    return graph_function

  def _get_concrete_function_internal(self, *args, **kwargs):
    """Bypasses error checking when getting a graph function."""
    graph_function = self._get_concrete_function_internal_garbage_collected(
        *args, **kwargs)
    # We're returning this concrete function to someone, and they may keep a
    # reference to the FuncGraph without keeping a reference to the
    # ConcreteFunction object. So we won't clean up the reference cycles
    # manually and instead will leave them to Python's garbage collector.
    graph_function._garbage_collector.release()  # pylint: disable=protected-access
    return graph_function

  def _get_concrete_function_garbage_collected(self, *args, **kwargs):
    """Returns a `ConcreteFunction` specialized to inputs and execution context.

    Unlike `get_concrete_function(...)`, the graph will be deleted when the
    returned function is deleted.  It's useful to avoid creating a reference
    cycle when you know for sure that the graph will be no longer used without
    the returned function.

    Args:
      *args: inputs to specialize on.
      **kwargs: inputs to specialize on.
    """
    if self.input_signature:
      if kwargs:
        raise ValueError("Cannot define a TensorFlow function from a Python "
                         "function with keyword arguments when "
                         "input_signature is provided, got keyword arguments "
                         f"({kwargs}) with input_signature "
                         f"({self.input_signature}).")
      if args:
        # If args are provided, they must match the input signature.
        if not is_same_structure(self.input_signature, args):
          raise ValueError("Structure of Python function inputs does not match "
                           f"input_signature: inputs ({args}), "
                           f"input_signature ({self.input_signature}).")
        flat_inputs = nest.flatten(args, expand_composites=True)
        if any(not isinstance(arg, (ops.Tensor, tensor_spec.DenseSpec,
                                    resource_variable_ops.BaseResourceVariable))
               for arg in flat_inputs):
          raise ValueError("When input_signature is provided, all inputs to "
                           "the Python function must be Tensors, Variables, "
                           "tf.TensorSpec or tf.VariableSpec objects.")
        if any(not spec.is_compatible_with(other)
               for spec, other in zip(self.flat_input_signature, flat_inputs)):
          raise ValueError("Python inputs incompatible with input_signature: "
                           f"inputs ({args}), input_signature "
                           f"({self.input_signature}).")
      args, kwargs = None, None
    with self._lock:
      graph_function, _ = self._maybe_define_function(args, kwargs)
      seen_names = set()
      captured = object_identity.ObjectIdentitySet(
          graph_function.graph.internal_captures)
      # pylint: disable=protected-access
      graph_function._arg_keywords = []
      prefix_counts = {}
      # pylint: enable=protected-access
      num_positional = 0
      for arg in graph_function.graph.inputs:
        if arg in captured:
          break
        num_positional += 1
        user_arg_name = compat.as_str(arg.op.get_attr("_user_specified_name"))
        proposal = user_arg_name
        while proposal in seen_names:
          index = prefix_counts.get(user_arg_name, 1)
          proposal = "{}_{}".format(user_arg_name, index)
          prefix_counts[user_arg_name] = index + 1
        seen_names.add(proposal)
        graph_function._arg_keywords.append(proposal)  # pylint: disable=protected-access
      # Anything can be a positional argument, in the same order as .inputs
      graph_function._num_positional_args = num_positional  # pylint: disable=protected-access
      return graph_function

  def get_concrete_function(self, *args, **kwargs):
    """Returns a `ConcreteFunction` specialized to inputs and execution context.

    Args:
      *args: inputs to specialize on. Can be concrete values (e.g. 1)
         or `tf.Tensor` or `tf.TensorSpec`.
      **kwargs: keyword inputs to specialize on. Concrete values (e.g. 1)
         or `tf.Tensor` or `tf.TensorSpec`.
    """
    graph_function = self._get_concrete_function_garbage_collected(
        *args, **kwargs)
    graph_function._garbage_collector.release()  # pylint: disable=protected-access
    return graph_function

  def _list_all_concrete_functions(self) -> List[ConcreteFunction]:
    return self._function_cache.values()

  def __get__(self, instance, owner):
    """Makes it possible to defun instance methods."""
    del owner
    # `instance` here is the instance that this `Function` was accessed through
    # e.g., for
    #
    #   class Foo(object):
    #
    #     @function.defun
    #     def bar(self):
    #       ...
    #
    #   foo = Foo()
    #   foo.bar()  # `foo.bar` is a `Function` instance
    #
    # then `instance` will be `foo` (and `owner` will be `Foo`).  We create a
    # new instance of `Function` here to allow different instances each
    # to create variables once, thereby allowing methods to be decorated with
    # defun. Keeps a cache to avoid retracing the function every time the
    # descriptor is accessed.
    if instance not in self._descriptor_cache:
      if instance is None:
        return self
      # If there is no instance-specific `Function` in the cache, we construct
      # an instance-specific `Function` that uses a weak reference to the
      # instance (so that the instance will be correctly gc'd).

      # And finally add the wrapped function to the description cache
      self._descriptor_cache[instance] = class_method_to_instance_method(
          self, instance)

    # Return the cached `Function` for the instance
    return self._descriptor_cache[instance]

  def _create_graph_function(self, args, kwargs, override_flat_arg_shapes=None):
    """Create a `ConcreteFunction` from `args` and `kwargs`."""
    self.tracing_count += 1

    if self.input_signature is None:
      arglen = len(args)
    else:
      arglen = len(self.input_signature)
    base_arg_names = self._function_spec.arg_names[:arglen]
    num_missing_args = arglen - len(self._function_spec.arg_names)
    missing_arg_names = [self._function_spec.vararg_name] * num_missing_args
    # Produce a list of missing args of the form ["arg_0", "arg_1", ...],
    # where arg is based on the self._function_spec.vararg_name.
    missing_arg_names = [
        "%s_%d" % (arg, i) for i, arg in enumerate(missing_arg_names)
    ]
    arg_names = base_arg_names + missing_arg_names
    graph_function = ConcreteFunction(
        func_graph_module.func_graph_from_py_func(
            self._name,
            self._python_function,
            args,
            kwargs,
            self.input_signature,
            autograph=self._autograph,
            autograph_options=self._autograph_options,
            arg_names=arg_names,
            override_flat_arg_shapes=override_flat_arg_shapes,
            capture_by_value=self._capture_by_value),
        self._function_attributes,
        function_spec=self.function_spec,
        # Tell the ConcreteFunction to clean up its graph once it goes out of
        # scope. This is not the default behavior since it gets used in some
        # places (like Keras) where the FuncGraph lives longer than the
        # ConcreteFunction.
        shared_func_graph=False)
    return graph_function

  def _define_function_with_shape_relaxation(self, args, kwargs, flat_args,
                                             filtered_flat_args):
    """Define a function, relaxing arg shapes to avoid unnecessary retracing."""
    flat_no_comp = nest.flatten((args, kwargs), expand_composites=False)

    any_composite_args = any(
        isinstance(x, composite_tensor.CompositeTensor) for x in flat_no_comp)

    # Build a cache key where TensorShapes include only rank information (and
    # not information about the size of each dimension).
    if not any_composite_args:
      rank_only_cache_key, _ = function_cache.make_cache_key(
          (args, kwargs), include_tensor_ranks_only=True)
    else:
      # For the rank-only cache key, replace any composite tensors with
      # shape-relaxed TypeSpecs.
      relaxed_args = nest.map_structure(
          _shape_relaxed_type_for_composite_tensor, (args, kwargs))
      rank_only_cache_key, _ = function_cache.make_cache_key(
          relaxed_args, include_tensor_ranks_only=True)

    arg_specs = [_type_spec_for(x) for x in flat_no_comp]
    relaxed_arg_specs = self._function_cache.arg_relaxed_specs.get(
        rank_only_cache_key, None)
    relaxed_arg_function = self._function_cache.arg_relaxed.get(
        rank_only_cache_key, None)

    if (relaxed_arg_function is not None
        and all(_is_type_subset(x, y) for (x, y) in
                zip(relaxed_arg_specs, arg_specs))):
      return relaxed_arg_function, filtered_flat_args

    if relaxed_arg_specs is None:
      relaxed_arg_specs = arg_specs
    else:
      if len(arg_specs) != len(relaxed_arg_specs):
        raise RuntimeError("Expected arg_specs len to match relaxed_arg_specs "
                           f"len: {len(arg_specs):d} vs. "
                           f"{len(relaxed_arg_specs):d}.")
      relaxed_arg_specs = [
          x if x is None else x.most_specific_compatible_type(y)
          for (x, y) in zip(arg_specs, relaxed_arg_specs)]
    self._function_cache.arg_relaxed_specs[rank_only_cache_key] = (
        relaxed_arg_specs)
    relaxed_arg_shapes = [
        x if x is None else x.shape
        for x in nest.flatten(relaxed_arg_specs, expand_composites=True)]

    if any_composite_args:
      # Rebuild composite tensors with the relaxed TypeSpecs.  For example,
      # if a tf.data iterator is passed as an argument, then we need to relax
      # the TensorShapes in its element_spec.
      (relaxed_arg_specs, relaxed_kwarg_specs) = nest.pack_sequence_as(
          (args, kwargs), relaxed_arg_specs, expand_composites=False)
      (args, kwargs) = nest.pack_sequence_as(
          (relaxed_arg_specs, relaxed_kwarg_specs),
          flat_args,
          expand_composites=True)

    graph_function = self._create_graph_function(
        args, kwargs, override_flat_arg_shapes=relaxed_arg_shapes)
    self._function_cache.arg_relaxed[rank_only_cache_key] = graph_function

    return (graph_function, [
        t for t in nest.flatten((args, kwargs), expand_composites=True)
        if isinstance(t, (ops.Tensor,
                          resource_variable_ops.BaseResourceVariable))
    ])

  def _maybe_define_function(self, args, kwargs):
    """Gets a function for these inputs, defining it if necessary.

    `args` and `kwargs` can be None if this `Function` was created with an
    `input_signature`.

    Caller must hold self._lock.

    Args:
      args: The varargs for the Python function.
      kwargs: The keyword args for the Python function.

    Returns:
      A graph function corresponding to the input signature implied by args and
      kwargs, as well as filtered flattened inputs (only Tensors and Variables)
      that the object should be called with.

    Raises:
      ValueError: If inputs are incompatible with the input signature.
      TypeError: If the function inputs include non-hashable objects
      RuntimeError: If there's an internal bug (inconsistency) in handling
        shape relaxation retracing.
    """
    if self.input_signature is None or args is not None or kwargs is not None:
      args, kwargs, flat_args, filtered_flat_args = \
          self._function_spec.canonicalize_function_inputs(*args, **kwargs)
    else:
      flat_args, filtered_flat_args = [None], []

    if self.input_signature is None:
      cache_key, cache_key_deletion_observer = function_cache.make_cache_key(
          (args, kwargs))
    else:
      cache_key, cache_key_deletion_observer = function_cache.make_cache_key(
          self.flat_input_signature)

    try:
      hash(cache_key)
    except TypeError as e:
      raise TypeError(
          "Arguments supplied to `defun`-generated functions must be "
          f"hashable.  Original error: {e}.")

    graph_function = self._function_cache.lookup(cache_key,
                                                 USE_FUNCTION_SUBTYPING)
    if graph_function is not None:
      return graph_function, filtered_flat_args

    with monitoring.MonitoredTimer(_graph_building_time_counter.get_cell()):
      with trace.Trace("tf.function-graph_building"):
        logging.vlog(1,
                     "Creating new FuncGraph for Python function %r (key: %r)",
                     self._python_function, cache_key)
        logging.vlog(2, "Python function signature [args: %s] [kwargs: %s]",
                     args, kwargs)

        ag_status = (
            ag_ctx.Status.ENABLED
            if self._autograph else ag_ctx.Status.DISABLED)
        with ag_ctx.ControlStatusCtx(
            status=ag_status, options=self._autograph_options):

          # Build a function with shape relaxation retracing if:
          # 1. shape relaxation is explicitly enabled
          # and 2. there's no provided input signature
          # and 3. there's been a cache miss for this calling context
          if (self._experimental_relax_shapes and
              self.input_signature is None and
              self._function_cache.has_call_context(cache_key.call_context)):
            return self._define_function_with_shape_relaxation(
                args, kwargs, flat_args, filtered_flat_args)

          self._function_cache.add_call_context(cache_key.call_context)
          graph_function = self._create_graph_function(args, kwargs)
          self._function_cache.add(cache_key, cache_key_deletion_observer,
                                   graph_function)

          return graph_function, filtered_flat_args


def register(func, *args, **kwargs):
  """Register a specialization of a `Function` into the graph.

  This won't actually call the function with the inputs, and only put the
  function definition into graph. Register function with different input param
  will result into multiple version of functions registered in graph.

  Args:
    func: the `Function` instance that generated by a @defun
    *args: input arguments for the Python function.
    **kwargs: input keyword arguments for the Python function.

  Returns:
    a `ConcreteFunction` object specialized to inputs and execution context.

  Raises:
    ValueError: When the input function is not a defun wrapped python function.
  """
  if not isinstance(func, Function):
    raise ValueError("Only defun function is allowed to be registered. "
                     f"Got {func} with type {type(func)}.")
  concrete_func = func.get_concrete_function(*args, **kwargs)
  concrete_func.add_to_graph()
  concrete_func.add_gradient_functions_to_graph()
  return concrete_func


def validate_signature(signature):
  if not isinstance(signature, (tuple, list)):
    raise TypeError("input_signature must be either a tuple or a list, got "
                    f"{type(signature)}.")

  if any(not isinstance(arg, tensor_spec.DenseSpec)
         for arg in nest.flatten(signature, expand_composites=True)):
    bad_args = [arg for arg in nest.flatten(signature, expand_composites=True)
                if not isinstance(arg, tensor_spec.DenseSpec)]
    raise TypeError("input_signature must be a possibly nested sequence of "
                    f"TensorSpec objects, got invalid args {bad_args} with "
                    f"types {list(map(type, bad_args))}.")


def validate_python_function(python_function):
  if not callable(python_function):
    raise TypeError(f"{python_function} is not a callable object.")


def defun(func=None,
          input_signature=None,
          autograph=True,
          experimental_autograph_options=None,
          experimental_relax_shapes=False):
  """Compiles a Python function into a callable TensorFlow graph.

  `defun` (short for "define function") compiles a Python function
  composed of TensorFlow operations into a callable that executes a `tf.Graph`
  containing those operations. The callable produced by `defun` contains only
  the subgraph of TensorFlow operations that were executed when the Python
  function was called with a particular input signature, defined as a list
  of the shapes and dtypes of the Python function's Tensor-valued arguments and
  the values of its non-Tensor Python objects.

  When eager execution is enabled, the ability to create graphs from Python
  functions makes it possible to incrementally trade off debuggability and
  interactivity for performance.  Functions compiled with `defun` cannot be
  inspected with `pdb`; however, executing a graph
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

  tf.compat.v1.enable_eager_execution()

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
  optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
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
  `F(tf.random.uniform([2])` will execute a different graph than
  `F(tf.random.uniform([3])` because the two inputs have different shapes.
  The first time that `F(*args, **kwargs)` is called with a particular sequence
  of Tensor shapes and dtypes and Python values, it constructs a graph by
  tracing the execution of `f(*args, **kwargs)`; this graph is bound to an
  input signature inferred from `(*args, **kwargs)` and cached for future reuse.

  NumPy arrays passed as inputs to `F` are converted to `tf.Tensor` objects
  before being passed to `f`, and are treated as Tensors for caching. This
  allows a function to be called multiple times with NumPy arrays having
  different values but the same shape and dtype without re-tracing each time.

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
  words = tf.random.uniform(([50, 300, 10])
  second_input = tf.random.uniform([300, 100])
  my_sequence_model(words, second_input)

  words = tf.random.uniform(([50, 300, 20])
  my_sequence_model(words, second_input)

  # Passing an input with an incompatible shape will raise an error.
  words = tf.random.uniform(([50, 100, 20])
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

  tf.compat.v1.enable_eager_execution()

  def add_noise():
    return tf.eye(5) + np.random.randn(5, 5)
  ```

  will return a different output everytime it is invoked, the compiled function
  `compiled = tf.contrib.eager.defun(add_noise)` will return the same value
  every time it is called, since a particular random offset generated by NumPy
  will be inserted into the graph as a TensorFlow constant. The solution is to
  replace the call to `np.random.randn` with `tf.random.normal((5, 5))`.

  _Python Side-Effects_

  A corollary of the previous discussion on tracing is the following: If a
  Python function `f` has Python side-effects, then executing `f` multiple times
  will not necessarily be semantically equivalent to executing `F =
  tf.contrib.eager.defun(f)` multiple times; this difference is due to the fact
  that `defun` only captures the subgraph of TensorFlow operations that is
  constructed when `f` is called in a graph-building context.

  _Python Control Flow_

  The structure of many machine learning computations depend upon whether one is
  training or validating, and it is common to nest specialized logic under `if
  training:` blocks. By mapping each input signature to a unique graph, `defun`
  lets users transparently compile such code, as the following code snippet
  demonstrates:

  ```python
  import tensorflow as tf

  tf.compat.v1.enable_eager_execution()

  @tf.contrib.eager.defun
  def lossy_matmul(W, x, training=True):
    outputs = tf.matmul(W, x)
    if training:
      outputs = tf.nn.dropout(outputs, keep_probability=0.2)
    return outputs

  W = tf.random.normal((3, 5))
  x = tf.random.normal((5, 1))

  # Executes a graph that applies dropout.
  lossy_outputs = lossy_matmul(W, x, training=True)

  # Executes a graph that does not apply dropout.
  exact_outputs = lossy_matmul(W, x, training=False)
  ```

  _TensorFlow Control Flow_

  When `autograph` is `True`, data-dependent control flow is allowed as well.
  Control flow statements that depend on `Tensor` values are staged into
  corresponding TensorFlow ops. For example, the following code will work as
  expected:

  ```python
  @tf.contrib.eager.defun
  def dynamic_rnn_loop(cell, seq):
    state, output = cell.zero_state()
    for input in seq:
      state, output = cell(input, state)
    return output
  ```

  For more information see `tf.autograph`.

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

  tf.compat.v1.enable_eager_execution()

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
    autograph: Whether `func` should be compiled before
      constructing the graph. See https://www.tensorflow.org/guide/autograph
      for more information.
    experimental_autograph_options: Experimental knobs (in the form of a tuple
      of tensorflow.autograph.Feature values) to control behavior when
      autograph=True.
    experimental_relax_shapes: When true, argument shapes may be relaxed to
      avoid unnecessary retracing.

  Returns:
     If `func` is not None, returns a callable that will execute the compiled
     function (and return zero or more `tf.Tensor` objects).
     If `func` is None, returns a decorator that, when invoked with a single
     `func` argument, returns a callable equivalent to the case above.

  Raises:
    TypeError: If `input_signature` is neither `None` nor a sequence of
      `tf.contrib.eager.TensorSpec` objects.
  """
  return defun_with_attributes(
      func=func,
      input_signature=input_signature,
      autograph=autograph,
      experimental_autograph_options=experimental_autograph_options,
      experimental_relax_shapes=experimental_relax_shapes)


@tf_export("__internal__.function.defun_with_attributes", v1=[])
def defun_with_attributes(func=None,
                          input_signature=None,
                          attributes=None,
                          autograph=True,
                          experimental_autograph_options=None,
                          jit_compile=None,
                          experimental_relax_shapes=False,
                          experimental_follow_type_hints=False):
  """Compiles a Python function into a callable TensorFlow graph.

  This function supports adding extra function attributes. See detailed
  documentation in defun(). Currently this is not exposed in public API since we
  don't expect user to directly use attributes, and attribute won't work by
  itself. This assumption might change in future.

  Args:
    func: function to be compiled.
    input_signature: same as defun()'s input_signature.
    attributes: A dictionary of arguments which will be added to function def as
      attributes. Currently only support primitive types as value, and only
      allowlisted attribute name is allowed. Unallowlisted attribute name or
      unsupported value will result into ValueError. `func_name` is also one of
      the allowlisted argument which is a python string, and sets the name for
      this `ConcreteFunction` in the graph.
    autograph: same as defun()'s autograph.
    experimental_autograph_options: same as defun()'s
      experimental_autograph_options.
    jit_compile: same as defun()'s jit_compile.
    experimental_relax_shapes: same as defun()'s experimental_relax_shapes
    experimental_follow_type_hints: see `tf.function`.

  Returns:
    Same as the return value of defun, with attributes added to the function in
    graph.
  """
  if input_signature is not None:
    validate_signature(input_signature)

  # TODO(apassos): deal with captured global state. Deal with control flow.
  def decorated(function):
    try:
      if attributes:
        name = attributes.pop("func_name", function.__name__)
      else:
        name = function.__name__
    except AttributeError:
      name = "function"
    return tf_decorator.make_decorator(
        function,
        Function(
            function,
            name,
            input_signature=input_signature,
            attributes=attributes,
            autograph=autograph,
            autograph_options=experimental_autograph_options,
            jit_compile=jit_compile,
            experimental_relax_shapes=experimental_relax_shapes,
            experimental_follow_type_hints=experimental_follow_type_hints))

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


# When a method is bound to objects of this type, it allows AutoGraph to
# recover a weak reference the original method's self pointer, so that it can
# execute it consistent with class_method_to_instance_method's
# bound_method_wrapper.
# TODO(b/119246461): This is not pretty. Use a descriptor instead?
class TfMethodTarget(object):
  """Binding target for methods replaced by function and defun."""

  __slots__ = ("weakrefself_target__", "weakrefself_func__")

  def __init__(self, target, original_python_function):
    self.weakrefself_target__ = target
    self.weakrefself_func__ = weakref.ref(original_python_function)

  @property
  def target(self):
    return self.weakrefself_target__()

  @property
  def target_class(self):
    true_self = self.weakrefself_target__()
    if tf_inspect.isclass(true_self):
      # Class method
      return true_self
    else:
      return true_self.__class__

  def call(self, args, kwargs):
    wrapped_fn = self.weakrefself_func__()
    if tf_inspect.ismethod(wrapped_fn):
      wrapped_fn = six.get_unbound_function(wrapped_fn)
    return wrapped_fn(self.weakrefself_target__(), *args, **kwargs)


def class_method_to_instance_method(original_function, instance):
  """Constructs a new `Function` with `self` bound."""
  weak_instance = weakref.ref(instance)

  # Note: while we could bind to a weakref proxy instead, that causes the
  # bound method to be unhashable.
  bound_method = types_lib.MethodType(
      original_function.python_function,
      TfMethodTarget(weak_instance, original_function.python_function))

  # original_function is expected to be of one of the two `Function` types
  # (defined either in function.py or def_function.py).
  assert hasattr(original_function, "_name")
  assert hasattr(original_function, "_autograph")
  assert hasattr(original_function, "_function_spec")
  assert hasattr(original_function, "python_function")

  weak_bound_method_wrapper = None
  def bound_method_wrapper(*args, **kwargs):
    """Wraps either a dummy MethodType or a converted AutoGraph function."""
    # __wrapped__ allows AutoGraph to swap in a converted function.
    strong_bound_method_wrapper = weak_bound_method_wrapper()
    wrapped_fn = strong_bound_method_wrapper.__wrapped__

    if wrapped_fn is strong_bound_method_wrapper.__original_wrapped__:
      # If __wrapped__ was not replaced, then call original_function.
      # TODO(mdan): For better consistency, use the wrapper's call().
      wrapped_fn = original_function.python_function
      if tf_inspect.ismethod(wrapped_fn):
        wrapped_fn = six.get_unbound_function(wrapped_fn)
      return wrapped_fn(weak_instance(), *args, **kwargs)

    # If __wrapped__ was replaced, then it is always an unbound function.
    # However, the replacer is still responsible for attaching self properly.
    # TODO(mdan): Is it possible to do it here instead?
    return wrapped_fn(*args, **kwargs)
  weak_bound_method_wrapper = weakref.ref(bound_method_wrapper)

  # pylint: disable=protected-access
  # We make a dummy MethodType object to generate the correct bound method
  # signature. The actual call is to a function with a weak reference to
  # `instance`.
  instance_func = type(original_function)(
      tf_decorator.make_decorator(bound_method, bound_method_wrapper),
      name=original_function._name,
      autograph=original_function._autograph,
      input_signature=original_function.input_signature,
      experimental_relax_shapes=original_function._experimental_relax_shapes,
      jit_compile=original_function._jit_compile)
  # pylint: enable=protected-access

  # We wrap the the bound method with tf_decorator so inspection works correctly
  wrapped_instance_func = tf_decorator.make_decorator(bound_method,
                                                      instance_func)
  return wrapped_instance_func


class ConcreteFunctionGarbageCollector(object):
  """Cleans up reference cycles when a `ConcreteFunction` goes out of scope."""

  __slots__ = ["_func_graph"]

  def __init__(self, func_graph):
    self._func_graph = func_graph

  def release(self):
    """Call off the FuncGraph deletion."""
    self._func_graph = None

  def __del__(self):
    if func_graph_module is None or memory is None or self._func_graph is None:
      return
    try:
      func_graph_module.dismantle_func_graph(self._func_graph)
    except:  # pylint: disable=bare-except
      pass


class _Marker(object):
  """Markers used to pretty-print nested args in function signatures."""

  __slots__ = ["_s"]

  def __init__(self, s):
    self._s = s

  def __repr__(self):
    return str(self._s)


def _structure_summary(structure):
  """Displays a summary of the nesting structure of the given value."""

  def type_name(x):
    if isinstance(x, type_spec.TypeSpec):
      return x.value_type.__name__
    else:
      return type(x).__name__

  markers = [_Marker(type_name(v)) for v in nest.flatten(structure)]
  return str(nest.pack_sequence_as(structure, markers))


def _contains_type_spec(value):
  return any(isinstance(x, type_spec.TypeSpec) for x in nest.flatten(value))
