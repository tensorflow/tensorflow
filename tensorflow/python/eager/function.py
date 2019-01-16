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
import re
import sys
import threading
import types as types_lib
import weakref

import numpy as np
import six

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import tape
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import error_interpolation
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import function_utils
from tensorflow.python.util import memory
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

# This is to avoid a circular dependency with gradients_impl
gradients_impl._function = sys.modules[__name__]  # pylint: disable=protected-access

FORWARD_FUNCTION_ATTRIBUTE_NAME = "forward_function_name"
BACKWARD_FUNCTION_ATTRIBUTE_NAME = "backward_function_name"

# TODO(scottzhu): Update this to allow arbitrary attribute names in future.
WHITELIST_FUNCTION_ATTRIBUTE_REGEX = [
    "experimental_.*",
    FORWARD_FUNCTION_ATTRIBUTE_NAME,
    BACKWARD_FUNCTION_ATTRIBUTE_NAME
]

CacheKey = collections.namedtuple("CacheKey", [
    "input_signature", "parent_graph", "device_functions", "colocation_stack",
    "uses_xla"
])


def is_same_structure(structure1,
                      structure2,
                      check_values=False):
  """Check two structures for equality, optionally of types and of values."""
  try:
    nest.assert_same_structure(structure1, structure2)
  except (ValueError, TypeError):
    return False
  if check_values:
    flattened1 = nest.flatten(structure1)
    flattened2 = nest.flatten(structure2)
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
    ValueError: If the kwargs contains unwhitelisted name or unsupported value
      types.
  """
  attrs = {}
  for key, value in attributes.items():
    if not any(re.match(reg, key)
               for reg in WHITELIST_FUNCTION_ATTRIBUTE_REGEX):
      raise ValueError("Attribute name is not whitelisted. "
                       "Whitelisted: prefix %s, got: %s" %
                       (WHITELIST_FUNCTION_ATTRIBUTE_REGEX, key))

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
      raise ValueError("Unsupported attribute type for %s with type %s" %
                       (key, type(value)))
  return attrs


class _InterpolateFunctionError(object):
  """Context Manager that interpolates the exception from 'top_level_func'."""

  def __init__(self, top_level_func):
    self._func = top_level_func

  def __enter__(self):
    pass

  def __exit__(self, typ, exc, tb):
    if not exc or not isinstance(exc, errors.OpError):
      return False
    message = compat.as_text(exc.message)
    _, tags = error_interpolation.parse_message(message)
    g = None
    func_stack = []
    # pylint: disable=protected-access
    for t in tags:
      if t.type == "function_node":
        if t.name == compat.as_str(self._func.name):
          g = self._func._graph
        elif g:
          next_func = g._get_function(t.name)
          if next_func is not None and isinstance(next_func,
                                                  _EagerDefinedFunction):
            g = next_func._graph
        if g:
          func_stack.append(g.name)
        else:
          func_stack.append("<unknown>")
    # pylint: enable=protected-access
    if g:
      message = error_interpolation.interpolate(message, g)
      message += "\n\nFunction call stack:\n"
      message += " -> ".join(func_stack)
      message += "\n"
      exc._message = message  # pylint: disable=protected-access
    return False


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
      outputs: the tensors in the graph which will be outputs to the function
      attrs: dict mapping names of attributes to their AttrValue values
    """
    input_ops = set(arg.op for arg in inputs)
    operations = [op for op in graph.get_operations() if op not in input_ops]

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
    with ops.init_scope():
      if context.executing_eagerly():
        context.add_function(fn)
    self.definition = function_def
    self.name = compat.as_bytes(function_def.signature.name)
    self.signature = function_def.signature
    self._num_outputs = len(self.signature.output_arg)
    self._output_types = [o.type for o in self.signature.output_arg]
    self._output_shapes = [o.shape for o in outputs]
    self._func_graph_outputs = outputs
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

    `ConcreteFunction` execution respects device annotations only if the
    function won't be compiled with xla.

    Args:
      ctx: a Context object
      args: a list of arguments to supply this function with.

    Returns:
      The outputs of the function call.

    Raises:
      ValueError: if the number of arguments is incorrect.
    """
    if len(args) != len(self.signature.input_arg):
      raise ValueError(
          "Arguments and signature arguments do not match: %s %s " %
          (len(args), len(list(self.signature.input_arg))))

    function_call_options = ctx.get_function_call_options()
    if function_call_options.config_proto_serialized is None:
      config = function_utils.get_disabled_rewriter_config()
    else:
      config = function_call_options.config_proto_serialized
    executor_type = function_call_options.executor_type or ""

    executing_eagerly = ctx.executing_eagerly()
    if executing_eagerly:
      with _InterpolateFunctionError(self):
        outputs = execute.execute(
            str(self.signature.name),
            num_outputs=self._num_outputs,
            inputs=args,
            attrs=("executor_type", executor_type,
                   "config_proto", config),
            ctx=ctx)
      # Replace empty list with None
      outputs = outputs or None
    elif self._graph._xla_compile:  # pylint: disable=protected-access
      g = ops.get_default_graph()
      self.add_to_graph(g)
      signature = self.signature
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
      if isinstance(outputs, (ops.Tensor, type(None))):
        outputs = [outputs]
      else:
        outputs = list(outputs)
    else:
      # TODO(akshayka): Either remove this if the FunctionLibraryRuntime
      # creates `PartitionedCallOp` kernels by default, or remove the previous
      # branch if a TPU kernel is registered for `PartitionedCall`.
      with _InterpolateFunctionError(self):
        outputs = functional_ops.partitioned_call(
            args=args,
            f=self,
            tout=self._output_types,
            executing_eagerly=executing_eagerly,
            config=config,
            executor_type=executor_type)

    if executing_eagerly:
      return outputs
    else:
      for i, shape in enumerate(self._output_shapes):
        outputs[i].set_shape(shape)
      for i, func_graph_output in enumerate(self._func_graph_outputs):
        custom_gradient.copy_handle_data(func_graph_output, outputs[i])
      return outputs


class ConcreteFunction(object):
  """Callable object encapsulating a function definition and its gradient.

  `ConcreteFunction` is a callable that encapsulates a function definition and
  is differentiable under `tf.GradientTape` objects.
  """

  def __init__(self, func_graph, attrs=None, signature=None):
    """Initialize a `ConcreteFunction`.

    Args:
      func_graph: An instance of FuncGraph: the function body to wrap.
      attrs: (optional) dict mapping names of attributes to their AttrValue
        values. Attributes in `attrs` will be included in this function's
        definition.
     signature: a nested sequence of `TensorSpec` objects specifying the input
       signature of this function.
    Raises:
      ValueError: If number of input_placeholders is not equal to the number
        of function inputs.
    """
    self._arg_keywords = None
    self._num_positional_args = None
    self._func_graph = func_graph
    self._captured_inputs = list(self._func_graph.captures.keys())
    self._num_outputs = len(self._func_graph.outputs)
    self._output_shapes = tuple(
        output.shape for output in self._func_graph.outputs)
    self._attrs = _parse_func_attrs(attrs or {})

    self._inference_function = _EagerDefinedFunction(
        _inference_name(self._func_graph.name), self._func_graph,
        self._func_graph.inputs, self._func_graph.outputs, self._attrs)
    self._backward_graph_function = None
    self._signature = signature
    self._gradient_name = None

  def __call__(self, *args, **kwargs):
    """Executes the wrapped function.

    Args:
      *args: Tensors or Variables. Positional arguments are only accepted when
        they correspond one-to-one with arguments of the traced Python function.
      **kwargs: Tensors or Variables specified by name. When
        `get_concrete_function` was called to create this `ConcreteFunction`,
        each Tensor input was given a name, defaulting to the name of the Python
        function's argument but possibly overridden by the `name=` argument to
        `tf.TensorSpec`. These names become the argument names for the concrete
        function.

    Returns:
      The result of applying the TF function on the given Tensors.

    Raises:
      AssertionError: If this `ConcreteFunction` was not created through
        `get_concrete_function`.
      ValueError: If arguments contains anything other than Tensors or
        Variables.
      TypeError: For invalid positional/keyword argument combinations.
    """
    if self._arg_keywords is None or self._num_positional_args is None:
      if self._signature is not None:
        if kwargs:
          raise NotImplementedError(
              "Keyword arguments not supported when calling a "
              "wrap_function-decorated function.")
        return self._call_flat(args)
      raise AssertionError(
          "Tried to call a concrete function obtained from an internal API "
          "through the public interface. Use get_concrete_function instead.")
    if len(args) > self._num_positional_args:
      raise TypeError(
          ("Expected at most {} positional arguments ({}), got {}. When "
           "calling a concrete function, positional arguments may not be bound "
           "to Tensors within nested structures.").format(
               self._num_positional_args,
               self._arg_keywords[:self._num_positional_args],
               args))
    args = list(args)
    for keyword in self._arg_keywords[len(args):]:
      args.append(kwargs.pop(compat.as_str(keyword)))
    if kwargs:
      positional_arg_keywords = set(self._arg_keywords[:len(args)])
      for unused_key in kwargs:
        if unused_key in positional_arg_keywords:
          raise TypeError("Got two values for keyword '{}'.".format(unused_key))
      raise TypeError("Keyword arguments {} unknown.".format(kwargs.keys()))
    return self._call_flat(args)

  def _filtered_call(self, args, kwargs):
    """Executes the function, filtering arguments from the Python function.

    Objects aside from Tensors and Variables are ignored.

    Args:
      args: Canonicalized positional arguments of the Python function.
      kwargs: Canonicalized keyword arguments of the Python function.

    Returns:
      The result of applying the function on the Tensors/Variables contained in
      `args` and `kwargs`.
    """
    return self._call_flat(
        (t for t in nest.flatten((args, kwargs))
         if isinstance(t, (ops.Tensor,
                           resource_variable_ops.ResourceVariable))))

  def _call_flat(self, args):
    """Executes the wrapped function.

    Args:
      args: a list of Tensors or Variables.

    Returns:
      The result of applying the TF function to `args`.

    Raises:
      ValueError: If `args` contains anything other than Tensors or Variables.
    """
    ctx = context.context()

    tape.variables_accessed(self._func_graph.variables)

    tensor_inputs = []
    for i, arg in enumerate(args):
      if isinstance(arg, resource_variable_ops.ResourceVariable):
        if arg.trainable:
          tape.variable_accessed(arg)
        tensor_inputs.append(arg.handle)
      elif isinstance(arg, ops.Tensor):
        tensor_inputs.append(arg)
      elif (self._signature is not None and
            isinstance(self._signature[i], tensor_spec.TensorSpec)):
        tensor_inputs.append(
            ops.convert_to_tensor(arg, self._signature[i].dtype))
      else:
        raise ValueError("All inputs to `ConcreteFunction`s must be Tensors; "
                         "on invocation of %s, the %d-th input (%s) was not a "
                         "Tensor." % (self._func_graph.name, i, str(arg)))
    args = tensor_inputs + self._captured_inputs

    if (tape.should_record(tensor_inputs) or
        tape.should_record(self._captured_inputs)):
      if context.executing_eagerly():
        return self._eager_backprop_call(args)
      else:
        return self._backprop_call_with_delayed_rewrite(args)

    # Only need to override the gradient in graph mode and when we have outputs.
    if context.executing_eagerly() or not self.outputs:
      outputs = self._inference_function.call(ctx, args)
    else:
      if not self._gradient_name:
        self._gradient_name = "PartitionedCall-%s" % ops.uid()
        self._register_gradient(self._gradient_name)
      with ops.get_default_graph().gradient_override_map(
          {"PartitionedCall": self._gradient_name,
           "StatefulPartitionedCall": self._gradient_name}):
        outputs = self._inference_function.call(ctx, args)
    return self._build_call_outputs(outputs)

  def _register_gradient(self, name):
    """Registers the gradient for the current `ConcreteFunction` under the given
    name.

    The gradient rewrites an inference call op to a forward call op, but does
    not modify a pre-existing forward call op. It then computes the gradient
    from the output's gradients and the side outputs of the forward op.

    Args:
      name: The name to register the gradient as.
    """
    @ops.RegisterGradient(name)
    def _registered_grad_fn(op, *doutputs):  # pylint: disable=unused-variable
      return self._grad_fn(op, *doutputs)

  def _grad_fn(self, op, *doutputs):
    """Gradients of this function."""
    if self._backward_graph_function is None:
      self._construct_backprop_function()

    # pylint: disable=protected-access
    self._forward_function.add_to_graph(op.graph)
    num_inference_outputs = self._inference_function._num_outputs

    # Rewrite an inference call op to be a forward call op
    if op.get_attr("f").name.encode() == self._inference_function.name:
      op._set_func_attr("f", self._forward_function.name)
      op._set_type_list_attr("Tout", self._forward_function._output_types)
      op._add_outputs(
          self._forward_function._output_types[num_inference_outputs:],
          self._forward_function._output_shapes[num_inference_outputs:])
      for i in range(num_inference_outputs, len(op.outputs)):
        func_graph_output = self._forward_function._func_graph_outputs[i]
        custom_gradient.copy_handle_data(func_graph_output, op.outputs[i])
    # pylint: enable=protected-access
    # Compute the gradients using the side outputs
    side_outputs = op.outputs[num_inference_outputs:]
    args = list(doutputs[:num_inference_outputs]) + list(side_outputs)
    return self._backward_graph_function._call_flat(  # pylint: disable=protected-access
        (a for a in args if a is not None))

  @property
  def name(self):
    """`ConcreteFunction` name."""
    return self._inference_function.name

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
    """Returns structured signature of the original function."""
    return self._func_graph.structured_input_signature

  @property
  def outputs(self):
    """Returns tensors in `self.graph` corresponding to returned tensors."""
    return self._func_graph.outputs

  @property
  def structured_outputs(self):
    """Returns outputs in `self.graph` as returned by the original function."""
    return self._func_graph.structured_outputs

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

  def add_to_graph(self, g=None, register_gradient_functions=False):
    """Registers the function, adds it to the graph g or default graph."""
    # If we are not executing eagerly, adds the function to default graph if no
    # graph is specified.
    # In case of eager execution, function definition gets added to context
    # during construction itself.

    # TODO(allel/shivaniagrawal): rename this to register to reflect the
    # method's functionality better. Remove register_gradient_functions argument
    # and figure out if these needs to be registered.

    if not context.executing_eagerly() or g:
      if not g:
        g = ops.get_default_graph()
      self._inference_function.add_to_graph(g)  # pylint: disable=protected-access

      # pylint: disable=protected-access
      if register_gradient_functions:
        # There are two situations for the actual call of a defun:
        # 1. If none of the input args are resource variables or watch by any
        #   tape, and it will run the _inference_function of concrete_func for
        #   forward pass, the gradient will be generated by standard mechanism.
        # 2. Otherwise, defun will create two functions, one for forward pass,
        #   and the backward pass will be created via tape.
        #   When registering the function, we register both cases.
        if self._backward_graph_function is None:
          self._construct_backprop_function()
        forward_function = self._forward_function
        backward_function = self._backward_graph_function._inference_function
        # pylint: enable=protected-access
        forward_function.add_to_graph(g)
        backward_function.add_to_graph(g)

  def _construct_backprop_function(self):
    """Constructs the backprop function object for this function."""
    backwards_graph = func_graph_module.FuncGraph(
        _backward_name(self._func_graph.name))
    forward_function_name = _forward_name(self._func_graph.name)
    outputs = [x for x in self._func_graph.outputs
               if gradients_impl.IsTrainable(x)]
    with backwards_graph.as_default():
      gradients_wrt_outputs = [
          graph_placeholder(x.dtype, x.shape) for x in outputs
      ]
      gradients_wrt_inputs = gradients_impl._GradientsHelper(  # pylint: disable=protected-access
          outputs,
          self._func_graph.inputs,
          grad_ys=gradients_wrt_outputs,
          src_graph=self._func_graph)

    backwards_graph_captures = list(backwards_graph.captures.keys())

    backward_function_attr = _parse_func_attrs(
        {FORWARD_FUNCTION_ATTRIBUTE_NAME: forward_function_name})
    backward_function_attr.update(self._attrs)

    # The ordering of `backwards_graph.inputs` is important: inputs of
    # `self._backward_graph_function` correspond to outputs of
    # `self._forward_function`.
    backwards_graph.inputs = gradients_wrt_outputs + list(
        backwards_graph.captures.values())
    # Clear captures, since we pass them in as inputs.
    backwards_graph.captures = {}
    backwards_graph.outputs.extend(
        grad for grad in func_graph_module.flatten(gradients_wrt_inputs)
        if grad is not None)
    backwards_graph.structured_outputs = gradients_wrt_inputs
    self._backward_graph_function = ConcreteFunction(
        backwards_graph, attrs=backward_function_attr)

    forward_function_attr = _parse_func_attrs({
        BACKWARD_FUNCTION_ATTRIBUTE_NAME:
            self._backward_graph_function._inference_function.name})  # pylint: disable=protected-access
    forward_function_attr.update(self._attrs)
    self._forward_function = _EagerDefinedFunction(
        forward_function_name, self._func_graph, self._func_graph.inputs,
        self._func_graph.outputs + backwards_graph_captures,
        forward_function_attr)

  def _eager_backprop_call(self, args):
    """Calls the forward function and records the result on a tape.

    This method fully constructs the forward and backward functions before
    calling the function and recording them on the tape.

    (Only records results on a tape if the function has outputs).

    Args:
      args: All inputs to the function, including resolved captured inputs

    Returns:
      The call output.
    """
    if self._backward_graph_function is None:
      self._construct_backprop_function()

    ctx = context.context()

    if not self._gradient_name:
      self._gradient_name = "PartitionedCall-%s" % ops.uid()
      self._register_gradient(self._gradient_name)
    with ops.get_default_graph().gradient_override_map(
        {"PartitionedCall": self._gradient_name,
         "StatefulPartitionedCall": self._gradient_name}):
      outputs = self._forward_function.call(ctx, args)

    if isinstance(outputs, ops.Operation) or outputs is None:
      return outputs

    # `real_outputs` are the actual outputs of the inference graph function;
    # `side_outputs` are the intermediate Tensors that were added as outputs to
    # the forward graph function so that we can compute its gradient.
    real_outputs = outputs[:self._num_outputs]
    skip_positions = [i for i, t in enumerate(real_outputs)
                      if not gradients_impl.IsTrainable(t)]
    side_outputs = outputs[self._num_outputs:]

    def backward_function(*args):
      args = [a for i, a in enumerate(args)
              if a is not None and i not in skip_positions]
      return self._backward_graph_function._call_flat(  # pylint: disable=protected-access
          list(args) + side_outputs)

    tape.record_operation(self._forward_function.signature.name, real_outputs,
                          args, backward_function)
    return self._build_call_outputs(real_outputs)

  def _backprop_call_with_delayed_rewrite(self, args):
    """Calls the inference function and records the result on a tape.

    The recorded backwards function will construct the backwards graph and
    rewrite the inference function to the forward function. This only happens
    if the recorded backwards function ends up being used to compute gradients.

    This approach avoids constructing unnecessary graphs, but it only works if
    we are calling this function when not executing eagerly.

    (Only records results on a tape if the function has outputs)

    Args:
      args: All inputs to the function, including resolved captured inputs

    Returns:
      The call output.
    """
    ctx = context.context()

    if not self._gradient_name:
      self._gradient_name = "PartitionedCall-%s" % ops.uid()
      self._register_gradient(self._gradient_name)
    with ops.get_default_graph().gradient_override_map(
        {"PartitionedCall": self._gradient_name,
         "StatefulPartitionedCall": self._gradient_name}):
      outputs = self._inference_function.call(ctx, args)

    if isinstance(outputs, ops.Operation) or outputs is None:
      return outputs

    call_op = outputs[0].op

    def backward_function(*args):
      return self._grad_fn(call_op, *args)

    tape.record_operation(self._inference_function.signature.name, outputs,
                          args, backward_function)
    return self._build_call_outputs(outputs)

  def _build_call_outputs(self, result):
    """Maps the fdef output list to actual output structure.

    Args:
      result: Output lists defined by FunctionDef.
    Returns:
      The actual call output.
    """
    if self._func_graph.structured_outputs is None:
      return result

    # Use `nest.flatten` instead of `func_graph_module.flatten` in order to
    # preserve any IndexedSlices in `self._func_graph.structured_outputs`.
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


pywrap_tensorflow.RegisterType("Tensor", ops.Tensor)
pywrap_tensorflow.RegisterType("IndexedSlices", ops.IndexedSlices)


def _deterministic_dict_values(dictionary):
  return tuple(dictionary[key] for key in sorted(dictionary))


class FunctionSpec(object):
  """Specification of how to bind arguments to a function."""

  @staticmethod
  def from_function_and_signature(python_function, input_signature):
    """Create a FunctionSpec instance given a python function and signature."""
    if isinstance(python_function, functools.partial):
      python_function_to_inspect = python_function.func
      args_to_prepend = python_function.args or tuple()
      kwargs_to_include = python_function.keywords or {}
    else:
      python_function_to_inspect = python_function
      args_to_prepend = tuple()
      kwargs_to_include = {}

    fullargspec = tf_inspect.getfullargspec(python_function_to_inspect)
    is_method = tf_inspect.ismethod(python_function_to_inspect)

    return FunctionSpec(fullargspec, is_method, args_to_prepend,
                        kwargs_to_include, input_signature)

  def __init__(self, fullargspec, is_method, args_to_prepend, kwargs_to_include,
               input_signature):
    self._fullargspec = fullargspec
    self._is_method = is_method
    self._args_to_prepend = args_to_prepend
    self._kwargs_to_include = kwargs_to_include
    self._default_values = fullargspec.defaults

    if self._is_method:
      # Remove `self`: default arguments shouldn't be matched to it.
      args = fullargspec.args[1:]
    else:
      args = fullargspec.args

    # A cache mapping from argument name to index, for canonicalizing
    # arguments that are called in a keyword-like fashion.
    self._args_to_indices = {arg: i for i, arg in enumerate(args)}
    self.arg_names = args
    self.vararg_name = fullargspec.varargs

    # A cache mapping from arg index to default value, for canonicalization.
    offset = len(args) - len(fullargspec.defaults or [])
    self._arg_indices_to_default_values = {
        offset + index: default
        for index, default in enumerate(fullargspec.defaults or [])
    }
    self._default_values_start_index = offset
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

  @property
  def fullargspec(self):
    return self._fullargspec

  @property
  def is_method(self):
    return self._is_method

  @property
  def args_to_prepend(self):
    return self._args_to_prepend

  @property
  def kwargs_to_include(self):
    return self._kwargs_to_include

  @property
  def input_signature(self):
    return self._input_signature

  @property
  def flat_input_signature(self):
    return self._flat_input_signature

  def canonicalize_function_inputs(self, *args, **kwargs):
    """Canonicalizes `args` and `kwargs`.

    Canonicalize the inputs to the Python function using a `FunctionSpec`
    instance. In particular, we parse the varags and kwargs that the
    original function was called with into a tuple corresponding to the
    Python function's positional (named) arguments and a dictionary
    corresponding to its kwargs.

    Args:
      *args: The varargs this object was called with.
      **kwargs: The keyword args this function was called with.

    Returns:
      A canonicalized ordering of the inputs representened by a tuple in the
      form (args, kwargs). Here: `args` is a full list of bound arguments, and
      `kwargs` contains only true keyword arguments, as opposed to named
      arguments called in a keyword-like fashion.

    Raises:
      ValueError: If a keyword in `kwargs` cannot be matched with a positional
        argument when an input signature is specified, or when the inputs
        do not conform to the input signature.
    """
    args = self._args_to_prepend + args
    kwargs = dict(kwargs, **self._kwargs_to_include)
    if not kwargs:
      if self._default_values:
        inputs = args + self._default_values[
            len(args) - self._default_values_start_index:]
      else:
        inputs = args
    else:
      # Maps from index of arg to its corresponding value, according to `args`
      # and `kwargs`; seeded with the default values for the named args that
      # aren't in `args`.
      arg_indices_to_values = {
          index: default for index, default in six.iteritems(
              self._arg_indices_to_default_values) if index >= len(args)
      }
      consumed_args = []
      for arg, value in six.iteritems(kwargs):
        index = self._args_to_indices.get(arg, None)
        if index is not None:
          arg_indices_to_values[index] = value
          consumed_args.append(arg)
        elif self._input_signature is not None:
          raise ValueError("Cannot define a TensorFlow function from a Python "
                           "function with keyword arguments when "
                           "input_signature is provided.")
      for arg in consumed_args:
        # After this loop, `kwargs` will only contain true keyword arguments, as
        # opposed to named arguments called in a keyword-like fashion.
        kwargs.pop(arg)
      inputs = args + _deterministic_dict_values(arg_indices_to_values)
    flat_inputs = nest.flatten(inputs)

    # Check for NumPy arrays in arguments and convert them to Tensors.
    # TODO(nareshmodi): Skip ndarray conversion to tensor altogether, perhaps
    # finding a way to store them directly in the cache key (currently not
    # possible since ndarrays are not hashable).
    need_packing = False
    for index, value in enumerate(flat_inputs):
      if type(value) == np.ndarray:
        flat_inputs[index] = constant_op.constant(value)
        need_packing = True
    if need_packing:
      inputs = nest.pack_sequence_as(
          structure=inputs, flat_sequence=flat_inputs)
    if self._input_signature is None:
      return inputs, kwargs
    else:
      assert not kwargs
      signature_relevant_inputs = inputs[:len(self._input_signature)]
      if not is_same_structure(self._input_signature,
                               signature_relevant_inputs):
        raise ValueError("Structure of Python function inputs does not match "
                         "input_signature.")
      signature_inputs_flat = nest.flatten(signature_relevant_inputs)
      if any(
          not pywrap_tensorflow.IsTensor(arg) for arg in signature_inputs_flat):
        raise ValueError("When input_signature is provided, all inputs to "
                         "the Python function must be Tensors.")
      if any(not spec.is_compatible_with(other) for spec, other in zip(
          self._flat_input_signature, signature_inputs_flat)):
        raise ValueError("Python inputs incompatible with input_signature: "
                         "inputs (%s), input_signature (%s)" %
                         (str(inputs), str(self._input_signature)))
      return inputs, {}


class Function(object):
  """Wrapper class for the graph functions defined for a Python function.

  See the documentation for `defun` for more information on the semantics of
  defined functions.

  `Function` class is thread-compatible meaning that minimal usage of defuns
  (defining and calling) is thread-safe, but if users call other methods or
  invoke the base `python_function` themselves, external synchronization is
  necessary.
  """

  def __init__(self,
               python_function,
               name,
               input_signature=None,
               attributes=None,
               autograph=True):
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

    Raises:
      ValueError: if `input_signature` is not None and the `python_function`'s
        argspec has keyword arguments.
    """
    if isinstance(python_function, functools.partial):
      self._python_function = python_function.func
    else:
      self._python_function = python_function
    self._function_spec = FunctionSpec.from_function_and_signature(
        python_function, input_signature)
    self._name = name
    self._autograph = autograph
    self._function_cache = collections.OrderedDict()
    self._garbage_collector = _FunctionGarbageCollector(
        self._function_cache)
    self._function_attributes = attributes or {}

    self._lock = threading.Lock()
    # _descriptor_cache is a of instance of a class to an instance-specific
    # `Function`, used to make sure defun-decorated methods create different
    # functions for each instance.
    self._descriptor_cache = weakref.WeakKeyDictionary()

  def __call__(self, *args, **kwargs):
    """Calls a graph function specialized to the inputs."""
    graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
    return graph_function._filtered_call(args, kwargs)  # pylint: disable=protected-access

  @property
  def python_function(self):
    """Returns the wrapped Python function."""
    return self._python_function  # pylint: disable=protected-access

  @property
  def function_spec(self):
    return self._function_spec

  @property
  def _input_signature(self):
    """Returns the wrapped Python function."""
    return self._function_spec.input_signature  # pylint: disable=protected-access

  @property
  def _flat_input_signature(self):
    """Returns the wrapped Python function."""
    return self._function_spec.flat_input_signature  # pylint: disable=protected-access

  def _get_concrete_function_internal_garbage_collected(self, *args, **kwargs):
    """Returns a concrete function which cleans up its graph function."""
    if self._input_signature:
      args, kwargs = None, None
    graph_function, _, _ = self._maybe_define_function(args, kwargs)
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

  def get_concrete_function(self, *args, **kwargs):
    """Returns a `ConcreteFunction` object specialized to inputs and execution
    context.

    Args:
      *args: inputs to specialize on.
      **kwargs: inputs to specialize on.
    """
    if self._input_signature:
      if kwargs:
        raise ValueError("Cannot define a TensorFlow function from a Python "
                         "function with keyword arguments when "
                         "input_signature is provided.")
      if args:
        # If args are provided, they must match the input signature.
        if not is_same_structure(self._input_signature, args):
          raise ValueError("Structure of Python function inputs does not match "
                           "input_signature.")
        flat_inputs = nest.flatten(args)
        if any(not isinstance(arg, (ops.Tensor, tensor_spec.TensorSpec))
               for arg in flat_inputs):
          raise ValueError("When input_signature is provided, all inputs to "
                           "the Python function must be Tensors or "
                           "tf.TensorSpec objects.")
        if any(not spec.is_compatible_with(other)
               for spec, other in zip(self._flat_input_signature, flat_inputs)):
          raise ValueError("Python inputs incompatible with input_signature: "
                           "inputs (%s), input_signature (%s)" %
                           (str(args), str(self._input_signature)))
      args, kwargs = None, None
    graph_function, args, kwargs = self._maybe_define_function(args, kwargs)
    if self._input_signature:
      args = self._input_signature
      kwargs = {}
    seen_names = set()
    captured = frozenset(graph_function.graph.internal_captures)
    allowed_positional = 0
    if args:
      for outer_arg in args:
        # TODO(allenl): Consider allowing arguments with defaults in the Python
        # function's signature to be passed as positional arguments to the
        # concrete function.
        if not isinstance(
            outer_arg,
            (ops.Tensor, resource_variable_ops.ResourceVariable,
             tensor_spec.TensorSpec)):
          break
        allowed_positional += 1
    # pylint: disable=protected-access
    graph_function._num_positional_args = allowed_positional
    graph_function._arg_keywords = []
    # pylint: enable=protected-access
    for arg in graph_function.graph.inputs:
      if arg in captured:
        break
      user_arg_name = arg.op.get_attr("_user_specified_name")
      if user_arg_name in seen_names:
        raise ValueError(
            ("Unable to construct a concrete function for {} since some "
             "arguments do not have unique names. Got two arguments named "
             "'{}'. When constructing a concrete TensorFlow function from a "
             "Python function which takes nested structures or variadic "
             "positional arguments, pass unique names to tf.TensorSpec objects "
             "used to identify these Tensor inputs. These names may then be "
             "used as keyword arguments to the concrete function.")
            .format(
                self._python_function,
                compat.as_str(arg.op.get_attr("_user_specified_name"))))
      seen_names.add(user_arg_name)
      graph_function._arg_keywords.append(user_arg_name)  # pylint: disable=protected-access
    return graph_function

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

  def _cache_key(self, args, kwargs):
    """Computes the cache key given inputs and execution context."""
    if self._input_signature is None:
      inputs = (args, kwargs) if kwargs else args
      input_signature = pywrap_tensorflow.TFE_Py_EncodeArg(inputs)
    else:
      del args, kwargs
      input_signature = self._flat_input_signature

    ctx = context.context()

    # Don't need to open an init_scope if the _cache_key call is in eager mode
    # already.
    executing_eagerly = ctx.executing_eagerly()
    parent_graph = None
    if not executing_eagerly:
      with ops.init_scope():
        # The graph, or whether we're executing eagerly, should be a part of the
        # cache key so we don't improperly capture tensors such as variables.
        executing_eagerly = ctx.executing_eagerly()
        parent_graph = None if executing_eagerly else ops.get_default_graph()

    # pylint: disable=protected-access
    default_graph = ops.get_default_graph()
    # TODO(b/117617952): The current distribution strategy will affect graph
    # building (e.g. accessing different variables from different devices) and
    # so requires retracing for each device.
    uses_distribution_strategy = bool(
        default_graph._distribution_strategy_stack)
    if executing_eagerly:
      colocation_stack = ()
      uses_xla = ctx.device_spec.device_type == "TPU"
      if uses_distribution_strategy or uses_xla:
        device_functions = (pydev.merge_device(ctx.device_name),)
      else:
        device_functions = ()
    else:
      colocation_stack = tuple(default_graph._colocation_stack.peek_objs())
      uses_xla = getattr(default_graph, "_xla_compile", False)
      if (uses_distribution_strategy
          or uses_xla
          or func_graph_module.device_stack_has_callable(
              default_graph._device_function_stack)):
        # Putting the device in the cache key ensures that call-site device
        # annotations are respected.
        device_functions = tuple(default_graph._device_functions_outer_to_inner)
      else:
        device_functions = ()
    # pylint: enable=protected-access
    return CacheKey(input_signature, parent_graph, device_functions,
                    colocation_stack, uses_xla)

  def _maybe_define_function(self, args, kwargs):
    """Gets a function for these inputs, defining it if necessary.

    `args` and `kwargs` can be None if this `Function` was created with an
    `input_signature`.

    Args:
      args: The varargs for the Python function.
      kwargs: The keyword args for the Python function.

    Returns:
      A graph function corresponding to the input signature implied by args and
      kwargs, as well as the inputs that the object should be called with.

    Raises:
      ValueError: If inputs are incompatible with the input signature.
      TypeError: If the function inputs include non-hashable objects
    """
    if self._input_signature is None or args is not None or kwargs is not None:
      args, kwargs = self._function_spec.canonicalize_function_inputs(
          *args, **kwargs)
    cache_key = self._cache_key(args, kwargs)
    with self._lock:
      try:
        graph_function = self._function_cache.get(cache_key, None)
      except TypeError:
        raise TypeError("Arguments supplied to `defun`-generated functions "
                        "must be hashable.")

      if graph_function is None:
        logging.vlog(1,
                     "Creating new FuncGraph for Python function %r (key: %r)",
                     self._python_function, cache_key)
        if self._input_signature is None:
          arglen = len(args)
        else:
          arglen = len(self._input_signature)
        arg_names = (
            self._function_spec.arg_names[:arglen]
            + [self._function_spec.vararg_name] *
            (arglen - len(self._function_spec.arg_names)))
        graph_function = ConcreteFunction(
            func_graph_module.func_graph_from_py_func(
                self._name,
                self._python_function,
                args,
                kwargs,
                self._input_signature,
                autograph=self._autograph,
                arg_names=arg_names),
            self._function_attributes)
        # pylint: disable=protected-access
        # Tell the ConcreteFunction to clean up its graph once it goes out of
        # scope. ConcreteFunction does not do this in its constructor since it
        # gets used in some places (like Keras) where the FuncGraph lives
        # longer than the ConcreteFunction.
        graph_function._garbage_collector = _ConcreteFunctionGarbageCollector(
            graph_function.graph)
        # pylint: enable=protected-access
        self._function_cache[cache_key] = graph_function
      return graph_function, args, kwargs


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
                     "Got type: %s" % type(func))
  concrete_func = func.get_concrete_function(*args, **kwargs)
  concrete_func.add_to_graph(register_gradient_functions=True)
  return concrete_func


def validate_signature(signature):
  if any(not isinstance(arg, tensor_spec.TensorSpec)
         for arg in nest.flatten(signature)):
    raise TypeError("Invalid input_signature %s; input_signature must be "
                    "a possibly nested sequence of TensorSpec objects.")


def defun(func=None, input_signature=None, autograph=True):
  """Compiles a Python function into a callable TensorFlow graph.

  `defun` (short for "define function") compiles a Python function
  composed of TensorFlow operations into a callable that executes a `tf.Graph`
  containing those operations. The callable produced by `defun` contains only
  the subgraph of TensorFlow operations that were executed when the Python
  function was called with a particular input signature, defined as a list
  of the shapes and dtypes of the Python function's Tensor-valued arguments and
  the values of its non-Tensor Python objects.

  When eager execution is enabled, the ability to create graphs from Python
  functions makes it possible to incrementally trade off debugability and
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

  _Python Control Flow_

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
    autograph: Whether `func` should be compiled before
      constructing the graph. See https://www.tensorflow.org/guide/autograph
      for more information.


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
      autograph=autograph)


def defun_with_attributes(func=None,
                          input_signature=None,
                          attributes=None,
                          autograph=True):
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
      whitelisted attribute name is allowed. Unwhitelisted attribute name or
      unsupported value will result into ValueError. `func_name` is also one of
      the whitelisted argument which is a python string, and sets the name for
      this `ConcreteFunction` in the graph.
    autograph: same as defun()'s autograph.

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
            autograph=autograph))

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
# recover a weak reference the original method's self pointer. This uses the
# mechanism from pyct.inspect_utils.getmethodclass.
# TODO(b/119246461): This is not pretty. Use a descriptor instead?
class _WeakrefSelf(object):

  def __init__(self, target):
    self.ag_self_weakref__ = target


def class_method_to_instance_method(original_function, instance):
  """Constructs a new `Function` with `self` bound."""
  weak_instance = weakref.ref(instance)

  # Note: while we could bind to a weakref proxy instead, that causes the
  # bound method to be unhashable.
  bound_method = types_lib.MethodType(original_function.python_function,
                                      _WeakrefSelf(weak_instance))

  # original_function is expected to be of one of the two `Function` types
  # (defined either in function.py or def_function.py).
  assert hasattr(original_function, "_name")
  assert hasattr(original_function, "_autograph")
  assert hasattr(original_function, "_input_signature")
  assert hasattr(original_function, "python_function")

  weak_bound_method_wrapper = None
  def bound_method_wrapper(*args, **kwargs):
    """Wraps either a dummy MethodType or a converted AutoGraph function."""
    # __wrapped__ allows AutoGraph to swap in a converted function.
    strong_bound_method_wrapper = weak_bound_method_wrapper()
    wrapped_fn = strong_bound_method_wrapper.__wrapped__

    if wrapped_fn is strong_bound_method_wrapper.__original_wrapped__:
      # If __wrapped__ was not replaced, then call original_function.
      wrapped_fn = original_function.python_function
      if tf_inspect.ismethod(wrapped_fn):
        wrapped_fn = six.get_unbound_function(wrapped_fn)
      return wrapped_fn(weak_instance(), *args, **kwargs)

    # If __wrapped__ was replaced, then it is always an unbound function
    # that takes self as first argument.
    return wrapped_fn(weak_instance(), *args, **kwargs)
  weak_bound_method_wrapper = weakref.ref(bound_method_wrapper)

  # pylint: disable=protected-access
  # We make a dummy MethodType object to generate the correct bound method
  # signature. The actual call is to a function with a weak reference to
  # `instance`.
  instance_func = type(original_function)(
      tf_decorator.make_decorator(bound_method, bound_method_wrapper),
      name=original_function._name,
      autograph=original_function._autograph,
      input_signature=original_function._input_signature)
  # pylint: enable=protected-access

  # And we wrap the function with tf_decorator so inspection works correctly
  wrapped_instance_func = tf_decorator.make_decorator(
      original_function.python_function, instance_func)
  return wrapped_instance_func


class _FunctionGarbageCollector(object):
  """Cleans up cycles when a defun goes out of scope."""

  def __init__(self, cache):
    self._cache = cache

  def __del__(self):
    if func_graph_module is None or memory is None:
      return
    try:
      while self._cache:
        self._cache.popitem()
      memory.dismantle_ordered_dict(self._cache)
    except:  # pylint: disable=bare-except
      pass


class _ConcreteFunctionGarbageCollector(object):
  """Cleans up reference cycles when a `ConcreteFunction` goes out of scope."""

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
