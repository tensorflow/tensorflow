# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Defines an input type specification for tf.function."""

import functools
import inspect
from typing import Any, Dict, Tuple

import numpy as np
import six

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.eager.polymorphic_function import composite_tensor_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import nest

# Sentinel value used by with ConcreteFunction's structured signature to
# indicate that a non-tensor parameter should use the value that was
# specified when the concrete function was created.
BOUND_VALUE = object()


def to_fullargspec(function_type: function_type_lib.FunctionType,
                   default_values: Dict[str, Any]) -> inspect.FullArgSpec:
  """Generates backwards compatible FullArgSpec from FunctionType."""
  args = []
  varargs = None
  varkw = None
  defaults = []
  kwonlyargs = []
  kwonlydefaults = {}

  for parameter in function_type.parameters.values():
    if parameter.kind in [
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]:
      args.append(parameter.name)
      if parameter.default is not inspect.Parameter.empty:
        defaults.append(default_values[parameter.name])
    elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
      kwonlyargs.append(parameter.name)
      if parameter.default is not inspect.Parameter.empty:
        kwonlydefaults[parameter.name] = default_values[parameter.name]
    elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
      varargs = parameter.name
    elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
      varkw = parameter.name

  return inspect.FullArgSpec(
      args,
      varargs,
      varkw,
      tuple(defaults) if defaults else None,
      kwonlyargs,
      kwonlydefaults if kwonlydefaults else None,
      annotations={})


def _to_default_values(fullargspec):
  """Returns default values from the function's inspected fullargspec."""
  if fullargspec.defaults is not None:
    defaults = {
        name: value for name, value in zip(
            fullargspec.args[-len(fullargspec.defaults):], fullargspec.defaults)
    }
  else:
    defaults = {}

  if fullargspec.kwonlydefaults is not None:
    defaults.update(fullargspec.kwonlydefaults)

  defaults = {
      function_type_lib.sanitize_arg_name(name): value
      for name, value in defaults.items()
  }

  return defaults


def to_function_type(fullargspec):
  """Generates FunctionType and default values from fullargspec."""
  default_values = _to_default_values(fullargspec)
  parameters = []

  for arg in fullargspec.args:
    arg_name = function_type_lib.sanitize_arg_name(arg)
    parameters.append(
        function_type_lib.Parameter(
            arg_name, function_type_lib.Parameter.POSITIONAL_OR_KEYWORD,
            arg_name in default_values, None))

  if fullargspec.varargs is not None:
    parameters.append(
        function_type_lib.Parameter(fullargspec.varargs,
                                    function_type_lib.Parameter.VAR_POSITIONAL,
                                    False, None))

  for kwarg in fullargspec.kwonlyargs:
    parameters.append(
        function_type_lib.Parameter(
            function_type_lib.sanitize_arg_name(kwarg),
            function_type_lib.Parameter.KEYWORD_ONLY, kwarg in default_values,
            None))

  if fullargspec.varkw is not None:
    parameters.append(
        function_type_lib.Parameter(fullargspec.varkw,
                                    function_type_lib.Parameter.VAR_KEYWORD,
                                    False, None))

  return function_type_lib.FunctionType(parameters), default_values


def to_input_signature(function_type):
  """Extracts an input_signature from function_type instance."""
  constrained_parameters = list(function_type.parameters.keys())

  # self does not have a constraint in input_signature
  if "self" in constrained_parameters:
    constrained_parameters.pop(0)

  # There are no parameters to constrain.
  if not constrained_parameters:
    return tuple()

  constraints = []
  is_auto_constrained = False

  for parameter_name in constrained_parameters:
    parameter = function_type.parameters[parameter_name]
    constraint = None
    if parameter.type_constraint:
      # Generate legacy constraint representation.
      constraint = parameter.type_constraint.placeholder_value(
          trace_type.InternalPlaceholderContext(unnest_only=True)
      )
      if any(
          not isinstance(arg, tensor_spec.TensorSpec)
          for arg in nest.flatten([constraint], expand_composites=True)):
        # input_signature only supports contiguous TensorSpec composites
        is_auto_constrained = True
        break
      else:
        constraints.append(constraint)

  # All constraints were generated by FunctionType
  if is_auto_constrained and not constraints:
    return tuple()

  # If the list is empty then there was no input_signature specified.
  return tuple(constraints) if constraints else None


# TODO(b/214462107): Clean up and migrate to core/function when unblocked.
class FunctionSpec(object):
  """Specification of how to bind arguments to a function."""

  @classmethod
  def from_function_and_signature(cls,
                                  python_function,
                                  input_signature,
                                  is_pure=False,
                                  jit_compile=None):
    """Creates a FunctionSpec instance given a python function and signature.

    Args:
      python_function: a function to inspect
      input_signature: a signature of the function (None, if variable)
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      jit_compile: see `tf.function`

    Returns:
      instance of FunctionSpec
    """
    _validate_signature(input_signature)

    function_type = function_type_lib.FunctionType.from_callable(
        python_function)
    default_values = function_type_lib.FunctionType.get_default_values(
        python_function)

    if input_signature is not None:
      input_signature = tuple(input_signature)
      function_type = function_type_lib.add_type_constraints(
          function_type, input_signature, default_values)

    # Get the function's name.  Remove functools.partial wrappers if necessary.
    while isinstance(python_function, functools.partial):
      python_function = python_function.func
    name = getattr(python_function, "__name__", "f")

    return FunctionSpec(
        function_type,
        default_values,
        is_pure=is_pure,
        jit_compile=jit_compile,
        name=name)

  @classmethod
  def from_fullargspec_and_signature(cls,
                                     fullargspec,
                                     input_signature,
                                     is_pure=False,
                                     name=None,
                                     jit_compile=None):
    """Construct FunctionSpec from legacy FullArgSpec format."""
    function_type, default_values = to_function_type(fullargspec)
    if input_signature:
      input_signature = tuple(input_signature)
      _validate_signature(input_signature)
      function_type = function_type_lib.add_type_constraints(
          function_type, input_signature, default_values)

    return FunctionSpec(function_type, default_values, is_pure,
                        name, jit_compile)

  def __init__(self,
               function_type,
               default_values,
               is_pure=False,
               name=None,
               jit_compile=None):
    """Constructs a FunctionSpec describing a python function.

    Args:
      function_type: A FunctionType describing the python function signature.
      default_values: Dictionary mapping parameter names to default values.
      is_pure: if True all input arguments (including variables and constants)
        will be converted to tensors and no variable changes allowed.
      name: Name of the function
      jit_compile: see `tf.function`.
    """
    self._function_type = function_type
    self._default_values = default_values
    self._fullargspec = to_fullargspec(function_type, default_values)
    self._is_pure = is_pure
    self._jit_compile = jit_compile

    # TODO(edloper): Include name when serializing for SavedModel?
    self._name = name or "f"
    self._input_signature = to_input_signature(function_type)

  @property
  def default_values(self):
    """Returns dict mapping parameter names to default values."""
    return self._default_values

  @property
  def function_type(self):
    """Returns a FunctionType representing the Python function signature."""
    return self._function_type

  @property
  def fullargspec(self):
    return self._fullargspec

  # TODO(fmuham): Replace usages with FunctionType and remove.
  @property
  def input_signature(self):
    return self._input_signature

  # TODO(fmuham): Replace usages with FunctionType and remove.
  @property
  def flat_input_signature(self):
    return tuple(nest.flatten(self.input_signature, expand_composites=True))

  @property
  def is_pure(self):
    return self._is_pure

  @property
  def jit_compile(self):
    return self._jit_compile

  # TODO(fmuham): Replace usages and remove.
  @property
  def arg_names(self):
    return list(
        p.name
        for p in self.function_type.parameters.values()
        if (
            p.kind is function_type_lib.Parameter.POSITIONAL_ONLY
            or p.kind is function_type_lib.Parameter.POSITIONAL_OR_KEYWORD
        )
    )

  def make_canonicalized_monomorphic_type(
      self,
      args: Any,
      kwargs: Any,
      captures: Any = None,
  ) -> Tuple[function_type_lib.FunctionType,
             trace_type.InternalTracingContext]:
    """Generates function type given the function arguments."""
    if captures is None:
      captures = dict()

    kwargs = {
        function_type_lib.sanitize_arg_name(name): value
        for name, value in kwargs.items()
    }

    _, function_type, type_context = (
        function_type_lib.canonicalize_to_monomorphic(
            args, kwargs, self.default_values, captures, self.function_type
        )
    )

    return function_type, type_context

  def signature_summary(self, default_values=False):
    """Returns a string summarizing this function's signature.

    Args:
      default_values: If true, then include default values in the signature.

    Returns:
      A `string`.
    """
    summary = f"{self._function_type!r}"
    if default_values:
      summary += f", defaults: {self.default_values!r}"
    return summary

  def canonicalize_function_inputs(self, args, kwargs):
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
      args: The varargs this object was called with.
      kwargs: The keyword args this function was called with.

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
    if self.is_pure:
      args, kwargs = _convert_variables_to_tensors(args, kwargs)
    args, kwargs = self.bind_function_inputs(args, kwargs)
    filtered_flat_args = filter_function_inputs(args, kwargs)

    return args, kwargs, filtered_flat_args

  def bind_function_inputs(self, args, kwargs):
    """Bind `args` and `kwargs` into a canonicalized signature args, kwargs."""
    sanitized_kwargs = {
        function_type_lib.sanitize_arg_name(k): v for k, v in kwargs.items()
    }
    if len(kwargs) != len(sanitized_kwargs):
      raise ValueError(f"Name collision after sanitization. Please rename "
                       f"tf.function input parameters. Original: "
                       f"{sorted(kwargs.keys())}, Sanitized: "
                       f"{sorted(sanitized_kwargs.keys())}")

    try:
      bound_arguments = self.function_type.bind_with_defaults(
          args, sanitized_kwargs, self.default_values)
    except Exception as e:
      raise TypeError(
          f"Binding inputs to tf.function `{self._name}` failed due to `{e}`. "
          f"Received args: {args} and kwargs: {sanitized_kwargs} for signature:"
          f" {self.function_type}."
      ) from e
    return bound_arguments.args, bound_arguments.kwargs


def _validate_signature(signature):
  """Checks the input_signature to be valid."""
  if signature is None:
    return

  if not isinstance(signature, (tuple, list)):
    raise TypeError("input_signature must be either a tuple or a list, got "
                    f"{type(signature)}.")

  # TODO(xjun): Allow VariableSpec once we figure out API for de-aliasing.
  variable_specs = _get_variable_specs(signature)
  if variable_specs:
    raise TypeError(
        f"input_signature doesn't support VariableSpec, got {variable_specs}")

  if any(not isinstance(arg, tensor_spec.TensorSpec)
         for arg in nest.flatten(signature, expand_composites=True)):
    bad_args = [
        arg for arg in nest.flatten(signature, expand_composites=True)
        if not isinstance(arg, tensor_spec.TensorSpec)
    ]
    raise TypeError("input_signature must be a possibly nested sequence of "
                    f"TensorSpec objects, got invalid args {bad_args} with "
                    f"types {list(six.moves.map(type, bad_args))}.")


def _to_tensor_or_tensor_spec(x):
  return (x if isinstance(x, (ops.Tensor, tensor_spec.TensorSpec)) else
          ops.convert_to_tensor(x))


def _convert_variables_to_tensors(args, kwargs):
  args = [_to_tensor_or_tensor_spec(x) for x in args]
  kwargs = {kw: _to_tensor_or_tensor_spec(x) for kw, x in kwargs.items()}
  return tuple(args), kwargs


# TODO(fmuham): Migrate to use TraceType/FunctionType _to_tensors.
def filter_function_inputs(args, kwargs):
  """Filters and flattens args and kwargs."""
  flat_inputs = composite_tensor_utils.flatten_with_variables(
      args) + composite_tensor_utils.flatten_with_variables(kwargs)

  for index, flat_input in enumerate(flat_inputs):
    if hasattr(flat_input, "__array__") and not (
        hasattr(flat_input, "_should_act_as_resource_variable")
        or isinstance(
            flat_input,
            (
                ops.Tensor,
                resource_variable_ops.BaseResourceVariable,
                np.str_,
                type,
                composite_tensor.CompositeTensor,
            ),
        )
    ):
      ndarray = flat_input.__array__()
      if not isinstance(ndarray, np.ndarray):
        raise TypeError(f"The output of __array__ must be an np.ndarray, "
                        f"got {type(ndarray)} from {flat_input}.")
      flat_inputs[index] = constant_op.constant(ndarray)

  filtered_flat_inputs = [
      t for t in flat_inputs
      if isinstance(t, (ops.Tensor, resource_variable_ops.BaseResourceVariable))
  ]

  return filtered_flat_inputs


def _get_variable_specs(args):
  """Returns `VariableSpecs` from `args`."""
  variable_specs = []
  for arg in nest.flatten(args):
    if not isinstance(arg, type_spec.TypeSpec):
      continue
    if isinstance(arg, resource_variable_ops.VariableSpec):
      variable_specs.append(arg)
    elif not isinstance(arg, tensor_spec.TensorSpec):
      # arg is a CompositeTensor spec.
      variable_specs.extend(_get_variable_specs(arg._component_specs))  # pylint: disable=protected-access
  return variable_specs


# TODO(fmuham): Replace usages with TraceType and remove.
def is_same_structure(structure1, structure2, check_values=False):
  """Check two structures for equality, optionally of types and of values."""
  try:
    nest.assert_same_structure(structure1, structure2, expand_composites=True)
  except (ValueError, TypeError):
    return False
  if check_values:
    flattened1 = nest.flatten(structure1, expand_composites=True)
    flattened2 = nest.flatten(structure2, expand_composites=True)
    # First check the types to avoid AttributeErrors.
    if any(type(f1) is not type(f2) for f1, f2 in zip(flattened1, flattened2)):
      return False
    return flattened1 == flattened2
  return True
