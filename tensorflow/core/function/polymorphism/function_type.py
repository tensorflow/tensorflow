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
"""Represents the types of TF functions."""

import collections
import inspect
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from absl import logging

from tensorflow.core.function import trace_type
from tensorflow.core.function.polymorphism import function_type_pb2
from tensorflow.core.function.trace_type import serialization
from tensorflow.python.types import trace

# Represents a defined parameter default value that is saved alongside the
# function's captures.
CAPTURED_DEFAULT_VALUE = object()

PROTO_TO_PY_ENUM = {
    function_type_pb2.Parameter.Kind.POSITIONAL_ONLY:
        inspect.Parameter.POSITIONAL_ONLY,
    function_type_pb2.Parameter.Kind.POSITIONAL_OR_KEYWORD:
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    function_type_pb2.Parameter.Kind.VAR_POSITIONAL:
        inspect.Parameter.VAR_POSITIONAL,
    function_type_pb2.Parameter.Kind.KEYWORD_ONLY:
        inspect.Parameter.KEYWORD_ONLY,
    function_type_pb2.Parameter.Kind.VAR_KEYWORD:
        inspect.Parameter.VAR_KEYWORD,
}

PY_TO_PROTO_ENUM = {v: k for k, v in PROTO_TO_PY_ENUM.items()}


class Parameter(inspect.Parameter):
  """Represents a parameter to a function."""

  def __init__(self, name: str, kind: Any, optional: bool,
               type_constraint: Optional[trace.TraceType]):
    if optional and kind not in [
        self.POSITIONAL_ONLY, self.KEYWORD_ONLY, self.POSITIONAL_OR_KEYWORD
    ]:
      raise ValueError(
          "Parameter " + name +
          " is optional and its kind must be one of {POSITIONAL_ONLY, " +
          "KEYWORD_ONLY, POSITIONAL_OR_KEYWORD}. Got: " + str(kind))

    if type_constraint and kind in [self.VAR_POSITIONAL, self.VAR_KEYWORD]:
      raise TypeError("Variable args/kwargs can not have type constraints.")

    if not isinstance(type_constraint, (trace.TraceType, type(None))):
      raise TypeError(
          "Type constraints can only be an instance of a TraceType but got " +
          "type_constraint=" + str(type_constraint) + " for Parameter " + name)

    super().__init__(
        name,
        kind,
        default=CAPTURED_DEFAULT_VALUE if optional else self.empty,
        annotation=type_constraint
        if type_constraint is not None else self.empty)

  @classmethod
  def from_proto(cls, proto: Any) -> "Parameter":
    deserialized_type_constraint = serialization.deserialize(
        proto.type_constraint) if proto.HasField("type_constraint") else None
    return Parameter(proto.name, PROTO_TO_PY_ENUM[proto.kind],
                     proto.is_optional, deserialized_type_constraint)

  def to_proto(self) -> function_type_pb2.Parameter:
    serialized_type_constraint = serialization.serialize(
        self.type_constraint) if self.type_constraint else None
    return function_type_pb2.Parameter(
        name=self.name,
        kind=PY_TO_PROTO_ENUM[self.kind],
        is_optional=self.optional,
        type_constraint=serialized_type_constraint)

  @property
  def optional(self) -> bool:
    """If this parameter might not be supplied for a call."""
    return self.default is not self.empty

  @property
  def type_constraint(self) -> Optional[trace.TraceType]:
    """A supertype that the parameter's type must subtype for validity."""
    return self.annotation if self.annotation is not self.empty else None

  def is_subtype_of(self, other: "Parameter") -> bool:
    """Returns True if self is a supertype of other Parameter."""
    if not self.type_constraint or not other.type_constraint:
      raise TypeError(
          "Can not determine relationship between partially specified types.")

    if ((self.name, self.kind, self.optional) !=
        (other.name, other.kind, other.optional)):
      return False

    return self.type_constraint.is_subtype_of(other.type_constraint)

  def most_specific_common_supertype(
      self, others: Sequence["Parameter"]) -> Optional["Parameter"]:
    """Returns a common supertype (if exists)."""
    if not self.type_constraint or any(
        not other.type_constraint for other in others):
      raise TypeError(
          "Can not determine relationship between partially specified types.")

    for other in others:
      if ((self.name, self.kind, self.optional) !=
          (other.name, other.kind, other.optional)):
        return None

    supertyped_constraint = self.type_constraint.most_specific_common_supertype(
        [other.type_constraint for other in others])
    if supertyped_constraint:
      return Parameter(self.name, self.kind, self.optional,
                       supertyped_constraint)
    else:
      return None

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, Parameter):
      return NotImplemented

    return ((self.name, self.kind, self.optional,
             self.type_constraint) == (other.name, other.kind, other.optional,
                                       other.type_constraint))

  def __hash__(self):
    return hash((self.name, self.kind, self.optional, self.type_constraint))

  def __repr__(self):
    return ("Parameter(name=" + self.name + ", kind=" + str(self.kind) +
            ", optional=" + repr(self.optional) + ", type_constraint=" +
            repr(self.type_constraint) + ")")

  def __reduce__(self):
    return (self.__class__, (self.name, self.kind, self.optional,
                             self.type_constraint))


class FunctionType(inspect.Signature):
  """Represents the signature of a polymorphic/monomorphic function."""

  def __init__(self,
               parameters: Sequence[inspect.Parameter],
               captures: Optional[collections.OrderedDict] = None,
               **kwargs):
    super().__init__(parameters, **kwargs)
    self._captures = captures if captures else collections.OrderedDict()

  @property
  def parameters(self) -> Mapping[str, Any]:
    return super().parameters

  @property
  def captures(self) -> collections.OrderedDict:
    return self._captures

  # TODO(fmuham): Use this method instead of fullargspec and tf_inspect.
  @classmethod
  def from_callable(cls,
                    obj: Callable[..., Any],
                    *,
                    follow_wrapped: bool = True) -> "FunctionType":
    """Generate FunctionType from a python Callable."""
    signature = super().from_callable(obj, follow_wrapped=follow_wrapped)
    # TODO(fmuham): Support TraceType-based annotations.
    parameters = [
        Parameter(p.name, p.kind, p.default is not p.empty, None)
        for p in signature.parameters.values()
    ]

    return FunctionType(parameters)

  @classmethod
  def get_default_values(cls,
                         obj: Callable[..., Any],
                         *,
                         follow_wrapped: bool = True) -> Dict[str, Any]:
    """Inspects and returns a dictionary of default values."""
    signature = super().from_callable(obj, follow_wrapped=follow_wrapped)
    default_values = {}
    for p in signature.parameters.values():
      if p.default is not p.empty:
        default_values[p.name] = p.default
    return default_values

  @classmethod
  def from_proto(cls, proto: Any) -> "FunctionType":
    return FunctionType([Parameter.from_proto(p) for p in proto.parameters],
                        collections.OrderedDict([
                            (c.name,
                             serialization.deserialize(c.type_constraint))
                            for c in proto.captures
                        ]))

  def to_proto(self) -> Any:
    return function_type_pb2.FunctionType(
        parameters=[p.to_proto() for p in self.parameters.values()],
        captures=[
            function_type_pb2.Capture(
                name=n, type_constraint=serialization.serialize(t))
            for n, t in self.captures.items()
        ])

  def bind_with_defaults(self, args, kwargs, default_values):
    """Returns BoundArguments with default values filled in."""
    bound_arguments = self.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    with_default_args = collections.OrderedDict()
    for name, value in bound_arguments.arguments.items():
      if value is CAPTURED_DEFAULT_VALUE:
        with_default_args[name] = default_values[name]
      else:
        with_default_args[name] = value

    for arg_name in with_default_args:
      constraint = self.parameters[arg_name].type_constraint
      if constraint:
        with_default_args[arg_name] = constraint._cast(  # pylint: disable=protected-access
            with_default_args[arg_name],
            trace_type.InternalCastContext(
                allow_specs=True, allow_supertype_tensors=True
            ),
        )
    bound_arguments = inspect.BoundArguments(self, with_default_args)
    return bound_arguments

  def is_supertype_of(self, other: "FunctionType") -> bool:
    """Returns True if self is a supertype of other FunctionType."""
    if len(self.parameters) != len(other.parameters):
      return False

    for self_param, other_param in zip(self.parameters.values(),
                                       other.parameters.values()):
      # Functions are contravariant on their parameter types.
      if not self_param.is_subtype_of(other_param):
        return False

    # Self must have all capture names of other.
    if not all(name in self.captures for name in other.captures):
      return False

    # Functions are contravariant upon the capture types.
    return all(self.captures[name].is_subtype_of(capture_type)
               for name, capture_type in other.captures.items())

  def most_specific_common_subtype(
      self, others: Sequence["FunctionType"]) -> Optional["FunctionType"]:
    """Returns a common subtype (if exists)."""
    subtyped_parameters = []

    for i, parameter in enumerate(self.parameters.values()):
      # Functions are contravariant on their parameter types.
      subtyped_parameter = parameter.most_specific_common_supertype(
          [list(other.parameters.values())[i] for other in others])
      if subtyped_parameter is None:
        return None
      subtyped_parameters.append(subtyped_parameter)

    if not all(subtyped_parameters):
      return None

    # Common subtype must use captures common to all.
    capture_names = set(self.captures.keys())
    for other in others:
      capture_names = capture_names.intersection(other.captures.keys())

    subtyped_captures = collections.OrderedDict()
    for name in capture_names:
      # Functions are contravariant upon the capture types.
      common_type = self.captures[name].most_specific_common_supertype(
          [other.captures[name] for other in others])
      if common_type is None:
        return None
      else:
        subtyped_captures[name] = common_type

    return FunctionType(subtyped_parameters, subtyped_captures)

  def placeholder_arguments(
      self, placeholder_context: trace.PlaceholderContext
  ) -> inspect.BoundArguments:
    """Returns BoundArguments of values that can be used for tracing."""
    arguments = collections.OrderedDict()
    for parameter in self.parameters.values():
      if parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
        raise ValueError("Can not generate placeholder values for "
                         "variable length function type.")

      if not parameter.type_constraint:
        raise ValueError("Can not generate placeholder value for "
                         "partially defined function type.")
      placeholder_context.update_naming_scope(parameter.name)
      arguments[parameter.name] = parameter.type_constraint.placeholder_value(
          placeholder_context)

    return inspect.BoundArguments(self, arguments)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, FunctionType):
      return NotImplemented

    return (self.parameters, self.captures) == (other.parameters,
                                                other.captures)

  def __hash__(self) -> int:
    return hash((tuple(self.parameters.items()), tuple(self.captures.items())))

  def __repr__(self):
    return (f"FunctionType(parameters={list(self.parameters.values())!r}, "
            f"captures={self.captures})")


MAX_SANITIZATION_WARNINGS = 5
sanitization_warnings_given = 0


# TODO(fmuham): In future, replace warning with exception.
# TODO(fmuham): Sanitize to graph node conventions.
def sanitize_arg_name(name: str) -> str:
  """Sanitizes function argument names.

  Matches Python symbol naming rules.

  Without sanitization, names that are not legal Python parameter names can be
  set which makes it challenging to represent callables supporting the named
  calling capability.

  Args:
    name: The name to sanitize.

  Returns:
    A string that meets Python parameter conventions.
  """
  # Replace non-alphanumeric chars with '_'
  swapped = "".join([c if c.isalnum() else "_" for c in name])
  result = swapped if swapped[0].isalpha() else "arg_" + swapped

  global sanitization_warnings_given
  if name != result and sanitization_warnings_given < MAX_SANITIZATION_WARNINGS:
    logging.warning(
        "`%s` is not a valid tf.function parameter name. Sanitizing to `%s`.",
        name, result)
    sanitization_warnings_given += 1

  return result


# TODO(fmuham): Consider forcing kind to be always POSITIONAL_OR_KEYWORD.
def _make_validated_mono_param(name, value, kind, type_context, poly_type):
  """Generates and validates a parameter for Monomorphic FunctionType."""
  mono_type = trace_type.from_value(value, type_context)

  if poly_type and not mono_type.is_subtype_of(poly_type):
    raise TypeError(f"Parameter `{name}` was expected to be of type "
                    f"{poly_type} but is {mono_type}")

  return Parameter(name, kind, False, mono_type)


def canonicalize_to_monomorphic(
    args: Tuple[Any, ...], kwargs: Dict[Any, Any], default_values: Dict[Any,
                                                                        Any],
    captures: Dict[Any, Any], polymorphic_type: FunctionType
) -> Tuple[inspect.BoundArguments, FunctionType,
           trace_type.InternalTracingContext]:
  """Converts polymorphic parameters to monomorphic and associated type."""
  poly_bound_arguments = polymorphic_type.bind(*args, **kwargs)
  poly_bound_arguments.apply_defaults()

  # Inject Default Values.
  default_values_injected = poly_bound_arguments.arguments
  for name, value in default_values_injected.items():
    if value is CAPTURED_DEFAULT_VALUE:
      default_values_injected[name] = default_values[name]
  poly_bound_arguments = inspect.BoundArguments(poly_bound_arguments.signature,
                                                default_values_injected)

  parameters = []
  type_context = trace_type.InternalTracingContext()
  has_var_positional = any(p.kind is Parameter.VAR_POSITIONAL
                           for p in polymorphic_type.parameters.values())

  for name, arg in poly_bound_arguments.arguments.items():
    poly_parameter = polymorphic_type.parameters[name]
    if (has_var_positional and
        poly_parameter.kind is Parameter.POSITIONAL_OR_KEYWORD):
      # If there is a VAR_POSITIONAL, all POSITIONAL_OR_KEYWORD become
      # POSITIONAL_ONLY.
      parameters.append(
          _make_validated_mono_param(name, arg, Parameter.POSITIONAL_ONLY,
                                     type_context,
                                     poly_parameter.type_constraint))

    elif poly_parameter.kind is Parameter.VAR_POSITIONAL:
      # Unbundle VAR_POSITIONAL into individual POSITIONAL_ONLY args.
      for i, value in enumerate(arg):
        parameters.append(
            _make_validated_mono_param(f"{poly_parameter.name}_{i}", value,
                                       Parameter.POSITIONAL_ONLY, type_context,
                                       poly_parameter.type_constraint))

    elif poly_parameter.kind is Parameter.VAR_KEYWORD:
      # Unbundle VAR_KEYWORD into individual KEYWORD_ONLY args.
      for kwarg_name in sorted(arg.keys()):
        parameters.append(
            _make_validated_mono_param(kwarg_name, arg[kwarg_name],
                                       Parameter.KEYWORD_ONLY, type_context,
                                       poly_parameter.type_constraint))
    else:
      parameters.append(
          _make_validated_mono_param(name, arg, poly_parameter.kind,
                                     type_context,
                                     poly_parameter.type_constraint))

  capture_types = collections.OrderedDict()
  for name, value in captures.items():
    capture_types[name] = trace_type.from_value(value, type_context)

  monomorphic_function_type = FunctionType(parameters, capture_types)
  mono_bound_arguments = monomorphic_function_type.bind(
      *poly_bound_arguments.args, **poly_bound_arguments.kwargs)

  return mono_bound_arguments, monomorphic_function_type, type_context


# TODO(fmuham): Share code with canonicalize_to_monomorphic.
# TODO(fmuham): Lift unnecessary restrictions on input_signature validity.
def add_type_constraints(function_type: FunctionType, input_signature: Any,
                         default_values: Dict[str, Any]):
  """Adds type constraints to a FunctionType based on the input_signature."""
  context = trace_type.InternalTracingContext(is_legacy_signature=True)
  constraints = [trace_type.from_value(c, context) for c in input_signature]
  parameters = []

  has_var_pos = any(
      p.kind is p.VAR_POSITIONAL for p in function_type.parameters.values())

  for param in function_type.parameters.values():
    # VAR_POSITIONAL does not allow POSITIONAL_OR_KEYWORD args.
    sanitized_kind = (
        param.POSITIONAL_ONLY if has_var_pos and
        param.kind is param.POSITIONAL_OR_KEYWORD else param.kind)

    if param.name == "self":
      # Type constraints do not apply on them.
      parameters.append(Parameter("self", sanitized_kind, param.optional, None))

    elif param.kind is param.VAR_KEYWORD:
      # Disabled when input_signature is specified.
      continue

    elif param.kind is param.VAR_POSITIONAL:
      # Convert into Positional Only args based on length of constraints.
      for i in range(len(constraints)):
        parameters.append(
            Parameter(param.name + "_" + str(i), Parameter.POSITIONAL_ONLY,
                      False, constraints.pop(0)))

    elif (param.kind in [
        param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY
    ]):
      if param.kind is param.KEYWORD_ONLY and param.name not in default_values:
        raise TypeError(
            "Since input_signature is defined, keyword-only parameter"
            f" `{param.name}` must have a default value"
        )

      if constraints:
        parameters.append(
            Parameter(param.name, sanitized_kind, param.optional,
                      constraints.pop(0)))
      elif param.name in default_values:
        type_constraint = trace_type.from_value(default_values[param.name])
        parameters.append(
            Parameter(param.name, sanitized_kind, param.optional,
                      type_constraint))
      else:
        raise TypeError(
            f"input_signature missing type constraint for {param.name}")

  if constraints:
    raise TypeError(
        f"input_signature contains {len(constraints)} extra type constraints.")

  return FunctionType(parameters)
