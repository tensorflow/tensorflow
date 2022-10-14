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
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, OrderedDict

from tensorflow.core.function import trace_type
from tensorflow.python.types import trace

# Represents a defined parameter default value that is saved alongside the
# function's captures.
CAPTURED_DEFAULT_VALUE = object()


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
    return ("Parameter(name=" + self.name + ", kind" + str(self.kind) +
            ", optional=" + repr(self.optional) + ", type_constraint=" +
            repr(self.type_constraint) + ")")

  def __reduce__(self):
    return (self.__class__, (self.name, self.kind, self.optional,
                             self.type_constraint))


class FunctionType(inspect.Signature):
  """Represents the parameters of a polymorphic function."""

  def __init__(self,
               parameters: Sequence[inspect.Parameter],
               captures: Optional[OrderedDict[str, trace.TraceType]] = None,
               **kwargs):
    super().__init__(parameters, **kwargs)
    self._captures = captures if captures else collections.OrderedDict()

  @property
  def parameters(self) -> Mapping[str, Any]:
    return super().parameters

  @property
  def captures(self) -> OrderedDict[str, trace.TraceType]:
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

    if inspect.ismethod(obj):
      parameters = [
          Parameter("self", Parameter.POSITIONAL_OR_KEYWORD, False, None)
      ] + parameters

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

  def placeholder_arguments(self) -> inspect.BoundArguments:
    """Returns BoundArguments of values that can be used for tracing."""
    arguments = collections.OrderedDict()
    for parameter in self.parameters.values():
      if parameter.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
        raise ValueError("Can not generate placeholder values for "
                         "variable length function type.")

      if not parameter.type_constraint:
        raise ValueError("Can not generate placeholder value for "
                         "partially defined function type.")

      arguments[parameter.name] = parameter.type_constraint._placeholder_value()    # pylint: disable=protected-access

    return inspect.BoundArguments(self, arguments)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, FunctionType):
      return NotImplemented

    return (self.parameters, self.captures) == (other.parameters,
                                                other.captures)

  def __hash__(self) -> int:
    return hash(
        (tuple(self.parameters.items()), tuple(self.captures.items())))

  def __repr__(self):
    return (
        f"FunctionType(parameters={list(self.parameters.values())!r}, "
        f"captures={self.captures})"
    )


# TODO(fmuham): Consider forcing kind to be always POSITIONAL_OR_KEYWORD.
def _make_validated_mono_param(name, value, kind, type_context, poly_type):
  """Generates and validates a parameter for Monomorphic FunctionType."""
  mono_type = trace_type.from_value(value, type_context)

  if poly_type and not mono_type.is_subtype_of(poly_type):
    raise TypeError(f"Parameter {name} was expected to be of type "
                    f"{poly_type} but is {mono_type}")

  return Parameter(name, kind, False, mono_type)


def canonicalize_to_monomorphic(
    args: Tuple[Any, ...], kwargs: Dict[Any,
                                        Any], polymorphic_type: FunctionType,
    monomorphic_type_context: trace_type.InternalTracingContext
) -> Tuple[inspect.BoundArguments, FunctionType]:
  """Converts polymorphic parameters to monomorphic and associated type."""
  poly_bound_arguments = polymorphic_type.bind(*args, **kwargs)
  parameters = []

  for name, arg in poly_bound_arguments.arguments.items():
    poly_parameter = polymorphic_type.parameters[name]
    if poly_parameter.kind is Parameter.VAR_POSITIONAL:
      for i, value in enumerate(arg):
        parameters.append(
            _make_validated_mono_param(f"{poly_parameter.name}_{i}", value,
                                       Parameter.POSITIONAL_ONLY,
                                       monomorphic_type_context,
                                       poly_parameter.type_constraint))
    elif poly_parameter.kind is Parameter.VAR_KEYWORD:
      for kwarg_name, kwarg_value in arg.items():
        parameters.append(
            _make_validated_mono_param(kwarg_name, kwarg_value,
                                       Parameter.KEYWORD_ONLY,
                                       monomorphic_type_context,
                                       poly_parameter.type_constraint))
    else:
      parameters.append(
          _make_validated_mono_param(name, arg, poly_parameter.kind,
                                     monomorphic_type_context,
                                     poly_parameter.type_constraint))

  monomorphic_function_type = FunctionType(parameters)
  mono_bound_arguments = monomorphic_function_type.bind(
      *poly_bound_arguments.args, **poly_bound_arguments.kwargs)

  return mono_bound_arguments, monomorphic_function_type
