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

import inspect
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

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

  def __repr__(self):
    return ("Parameter(name=" + self.name + ", kind" + str(self.kind) +
            ", optional=" + repr(self.optional) + ", type_constraint=" +
            repr(self.type_constraint) + ")")

  def __reduce__(self):
    return (self.__class__, (self.name, self.kind, self.optional,
                             self.type_constraint))


class FunctionType(inspect.Signature):
  """Represents the parameters of a polymorphic function."""

  @property
  def parameters(self) -> Mapping[str, Any]:
    return super().parameters

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

  def __repr__(self):
    return ("FunctionType(" +
            ", ".join([repr(p) for p in self.parameters.values()]) + ")")


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
            _make_validated_mono_param(f"{poly_parameter.name}_{i}",
                                       value, Parameter.POSITIONAL_ONLY,
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
