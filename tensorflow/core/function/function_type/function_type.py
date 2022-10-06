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
from typing import Any, Optional, Dict, Callable, Mapping

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
