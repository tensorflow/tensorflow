# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""tf.function tracing types.

See `core.GenericFunction` and `core.ConcreteFunction`.

`GenericFunction` assigns types to call arguments, forming a signature.
Function signatures are used to match arguments to `ConcreteFunction`s.
For example, when a new `ConcreteFunction` is traced, it is assigned a
the signature of the arguments it was traced with. Subsequent call arguments
which match its signature will be dispatched to the same `ConcreteFunction`.
If no `ConcreteFunction` with a matching signature is found, a new one may be
traced (a process known as retracing).
"""

import abc
from typing import Optional, Sequence
from typing_extensions import Protocol
from typing_extensions import runtime_checkable


class TraceType(abc.ABC):
  """Represents the type of object(s) for Function Tracing purposes.

  `TraceType` is an abstract class that other classes might inherit from to
  provide information regarding associated class(es) for the purposes of
  Function Tracing. The typing logic provided through this mechanism will be
  used to make decisions regarding usage of cached functions and retracing.
  """

  @abc.abstractmethod
  def is_subtype_of(self, other: "TraceType") -> bool:
    pass

  @abc.abstractmethod
  def most_specific_common_supertype(
      self, others: Sequence["TraceType"]) -> Optional["TraceType"]:
    pass

  @abc.abstractmethod
  def __hash__(self) -> int:
    pass

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    pass


class TracingContext():
  """Contains information scoped to the tracing of multiple objects.

  `TracingContext` is a container class for flags and variables that have
  any kind of influence on the tracing behaviour of the class implementing
  the __tf_tracing_type__. This context will be shared across all
  __tf_tracing_type__ calls while constructing the TraceType for a particular
  set of objects.
  """
  pass


@runtime_checkable
class SupportsTracingType(Protocol):
  """The Trace Control Protocol for functions.

  Classes that implement this protocol can expect the TensorFlow function
  caching and function retracing mechanisms to treat instances of those
  classes according to the behaviour specified by their TraceType.
  """

  @abc.abstractmethod
  def __tf_tracing_type__(self, context: TracingContext) -> TraceType:
    pass
