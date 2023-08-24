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
from typing import Any, List, Optional, Sequence, Iterator

from typing_extensions import Protocol
from typing_extensions import runtime_checkable

from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls


@tf_export("types.experimental.TraceType", v1=[])
class TraceType(metaclass=abc.ABCMeta):
  """Represents the type of object(s) for tf.function tracing purposes.

  `TraceType` is an abstract class that other classes might inherit from to
  provide information regarding associated class(es) for the purposes of
  tf.function tracing. The typing logic provided through this mechanism will be
  used to make decisions regarding usage of cached concrete functions and
  retracing.

  For example, if we have the following tf.function and classes:
  ```python
  @tf.function
  def get_mixed_flavor(fruit_a, fruit_b):
    return fruit_a.flavor + fruit_b.flavor

  class Fruit:
    flavor = tf.constant([0, 0])

  class Apple(Fruit):
    flavor = tf.constant([1, 2])

  class Mango(Fruit):
    flavor = tf.constant([3, 4])
  ```

  tf.function does not know when to re-use an existing concrete function in
  regards to the `Fruit` class so naively it retraces for every new instance.
  ```python
  get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function
  get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function again
  ```

  However, we, as the designers of the `Fruit` class, know that each subclass
  has a fixed flavor and we can reuse an existing traced concrete function if
  it was the same subclass. Avoiding such unnecessary tracing of concrete
  functions can have significant performance benefits.

  ```python
  class FruitTraceType(tf.types.experimental.TraceType):
    def __init__(self, fruit):
      self.fruit_type = type(fruit)
      self.fruit_value = fruit

    def is_subtype_of(self, other):
       return (type(other) is FruitTraceType and
               self.fruit_type is other.fruit_type)

    def most_specific_common_supertype(self, others):
       return self if all(self == other for other in others) else None

    def placeholder_value(self, placeholder_context=None):
      return self.fruit_value

  class Fruit:

   def __tf_tracing_type__(self, context):
     return FruitTraceType(self)
  ```

  Now if we try calling it again:
  ```python
  get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function
  get_mixed_flavor(Apple(), Mango()) # Re-uses the traced concrete function
  ```
  """

  @abc.abstractmethod
  def is_subtype_of(self, other: "TraceType") -> bool:
    """Returns True if `self` is a subtype of `other`.

    For example, `tf.function` uses subtyping for dispatch:
    if `a.is_subtype_of(b)` is True, then an argument of `TraceType`
    `a` can be used as argument to a `ConcreteFunction` traced with an
    a `TraceType` `b`.

    Args:
     other: A TraceType object to be compared against.

    Example:

    ```python
    class Dimension(TraceType):
      def __init__(self, value: Optional[int]):
        self.value = value

      def is_subtype_of(self, other):
        # Either the value is the same or other has a generalized value that
        # can represent any specific ones.
        return (self.value == other.value) or (other.value is None)
    ```
    """

  @abc.abstractmethod
  def most_specific_common_supertype(
      self, others: Sequence["TraceType"]) -> Optional["TraceType"]:
    """Returns the most specific supertype of `self` and `others`, if exists.

    The returned `TraceType` is a supertype of `self` and `others`, that is,
    they are all subtypes (see `is_subtype_of`) of it.
    It is also most specific, that is, there it has no subtype that is also
    a common supertype of `self` and `others`.

    If `self` and `others` have no common supertype, this returns `None`.

    Args:
     others: A sequence of TraceTypes.

    Example:
    ```python
     class Dimension(TraceType):
       def __init__(self, value: Optional[int]):
         self.value = value

       def most_specific_common_supertype(self, other):
          # Either the value is the same or other has a generalized value that
          # can represent any specific ones.
          if self.value == other.value:
            return self.value
          else:
            return Dimension(None)
    ```
    """

  @abc.abstractmethod
  def placeholder_value(self, placeholder_context) -> Any:
    """Creates a placeholder for tracing.

    tf.funcion traces with the placeholder value rather than the actual value.
    For example, a placeholder value can represent multiple different
    actual values. This means that the trace generated with that placeholder
    value is more general and reusable which saves expensive retracing.

    Args:
      placeholder_context: A `PlaceholderContext` container for context
                           information when creating a placeholder value.

    For the `Fruit` example shared above, implementing:

    ```python
    class FruitTraceType:
      def placeholder_value(self, placeholder_context):
        return Fruit()
    ```
    instructs tf.function to trace with the `Fruit()` objects
    instead of the actual `Apple()` and `Mango()` objects when it receives a
    call to `get_mixed_flavor(Apple(), Mango())`. For example, Tensor arguments
    are replaced with Tensors of similar shape and dtype, output from
    a tf.Placeholder op.

    More generally, placeholder values are the arguments of a tf.function,
    as seen from the function's body:
    ```python
    @tf.function
    def foo(x):
      # Here `x` is be the placeholder value
      ...

    foo(x) # Here `x` is the actual value
    ```
    """

  @doc_controls.do_not_doc_inheritable
  def _to_tensors(self, value: Any) -> List[core.Tensor]:
    """Breaks down a value of this type into Tensors.

    Args:
      value: An input value belonging to this TraceType

    Returns:
      List of Tensors.
    """
    del value
    return []

  @doc_controls.do_not_doc_inheritable
  def _from_tensors(self, tensors: Iterator[core.Tensor]) -> Any:
    """Regenerates a value of this type from Tensors.

    Must use the same fixed amount of tensors as `_to_tensors`.

    Args:
      tensors: An iterator from which the tensors can be pulled.

    Returns:
      A value of this type.
    """
    del tensors
    return self.placeholder_value(PlaceholderContext())

  @doc_controls.do_not_doc_inheritable
  def _flatten(self) -> List["TraceType"]:
    """Returns a list of TensorSpecs corresponding to `_to_tensors` values."""
    return []

  @doc_controls.do_not_doc_inheritable
  def _cast(self, value, casting_context) -> Any:  # pylint:disable=unused-argument
    """Cast value to this type.

    Args:
      value: An input value belonging to this TraceType.
      casting_context: A context reserved for future usage such as to determine
        casting rules.

    Returns:
      The value casted to this TraceType.

    Raises:
      AssertionError: When _cast is not overloaded in subclass,
        the value is returned directly, and it should be the same to
        self.placeholder_value().
    """
    assert value == self.placeholder_value(
        PlaceholderContext()), f"Can not cast {value!r} to type {self!r}"
    return value

  @abc.abstractmethod
  def __hash__(self) -> int:
    pass

  @abc.abstractmethod
  def __eq__(self, other) -> bool:
    pass


@tf_export("types.experimental.TracingContext", v1=[])
class TracingContext(metaclass=abc.ABCMeta):
  """Contains information scoped to the tracing of multiple objects.

  `TracingContext` is a container class for flags and variables that have
  any kind of influence on the tracing behaviour of the class implementing
  the __tf_tracing_type__. This context will be shared across all
  __tf_tracing_type__ calls while constructing the TraceType for a particular
  set of objects.
  """


class PlaceholderContext():
  """Contains context information for generating placeholders within a scope."""


class CastContext():
  """Contains context info and rules for casting values to a TypeSpec."""


@runtime_checkable
class SupportsTracingProtocol(Protocol):
  """A protocol allowing custom classes to control tf.function retracing."""

  @doc_controls.doc_private
  @abc.abstractmethod
  def __tf_tracing_type__(self, context: TracingContext) -> TraceType:
    """Returns the tracing type of this object.

    The tracing type is used to build the signature of a tf.function
    when traced, and to match arguments with existing signatures.
    When a Function object is called, tf.function looks at the tracing type
    of the call arguments. If an existing signature of matching type exists,
    it will be used. Otherwise, a new function is traced, and its signature
    will use the tracing type of the call arguments.

    Args:
      context: a context object created for each function call for tracking
        information about the call arguments as a whole
    Returns:
      The tracing type of this object.
    """

# TODO(b/219556836): Direct tf_export decorator adds non-method members to the
# Protocol which breaks @runtime_checkable since it does not support them.
tf_export(
    "types.experimental.SupportsTracingProtocol",
    v1=[]).export_constant(__name__, "SupportsTracingProtocol")
