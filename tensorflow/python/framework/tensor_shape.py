# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Helper classes for tensor shape inference."""
import functools
import operator
from typing import Optional, Sequence, Type

from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.function import trace_type
from tensorflow.python import tf2
from tensorflow.python.eager import monitoring
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import trace
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls

_TENSORSHAPE_V2_OVERRIDE = None

_api_usage_gauge = monitoring.BoolGauge(
    "/tensorflow/api/v2_tensorshape",
    "Whether tensor_shape.enable_v2_tensorshape() is called.")


@tf_export(v1=["enable_v2_tensorshape"])
def enable_v2_tensorshape():
  """In TensorFlow 2.0, iterating over a TensorShape instance returns values.

  This enables the new behavior.

  Concretely, `tensor_shape[i]` returned a Dimension instance in V1, but
  it V2 it returns either an integer, or None.

  Examples:

  ```
  #######################
  # If you had this in V1:
  value = tensor_shape[i].value

  # Do this in V2 instead:
  value = tensor_shape[i]

  #######################
  # If you had this in V1:
  for dim in tensor_shape:
    value = dim.value
    print(value)

  # Do this in V2 instead:
  for value in tensor_shape:
    print(value)

  #######################
  # If you had this in V1:
  dim = tensor_shape[i]
  dim.assert_is_compatible_with(other_shape)  # or using any other shape method

  # Do this in V2 instead:
  if tensor_shape.rank is None:
    dim = Dimension(None)
  else:
    dim = tensor_shape.dims[i]
  dim.assert_is_compatible_with(other_shape)  # or using any other shape method

  # The V2 suggestion above is more explicit, which will save you from
  # the following trap (present in V1):
  # you might do in-place modifications to `dim` and expect them to be reflected
  # in `tensor_shape[i]`, but they would not be.
  ```
  """
  global _TENSORSHAPE_V2_OVERRIDE  # pylint: disable=invalid-name
  _TENSORSHAPE_V2_OVERRIDE = True
  logging.vlog(1, "Enabling v2 tensorshape")
  _api_usage_gauge.get_cell().set(True)


@tf_export(v1=["disable_v2_tensorshape"])
def disable_v2_tensorshape():
  """Disables the V2 TensorShape behavior and reverts to V1 behavior.

  See docstring for `enable_v2_tensorshape` for details about the new behavior.
  """
  global _TENSORSHAPE_V2_OVERRIDE  # pylint: disable=invalid-name
  _TENSORSHAPE_V2_OVERRIDE = False
  logging.vlog(1, "Disabling v2 tensorshape")
  _api_usage_gauge.get_cell().set(False)


@tf_export(
    "compat.dimension_value", v1=["dimension_value", "compat.dimension_value"])
def dimension_value(dimension):
  """Compatibility utility required to allow for both V1 and V2 behavior in TF.

  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to
  coexist with the new behavior. This utility is a bridge between the two.

  When accessing the value of a TensorShape dimension,
  use this utility, like this:

  ```
  # If you had this in your V1 code:
  value = tensor_shape[i].value

  # Use `dimension_value` as direct replacement compatible with both V1 & V2:
  value = dimension_value(tensor_shape[i])

  # This would be the V2 equivalent:
  value = tensor_shape[i]  # Warning: this will return the dim value in V2!
  ```

  Args:
    dimension: Either a `Dimension` instance, an integer, or None.

  Returns:
    A plain value, i.e. an integer or None.
  """
  if isinstance(dimension, Dimension):
    return dimension.value
  return dimension


@tf_export(
    "compat.dimension_at_index",
    v1=["dimension_at_index", "compat.dimension_at_index"])
def dimension_at_index(shape, index):
  """Compatibility utility required to allow for both V1 and V2 behavior in TF.

  Until the release of TF 2.0, we need the legacy behavior of `TensorShape` to
  coexist with the new behavior. This utility is a bridge between the two.

  If you want to retrieve the Dimension instance corresponding to a certain
  index in a TensorShape instance, use this utility, like this:

  ```
  # If you had this in your V1 code:
  dim = tensor_shape[i]

  # Use `dimension_at_index` as direct replacement compatible with both V1 & V2:
  dim = dimension_at_index(tensor_shape, i)

  # Another possibility would be this, but WARNING: it only works if the
  # tensor_shape instance has a defined rank.
  dim = tensor_shape.dims[i]  # `dims` may be None if the rank is undefined!

  # In native V2 code, we recommend instead being more explicit:
  if tensor_shape.rank is None:
    dim = Dimension(None)
  else:
    dim = tensor_shape.dims[i]

  # Being more explicit will save you from the following trap (present in V1):
  # you might do in-place modifications to `dim` and expect them to be reflected
  # in `tensor_shape[i]`, but they would not be (as the Dimension object was
  # instantiated on the fly.
  ```

  Args:
    shape: A TensorShape instance.
    index: An integer index.

  Returns:
    A dimension object.
  """
  assert isinstance(shape, TensorShape)
  if shape.rank is None:
    return Dimension(None)
  else:
    return shape.dims[index]


@tf_export(v1=["Dimension"])
class Dimension(object):
  """Represents the value of one dimension in a TensorShape.

  @compatibility(TF2)
  In TF2, members of a `TensorShape` object are integers. The `Dimension` class
  is not part of TF2's data model.

  Please refer to the [TensorShape section of the migration guide]
  (https://www.tensorflow.org/guide/migrate/index#tensorshape) on common code
  patterns adapting Dimension objects to a TF2 syntax.
  @end_compatibility
  """

  __slots__ = ["_value"]

  def __init__(self, value):
    """Creates a new Dimension with the given value."""
    if isinstance(value, int):  # Most common case.
      if value < 0:
        raise ValueError("Dimension %d must be >= 0" % value)
      self._value = value
    elif value is None:
      self._value = None
    elif isinstance(value, Dimension):
      self._value = value._value
    else:
      try:
        # int(...) compensates for the int/long dichotomy on Python 2.X.
        # TODO(b/143206389): Remove once we fully migrate to 3.X.
        self._value = int(value.__index__())
      except AttributeError:
        raise TypeError(
            "Dimension value must be integer or None or have "
            "an __index__ method, got value '{0!r}' with type '{1!r}'".format(
                value, type(value))) from None
      if self._value < 0:
        raise ValueError("Dimension %d must be >= 0" % self._value)

  def __repr__(self):
    return "Dimension(%s)" % repr(self._value)

  def __str__(self):
    value = self._value
    return "?" if value is None else str(value)

  def __eq__(self, other):
    """Returns true if `other` has the same known value as this Dimension."""
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return None
    return self._value == other.value

  def __ne__(self, other):
    """Returns true if `other` has a different known value from `self`."""
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return None
    return self._value != other.value

  def __bool__(self):
    """Equivalent to `bool(self.value)`."""
    return bool(self._value)

  def __int__(self):
    return self._value

  # This is needed for Windows.
  # See https://github.com/tensorflow/tensorflow/pull/9780
  def __long__(self):
    return self._value

  def __index__(self):
    # Allow use in Python 3 range
    return self._value

  @property
  def value(self):
    """The value of this dimension, or None if it is unknown."""
    return self._value

  # TODO(b/225058047): Reconsider semantics.
  def is_compatible_with(self, other):
    """Returns true if `other` is compatible with this Dimension.

    Two known Dimensions are compatible if they have the same value.
    An unknown Dimension is compatible with all other Dimensions.

    Args:
      other: Another Dimension.

    Returns:
      True if this Dimension and `other` are compatible.
    """
    other = as_dimension(other)
    return (self._value is None or other.value is None or
            self._value == other.value)

  def assert_is_compatible_with(self, other):
    """Raises an exception if `other` is not compatible with this Dimension.

    Args:
      other: Another Dimension.

    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    """
    if not self.is_compatible_with(other):
      raise ValueError("Dimensions %s and %s are not compatible" %
                       (self, other))

  def merge_with(self, other):
    """Returns a Dimension that combines the information in `self` and `other`.

    Dimensions are combined as follows:

    ```python
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(n))     ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(None))  ==
    tf.compat.v1.Dimension(n)
    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(n))     ==
    tf.compat.v1.Dimension(n)
    # equivalent to tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None).merge_with(tf.compat.v1.Dimension(None))

    # raises ValueError for n != m
    tf.compat.v1.Dimension(n)   .merge_with(tf.compat.v1.Dimension(m))
    ```

    Args:
      other: Another Dimension.

    Returns:
      A Dimension containing the combined information of `self` and
      `other`.

    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    """
    other = as_dimension(other)
    self.assert_is_compatible_with(other)
    if self._value is None:
      return Dimension(other.value)
    else:
      return Dimension(self._value)

  def __add__(self, other):
    """Returns the sum of `self` and `other`.

    Dimensions are summed as follows:

    ```python
    tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m + n)
    tf.compat.v1.Dimension(m)    + tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) + tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) + tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the sum of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value + other.value)

  def __radd__(self, other):
    """Returns the sum of `other` and `self`.

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the sum of `self` and `other`.
    """
    return self + other

  def __sub__(self, other):
    """Returns the subtraction of `other` from `self`.

    Dimensions are subtracted as follows:

    ```python
    tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m - n)
    tf.compat.v1.Dimension(m)    - tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) - tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) - tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the subtraction of `other` from `self`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value - other.value)

  def __rsub__(self, other):
    """Returns the subtraction of `self` from `other`.

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the subtraction of `self` from `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(other.value - self._value)

  def __mul__(self, other):
    """Returns the product of `self` and `other`.

    Dimensions are summed as follows:

    ```python
    tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m * n)
    tf.compat.v1.Dimension(m)    * tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) * tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) * tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the product of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented

    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value * other.value)

  def __rmul__(self, other):
    """Returns the product of `self` and `other`.

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is the product of `self` and `other`.
    """
    return self * other

  def __floordiv__(self, other):
    """Returns the quotient of `self` and `other` rounded down.

    Dimensions are divided as follows:

    ```python
    tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m // n)
    tf.compat.v1.Dimension(m)    // tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) // tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) // tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    try:
      other = as_dimension(other)
    except (TypeError, ValueError):
      return NotImplemented
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value // other.value)

  def __rfloordiv__(self, other):
    """Returns the quotient of `other` and `self` rounded down.

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(other.value // self._value)

  def __div__(self, other):
    """DEPRECATED: Use `__floordiv__` via `x // y` instead.

    This function exists only for backwards compatibility purposes; new code
    should use `__floordiv__` via the syntax `x // y`.  Using `x // y`
    communicates clearly that the result rounds down, and is forward compatible
    to Python 3.

    Args:
      other: Another `Dimension`.

    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    return self // other

  def __rdiv__(self, other):
    """Use `__floordiv__` via `x // y` instead.

    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,
    this function will explicitly call for usage of `//` instead.

    Args:
      other: Another `Dimension`.

    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: '{}' and 'Dimension', "
                    "please use // instead".format(type(other).__name__))

  def __truediv__(self, other):
    """Use `__floordiv__` via `x // y` instead.

    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'Dimension' and 'int'`,
    this function will explicitly call for usage of `//` instead.

    Args:
      other: Another `Dimension`.

    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: 'Dimension' and '{}', "
                    "please use // instead".format(type(other).__name__))

  def __rtruediv__(self, other):
    """Use `__floordiv__` via `x // y` instead.

    This function exists only to have a better error message. Instead of:
    `TypeError: unsupported operand type(s) for /: 'int' and 'Dimension'`,
    this function will explicitly call for usage of `//` instead.

    Args:
      other: Another `Dimension`.

    Raises:
      TypeError.
    """
    raise TypeError("unsupported operand type(s) for /: '{}' and 'Dimension', "
                    "please use // instead".format(type(other).__name__))

  def __mod__(self, other):
    """Returns `self` modulo `other`.

    Dimension modulo are computed as follows:

    ```python
    tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(n)     ==
    tf.compat.v1.Dimension(m % n)
    tf.compat.v1.Dimension(m)    % tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) % tf.compat.v1.Dimension(n)     # equiv. to
    tf.compat.v1.Dimension(None)
    tf.compat.v1.Dimension(None) % tf.compat.v1.Dimension(None)  # equiv. to
    tf.compat.v1.Dimension(None)
    ```

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is `self` modulo `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value % other.value)

  def __rmod__(self, other):
    """Returns `other` modulo `self`.

    Args:
      other: Another Dimension, or a value accepted by `as_dimension`.

    Returns:
      A Dimension whose value is `other` modulo `self`.
    """
    other = as_dimension(other)
    return other % self

  def __lt__(self, other):
    """Returns True if `self` is known to be less than `other`.

    Dimensions are compared as follows:

    ```python
    (tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(n))    == (m < n)
    (tf.compat.v1.Dimension(m)    < tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) < tf.compat.v1.Dimension(None)) == None
    ```

    Args:
      other: Another Dimension.

    Returns:
      The value of `self.value < other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value < other.value

  def __le__(self, other):
    """Returns True if `self` is known to be less than or equal to `other`.

    Dimensions are compared as follows:

    ```python
    (tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(n))    == (m <= n)
    (tf.compat.v1.Dimension(m)    <= tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) <= tf.compat.v1.Dimension(None)) == None
    ```

    Args:
      other: Another Dimension.

    Returns:
      The value of `self.value <= other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value <= other.value

  def __gt__(self, other):
    """Returns True if `self` is known to be greater than `other`.

    Dimensions are compared as follows:

    ```python
    (tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(n))    == (m > n)
    (tf.compat.v1.Dimension(m)    > tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) > tf.compat.v1.Dimension(None)) == None
    ```

    Args:
      other: Another Dimension.

    Returns:
      The value of `self.value > other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value > other.value

  def __ge__(self, other):
    """Returns True if `self` is known to be greater than or equal to `other`.

    Dimensions are compared as follows:

    ```python
    (tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(n))    == (m >= n)
    (tf.compat.v1.Dimension(m)    >= tf.compat.v1.Dimension(None)) == None
    (tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(n))    == None
    (tf.compat.v1.Dimension(None) >= tf.compat.v1.Dimension(None)) == None
    ```

    Args:
      other: Another Dimension.

    Returns:
      The value of `self.value >= other.value` if both are known, otherwise
      None.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    else:
      return self._value >= other.value

  def __reduce__(self):
    return Dimension, (self._value,)


def as_dimension(value):
  """Converts the given value to a Dimension.

  A Dimension input will be returned unmodified.
  An input of `None` will be converted to an unknown Dimension.
  An integer input will be converted to a Dimension with that value.

  Args:
    value: The value to be converted.

  Returns:
    A Dimension corresponding to the given value.
  """
  if isinstance(value, Dimension):
    return value
  else:
    return Dimension(value)


@tf_export("TensorShape")
class TensorShape(trace.TraceType, trace_type.Serializable):
  """Represents the shape of a `Tensor`.

  >>> t = tf.constant([[1,2,3],[4,5,6]])
  >>> t.shape
  TensorShape([2, 3])

  `TensorShape` is the *static* shape representation of a Tensor.
  During eager execution a Tensor always has a fully specified shape but
  when tracing a `tf.function` it may be one of the following:

  * *Fully-known shape:* has a known number of dimensions and a known size
    for each dimension. e.g. `TensorShape([16, 256])`
  * *Partially-known shape:* has a known number of dimensions, and an unknown
    size for one or more dimension. e.g. `TensorShape([None, 256])`
  * *Unknown shape:* has an unknown number of dimensions, and an unknown
    size in all dimensions. e.g. `TensorShape(None)`

  During function tracing `t.shape` will return a `TensorShape` object
  representing the shape of Tensor as it is known during tracing.
  This static representation will be partially defined in cases where the
  exact shape depends on the values within the tensors. To get the
  *dynamic* representation, please use `tf.shape(t)`
  which will return Tensor representing the fully defined shape of `t`.
  This way, you can express logic that manipulates the shapes of tensors by
  building other tensors that depend on the dynamic shape of `t`.

  Note: `tf.RaggedTensor.shape` also returns a `tf.TensorShape`,
  the lengths of any ragged dimensions are unknown (`None`).

  For example, this function prints the `TensorShape' (`t.shape`), when you
  trace the function, and returns a tensor `tf.shape(t)` for given input `t`:

  >>> @tf.function
  ... def get_dynamic_shape(t):
  ...   print("tracing...")
  ...   print(f"static shape is {t.shape}")
  ...   return tf.shape(t)

  Just calling the function traces it with a fully-specified static shape:

  >>> result = get_dynamic_shape(tf.constant([[1, 1, 1], [0, 0, 0]]))
  tracing...
  static shape is (2, 3)
  >>> result.numpy()
  array([2, 3], dtype=int32)

  But `tf.function` can also trace the function with a partially specified
  (or even unspecified) shape:

  >>> cf1 = get_dynamic_shape.get_concrete_function(tf.TensorSpec(
  ...                                               shape=[None, 2]))
  tracing...
  static shape is (None, 2)
  >>> cf1(tf.constant([[1., 0],[1, 0],[1, 0]])).numpy()
  array([3, 2], dtype=int32)

  >>> cf2 = get_dynamic_shape.get_concrete_function(tf.TensorSpec(shape=None))
  tracing...
  static shape is <unknown>
  >>> cf2(tf.constant([[[[[1., 0]]]]])).numpy()
  array([1, 1, 1, 1, 2], dtype=int32)

  If a tensor is produced by an operation of type `"Foo"`, its shape
  may be inferred if there is a registered shape function for
  `"Foo"`. See [Shape
  functions](https://www.tensorflow.org/guide/create_op#shape_functions_in_c)
  for details of shape functions and how to register them. Alternatively,
  you may set the shape explicitly using `tf.Tensor.ensure_shape`.
  """
  __slots__ = ["_dims"]

  def __init__(self, dims):
    """Creates a new TensorShape with the given dimensions.

    Args:
      dims: A list of Dimensions, or None if the shape is unspecified.

    Raises:
      TypeError: If dims cannot be converted to a list of dimensions.
    """
    if isinstance(dims, (tuple, list)):  # Most common case.
      self._dims = tuple(as_dimension(d).value for d in dims)
    elif dims is None:
      self._dims = None
    elif isinstance(dims, tensor_shape_pb2.TensorShapeProto):
      if dims.unknown_rank:
        self._dims = None
      else:
        self._dims = tuple(
            # Protos store variable-size dimensions as -1
            dim.size if dim.size != -1 else None
            for dim in dims.dim
            )
    elif isinstance(dims, TensorShape):
      self._dims = dims._dims
    else:
      try:
        dims_iter = iter(dims)
      except TypeError:
        # Treat as a singleton dimension
        self._dims = (as_dimension(dims).value,)
      else:
        self._dims = []
        for d in dims_iter:
          try:
            self._dims.append(as_dimension(d).value)
          except TypeError as e:
            raise TypeError(
                "Failed to convert '{0!r}' to a shape: '{1!r}'"
                "could not be converted to a dimension. A shape should "
                "either be single dimension (e.g. 10), or an iterable of "
                "dimensions (e.g. [1, 10, None]).".format(dims, d)) from e
        self._dims = tuple(self._dims)

  @property
  def _v2_behavior(self):
    if _TENSORSHAPE_V2_OVERRIDE is None:
      return tf2.enabled()
    return _TENSORSHAPE_V2_OVERRIDE

  def __repr__(self):
    if self._v2_behavior:
      if self._dims is not None:
        return f"TensorShape({list(self._dims)})"
      else:
        return "TensorShape(None)"
    else:
      return f"TensorShape({self.dims})"

  def __str__(self):
    if self.rank is None:
      return "<unknown>"
    elif self.rank == 1:
      if self._v2_behavior:
        return "(%s,)" % self._dims[0]
      else:
        return "(%s,)" % self.dims[0]
    else:
      if self._v2_behavior:
        return "(%s)" % ", ".join(str(d) for d in self._dims)
      else:
        return "(%s)" % ", ".join(str(d) for d in self.dims)

  @property
  def rank(self):
    """Returns the rank of this shape, or None if it is unspecified."""
    if self._dims is not None:
      return len(self._dims)
    return None

  @property
  def dims(self):
    """Deprecated.  Returns list of dimensions for this shape.

    Suggest `TensorShape.as_list` instead.

    Returns:
      A list containing `tf.compat.v1.Dimension`s, or None if the shape is
      unspecified.
    """
    if self._dims is None:
      return None
    return [as_dimension(d) for d in self._dims]

  @property
  def ndims(self):
    """Deprecated accessor for `rank`."""
    return self.rank

  def __len__(self):
    """Returns the rank of this shape, or raises ValueError if unspecified."""
    if self._dims is None:
      raise ValueError("Cannot take the length of shape with unknown rank.")
    return len(self._dims)

  def __bool__(self):
    """Returns True if this shape contains non-zero information."""
    return self._dims is not None

  # Python 3 wants __bool__, Python 2.7 wants __nonzero__
  __nonzero__ = __bool__

  def __iter__(self):
    """Returns `self.dims` if the rank is known, otherwise raises ValueError."""
    if self._dims is None:
      raise ValueError("Cannot iterate over a shape with unknown rank.")
    else:
      if self._v2_behavior:
        return iter(d for d in self._dims)
      else:
        return iter(d for d in self.dims)

  def __getitem__(self, key):
    """Returns the value of a dimension or a shape, depending on the key.

    Args:
      key: If `key` is an integer, returns the dimension at that index;
        otherwise if `key` is a slice, returns a TensorShape whose dimensions
        are those selected by the slice from `self`.

    Returns:
      An integer if `key` is an integer, or a `TensorShape` if `key` is a
      slice.

    Raises:
      ValueError: If `key` is a slice and `self` is completely unknown and
        the step is set.
    """
    if self._dims is not None:
      if isinstance(key, slice):
        return TensorShape(self._dims[key])
      else:
        if self._v2_behavior:
          return self._dims[key]
        else:
          return self.dims[key]
    else:
      if isinstance(key, slice):
        start = key.start if key.start is not None else 0
        stop = key.stop

        if key.step is not None:
          # TODO(mrry): Handle these maybe.
          raise ValueError("Steps are not yet handled")
        if stop is None:
          # NOTE(mrry): This implies that TensorShape(None) is compatible with
          # TensorShape(None)[1:], which is obviously not true. It would be
          # possible to track the number of dimensions symbolically,
          # and perhaps we should do that.
          return unknown_shape()
        elif start < 0 or stop < 0:
          # TODO(mrry): Handle this better, as it will be useful for handling
          # suffixes of otherwise unknown shapes.
          return unknown_shape()
        else:
          return unknown_shape(rank=stop - start)
      else:
        if self._v2_behavior:
          return None
        else:
          return Dimension(None)

  def num_elements(self):
    """Returns the total number of elements, or none for incomplete shapes."""
    if self.is_fully_defined():
      return functools.reduce(operator.mul, self.as_list(), 1)
    else:
      return None

  def merge_with(self, other):
    """Returns a `TensorShape` combining the information in `self` and `other`.

    The dimensions in `self` and `other` are merged element-wise,
    according to the rules below:

    ```python
    Dimension(n).merge_with(Dimension(None)) == Dimension(n)
    Dimension(None).merge_with(Dimension(n)) == Dimension(n)
    Dimension(None).merge_with(Dimension(None)) == Dimension(None)
    # raises ValueError for n != m
    Dimension(n).merge_with(Dimension(m))
    ```
    >> ts = tf.TensorShape([1,2])
    >> ot1 = tf.TensorShape([1,2])
    >> ts.merge_with(ot).as_list()
    [1,2]

    >> ot2 = tf.TensorShape([1,None])
    >> ts.merge_with(ot2).as_list()
    [1,2]

    >> ot3 = tf.TensorShape([None, None])
    >> ot3.merge_with(ot2).as_list()
    [1, None]

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` containing the combined information of `self` and
      `other`.

    Raises:
      ValueError: If `self` and `other` are not compatible.
    """
    other = as_shape(other)
    if self.dims is None:
      return other
    if other.dims is None:
      return self
    else:
      try:
        self.assert_same_rank(other)
        new_dims = [
            dim.merge_with(other_dim)
            for dim, other_dim in zip(self.dims, other.dims)
        ]
        return TensorShape(new_dims)
      except ValueError:
        raise ValueError("Shapes %s and %s are not compatible" % (self, other))

  def __add__(self, other):
    return self.concatenate(other)

  def __radd__(self, other):
    if not isinstance(other, TensorShape):
      other = TensorShape(other)
    return other.concatenate(self)

  def concatenate(self, other):
    """Returns the concatenation of the dimension in `self` and `other`.

    *N.B.* If either `self` or `other` is completely unknown,
    concatenation will discard information about the other shape. In
    future, we might support concatenation that preserves this
    information for use with slicing.

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` whose dimensions are the concatenation of the
      dimensions in `self` and `other`.
    """
    # TODO(mrry): Handle the case where we concatenate a known shape with a
    # completely unknown shape, so that we can use the partial information.
    other = as_shape(other)
    if self.dims is None or other.dims is None:
      return unknown_shape()
    else:
      return TensorShape(self.dims + other.dims)

  def assert_same_rank(self, other):
    """Raises an exception if `self` and `other` do not have compatible ranks.

    Args:
      other: Another `TensorShape`.

    Raises:
      ValueError: If `self` and `other` do not represent shapes with the
        same rank.
    """
    other = as_shape(other)
    if self.rank is not None and other.rank is not None:
      if self.rank != other.rank:
        raise ValueError("Shapes %s and %s must have the same rank" %
                         (self, other))

  def assert_has_rank(self, rank):
    """Raises an exception if `self` is not compatible with the given `rank`.

    Args:
      rank: An integer.

    Raises:
      ValueError: If `self` does not represent a shape with the given `rank`.
    """
    if self.rank not in (None, rank):
      raise ValueError("Shape %s must have rank %d" % (self, rank))

  def with_rank(self, rank):
    """Returns a shape based on `self` with the given rank.

    This method promotes a completely unknown shape to one with a
    known rank.

    Args:
      rank: An integer.

    Returns:
      A shape that is at least as specific as `self` with the given rank.

    Raises:
      ValueError: If `self` does not represent a shape with the given `rank`.
    """
    try:
      return self.merge_with(unknown_shape(rank=rank))
    except ValueError:
      raise ValueError("Shape %s must have rank %d" % (self, rank))

  def with_rank_at_least(self, rank):
    """Returns a shape based on `self` with at least the given rank.

    Args:
      rank: An integer.

    Returns:
      A shape that is at least as specific as `self` with at least the given
      rank.

    Raises:
      ValueError: If `self` does not represent a shape with at least the given
        `rank`.
    """
    if self.rank is not None and self.rank < rank:
      raise ValueError("Shape %s must have rank at least %d" % (self, rank))
    else:
      return self

  def with_rank_at_most(self, rank):
    """Returns a shape based on `self` with at most the given rank.

    Args:
      rank: An integer.

    Returns:
      A shape that is at least as specific as `self` with at most the given
      rank.

    Raises:
      ValueError: If `self` does not represent a shape with at most the given
        `rank`.
    """
    if self.rank is not None and self.rank > rank:
      raise ValueError("Shape %s must have rank at most %d" % (self, rank))
    else:
      return self

  def is_subtype_of(self, other: trace.TraceType) -> bool:
    """Returns True iff `self` is subtype of `other`.

    Shape A is a subtype of shape B if shape B can successfully represent it:

    * A `TensorShape` of any rank is a subtype of `TensorShape(None)`.

    *  TensorShapes of equal ranks are covariant, i.e.
      `TensorShape([A1, A2, ..])` is a subtype of
      `TensorShape([B1, B2, ..])` iff An is a subtype of Bn.

      An is subtype of Bn iff An == Bn or Bn is None.

    * TensorShapes of different defined ranks have no subtyping relation.

    The subtyping relation is reflexive and transitive, but not symmetric.

    Some examples:
    * `TensorShape([32, 784])` is a subtype of `TensorShape(None)`, and
      `TensorShape([4, 4])` is also a subtype of `TensorShape(None)` but
      `TensorShape([32, 784])` and `TensorShape([4, 4])` are not subtypes of
      each other.

    * All two-dimensional shapes are subtypes of `TensorShape([None, None])`,
      such as `TensorShape([32, 784])`. There is no subtype relationship with,
      for example, `TensorShape([None])` or `TensorShape([None, None, None])`.

    * `TensorShape([32, None])` is also a subtype of `TensorShape([None, None])`
      and `TensorShape(None)`. It is not a subtype of, for example,
      `TensorShape([32])`, `TensorShape([32, None, 1])`,
      `TensorShape([64, None])` or `TensorShape([None, 32])`.

    * `TensorShape([32, 784])` is a subtype of itself, and also
      `TensorShape([32, None])`, `TensorShape([None, 784])`,
      `TensorShape([None, None])` and `TensorShape(None)`.
      It has no subtype relation with, for example, `TensorShape([32, 1, 784])`
      or `TensorShape([None])`.

    Args:
      other: Another `TensorShape`.

    Returns:
      True iff `self` is subtype of `other`.

    """
    if not isinstance(other, TensorShape):
      return False

    # All Tensors are subtypes of a Tensor with no shape.
    if other.rank is None:
      return True

    # Tensor with a defined shape can only be subtype of another with a defined
    # shape if they have the same number of dimensions.
    if self.rank != other.rank:
      return False

    # A Tensor is a subtype if each corresponding dimension is a subtype.
    return all(o is None or s == o for s, o in zip(self._dims, other._dims))  # pylint: disable=protected-access

  def most_specific_common_supertype(
      self, others: Sequence[trace.TraceType]) -> Optional["TensorShape"]:
    """Returns the most specific supertype `TensorShape` of self and others.

    * `TensorShape([None, 1])` is the most specific `TensorShape` supertyping
      both `TensorShape([2, 1])` and `TensorShape([5, 1])`. Note that
      `TensorShape(None)` is also a supertype but it is not "most specific".

    * `TensorShape([1, 2, 3])` is the most specific `TensorShape` supertyping
      both `TensorShape([1, 2, 3])` and `TensorShape([1, 2, 3]`). There are
      other less specific TensorShapes that supertype above mentioned
      TensorShapes, e.g. `TensorShape([1, 2, None])`, `TensorShape(None)`.

     * `TensorShape([None, None])` is the most specific `TensorShape`
       supertyping both `TensorShape([2, None])` and `TensorShape([None, 3])`.
       As always, `TensorShape(None)` is also a supertype but not the most
       specific one.

     * `TensorShape(None`) is the only `TensorShape` supertyping both
       `TensorShape([1, 2, 3])` and `TensorShape([1, 2])`. In general, any two
       shapes that have different ranks will only have `TensorShape(None)`
       as a common supertype.

     * `TensorShape(None)` is the only `TensorShape` supertyping both
       `TensorShape([1, 2, 3])` and `TensorShape(None)`. In general, the common
       supertype of any shape with `TensorShape(None)` is `TensorShape(None)`.

    Args:
      others: Sequence of `TensorShape`.

    Returns:
      A `TensorShape` which is the most specific supertype shape of `self`
      and `others`. None if it does not exist.
    """
    if any(not isinstance(other, TensorShape) for other in others):
      return None

    # A Rankless TensorShape is already a global supertype so we return another
    # instance of it.
    if self.rank is None:
      return unknown_shape()

    # A Rankless TensorShape is the most specific supertype for shapes whose
    # ranks do not match.
    if any(other.dims is None or self.rank != other.rank for other in others):
      return unknown_shape()

    # Retain the integer dimension if it is the same across all others, else
    # use an undefined dimension.
    dims = [
        dim if all(dim == other._dims[i]
                   for other in others) else None
        for i, dim in enumerate(self._dims)
    ]
    return TensorShape(dims)

  @doc_controls.do_not_doc_inheritable
  def placeholder_value(self, placeholder_context=None):
    raise NotImplementedError("A graph placeholder is not currently supported"
                              "for an object of type: TensorShape.")

  @classmethod
  def experimental_type_proto(cls) -> Type[tensor_shape_pb2.TensorShapeProto]:
    """Returns the type of proto associated with TensorShape serialization."""
    return tensor_shape_pb2.TensorShapeProto

  @classmethod
  def experimental_from_proto(
      cls, proto: tensor_shape_pb2.TensorShapeProto) -> "TensorShape":
    """Returns a TensorShape instance based on the serialized proto."""
    return TensorShape(proto)

  def experimental_as_proto(self) -> tensor_shape_pb2.TensorShapeProto:
    """Returns a proto representation of the TensorShape instance."""
    return self.as_proto()

  # TODO(b/216206374): Consider deprecation at TraceType release.
  def is_compatible_with(self, other):
    """Returns True iff `self` is compatible with `other`.

    Two possibly-partially-defined shapes are compatible if there
    exists a fully-defined shape that both shapes can represent. Thus,
    compatibility allows the shape inference code to reason about
    partially-defined shapes. For example:

    * TensorShape(None) is compatible with all shapes.

    * TensorShape([None, None]) is compatible with all two-dimensional
      shapes, such as TensorShape([32, 784]), and also TensorShape(None). It is
      not compatible with, for example, TensorShape([None]) or
      TensorShape([None, None, None]).

    * TensorShape([32, None]) is compatible with all two-dimensional shapes
      with size 32 in the 0th dimension, and also TensorShape([None, None])
      and TensorShape(None). It is not compatible with, for example,
      TensorShape([32]), TensorShape([32, None, 1]) or TensorShape([64, None]).

    * TensorShape([32, 784]) is compatible with itself, and also
      TensorShape([32, None]), TensorShape([None, 784]), TensorShape([None,
      None]) and TensorShape(None). It is not compatible with, for example,
      TensorShape([32, 1, 784]) or TensorShape([None]).

    The compatibility relation is reflexive and symmetric, but not
    transitive. For example, TensorShape([32, 784]) is compatible with
    TensorShape(None), and TensorShape(None) is compatible with
    TensorShape([4, 4]), but TensorShape([32, 784]) is not compatible with
    TensorShape([4, 4]).

    Args:
      other: Another TensorShape.

    Returns:
      True iff `self` is compatible with `other`.

    """
    other = as_shape(other)
    if self.dims is not None and other.dims is not None:
      if self.rank != other.rank:
        return False
      for x_dim, y_dim in zip(self.dims, other.dims):
        if not x_dim.is_compatible_with(y_dim):
          return False
    return True

  def assert_is_compatible_with(self, other):
    """Raises exception if `self` and `other` do not represent the same shape.

    This method can be used to assert that there exists a shape that both
    `self` and `other` represent.

    Args:
      other: Another TensorShape.

    Raises:
      ValueError: If `self` and `other` do not represent the same shape.
    """
    if not self.is_compatible_with(other):
      raise ValueError("Shapes %s and %s are incompatible" % (self, other))

  def most_specific_compatible_shape(self, other):
    """Returns the most specific TensorShape compatible with `self` and `other`.

    * TensorShape([None, 1]) is the most specific TensorShape compatible with
      both TensorShape([2, 1]) and TensorShape([5, 1]). Note that
      TensorShape(None) is also compatible with above mentioned TensorShapes.

    * TensorShape([1, 2, 3]) is the most specific TensorShape compatible with
      both TensorShape([1, 2, 3]) and TensorShape([1, 2, 3]). There are more
      less specific TensorShapes compatible with above mentioned TensorShapes,
      e.g. TensorShape([1, 2, None]), TensorShape(None).

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` which is the most specific compatible shape of `self`
      and `other`.
    """

    other = as_shape(other)
    if self.dims is None or other.dims is None or self.rank != other.rank:
      return unknown_shape()

    dims = [
        d1 if d1 is not None and d2 is not None and d1 == d2 else None
        for d1, d2 in zip(self.dims, other.dims)
    ]
    return TensorShape(dims)

  def is_fully_defined(self):
    """Returns True iff `self` is fully defined in every dimension."""
    return (self._dims is not None and
            all(dim is not None for dim in self._dims))

  def assert_is_fully_defined(self):
    """Raises an exception if `self` is not fully defined in every dimension.

    Raises:
      ValueError: If `self` does not have a known value for every dimension.
    """
    if not self.is_fully_defined():
      raise ValueError("Shape %s is not fully defined" % self)

  def as_list(self):
    """Returns a list of integers or `None` for each dimension.

    Returns:
      A list of integers or `None` for each dimension.

    Raises:
      ValueError: If `self` is an unknown shape with an unknown rank.
    """
    if self._dims is None:
      raise ValueError("as_list() is not defined on an unknown TensorShape.")
    return list(self._dims)

  def as_proto(self):
    """Returns this shape as a `TensorShapeProto`."""
    if self._dims is None:
      return tensor_shape_pb2.TensorShapeProto(unknown_rank=True)
    else:
      return tensor_shape_pb2.TensorShapeProto(dim=[
          tensor_shape_pb2.TensorShapeProto.Dim(
              size=-1 if d is None else d) for d in self._dims
      ])

  def __eq__(self, other):
    """Returns True if `self` is equivalent to `other`.

    It first tries to convert `other` to `TensorShape`. `TypeError` is thrown
    when the conversion fails. Otherwise, it compares each element in the
    TensorShape dimensions.

    * Two *Fully known* shapes, return True iff each element is equal.
    >>> t_a = tf.TensorShape([1,2])
    >>> a = [1, 2]
    >>> t_b = tf.TensorShape([1,2])
    >>> t_c = tf.TensorShape([1,2,3])
    >>> t_a.__eq__(a)
    True
    >>> t_a.__eq__(t_b)
    True
    >>> t_a.__eq__(t_c)
    False

    * Two *Partially-known* shapes, return True iff each element is equal.
    >>> p_a = tf.TensorShape([1,None])
    >>> p_b = tf.TensorShape([1,None])
    >>> p_c = tf.TensorShape([2,None])
    >>> p_a.__eq__(p_b)
    True
    >>> t_a.__eq__(p_a)
    False
    >>> p_a.__eq__(p_c)
    False

    * Two *Unknown shape*, return True.
    >>> unk_a = tf.TensorShape(None)
    >>> unk_b = tf.TensorShape(None)
    >>> unk_a.__eq__(unk_b)
    True
    >>> unk_a.__eq__(t_a)
    False

    Args:
      other: A `TensorShape` or type that can be converted to `TensorShape`.

    Returns:
      True if the dimensions are all equal.

    Raises:
      TypeError if `other` can not be converted to `TensorShape`.
    """

    try:
      other = as_shape(other)
    except TypeError:
      return NotImplemented

    return self._dims == other._dims

  def __hash__(self):
    return hash(self._dims)

  def __reduce__(self):
    return TensorShape, (self.dims,)

  def __concat__(self, other):
    return self.concatenate(other)

trace_type.register_serializable(TensorShape)


def as_shape(shape):
  """Converts the given object to a TensorShape."""
  if isinstance(shape, TensorShape):
    return shape
  else:
    return TensorShape(shape)


def unknown_shape(rank=None, **kwargs):
  """Returns an unknown TensorShape, optionally with a known rank.

  Args:
    rank: (Optional) If specified, the number of dimensions in the shape.
    **kwargs: For backwards compatibility.

  Returns:
    An unknown TensorShape.

  Raises:
    TypeError: In case of invalid arguments.
  """
  if rank is None and "ndims" in kwargs:
    rank = kwargs.pop("ndims")
  if kwargs:
    raise TypeError("Unknown argument: %s" % kwargs)
  if rank is None:
    return TensorShape(None)
  else:
    return TensorShape([Dimension(None)] * rank)
