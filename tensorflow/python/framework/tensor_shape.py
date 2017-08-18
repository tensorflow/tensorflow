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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.util import compat


class Dimension(object):
  """Represents the value of one dimension in a TensorShape."""

  def __init__(self, value):
    """Creates a new Dimension with the given value."""
    if value is None:
      self._value = None
    else:
      self._value = int(value)
      if (not isinstance(value, compat.bytes_or_text_types) and
          self._value != value):
        raise ValueError("Ambiguous dimension: %s" % value)
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
      raise ValueError("Dimensions %s and %s are not compatible" % (self,
                                                                    other))

  def merge_with(self, other):
    """Returns a Dimension that combines the information in `self` and `other`.

    Dimensions are combined as follows:

    ```python
    tf.Dimension(n)   .merge_with(tf.Dimension(n))    == tf.Dimension(n)
    tf.Dimension(n)   .merge_with(tf.Dimension(None)) == tf.Dimension(n)
    tf.Dimension(None).merge_with(tf.Dimension(n))    == tf.Dimension(n)
    tf.Dimension(None).merge_with(tf.Dimension(None)) == tf.Dimension(None)
    tf.Dimension(n)   .merge_with(tf.Dimension(m))  # raises ValueError for n != m
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
    tf.Dimension(m)    + tf.Dimension(n)    == tf.Dimension(m + n)
    tf.Dimension(m)    + tf.Dimension(None) == tf.Dimension(None)
    tf.Dimension(None) + tf.Dimension(n)    == tf.Dimension(None)
    tf.Dimension(None) + tf.Dimension(None) == tf.Dimension(None)
    ```

    Args:
      other: Another Dimension.

    Returns:
      A Dimension whose value is the sum of `self` and `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value + other.value)

  def __sub__(self, other):
    """Returns the subtraction of `other` from `self`.

    Dimensions are subtracted as follows:

    ```python
    tf.Dimension(m)    - tf.Dimension(n)    == tf.Dimension(m - n)
    tf.Dimension(m)    - tf.Dimension(None) == tf.Dimension(None)
    tf.Dimension(None) - tf.Dimension(n)    == tf.Dimension(None)
    tf.Dimension(None) - tf.Dimension(None) == tf.Dimension(None)
    ```

    Args:
      other: Another Dimension.

    Returns:
      A Dimension whose value is the subtraction of sum of `other` from `self`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value - other.value)

  def __mul__(self, other):
    """Returns the product of `self` and `other`.

    Dimensions are summed as follows:

    ```python
    tf.Dimension(m)    * tf.Dimension(n)    == tf.Dimension(m * n)
    tf.Dimension(m)    * tf.Dimension(None) == tf.Dimension(None)
    tf.Dimension(None) * tf.Dimension(n)    == tf.Dimension(None)
    tf.Dimension(None) * tf.Dimension(None) == tf.Dimension(None)
    ```

    Args:
      other: Another Dimension.

    Returns:
      A Dimension whose value is the product of `self` and `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value * other.value)

  def __floordiv__(self, other):
    """Returns the quotient of `self` and `other` rounded down.

    Dimensions are divided as follows:

    ```python
    tf.Dimension(m)    // tf.Dimension(n)    == tf.Dimension(m // n)
    tf.Dimension(m)    // tf.Dimension(None) == tf.Dimension(None)
    tf.Dimension(None) // tf.Dimension(n)    == tf.Dimension(None)
    tf.Dimension(None) // tf.Dimension(None) == tf.Dimension(None)
    ```

    Args:
      other: Another `Dimension`.

    Returns:
      A `Dimension` whose value is the integer quotient of `self` and `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value // other.value)

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

  def __mod__(self, other):
    """Returns `self` modulo `other.

    Dimension moduli are computed as follows:

    ```python
    tf.Dimension(m)    % tf.Dimension(n)    == tf.Dimension(m % n)
    tf.Dimension(m)    % tf.Dimension(None) == tf.Dimension(None)
    tf.Dimension(None) % tf.Dimension(n)    == tf.Dimension(None)
    tf.Dimension(None) % tf.Dimension(None) == tf.Dimension(None)
    ```

    Args:
      other: Another Dimension.

    Returns:
      A Dimension whose value is `self` modulo `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value % other.value)

  def __lt__(self, other):
    """Returns True if `self` is known to be less than `other`.

    Dimensions are compared as follows:

    ```python
    (tf.Dimension(m)    < tf.Dimension(n))    == (m < n)
    (tf.Dimension(m)    < tf.Dimension(None)) == None
    (tf.Dimension(None) < tf.Dimension(n))    == None
    (tf.Dimension(None) < tf.Dimension(None)) == None
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
    (tf.Dimension(m)    <= tf.Dimension(n))    == (m <= n)
    (tf.Dimension(m)    <= tf.Dimension(None)) == None
    (tf.Dimension(None) <= tf.Dimension(n))    == None
    (tf.Dimension(None) <= tf.Dimension(None)) == None
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
    (tf.Dimension(m)    > tf.Dimension(n))    == (m > n)
    (tf.Dimension(m)    > tf.Dimension(None)) == None
    (tf.Dimension(None) > tf.Dimension(n))    == None
    (tf.Dimension(None) > tf.Dimension(None)) == None
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
    (tf.Dimension(m)    >= tf.Dimension(n))    == (m >= n)
    (tf.Dimension(m)    >= tf.Dimension(None)) == None
    (tf.Dimension(None) >= tf.Dimension(n))    == None
    (tf.Dimension(None) >= tf.Dimension(None)) == None
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


class TensorShape(object):
  """Represents the shape of a `Tensor`.

  A `TensorShape` represents a possibly-partial shape specification for a
  `Tensor`. It may be one of the following:

  * *Fully-known shape:* has a known number of dimensions and a known size
    for each dimension. e.g. `TensorShape([16, 256])`
  * *Partially-known shape:* has a known number of dimensions, and an unknown
    size for one or more dimension. e.g. `TensorShape([None, 256])`
  * *Unknown shape:* has an unknown number of dimensions, and an unknown
    size in all dimensions. e.g. `TensorShape(None)`

  If a tensor is produced by an operation of type `"Foo"`, its shape
  may be inferred if there is a registered shape function for
  `"Foo"`. See @{$adding_an_op#shape-functions-in-c$`Shape functions in C++`}
  for details of shape functions and how to register them. Alternatively,
  the shape may be set explicitly using @{tf.Tensor.set_shape}.
  """

  def __init__(self, dims):
    """Creates a new TensorShape with the given dimensions.

    Args:
      dims: A list of Dimensions, or None if the shape is unspecified.
        DEPRECATED: A single integer is treated as a singleton list.

    Raises:
      TypeError: If dims cannot be converted to a list of dimensions.
    """
    # TODO(irving): Eliminate the single integer special case.
    if dims is None:
      self._dims = None
    elif isinstance(dims, compat.bytes_or_text_types):
      raise TypeError("A string has ambiguous TensorShape, please wrap in a "
                      "list or convert to an int: %s" % dims)
    elif isinstance(dims, tensor_shape_pb2.TensorShapeProto):
      if dims.unknown_rank:
        self._dims = None
      else:
        self._dims = [
            # Protos store variable-size dimensions as -1
            as_dimension(dim.size if dim.size != -1 else None)
            for dim in dims.dim
        ]
    elif isinstance(dims, TensorShape):
      self._dims = dims.dims
    else:
      try:
        dims_iter = iter(dims)
      except TypeError:
        # Treat as a singleton dimension
        self._dims = [as_dimension(dims)]
      else:
        # Got a list of dimensions
        self._dims = [as_dimension(d) for d in dims_iter]

  def __repr__(self):
    return "TensorShape(%r)" % self._dims

  def __str__(self):
    if self.ndims is None:
      return "<unknown>"
    elif self.ndims == 1:
      return "(%s,)" % self._dims[0]
    else:
      return "(%s)" % ", ".join(str(d) for d in self._dims)

  @property
  def dims(self):
    """Returns a list of Dimensions, or None if the shape is unspecified."""
    return self._dims

  @property
  def ndims(self):
    """Returns the rank of this shape, or None if it is unspecified."""
    if self._dims is None:
      return None
    else:
      return len(self._dims)

  def __len__(self):
    """Returns the rank of this shape, or raises ValueError if unspecified."""
    if self._dims is None:
      raise ValueError("Cannot take the length of Shape with unknown rank.")
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
      return iter(self._dims)

  def __getitem__(self, key):
    """Returns the value of a dimension or a shape, depending on the key.

    Args:
      key: If `key` is an integer, returns the dimension at that index;
        otherwise if `key` is a slice, returns a TensorShape whose
        dimensions are those selected by the slice from `self`.

    Returns:
      A dimension if `key` is an integer, or a `TensorShape` if `key` is a
      slice.

    Raises:
      ValueError: If `key` is a slice, and any of its elements are negative, or
        if `self` is completely unknown and the step is set.
    """
    if self._dims is not None:
      if isinstance(key, slice):
        return TensorShape(self._dims[key])
      else:
        return self._dims[key]
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
          return unknown_shape(ndims=stop - start)
      else:
        return Dimension(None)

  def num_elements(self):
    """Returns the total number of elements, or none for incomplete shapes."""
    if self.is_fully_defined():
      size = 1
      for dim in self._dims:
        size *= dim.value
      return size
    else:
      return None

  def merge_with(self, other):
    """Returns a `TensorShape` combining the information in `self` and `other`.

    The dimensions in `self` and `other` are merged elementwise,
    according to the rules defined for `Dimension.merge_with()`.

    Args:
      other: Another `TensorShape`.

    Returns:
      A `TensorShape` containing the combined information of `self` and
      `other`.

    Raises:
      ValueError: If `self` and `other` are not compatible.
    """
    other = as_shape(other)
    if self._dims is None:
      return other
    else:
      try:
        self.assert_same_rank(other)
        new_dims = []
        for i, dim in enumerate(self._dims):
          new_dims.append(dim.merge_with(other[i]))
        return TensorShape(new_dims)
      except ValueError:
        raise ValueError("Shapes %s and %s are not compatible" % (self, other))

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
    if self._dims is None or other.dims is None:
      return unknown_shape()
    else:
      return TensorShape(self._dims + other.dims)

  def assert_same_rank(self, other):
    """Raises an exception if `self` and `other` do not have compatible ranks.

    Args:
      other: Another `TensorShape`.

    Raises:
      ValueError: If `self` and `other` do not represent shapes with the
        same rank.
    """
    other = as_shape(other)
    if self.ndims is not None and other.ndims is not None:
      if self.ndims != other.ndims:
        raise ValueError("Shapes %s and %s must have the same rank" % (self,
                                                                       other))

  def assert_has_rank(self, rank):
    """Raises an exception if `self` is not compatible with the given `rank`.

    Args:
      rank: An integer.

    Raises:
      ValueError: If `self` does not represent a shape with the given `rank`.
    """
    if self.ndims not in (None, rank):
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
      return self.merge_with(unknown_shape(ndims=rank))
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
    if self.ndims is not None and self.ndims < rank:
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
    if self.ndims is not None and self.ndims > rank:
      raise ValueError("Shape %s must have rank at most %d" % (self, rank))
    else:
      return self

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
    if self._dims is not None and other.dims is not None:
      if self.ndims != other.ndims:
        return False
      for x_dim, y_dim in zip(self._dims, other.dims):
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
    if self._dims is None or other.dims is None or self.ndims != other.ndims:
      return unknown_shape()

    dims = [(Dimension(None))] * self.ndims
    for i, (d1, d2) in enumerate(zip(self._dims, other.dims)):
      if d1 is not None and d2 is not None and d1 == d2:
        dims[i] = d1
    return TensorShape(dims)

  def is_fully_defined(self):
    """Returns True iff `self` is fully defined in every dimension."""
    return (self._dims is not None and all(dim.value is not None
                                           for dim in self._dims))

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
    return [dim.value for dim in self._dims]

  def as_proto(self):
    """Returns this shape as a `TensorShapeProto`."""
    if self._dims is None:
      return tensor_shape_pb2.TensorShapeProto(unknown_rank=True)
    else:
      return tensor_shape_pb2.TensorShapeProto(dim=[
          tensor_shape_pb2.TensorShapeProto.Dim(size=-1
                                                if d.value is None else d.value)
          for d in self._dims
      ])

  def __eq__(self, other):
    """Returns True if `self` is equivalent to `other`."""
    try:
      other = as_shape(other)
    except TypeError:
      return NotImplemented
    return self._dims == other.dims

  def __ne__(self, other):
    """Returns True if `self` is known to be different from `other`."""
    try:
      other = as_shape(other)
    except TypeError:
      return NotImplemented
    if self.ndims is None or other.ndims is None:
      raise ValueError("The inequality of unknown TensorShapes is undefined.")
    if self.ndims != other.ndims:
      return True
    return self._dims != other.dims


def as_shape(shape):
  """Converts the given object to a TensorShape."""
  if isinstance(shape, TensorShape):
    return shape
  else:
    return TensorShape(shape)


def unknown_shape(ndims=None):
  """Returns an unknown TensorShape, optionally with a known rank.

  Args:
    ndims: (Optional) If specified, the number of dimensions in the shape.

  Returns:
    An unknown TensorShape.
  """
  if ndims is None:
    return TensorShape(None)
  else:
    return TensorShape([Dimension(None)] * ndims)


def scalar():
  """Returns a shape representing a scalar."""
  return TensorShape([])


def vector(length):
  """Returns a shape representing a vector.

  Args:
    length: The length of the vector, which may be None if unknown.

  Returns:
    A TensorShape representing a vector of the given length.
  """
  return TensorShape([length])


def matrix(rows, cols):
  """Returns a shape representing a matrix.

  Args:
    rows: The number of rows in the matrix, which may be None if unknown.
    cols: The number of columns in the matrix, which may be None if unknown.

  Returns:
    A TensorShape representing a matrix of the given size.
  """
  return TensorShape([rows, cols])
