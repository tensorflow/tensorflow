"""Helper classes for tensor shape inference."""
import tensorflow.python.platform


class Dimension(object):
  """Represents the value of one dimension in a TensorShape."""

  def __init__(self, value):
    """Creates a new Dimension with the given value."""
    if value is None:
      self._value = None
    else:
      self._value = int(value)

  def __repr__(self):
    return "Dimension(%s)" % repr(self._value)

  def __eq__(self, other):
    """Returns true if `other` has the same known value as this Dimension."""
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    return self._value == other.value

  def __ne__(self, other):
    """Returns true if `other` has a different known value from `self`."""
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return None
    return self._value != other.value

  def __int__(self):
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
    return (self._value is None
            or other.value is None
            or self._value == other.value)

  def assert_is_compatible_with(self, other):
    """Raises an exception if `other` is not compatible with this Dimension.

    Args:
      other: Another Dimension.

    Raises:
      ValueError: If `self` and `other` are not compatible (see
        is_compatible_with).
    """
    if not self.is_compatible_with(other):
      raise ValueError("Dimensions %s and %s are not compatible"
                       % (self, other))

  def merge_with(self, other):
    """Returns a Dimension that combines the information in `self` and `other`.

    Dimensions are combined as follows:

      Dimension(n)   .merge_with(Dimension(n))    == Dimension(n)
      Dimension(n)   .merge_with(Dimension(None)) == Dimension(n)
      Dimension(None).merge_with(Dimension(n))    == Dimension(n)
      Dimension(None).merge_with(Dimension(None)) == Dimension(None)
      Dimension(n)   .merge_with(Dimension(m)) raises ValueError for n != m

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

      Dimension(m)    + Dimension(n)    == Dimension(m + n)
      Dimension(m)    + Dimension(None) == Dimension(None)
      Dimension(None) + Dimension(n)    == Dimension(None)
      Dimension(None) + Dimension(None) == Dimension(None)

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

      Dimension(m)    - Dimension(n)    == Dimension(m - n)
      Dimension(m)    - Dimension(None) == Dimension(None)
      Dimension(None) - Dimension(n)    == Dimension(None)
      Dimension(None) - Dimension(None) == Dimension(None)

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

      Dimension(m)    * Dimension(n)    == Dimension(m * n)
      Dimension(m)    * Dimension(None) == Dimension(None)
      Dimension(None) * Dimension(n)    == Dimension(None)
      Dimension(None) * Dimension(None) == Dimension(None)

    Args:
      other: Another Dimension.

    Returns:
      A Dimension whose value is the sum of `self` and `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value * other.value)

  def __div__(self, other):
    """Returns the quotient of `self` and `other`.

    Dimensions are summed as follows:

      Dimension(m)    / Dimension(n)    == Dimension(m / n)
      Dimension(m)    / Dimension(None) == Dimension(None)
      Dimension(None) / Dimension(n)    == Dimension(None)
      Dimension(None) / Dimension(None) == Dimension(None)

    Args:
      other: Another Dimension.

    Returns:
      A Dimension whose value is the sum of `self` and `other`.
    """
    other = as_dimension(other)
    if self._value is None or other.value is None:
      return Dimension(None)
    else:
      return Dimension(self._value / other.value)

  def __mod__(self, other):
    """Returns `self` modulo `other.

    Dimension moduli are computed  as follows:

      Dimension(m)    % Dimension(n)     == Dimension(m % n)
      Dimension(m)    % Dimension(None)  == Dimension(None)
      Dimension(None) % Dimension(n)     == Dimension(None)
      Dimension(None) %  Dimension(None) == Dimension(None)

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

      Dimension(m)    < Dimension(n)    == m < n
      Dimension(m)    < Dimension(None) == None
      Dimension(None) < Dimension(n)    == None
      Dimension(None) < Dimension(None) == None

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

      Dimension(m)    <= Dimension(n)    == m <= n
      Dimension(m)    <= Dimension(None) == None
      Dimension(None) <= Dimension(n)    == None
      Dimension(None) <= Dimension(None) == None

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

      Dimension(m)    > Dimension(n)    == m > n
      Dimension(m)    > Dimension(None) == None
      Dimension(None) > Dimension(n)    == None
      Dimension(None) > Dimension(None) == None

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

      Dimension(m)    >= Dimension(n)    == m >= n
      Dimension(m)    >= Dimension(None) == None
      Dimension(None) >= Dimension(n)    == None
      Dimension(None) >= Dimension(None) == None

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

  A Dimenson input will be returned unmodified.
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
    for each dimension.
  * *Partially-known shape:* has a known number of dimensions, and an unknown
    size for one or more dimension.
  * *Unknown shape:* has an unknown number of dimensions, and an unknown
    size in all dimensions.

  If a tensor is produced by an operation of type `"Foo"`, its shape
  may be inferred if there is a registered shape function for
  `"Foo"`. See [`tf.RegisterShape()`](../../api_docs/python/framework.md#RegisterShape)
  for details of shape
  functions and how to register them. Alternatively, the shape may be set
  explicitly using [`Tensor.set_shape()`](../../api_docs/python/framework.md#Tensor.set_shape).

  @@merge_with
  @@concatenate

  @@ndims
  @@dims
  @@as_list
  @@is_compatible_with
  @@is_fully_defined

  @@with_rank
  @@with_rank_at_least
  @@with_rank_at_most

  @@assert_has_rank
  @@assert_same_rank
  @@assert_is_compatible_with
  @@assert_is_fully_defined
  """

  def __init__(self, dims):
    """Creates a new TensorShape with the given dimensions.

    Args:
      dims: A list of Dimensions, or None if the shape is unspecified.
        DEPRECATED: A single integer is treated as a singleton list.
    """
    # TODO(irving): Eliminate the single integer special case.
    if dims is None:
      self._dims = None
    else:
      try:
        dims_iter = iter(dims)
      except TypeError:
        # Treat as a singleton dimension
        self._dims = [as_dimension(dims)]
      else:
        # Got a list of dimensions
        self._dims = map(as_dimension, dims_iter)

  def __repr__(self):
    return "TensorShape(%s)" % str(self._dims)

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

  def __nonzero__(self):
    """Returns True if this shape contains non-zero information."""
    return self._dims is not None

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
          return unknown_shape(ndims=stop-start)
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
      self.assert_same_rank(other)
      new_dims = []
      for i, dim in enumerate(self._dims):
        new_dims.append(dim.merge_with(other[i]))
      return TensorShape(new_dims)

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
        raise ValueError(
            "Shapes %s and %s must have the same rank" % (self, other))

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
    return self.merge_with(unknown_shape(ndims=rank))

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

  def is_fully_defined(self):
    """Returns True iff `self` is fully defined in every dimension."""
    return (self._dims is not None
            and all(dim.value is not None for dim in self._dims))

  def assert_is_fully_defined(self):
    """Raises an exception if `self` is not fully defined in every dimension.

    Raises:
      ValueError: If `self` does not have a known value for every dimension.
    """
    if not self.is_fully_defined():
      raise ValueError("Shape %s is not fully defined" % self)

  def as_dimension_list(self):
    """DEPRECATED: use as_list()."""
    self.assert_is_fully_defined()
    return self.as_list()

  def as_list(self):
    """Returns a list of integers or None for each dimension."""
    return [dim.value for dim in self._dims]

  def __eq__(self, other):
    """Returns True if `self` is equivalent to `other`."""
    other = as_shape(other)
    return self._dims == other.dims

  def __ne__(self, other):
    """Returns True if `self` is known to be different from `other`."""
    other = as_shape(other)
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
    return TensorShape([Dimension(None) for _ in range(ndims)])


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
