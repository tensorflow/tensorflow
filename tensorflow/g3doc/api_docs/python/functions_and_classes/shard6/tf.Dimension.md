Represents the value of one dimension in a TensorShape.
- - -

#### `tf.Dimension.__add__(other)` {#Dimension.__add__}

Returns the sum of `self` and `other`.

Dimensions are summed as follows:

  Dimension(m)    + Dimension(n)    == Dimension(m + n)
  Dimension(m)    + Dimension(None) == Dimension(None)
  Dimension(None) + Dimension(n)    == Dimension(None)
  Dimension(None) + Dimension(None) == Dimension(None)

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  A Dimension whose value is the sum of `self` and `other`.


- - -

#### `tf.Dimension.__div__(other)` {#Dimension.__div__}

DEPRECATED: Use `__floordiv__` via `x // y` instead.

This function exists only for backwards compatibility purposes; new code
should use `__floordiv__` via the syntax `x // y`.  Using `x // y`
communicates clearly that the result rounds down, and is forward compatible
to Python 3.

##### Args:


*  <b>`other`</b>: Another `Dimension`.

##### Returns:

  A `Dimension` whose value is the integer quotient of `self` and `other`.


- - -

#### `tf.Dimension.__eq__(other)` {#Dimension.__eq__}

Returns true if `other` has the same known value as this Dimension.


- - -

#### `tf.Dimension.__floordiv__(other)` {#Dimension.__floordiv__}

Returns the quotient of `self` and `other` rounded down.

Dimensions are divided as follows:

  Dimension(m)    // Dimension(n)    == Dimension(m // n)
  Dimension(m)    // Dimension(None) == Dimension(None)
  Dimension(None) // Dimension(n)    == Dimension(None)
  Dimension(None) // Dimension(None) == Dimension(None)

##### Args:


*  <b>`other`</b>: Another `Dimension`.

##### Returns:

  A `Dimension` whose value is the integer quotient of `self` and `other`.


- - -

#### `tf.Dimension.__ge__(other)` {#Dimension.__ge__}

Returns True if `self` is known to be greater than or equal to `other`.

Dimensions are compared as follows:

  Dimension(m)    >= Dimension(n)    == m >= n
  Dimension(m)    >= Dimension(None) == None
  Dimension(None) >= Dimension(n)    == None
  Dimension(None) >= Dimension(None) == None

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  The value of `self.value >= other.value` if both are known, otherwise
  None.


- - -

#### `tf.Dimension.__gt__(other)` {#Dimension.__gt__}

Returns True if `self` is known to be greater than `other`.

Dimensions are compared as follows:

  Dimension(m)    > Dimension(n)    == m > n
  Dimension(m)    > Dimension(None) == None
  Dimension(None) > Dimension(n)    == None
  Dimension(None) > Dimension(None) == None

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  The value of `self.value > other.value` if both are known, otherwise
  None.


- - -

#### `tf.Dimension.__index__()` {#Dimension.__index__}




- - -

#### `tf.Dimension.__init__(value)` {#Dimension.__init__}

Creates a new Dimension with the given value.


- - -

#### `tf.Dimension.__int__()` {#Dimension.__int__}




- - -

#### `tf.Dimension.__le__(other)` {#Dimension.__le__}

Returns True if `self` is known to be less than or equal to `other`.

Dimensions are compared as follows:

  Dimension(m)    <= Dimension(n)    == m <= n
  Dimension(m)    <= Dimension(None) == None
  Dimension(None) <= Dimension(n)    == None
  Dimension(None) <= Dimension(None) == None

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  The value of `self.value <= other.value` if both are known, otherwise
  None.


- - -

#### `tf.Dimension.__lt__(other)` {#Dimension.__lt__}

Returns True if `self` is known to be less than `other`.

Dimensions are compared as follows:

  Dimension(m)    < Dimension(n)    == m < n
  Dimension(m)    < Dimension(None) == None
  Dimension(None) < Dimension(n)    == None
  Dimension(None) < Dimension(None) == None

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  The value of `self.value < other.value` if both are known, otherwise
  None.


- - -

#### `tf.Dimension.__mod__(other)` {#Dimension.__mod__}

Returns `self` modulo `other.

Dimension moduli are computed  as follows:

  Dimension(m)    % Dimension(n)     == Dimension(m % n)
  Dimension(m)    % Dimension(None)  == Dimension(None)
  Dimension(None) % Dimension(n)     == Dimension(None)
  Dimension(None) %  Dimension(None) == Dimension(None)

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  A Dimension whose value is `self` modulo `other`.


- - -

#### `tf.Dimension.__mul__(other)` {#Dimension.__mul__}

Returns the product of `self` and `other`.

Dimensions are summed as follows:

```
  Dimension(m)    * Dimension(n)    == Dimension(m * n)
  Dimension(m)    * Dimension(None) == Dimension(None)
  Dimension(None) * Dimension(n)    == Dimension(None)
  Dimension(None) * Dimension(None) == Dimension(None)
```

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  A Dimension whose value is the product of `self` and `other`.


- - -

#### `tf.Dimension.__ne__(other)` {#Dimension.__ne__}

Returns true if `other` has a different known value from `self`.


- - -

#### `tf.Dimension.__repr__()` {#Dimension.__repr__}




- - -

#### `tf.Dimension.__str__()` {#Dimension.__str__}




- - -

#### `tf.Dimension.__sub__(other)` {#Dimension.__sub__}

Returns the subtraction of `other` from `self`.

Dimensions are subtracted as follows:

  Dimension(m)    - Dimension(n)    == Dimension(m - n)
  Dimension(m)    - Dimension(None) == Dimension(None)
  Dimension(None) - Dimension(n)    == Dimension(None)
  Dimension(None) - Dimension(None) == Dimension(None)

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  A Dimension whose value is the subtraction of sum of `other` from `self`.


- - -

#### `tf.Dimension.assert_is_compatible_with(other)` {#Dimension.assert_is_compatible_with}

Raises an exception if `other` is not compatible with this Dimension.

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Raises:


*  <b>`ValueError`</b>: If `self` and `other` are not compatible (see
    is_compatible_with).


- - -

#### `tf.Dimension.is_compatible_with(other)` {#Dimension.is_compatible_with}

Returns true if `other` is compatible with this Dimension.

Two known Dimensions are compatible if they have the same value.
An unknown Dimension is compatible with all other Dimensions.

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  True if this Dimension and `other` are compatible.


- - -

#### `tf.Dimension.merge_with(other)` {#Dimension.merge_with}

Returns a Dimension that combines the information in `self` and `other`.

Dimensions are combined as follows:

```python
    Dimension(n)   .merge_with(Dimension(n))    == Dimension(n)
    Dimension(n)   .merge_with(Dimension(None)) == Dimension(n)
    Dimension(None).merge_with(Dimension(n))    == Dimension(n)
    Dimension(None).merge_with(Dimension(None)) == Dimension(None)
    Dimension(n)   .merge_with(Dimension(m)) raises ValueError for n != m
```

##### Args:


*  <b>`other`</b>: Another Dimension.

##### Returns:

  A Dimension containing the combined information of `self` and
  `other`.

##### Raises:


*  <b>`ValueError`</b>: If `self` and `other` are not compatible (see
    is_compatible_with).


- - -

#### `tf.Dimension.value` {#Dimension.value}

The value of this dimension, or None if it is unknown.


