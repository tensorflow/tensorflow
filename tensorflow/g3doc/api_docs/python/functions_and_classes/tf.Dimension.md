Represents the value of one dimension in a TensorShape.
- - -

#### `tf.Dimension.__init__(value)` {#Dimension.__init__}

Creates a new Dimension with the given value.


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

    Dimension(n)   .merge_with(Dimension(n))    == Dimension(n)
    Dimension(n)   .merge_with(Dimension(None)) == Dimension(n)
    Dimension(None).merge_with(Dimension(n))    == Dimension(n)
    Dimension(None).merge_with(Dimension(None)) == Dimension(None)
    Dimension(n)   .merge_with(Dimension(m)) raises ValueError for n != m

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


