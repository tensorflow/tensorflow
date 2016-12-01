Represents the shape of a `Tensor`.

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
`"Foo"`. See [`Shape functions in
C++`](../../how_tos/adding_an_op/index.md#shape-functions-in-c) for
details of shape functions and how to register them. Alternatively,
the shape may be set explicitly using
[`Tensor.set_shape()`](../../api_docs/python/framework.md#Tensor.set_shape).

- - -

#### `tf.TensorShape.merge_with(other)` {#TensorShape.merge_with}

Returns a `TensorShape` combining the information in `self` and `other`.

The dimensions in `self` and `other` are merged elementwise,
according to the rules defined for `Dimension.merge_with()`.

##### Args:


*  <b>`other`</b>: Another `TensorShape`.

##### Returns:

  A `TensorShape` containing the combined information of `self` and
  `other`.

##### Raises:


*  <b>`ValueError`</b>: If `self` and `other` are not compatible.


- - -

#### `tf.TensorShape.concatenate(other)` {#TensorShape.concatenate}

Returns the concatenation of the dimension in `self` and `other`.

*N.B.* If either `self` or `other` is completely unknown,
concatenation will discard information about the other shape. In
future, we might support concatenation that preserves this
information for use with slicing.

##### Args:


*  <b>`other`</b>: Another `TensorShape`.

##### Returns:

  A `TensorShape` whose dimensions are the concatenation of the
  dimensions in `self` and `other`.



- - -

#### `tf.TensorShape.ndims` {#TensorShape.ndims}

Returns the rank of this shape, or None if it is unspecified.


- - -

#### `tf.TensorShape.dims` {#TensorShape.dims}

Returns a list of Dimensions, or None if the shape is unspecified.


- - -

#### `tf.TensorShape.as_list()` {#TensorShape.as_list}

Returns a list of integers or `None` for each dimension.

##### Returns:

  A list of integers or `None` for each dimension.

##### Raises:


*  <b>`ValueError`</b>: If `self` is an unknown shape with an unknown rank.


- - -

#### `tf.TensorShape.as_proto()` {#TensorShape.as_proto}

Returns this shape as a `TensorShapeProto`.


- - -

#### `tf.TensorShape.is_compatible_with(other)` {#TensorShape.is_compatible_with}

Returns True iff `self` is compatible with `other`.

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

##### Args:


*  <b>`other`</b>: Another TensorShape.

##### Returns:

  True iff `self` is compatible with `other`.


- - -

#### `tf.TensorShape.is_fully_defined()` {#TensorShape.is_fully_defined}

Returns True iff `self` is fully defined in every dimension.



- - -

#### `tf.TensorShape.with_rank(rank)` {#TensorShape.with_rank}

Returns a shape based on `self` with the given rank.

This method promotes a completely unknown shape to one with a
known rank.

##### Args:


*  <b>`rank`</b>: An integer.

##### Returns:

  A shape that is at least as specific as `self` with the given rank.

##### Raises:


*  <b>`ValueError`</b>: If `self` does not represent a shape with the given `rank`.


- - -

#### `tf.TensorShape.with_rank_at_least(rank)` {#TensorShape.with_rank_at_least}

Returns a shape based on `self` with at least the given rank.

##### Args:


*  <b>`rank`</b>: An integer.

##### Returns:

  A shape that is at least as specific as `self` with at least the given
  rank.

##### Raises:


*  <b>`ValueError`</b>: If `self` does not represent a shape with at least the given
    `rank`.


- - -

#### `tf.TensorShape.with_rank_at_most(rank)` {#TensorShape.with_rank_at_most}

Returns a shape based on `self` with at most the given rank.

##### Args:


*  <b>`rank`</b>: An integer.

##### Returns:

  A shape that is at least as specific as `self` with at most the given
  rank.

##### Raises:


*  <b>`ValueError`</b>: If `self` does not represent a shape with at most the given
    `rank`.



- - -

#### `tf.TensorShape.assert_has_rank(rank)` {#TensorShape.assert_has_rank}

Raises an exception if `self` is not compatible with the given `rank`.

##### Args:


*  <b>`rank`</b>: An integer.

##### Raises:


*  <b>`ValueError`</b>: If `self` does not represent a shape with the given `rank`.


- - -

#### `tf.TensorShape.assert_same_rank(other)` {#TensorShape.assert_same_rank}

Raises an exception if `self` and `other` do not have compatible ranks.

##### Args:


*  <b>`other`</b>: Another `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: If `self` and `other` do not represent shapes with the
    same rank.


- - -

#### `tf.TensorShape.assert_is_compatible_with(other)` {#TensorShape.assert_is_compatible_with}

Raises exception if `self` and `other` do not represent the same shape.

This method can be used to assert that there exists a shape that both
`self` and `other` represent.

##### Args:


*  <b>`other`</b>: Another TensorShape.

##### Raises:


*  <b>`ValueError`</b>: If `self` and `other` do not represent the same shape.


- - -

#### `tf.TensorShape.assert_is_fully_defined()` {#TensorShape.assert_is_fully_defined}

Raises an exception if `self` is not fully defined in every dimension.

##### Raises:


*  <b>`ValueError`</b>: If `self` does not have a known value for every dimension.



#### Other Methods
- - -

#### `tf.TensorShape.__bool__()` {#TensorShape.__bool__}

Returns True if this shape contains non-zero information.


- - -

#### `tf.TensorShape.__eq__(other)` {#TensorShape.__eq__}

Returns True if `self` is equivalent to `other`.


- - -

#### `tf.TensorShape.__getitem__(key)` {#TensorShape.__getitem__}

Returns the value of a dimension or a shape, depending on the key.

##### Args:


*  <b>`key`</b>: If `key` is an integer, returns the dimension at that index;
    otherwise if `key` is a slice, returns a TensorShape whose
    dimensions are those selected by the slice from `self`.

##### Returns:

  A dimension if `key` is an integer, or a `TensorShape` if `key` is a
  slice.

##### Raises:


*  <b>`ValueError`</b>: If `key` is a slice, and any of its elements are negative, or
    if `self` is completely unknown and the step is set.


- - -

#### `tf.TensorShape.__init__(dims)` {#TensorShape.__init__}

Creates a new TensorShape with the given dimensions.

##### Args:


*  <b>`dims`</b>: A list of Dimensions, or None if the shape is unspecified.
*  <b>`DEPRECATED`</b>: A single integer is treated as a singleton list.

##### Raises:


*  <b>`TypeError`</b>: If dims cannot be converted to a list of dimensions.


- - -

#### `tf.TensorShape.__iter__()` {#TensorShape.__iter__}

Returns `self.dims` if the rank is known, otherwise raises ValueError.


- - -

#### `tf.TensorShape.__len__()` {#TensorShape.__len__}

Returns the rank of this shape, or raises ValueError if unspecified.


- - -

#### `tf.TensorShape.__ne__(other)` {#TensorShape.__ne__}

Returns True if `self` is known to be different from `other`.


- - -

#### `tf.TensorShape.__nonzero__()` {#TensorShape.__nonzero__}

Returns True if this shape contains non-zero information.


- - -

#### `tf.TensorShape.__repr__()` {#TensorShape.__repr__}




- - -

#### `tf.TensorShape.__str__()` {#TensorShape.__str__}




- - -

#### `tf.TensorShape.num_elements()` {#TensorShape.num_elements}

Returns the total number of elements, or none for incomplete shapes.


