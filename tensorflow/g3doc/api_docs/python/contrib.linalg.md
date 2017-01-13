<!-- This file is machine generated: DO NOT EDIT! -->

# Linear Algebra (contrib)
[TOC]

Linear algebra libraries for TensorFlow.

## `LinearOperator`

Subclasses of `LinearOperator` provide a access to common methods on a
(batch) matrix, without the need to materialize the matrix.  This allows:

* Matrix free computations
* Different operators to take advantage of special strcture, while providing a
  consistent API to users.

### Base class

- - -

### `class tf.contrib.linalg.LinearOperator` {#LinearOperator}

Base class defining a [batch of] linear operator[s].

Subclasses of `LinearOperator` provide a access to common methods on a
(batch) matrix, without the need to materialize the matrix.  This allows:

* Matrix free computations
* Operators that take advantage of special structure, while providing a
  consistent API to users.

#### Subclassing

To enable a public method, subclasses should implement the leading-underscore
version of the method.  The argument signature should be identical except for
the omission of `name="..."`.  For example, to enable
`apply(x, adjoint=False, name="apply")` a subclass should implement
`_apply(x, adjoint=False)`.

#### Performance contract

Subclasses should implement a method only if it can be done with a reasonable
performance increase over generic dense operations, either in time, parallel
scalability, or memory usage.  For example, if the determinant can only be
computed using `tf.matrix_determinant(self.to_dense())`, then determinants
should not be implemented.

Class docstrings should contain an explanation of computational complexity.
Since this is a high-performance library, attention should be paid to detail,
and explanations can include constants as well as Big-O notation.

#### Shape compatibility

`LinearOperator` sub classes should operate on a [batch] matrix with
compatible shape.  Class docstrings should define what is meant by compatible
shape.  Some sub-classes may not support batching.

An example is:

`x` is a batch matrix with compatible shape for `apply` if

```
operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
x.shape =   [B1,...,Bb] + [N, R]
```

`rhs` is a batch matrix with compatible shape for `solve` if

```
operator.shape = [B1,...,Bb] + [M, N],  b >= 0,
rhs.shape =   [B1,...,Bb] + [M, R]
```

#### Example docstring for subclasses.

This operator acts like a (batch) matrix `A` with shape
`[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
an `m x n` matrix.  Again, this matrix `A` may not be materialized, but for
purposes of identifying and working with compatible arguments the shape is
relevant.

Examples:

```python
some_tensor = ... shape = ????
operator = MyLinOp(some_tensor)

operator.shape()
==> [2, 4, 4]

operator.log_determinant()
==> Shape [2] Tensor

x = ... Shape [2, 4, 5] Tensor

operator.apply(x)
==> Shape [2, 4, 5] Tensor
```

#### Shape compatibility

This operator acts on batch matrices with compatible shape.
FILL IN WHAT IS MEANT BY COMPATIBLE SHAPE

#### Performance

FILL THIS IN

#### Matrix property hints

This `LinearOperator` is initialized with boolean flags of the form `is_X`,
for `X = non_singular, self_adjoint, positive_definite`.
These have the following meaning
* If `is_X == True`, callers should expect the operator to have the
  property `X`.  This is a promise that should be fulfilled, but is *not* a
  runtime assert.  For example, finite floating point precision may result
  in these promises being violated.
* If `is_X == False`, callers should expect the operator to not have `X`.
* If `is_X == None` (the default), callers should have no expectation either
  way.
- - -

#### `tf.contrib.linalg.LinearOperator.__init__(dtype, graph_parents=None, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, name=None)` {#LinearOperator.__init__}

Initialize the `LinearOperator`.

**This is a private method for subclass use.**
**Subclasses should copy-paste this `__init__` documentation.**

##### Args:


*  <b>`dtype`</b>: The type of the this `LinearOperator`.  Arguments to `apply` and
    `solve` will have to be this type.
*  <b>`graph_parents`</b>: Python list of graph prerequisites of this `LinearOperator`
    Typically tensors that are passed during initialization.
*  <b>`is_non_singular`</b>: Expect that this operator is non-singular.
*  <b>`is_self_adjoint`</b>: Expect that this operator is equal to its hermitian
    transpose.  If `dtype` is real, this is equivalent to being symmetric.
*  <b>`is_positive_definite`</b>: Expect that this operator is positive definite,
    meaning the real part of all eigenvalues is positive.  We do not require
    the operator to be self-adjoint to be positive-definite.  See:
*  <b>`https`</b>: //en.wikipedia.org/wiki/Positive-definite_matrix\
        #Extension_for_non_symmetric_matrices
*  <b>`name`</b>: A name for this `LinearOperator`.

##### Raises:


*  <b>`ValueError`</b>: if any member of graph_parents is `None` or not a `Tensor`.


- - -

#### `tf.contrib.linalg.LinearOperator.add_to_tensor(x, name='add_to_tensor')` {#LinearOperator.add_to_tensor}

Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

##### Args:


*  <b>`x`</b>: `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with broadcast shape and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperator.apply(x, adjoint=False, name='apply')` {#LinearOperator.apply}

Transform `x` with left multiplication:  `x --> Ax`.

##### Args:


*  <b>`x`</b>: `Tensor` with compatible shape and same `dtype` as `self`.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, left multiply by the adjoint.
*  <b>`name`</b>: A name for this `Op.

##### Returns:

  A `Tensor` with shape `[..., M, R]` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperator.assert_non_singular(name='assert_non_singular')` {#LinearOperator.assert_non_singular}

Returns an `Op` that asserts this operator is non singular.


- - -

#### `tf.contrib.linalg.LinearOperator.assert_positive_definite(name='assert_positive_definite')` {#LinearOperator.assert_positive_definite}

Returns an `Op` that asserts this operator is positive definite.

Here, positive definite means the real part of all eigenvalues is positive.
We do not require the operator to be self-adjoint.

##### Args:


*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  An `Op` that asserts this operator is positive definite.


- - -

#### `tf.contrib.linalg.LinearOperator.assert_self_adjoint(name='assert_self_adjoint')` {#LinearOperator.assert_self_adjoint}

Returns an `Op` that asserts this operator is self-adjoint.


- - -

#### `tf.contrib.linalg.LinearOperator.batch_shape` {#LinearOperator.batch_shape}

`TensorShape` of batch dimensions of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperator.batch_shape_dynamic(name='batch_shape_dynamic')` {#LinearOperator.batch_shape_dynamic}

Shape of batch dimensions of this operator, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb]`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperator.determinant(name='det')` {#LinearOperator.determinant}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperator.domain_dimension` {#LinearOperator.domain_dimension}

Dimension (in the sense of vector spaces) of the domain of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperator.domain_dimension_dynamic(name='domain_dimension_dynamic')` {#LinearOperator.domain_dimension_dynamic}

Dimension (in the sense of vector spaces) of the domain of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperator.dtype` {#LinearOperator.dtype}

The `DType` of `Tensor`s handled by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperator.graph_parents` {#LinearOperator.graph_parents}

List of graph dependencies of this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperator.is_non_singular` {#LinearOperator.is_non_singular}




- - -

#### `tf.contrib.linalg.LinearOperator.is_positive_definite` {#LinearOperator.is_positive_definite}




- - -

#### `tf.contrib.linalg.LinearOperator.is_self_adjoint` {#LinearOperator.is_self_adjoint}




- - -

#### `tf.contrib.linalg.LinearOperator.log_abs_determinant(name='log_abs_det')` {#LinearOperator.log_abs_determinant}

Log absolute value of determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperator.name` {#LinearOperator.name}

Name prepended to all ops created by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperator.range_dimension` {#LinearOperator.range_dimension}

Dimension (in the sense of vector spaces) of the range of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperator.range_dimension_dynamic(name='range_dimension_dynamic')` {#LinearOperator.range_dimension_dynamic}

Dimension (in the sense of vector spaces) of the range of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperator.shape` {#LinearOperator.shape}

`TensorShape` of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperator.shape_dynamic(name='shape_dynamic')` {#LinearOperator.shape_dynamic}

Shape of this `LinearOperator`, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperator.solve(rhs, adjoint=False, name='solve')` {#LinearOperator.solve}

Solve `R` (batch) systems of equations exactly: `A X = rhs`.

Examples:

```python
# Create an operator acting like a 10 x 2 x 2 matrix.
operator = LinearOperator(...)
operator.shape # = 10 x 2 x 2

# Solve one linear system (R = 1) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 1
X = operator.solve(RHS)  # shape 10 x 2 x 1

# Solve five linear systems (R = 5) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 5
X = operator.solve(RHS)
X[3, :, 2]  # Solution to the linear system A[3, :, :] X = RHS[3, :, 2]
```

##### Args:


*  <b>`rhs`</b>: `Tensor` with same `dtype` as this operator and compatible shape.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, solve the system involving the adjoint
    of this `LinearOperator`.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

##### Raises:


*  <b>`ValueError`</b>: If self.is_non_singular is False.


- - -

#### `tf.contrib.linalg.LinearOperator.tensor_rank` {#LinearOperator.tensor_rank}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  Python integer, or None if the tensor rank is undefined.


- - -

#### `tf.contrib.linalg.LinearOperator.tensor_rank_dynamic(name='tensor_rank_dynamic')` {#LinearOperator.tensor_rank_dynamic}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`, determined at runtime.


- - -

#### `tf.contrib.linalg.LinearOperator.to_dense(name='to_dense')` {#LinearOperator.to_dense}

Return a dense (batch) matrix representing this operator.




### Individual operators

- - -

### `class tf.contrib.linalg.LinearOperatorDiag` {#LinearOperatorDiag}

`LinearOperator` acting like a [batch] square diagonal matrix.

This operator acts like a [batch] diagonal matrix `A` with shape
`[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
an `N x N` matrix.  This matrix `A` is not materialized, but for
purposes of broadcasting this shape will be relevant.

`LinearOperatorDiag` is initialized with a (batch) vector.

```python
# Create a 2 x 2 diagonal linear operator.
diag = [1., -1.]
operator = LinearOperatorDiag(diag)

operator.to_dense()
==> [[1.,  0.]
     [0., -1.]]

operator.shape
==> [2, 2]

operator.log_determinant()
==> scalar Tensor

x = ... Shape [2, 4] Tensor
operator.apply(x)
==> Shape [2, 4] Tensor

# Create a [2, 3] batch of 4 x 4 linear operators.
diag = tf.random_normal(shape=[2, 3, 4])
operator = LinearOperatorDiag(diag)

# Create a shape [2, 1, 4, 2] vector.  Note that this shape is compatible
# since the batch dimensions, [2, 1], are brodcast to
# operator.batch_shape = [2, 3].
y = tf.random_normal(shape=[2, 1, 4, 2])
x = operator.solve(y)
==> operator.apply(x) = y
```

#### Shape compatibility

This operator acts on [batch] matrix with compatible shape.
`x` is a batch matrix with compatible shape for `apply` and `solve` if

```
operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
x.shape =   [C1,...,Cc] + [N, R],
and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
```

#### Performance

Suppose `operator` is a `LinearOperatorDiag` of shape `[N, N]`,
and `x.shape = [N, R]`.  Then

* `operator.apply(x)` involves `N * R` multiplications.
* `operator.solve(x)` involves `N` divisions and `N * R` multiplications.
* `operator.determinant()` involves a size `N` `reduce_prod`.

If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
`[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

#### Matrix property hints

This `LinearOperator` is initialized with boolean flags of the form `is_X`,
for `X = non_singular, self_adjoint, positive_definite`.
These have the following meaning
* If `is_X == True`, callers should expect the operator to have the
  property `X`.  This is a promise that should be fulfilled, but is *not* a
  runtime assert.  For example, finite floating point precision may result
  in these promises being violated.
* If `is_X == False`, callers should expect the operator to not have `X`.
* If `is_X == None` (the default), callers should have no expectation either
  way.
- - -

#### `tf.contrib.linalg.LinearOperatorDiag.__init__(diag, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, name='LinearOperatorDiag')` {#LinearOperatorDiag.__init__}

Initialize a `LinearOperatorDiag`.

##### Args:


*  <b>`diag`</b>: Shape `[B1,...,Bb, N]` `Tensor` with `b >= 0` `N >= 0`.
    The diagonal of the operator.  Allowed dtypes: `float32`, `float64`,
      `complex64`, `complex128`.
*  <b>`is_non_singular`</b>: Expect that this operator is non-singular.
*  <b>`is_self_adjoint`</b>: Expect that this operator is equal to its hermitian
    transpose.  If `diag.dtype` is real, this is auto-set to `True`.
*  <b>`is_positive_definite`</b>: Expect that this operator is positive definite,
    meaning the real part of all eigenvalues is positive.  We do not require
    the operator to be self-adjoint to be positive-definite.  See:
*  <b>`https`</b>: //en.wikipedia.org/wiki/Positive-definite_matrix
        #Extension_for_non_symmetric_matrices
*  <b>`name`</b>: A name for this `LinearOperator`.

##### Raises:


*  <b>`TypeError`</b>: If `diag.dtype` is not an allowed type.
*  <b>`ValueError`</b>: If `diag.dtype` is real, and `is_self_adjoint` is not `True`.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.add_to_tensor(x, name='add_to_tensor')` {#LinearOperatorDiag.add_to_tensor}

Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

##### Args:


*  <b>`x`</b>: `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with broadcast shape and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.apply(x, adjoint=False, name='apply')` {#LinearOperatorDiag.apply}

Transform `x` with left multiplication:  `x --> Ax`.

##### Args:


*  <b>`x`</b>: `Tensor` with compatible shape and same `dtype` as `self`.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, left multiply by the adjoint.
*  <b>`name`</b>: A name for this `Op.

##### Returns:

  A `Tensor` with shape `[..., M, R]` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.assert_non_singular(name='assert_non_singular')` {#LinearOperatorDiag.assert_non_singular}

Returns an `Op` that asserts this operator is non singular.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.assert_positive_definite(name='assert_positive_definite')` {#LinearOperatorDiag.assert_positive_definite}

Returns an `Op` that asserts this operator is positive definite.

Here, positive definite means the real part of all eigenvalues is positive.
We do not require the operator to be self-adjoint.

##### Args:


*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  An `Op` that asserts this operator is positive definite.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.assert_self_adjoint(name='assert_self_adjoint')` {#LinearOperatorDiag.assert_self_adjoint}

Returns an `Op` that asserts this operator is self-adjoint.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.batch_shape` {#LinearOperatorDiag.batch_shape}

`TensorShape` of batch dimensions of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.batch_shape_dynamic(name='batch_shape_dynamic')` {#LinearOperatorDiag.batch_shape_dynamic}

Shape of batch dimensions of this operator, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb]`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.determinant(name='det')` {#LinearOperatorDiag.determinant}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.domain_dimension` {#LinearOperatorDiag.domain_dimension}

Dimension (in the sense of vector spaces) of the domain of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.domain_dimension_dynamic(name='domain_dimension_dynamic')` {#LinearOperatorDiag.domain_dimension_dynamic}

Dimension (in the sense of vector spaces) of the domain of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.dtype` {#LinearOperatorDiag.dtype}

The `DType` of `Tensor`s handled by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.graph_parents` {#LinearOperatorDiag.graph_parents}

List of graph dependencies of this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.is_non_singular` {#LinearOperatorDiag.is_non_singular}




- - -

#### `tf.contrib.linalg.LinearOperatorDiag.is_positive_definite` {#LinearOperatorDiag.is_positive_definite}




- - -

#### `tf.contrib.linalg.LinearOperatorDiag.is_self_adjoint` {#LinearOperatorDiag.is_self_adjoint}




- - -

#### `tf.contrib.linalg.LinearOperatorDiag.log_abs_determinant(name='log_abs_det')` {#LinearOperatorDiag.log_abs_determinant}

Log absolute value of determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.name` {#LinearOperatorDiag.name}

Name prepended to all ops created by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.range_dimension` {#LinearOperatorDiag.range_dimension}

Dimension (in the sense of vector spaces) of the range of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.range_dimension_dynamic(name='range_dimension_dynamic')` {#LinearOperatorDiag.range_dimension_dynamic}

Dimension (in the sense of vector spaces) of the range of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.shape` {#LinearOperatorDiag.shape}

`TensorShape` of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.shape_dynamic(name='shape_dynamic')` {#LinearOperatorDiag.shape_dynamic}

Shape of this `LinearOperator`, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.solve(rhs, adjoint=False, name='solve')` {#LinearOperatorDiag.solve}

Solve `R` (batch) systems of equations exactly: `A X = rhs`.

Examples:

```python
# Create an operator acting like a 10 x 2 x 2 matrix.
operator = LinearOperator(...)
operator.shape # = 10 x 2 x 2

# Solve one linear system (R = 1) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 1
X = operator.solve(RHS)  # shape 10 x 2 x 1

# Solve five linear systems (R = 5) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 5
X = operator.solve(RHS)
X[3, :, 2]  # Solution to the linear system A[3, :, :] X = RHS[3, :, 2]
```

##### Args:


*  <b>`rhs`</b>: `Tensor` with same `dtype` as this operator and compatible shape.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, solve the system involving the adjoint
    of this `LinearOperator`.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

##### Raises:


*  <b>`ValueError`</b>: If self.is_non_singular is False.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.tensor_rank` {#LinearOperatorDiag.tensor_rank}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  Python integer, or None if the tensor rank is undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.tensor_rank_dynamic(name='tensor_rank_dynamic')` {#LinearOperatorDiag.tensor_rank_dynamic}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`, determined at runtime.


- - -

#### `tf.contrib.linalg.LinearOperatorDiag.to_dense(name='to_dense')` {#LinearOperatorDiag.to_dense}

Return a dense (batch) matrix representing this operator.



- - -

### `class tf.contrib.linalg.LinearOperatorIdentity` {#LinearOperatorIdentity}

`LinearOperator` acting like a [batch] square identity matrix.

This operator acts like a [batch] identity matrix `A` with shape
`[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
an `N x N` matrix.  This matrix `A` is not materialized, but for
purposes of broadcasting this shape will be relevant.

`LinearOperatorIdentity` is initialized with `num_rows`, and optionally
`batch_shape`, and `dtype` arguments.  If `batch_shape` is `None`, this
operator efficiently passes through all arguments.  If `batch_shape` is
provided, broadcasting may occur, which will require making copies.

```python
# Create a 2 x 2 identity matrix.
operator = LinearOperatorIdentity(num_rows=2, dtype=tf.float32)

operator.to_dense()
==> [[1., 0.]
     [0., 1.]]

operator.shape
==> [2, 2]

operator.log_determinant()
==> 0.

x = ... Shape [2, 4] Tensor
operator.apply(x)
==> Shape [2, 4] Tensor, same as x.

y = tf.random_normal(shape=[3, 2, 4])
# Note that y.shape is compatible with operator.shape because operator.shape
# is broadcast to [3, 2, 2].
# This broadcast does NOT require copying data, since we can infer that y
# will be passed through without changing shape.  We are always able to infer
# this if the operator has no batch_shape.
x = operator.solve(y)
==> Shape [3, 2, 4] Tensor, same as y.

# Create a 2-batch of 2x2 identity matrices
operator = LinearOperatorIdentity(num_rows=2, batch_shape=[2])
operator.to_dense()
==> [[[1., 0.]
      [0., 1.]],
     [[1., 0.]
      [0., 1.]]]

# Here, even though the operator has a batch shape, the input is the same as
# the output, so x can be passed through without a copy.  The operator is able
# to detect that no broadcast is necessary because both x and the operator
# have statically defined shape.
x = ... Shape [2, 2, 3]
operator.apply(x)
==> Shape [2, 2, 3] Tensor, same as x

# Here the operator and x have different batch_shape, and are broadcast.
# This requires a copy, since the output is different size than the input.
x = ... Shape [1, 2, 3]
operator.apply(x)
==> Shape [2, 2, 3] Tensor, equal to [x, x]
```

### Shape compatibility

This operator acts on [batch] matrix with compatible shape.
`x` is a batch matrix with compatible shape for `apply` and `solve` if

```
operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
x.shape =   [C1,...,Cc] + [N, R],
and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
```

### Performance

If `batch_shape` initialization arg is `None`:

* `operator.apply(x)` is `O(1)`
* `operator.solve(x)` is `O(1)`
* `operator.determinant()` is `O(1)`

If `batch_shape` initialization arg is provided, and static checks cannot
rule out the need to broadcast:

* `operator.apply(x)` is `O(D1*...*Dd*N*R)`
* `operator.solve(x)` is `O(D1*...*Dd*N*R)`
* `operator.determinant()` is `O(B1*...*Bb)`

#### Matrix property hints

This `LinearOperator` is initialized with boolean flags of the form `is_X`,
for `X = non_singular, self_adjoint, positive_definite`.
These have the following meaning
* If `is_X == True`, callers should expect the operator to have the
  property `X`.  This is a promise that should be fulfilled, but is *not* a
  runtime assert.  For example, finite floating point precision may result
  in these promises being violated.
* If `is_X == False`, callers should expect the operator to not have `X`.
* If `is_X == None` (the default), callers should have no expectation either
  way.
- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.__init__(num_rows, batch_shape=None, dtype=None, is_non_singular=True, is_self_adjoint=True, is_positive_definite=True, assert_proper_shapes=False, name='LinearOperatorIdentity')` {#LinearOperatorIdentity.__init__}

Initialize a `LinearOperatorIdentity`.

The `LinearOperatorIdentity` is initialized with arguments defining `dtype`
and shape.

This operator is able to broadcast the leading (batch) dimensions, which
sometimes requires copying data.  If `batch_shape` is `None`, the operator
can take arguments of any batch shape without copying.  See examples.

##### Args:


*  <b>`num_rows`</b>: Scalar non-negative integer `Tensor`.  Number of rows in the
    corresponding identity matrix.
*  <b>`batch_shape`</b>: Optional `1-D` integer `Tensor`.  The shape of the leading
    dimensions.  If `None`, this operator has no leading dimensions.
*  <b>`dtype`</b>: Data type of the matrix that this operator represents.
*  <b>`is_non_singular`</b>: Expect that this operator is non-singular.
*  <b>`is_self_adjoint`</b>: Expect that this operator is equal to its hermitian
    transpose.
*  <b>`is_positive_definite`</b>: Expect that this operator is positive definite.
*  <b>`assert_proper_shapes`</b>: Python `bool`.  If `False`, only perform static
    checks that initialization and method arguments have proper shape.
    If `True`, and static checks are inconclusive, add asserts to the graph.
*  <b>`name`</b>: A name for this `LinearOperator`

##### Raises:


*  <b>`ValueError`</b>: If `num_rows` is determined statically to be non-scalar, or
    negative.
*  <b>`ValueError`</b>: If `batch_shape` is determined statically to not be 1-D, or
    negative.
*  <b>`ValueError`</b>: If any of the following is not `True`:
    `{is_self_adjoint, is_non_singular, is_positive_definite}`.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.add_to_tensor(mat, name='add_to_tensor')` {#LinearOperatorIdentity.add_to_tensor}

Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.

##### Args:


*  <b>`mat`</b>: `Tensor` with same `dtype` and shape broadcastable to `self`.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with broadcast shape and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.apply(x, adjoint=False, name='apply')` {#LinearOperatorIdentity.apply}

Transform `x` with left multiplication:  `x --> Ax`.

##### Args:


*  <b>`x`</b>: `Tensor` with compatible shape and same `dtype` as `self`.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, left multiply by the adjoint.
*  <b>`name`</b>: A name for this `Op.

##### Returns:

  A `Tensor` with shape `[..., M, R]` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.assert_non_singular(name='assert_non_singular')` {#LinearOperatorIdentity.assert_non_singular}

Returns an `Op` that asserts this operator is non singular.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.assert_positive_definite(name='assert_positive_definite')` {#LinearOperatorIdentity.assert_positive_definite}

Returns an `Op` that asserts this operator is positive definite.

Here, positive definite means the real part of all eigenvalues is positive.
We do not require the operator to be self-adjoint.

##### Args:


*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  An `Op` that asserts this operator is positive definite.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.assert_self_adjoint(name='assert_self_adjoint')` {#LinearOperatorIdentity.assert_self_adjoint}

Returns an `Op` that asserts this operator is self-adjoint.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.batch_shape` {#LinearOperatorIdentity.batch_shape}

`TensorShape` of batch dimensions of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.batch_shape_dynamic(name='batch_shape_dynamic')` {#LinearOperatorIdentity.batch_shape_dynamic}

Shape of batch dimensions of this operator, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb]`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.determinant(name='det')` {#LinearOperatorIdentity.determinant}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.domain_dimension` {#LinearOperatorIdentity.domain_dimension}

Dimension (in the sense of vector spaces) of the domain of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.domain_dimension_dynamic(name='domain_dimension_dynamic')` {#LinearOperatorIdentity.domain_dimension_dynamic}

Dimension (in the sense of vector spaces) of the domain of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.dtype` {#LinearOperatorIdentity.dtype}

The `DType` of `Tensor`s handled by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.graph_parents` {#LinearOperatorIdentity.graph_parents}

List of graph dependencies of this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.is_non_singular` {#LinearOperatorIdentity.is_non_singular}




- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.is_positive_definite` {#LinearOperatorIdentity.is_positive_definite}




- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.is_self_adjoint` {#LinearOperatorIdentity.is_self_adjoint}




- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.log_abs_determinant(name='log_abs_det')` {#LinearOperatorIdentity.log_abs_determinant}

Log absolute value of determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.name` {#LinearOperatorIdentity.name}

Name prepended to all ops created by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.range_dimension` {#LinearOperatorIdentity.range_dimension}

Dimension (in the sense of vector spaces) of the range of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.range_dimension_dynamic(name='range_dimension_dynamic')` {#LinearOperatorIdentity.range_dimension_dynamic}

Dimension (in the sense of vector spaces) of the range of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.shape` {#LinearOperatorIdentity.shape}

`TensorShape` of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.shape_dynamic(name='shape_dynamic')` {#LinearOperatorIdentity.shape_dynamic}

Shape of this `LinearOperator`, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.solve(rhs, adjoint=False, name='solve')` {#LinearOperatorIdentity.solve}

Solve `R` (batch) systems of equations exactly: `A X = rhs`.

Examples:

```python
# Create an operator acting like a 10 x 2 x 2 matrix.
operator = LinearOperator(...)
operator.shape # = 10 x 2 x 2

# Solve one linear system (R = 1) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 1
X = operator.solve(RHS)  # shape 10 x 2 x 1

# Solve five linear systems (R = 5) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 5
X = operator.solve(RHS)
X[3, :, 2]  # Solution to the linear system A[3, :, :] X = RHS[3, :, 2]
```

##### Args:


*  <b>`rhs`</b>: `Tensor` with same `dtype` as this operator and compatible shape.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, solve the system involving the adjoint
    of this `LinearOperator`.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

##### Raises:


*  <b>`ValueError`</b>: If self.is_non_singular is False.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.tensor_rank` {#LinearOperatorIdentity.tensor_rank}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  Python integer, or None if the tensor rank is undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.tensor_rank_dynamic(name='tensor_rank_dynamic')` {#LinearOperatorIdentity.tensor_rank_dynamic}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`, determined at runtime.


- - -

#### `tf.contrib.linalg.LinearOperatorIdentity.to_dense(name='to_dense')` {#LinearOperatorIdentity.to_dense}

Return a dense (batch) matrix representing this operator.



- - -

### `class tf.contrib.linalg.LinearOperatorMatrix` {#LinearOperatorMatrix}

`LinearOperator` that wraps a [batch] matrix.

This operator wraps a [batch] matrix `A` (which is a `Tensor`) with shape
`[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
an `M x N` matrix.

```python
# Create a 2 x 2 linear operator.
matrix = [[1., 2.], [3., 4.]]
operator = LinearOperatorMatrix(matrix)

operator.to_dense()
==> [[1., 2.]
     [3., 4.]]

operator.shape
==> [2, 2]

operator.log_determinant()
==> scalar Tensor

x = ... Shape [2, 4] Tensor
operator.apply(x)
==> Shape [2, 4] Tensor

# Create a [2, 3] batch of 4 x 4 linear operators.
matrix = tf.random_normal(shape=[2, 3, 4, 4])
operator = LinearOperatorMatrix(matrix)
```

#### Shape compatibility

This operator acts on [batch] matrix with compatible shape.
`x` is a batch matrix with compatible shape for `apply` and `solve` if

```
operator.shape = [B1,...,Bb] + [M, N],  with b >= 0
x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
```

#### Performance

`LinearOperatorMatrix` has exactly the same performance as would be achieved
by using standard `TensorFlow` matrix ops.  Intelligent choices are made
based on the following initialization hints.

* If `dtype` is real, and `is_self_adjoint` and `is_positive_definite`, a
  Cholesky factorization is used for the determinant and solve.

In all cases, suppose `operator` is a `LinearOperatorMatrix` of shape
`[M, N]`, and `x.shape = [N, R]`.  Then

* `operator.apply(x)` is `O(M * N * R)`.
* If `M=N`, `operator.solve(x)` is `O(N^3 * R)`.
* If `M=N`, `operator.determinant()` is `O(N^3)`.

If instead `operator` and `x` have shape `[B1,...,Bb, M, N]` and
`[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

#### Matrix property hints

This `LinearOperator` is initialized with boolean flags of the form `is_X`,
for `X = non_singular, self_adjoint, positive_definite`.
These have the following meaning
* If `is_X == True`, callers should expect the operator to have the
  property `X`.  This is a promise that should be fulfilled, but is *not* a
  runtime assert.  For example, finite floating point precision may result
  in these promises being violated.
* If `is_X == False`, callers should expect the operator to not have `X`.
* If `is_X == None` (the default), callers should have no expectation either
  way.
- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.__init__(matrix, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, name='LinearOperatorMatrix')` {#LinearOperatorMatrix.__init__}

Initialize a `LinearOperatorMatrix`.

##### Args:


*  <b>`matrix`</b>: Shape `[B1,...,Bb, M, N]` with `b >= 0`, `M, N >= 0`.
    Allowed dtypes: `float32`, `float64`, `complex64`, `complex128`.
*  <b>`is_non_singular`</b>: Expect that this operator is non-singular.
*  <b>`is_self_adjoint`</b>: Expect that this operator is equal to its hermitian
    transpose.
*  <b>`is_positive_definite`</b>: Expect that this operator is positive definite,
    meaning the real part of all eigenvalues is positive.  We do not require
    the operator to be self-adjoint to be positive-definite.  See:
*  <b>`https`</b>: //en.wikipedia.org/wiki/Positive-definite_matrix
        #Extension_for_non_symmetric_matrices
*  <b>`name`</b>: A name for this `LinearOperator`.

##### Raises:


*  <b>`TypeError`</b>: If `diag.dtype` is not an allowed type.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.add_to_tensor(x, name='add_to_tensor')` {#LinearOperatorMatrix.add_to_tensor}

Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

##### Args:


*  <b>`x`</b>: `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with broadcast shape and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.apply(x, adjoint=False, name='apply')` {#LinearOperatorMatrix.apply}

Transform `x` with left multiplication:  `x --> Ax`.

##### Args:


*  <b>`x`</b>: `Tensor` with compatible shape and same `dtype` as `self`.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, left multiply by the adjoint.
*  <b>`name`</b>: A name for this `Op.

##### Returns:

  A `Tensor` with shape `[..., M, R]` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.assert_non_singular(name='assert_non_singular')` {#LinearOperatorMatrix.assert_non_singular}

Returns an `Op` that asserts this operator is non singular.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.assert_positive_definite(name='assert_positive_definite')` {#LinearOperatorMatrix.assert_positive_definite}

Returns an `Op` that asserts this operator is positive definite.

Here, positive definite means the real part of all eigenvalues is positive.
We do not require the operator to be self-adjoint.

##### Args:


*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  An `Op` that asserts this operator is positive definite.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.assert_self_adjoint(name='assert_self_adjoint')` {#LinearOperatorMatrix.assert_self_adjoint}

Returns an `Op` that asserts this operator is self-adjoint.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.batch_shape` {#LinearOperatorMatrix.batch_shape}

`TensorShape` of batch dimensions of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.batch_shape_dynamic(name='batch_shape_dynamic')` {#LinearOperatorMatrix.batch_shape_dynamic}

Shape of batch dimensions of this operator, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb]`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.determinant(name='det')` {#LinearOperatorMatrix.determinant}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.domain_dimension` {#LinearOperatorMatrix.domain_dimension}

Dimension (in the sense of vector spaces) of the domain of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.domain_dimension_dynamic(name='domain_dimension_dynamic')` {#LinearOperatorMatrix.domain_dimension_dynamic}

Dimension (in the sense of vector spaces) of the domain of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.dtype` {#LinearOperatorMatrix.dtype}

The `DType` of `Tensor`s handled by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.graph_parents` {#LinearOperatorMatrix.graph_parents}

List of graph dependencies of this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.is_non_singular` {#LinearOperatorMatrix.is_non_singular}




- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.is_positive_definite` {#LinearOperatorMatrix.is_positive_definite}




- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.is_self_adjoint` {#LinearOperatorMatrix.is_self_adjoint}




- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.log_abs_determinant(name='log_abs_det')` {#LinearOperatorMatrix.log_abs_determinant}

Log absolute value of determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.name` {#LinearOperatorMatrix.name}

Name prepended to all ops created by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.range_dimension` {#LinearOperatorMatrix.range_dimension}

Dimension (in the sense of vector spaces) of the range of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.range_dimension_dynamic(name='range_dimension_dynamic')` {#LinearOperatorMatrix.range_dimension_dynamic}

Dimension (in the sense of vector spaces) of the range of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.shape` {#LinearOperatorMatrix.shape}

`TensorShape` of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.shape_dynamic(name='shape_dynamic')` {#LinearOperatorMatrix.shape_dynamic}

Shape of this `LinearOperator`, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.solve(rhs, adjoint=False, name='solve')` {#LinearOperatorMatrix.solve}

Solve `R` (batch) systems of equations exactly: `A X = rhs`.

Examples:

```python
# Create an operator acting like a 10 x 2 x 2 matrix.
operator = LinearOperator(...)
operator.shape # = 10 x 2 x 2

# Solve one linear system (R = 1) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 1
X = operator.solve(RHS)  # shape 10 x 2 x 1

# Solve five linear systems (R = 5) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 5
X = operator.solve(RHS)
X[3, :, 2]  # Solution to the linear system A[3, :, :] X = RHS[3, :, 2]
```

##### Args:


*  <b>`rhs`</b>: `Tensor` with same `dtype` as this operator and compatible shape.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, solve the system involving the adjoint
    of this `LinearOperator`.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

##### Raises:


*  <b>`ValueError`</b>: If self.is_non_singular is False.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.tensor_rank` {#LinearOperatorMatrix.tensor_rank}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  Python integer, or None if the tensor rank is undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.tensor_rank_dynamic(name='tensor_rank_dynamic')` {#LinearOperatorMatrix.tensor_rank_dynamic}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`, determined at runtime.


- - -

#### `tf.contrib.linalg.LinearOperatorMatrix.to_dense(name='to_dense')` {#LinearOperatorMatrix.to_dense}

Return a dense (batch) matrix representing this operator.



- - -

### `class tf.contrib.linalg.LinearOperatorTriL` {#LinearOperatorTriL}

`LinearOperator` acting like a [batch] square lower triangular matrix.

This operator acts like a [batch] lower triangular matrix `A` with shape
`[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
an `N x N` matrix.

`LinearOperatorTriL` is initialized with a `Tensor` having dimensions
`[B1,...,Bb, N, N]`. The upper triangle of the last two dimensions is ignored.

```python
# Create a 2 x 2 lower-triangular linear operator.
tril = [[1., 2.], [3., 4.]]
operator = LinearOperatorTriL(tril)

# The upper triangle is ignored.
operator.to_dense()
==> [[1., 0.]
     [3., 4.]]

operator.shape
==> [2, 2]

operator.log_determinant()
==> scalar Tensor

x = ... Shape [2, 4] Tensor
operator.apply(x)
==> Shape [2, 4] Tensor

# Create a [2, 3] batch of 4 x 4 linear operators.
tril = tf.random_normal(shape=[2, 3, 4, 4])
operator = LinearOperatorTriL(tril)
```

#### Shape compatibility

This operator acts on [batch] matrix with compatible shape.
`x` is a batch matrix with compatible shape for `apply` and `solve` if

```
operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
```

#### Performance

Suppose `operator` is a `LinearOperatorTriL` of shape `[N, N]`,
and `x.shape = [N, R]`.  Then

* `operator.apply(x)` involves `N^2 * R` multiplications.
* `operator.solve(x)` involves `N * R` size `N` back-substitutions.
* `operator.determinant()` involves a size `N` `reduce_prod`.

If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
`[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

#### Matrix property hints

This `LinearOperator` is initialized with boolean flags of the form `is_X`,
for `X = non_singular, self_adjoint, positive_definite`.
These have the following meaning
* If `is_X == True`, callers should expect the operator to have the
  property `X`.  This is a promise that should be fulfilled, but is *not* a
  runtime assert.  For example, finite floating point precision may result
  in these promises being violated.
* If `is_X == False`, callers should expect the operator to not have `X`.
* If `is_X == None` (the default), callers should have no expectation either
  way.
- - -

#### `tf.contrib.linalg.LinearOperatorTriL.__init__(tril, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, name='LinearOperatorTriL')` {#LinearOperatorTriL.__init__}

Initialize a `LinearOperatorTriL`.

##### Args:


*  <b>`tril`</b>: Shape `[B1,...,Bb, N, N]` with `b >= 0`, `N >= 0`.
    The lower triangular part of `tril` defines this operator.  The strictly
    upper triangle is ignored.  Allowed dtypes: `float32`, `float64`.
*  <b>`is_non_singular`</b>: Expect that this operator is non-singular.
    This operator is non-singular if and only if its diagonal elements are
    all non-zero.
*  <b>`is_self_adjoint`</b>: Expect that this operator is equal to its hermitian
    transpose.  This operator is self-adjoint only if it is diagonal with
    real-valued diagonal entries.  In this case it is advised to use
    `LinearOperatorDiag`.
*  <b>`is_positive_definite`</b>: Expect that this operator is positive definite,
    meaning the real part of all eigenvalues is positive.  We do not require
    the operator to be self-adjoint to be positive-definite.  See:
*  <b>`https`</b>: //en.wikipedia.org/wiki/Positive-definite_matrix
        #Extension_for_non_symmetric_matrices
*  <b>`name`</b>: A name for this `LinearOperator`.

##### Raises:


*  <b>`TypeError`</b>: If `diag.dtype` is not an allowed type.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.add_to_tensor(x, name='add_to_tensor')` {#LinearOperatorTriL.add_to_tensor}

Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

##### Args:


*  <b>`x`</b>: `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with broadcast shape and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.apply(x, adjoint=False, name='apply')` {#LinearOperatorTriL.apply}

Transform `x` with left multiplication:  `x --> Ax`.

##### Args:


*  <b>`x`</b>: `Tensor` with compatible shape and same `dtype` as `self`.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, left multiply by the adjoint.
*  <b>`name`</b>: A name for this `Op.

##### Returns:

  A `Tensor` with shape `[..., M, R]` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.assert_non_singular(name='assert_non_singular')` {#LinearOperatorTriL.assert_non_singular}

Returns an `Op` that asserts this operator is non singular.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.assert_positive_definite(name='assert_positive_definite')` {#LinearOperatorTriL.assert_positive_definite}

Returns an `Op` that asserts this operator is positive definite.

Here, positive definite means the real part of all eigenvalues is positive.
We do not require the operator to be self-adjoint.

##### Args:


*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  An `Op` that asserts this operator is positive definite.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.assert_self_adjoint(name='assert_self_adjoint')` {#LinearOperatorTriL.assert_self_adjoint}

Returns an `Op` that asserts this operator is self-adjoint.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.batch_shape` {#LinearOperatorTriL.batch_shape}

`TensorShape` of batch dimensions of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.batch_shape_dynamic(name='batch_shape_dynamic')` {#LinearOperatorTriL.batch_shape_dynamic}

Shape of batch dimensions of this operator, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb]`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.determinant(name='det')` {#LinearOperatorTriL.determinant}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.domain_dimension` {#LinearOperatorTriL.domain_dimension}

Dimension (in the sense of vector spaces) of the domain of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.domain_dimension_dynamic(name='domain_dimension_dynamic')` {#LinearOperatorTriL.domain_dimension_dynamic}

Dimension (in the sense of vector spaces) of the domain of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.dtype` {#LinearOperatorTriL.dtype}

The `DType` of `Tensor`s handled by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.graph_parents` {#LinearOperatorTriL.graph_parents}

List of graph dependencies of this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.is_non_singular` {#LinearOperatorTriL.is_non_singular}




- - -

#### `tf.contrib.linalg.LinearOperatorTriL.is_positive_definite` {#LinearOperatorTriL.is_positive_definite}




- - -

#### `tf.contrib.linalg.LinearOperatorTriL.is_self_adjoint` {#LinearOperatorTriL.is_self_adjoint}




- - -

#### `tf.contrib.linalg.LinearOperatorTriL.log_abs_determinant(name='log_abs_det')` {#LinearOperatorTriL.log_abs_determinant}

Log absolute value of determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.name` {#LinearOperatorTriL.name}

Name prepended to all ops created by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.range_dimension` {#LinearOperatorTriL.range_dimension}

Dimension (in the sense of vector spaces) of the range of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.range_dimension_dynamic(name='range_dimension_dynamic')` {#LinearOperatorTriL.range_dimension_dynamic}

Dimension (in the sense of vector spaces) of the range of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.shape` {#LinearOperatorTriL.shape}

`TensorShape` of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.shape_dynamic(name='shape_dynamic')` {#LinearOperatorTriL.shape_dynamic}

Shape of this `LinearOperator`, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.solve(rhs, adjoint=False, name='solve')` {#LinearOperatorTriL.solve}

Solve `R` (batch) systems of equations exactly: `A X = rhs`.

Examples:

```python
# Create an operator acting like a 10 x 2 x 2 matrix.
operator = LinearOperator(...)
operator.shape # = 10 x 2 x 2

# Solve one linear system (R = 1) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 1
X = operator.solve(RHS)  # shape 10 x 2 x 1

# Solve five linear systems (R = 5) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 5
X = operator.solve(RHS)
X[3, :, 2]  # Solution to the linear system A[3, :, :] X = RHS[3, :, 2]
```

##### Args:


*  <b>`rhs`</b>: `Tensor` with same `dtype` as this operator and compatible shape.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, solve the system involving the adjoint
    of this `LinearOperator`.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

##### Raises:


*  <b>`ValueError`</b>: If self.is_non_singular is False.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.tensor_rank` {#LinearOperatorTriL.tensor_rank}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  Python integer, or None if the tensor rank is undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.tensor_rank_dynamic(name='tensor_rank_dynamic')` {#LinearOperatorTriL.tensor_rank_dynamic}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`, determined at runtime.


- - -

#### `tf.contrib.linalg.LinearOperatorTriL.to_dense(name='to_dense')` {#LinearOperatorTriL.to_dense}

Return a dense (batch) matrix representing this operator.




### Transformations and Combinations of operators

- - -

### `class tf.contrib.linalg.LinearOperatorComposition` {#LinearOperatorComposition}

Composes one or more `LinearOperators`.

This operator composes one or more linear operators `[op1,...,opJ]`,
building a new `LinearOperator` with action defined by:

```
op_composed(x) := op1(op2(...(opJ(x)...))
```

If `opj` acts like [batch] matrix `Aj`, then `op_composed` acts like the
[batch] matrix formed with the multiplication `A1 A2...AJ`.

If `opj` has shape `batch_shape_j + [M_j, N_j]`, then we must have
`N_j = M_{j+1}`, in which case the composed operator has shape equal to
`broadcast_batch_shape + [M_1, N_J]`, where `broadcast_batch_shape` is the
mutual broadcast of `batch_shape_j`, `j = 1,...,J`, assuming the intermediate
batch shapes broadcast.  Even if the composed shape is well defined, the
composed operator's methods may fail due to lack of broadcasting ability in
the defining operators' methods.

```python
# Create a 2 x 2 linear operator composed of two 2 x 2 operators.
operator_1 = LinearOperatorMatrix([[1., 2.], [3., 4.]])
operator_2 = LinearOperatorMatrix([[1., 0.], [0., 1.]])
operator = LinearOperatorComposition([operator_1, operator_2])

operator.to_dense()
==> [[1., 2.]
     [3., 4.]]

operator.shape
==> [2, 2]

operator.log_determinant()
==> scalar Tensor

x = ... Shape [2, 4] Tensor
operator.apply(x)
==> Shape [2, 4] Tensor

# Create a [2, 3] batch of 4 x 5 linear operators.
matrix_45 = tf.random_normal(shape=[2, 3, 4, 5])
operator_45 = LinearOperatorMatrix(matrix)

# Create a [2, 3] batch of 5 x 6 linear operators.
matrix_56 = tf.random_normal(shape=[2, 3, 5, 6])
operator_56 = LinearOperatorMatrix(matrix_56)

# Compose to create a [2, 3] batch of 4 x 6 operators.
opeartor_46 = LinearOperatorComposition([operator_45, operator_56])

# Create a shape [2, 3, 6, 2] vector.
x = tf.random_normal(shape=[2, 3, 6, 2])
operator.apply(x)
==> Shape [2, 3, 4, 2] Tensor
```

#### Performance

The performance of `LinearOperatorComposition` on any operation is equal to
the sum of the individual operators' operations.


#### Matrix property hints

This `LinearOperator` is initialized with boolean flags of the form `is_X`,
for `X = non_singular, self_adjoint, positive_definite`.
These have the following meaning
* If `is_X == True`, callers should expect the operator to have the
  property `X`.  This is a promise that should be fulfilled, but is *not* a
  runtime assert.  For example, finite floating point precision may result
  in these promises being violated.
* If `is_X == False`, callers should expect the operator to not have `X`.
* If `is_X == None` (the default), callers should have no expectation either
  way.
- - -

#### `tf.contrib.linalg.LinearOperatorComposition.__init__(operators, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, name=None)` {#LinearOperatorComposition.__init__}

Initialize a `LinearOperatorComposition`.

`LinearOperatorComposition` is initialized with a list of operators
`[op_1,...,op_J]`.  For the `apply` method to be well defined, the
composition `op_i.apply(op_{i+1}(x))` must be defined.  Other methods have
similar constraints.

##### Args:


*  <b>`operators`</b>: Iterable of `LinearOperator` objects, each with
    the same `dtype` and composible shape.
*  <b>`is_non_singular`</b>: Expect that this operator is non-singular.
*  <b>`is_self_adjoint`</b>: Expect that this operator is equal to its hermitian
    transpose.
*  <b>`is_positive_definite`</b>: Expect that this operator is positive definite,
    meaning the real part of all eigenvalues is positive.  We do not require
    the operator to be self-adjoint to be positive-definite.  See:
*  <b>`https`</b>: //en.wikipedia.org/wiki/Positive-definite_matrix
        #Extension_for_non_symmetric_matrices
*  <b>`name`</b>: A name for this `LinearOperator`.  Default is the individual
    operators names joined with `_o_`.

##### Raises:


*  <b>`TypeError`</b>: If all operators do not have the same `dtype`.
*  <b>`ValueError`</b>: If `operators` is empty.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.add_to_tensor(x, name='add_to_tensor')` {#LinearOperatorComposition.add_to_tensor}

Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

##### Args:


*  <b>`x`</b>: `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with broadcast shape and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.apply(x, adjoint=False, name='apply')` {#LinearOperatorComposition.apply}

Transform `x` with left multiplication:  `x --> Ax`.

##### Args:


*  <b>`x`</b>: `Tensor` with compatible shape and same `dtype` as `self`.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, left multiply by the adjoint.
*  <b>`name`</b>: A name for this `Op.

##### Returns:

  A `Tensor` with shape `[..., M, R]` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.assert_non_singular(name='assert_non_singular')` {#LinearOperatorComposition.assert_non_singular}

Returns an `Op` that asserts this operator is non singular.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.assert_positive_definite(name='assert_positive_definite')` {#LinearOperatorComposition.assert_positive_definite}

Returns an `Op` that asserts this operator is positive definite.

Here, positive definite means the real part of all eigenvalues is positive.
We do not require the operator to be self-adjoint.

##### Args:


*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  An `Op` that asserts this operator is positive definite.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.assert_self_adjoint(name='assert_self_adjoint')` {#LinearOperatorComposition.assert_self_adjoint}

Returns an `Op` that asserts this operator is self-adjoint.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.batch_shape` {#LinearOperatorComposition.batch_shape}

`TensorShape` of batch dimensions of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.batch_shape_dynamic(name='batch_shape_dynamic')` {#LinearOperatorComposition.batch_shape_dynamic}

Shape of batch dimensions of this operator, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb]`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.determinant(name='det')` {#LinearOperatorComposition.determinant}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.domain_dimension` {#LinearOperatorComposition.domain_dimension}

Dimension (in the sense of vector spaces) of the domain of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.domain_dimension_dynamic(name='domain_dimension_dynamic')` {#LinearOperatorComposition.domain_dimension_dynamic}

Dimension (in the sense of vector spaces) of the domain of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.dtype` {#LinearOperatorComposition.dtype}

The `DType` of `Tensor`s handled by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.graph_parents` {#LinearOperatorComposition.graph_parents}

List of graph dependencies of this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.is_non_singular` {#LinearOperatorComposition.is_non_singular}




- - -

#### `tf.contrib.linalg.LinearOperatorComposition.is_positive_definite` {#LinearOperatorComposition.is_positive_definite}




- - -

#### `tf.contrib.linalg.LinearOperatorComposition.is_self_adjoint` {#LinearOperatorComposition.is_self_adjoint}




- - -

#### `tf.contrib.linalg.LinearOperatorComposition.log_abs_determinant(name='log_abs_det')` {#LinearOperatorComposition.log_abs_determinant}

Log absolute value of determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.name` {#LinearOperatorComposition.name}

Name prepended to all ops created by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.operators` {#LinearOperatorComposition.operators}




- - -

#### `tf.contrib.linalg.LinearOperatorComposition.range_dimension` {#LinearOperatorComposition.range_dimension}

Dimension (in the sense of vector spaces) of the range of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.range_dimension_dynamic(name='range_dimension_dynamic')` {#LinearOperatorComposition.range_dimension_dynamic}

Dimension (in the sense of vector spaces) of the range of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.shape` {#LinearOperatorComposition.shape}

`TensorShape` of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.shape_dynamic(name='shape_dynamic')` {#LinearOperatorComposition.shape_dynamic}

Shape of this `LinearOperator`, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.solve(rhs, adjoint=False, name='solve')` {#LinearOperatorComposition.solve}

Solve `R` (batch) systems of equations exactly: `A X = rhs`.

Examples:

```python
# Create an operator acting like a 10 x 2 x 2 matrix.
operator = LinearOperator(...)
operator.shape # = 10 x 2 x 2

# Solve one linear system (R = 1) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 1
X = operator.solve(RHS)  # shape 10 x 2 x 1

# Solve five linear systems (R = 5) for every member of the length 10 batch.
RHS = ... # shape 10 x 2 x 5
X = operator.solve(RHS)
X[3, :, 2]  # Solution to the linear system A[3, :, :] X = RHS[3, :, 2]
```

##### Args:


*  <b>`rhs`</b>: `Tensor` with same `dtype` as this operator and compatible shape.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, solve the system involving the adjoint
    of this `LinearOperator`.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` with shape `[...,N, R]` and same `dtype` as `rhs`.

##### Raises:


*  <b>`ValueError`</b>: If self.is_non_singular is False.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.tensor_rank` {#LinearOperatorComposition.tensor_rank}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  Python integer, or None if the tensor rank is undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.tensor_rank_dynamic(name='tensor_rank_dynamic')` {#LinearOperatorComposition.tensor_rank_dynamic}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`, determined at runtime.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.to_dense(name='to_dense')` {#LinearOperatorComposition.to_dense}

Return a dense (batch) matrix representing this operator.



