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

### Subclassing

To enable a public method, subclasses should implement the leading-underscore
version of the method.  The argument signature should be identical except for
the omission of `name="..."`.  For example, to enable
`apply(x, adjoint=False, name="apply")` a subclass should implement
`_apply(x, adjoint=False)`.

### Performance contract

Subclasses should implement a method only if it can be done with a reasonable
performance increase over generic dense operations, either in time, parallel
scalability, or memory usage.  For example, if the determinant can only be
computed using `tf.matrix_determinant(self.to_dense())`, then determinants
should not be implemented.

Class docstrings should contain an explanation of computational complexity.
Since this is a high-performance library, attention should be paid to detail,
and explanations can include constants as well as Big-O notation.

### Shape compatibility

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

### Example docstring for subclasses.

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

### Shape compatibility

This operator acts on batch matrices with compatible shape.
FILL IN WHAT IS MEANT BY COMPATIBLE SHAPE

### Performance

FILL THIS IN
- - -

#### `tf.contrib.linalg.LinearOperator.__init__(dtype, graph_parents=None, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, name=None)` {#LinearOperator.__init__}

Initialize the `LinearOperator`.

**This is a private method for subclass use.**
**Subclasses should copy-paste this `__init__` documentation.**

For `X = non_singular, self_adjoint` etc...
`is_X` is a Python `bool` initialization argument with the following meaning
* If `is_X == True`, callers should expect the operator to have the
  attribute `X`.  This is a promise that should be fulfilled, but is *not* a
  runtime assert.  Issues, such as floating point error, could mean the
  operator violates this promise.
* If `is_X == False`, callers should expect the operator to not have `X`.
* If `is_X == None` (the default), callers should have no expectation either
  way.

##### Args:


*  <b>`dtype`</b>: The type of the this `LinearOperator`.  Arguments to `apply` and
    `solve` will have to be this type.
*  <b>`graph_parents`</b>: Python list of graph prerequisites of this `LinearOperator`
    Typically tensors that are passed during initialization.
*  <b>`is_non_singular`</b>: Expect that this operator is non-singular.
*  <b>`is_self_adjoint`</b>: Expect that this operator is equal to its hermitian
    transpose.  If `dtype` is real, this is equivalent to being symmetric.
*  <b>`is_positive_definite`</b>: Expect that this operator is positive definite.
*  <b>`name`</b>: A name for this `LinearOperator`. Default: subclass name.

##### Raises:


*  <b>`ValueError`</b>: if any member of graph_parents is `None` or not a `Tensor`.


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

  Python integer if vector space dimension can be determined statically,
    otherwise `None`.


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

  Python integer if vector space dimension can be determined statically,
    otherwise `None`.


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

This operator acts like a [batch] matrix `A` with shape
`[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
an `m x n` matrix.  Again, this matrix `A` may not be materialized, but for
purposes of broadcasting this shape will be relevant.

`LinearOperatorDiag` is initialized with a (batch) vector.

```python
# Create a 2 x 2 diagonal linear operator.
diag = [1., -1.]
operator = LinearOperatorDiag(diag)

operator.to_dense()
==> [[1.,  0.]
     [0., -1.]]

operator.shape()
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

### Shape compatibility

This operator acts on [batch] matrix with compatible shape.
`x` is a batch matrix with compatible shape for `apply` and `solve` if

```
operator.shape = [B1,...,Bb] + [N, N],  with b >= 0
x.shape =   [C1,...,Cc] + [N, R],
and [C1,...,Cc] broadcasts with [B1,...,Bb] to [D1,...,Dd]
```

### Performance

Suppose `operator` is a `LinearOperatorDiag` is of shape `[N, N]`,
and `x.shape = [N, R]`.  Then

* `operator.apply(x)` involves `N*R` multiplications.
* `operator.solve(x)` involves `N` divisions and `N*R` multiplications.
* `operator.determinant()` involves a size `N` `reduce_prod`.

If instead `operator` and `x` have shape `[B1,...,Bb, N, N]` and
`[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.
- - -

#### `tf.contrib.linalg.LinearOperatorDiag.__init__(diag, is_non_singular=None, is_self_adjoint=True, is_positive_definite=None, name='LinearOperatorDiag')` {#LinearOperatorDiag.__init__}

Initialize a `LinearOperatorDiag`.

For `X = non_singular, self_adjoint` etc...
`is_X` is a Python `bool` initialization argument with the following meaning
* If `is_X == True`, callers should expect the operator to have the
  attribute `X`.  This is a promise that should be fulfilled, but is *not* a
  runtime assert.  Issues, such as floating point error, could mean the
  operator violates this promise.
* If `is_X == False`, callers should expect the operator to not have `X`.
* If `is_X == None` (the default), callers should have no expectation either
  way.

##### Args:


*  <b>`diag`</b>: Shape `[B1,...,Bb, N]` real float type `Tensor` with `b >= 0`,
    `N >= 0`.  The diagonal of the operator.
*  <b>`is_non_singular`</b>: Expect that this operator is non-singular.
*  <b>`is_self_adjoint`</b>: Expect that this operator is equal to its hermitian
    transpose.  Since this is a real (not complex) diagonal operator, it is
    always self adjoint.
*  <b>`is_positive_definite`</b>: Expect that this operator is positive definite.
*  <b>`name`</b>: A name for this `LinearOperator`. Default: subclass name.

##### Raises:


*  <b>`ValueError`</b>: If `diag.dtype` is not floating point.
*  <b>`ValueError`</b>: If `is_self_adjoint` is not `True`.


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

  Python integer if vector space dimension can be determined statically,
    otherwise `None`.


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

  Python integer if vector space dimension can be determined statically,
    otherwise `None`.


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



