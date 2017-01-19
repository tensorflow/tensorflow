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

#### `tf.contrib.linalg.LinearOperatorTriL.batch_shape_tensor(name='batch_shape_tensor')` {#LinearOperatorTriL.batch_shape_tensor}

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

#### `tf.contrib.linalg.LinearOperatorTriL.domain_dimension_tensor(name='domain_dimension_tensor')` {#LinearOperatorTriL.domain_dimension_tensor}

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

#### `tf.contrib.linalg.LinearOperatorTriL.range_dimension_tensor(name='range_dimension_tensor')` {#LinearOperatorTriL.range_dimension_tensor}

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

#### `tf.contrib.linalg.LinearOperatorTriL.shape_tensor(name='shape_tensor')` {#LinearOperatorTriL.shape_tensor}

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

#### `tf.contrib.linalg.LinearOperatorTriL.tensor_rank_tensor(name='tensor_rank_tensor')` {#LinearOperatorTriL.tensor_rank_tensor}

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


