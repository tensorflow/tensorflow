`LinearOperator` acting like a scaled [batch] identity matrix `A = c I`.

This operator acts like a scaled [batch] identity matrix `A` with shape
`[B1,...,Bb, N, N]` for some `b >= 0`.  The first `b` indices index a
batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
a scaled version of the `N x N` identity matrix.

`LinearOperatorIdentity` is initialized with `num_rows`, and a `multiplier`
(a `Tensor`) of shape `[B1,...,Bb]`.  `N` is set to `num_rows`, and the
`multiplier` determines the scale for each batch member.

```python
# Create a 2 x 2 scaled identity matrix.
operator = LinearOperatorIdentity(num_rows=2, multiplier=3.)

operator.to_dense()
==> [[3., 0.]
     [0., 3.]]

operator.shape
==> [2, 2]

operator.log_determinant()
==> 2 * Log[3]

x = ... Shape [2, 4] Tensor
operator.apply(x)
==> 3 * x

y = tf.random_normal(shape=[3, 2, 4])
# Note that y.shape is compatible with operator.shape because operator.shape
# is broadcast to [3, 2, 2].
x = operator.solve(y)
==> 3 * x

# Create a 2-batch of 2x2 identity matrices
operator = LinearOperatorIdentity(num_rows=2, multiplier=5.)
operator.to_dense()
==> [[[5., 0.]
      [0., 5.]],
     [[5., 0.]
      [0., 5.]]]

x = ... Shape [2, 2, 3]
operator.apply(x)
==> 5 * x

# Here the operator and x have different batch_shape, and are broadcast.
x = ... Shape [1, 2, 3]
operator.apply(x)
==> 5 * x
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

* `operator.apply(x)` is `O(D1*...*Dd*N*R)`
* `operator.solve(x)` is `O(D1*...*Dd*N*R)`
* `operator.determinant()` is `O(D1*...*Dd)`

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

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.__init__(num_rows, multiplier, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, assert_proper_shapes=False, name='LinearOperatorScaledIdentity')` {#LinearOperatorScaledIdentity.__init__}

Initialize a `LinearOperatorScaledIdentity`.

The `LinearOperatorScaledIdentity` is initialized with `num_rows`, which
determines the size of each identity matrix, and a `multiplier`,
which defines `dtype`, batch shape, and scale of each matrix.

This operator is able to broadcast the leading (batch) dimensions.

##### Args:


*  <b>`num_rows`</b>: Scalar non-negative integer `Tensor`.  Number of rows in the
    corresponding identity matrix.
*  <b>`multiplier`</b>: `Tensor` of shape `[B1,...,Bb]`, or `[]` (a scalar).
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


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.add_to_tensor(mat, name='add_to_tensor')` {#LinearOperatorScaledIdentity.add_to_tensor}

Add matrix represented by this operator to `mat`.  Equiv to `I + mat`.

##### Args:


*  <b>`mat`</b>: `Tensor` with same `dtype` and shape broadcastable to `self`.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with broadcast shape and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.apply(x, adjoint=False, name='apply')` {#LinearOperatorScaledIdentity.apply}

Transform `x` with left multiplication:  `x --> Ax`.

##### Args:


*  <b>`x`</b>: `Tensor` with compatible shape and same `dtype` as `self`.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, left multiply by the adjoint.
*  <b>`name`</b>: A name for this `Op.

##### Returns:

  A `Tensor` with shape `[..., M, R]` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.assert_non_singular(name='assert_non_singular')` {#LinearOperatorScaledIdentity.assert_non_singular}

Returns an `Op` that asserts this operator is non singular.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.assert_positive_definite(name='assert_positive_definite')` {#LinearOperatorScaledIdentity.assert_positive_definite}

Returns an `Op` that asserts this operator is positive definite.

Here, positive definite means the real part of all eigenvalues is positive.
We do not require the operator to be self-adjoint.

##### Args:


*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  An `Op` that asserts this operator is positive definite.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.assert_self_adjoint(name='assert_self_adjoint')` {#LinearOperatorScaledIdentity.assert_self_adjoint}

Returns an `Op` that asserts this operator is self-adjoint.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.batch_shape` {#LinearOperatorScaledIdentity.batch_shape}

`TensorShape` of batch dimensions of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.batch_shape_tensor(name='batch_shape_tensor')` {#LinearOperatorScaledIdentity.batch_shape_tensor}

Shape of batch dimensions of this operator, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb]`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.determinant(name='det')` {#LinearOperatorScaledIdentity.determinant}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.domain_dimension` {#LinearOperatorScaledIdentity.domain_dimension}

Dimension (in the sense of vector spaces) of the domain of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.domain_dimension_tensor(name='domain_dimension_tensor')` {#LinearOperatorScaledIdentity.domain_dimension_tensor}

Dimension (in the sense of vector spaces) of the domain of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.dtype` {#LinearOperatorScaledIdentity.dtype}

The `DType` of `Tensor`s handled by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.graph_parents` {#LinearOperatorScaledIdentity.graph_parents}

List of graph dependencies of this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.is_non_singular` {#LinearOperatorScaledIdentity.is_non_singular}




- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.is_positive_definite` {#LinearOperatorScaledIdentity.is_positive_definite}




- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.is_self_adjoint` {#LinearOperatorScaledIdentity.is_self_adjoint}




- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.log_abs_determinant(name='log_abs_det')` {#LinearOperatorScaledIdentity.log_abs_determinant}

Log absolute value of determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.multiplier` {#LinearOperatorScaledIdentity.multiplier}

The [batch] scalar `Tensor`, `c` in `cI`.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.name` {#LinearOperatorScaledIdentity.name}

Name prepended to all ops created by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.range_dimension` {#LinearOperatorScaledIdentity.range_dimension}

Dimension (in the sense of vector spaces) of the range of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.range_dimension_tensor(name='range_dimension_tensor')` {#LinearOperatorScaledIdentity.range_dimension_tensor}

Dimension (in the sense of vector spaces) of the range of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.shape` {#LinearOperatorScaledIdentity.shape}

`TensorShape` of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.shape_tensor(name='shape_tensor')` {#LinearOperatorScaledIdentity.shape_tensor}

Shape of this `LinearOperator`, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.solve(rhs, adjoint=False, name='solve')` {#LinearOperatorScaledIdentity.solve}

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

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.tensor_rank` {#LinearOperatorScaledIdentity.tensor_rank}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  Python integer, or None if the tensor rank is undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.tensor_rank_tensor(name='tensor_rank_tensor')` {#LinearOperatorScaledIdentity.tensor_rank_tensor}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`, determined at runtime.


- - -

#### `tf.contrib.linalg.LinearOperatorScaledIdentity.to_dense(name='to_dense')` {#LinearOperatorScaledIdentity.to_dense}

Return a dense (batch) matrix representing this operator.


