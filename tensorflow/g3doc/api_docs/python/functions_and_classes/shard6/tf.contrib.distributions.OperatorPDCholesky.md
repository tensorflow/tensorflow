Class representing a (batch) of positive definite matrices `A`.

This class provides access to functions of a batch of symmetric positive
definite (PD) matrices `A` in `R^{k x k}` defined by Cholesky factor(s).
Determinants and solves are `O(k^2)`.

In practice, this operator represents a (batch) matrix `A` with shape
`[N1,...,Nn, k, k]` for some `n >= 0`.  The first `n` indices designate a
batch member.  For every batch member `(i1,...,in)`, `A[i1,...,ib, : :]` is
a `k x k` matrix.

Since `A` is (batch) positive definite, it has a (or several) square roots `S`
such that `A = SS^T`.

For example,

```python
distributions = tf.contrib.distributions
chol = [[1.0, 0.0], [1.0, 2.0]]
operator = OperatorPDCholesky(chol)
operator.log_det()

# Compute the quadratic form x^T A^{-1} x for vector x.
x = [1.0, 2.0]
operator.inv_quadratic_form_on_vectors(x)

# Matrix multiplication by the square root, S w.
# If w is iid normal, S w has covariance A.
w = [[1.0], [2.0]]
operator.sqrt_matmul(w)
```

The above three methods, `log_det`, `inv_quadratic_form_on_vectors`, and
`sqrt_matmul` provide "all" that is necessary to use a covariance matrix
in a multi-variate normal distribution.  See the class
`MultivariateNormalCholesky`.
- - -

#### `tf.contrib.distributions.OperatorPDCholesky.__init__(chol, verify_pd=True, name='OperatorPDCholesky')` {#OperatorPDCholesky.__init__}

Initialize an OperatorPDCholesky.

##### Args:


*  <b>`chol`</b>: Shape `[N1,...,Nn, k, k]` tensor with `n >= 0`, `k >= 1`, and
    positive diagonal elements.  The strict upper triangle of `chol` is
    never used, and the user may set these elements to zero, or ignore them.
*  <b>`verify_pd`</b>: Whether to check that `chol` has positive diagonal (this is
    equivalent to it being a Cholesky factor of a symmetric positive
    definite matrix.  If `verify_pd` is `False`, correct behavior is not
    guaranteed.
*  <b>`name`</b>: A name to prepend to all ops created by this class.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.batch_shape(name='batch_shape')` {#OperatorPDCholesky.batch_shape}

Shape of batches associated with this operator.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, the `batch_shape` is `[N1,...,Nn]`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.det(name='det')` {#OperatorPDCholesky.det}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  Determinant for every batch member.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.dtype` {#OperatorPDCholesky.dtype}




- - -

#### `tf.contrib.distributions.OperatorPDCholesky.get_batch_shape()` {#OperatorPDCholesky.get_batch_shape}

`TensorShape` with batch shape.  Statically determined if possible.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, then this returns `TensorShape([N1,...,Nn])`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.get_shape()` {#OperatorPDCholesky.get_shape}

`TensorShape` giving static shape.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.get_vector_shape()` {#OperatorPDCholesky.get_vector_shape}

`TensorShape` of vectors this operator will work with.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, then this returns
`TensorShape([N1,...,Nn, k])`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.inputs` {#OperatorPDCholesky.inputs}

List of tensors that were provided as initialization inputs.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.inv_quadratic_form_on_vectors(x, name='inv_quadratic_form_on_vectors')` {#OperatorPDCholesky.inv_quadratic_form_on_vectors}

Compute the quadratic form: `x^T A^{-1} x` where `x` is a batch vector.

`x` is a batch vector with compatible shape if

```
self.shape = [N1,...,Nn] + [k, k]
x.shape = [M1,...,Mm] + [N1,...,Nn] + [k]
```

##### Args:


*  <b>`x`</b>: `Tensor` with compatible batch vector shape and same `dtype` as self.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` with shape `[M1,...,Mm] + [N1,...,Nn]` and same `dtype`
    as `self`.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.log_det(name='log_det')` {#OperatorPDCholesky.log_det}

Log of the determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  Logarithm of determinant for every batch member.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.matmul(x, transpose_x=False, name='matmul')` {#OperatorPDCholesky.matmul}

Left (batch) matmul `x` by this matrix:  `Ax`.

`x` is a batch matrix with compatible shape if

```
self.shape = [N1,...,Nn] + [k, k]
x.shape = [N1,...,Nn] + [k, r]
```

##### Args:


*  <b>`x`</b>: `Tensor` with shape `self.batch_shape + [k, r]` and same `dtype` as
    this `Operator`.
*  <b>`transpose_x`</b>: If `True`, `x` is transposed before multiplication.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A result equivalent to `tf.batch_matmul(self.to_dense(), x)`.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.name` {#OperatorPDCholesky.name}




- - -

#### `tf.contrib.distributions.OperatorPDCholesky.rank(name='rank')` {#OperatorPDCholesky.rank}

Tensor rank.  Equivalent to `tf.rank(A)`.  Will equal `n + 2`.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, the `rank` is `n + 2`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.shape(name='shape')` {#OperatorPDCholesky.shape}

Equivalent to `tf.shape(A).`  Equal to `[N1,...,Nn, k, k]`, `n >= 0`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.solve(rhs, name='solve')` {#OperatorPDCholesky.solve}

Solve `r` batch systems: `A X = rhs`.

`rhs` is a batch matrix with compatible shape if

```python
self.shape = [N1,...,Nn] + [k, k]
rhs.shape = [N1,...,Nn] + [k, r]
```

For every batch member, this is done in `O(r*k^2)` complexity using back
substitution.

```python
# Solve one linear system (r = 1) for every member of the length 10 batch.
A = ... # shape 10 x 2 x 2
RHS = ... # shape 10 x 2 x 1
operator.shape # = 10 x 2 x 2
X = operator.squrt_solve(RHS)  # shape 10 x 2 x 1
# operator.squrt_matmul(X) ~ RHS
X[3, :, 0]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 0]

# Solve five linear systems (r = 5) for every member of the length 10 batch.
operator.shape # = 10 x 2 x 2
RHS = ... # shape 10 x 2 x 5
...
X[3, :, 2]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 2]
```

##### Args:


*  <b>`rhs`</b>: `Tensor` with same `dtype` as this operator and compatible shape,
    `rhs.shape = self.shape[:-1] + [r]` for `r >= 1`.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` with same `dtype` and shape as `x`.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.sqrt_matmul(x, transpose_x=False, name='sqrt_matmul')` {#OperatorPDCholesky.sqrt_matmul}

Left (batch) matmul `x` by a sqrt of this matrix: `Sx` where `A = S S^T`.

`x` is a batch matrix with compatible shape if

```
self.shape = [N1,...,Nn] + [k, k]
x.shape = [N1,...,Nn] + [k, r]
```

##### Args:


*  <b>`x`</b>: `Tensor` with shape `self.batch_shape + [k, r]` and same `dtype` as
    this `Operator`.
*  <b>`transpose_x`</b>: If `True`, `x` is transposed before multiplication.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  A result equivalent to `tf.batch_matmul(self.sqrt_to_dense(), x)`.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.sqrt_solve(rhs, name='sqrt_solve')` {#OperatorPDCholesky.sqrt_solve}

Solve `r` batch systems involving sqrt: `S X = rhs` where `A = SS^T`.

`rhs` is a batch matrix with compatible shape if

```python
self.shape = [N1,...,Nn] + [k, k]
rhs.shape = [N1,...,Nn] + [k, r]
```

For every batch member, this is done in `O(r*k^2)` complexity using back
substitution.

```python
# Solve one linear system (r = 1) for every member of the length 10 batch.
A = ... # shape 10 x 2 x 2
RHS = ... # shape 10 x 2 x 1
operator.shape # = 10 x 2 x 2
X = operator.squrt_solve(RHS)  # shape 10 x 2 x 1
# operator.squrt_matmul(X) ~ RHS
X[3, :, 0]  # Solution to the linear system S[3, :, :] x = RHS[3, :, 0]

# Solve five linear systems (r = 5) for every member of the length 10 batch.
operator.shape # = 10 x 2 x 2
RHS = ... # shape 10 x 2 x 5
...
X[3, :, 2]  # Solution to the linear system S[3, :, :] x = RHS[3, :, 2]
```

##### Args:


*  <b>`rhs`</b>: `Tensor` with same `dtype` as this operator and compatible shape,
    `rhs.shape = self.shape[:-1] + [r]` for `r >= 1`.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` with same `dtype` and shape as `x`.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.sqrt_to_dense(name='sqrt_to_dense')` {#OperatorPDCholesky.sqrt_to_dense}

Return a dense (batch) matrix representing sqrt of this operator.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.to_dense(name='to_dense')` {#OperatorPDCholesky.to_dense}

Return a dense (batch) matrix representing this operator.


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.vector_shape(name='vector_shape')` {#OperatorPDCholesky.vector_shape}

Shape of (batch) vectors that this (batch) matrix will multiply.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, the `vector_shape` is `[N1,...,Nn, k]`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.vector_space_dimension(name='vector_space_dimension')` {#OperatorPDCholesky.vector_space_dimension}

Dimension of vector space on which this acts.  The `k` in `R^k`.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, the `vector_space_dimension` is `k`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDCholesky.verify_pd` {#OperatorPDCholesky.verify_pd}

Whether to verify that this `Operator` is positive definite.


