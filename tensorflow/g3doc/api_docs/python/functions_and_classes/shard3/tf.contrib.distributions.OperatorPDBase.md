Class representing a (batch) of positive definite matrices `A`.

This class provides access to functions of a (batch) symmetric positive
definite (PD) matrix, without the need to materialize them.  In other words,
this provides means to do "matrix free" computations.

### Basics

For example, `my_operator.matmul(x)` computes the result of matrix
multiplication, and this class is free to do this computation with or without
ever materializing a matrix.

In practice, this operator represents a (batch) matrix `A` with shape
`[N1,...,Nn, k, k]` for some `n >= 0`.  The first `n` indices index a
batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,in, : :]` is
a `k x k` matrix.  Again, this matrix `A` may not be materialized, but for
purposes of broadcasting this shape will be relevant.

Since `A` is (batch) positive definite, it has a (or several) square roots `S`
such that `A = SS^T`.

For example, if `MyOperator` inherits from `OperatorPDBase`, the user can do

```python
operator = MyOperator(...)  # Initialize with some tensors.
operator.log_det()

# Compute the quadratic form x^T A^{-1} x for vector x.
x = ... # some shape [M1,...,Mm, N1,...,Nn, k] tensor
operator.inv_quadratic_form_on_vectors(x)

# Matrix multiplication by the square root, S w.
# If w is iid normal, S w has covariance A.
w = ... # some shape [N1,...,Nn, k, r] tensor, r >= 1
operator.sqrt_matmul(w)
```

The above three methods, `log_det`, `inv_quadratic_form_on_vectors`, and
`sqrt_matmul` provide "all" that is necessary to use a covariance matrix
in a multi-variate normal distribution.  See the class `MVNOperatorPD`.

### Details about shape requirements

The `Operator` classes operate on batch vectors and batch matrices with
compatible shapes.  `matrix` is a batch matrix with compatible shape if

```
operator.shape = [N1,...,Nn] + [j, k]
matrix.shape =   [N1,...,Nn] + [k, r]
```

This is the same requirement as `tf.matmul`.  `vec` is a batch vector with
compatible shape if

```
operator.shape = [N1,...,Nn] + [j, k]
vec.shape =   [N1,...,Nn] + [k]
OR
vec.shape = [M1,...,Mm] + [N1,...,Nn] + [k]
```

We are strict with the matrix shape requirements since we do not want to
require `Operator` broadcasting.  The `Operator` may be defined by large
tensors (thus broadcasting is expensive), or the `Operator` may be matrix
free, in which case there is no guarantee that the underlying implementation
will broadcast.

We are more flexible with vector shapes since extra leading dimensions can
be "flipped" to the end to change the vector to a compatible matrix.
- - -

#### `tf.contrib.distributions.OperatorPDBase.batch_shape(name='batch_shape')` {#OperatorPDBase.batch_shape}

Shape of batches associated with this operator.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, the `batch_shape` is `[N1,...,Nn]`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDBase.det(name='det')` {#OperatorPDBase.det}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  Determinant for every batch member.


- - -

#### `tf.contrib.distributions.OperatorPDBase.dtype` {#OperatorPDBase.dtype}

Data type of matrix elements of `A`.


- - -

#### `tf.contrib.distributions.OperatorPDBase.get_batch_shape()` {#OperatorPDBase.get_batch_shape}

`TensorShape` with batch shape.  Statically determined if possible.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, then this returns `TensorShape([N1,...,Nn])`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.distributions.OperatorPDBase.get_shape()` {#OperatorPDBase.get_shape}

Static `TensorShape` of entire operator.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, then this returns
`TensorShape([N1,...,Nn, k, k])`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.distributions.OperatorPDBase.get_vector_shape()` {#OperatorPDBase.get_vector_shape}

`TensorShape` of vectors this operator will work with.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, then this returns
`TensorShape([N1,...,Nn, k])`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.distributions.OperatorPDBase.inputs` {#OperatorPDBase.inputs}

List of tensors that were provided as initialization inputs.


- - -

#### `tf.contrib.distributions.OperatorPDBase.inv_quadratic_form_on_vectors(x, name='inv_quadratic_form_on_vectors')` {#OperatorPDBase.inv_quadratic_form_on_vectors}

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

#### `tf.contrib.distributions.OperatorPDBase.log_det(name='log_det')` {#OperatorPDBase.log_det}

Log of the determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  Logarithm of determinant for every batch member.


- - -

#### `tf.contrib.distributions.OperatorPDBase.matmul(x, transpose_x=False, name='matmul')` {#OperatorPDBase.matmul}

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

#### `tf.contrib.distributions.OperatorPDBase.name` {#OperatorPDBase.name}

String name identifying this `Operator`.


- - -

#### `tf.contrib.distributions.OperatorPDBase.rank(name='rank')` {#OperatorPDBase.rank}

Tensor rank.  Equivalent to `tf.rank(A)`.  Will equal `n + 2`.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, the `rank` is `n + 2`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDBase.shape(name='shape')` {#OperatorPDBase.shape}

Equivalent to `tf.shape(A).`  Equal to `[N1,...,Nn, k, k]`, `n >= 0`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDBase.solve(rhs, name='solve')` {#OperatorPDBase.solve}

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

#### `tf.contrib.distributions.OperatorPDBase.sqrt_matmul(x, transpose_x=False, name='sqrt_matmul')` {#OperatorPDBase.sqrt_matmul}

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

#### `tf.contrib.distributions.OperatorPDBase.sqrt_solve(rhs, name='sqrt_solve')` {#OperatorPDBase.sqrt_solve}

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

#### `tf.contrib.distributions.OperatorPDBase.sqrt_to_dense(name='sqrt_to_dense')` {#OperatorPDBase.sqrt_to_dense}

Return a dense (batch) matrix representing sqrt of this operator.


- - -

#### `tf.contrib.distributions.OperatorPDBase.to_dense(name='to_dense')` {#OperatorPDBase.to_dense}

Return a dense (batch) matrix representing this operator.


- - -

#### `tf.contrib.distributions.OperatorPDBase.vector_shape(name='vector_shape')` {#OperatorPDBase.vector_shape}

Shape of (batch) vectors that this (batch) matrix will multiply.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, the `vector_shape` is `[N1,...,Nn, k]`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDBase.vector_space_dimension(name='vector_space_dimension')` {#OperatorPDBase.vector_space_dimension}

Dimension of vector space on which this acts.  The `k` in `R^k`.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nn, k, k]`, the `vector_space_dimension` is `k`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDBase.verify_pd` {#OperatorPDBase.verify_pd}

Whether to verify that this `Operator` is positive definite.


