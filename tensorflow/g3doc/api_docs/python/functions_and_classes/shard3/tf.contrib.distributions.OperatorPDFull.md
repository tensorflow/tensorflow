Class representing a (batch) of positive definite matrices `A`.

This class provides access to functions of a batch of symmetric positive
definite (PD) matrices `A` in `R^{k x k}` defined by dense matrices.
Determinants and solves are `O(k^3)`.

In practice, this operator represents a (batch) matrix `A` with shape
`[N1,...,Nb, k, k]` for some `b >= 0`.  The first `b` indices designate a
batch member.  For every batch member `(n1,...,nb)`, `A[n1,...,nb, : :]` is
a `k x k` matrix.

Since `A` is (batch) positive definite, it has a (or several) square roots `S`
such that `A = SS^T`.

For example,

```python
distributions = tf.contrib.distributions
matrix = [[1.0, 0.5], [1.0, 2.0]]
operator = OperatorPDFull(matrix)
operator.log_det()

# Compute the quadratic form x^T A^{-1} x for vector x.
x = [1.0, 2.0]
operator.inv_quadratic_form(x)

# Matrix multiplication by the square root, S w.
# If w is iid normal, S w has covariance A.
w = [[1.0], [2.0]]
operator.sqrt_matmul(w)
```

The above three methods, `log_det`, `inv_quadratic_form`, and
`sqrt_matmul` provide "all" that is necessary to use a covariance matrix
in a multi-variate normal distribution.  See the class `MVNOperatorPD`.
- - -

#### `tf.contrib.distributions.OperatorPDFull.__init__(matrix, verify_pd=True, name='OperatorPDFull')` {#OperatorPDFull.__init__}

Initialize an OperatorPDFull.

##### Args:


*  <b>`matrix`</b>: Shape `[N1,...,Nb, k, k]` tensor with `b >= 0`, `k >= 1`.  The
    last two dimensions should be `k x k` symmetric positive definite
    matrices.
*  <b>`verify_pd`</b>: Whether to check that `matrix` is symmetric positive definite.
    If `verify_pd` is `False`, correct behavior is not guaranteed.
*  <b>`name`</b>: A name to prepend to all ops created by this class.


- - -

#### `tf.contrib.distributions.OperatorPDFull.batch_shape(name='batch_shape')` {#OperatorPDFull.batch_shape}

Shape of batches associated with this operator.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nb, k, k]`, the `batch_shape` is `[N1,...,Nb]`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDFull.det(name='det')` {#OperatorPDFull.det}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  Determinant for every batch member.


- - -

#### `tf.contrib.distributions.OperatorPDFull.dtype` {#OperatorPDFull.dtype}




- - -

#### `tf.contrib.distributions.OperatorPDFull.get_batch_shape()` {#OperatorPDFull.get_batch_shape}

`TensorShape` with batch shape.


- - -

#### `tf.contrib.distributions.OperatorPDFull.get_shape()` {#OperatorPDFull.get_shape}

`TensorShape` giving static shape.


- - -

#### `tf.contrib.distributions.OperatorPDFull.get_vector_shape()` {#OperatorPDFull.get_vector_shape}

`TensorShape` of vectors this operator will work with.


- - -

#### `tf.contrib.distributions.OperatorPDFull.inputs` {#OperatorPDFull.inputs}

List of tensors that were provided as initialization inputs.


- - -

#### `tf.contrib.distributions.OperatorPDFull.inv_quadratic_form(x, name='inv_quadratic_form')` {#OperatorPDFull.inv_quadratic_form}

Compute the induced vector norm (squared): ||x||^2 := x^T A^{-1} x.

For every batch member, this is done in `O(k^2)` complexity.  The efficiency
depends on the shape of `x`.
* If `x.shape = [M1,...,Mm, N1,...,Nb, k]`, `m >= 0`, and
  `self.shape = [N1,...,Nb, k, k]`, `x` will be reshaped and the
  initialization matrix `chol` does not need to be copied.
* Otherwise, data will be broadcast and copied.

##### Args:


*  <b>`x`</b>: `Tensor` with shape broadcastable to `[N1,...,Nb, k]` and same `dtype`
    as self.  If the batch dimensions of `x` do not match exactly with those
    of self, `x` and/or self's Cholesky factor will broadcast to match, and
    the resultant set of linear systems are solved independently.  This may
    result in inefficient operation.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` holding the square of the norm induced by inverse of `A`.  For
  every broadcast batch member.


- - -

#### `tf.contrib.distributions.OperatorPDFull.log_det(name='log_det')` {#OperatorPDFull.log_det}

Log determinant of every batch member.


- - -

#### `tf.contrib.distributions.OperatorPDFull.matmul(x, name='matmul')` {#OperatorPDFull.matmul}

Left (batch) matrix multiplication of `x` by this operator.


- - -

#### `tf.contrib.distributions.OperatorPDFull.name` {#OperatorPDFull.name}




- - -

#### `tf.contrib.distributions.OperatorPDFull.rank(name='rank')` {#OperatorPDFull.rank}

Tensor rank.  Equivalent to `tf.rank(A)`.  Will equal `b + 2`.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nb, k, k]`, the `rank` is `b + 2`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDFull.shape(name='shape')` {#OperatorPDFull.shape}




- - -

#### `tf.contrib.distributions.OperatorPDFull.sqrt_matmul(x, name='sqrt_matmul')` {#OperatorPDFull.sqrt_matmul}

Left (batch) matmul `x` by a sqrt of this matrix:  `Sx` where `A = S S^T.

##### Args:


*  <b>`x`</b>: `Tensor` with shape broadcastable to `[N1,...,Nb, k]` and same `dtype`
    as self.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  Shape `[N1,...,Nb, k]` `Tensor` holding the product `S x`.


- - -

#### `tf.contrib.distributions.OperatorPDFull.to_dense(name='to_dense')` {#OperatorPDFull.to_dense}

Return a dense (batch) matrix representing this covariance.


- - -

#### `tf.contrib.distributions.OperatorPDFull.to_dense_sqrt(name='to_dense_sqrt')` {#OperatorPDFull.to_dense_sqrt}

Return a dense (batch) matrix representing sqrt of this covariance.


- - -

#### `tf.contrib.distributions.OperatorPDFull.vector_shape(name='vector_shape')` {#OperatorPDFull.vector_shape}

Shape of (batch) vectors that this (batch) matrix will multiply.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nb, k, k]`, the `vector_shape` is `[N1,...,Nb, k]`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDFull.vector_space_dimension(name='vector_space_dimension')` {#OperatorPDFull.vector_space_dimension}

Dimension of vector space on which this acts.  The `k` in `R^k`.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nb, k, k]`, the `vector_space_dimension` is `k`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDFull.verify_pd` {#OperatorPDFull.verify_pd}

Whether to verify that this `Operator` is positive definite.


