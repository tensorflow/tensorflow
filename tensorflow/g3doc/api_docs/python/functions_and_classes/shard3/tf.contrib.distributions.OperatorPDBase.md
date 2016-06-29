Class representing a (batch) of positive definite matrices `A`.

This class provides access to functions of a (batch) symmetric positive
definite (PD) matrix, without the need to materialize them.  In other words,
this provides means to do "matrix free" computations.

For example, `my_operator.matmul(x)` computes the result of matrix
multiplication, and this class is free to do this computation with or without
ever materializing a matrix.

In practice, this operator represents a (batch) matrix `A` with shape
`[N1,...,Nb, k, k]` for some `b >= 0`.  The first `b` indices index a
batch member.  For every batch index `(n1,...,nb)`, `A[n1,...,nb, : :]` is
a `k x k` matrix.  Again, this matrix `A` may not be materialized, but for
purposes of broadcasting this shape will be relevant.

Since `A` is (batch) positive definite, it has a (or several) square roots `S`
such that `A = SS^T`.

For example, if `MyOperator` inherits from `OperatorPDBase`, the user can do

```python
operator = MyOperator(...)  # Initialize with some tensors.
operator.log_det()

# Compute the quadratic form x^T A^{-1} x for vector x.
x = ... # some shape [..., k] tensor
operator.inv_quadratic_form(x)

# Matrix multiplication by the square root, S w.
# If w is iid normal, S w has covariance A.
w = ... # some shape [..., k, L] tensor, L >= 1
operator.sqrt_matmul(w)
```

The above three methods, `log_det`, `inv_quadratic_form`, and
`sqrt_matmul` provide "all" that is necessary to use a covariance matrix
in a multi-variate normal distribution.  See the class `MVNOperatorPD`.
- - -

#### `tf.contrib.distributions.OperatorPDBase.batch_shape(name='batch_shape')` {#OperatorPDBase.batch_shape}

Shape of batches associated with this operator.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nb, k, k]`, the `batch_shape` is `[N1,...,Nb]`.

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

`TensorShape` with batch shape.


- - -

#### `tf.contrib.distributions.OperatorPDBase.get_shape()` {#OperatorPDBase.get_shape}

`TensorShape` giving static shape.


- - -

#### `tf.contrib.distributions.OperatorPDBase.get_vector_shape()` {#OperatorPDBase.get_vector_shape}

`TensorShape` of vectors this operator will work with.


- - -

#### `tf.contrib.distributions.OperatorPDBase.inputs` {#OperatorPDBase.inputs}

List of tensors that were provided as initialization inputs.


- - -

#### `tf.contrib.distributions.OperatorPDBase.inv_quadratic_form(x, name='inv_quadratic_form')` {#OperatorPDBase.inv_quadratic_form}

Compute the quadratic form: x^T A^{-1} x.

##### Args:


*  <b>`x`</b>: `Tensor` with shape broadcastable to `[N1,...,Nb, k]` and same `dtype`
    as self.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `Tensor` holding the square of the norm induced by inverse of `A`.  For
  every broadcast batch member.


- - -

#### `tf.contrib.distributions.OperatorPDBase.log_det(name='log_det')` {#OperatorPDBase.log_det}

Log of the determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  Logarithm of determinant for every batch member.


- - -

#### `tf.contrib.distributions.OperatorPDBase.matmul(x, name='matmul')` {#OperatorPDBase.matmul}

Left multiply `x` by this operator.

##### Args:


*  <b>`x`</b>: Shape `[N1,...,Nb, k, L]` `Tensor` with same `dtype` as this operator
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A result equivalent to `tf.batch_matmul(self.to_dense(), x)`.


- - -

#### `tf.contrib.distributions.OperatorPDBase.name` {#OperatorPDBase.name}

String name identifying this `Operator`.


- - -

#### `tf.contrib.distributions.OperatorPDBase.rank(name='rank')` {#OperatorPDBase.rank}

Tensor rank.  Equivalent to `tf.rank(A)`.  Will equal `b + 2`.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nb, k, k]`, the `rank` is `b + 2`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDBase.shape(name='shape')` {#OperatorPDBase.shape}

Equivalent to `tf.shape(A).`  Equal to `[N1,...,Nb, k, k]`, `b >= 0`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDBase.sqrt_matmul(x, name='sqrt_matmul')` {#OperatorPDBase.sqrt_matmul}

Left (batch) matmul `x` by a sqrt of this matrix:  `Sx` where `A = S S^T.

##### Args:


*  <b>`x`</b>: `Tensor` with shape broadcastable to `[N1,...,Nb, k]` and same `dtype`
    as self.
*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  Shape `[N1,...,Nb, k]` `Tensor` holding the product `S x`.


- - -

#### `tf.contrib.distributions.OperatorPDBase.to_dense(name='to_dense')` {#OperatorPDBase.to_dense}

Return a dense (batch) matrix representing this operator.


- - -

#### `tf.contrib.distributions.OperatorPDBase.to_dense_sqrt(name='to_dense_sqrt')` {#OperatorPDBase.to_dense_sqrt}

Return a dense (batch) matrix representing sqrt of this operator.


- - -

#### `tf.contrib.distributions.OperatorPDBase.vector_shape(name='vector_shape')` {#OperatorPDBase.vector_shape}

Shape of (batch) vectors that this (batch) matrix will multiply.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nb, k, k]`, the `vector_shape` is `[N1,...,Nb, k]`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDBase.vector_space_dimension(name='vector_space_dimension')` {#OperatorPDBase.vector_space_dimension}

Dimension of vector space on which this acts.  The `k` in `R^k`.

If this operator represents the batch matrix `A` with
`A.shape = [N1,...,Nb, k, k]`, the `vector_space_dimension` is `k`.

##### Args:


*  <b>`name`</b>: A name scope to use for ops added by this method.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.distributions.OperatorPDBase.verify_pd` {#OperatorPDBase.verify_pd}

Whether to verify that this `Operator` is positive definite.


