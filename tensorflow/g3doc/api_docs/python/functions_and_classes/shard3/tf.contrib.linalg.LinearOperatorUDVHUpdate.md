Perturb a `LinearOperator` with a rank `K` update.

This operator acts like a [batch] matrix `A` with shape
`[B1,...,Bb, M, N]` for some `b >= 0`.  The first `b` indices index a
batch member.  For every batch index `(i1,...,ib)`, `A[i1,...,ib, : :]` is
an `M x N` matrix.

`LinearOperatorUDVHUpdate` represents `A = L + U D V^H`, where

```
L, is a LinearOperator representing [batch] M x N matrices
U, is a [batch] M x K matrix.  Typically K << M.
D, is a [batch] K x K matrix.
V, is a [batch] N x K matrix.  Typically K << N.
V^H is the Hermitian transpose (adjoint) of V.
```

If `M = N`, determinants and solves are done using the matrix determinant
lemma and Woodbury identities, and thus require L and D to be non-singular.

Solves and determinants will be attempted unless the "is_non_singular"
property of L and D is False.

In the event that L and D are positive-definite, and U = V, solves and
determinants can be done using a Cholesky factorization.

```python
# Create a 3 x 3 diagonal linear operator.
diag_operator = LinearOperatorDiag(
    diag=[1., 2., 3.], is_non_singular=True, is_self_adjoint=True,
    is_positive_definite=True)

# Perturb with a rank 2 perturbation
operator = LinearOperatorUDVHUpdate(
    operator=diag_operator,
    u=[[1., 2.], [-1., 3.], [0., 0.]],
    diag=[11., 12.],
    v=[[1., 2.], [-1., 3.], [10., 10.]])

operator.shape
==> [3, 3]

operator.log_determinant()
==> scalar Tensor

x = ... Shape [3, 4] Tensor
operator.apply(x)
==> Shape [3, 4] Tensor
```

### Shape compatibility

This operator acts on [batch] matrix with compatible shape.
`x` is a batch matrix with compatible shape for `apply` and `solve` if

```
operator.shape = [B1,...,Bb] + [M, N],  with b >= 0
x.shape =        [B1,...,Bb] + [N, R],  with R >= 0.
```

### Performance

Suppose `operator` is a `LinearOperatorUDVHUpdate` of shape `[M, N]`,
made from a rank `K` update of `base_operator` which performs `.apply(x)` on
`x` having `x.shape = [N, R]` with `O(L_apply*N*R)` complexity (and similarly
for `solve`, `determinant`.  Then, if `x.shape = [N, R]`,

* `operator.apply(x)` is `O(L_apply*N*R + K*N*R)`

and if `M = N`,

* `operator.solve(x)` is `O(L_apply*N*R + N*K*R + K^2*R + K^3)`
* `operator.determinant()` is `O(L_determinant + L_solve*N*K + K^2*N + K^3)`

If instead `operator` and `x` have shape `[B1,...,Bb, M, N]` and
`[B1,...,Bb, N, R]`, every operation increases in complexity by `B1*...*Bb`.

#### Matrix property hints

This `LinearOperator` is initialized with boolean flags of the form `is_X`,
for `X = non_singular, self_adjoint, positive_definite, diag_positive, square`
These have the following meaning
* If `is_X == True`, callers should expect the operator to have the
  property `X`.  This is a promise that should be fulfilled, but is *not* a
  runtime assert.  For example, finite floating point precision may result
  in these promises being violated.
* If `is_X == False`, callers should expect the operator to not have `X`.
* If `is_X == None` (the default), callers should have no expectation either
  way.
- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.__init__(base_operator, u, diag=None, v=None, is_diag_positive=None, is_non_singular=None, is_self_adjoint=None, is_positive_definite=None, is_square=None, name='LinearOperatorUDVHUpdate')` {#LinearOperatorUDVHUpdate.__init__}

Initialize a `LinearOperatorUDVHUpdate`.

This creates a `LinearOperator` of the form `A = L + U D V^H`, with
`L` a `LinearOperator`, `U, V` both [batch] matrices, and `D` a [batch]
diagonal matrix.

If `L` is non-singular, solves and determinants are available.
Solves/determinants both involve a solve/determinant of a `K x K` system.
In the event that L and D are self-adjoint positive-definite, and U = V,
this can be done using a Cholesky factorization.  The user should set the
`is_X` matrix property hints, which will trigger the appropriate code path.

##### Args:


*  <b>`base_operator`</b>: Shape `[B1,...,Bb, M, N]` real `float32` or `float64`
    `LinearOperator`.  This is `L` above.
*  <b>`u`</b>: Shape `[B1,...,Bb, M, K]` `Tensor` of same `dtype` as `base_operator`.
    This is `U` above.
*  <b>`diag`</b>: Optional shape `[B1,...,Bb, K]` `Tensor` with same `dtype` as
    `base_operator`.  This is the diagonal of `D` above.
     Defaults to `D` being the identity operator.
*  <b>`v`</b>: Optional `Tensor` of same `dtype` as `u` and shape `[B1,...,Bb, N, K]`
     Defaults to `v = u`, in which case the perturbation is symmetric.
     If `M != N`, then `v` must be set since the pertrubation is not square.
*  <b>`is_diag_positive`</b>: Python `bool`.  If `True`, expect `diag > 0`.
*  <b>`is_non_singular`</b>: Expect that this operator is non-singular.
    Default is `None`, unless `is_positive_definite` is auto-set to be
    `True` (see below).
*  <b>`is_self_adjoint`</b>: Expect that this operator is equal to its hermitian
    transpose.  Default is `None`, unless `base_operator` is self-adjoint
    and `v = None` (meaning `u=v`), in which case this defaults to `True`.
*  <b>`is_positive_definite`</b>: Expect that this operator is positive definite.
    Default is `None`, unless `base_operator` is positive-definite
    `v = None` (meaning `u=v`), and `is_diag_positive`, in which case this
    defaults to `True`.
*  <b>`is_square`</b>: Expect that this operator acts like square [batch] matrices.
*  <b>`name`</b>: A name for this `LinearOperator`.

##### Raises:


*  <b>`ValueError`</b>: If `is_X` flags are set in an inconsistent way.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.add_to_tensor(x, name='add_to_tensor')` {#LinearOperatorUDVHUpdate.add_to_tensor}

Add matrix represented by this operator to `x`.  Equivalent to `A + x`.

##### Args:


*  <b>`x`</b>: `Tensor` with same `dtype` and shape broadcastable to `self.shape`.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with broadcast shape and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.apply(x, adjoint=False, name='apply')` {#LinearOperatorUDVHUpdate.apply}

Transform `x` with left multiplication:  `x --> Ax`.

##### Args:


*  <b>`x`</b>: `Tensor` with compatible shape and same `dtype` as `self`.
    See class docstring for definition of compatibility.
*  <b>`adjoint`</b>: Python `bool`.  If `True`, left multiply by the adjoint.
*  <b>`name`</b>: A name for this `Op.

##### Returns:

  A `Tensor` with shape `[..., M, R]` and same `dtype` as `self`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.assert_non_singular(name='assert_non_singular')` {#LinearOperatorUDVHUpdate.assert_non_singular}

Returns an `Op` that asserts this operator is non singular.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.assert_positive_definite(name='assert_positive_definite')` {#LinearOperatorUDVHUpdate.assert_positive_definite}

Returns an `Op` that asserts this operator is positive definite.

Here, positive definite means the real part of all eigenvalues is positive.
We do not require the operator to be self-adjoint.

##### Args:


*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  An `Op` that asserts this operator is positive definite.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.assert_self_adjoint(name='assert_self_adjoint')` {#LinearOperatorUDVHUpdate.assert_self_adjoint}

Returns an `Op` that asserts this operator is self-adjoint.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.base_operator` {#LinearOperatorUDVHUpdate.base_operator}

If this operator is `A = L + U D V^H`, this is the `L`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.batch_shape` {#LinearOperatorUDVHUpdate.batch_shape}

`TensorShape` of batch dimensions of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb])`, equivalent to `A.get_shape()[:-2]`

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.batch_shape_tensor(name='batch_shape_tensor')` {#LinearOperatorUDVHUpdate.batch_shape_tensor}

Shape of batch dimensions of this operator, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb]`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.determinant(name='det')` {#LinearOperatorUDVHUpdate.determinant}

Determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

##### Raises:


*  <b>`NotImplementedError`</b>: If `self.is_square` is `False`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.diag` {#LinearOperatorUDVHUpdate.diag}

If this operator is `A = L + U D V^H`, this is the diagonal of `D`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.diag_operator` {#LinearOperatorUDVHUpdate.diag_operator}

If this operator is `A = L + U D V^H`, this is `D`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.domain_dimension` {#LinearOperatorUDVHUpdate.domain_dimension}

Dimension (in the sense of vector spaces) of the domain of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.domain_dimension_tensor(name='domain_dimension_tensor')` {#LinearOperatorUDVHUpdate.domain_dimension_tensor}

Dimension (in the sense of vector spaces) of the domain of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.dtype` {#LinearOperatorUDVHUpdate.dtype}

The `DType` of `Tensor`s handled by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.graph_parents` {#LinearOperatorUDVHUpdate.graph_parents}

List of graph dependencies of this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.is_non_singular` {#LinearOperatorUDVHUpdate.is_non_singular}




- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.is_positive_definite` {#LinearOperatorUDVHUpdate.is_positive_definite}




- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.is_self_adjoint` {#LinearOperatorUDVHUpdate.is_self_adjoint}




- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.is_square` {#LinearOperatorUDVHUpdate.is_square}

Return `True/False` depending on if this operator is square.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.log_abs_determinant(name='log_abs_det')` {#LinearOperatorUDVHUpdate.log_abs_determinant}

Log absolute value of determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

##### Raises:


*  <b>`NotImplementedError`</b>: If `self.is_square` is `False`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.name` {#LinearOperatorUDVHUpdate.name}

Name prepended to all ops created by this `LinearOperator`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.range_dimension` {#LinearOperatorUDVHUpdate.range_dimension}

Dimension (in the sense of vector spaces) of the range of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.range_dimension_tensor(name='range_dimension_tensor')` {#LinearOperatorUDVHUpdate.range_dimension_tensor}

Dimension (in the sense of vector spaces) of the range of this operator.

Determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `M`.

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.shape` {#LinearOperatorUDVHUpdate.shape}

`TensorShape` of this `LinearOperator`.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns
`TensorShape([B1,...,Bb, M, N])`, equivalent to `A.get_shape()`.

##### Returns:

  `TensorShape`, statically determined, may be undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.shape_tensor(name='shape_tensor')` {#LinearOperatorUDVHUpdate.shape_tensor}

Shape of this `LinearOperator`, determined at runtime.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns a `Tensor` holding
`[B1,...,Bb, M, N]`, equivalent to `tf.shape(A)`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.solve(rhs, adjoint=False, name='solve')` {#LinearOperatorUDVHUpdate.solve}

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


*  <b>`NotImplementedError`</b>: If `self.is_non_singular` or `is_square` is False.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.tensor_rank` {#LinearOperatorUDVHUpdate.tensor_rank}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  Python integer, or None if the tensor rank is undefined.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.tensor_rank_tensor(name='tensor_rank_tensor')` {#LinearOperatorUDVHUpdate.tensor_rank_tensor}

Rank (in the sense of tensors) of matrix corresponding to this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `b + 2`.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `int32` `Tensor`, determined at runtime.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.to_dense(name='to_dense')` {#LinearOperatorUDVHUpdate.to_dense}

Return a dense (batch) matrix representing this operator.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.u` {#LinearOperatorUDVHUpdate.u}

If this operator is `A = L + U D V^H`, this is the `U`.


- - -

#### `tf.contrib.linalg.LinearOperatorUDVHUpdate.v` {#LinearOperatorUDVHUpdate.v}

If this operator is `A = L + U D V^H`, this is the `V`.


