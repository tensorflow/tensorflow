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

#### `tf.contrib.linalg.LinearOperatorComposition.batch_shape_tensor(name='batch_shape_tensor')` {#LinearOperatorComposition.batch_shape_tensor}

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

##### Raises:


*  <b>`NotImplementedError`</b>: If `self.is_square` is `False`.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.diag_part(name='diag_part')` {#LinearOperatorComposition.diag_part}

Efficiently get the [batch] diagonal part of this operator.

If this operator has shape `[B1,...,Bb, M, N]`, this returns a
`Tensor` `diagonal`, of shape `[B1,...,Bb, min(M, N)]`, where
`diagonal[b1,...,bb, i] = self.to_dense()[b1,...,bb, i, i]`.

```
my_operator = LinearOperatorDiag([1., 2.])

# Efficiently get the diagonal
my_operator.diag_part()
==> [1., 2.]

# Equivalent, but inefficient method
tf.matrix_diag_part(my_operator.to_dense())
==> [1., 2.]
```

##### Args:


*  <b>`name`</b>: A name for this `Op`.

##### Returns:


*  <b>`diag_part`</b>: A `Tensor` of same `dtype` as self.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.domain_dimension` {#LinearOperatorComposition.domain_dimension}

Dimension (in the sense of vector spaces) of the domain of this operator.

If this operator acts like the batch matrix `A` with
`A.shape = [B1,...,Bb, M, N]`, then this returns `N`.

##### Returns:

  `Dimension` object.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.domain_dimension_tensor(name='domain_dimension_tensor')` {#LinearOperatorComposition.domain_dimension_tensor}

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

#### `tf.contrib.linalg.LinearOperatorComposition.is_square` {#LinearOperatorComposition.is_square}

Return `True/False` depending on if this operator is square.


- - -

#### `tf.contrib.linalg.LinearOperatorComposition.log_abs_determinant(name='log_abs_det')` {#LinearOperatorComposition.log_abs_determinant}

Log absolute value of determinant for every batch member.

##### Args:


*  <b>`name`</b>: A name for this `Op.

##### Returns:

  `Tensor` with shape `self.batch_shape` and same `dtype` as `self`.

##### Raises:


*  <b>`NotImplementedError`</b>: If `self.is_square` is `False`.


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

#### `tf.contrib.linalg.LinearOperatorComposition.range_dimension_tensor(name='range_dimension_tensor')` {#LinearOperatorComposition.range_dimension_tensor}

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

#### `tf.contrib.linalg.LinearOperatorComposition.shape_tensor(name='shape_tensor')` {#LinearOperatorComposition.shape_tensor}

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


*  <b>`NotImplementedError`</b>: If `self.is_non_singular` or `is_square` is False.


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

#### `tf.contrib.linalg.LinearOperatorComposition.tensor_rank_tensor(name='tensor_rank_tensor')` {#LinearOperatorComposition.tensor_rank_tensor}

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


