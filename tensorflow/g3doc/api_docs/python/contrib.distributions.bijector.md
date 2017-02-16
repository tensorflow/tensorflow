<!-- This file is machine generated: DO NOT EDIT! -->

# Random variable transformations (contrib)
[TOC]

Bijector Ops. See the @{$python/contrib.distributions.bijector} guide.

- - -

### `class tf.contrib.distributions.bijector.Affine` {#Affine}

Compute `Y = g(X; shift, scale) = scale @ X + shift`.

Here `scale = c * I + diag(D1) + tril(L) + V @ diag(D2) @ V.T`.

In TF parlance, the `scale` term is logically equivalent to:

```python
scale = (
  scale_identity_multiplier * tf.diag(tf.ones(d)) +
  tf.diag(scale_diag) +
  scale_tril +
  scale_perturb_factor @ diag(scale_perturb_diag) @
    tf.transpose([scale_perturb_factor])
)
```

The `scale` term is applied without necessarily materializing constituent
matrices, i.e., the matmul is [matrix-free](
https://en.wikipedia.org/wiki/Matrix-free_methods) when possible.

Examples:

```python
# Y = X
b = Affine()

# Y = X + shift
b = Affine(shift=[1., 2, 3])

# Y = 2 * I @ X.T + shift
b = Affine(shift=[1., 2, 3],
           scale_identity_multiplier=2.)

# Y = tf.diag(d1) @ X.T + shift
b = Affine(shift=[1., 2, 3],
           scale_diag=[-1., 2, 1])         # Implicitly 3x3.

# Y = (I + v * v.T) @ X.T + shift
b = Affine(shift=[1., 2, 3],
           scale_perturb_factor=[[1., 0],
                                 [0, 1],
                                 [1, 1]])

# Y = (diag(d1) + v * diag(d2) * v.T) @ X.T + shift
b = Affine(shift=[1., 2, 3],
           scale_diag=[1., 3, 3],          # Implicitly 3x3.
           scale_perturb_diag=[2., 1],     # Implicitly 2x2.
           scale_perturb_factor=[[1., 0],
                                 [0, 1],
                                 [1, 1]])

```
- - -

#### `tf.contrib.distributions.bijector.Affine.__init__(shift=None, scale_identity_multiplier=None, scale_diag=None, scale_tril=None, scale_perturb_factor=None, scale_perturb_diag=None, event_ndims=1, validate_args=False, name='affine')` {#Affine.__init__}

Instantiates the `Affine` bijector.

This `Bijector` is initialized with `shift` `Tensor` and `scale` arguments,
giving the forward operation:

```none
Y = g(X) = scale @ X + shift
```

where the `scale` term is logically equivalent to:

```python
scale = (
  scale_identity_multiplier * tf.diag(tf.ones(d)) +
  tf.diag(scale_diag) +
  scale_tril +
  scale_perturb_factor @ diag(scale_perturb_diag) @
    tf.transpose([scale_perturb_factor])
)
```

If none of `scale_identity_multiplier`, `scale_diag`, or `scale_tril` are
specified then `scale += IdentityMatrix`. Otherwise specifying a
`scale` argument has the semantics of `scale += Expand(arg)`, i.e.,
`scale_diag != None` means `scale += tf.diag(scale_diag)`.

##### Args:


*  <b>`shift`</b>: Floating-point `Tensor`. If this is set to `None`, no shift is
    applied.
*  <b>`scale_identity_multiplier`</b>: floating point rank 0 `Tensor` representing a
    scaling done to the identity matrix.
    When `scale_identity_multiplier = scale_diag = scale_tril = None` then
    `scale += IdentityMatrix`. Otherwise no scaled-identity-matrix is added
    to `scale`.
*  <b>`scale_diag`</b>: Floating-point `Tensor` representing the diagonal matrix.
    `scale_diag` has shape [N1, N2, ...  k], which represents a k x k
    diagonal matrix.
    When `None` no diagonal term is added to `scale`.
*  <b>`scale_tril`</b>: Floating-point `Tensor` representing the diagonal matrix.
    `scale_diag` has shape [N1, N2, ...  k, k], which represents a k x k
    lower triangular matrix.
    When `None` no `scale_tril` term is added to `scale`.
    The upper triangular elements above the diagonal are ignored.
*  <b>`scale_perturb_factor`</b>: Floating-point `Tensor` representing factor matrix
    with last two dimensions of shape `(k, r)`. When `None`, no rank-r
    update is added to `scale`.
*  <b>`scale_perturb_diag`</b>: Floating-point `Tensor` representing the diagonal
    matrix. `scale_perturb_diag` has shape [N1, N2, ...  r], which
    represents an `r x r` diagonal matrix. When `None` low rank updates will
    take the form `scale_perturb_factor * scale_perturb_factor.T`.
*  <b>`event_ndims`</b>: Scalar `int32` `Tensor` indicating the number of dimensions
    associated with a particular draw from the distribution. Must be 0 or 1.
*  <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
    checked for correctness.
*  <b>`name`</b>: Python `str` name given to ops managed by this object.

##### Raises:


*  <b>`ValueError`</b>: if `perturb_diag` is specified but not `perturb_factor`.
*  <b>`TypeError`</b>: if `shift` has different `dtype` from `scale` arguments.


- - -

#### `tf.contrib.distributions.bijector.Affine.dtype` {#Affine.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Affine.event_ndims` {#Affine.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.Affine.forward(x, name='forward')` {#Affine.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.Affine.forward_event_shape(input_shape)` {#Affine.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Affine.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#Affine.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Affine.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#Affine.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Affine.graph_parents` {#Affine.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Affine.inverse(y, name='inverse')` {#Affine.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Affine.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#Affine.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Affine.inverse_event_shape(output_shape)` {#Affine.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Affine.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#Affine.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Affine.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#Affine.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Affine.is_constant_jacobian` {#Affine.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.Affine.name` {#Affine.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Affine.scale` {#Affine.scale}

The `scale` `LinearOperator` in `Y = scale @ X + shift`.


- - -

#### `tf.contrib.distributions.bijector.Affine.shift` {#Affine.shift}

The `shift` `Tensor` in `Y = scale @ X + shift`.


- - -

#### `tf.contrib.distributions.bijector.Affine.validate_args` {#Affine.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.AffineLinearOperator` {#AffineLinearOperator}

Compute `Y = g(X; shift, scale) = scale @ X + shift`.

`shift` is a numeric `Tensor` and `scale` is a `LinearOperator`.

If `X` is a scalar then the forward transformation is: `scale * X + shift`
where `*` denotes the scalar product.

Note: we don't always simply transpose `X` (but write it this way for
brevity). Actually the input `X` undergoes the following transformation
before being premultiplied by `scale`:

1. If there are no sample dims, we call `X = tf.expand_dims(X, 0)`, i.e.,
   `new_sample_shape = [1]`. Otherwise do nothing.
2. The sample shape is flattened to have one dimension, i.e.,
   `new_sample_shape = [n]` where `n = tf.reduce_prod(old_sample_shape)`.
3. The sample dim is cyclically rotated left by 1, i.e.,
   `new_shape = [B1,...,Bb, k, n]` where `n` is as above, `k` is the
   event_shape, and `B1,...,Bb` are the batch shapes for each of `b` batch
   dimensions.

(For more details see `shape.make_batch_of_event_sample_matrices`.)

The result of the above transformation is that `X` can be regarded as a batch
of matrices where each column is a draw from the distribution. After
premultiplying by `scale`, we take the inverse of this procedure. The input
`Y` also undergoes the same transformation before/after premultiplying by
`inv(scale)`.

Example Use:

```python
linalg = tf.contrib.linalg

x = [1., 2, 3]

shift = [-1., 0., 1]
diag = [1., 2, 3]
scale = linalg.LinearOperatorDiag(diag)
affine = AffineLinearOperator(shift, scale)
# In this case, `forward` is equivalent to:
# y = scale @ x + shift
y = affine.forward(x)  # [0., 4, 10]

shift = [2., 3, 1]
tril = [[1., 0, 0],
        [2, 1, 0],
        [3, 2, 1]]
scale = linalg.LinearOperatorTriL(tril)
affine = AffineLinearOperator(shift, scale)
# In this case, `forward` is equivalent to:
# np.squeeze(np.matmul(tril, np.expand_dims(x, -1)), -1) + shift
y = affine.forward(x)  # [3., 7, 11]
```
- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.__init__(shift=None, scale=None, event_ndims=1, validate_args=False, name='affine_linear_operator')` {#AffineLinearOperator.__init__}

Instantiates the `AffineLinearOperator` bijector.

##### Args:


*  <b>`shift`</b>: Floating-point `Tensor`.
*  <b>`scale`</b>: Subclass of `LinearOperator`. Represents the (batch) positive
    definite matrix `M` in `R^{k x k}`.
*  <b>`event_ndims`</b>: Scalar `integer` `Tensor` indicating the number of dimensions
    associated with a particular draw from the distribution. Must be 0 or 1.
*  <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
    checked for correctness.
*  <b>`name`</b>: Python `str` name given to ops managed by this object.

##### Raises:


*  <b>`ValueError`</b>: if `event_ndims` is not 0 or 1.
*  <b>`TypeError`</b>: if `scale` is not a `LinearOperator`.
*  <b>`TypeError`</b>: if `shift.dtype` does not match `scale.dtype`.
*  <b>`ValueError`</b>: if not `scale.is_non_singular`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.dtype` {#AffineLinearOperator.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.event_ndims` {#AffineLinearOperator.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.forward(x, name='forward')` {#AffineLinearOperator.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.forward_event_shape(input_shape)` {#AffineLinearOperator.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#AffineLinearOperator.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#AffineLinearOperator.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.graph_parents` {#AffineLinearOperator.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.inverse(y, name='inverse')` {#AffineLinearOperator.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#AffineLinearOperator.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.inverse_event_shape(output_shape)` {#AffineLinearOperator.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#AffineLinearOperator.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#AffineLinearOperator.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.is_constant_jacobian` {#AffineLinearOperator.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.name` {#AffineLinearOperator.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.scale` {#AffineLinearOperator.scale}

The `scale` `LinearOperator` in `Y = scale @ X + shift`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.shift` {#AffineLinearOperator.shift}

The `shift` `Tensor` in `Y = scale @ X + shift`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.validate_args` {#AffineLinearOperator.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.Bijector` {#Bijector}

Interface for transforming a `Distribution` sample.

A `Bijector` implements a
[diffeomorphism](https://en.wikipedia.org/wiki/Diffeomorphism), i.e., a
bijective, differentiable function. A `Bijector` is used by
`TransformedDistribution` but can be generally used for transforming a
`Distribution` generated `Tensor`. A `Bijector` is characterized by three
operations:

1. Forward Evaluation

   Useful for turning one random outcome into another random outcome from a
   different distribution.

2. Inverse Evaluation

   Useful for "reversing" a transformation to compute one probability in
   terms of another.

3. (log o det o Jacobian o inverse)(x)

   "The log of the determinant of the matrix of all first-order partial
   derivatives of the inverse function."
   Useful for inverting a transformation to compute one probability in terms
   of another. Geometrically, the det(Jacobian) is the volume of the
   transformation and is used to scale the probability.

By convention, transformations of random variables are named in terms of the
forward transformation. The forward transformation creates samples, the
inverse is useful for computing probabilities.

Example Use:

  - Basic properties:

  ```python
  x = ...  # A tensor.
  # Evaluate forward transformation.
  fwd_x = my_bijector.forward(x)
  x == my_bijector.inverse(fwd_x)
  x != my_bijector.forward(fwd_x)  # Not equal because g(x) != g(g(x)).
  ```

  - Computing a log-likelihood:

  ```python
  def transformed_log_prob(bijector, log_prob, x):
    return (bijector.inverse_log_det_jacobian(x) +
            log_prob(bijector.inverse(x)))
  ```

  - Transforming a random outcome:

  ```python
  def transformed_sample(bijector, x):
    return bijector.forward(x)
  ```

Example transformations:

  - "Exponential"

    ```
    Y = g(X) = exp(X)
    X ~ Normal(0, 1)  # Univariate.
    ```

    Implies:

    ```
      g^{-1}(Y) = log(Y)
      |Jacobian(g^{-1})(y)| = 1 / y
      Y ~ LogNormal(0, 1), i.e.,
      prob(Y=y) = |Jacobian(g^{-1})(y)| * prob(X=g^{-1}(y))
                = (1 / y) Normal(log(y); 0, 1)
    ```

    Here is an example of how one might implement the `Exp` bijector:

    ```
      class Exp(Bijector):
        def __init__(self, event_ndims=0, validate_args=False, name="exp"):
          super(Exp, self).__init__(event_ndims=event_ndims,
                                    validate_args=validate_args, name=name)
        def _forward(self, x):
          return math_ops.exp(x)
        def _inverse_and_inverse_log_det_jacobian(self, y):
          x = math_ops.log(y)
          return x, -self._forward_log_det_jacobian(x)
        def _forward_log_det_jacobian(self, x):
          if self.event_ndims is None:
            raise ValueError("Jacobian requires known event_ndims.")
          event_dims = array_ops.shape(x)[-self.event_ndims:]
          return math_ops.reduce_sum(x, axis=event_dims)
      ```

  - "Affine"

    ```
    Y = g(X) = sqrtSigma * X + mu
    X ~ MultivariateNormal(0, I_d)
    ```

    Implies:

    ```
      g^{-1}(Y) = inv(sqrtSigma) * (Y - mu)
      |Jacobian(g^{-1})(y)| = det(inv(sqrtSigma))
      Y ~ MultivariateNormal(mu, sqrtSigma) , i.e.,
      prob(Y=y) = |Jacobian(g^{-1})(y)| * prob(X=g^{-1}(y))
                = det(sqrtSigma)^(-d) *
                  MultivariateNormal(inv(sqrtSigma) * (y - mu); 0, I_d)
    ```

Example of why a `Bijector` needs to understand sample, batch, event
partitioning:

- Consider the `Exp` `Bijector` applied to a `Tensor` which has sample, batch,
  and event (S, B, E) shape semantics. Suppose the `Tensor`'s
  partitioned-shape is `(S=[4], B=[2], E=[3, 3])`.

  For `Exp`, the shape of the `Tensor` returned by `forward` and `inverse` is
  unchanged, i.e., `[4, 2, 3, 3]`. However the shape returned by
  `inverse_log_det_jacobian` is `[4, 2]` because the Jacobian is a reduction
  over the event dimensions.

Subclass Requirements:

- Typically subclasses implement `_forward` and one or both of:
    - `_inverse`, `_inverse_log_det_jacobian`,
    - `_inverse_and_inverse_log_det_jacobian`.

- If the `Bijector`'s use is limited to `TransformedDistribution` (or friends
  like `QuantizedDistribution`) then depending on your use, you may not need
  to implement all of `_forward` and `_inverse` functions. Examples:
    1. Sampling (e.g., `sample`) only requires `_forward`.
    2. Probability functions (e.g., `prob`, `cdf`, `survival`) only require
       `_inverse` (and related).
    3. Only calling probability functions on the output of `sample` means
      `_inverse` can be implemented as a cache lookup.

  See `Example Use` [above] which shows how these functions are used to
  transform a distribution. (Note: `_forward` could theoretically be
  implemented as a cache lookup but this would require controlling the
  underlying sample generation mechanism.)

- If computation can be shared among `_inverse` and
  `_inverse_log_det_jacobian` it is preferable to implement
  `_inverse_and_inverse_log_det_jacobian`. This usually reduces
  graph-construction overhead because a `Distribution`'s implementation of
  `log_prob` will need to evaluate both the inverse Jacobian as well as the
  inverse function.

- If an additional use case needs just `inverse` or just
  `inverse_log_det_jacobian` then he or she may also wish to implement these
  functions to avoid computing the `inverse_log_det_jacobian` or the
  `inverse`, respectively.

- Subclasses should implement `_forward_event_shape`,
  `_forward_event_shape_tensor` (and `inverse` counterparts) if the
  transformation is shape-changing. By default the event-shape is assumed
  unchanged from input.

Tips for implementing `_inverse` and `_inverse_log_det_jacobian`:

- As case 3 [above] indicates, under some circumstances the inverse function
  can be implemented as a cache lookup.

- The inverse `log o det o Jacobian` can be implemented as the negative of the
  forward `log o det o Jacobian`. This is useful if the `inverse` is
  implemented as a cache or the inverse Jacobian is computationally more
  expensive (e.g., `CholeskyOuterProduct` `Bijector`). The following
  demonstrates the suggested implementation.

  ```python
  def _inverse_and_log_det_jacobian(self, y):
     x = ...  # implement inverse, possibly via cache.
     return x, -self._forward_log_det_jac(x)  # Note negation.
  ```

  By overriding the `_inverse_and_log_det_jacobian` function we have access to
  the inverse in one call.

  The correctness of this approach can be seen from the following claim.

  - Claim:

      Assume `Y=g(X)` is a bijection whose derivative exists and is nonzero
      for its domain, i.e., `d/dX g(X)!=0`. Then:

      ```none
      (log o det o jacobian o g^{-1})(Y) = -(log o det o jacobian o g)(X)
      ```

  - Proof:

      From the bijective, nonzero differentiability of `g`, the
      [inverse function theorem](
          https://en.wikipedia.org/wiki/Inverse_function_theorem)
      implies `g^{-1}` is differentiable in the image of `g`.
      Applying the chain rule to `y = g(x) = g(g^{-1}(y))` yields
      `I = g'(g^{-1}(y))*g^{-1}'(y)`.
      The same theorem also implies `g{-1}'` is non-singular therefore:
      `inv[ g'(g^{-1}(y)) ] = g^{-1}'(y)`.
      The claim follows from [properties of determinant](
https://en.wikipedia.org/wiki/Determinant#Multiplicativity_and_matrix_groups).

- If possible, prefer a direct implementation of the inverse Jacobian. This
  should have superior numerical stability and will often share subgraphs with
  the `_inverse` implementation.
- - -

#### `tf.contrib.distributions.bijector.Bijector.__init__(event_ndims=None, graph_parents=None, is_constant_jacobian=False, validate_args=False, dtype=None, name=None)` {#Bijector.__init__}

Constructs Bijector.

A `Bijector` transforms random variables into new random variables.

Examples:

```python
# Create the Y = g(X) = X transform which operates on vector events.
identity = Identity(event_ndims=1)

# Create the Y = g(X) = exp(X) transform which operates on matrices.
exp = Exp(event_ndims=2)
```

See `Bijector` subclass docstring for more details and specific examples.

##### Args:


*  <b>`event_ndims`</b>: number of dimensions associated with event coordinates.
*  <b>`graph_parents`</b>: Python list of graph prerequisites of this `Bijector`.
*  <b>`is_constant_jacobian`</b>: Python `bool` indicating that the Jacobian is not a
    function of the input.
*  <b>`validate_args`</b>: Python `bool`, default `False`. Whether to validate input
    with asserts. If `validate_args` is `False`, and the inputs are invalid,
    correct behavior is not guaranteed.
*  <b>`dtype`</b>: `tf.dtype` supported by this `Bijector`. `None` means dtype is not
    enforced.
*  <b>`name`</b>: The name to give Ops created by the initializer.


- - -

#### `tf.contrib.distributions.bijector.Bijector.dtype` {#Bijector.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Bijector.event_ndims` {#Bijector.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.Bijector.forward(x, name='forward')` {#Bijector.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.Bijector.forward_event_shape(input_shape)` {#Bijector.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Bijector.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#Bijector.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Bijector.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#Bijector.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Bijector.graph_parents` {#Bijector.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Bijector.inverse(y, name='inverse')` {#Bijector.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Bijector.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#Bijector.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Bijector.inverse_event_shape(output_shape)` {#Bijector.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Bijector.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#Bijector.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Bijector.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#Bijector.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Bijector.is_constant_jacobian` {#Bijector.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.Bijector.name` {#Bijector.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Bijector.validate_args` {#Bijector.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.Chain` {#Chain}

Bijector which applies a sequence of bijectors.

Example Use:

```python
chain = Chain([Exp(), Softplus()], name="one_plus_exp")
```

Results in:

* Forward:

 ```python
 exp = Exp()
 softplus = Softplus()
 Chain([exp, softplus]).forward(x)
 = exp.forward(softplus.forward(x))
 = tf.exp(tf.log(1. + tf.exp(x)))
 = 1. + tf.exp(x)
 ```

* Inverse:

 ```python
 exp = Exp()
 softplus = Softplus()
 Chain([exp, softplus]).inverse(y)
 = softplus.inverse(exp.inverse(y))
 = tf.log(tf.exp(tf.log(y)) - 1.)
 = tf.log(y - 1.)
 ```
- - -

#### `tf.contrib.distributions.bijector.Chain.__init__(bijectors=(), validate_args=False, name=None)` {#Chain.__init__}

Instantiates `Chain` bijector.

##### Args:


*  <b>`bijectors`</b>: Python list of bijector instances. An empty list makes this
    bijector equivalent to the `Identity` bijector.
*  <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
    checked for correctness.
*  <b>`name`</b>: Python `str`, name given to ops managed by this object. Default:
    E.g., `Chain([Exp(), Softplus()]).name == "chain_of_exp_of_softplus"`.

##### Raises:


*  <b>`ValueError`</b>: if bijectors have different dtypes.


- - -

#### `tf.contrib.distributions.bijector.Chain.bijectors` {#Chain.bijectors}




- - -

#### `tf.contrib.distributions.bijector.Chain.dtype` {#Chain.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Chain.event_ndims` {#Chain.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.Chain.forward(x, name='forward')` {#Chain.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.Chain.forward_event_shape(input_shape)` {#Chain.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Chain.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#Chain.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Chain.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#Chain.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Chain.graph_parents` {#Chain.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Chain.inverse(y, name='inverse')` {#Chain.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Chain.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#Chain.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Chain.inverse_event_shape(output_shape)` {#Chain.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Chain.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#Chain.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Chain.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#Chain.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Chain.is_constant_jacobian` {#Chain.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.Chain.name` {#Chain.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Chain.validate_args` {#Chain.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.CholeskyOuterProduct` {#CholeskyOuterProduct}

Compute `g(X) = X @ X.T`; X is lower-triangular, positive-diagonal matrix.

`event_ndims` must be 0 or 2, i.e., scalar or matrix.

Note: the upper-triangular part of X is ignored (whether or not its zero).

Examples:

```python
bijector.CholeskyOuterProduct(event_ndims=2).forward(x=[[1., 0], [2, 1]])
# Result: [[1., 2], [2, 5]], i.e., x @ x.T

bijector.CholeskyOuterProduct(event_ndims=2).inverse(y=[[1., 2], [2, 5]])
# Result: [[1., 0], [2, 1]], i.e., cholesky(y).
```
- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.__init__(event_ndims=2, validate_args=False, name='cholesky_outer_product')` {#CholeskyOuterProduct.__init__}

Instantiates the `CholeskyOuterProduct` bijector.

##### Args:


*  <b>`event_ndims`</b>: `constant` `int32` scalar `Tensor` indicating the number of
    dimensions associated with a particular draw from the distribution. Must
    be 0 or 2.
*  <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
    checked for correctness.
*  <b>`name`</b>: Python `str` name given to ops managed by this object.

##### Raises:


*  <b>`ValueError`</b>: if event_ndims is neither 0 or 2.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.dtype` {#CholeskyOuterProduct.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.event_ndims` {#CholeskyOuterProduct.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.forward(x, name='forward')` {#CholeskyOuterProduct.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.forward_event_shape(input_shape)` {#CholeskyOuterProduct.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#CholeskyOuterProduct.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#CholeskyOuterProduct.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.graph_parents` {#CholeskyOuterProduct.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse(y, name='inverse')` {#CholeskyOuterProduct.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#CholeskyOuterProduct.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse_event_shape(output_shape)` {#CholeskyOuterProduct.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#CholeskyOuterProduct.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#CholeskyOuterProduct.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.is_constant_jacobian` {#CholeskyOuterProduct.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.name` {#CholeskyOuterProduct.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.validate_args` {#CholeskyOuterProduct.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.Exp` {#Exp}

Compute `Y = g(X) = exp(X)`.

Example Use:

```python
# Create the Y=g(X)=exp(X) transform which works only on Tensors with 1
# batch ndim and 2 event ndims (i.e., vector of matrices).
exp = Exp(event_ndims=2)
x = [[[1., 2],
       [3, 4]],
      [[5, 6],
       [7, 8]]]
exp(x) == exp.forward(x)
log(x) == exp.inverse(x)
```

Note: the exp(.) is applied element-wise but the Jacobian is a reduction
over the event space.
- - -

#### `tf.contrib.distributions.bijector.Exp.__init__(event_ndims=0, validate_args=False, name='exp')` {#Exp.__init__}

Instantiates the `Exp` bijector.

##### Args:


*  <b>`event_ndims`</b>: Scalar `int32` `Tensor` indicating the number of dimensions
    associated with a particular draw from the distribution.
*  <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
    checked for correctness.
*  <b>`name`</b>: Python `str` name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.Exp.dtype` {#Exp.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Exp.event_ndims` {#Exp.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.Exp.forward(x, name='forward')` {#Exp.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.Exp.forward_event_shape(input_shape)` {#Exp.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Exp.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#Exp.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Exp.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#Exp.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Exp.graph_parents` {#Exp.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Exp.inverse(y, name='inverse')` {#Exp.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Exp.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#Exp.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Exp.inverse_event_shape(output_shape)` {#Exp.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Exp.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#Exp.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Exp.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#Exp.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Exp.is_constant_jacobian` {#Exp.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.Exp.name` {#Exp.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Exp.power` {#Exp.power}

The `c` in: `Y = g(X) = (1 + X * c)**(1 / c)`.


- - -

#### `tf.contrib.distributions.bijector.Exp.validate_args` {#Exp.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.Identity` {#Identity}

Compute Y = g(X) = X.

Example Use:

```python
# Create the Y=g(X)=X transform which is intended for Tensors with 1 batch
# ndim and 1 event ndim (i.e., vector of vectors).
identity = Identity(event_ndims=1)
x = [[1., 2],
     [3, 4]]
x == identity.forward(x) == identity.inverse(x)
```
- - -

#### `tf.contrib.distributions.bijector.Identity.__init__(validate_args=False, event_ndims=0, name='identity')` {#Identity.__init__}




- - -

#### `tf.contrib.distributions.bijector.Identity.dtype` {#Identity.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Identity.event_ndims` {#Identity.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.Identity.forward(x, name='forward')` {#Identity.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.Identity.forward_event_shape(input_shape)` {#Identity.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Identity.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#Identity.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Identity.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#Identity.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Identity.graph_parents` {#Identity.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Identity.inverse(y, name='inverse')` {#Identity.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Identity.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#Identity.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Identity.inverse_event_shape(output_shape)` {#Identity.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Identity.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#Identity.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Identity.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#Identity.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Identity.is_constant_jacobian` {#Identity.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.Identity.name` {#Identity.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Identity.validate_args` {#Identity.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.Inline` {#Inline}

Bijector constructed from custom callables.

Example Use:

```python
exp = Inline(
  forward_fn=tf.exp,
  inverse_fn=tf.log,
  inverse_log_det_jacobian_fn=(
    lambda y: -tf.reduce_sum(tf.log(y), axis=-1)),
  name="exp")
```

The above example is equivalent to the `Bijector` `Exp(event_ndims=1)`.
- - -

#### `tf.contrib.distributions.bijector.Inline.__init__(forward_fn=None, inverse_fn=None, inverse_log_det_jacobian_fn=None, forward_log_det_jacobian_fn=None, forward_event_shape_fn=None, forward_event_shape_tensor_fn=None, inverse_event_shape_fn=None, inverse_event_shape_tensor_fn=None, is_constant_jacobian=False, validate_args=False, name='inline')` {#Inline.__init__}

Creates a `Bijector` from callables.

##### Args:


*  <b>`forward_fn`</b>: Python callable implementing the forward transformation.
*  <b>`inverse_fn`</b>: Python callable implementing the inverse transformation.
*  <b>`inverse_log_det_jacobian_fn`</b>: Python callable implementing the
    log o det o jacobian of the inverse transformation.
*  <b>`forward_log_det_jacobian_fn`</b>: Python callable implementing the
    log o det o jacobian of the forward transformation.
*  <b>`forward_event_shape_fn`</b>: Python callable implementing non-identical
    static event shape changes. Default: shape is assumed unchanged.
*  <b>`forward_event_shape_tensor_fn`</b>: Python callable implementing non-identical
    event shape changes. Default: shape is assumed unchanged.
*  <b>`inverse_event_shape_fn`</b>: Python callable implementing non-identical
    static event shape changes. Default: shape is assumed unchanged.
*  <b>`inverse_event_shape_tensor_fn`</b>: Python callable implementing non-identical
    event shape changes. Default: shape is assumed unchanged.
*  <b>`is_constant_jacobian`</b>: Python `bool` indicating that the Jacobian is
    constant for all input arguments.
*  <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
    checked for correctness.
*  <b>`name`</b>: Python `str`, name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.Inline.dtype` {#Inline.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Inline.event_ndims` {#Inline.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.Inline.forward(x, name='forward')` {#Inline.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.Inline.forward_event_shape(input_shape)` {#Inline.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Inline.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#Inline.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Inline.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#Inline.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Inline.graph_parents` {#Inline.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Inline.inverse(y, name='inverse')` {#Inline.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Inline.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#Inline.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Inline.inverse_event_shape(output_shape)` {#Inline.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Inline.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#Inline.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Inline.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#Inline.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Inline.is_constant_jacobian` {#Inline.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.Inline.name` {#Inline.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Inline.validate_args` {#Inline.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.Invert` {#Invert}

Bijector which inverts another Bijector.

Example Use: [ExpGammaDistribution (see Background & Context)](
https://reference.wolfram.com/language/ref/ExpGammaDistribution.html)
models `Y=log(X)` where `X ~ Gamma`.

```python
exp_gamma_distribution = TransformedDistribution(
  distribution=Gamma(concentration=1., rate=2.),
  bijector=bijector.Invert(bijector.Exp())
```
- - -

#### `tf.contrib.distributions.bijector.Invert.__init__(bijector, validate_args=False, name=None)` {#Invert.__init__}

Creates a `Bijector` which swaps the meaning of `inverse` and `forward`.

Note: An inverted bijector's `inverse_log_det_jacobian` is often more
efficient if the base bijector implements `_forward_log_det_jacobian`. If
`_forward_log_det_jacobian` is not implemented then the following code is
used:

```python
y = self.inverse(x, **kwargs)
return -self.inverse_log_det_jacobian(y, **kwargs)
```

##### Args:


*  <b>`bijector`</b>: Bijector instance.
*  <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
    checked for correctness.
*  <b>`name`</b>: Python `str`, name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.Invert.bijector` {#Invert.bijector}




- - -

#### `tf.contrib.distributions.bijector.Invert.dtype` {#Invert.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Invert.event_ndims` {#Invert.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.Invert.forward(x, name='forward')` {#Invert.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.Invert.forward_event_shape(input_shape)` {#Invert.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Invert.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#Invert.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Invert.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#Invert.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Invert.graph_parents` {#Invert.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Invert.inverse(y, name='inverse')` {#Invert.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Invert.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#Invert.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Invert.inverse_event_shape(output_shape)` {#Invert.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Invert.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#Invert.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Invert.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#Invert.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Invert.is_constant_jacobian` {#Invert.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.Invert.name` {#Invert.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Invert.validate_args` {#Invert.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.PowerTransform` {#PowerTransform}

Compute `Y = g(X) = (1 + X * c)**(1 / c), X >= -1 / c`.

The [power transform](https://en.wikipedia.org/wiki/Power_transform) maps
inputs from `[0, inf]` to `[-1/c, inf]`; this is equivalent to the `inverse`
of this bijector.

This bijector is equivalent to the `Exp` bijector when `c=0`.
- - -

#### `tf.contrib.distributions.bijector.PowerTransform.__init__(power=0.0, event_ndims=0, validate_args=False, name='power_transform')` {#PowerTransform.__init__}

Instantiates the `PowerTransform` bijector.

##### Args:


*  <b>`power`</b>: Python `float` scalar indicating the transform power, i.e.,
    `Y = g(X) = (1 + X * c)**(1 / c)` where `c` is the `power`.
*  <b>`event_ndims`</b>: Python scalar indicating the number of dimensions associated
    with a particular draw from the distribution.
*  <b>`validate_args`</b>: Python `bool` indicating whether arguments should be
    checked for correctness.
*  <b>`name`</b>: Python `str` name given to ops managed by this object.

##### Raises:


*  <b>`ValueError`</b>: if `power < 0` or is not known statically.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.dtype` {#PowerTransform.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.event_ndims` {#PowerTransform.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.forward(x, name='forward')` {#PowerTransform.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.forward_event_shape(input_shape)` {#PowerTransform.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#PowerTransform.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#PowerTransform.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.graph_parents` {#PowerTransform.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.inverse(y, name='inverse')` {#PowerTransform.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#PowerTransform.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.inverse_event_shape(output_shape)` {#PowerTransform.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#PowerTransform.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#PowerTransform.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.is_constant_jacobian` {#PowerTransform.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.name` {#PowerTransform.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.power` {#PowerTransform.power}

The `c` in: `Y = g(X) = (1 + X * c)**(1 / c)`.


- - -

#### `tf.contrib.distributions.bijector.PowerTransform.validate_args` {#PowerTransform.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.SigmoidCentered` {#SigmoidCentered}

Bijector which computes Y = g(X) = exp([X 0]) / (1 + exp(-X)).

Equivalent to: `bijector.SoftmaxCentered(event_ndims=0)`.

See `bijector.SoftmaxCentered` for more details.
- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.__init__(validate_args=False, name='sigmoid_centered')` {#SigmoidCentered.__init__}




- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.dtype` {#SigmoidCentered.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.event_ndims` {#SigmoidCentered.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.forward(x, name='forward')` {#SigmoidCentered.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.forward_event_shape(input_shape)` {#SigmoidCentered.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#SigmoidCentered.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#SigmoidCentered.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.graph_parents` {#SigmoidCentered.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.inverse(y, name='inverse')` {#SigmoidCentered.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#SigmoidCentered.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.inverse_event_shape(output_shape)` {#SigmoidCentered.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#SigmoidCentered.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#SigmoidCentered.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.is_constant_jacobian` {#SigmoidCentered.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.name` {#SigmoidCentered.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.validate_args` {#SigmoidCentered.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.SoftmaxCentered` {#SoftmaxCentered}

Bijector which computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`.

To implement [softmax](https://en.wikipedia.org/wiki/Softmax_function) as a
bijection, the forward transformation appends a value to the input and the
inverse removes this coordinate. The appended coordinate represents a pivot,
e.g., `softmax(x) = exp(x-c) / sum(exp(x-c))` where `c` is the implicit last
coordinate.

Because we append a coordinate, this bijector only supports `event_ndim in [0,
1]`, i.e., scalars and vectors.

Example Use:

```python
bijector.SoftmaxCentered(event_ndims=1).forward(tf.log([2, 3, 4]))
# Result: [0.2, 0.3, 0.4, 0.1]
# Extra result: 0.1

bijector.SoftmaxCentered(event_ndims=1).inverse([0.2, 0.3, 0.4, 0.1])
# Result: tf.log([2, 3, 4])
# Extra coordinate removed.
```

At first blush it may seem like the [Invariance of domain](
https://en.wikipedia.org/wiki/Invariance_of_domain) theorem implies this
implementation is not a bijection. However, the appended dimension
makes the (forward) image non-open and the theorem does not directly apply.
- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.__init__(event_ndims=0, validate_args=False, name='softmax_centered')` {#SoftmaxCentered.__init__}




- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.dtype` {#SoftmaxCentered.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.event_ndims` {#SoftmaxCentered.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward(x, name='forward')` {#SoftmaxCentered.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward_event_shape(input_shape)` {#SoftmaxCentered.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#SoftmaxCentered.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#SoftmaxCentered.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.graph_parents` {#SoftmaxCentered.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse(y, name='inverse')` {#SoftmaxCentered.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#SoftmaxCentered.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_event_shape(output_shape)` {#SoftmaxCentered.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#SoftmaxCentered.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#SoftmaxCentered.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.is_constant_jacobian` {#SoftmaxCentered.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.name` {#SoftmaxCentered.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.validate_args` {#SoftmaxCentered.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.Softplus` {#Softplus}

Bijector which computes `Y = g(X) = Log[1 + exp(X)]`.

The softplus `Bijector` has the following two useful properties:

* The domain is the positive real numbers
* `softplus(x) approx x`, for large `x`, so it does not overflow as easily as
  the `Exp` `Bijector`.

  Example Use:

  ```python
  # Create the Y=g(X)=softplus(X) transform which works only on Tensors with 1
  # batch ndim and 2 event ndims (i.e., vector of matrices).
  softplus = Softplus(event_ndims=2)
  x = [[[1., 2],
         [3, 4]],
        [[5, 6],
         [7, 8]]]
  log(1 + exp(x)) == softplus.forward(x)
  log(exp(x) - 1) == softplus.inverse(x)
  ```

  Note: log(.) and exp(.) are applied element-wise but the Jacobian is a
  reduction over the event space.
- - -

#### `tf.contrib.distributions.bijector.Softplus.__init__(event_ndims=0, validate_args=False, name='softplus')` {#Softplus.__init__}




- - -

#### `tf.contrib.distributions.bijector.Softplus.dtype` {#Softplus.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Softplus.event_ndims` {#Softplus.event_ndims}

Returns then number of event dimensions this bijector operates on.


- - -

#### `tf.contrib.distributions.bijector.Softplus.forward(x, name='forward')` {#Softplus.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.Softplus.forward_event_shape(input_shape)` {#Softplus.forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Softplus.forward_event_shape_tensor(input_shape, name='forward_event_shape_tensor')` {#Softplus.forward_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Softplus.forward_log_det_jacobian(x, name='forward_log_det_jacobian')` {#Softplus.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Softplus.graph_parents` {#Softplus.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.Softplus.inverse(y, name='inverse')` {#Softplus.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Softplus.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian')` {#Softplus.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.Softplus.inverse_event_shape(output_shape)` {#Softplus.inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape_tensor`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `TensorShape` indicating event-portion shape
    after applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Softplus.inverse_event_shape_tensor(output_shape, name='inverse_event_shape_tensor')` {#Softplus.inverse_event_shape_tensor}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape_tensor`</b>: `Tensor`, `int32` vector indicating
    event-portion shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Softplus.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian')` {#Softplus.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_log_det_jacobian` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.Softplus.is_constant_jacobian` {#Softplus.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:


*  <b>`is_constant_jacobian`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.bijector.Softplus.name` {#Softplus.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Softplus.validate_args` {#Softplus.validate_args}

Returns True if Tensor arguments will be validated.



