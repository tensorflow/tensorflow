Bijector which computes `Y = g(X; shift, scale) = scale @ X.T + shift`.

`shift` is a numeric `Tensor` and `scale` is a `LinearOperator`.

If `X` is a scalar then the forward transformation is: `scale * X + shift`
where `*` denotes the scalar product.

Note: we don't always simply transpose `X` (but write it this way for
brevity).  Actually the input `X` undergoes the following transformation
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
of matrices where each column is a draw from the distribution.  After
premultiplying by `scale`, we take the inverse of this procedure.  The input
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
# diag * scale + shift
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


*  <b>`shift`</b>: Numeric `Tensor`.
*  <b>`scale`</b>: Subclass of `LinearOperator`.  Represents the (batch) positive
    definite matrix `M` in `R^{k x k}`.
*  <b>`event_ndims`</b>: Scalar `integer` `Tensor` indicating the number of dimensions
    associated with a particular draw from the distribution. Must be 0 or 1.
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String` name given to ops managed by this object.

##### Raises:


*  <b>`ValueError`</b>: if `event_ndims` is not 0 or 1.
*  <b>`TypeError`</b>: if `scale` is not a `LinearOperator`.
*  <b>`TypeError`</b>: if `shift.dtype` does not match `scale.dtype`.
*  <b>`ValueError`</b>: if not `scale.is_non_singular`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.dtype` {#AffineLinearOperator.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.forward(x, name='forward', **condition_kwargs)` {#AffineLinearOperator.forward}

Returns the forward `Bijector` evaluation, i.e., X = g(Y).

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `x.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if `_forward` is not implemented.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.forward_event_shape(input_shape, name='forward_event_shape')` {#AffineLinearOperator.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#AffineLinearOperator.forward_log_det_jacobian}

Returns both the forward_log_det_jacobian.

##### Args:


*  <b>`x`</b>: `Tensor`. The input to the "forward" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_forward_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.get_forward_event_shape(input_shape)` {#AffineLinearOperator.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.get_inverse_event_shape(output_shape)` {#AffineLinearOperator.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.graph_parents` {#AffineLinearOperator.graph_parents}

Returns this `Bijector`'s graph_parents as a Python list.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.inverse(y, name='inverse', **condition_kwargs)` {#AffineLinearOperator.inverse}

Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse` nor
    `_inverse_and_inverse_log_det_jacobian` are implemented.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#AffineLinearOperator.inverse_and_inverse_log_det_jacobian}

Returns both the inverse evaluation and inverse_log_det_jacobian.

Enables possibly more efficient calculation when both inverse and
corresponding Jacobian are needed.

See `inverse()`, `inverse_log_det_jacobian()` for more details.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor`.

##### Raises:


*  <b>`TypeError`</b>: if `self.dtype` is specified and `y.dtype` is not
    `self.dtype`.
*  <b>`NotImplementedError`</b>: if neither `_inverse_and_inverse_log_det_jacobian`
    nor {`_inverse`, `_inverse_log_det_jacobian`} are implemented.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.inverse_event_shape(output_shape, name='inverse_event_shape')` {#AffineLinearOperator.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#AffineLinearOperator.inverse_log_det_jacobian}

Returns the (log o det o Jacobian o inverse)(y).

Mathematically, returns: `log(det(dX/dY))(Y)`. (Recall that: `X=g^{-1}(Y)`.)

Note that `forward_log_det_jacobian` is the negative of this function.

##### Args:


*  <b>`y`</b>: `Tensor`. The input to the "inverse" Jacobian evaluation.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

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

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.name` {#AffineLinearOperator.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.scale` {#AffineLinearOperator.scale}

The `scale` `LinearOperator` in `Y = scale @ X.T + shift`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.shaper` {#AffineLinearOperator.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.shift` {#AffineLinearOperator.shift}

The `shift` `Tensor` in `Y = scale @ X.T + shift`.


- - -

#### `tf.contrib.distributions.bijector.AffineLinearOperator.validate_args` {#AffineLinearOperator.validate_args}

Returns True if Tensor arguments will be validated.


