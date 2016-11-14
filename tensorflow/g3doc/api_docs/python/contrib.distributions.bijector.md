<!-- This file is machine generated: DO NOT EDIT! -->

# Random variable transformations (contrib)
[TOC]

Bijector Ops.

An API for invertible, differentiable transformations of random variables.

## Background

Differentiable, bijective transformations of continuous random variables alter
the calculations made in the cumulative/probability distribution functions and
sample function.  This module provides a standard interface for making these
manipulations.

For more details and examples, see the `Bijector` docstring.

To apply a `Bijector`, use `distributions.TransformedDistribution`.

## Bijectors

- - -

### `class tf.contrib.distributions.bijector.Bijector` {#Bijector}

Interface for transforming a `Distribution` sample.

A `Bijector` implements a
[diffeomorphism](https://en.wikipedia.org/wiki/Diffeomorphism), i.e., a
bijective, differentiable function. A `Bijector` is used by
`TransformedDistribution` but can be generally used for transforming a
`Distribution` generated `Tensor`.  A `Bijector` is characterized by three
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
   of another.  Geometrically, the det(Jacobian) is the volume of the
   transformation and is used to scale the probability.

By convention, transformations of random variables are named in terms of the
forward transformation. The forward transformation creates samples, the
inverse is useful for computing probabilities.

Example Use:

  - Basic properties:

  ```python
  x = ... # A tensor.
  # Evaluate forward transformation.
  fwd_x = my_bijector.forward(x)
  x == my_bijector.inverse(fwd_x)
  x != my_bijector.forward(fwd_x)  # Not equal because g(x) != g(g(x)).
  ```

  - Computing a log-likelihood:

  ```python
  def transformed_log_pdf(bijector, log_pdf, x):
    return (bijector.inverse_log_det_jacobian(x) +
            log_pdf(bijector.inverse(x)))
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

  - "ScaleAndShift"

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
  and event (S, B, E) shape semantics.  Suppose
  the `Tensor`'s partitioned-shape is `(S=[4], B=[2], E=[3, 3])`.

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
  to implement all of `_forward` and `_inverese` functions.  Examples:
    1. Sampling (e.g., `sample`) only requires `_forward`.
    2. Probability functions (e.g., `prob`, `cdf`, `survival`) only require
       `_inverse` (and related).
    3. Only calling probability functions on the output of `sample` means
      `_inverse` can be implemented as a cache lookup.

  See `Example Use` [above] which shows how these functions are used to
  transform a distribution.  (Note: `_forward` could theoretically be
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

- Subclasses should implement `_get_forward_event_shape`,
  `_forward_event_shape` (and `inverse` counterparts) if the transformation is
  shape-changing.  By default the event-shape is assumed unchanged from input.

Tips for implementing `_inverse` and `_inverse_log_det_jacobian`:

- As case 3 [above] indicates, under some circumstances the inverse function
  can be implemented as a cache lookup.

- The inverse `log o det o Jacobian` can be implemented as the negative of the
  forward `log o det o Jacobian`.  This is useful if the `inverse` is
  implemented as a cache or the inverse Jacobian is computationally more
  expensive (e.g., `CholeskyOuterProduct` `Bijector`). The following
  demonstrates the suggested implementation.

  ```python
  def _inverse_and_log_det_jacobian(self, y):
     x = # ... implement inverse, possibly via cache.
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

#### `tf.contrib.distributions.bijector.Bijector.__init__(batch_ndims=None, event_ndims=None, parameters=None, is_constant_jacobian=False, validate_args=False, dtype=None, name=None)` {#Bijector.__init__}

Constructs Bijector.

A `Bijector` transforms random variables into new random variables.

Examples:

```python
# Create the Y = g(X) = X transform which operates on 4-Tensors of vectors.
identity = Identity(batch_ndims=4, event_ndims=1)

# Create the Y = g(X) = exp(X) transform which operates on matrices.
exp = Exp(batch_ndims=0, event_ndims=2)
```

See `Bijector` subclass docstring for more details and specific examples.

##### Args:


*  <b>`batch_ndims`</b>: number of dimensions associated with batch coordinates.
*  <b>`event_ndims`</b>: number of dimensions associated with event coordinates.
*  <b>`parameters`</b>: Dictionary of parameters used by this `Bijector`
*  <b>`is_constant_jacobian`</b>: `Boolean` indicating that the Jacobian is not a
    function of the input.
*  <b>`validate_args`</b>: `Boolean`, default `False`.  Whether to validate input with
    asserts. If `validate_args` is `False`, and the inputs are invalid,
    correct behavior is not guaranteed.
*  <b>`dtype`</b>: `tf.dtype` supported by this `Bijector`. `None` means dtype is not
    enforced.
*  <b>`name`</b>: The name to give Ops created by the initializer.


- - -

#### `tf.contrib.distributions.bijector.Bijector.dtype` {#Bijector.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Bijector.forward(x, name='forward', **condition_kwargs)` {#Bijector.forward}

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

#### `tf.contrib.distributions.bijector.Bijector.forward_event_shape(input_shape, name='forward_event_shape')` {#Bijector.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Bijector.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#Bijector.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Bijector.get_forward_event_shape(input_shape)` {#Bijector.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Bijector.get_inverse_event_shape(output_shape)` {#Bijector.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Bijector.inverse(y, name='inverse', **condition_kwargs)` {#Bijector.inverse}

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

#### `tf.contrib.distributions.bijector.Bijector.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#Bijector.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Bijector.inverse_event_shape(output_shape, name='inverse_event_shape')` {#Bijector.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Bijector.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#Bijector.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Bijector.is_constant_jacobian` {#Bijector.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Bijector.name` {#Bijector.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Bijector.parameters` {#Bijector.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.Bijector.shaper` {#Bijector.shaper}

Returns shape object used to manage shape constraints.


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
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String`, name given to ops managed by this object. Default: E.g.,
    `Chain([Exp(), Softplus()]).name == "chain_of_exp_of_softplus"`.

##### Raises:


*  <b>`ValueError`</b>: if bijectors have different dtypes.


- - -

#### `tf.contrib.distributions.bijector.Chain.bijectors` {#Chain.bijectors}




- - -

#### `tf.contrib.distributions.bijector.Chain.dtype` {#Chain.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Chain.forward(x, name='forward', **condition_kwargs)` {#Chain.forward}

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

#### `tf.contrib.distributions.bijector.Chain.forward_event_shape(input_shape, name='forward_event_shape')` {#Chain.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Chain.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#Chain.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Chain.get_forward_event_shape(input_shape)` {#Chain.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Chain.get_inverse_event_shape(output_shape)` {#Chain.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Chain.inverse(y, name='inverse', **condition_kwargs)` {#Chain.inverse}

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

#### `tf.contrib.distributions.bijector.Chain.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#Chain.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Chain.inverse_event_shape(output_shape, name='inverse_event_shape')` {#Chain.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Chain.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#Chain.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Chain.is_constant_jacobian` {#Chain.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Chain.name` {#Chain.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Chain.parameters` {#Chain.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.Chain.shaper` {#Chain.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.Chain.validate_args` {#Chain.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.CholeskyOuterProduct` {#CholeskyOuterProduct}

Bijector which computes Y = g(X) = X X^T where X is a lower-triangular, positive-diagonal matrix.

`event_ndims` must be 0 or 2, i.e., scalar or matrix.

Note: the upper-triangular part of X is ignored (whether or not its zero).

Examples:

```python
bijector.CholeskyOuterProduct(event_ndims=2).forward(x=[[1., 0], [2, 1]])
# Result: [[1, 1], [1, 5]], i.e., x x^T

bijector.SoftmaxCentered(event_ndims=2).inverse(y=[[1., 1], [1, 5]])
# Result: [[1, 0], [2, 1]], i.e., chol(y).
```
- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.__init__(event_ndims=2, validate_args=False, name='cholesky_outer_product')` {#CholeskyOuterProduct.__init__}

Instantiates the `CholeskyOuterProduct` bijector.

##### Args:


*  <b>`event_ndims`</b>: `constant` `int32` scalar `Tensor` indicating the number of
    dimensions associated with a particular draw from the distribution. Must
    be 0 or 2.
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String` name given to ops managed by this object.

##### Raises:


*  <b>`ValueError`</b>: if event_ndims is neither 0 or 2.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.dtype` {#CholeskyOuterProduct.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.forward(x, name='forward', **condition_kwargs)` {#CholeskyOuterProduct.forward}

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

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.forward_event_shape(input_shape, name='forward_event_shape')` {#CholeskyOuterProduct.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#CholeskyOuterProduct.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.get_forward_event_shape(input_shape)` {#CholeskyOuterProduct.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.get_inverse_event_shape(output_shape)` {#CholeskyOuterProduct.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse(y, name='inverse', **condition_kwargs)` {#CholeskyOuterProduct.inverse}

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

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#CholeskyOuterProduct.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse_event_shape(output_shape, name='inverse_event_shape')` {#CholeskyOuterProduct.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#CholeskyOuterProduct.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.is_constant_jacobian` {#CholeskyOuterProduct.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.name` {#CholeskyOuterProduct.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.parameters` {#CholeskyOuterProduct.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.shaper` {#CholeskyOuterProduct.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.CholeskyOuterProduct.validate_args` {#CholeskyOuterProduct.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.Exp` {#Exp}

Bijector which computes Y = g(X) = exp(X).

Example Use:

```python
# Create the Y=g(X)=exp(X) transform which works only on Tensors with 1
# batch ndim and 2 event ndims (i.e., vector of matrices).
exp = Exp(batch_ndims=1, event_ndims=2)
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
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String` name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.Exp.dtype` {#Exp.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Exp.forward(x, name='forward', **condition_kwargs)` {#Exp.forward}

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

#### `tf.contrib.distributions.bijector.Exp.forward_event_shape(input_shape, name='forward_event_shape')` {#Exp.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Exp.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#Exp.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Exp.get_forward_event_shape(input_shape)` {#Exp.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Exp.get_inverse_event_shape(output_shape)` {#Exp.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Exp.inverse(y, name='inverse', **condition_kwargs)` {#Exp.inverse}

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

#### `tf.contrib.distributions.bijector.Exp.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#Exp.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Exp.inverse_event_shape(output_shape, name='inverse_event_shape')` {#Exp.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Exp.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#Exp.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Exp.is_constant_jacobian` {#Exp.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Exp.name` {#Exp.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Exp.parameters` {#Exp.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.Exp.shaper` {#Exp.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.Exp.validate_args` {#Exp.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.Identity` {#Identity}

Bijector which computes Y = g(X) = X.

Example Use:

```python
# Create the Y=g(X)=X transform which is intended for Tensors with 1 batch
# ndim and 1 event ndim (i.e., vector of vectors).
identity = Identity(batch_ndims=1, event_ndims=1)
x = [[1., 2],
     [3, 4]]
x == identity.forward(x) == identity.inverse(x)
```
- - -

#### `tf.contrib.distributions.bijector.Identity.__init__(validate_args=False, name='identity')` {#Identity.__init__}




- - -

#### `tf.contrib.distributions.bijector.Identity.dtype` {#Identity.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Identity.forward(x, name='forward', **condition_kwargs)` {#Identity.forward}

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

#### `tf.contrib.distributions.bijector.Identity.forward_event_shape(input_shape, name='forward_event_shape')` {#Identity.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Identity.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#Identity.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Identity.get_forward_event_shape(input_shape)` {#Identity.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Identity.get_inverse_event_shape(output_shape)` {#Identity.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Identity.inverse(y, name='inverse', **condition_kwargs)` {#Identity.inverse}

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

#### `tf.contrib.distributions.bijector.Identity.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#Identity.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Identity.inverse_event_shape(output_shape, name='inverse_event_shape')` {#Identity.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Identity.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#Identity.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Identity.is_constant_jacobian` {#Identity.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Identity.name` {#Identity.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Identity.parameters` {#Identity.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.Identity.shaper` {#Identity.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.Identity.validate_args` {#Identity.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.Inline` {#Inline}

Bijector constructed from callables implementing forward, inverse, and inverse_log_det_jacobian.

Example Use:

```python
exp = Inline(
  forward_fn=tf.exp,
  inverse_fn=tf.log,
  inverse_log_det_jacobian_fn=(
    lambda y: -tf.reduce_sum(tf.log(y), reduction_indices=-1)),
  name="exp")
```

The above example is equivalent to the `Bijector` `Exp(event_ndims=1)`.
- - -

#### `tf.contrib.distributions.bijector.Inline.__init__(forward_fn=None, inverse_fn=None, inverse_log_det_jacobian_fn=None, forward_log_det_jacobian_fn=None, get_forward_event_shape_fn=None, forward_event_shape_fn=None, get_inverse_event_shape_fn=None, inverse_event_shape_fn=None, is_constant_jacobian=False, validate_args=False, name='inline')` {#Inline.__init__}

Creates a `Bijector` from callables.

##### Args:


*  <b>`forward_fn`</b>: Python callable implementing the forward transformation.
*  <b>`inverse_fn`</b>: Python callable implementing the inverse transformation.
*  <b>`inverse_log_det_jacobian_fn`</b>: Python callable implementing the
    log o det o jacobian of the inverse transformation.
*  <b>`forward_log_det_jacobian_fn`</b>: Python callable implementing the
    log o det o jacobian of the forward transformation.
*  <b>`get_forward_event_shape_fn`</b>: Python callable implementing non-identical
    static event shape changes. Default: shape is assumed unchanged.
*  <b>`forward_event_shape_fn`</b>: Python callable implementing non-identical event
    shape changes. Default: shape is assumed unchanged.
*  <b>`get_inverse_event_shape_fn`</b>: Python callable implementing non-identical
    static event shape changes. Default: shape is assumed unchanged.
*  <b>`inverse_event_shape_fn`</b>: Python callable implementing non-identical event
    shape changes. Default: shape is assumed unchanged.
*  <b>`is_constant_jacobian`</b>: `Boolean` indicating that the Jacobian is constant
    for all input arguments.
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String`, name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.Inline.dtype` {#Inline.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Inline.forward(x, name='forward', **condition_kwargs)` {#Inline.forward}

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

#### `tf.contrib.distributions.bijector.Inline.forward_event_shape(input_shape, name='forward_event_shape')` {#Inline.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Inline.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#Inline.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Inline.get_forward_event_shape(input_shape)` {#Inline.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Inline.get_inverse_event_shape(output_shape)` {#Inline.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Inline.inverse(y, name='inverse', **condition_kwargs)` {#Inline.inverse}

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

#### `tf.contrib.distributions.bijector.Inline.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#Inline.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Inline.inverse_event_shape(output_shape, name='inverse_event_shape')` {#Inline.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Inline.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#Inline.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Inline.is_constant_jacobian` {#Inline.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Inline.name` {#Inline.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Inline.parameters` {#Inline.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.Inline.shaper` {#Inline.shaper}

Returns shape object used to manage shape constraints.


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
  Gamma(alpha=1., beta=2.),
  bijector.Invert(bijector.Exp())
```
- - -

#### `tf.contrib.distributions.bijector.Invert.__init__(bijector, validate_args=False, name=None)` {#Invert.__init__}

Creates a `Bijector` which swaps the meaning of `inverse` and `forward`.

Note: An inverted bijector's `inverse_log_det_jacobian` is often more
efficient if the base bijector implements `_forward_log_det_jacobian`. If
`_forward_log_det_jacobian` is not implemented then the following code is
used:

```python
y = self.inverse(x, **condition_kwargs)
return -self.inverse_log_det_jacobian(y, **condition_kwargs)
```

##### Args:


*  <b>`bijector`</b>: Bijector instance.
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String`, name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.Invert.bijector` {#Invert.bijector}




- - -

#### `tf.contrib.distributions.bijector.Invert.dtype` {#Invert.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.Invert.forward(x, name='forward', **condition_kwargs)` {#Invert.forward}

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

#### `tf.contrib.distributions.bijector.Invert.forward_event_shape(input_shape, name='forward_event_shape')` {#Invert.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Invert.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#Invert.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Invert.get_forward_event_shape(input_shape)` {#Invert.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Invert.get_inverse_event_shape(output_shape)` {#Invert.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Invert.inverse(y, name='inverse', **condition_kwargs)` {#Invert.inverse}

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

#### `tf.contrib.distributions.bijector.Invert.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#Invert.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Invert.inverse_event_shape(output_shape, name='inverse_event_shape')` {#Invert.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Invert.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#Invert.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Invert.is_constant_jacobian` {#Invert.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Invert.name` {#Invert.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Invert.parameters` {#Invert.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.Invert.shaper` {#Invert.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.Invert.validate_args` {#Invert.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.ScaleAndShift` {#ScaleAndShift}

Bijector which computes Y = g(X; shift, scale) = matmul(scale, X) + shift.

`scale` is either a non-zero scalar, or a lower triangular matrix with
non-zero diagonal.  This means the `Bijector` will be invertible and
computation of determinant and inverse will be efficient.

As a result, the mean and covariance are transformed:

```
E[Y] = matmul(scale, E[X])
Cov[Y] = matmul(scale, matmul(Cov[X], scale, transpose_b=True))
```

Example Use:

```python
# No batch, scalar
mu = 0     # shape=[]
sigma = 1  # shape=[], treated like a 1x1 matrix.
b = ScaleAndShift(shift=mu, scale=sigma)
# b.shaper.batch_ndims == 0
# b.shaper.event_ndims == 0

# One batch, scalar.
mu = ...    # shape=[b], b>0
sigma = ... # shape=[b], b>0, treated like a batch of 1x1 matrices
b = ScaleAndShift(shift=mu, scale=sigma)
# b.shaper.batch_ndims == 1
# b.shaper.event_ndims == 0

# No batch, multivariate.
mu = ...    # shape=[d],    d>0
sigma = ... # shape=[d, d], d>0, treated like a single dxd matrix.
b = ScaleAndShift(shift=mu, scale=sigma, event_ndims=1)
# b.shaper.batch_ndims == 0
# b.shaper.event_ndims == 1

# (B1*B2*...*Bb)-batch, multivariate.
mu = ...    # shape=[B1,...,Bb, d],    b>0, d>0
sigma = ... # shape=[B1,...,Bb, d, d], b>0, d>0
b = ScaleAndShift(shift=mu, scale=sigma, event_ndims=1)
# b.shaper.batch_ndims == b
# b.shaper.event_ndims == 1

# Mu is broadcast:
mu = 1
sigma = [I, I]  # I is a 3x3 identity matrix.
b = ScaleAndShift(shift=mu, scale=sigma, event_ndims=1)
x = numpy.ones(S + sigma.shape)
b.forward(x) # == x + 1
```
- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.__init__(shift, scale, event_ndims=0, validate_args=False, name='scale_and_shift')` {#ScaleAndShift.__init__}

Instantiates the `ScaleAndShift` bijector.

This `Bijector` is initialized with `scale` and `shift` `Tensors`, giving
the forward operation:

```Y = g(X) = matmul(scale, X) + shift```

##### Args:


*  <b>`shift`</b>: Numeric `Tensor`.
*  <b>`scale`</b>: Numeric `Tensor` of same `dtype` as `shift`.  If `event_ndims = 0`,
    `scale` is treated like a `1x1` matrix or a batch thereof.
    Otherwise, the last two dimensions of `scale` define a matrix.
    `scale` must have non-negative diagonal entries.  The upper triangular
    part of `scale` is ignored, effectively making it lower triangular.
*  <b>`event_ndims`</b>: Scalar `int32` `Tensor` indicating the number of dimensions
    associated with a particular draw from the distribution.  Must be 0 or 1
*  <b>`validate_args`</b>: `Boolean` indicating whether arguments should be checked
    for correctness.
*  <b>`name`</b>: `String` name given to ops managed by this object.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.dtype` {#ScaleAndShift.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.forward(x, name='forward', **condition_kwargs)` {#ScaleAndShift.forward}

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

#### `tf.contrib.distributions.bijector.ScaleAndShift.forward_event_shape(input_shape, name='forward_event_shape')` {#ScaleAndShift.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#ScaleAndShift.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.ScaleAndShift.get_forward_event_shape(input_shape)` {#ScaleAndShift.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.get_inverse_event_shape(output_shape)` {#ScaleAndShift.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.inverse(y, name='inverse', **condition_kwargs)` {#ScaleAndShift.inverse}

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

#### `tf.contrib.distributions.bijector.ScaleAndShift.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#ScaleAndShift.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.ScaleAndShift.inverse_event_shape(output_shape, name='inverse_event_shape')` {#ScaleAndShift.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#ScaleAndShift.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.ScaleAndShift.is_constant_jacobian` {#ScaleAndShift.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.name` {#ScaleAndShift.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.parameters` {#ScaleAndShift.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.scale` {#ScaleAndShift.scale}




- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.shaper` {#ScaleAndShift.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.shift` {#ScaleAndShift.shift}




- - -

#### `tf.contrib.distributions.bijector.ScaleAndShift.validate_args` {#ScaleAndShift.validate_args}

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

#### `tf.contrib.distributions.bijector.SigmoidCentered.forward(x, name='forward', **condition_kwargs)` {#SigmoidCentered.forward}

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

#### `tf.contrib.distributions.bijector.SigmoidCentered.forward_event_shape(input_shape, name='forward_event_shape')` {#SigmoidCentered.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#SigmoidCentered.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.SigmoidCentered.get_forward_event_shape(input_shape)` {#SigmoidCentered.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.get_inverse_event_shape(output_shape)` {#SigmoidCentered.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.inverse(y, name='inverse', **condition_kwargs)` {#SigmoidCentered.inverse}

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

#### `tf.contrib.distributions.bijector.SigmoidCentered.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#SigmoidCentered.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.SigmoidCentered.inverse_event_shape(output_shape, name='inverse_event_shape')` {#SigmoidCentered.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#SigmoidCentered.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.SigmoidCentered.is_constant_jacobian` {#SigmoidCentered.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.name` {#SigmoidCentered.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.parameters` {#SigmoidCentered.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.shaper` {#SigmoidCentered.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.SigmoidCentered.validate_args` {#SigmoidCentered.validate_args}

Returns True if Tensor arguments will be validated.



- - -

### `class tf.contrib.distributions.bijector.SoftmaxCentered` {#SoftmaxCentered}

Bijector which computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`.

To implement [softmax](https://en.wikipedia.org/wiki/Softmax_function) as a
bijection, the forward transformation appends a value to the input and the
inverse removes this coordinate.  The appended coordinate represents a pivot,
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
implementation is not a bijection.  However, the appended dimension
makes the (forward) image non-open and the theorem does not directly apply.
- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.__init__(event_ndims=0, validate_args=False, name='softmax_centered')` {#SoftmaxCentered.__init__}




- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.dtype` {#SoftmaxCentered.dtype}

dtype of `Tensor`s transformable by this distribution.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward(x, name='forward', **condition_kwargs)` {#SoftmaxCentered.forward}

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

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward_event_shape(input_shape, name='forward_event_shape')` {#SoftmaxCentered.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#SoftmaxCentered.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.SoftmaxCentered.get_forward_event_shape(input_shape)` {#SoftmaxCentered.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.get_inverse_event_shape(output_shape)` {#SoftmaxCentered.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse(y, name='inverse', **condition_kwargs)` {#SoftmaxCentered.inverse}

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

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#SoftmaxCentered.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_event_shape(output_shape, name='inverse_event_shape')` {#SoftmaxCentered.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#SoftmaxCentered.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.SoftmaxCentered.is_constant_jacobian` {#SoftmaxCentered.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.name` {#SoftmaxCentered.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.parameters` {#SoftmaxCentered.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.SoftmaxCentered.shaper` {#SoftmaxCentered.shaper}

Returns shape object used to manage shape constraints.


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
  softplus = Softplus(batch_ndims=1, event_ndims=2)
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

#### `tf.contrib.distributions.bijector.Softplus.forward(x, name='forward', **condition_kwargs)` {#Softplus.forward}

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

#### `tf.contrib.distributions.bijector.Softplus.forward_event_shape(input_shape, name='forward_event_shape')` {#Softplus.forward_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`input_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `forward` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`forward_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `forward`.


- - -

#### `tf.contrib.distributions.bijector.Softplus.forward_log_det_jacobian(x, name='forward_log_det_jacobian', **condition_kwargs)` {#Softplus.forward_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Softplus.get_forward_event_shape(input_shape)` {#Softplus.get_forward_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `forward_event_shape`. May be only partially defined.

##### Args:


*  <b>`input_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `forward` function.

##### Returns:


*  <b>`forward_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `forward`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Softplus.get_inverse_event_shape(output_shape)` {#Softplus.get_inverse_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `inverse_event_shape`. May be only partially defined.

##### Args:


*  <b>`output_shape`</b>: `TensorShape` indicating event-portion shape passed into
    `inverse` function.

##### Returns:


*  <b>`inverse_event_shape`</b>: `TensorShape` indicating event-portion shape after
    applying `inverse`. Possibly unknown.


- - -

#### `tf.contrib.distributions.bijector.Softplus.inverse(y, name='inverse', **condition_kwargs)` {#Softplus.inverse}

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

#### `tf.contrib.distributions.bijector.Softplus.inverse_and_inverse_log_det_jacobian(y, name='inverse_and_inverse_log_det_jacobian', **condition_kwargs)` {#Softplus.inverse_and_inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Softplus.inverse_event_shape(output_shape, name='inverse_event_shape')` {#Softplus.inverse_event_shape}

Shape of a single sample from a single batch as an `int32` 1D `Tensor`.

##### Args:


*  <b>`output_shape`</b>: `Tensor`, `int32` vector indicating event-portion shape
    passed into `inverse` function.
*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`inverse_event_shape`</b>: `Tensor`, `int32` vector indicating event-portion
    shape after applying `inverse`.


- - -

#### `tf.contrib.distributions.bijector.Softplus.inverse_log_det_jacobian(y, name='inverse_log_det_jacobian', **condition_kwargs)` {#Softplus.inverse_log_det_jacobian}

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

#### `tf.contrib.distributions.bijector.Softplus.is_constant_jacobian` {#Softplus.is_constant_jacobian}

Returns true iff the Jacobian is not a function of x.

Note: Jacobian is either constant for both forward and inverse or neither.

##### Returns:

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Softplus.name` {#Softplus.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Softplus.parameters` {#Softplus.parameters}

Returns this `Bijector`'s parameters as a name/value dictionary.


- - -

#### `tf.contrib.distributions.bijector.Softplus.shaper` {#Softplus.shaper}

Returns shape object used to manage shape constraints.


- - -

#### `tf.contrib.distributions.bijector.Softplus.validate_args` {#Softplus.validate_args}

Returns True if Tensor arguments will be validated.



