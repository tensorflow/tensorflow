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
          return math_ops.reduce_sum(x, reduction_indices=event_dims)
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
  to implement all of `_forward` and `_inverse` functions.  Examples:
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

- Subclasses should implement `_forward_event_shape`,
  `_forward_event_shape_tensor` (and `inverse` counterparts) if the
  transformation is shape-changing.  By default the event-shape is assumed
  unchanged from input.

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

  `Boolean`.


- - -

#### `tf.contrib.distributions.bijector.Bijector.name` {#Bijector.name}

Returns the string name of this `Bijector`.


- - -

#### `tf.contrib.distributions.bijector.Bijector.validate_args` {#Bijector.validate_args}

Returns True if Tensor arguments will be validated.


