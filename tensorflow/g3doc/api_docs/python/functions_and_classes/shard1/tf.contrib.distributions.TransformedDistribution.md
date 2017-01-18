A Transformed Distribution.

A `TransformedDistribution` models `p(y)` given a base distribution `p(x)`,
and a deterministic, invertible, differentiable transform, `Y = g(X)`. The
transform is typically an instance of the `Bijector` class and the base
distribution is typically an instance of the `Distribution` class.

A `Bijector` is expected to implement the following functions:
- `forward`,
- `inverse`,
- `inverse_log_det_jacobian`.
The semantics of these functions are outlined in the `Bijector` documentation.

We now describe how a `TransformedDistribution` alters the input/outputs of a
`Distribution` associated with a random variable (rv) `X`.

Write `cdf(Y=y)` for an absolutely continuous cumulative distribution function
of random variable `Y`; write the probability density function `pdf(Y=y) :=
d^k / (dy_1,...,dy_k) cdf(Y=y)` for its derivative wrt to `Y` evaluated at
`y`.  Assume that `Y = g(X)` where `g` is a deterministic diffeomorphism,
i.e., a non-random, continuous, differentiable, and invertible function.
Write the inverse of `g` as `X = g^{-1}(Y)` and `(J o g)(x)` for the Jacobian
of `g` evaluated at `x`.

A `TransformedDistribution` implements the following operations:

  * `sample`:

    Mathematically:

    ```none
    Y = g(X)
    ```

    Programmatically:

    ```python
    return bijector.forward(distribution.sample(...))
    ```

  * `log_prob`:

    Mathematically:

    ```none
    (log o pdf)(Y=y) = (log o pdf o g^{-1})(y) +
                         (log o abs o det o J o g^{-1})(y)
    ```

    Programmatically:

    ```python
    return (distribution.log_prob(bijector.inverse(x)) +
            bijector.inverse_log_det_jacobian(x))
    ```

  * `log_cdf`:

    Mathematically:

    ```none
    (log o cdf)(Y=y) = (log o cdf o g^{-1})(y)
    ```

    Programmatically:

    ```python
    return distribution.log_cdf(bijector.inverse(x))
    ```

  * and similarly for: `cdf`, `prob`, `log_survival_function`,
   `survival_function`.

A simple example constructing a Log-Normal distribution from a Normal
distribution:

```python
ds = tf.contrib.distributions
log_normal = ds.TransformedDistribution(
  distribution=ds.Normal(mu=mu, sigma=sigma),
  bijector=ds.bijector.Exp(),
  name="LogNormalTransformedDistribution")
```

A `LogNormal` made from callables:

```python
ds = tf.contrib.distributions
log_normal = ds.TransformedDistribution(
  distribution=ds.Normal(mu=mu, sigma=sigma),
  bijector=ds.bijector.Inline(
    forward_fn=tf.exp,
    inverse_fn=tf.log,
    inverse_log_det_jacobian_fn=(
      lambda y: -tf.reduce_sum(tf.log(y), reduction_indices=-1)),
  name="LogNormalTransformedDistribution")
```

Another example constructing a Normal from a StandardNormal:

```python
ds = tf.contrib.distributions
normal = ds.TransformedDistribution(
  distribution=ds.Normal(mu=0, sigma=1),
  bijector=ds.bijector.ScaleAndShift(loc=mu, scale=sigma, event_ndims=0),
  name="NormalTransformedDistribution")
```

A `TransformedDistribution`'s batch- and event-shape are implied by the base
distribution unless explicitly overridden by `batch_shape` or `event_shape`
arguments.  Specifying an overriding `batch_shape` (`event_shape`) is
permitted only if the base distribution has scalar batch-shape (event-shape).
The bijector is applied to the distribution as if the distribution possessed
the overridden shape(s). The following example demonstrates how to construct a
multivariate Normal as a `TransformedDistribution`.

```python
bs = tf.contrib.distributions.bijector
ds = tf.contrib.distributions
# We will create two MVNs with batch_shape = event_shape = 2.
mean = [[-1., 0],      # batch:0
        [0., 1]]       # batch:1
chol_cov = [[[1., 0],
             [0, 1]],  # batch:0
            [[1, 0],
             [2, 2]]]  # batch:1
mvn1 = ds.TransformedDistribution(
    distribution=ds.Normal(mu=0., sigma=1.),
    bijector=bs.Affine(shift=mean, tril=chol_cov),
    batch_shape=[2],  # Valid because base_distribution.batch_shape == [].
    event_shape=[2])  # Valid because base_distribution.event_shape == [].
mvn2 = ds.MultivariateNormalCholesky(mu=mean, chol=chol_cov)
# mvn1.log_prob(x) == mvn2.log_prob(x)
```
- - -

#### `tf.contrib.distributions.TransformedDistribution.__init__(distribution, bijector=None, batch_shape=None, event_shape=None, validate_args=False, name=None)` {#TransformedDistribution.__init__}

Construct a Transformed Distribution.

##### Args:


*  <b>`distribution`</b>: The base distribution instance to transform. Typically an
    instance of `Distribution`.
*  <b>`bijector`</b>: The object responsible for calculating the transformation.
    Typically an instance of `Bijector`. `None` means `Identity()`.
*  <b>`batch_shape`</b>: `integer` vector `Tensor` which overrides `distribution`
    `batch_shape`; valid only if `distribution.is_scalar_batch()`.
*  <b>`event_shape`</b>: `integer` vector `Tensor` which overrides `distribution`
    `event_shape`; valid only if `distribution.is_scalar_event()`.
*  <b>`validate_args`</b>: Python Boolean.  Whether to validate input with asserts.
    If `validate_args` is `False`, and the inputs are invalid,
    correct behavior is not guaranteed.
*  <b>`name`</b>: The name for the distribution. Default:
    `bijector.name + distribution.name`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.allow_nan_stats` {#TransformedDistribution.allow_nan_stats}

Python boolean describing behavior when a stat is undefined.

Stats return +/- infinity when it makes sense.  E.g., the variance
of a Cauchy distribution is infinity.  However, sometimes the
statistic is undefined, e.g., if a distribution's pdf does not achieve a
maximum within the support of the distribution, the mode is undefined.
If the mean is undefined, then by definition the variance is undefined.
E.g. the mean for Student's T for df = 1 is undefined (no clear way to say
it is either + or - infinity), so the variance = E[(X - mean)^2] is also
undefined.

##### Returns:


*  <b>`allow_nan_stats`</b>: Python boolean.


- - -

#### `tf.contrib.distributions.TransformedDistribution.batch_shape(name='batch_shape')` {#TransformedDistribution.batch_shape}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.bijector` {#TransformedDistribution.bijector}

Function transforming x => y.


- - -

#### `tf.contrib.distributions.TransformedDistribution.cdf(value, name='cdf', **condition_kwargs)` {#TransformedDistribution.cdf}

Cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
cdf(x) := P[X <= x]
```


Additional documentation from `TransformedDistribution`:

##### `condition_kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`cdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.copy(**override_parameters_kwargs)` {#TransformedDistribution.copy}

Creates a deep copy of the distribution.

Note: the copy distribution may continue to depend on the original
intialization arguments.

##### Args:


*  <b>`**override_parameters_kwargs`</b>: String/value dictionary of initialization
    arguments to override with new values.

##### Returns:


*  <b>`distribution`</b>: A new instance of `type(self)` intitialized from the union
    of self.parameters and override_parameters_kwargs, i.e.,
    `dict(self.parameters, **override_parameters_kwargs)`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.distribution` {#TransformedDistribution.distribution}

Base distribution, p(x).


- - -

#### `tf.contrib.distributions.TransformedDistribution.dtype` {#TransformedDistribution.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.entropy(name='entropy')` {#TransformedDistribution.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.TransformedDistribution.event_shape(name='event_shape')` {#TransformedDistribution.event_shape}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.get_batch_shape()` {#TransformedDistribution.get_batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.TransformedDistribution.get_event_shape()` {#TransformedDistribution.get_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.TransformedDistribution.is_continuous` {#TransformedDistribution.is_continuous}




- - -

#### `tf.contrib.distributions.TransformedDistribution.is_reparameterized` {#TransformedDistribution.is_reparameterized}




- - -

#### `tf.contrib.distributions.TransformedDistribution.is_scalar_batch(name='is_scalar_batch')` {#TransformedDistribution.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `Boolean` `scalar` `Tensor`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.is_scalar_event(name='is_scalar_event')` {#TransformedDistribution.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `Boolean` `scalar` `Tensor`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_cdf(value, name='log_cdf', **condition_kwargs)` {#TransformedDistribution.log_cdf}

Log cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
log_cdf(x) := Log[ P[X <= x] ]
```

Often, a numerical approximation can be used for `log_cdf(x)` that yields
a more accurate answer than simply taking the logarithm of the `cdf` when
`x << -1`.


Additional documentation from `TransformedDistribution`:

##### `condition_kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`logcdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_pdf(value, name='log_pdf', **condition_kwargs)` {#TransformedDistribution.log_pdf}

Log probability density function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if not `is_continuous`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_pmf(value, name='log_pmf', **condition_kwargs)` {#TransformedDistribution.log_pmf}

Log probability mass function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`log_pmf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if `is_continuous`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_prob(value, name='log_prob', **condition_kwargs)` {#TransformedDistribution.log_prob}

Log probability density/mass function (depending on `is_continuous`).


Additional documentation from `TransformedDistribution`:

Implements `(log o p o g^{-1})(y) + (log o abs o det o J o g^{-1})(y)`,
      where `g^{-1}` is the inverse of `transform`.

      Also raises a `ValueError` if `inverse` was not provided to the
      distribution and `y` was not returned from `sample`.

##### `condition_kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_survival_function(value, name='log_survival_function', **condition_kwargs)` {#TransformedDistribution.log_survival_function}

Log survival function.

Given random variable `X`, the survival function is defined:

```
log_survival_function(x) = Log[ P[X > x] ]
                         = Log[ 1 - P[X <= x] ]
                         = Log[ 1 - cdf(x) ]
```

Typically, different numerical approximations can be used for the log
survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.


Additional documentation from `TransformedDistribution`:

##### `condition_kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.mean(name='mean')` {#TransformedDistribution.mean}

Mean.


- - -

#### `tf.contrib.distributions.TransformedDistribution.mode(name='mode')` {#TransformedDistribution.mode}

Mode.


- - -

#### `tf.contrib.distributions.TransformedDistribution.name` {#TransformedDistribution.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#TransformedDistribution.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.TransformedDistribution.param_static_shapes(cls, sample_shape)` {#TransformedDistribution.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.TransformedDistribution.parameters` {#TransformedDistribution.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.pdf(value, name='pdf', **condition_kwargs)` {#TransformedDistribution.pdf}

Probability density function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if not `is_continuous`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.pmf(value, name='pmf', **condition_kwargs)` {#TransformedDistribution.pmf}

Probability mass function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`pmf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if `is_continuous`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.prob(value, name='prob', **condition_kwargs)` {#TransformedDistribution.prob}

Probability density/mass function (depending on `is_continuous`).


Additional documentation from `TransformedDistribution`:

Implements `p(g^{-1}(y)) det|J(g^{-1}(y))|`, where `g^{-1}` is the
      inverse of `transform`.

      Also raises a `ValueError` if `inverse` was not provided to the
      distribution and `y` was not returned from `sample`.

##### `condition_kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.sample(sample_shape=(), seed=None, name='sample', **condition_kwargs)` {#TransformedDistribution.sample}

Generate samples of the specified shape.

Note that a call to `sample()` without arguments will generate a single
sample.

##### Args:


*  <b>`sample_shape`</b>: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`samples`</b>: a `Tensor` with prepended dimensions `sample_shape`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.std(name='std')` {#TransformedDistribution.std}

Standard deviation.


- - -

#### `tf.contrib.distributions.TransformedDistribution.survival_function(value, name='survival_function', **condition_kwargs)` {#TransformedDistribution.survival_function}

Survival function.

Given random variable `X`, the survival function is defined:

```
survival_function(x) = P[X > x]
                     = 1 - P[X <= x]
                     = 1 - cdf(x).
```


Additional documentation from `TransformedDistribution`:

##### `condition_kwargs`:

*  `bijector_kwargs`: Python dictionary of arg names/values forwarded to the bijector.
*  `distribution_kwargs`: Python dictionary of arg names/values forwarded to the distribution.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.validate_args` {#TransformedDistribution.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.TransformedDistribution.variance(name='variance')` {#TransformedDistribution.variance}

Variance.


