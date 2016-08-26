A Transformed Distribution.

A Transformed Distribution models `p(y)` given a base distribution `p(x)`,
an invertible transform, `y = f(x)`, and the determinant of the Jacobian of
`f(x)`.

Shapes, type, and reparameterization are taken from the base distribution.

#### Mathematical details

* `p(x)` - probability distribution for random variable X
* `p(y)` - probability distribution for random variable Y
* `f` - transform
* `g` - inverse transform, `g(f(x)) = x`
* `J(x)` - Jacobian of f(x)

A Transformed Distribution exposes `sample` and `pdf`:

  * `sample`: `y = f(x)`, after drawing a sample of X.
  * `pdf`: `p(y) = p(x) / det|J(x)| = p(g(y)) / det|J(g(y))|`

A simple example constructing a Log-Normal distribution from a Normal
distribution:

```
logit_normal = TransformedDistribution(
  base_dist=Normal(mu, sigma),
  transform=lambda x: tf.sigmoid(x),
  inverse=lambda y: tf.log(y) - tf.log(1. - y),
  log_det_jacobian=(lambda x:
      tf.reduce_sum(tf.log(tf.sigmoid(x)) + tf.log(1. - tf.sigmoid(x)),
                    reduction_indices=[-1])))
  name="LogitNormalTransformedDistribution"
)
```
- - -

#### `tf.contrib.distributions.TransformedDistribution.__init__(base_dist_cls, transform, inverse, log_det_jacobian, name='TransformedDistribution', **base_dist_args)` {#TransformedDistribution.__init__}

Construct a Transformed Distribution.

##### Args:


*  <b>`base_dist_cls`</b>: the base distribution class to transform. Must be a
      subclass of `Distribution`.
*  <b>`transform`</b>: a callable that takes a `Tensor` sample from `base_dist` and
      returns a `Tensor` of the same shape and type. `x => y`.
*  <b>`inverse`</b>: a callable that computes the inverse of transform. `y => x`. If
      None, users can only call `log_pdf` on values returned by `sample`.
*  <b>`log_det_jacobian`</b>: a callable that takes a `Tensor` sample from `base_dist`
      and returns the log of the determinant of the Jacobian of `transform`.
*  <b>`name`</b>: The name for the distribution.
*  <b>`**base_dist_args`</b>: kwargs to pass on to dist_cls on construction.

##### Raises:


*  <b>`TypeError`</b>: if `base_dist_cls` is not a subclass of
      `Distribution`.


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

#### `tf.contrib.distributions.TransformedDistribution.base_distribution` {#TransformedDistribution.base_distribution}

Base distribution, p(x).


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

#### `tf.contrib.distributions.TransformedDistribution.cdf(value, name='cdf')` {#TransformedDistribution.cdf}

Cumulative distribution function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.dtype` {#TransformedDistribution.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.entropy(name='entropy')` {#TransformedDistribution.entropy}

Shanon entropy in nats.


- - -

#### `tf.contrib.distributions.TransformedDistribution.event_shape(name='event_shape')` {#TransformedDistribution.event_shape}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.from_params(cls, make_safe=True, **kwargs)` {#TransformedDistribution.from_params}

Given (unconstrained) parameters, return an instantiated distribution.

Subclasses should implement a static method `_safe_transforms` that returns
a dict of parameter transforms, which will be used if `make_safe = True`.

Example usage:

```
# Let's say we want a sample of size (batch_size, 10)
shapes = MultiVariateNormalDiag.param_shapes([batch_size, 10])

# shapes has a Tensor shape for mu and sigma
# shapes == {
#   "mu": tf.constant([batch_size, 10]),
#   "sigma": tf.constant([batch_size, 10]),
# }

# Here we parameterize mu and sigma with the output of a linear
# layer. Note that sigma is unconstrained.
params = {}
for name, shape in shapes.items():
  params[name] = linear(x, shape[1])

# Note that you can forward other kwargs to the `Distribution`, like
# `allow_nan_stats` or `name`.
mvn = MultiVariateNormalDiag.from_params(**params, allow_nan_stats=True)
```

Distribution parameters may have constraints (e.g. `sigma` must be positive
for a `Normal` distribution) and the `from_params` method will apply default
parameter transforms. If a user wants to use their own transform, they can
apply it externally and set `make_safe=False`.

##### Args:


*  <b>`make_safe`</b>: Whether the `params` should be constrained. If True,
    `from_params` will apply default parameter transforms. If False, no
    parameter transforms will be applied.
*  <b>`**kwargs`</b>: dict of parameters for the distribution.

##### Returns:

  A distribution parameterized by possibly transformed parameters in
  `kwargs`.

##### Raises:


*  <b>`TypeError`</b>: if `make_safe` is `True` but `_safe_transforms` is not
    implemented directly for `cls`.


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

#### `tf.contrib.distributions.TransformedDistribution.inverse` {#TransformedDistribution.inverse}

Inverse function of transform, y => x.


- - -

#### `tf.contrib.distributions.TransformedDistribution.is_continuous` {#TransformedDistribution.is_continuous}




- - -

#### `tf.contrib.distributions.TransformedDistribution.is_reparameterized` {#TransformedDistribution.is_reparameterized}




- - -

#### `tf.contrib.distributions.TransformedDistribution.log_cdf(value, name='log_cdf')` {#TransformedDistribution.log_cdf}

Log cumulative distribution function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`logcdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_det_jacobian` {#TransformedDistribution.log_det_jacobian}

Function computing the log determinant of the Jacobian of transform.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_pdf(value, name='log_pdf')` {#TransformedDistribution.log_pdf}

Log probability density function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`AttributeError`</b>: if not `is_continuous`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_pmf(value, name='log_pmf')` {#TransformedDistribution.log_pmf}

Log probability mass function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_pmf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`AttributeError`</b>: if `is_continuous`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.log_prob(value, name='log_prob')` {#TransformedDistribution.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


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

Dictionary of parameters used by this `Distribution`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.pdf(value, name='pdf')` {#TransformedDistribution.pdf}

Probability density function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`AttributeError`</b>: if not `is_continuous`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.pmf(value, name='pmf')` {#TransformedDistribution.pmf}

Probability mass function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`pmf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`AttributeError`</b>: if `is_continuous`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.prob(value, name='prob')` {#TransformedDistribution.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.sample(sample_shape=(), seed=None, name='sample')` {#TransformedDistribution.sample}

Generate samples of the specified shape.

Note that a call to `sample()` without arguments will generate a single
sample.

##### Args:


*  <b>`sample_shape`</b>: 0D or 1D `int32` `Tensor`. Shape of the generated samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` with prepended dimensions `sample_shape`.


- - -

#### `tf.contrib.distributions.TransformedDistribution.sample_n(n, seed=None, name='sample_n')` {#TransformedDistribution.sample_n}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: `Scalar` `Tensor` of type `int32` or `int64`, the number of
    observations to sample.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` with a prepended dimension (n,).

##### Raises:


*  <b>`TypeError`</b>: if `n` is not an integer type.


- - -

#### `tf.contrib.distributions.TransformedDistribution.std(name='std')` {#TransformedDistribution.std}

Standard deviation.


- - -

#### `tf.contrib.distributions.TransformedDistribution.transform` {#TransformedDistribution.transform}

Function transforming x => y.


- - -

#### `tf.contrib.distributions.TransformedDistribution.validate_args` {#TransformedDistribution.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.TransformedDistribution.variance(name='variance')` {#TransformedDistribution.variance}

Variance.


