Exponential with softplus transform on `lam`.
- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.__init__(lam, validate_args=False, allow_nan_stats=True, name='ExponentialWithSoftplusLam')` {#ExponentialWithSoftplusLam.__init__}




- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.allow_nan_stats` {#ExponentialWithSoftplusLam.allow_nan_stats}

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

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.alpha` {#ExponentialWithSoftplusLam.alpha}

Shape parameter.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.batch_shape(name='batch_shape')` {#ExponentialWithSoftplusLam.batch_shape}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.beta` {#ExponentialWithSoftplusLam.beta}

Inverse scale parameter.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.cdf(value, name='cdf', **condition_kwargs)` {#ExponentialWithSoftplusLam.cdf}

Cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
cdf(x) := P[X <= x]
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`cdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.copy(**override_parameters_kwargs)` {#ExponentialWithSoftplusLam.copy}

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

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.dtype` {#ExponentialWithSoftplusLam.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.entropy(name='entropy')` {#ExponentialWithSoftplusLam.entropy}

Shannon entropy in nats.

Additional documentation from `Gamma`:

This is defined to be

```
entropy = alpha - log(beta) + log(Gamma(alpha))
+ (1-alpha)digamma(alpha)
```

where digamma(alpha) is the digamma function.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.event_shape(name='event_shape')` {#ExponentialWithSoftplusLam.event_shape}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.get_batch_shape()` {#ExponentialWithSoftplusLam.get_batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.get_event_shape()` {#ExponentialWithSoftplusLam.get_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.is_continuous` {#ExponentialWithSoftplusLam.is_continuous}




- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.is_reparameterized` {#ExponentialWithSoftplusLam.is_reparameterized}




- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.is_scalar_batch(name='is_scalar_batch')` {#ExponentialWithSoftplusLam.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `Boolean` `scalar` `Tensor`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.is_scalar_event(name='is_scalar_event')` {#ExponentialWithSoftplusLam.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `Boolean` `scalar` `Tensor`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.lam` {#ExponentialWithSoftplusLam.lam}




- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.log_cdf(value, name='log_cdf', **condition_kwargs)` {#ExponentialWithSoftplusLam.log_cdf}

Log cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
log_cdf(x) := Log[ P[X <= x] ]
```

Often, a numerical approximation can be used for `log_cdf(x)` that yields
a more accurate answer than simply taking the logarithm of the `cdf` when
`x << -1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`logcdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.log_pdf(value, name='log_pdf', **condition_kwargs)` {#ExponentialWithSoftplusLam.log_pdf}

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

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.log_pmf(value, name='log_pmf', **condition_kwargs)` {#ExponentialWithSoftplusLam.log_pmf}

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

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.log_prob(value, name='log_prob', **condition_kwargs)` {#ExponentialWithSoftplusLam.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.log_survival_function(value, name='log_survival_function', **condition_kwargs)` {#ExponentialWithSoftplusLam.log_survival_function}

Log survival function.

Given random variable `X`, the survival function is defined:

```
log_survival_function(x) = Log[ P[X > x] ]
                         = Log[ 1 - P[X <= x] ]
                         = Log[ 1 - cdf(x) ]
```

Typically, different numerical approximations can be used for the log
survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.mean(name='mean')` {#ExponentialWithSoftplusLam.mean}

Mean.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.mode(name='mode')` {#ExponentialWithSoftplusLam.mode}

Mode.

Additional documentation from `Gamma`:

The mode of a gamma distribution is `(alpha - 1) / beta` when
`alpha > 1`, and `NaN` otherwise.  If `self.allow_nan_stats` is `False`,
an exception will be raised rather than returning `NaN`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.name` {#ExponentialWithSoftplusLam.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#ExponentialWithSoftplusLam.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.param_static_shapes(cls, sample_shape)` {#ExponentialWithSoftplusLam.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.parameters` {#ExponentialWithSoftplusLam.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.pdf(value, name='pdf', **condition_kwargs)` {#ExponentialWithSoftplusLam.pdf}

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

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.pmf(value, name='pmf', **condition_kwargs)` {#ExponentialWithSoftplusLam.pmf}

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

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.prob(value, name='prob', **condition_kwargs)` {#ExponentialWithSoftplusLam.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.sample(sample_shape=(), seed=None, name='sample', **condition_kwargs)` {#ExponentialWithSoftplusLam.sample}

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

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.std(name='std')` {#ExponentialWithSoftplusLam.std}

Standard deviation.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.survival_function(value, name='survival_function', **condition_kwargs)` {#ExponentialWithSoftplusLam.survival_function}

Survival function.

Given random variable `X`, the survival function is defined:

```
survival_function(x) = P[X > x]
                     = 1 - P[X <= x]
                     = 1 - cdf(x).
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.
*  <b>`**condition_kwargs`</b>: Named arguments forwarded to subclass implementation.

##### Returns:

  Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.validate_args` {#ExponentialWithSoftplusLam.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.ExponentialWithSoftplusLam.variance(name='variance')` {#ExponentialWithSoftplusLam.variance}

Variance.


