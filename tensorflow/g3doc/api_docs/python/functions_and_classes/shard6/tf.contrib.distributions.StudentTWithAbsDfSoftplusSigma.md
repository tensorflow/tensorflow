StudentT with `df = floor(abs(df))` and `sigma = softplus(sigma)`.
- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.__init__(df, mu, sigma, validate_args=False, allow_nan_stats=True, name='StudentTWithAbsDfSoftplusSigma')` {#StudentTWithAbsDfSoftplusSigma.__init__}




- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.allow_nan_stats` {#StudentTWithAbsDfSoftplusSigma.allow_nan_stats}

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

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.batch_shape(name='batch_shape')` {#StudentTWithAbsDfSoftplusSigma.batch_shape}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.cdf(value, name='cdf')` {#StudentTWithAbsDfSoftplusSigma.cdf}

Cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
cdf(x) := P[X <= x]
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.df` {#StudentTWithAbsDfSoftplusSigma.df}

Degrees of freedom in these Student's t distribution(s).


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.dtype` {#StudentTWithAbsDfSoftplusSigma.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.entropy(name='entropy')` {#StudentTWithAbsDfSoftplusSigma.entropy}

Shanon entropy in nats.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.event_shape(name='event_shape')` {#StudentTWithAbsDfSoftplusSigma.event_shape}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.get_batch_shape()` {#StudentTWithAbsDfSoftplusSigma.get_batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.get_event_shape()` {#StudentTWithAbsDfSoftplusSigma.get_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.is_continuous` {#StudentTWithAbsDfSoftplusSigma.is_continuous}




- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.is_reparameterized` {#StudentTWithAbsDfSoftplusSigma.is_reparameterized}




- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.log_cdf(value, name='log_cdf')` {#StudentTWithAbsDfSoftplusSigma.log_cdf}

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

##### Returns:


*  <b>`logcdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.log_pdf(value, name='log_pdf')` {#StudentTWithAbsDfSoftplusSigma.log_pdf}

Log probability density function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if not `is_continuous`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.log_pmf(value, name='log_pmf')` {#StudentTWithAbsDfSoftplusSigma.log_pmf}

Log probability mass function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_pmf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if `is_continuous`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.log_prob(value, name='log_prob')` {#StudentTWithAbsDfSoftplusSigma.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.log_survival_function(value, name='log_survival_function')` {#StudentTWithAbsDfSoftplusSigma.log_survival_function}

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

##### Returns:

  `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.mean(name='mean')` {#StudentTWithAbsDfSoftplusSigma.mean}

Mean.

Additional documentation from `StudentT`:

The mean of Student's T equals `mu` if `df > 1`, otherwise it is `NaN`.
If `self.allow_nan_stats=True`, then an exception will be raised rather
than returning `NaN`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.mode(name='mode')` {#StudentTWithAbsDfSoftplusSigma.mode}

Mode.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.mu` {#StudentTWithAbsDfSoftplusSigma.mu}

Locations of these Student's t distribution(s).


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.name` {#StudentTWithAbsDfSoftplusSigma.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#StudentTWithAbsDfSoftplusSigma.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.param_static_shapes(cls, sample_shape)` {#StudentTWithAbsDfSoftplusSigma.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.parameters` {#StudentTWithAbsDfSoftplusSigma.parameters}

Dictionary of parameters used by this `Distribution`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.pdf(value, name='pdf')` {#StudentTWithAbsDfSoftplusSigma.pdf}

Probability density function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if not `is_continuous`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.pmf(value, name='pmf')` {#StudentTWithAbsDfSoftplusSigma.pmf}

Probability mass function.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`pmf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.

##### Raises:


*  <b>`TypeError`</b>: if `is_continuous`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.prob(value, name='prob')` {#StudentTWithAbsDfSoftplusSigma.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.sample(sample_shape=(), seed=None, name='sample')` {#StudentTWithAbsDfSoftplusSigma.sample}

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

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.sample_n(n, seed=None, name='sample_n')` {#StudentTWithAbsDfSoftplusSigma.sample_n}

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

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.sigma` {#StudentTWithAbsDfSoftplusSigma.sigma}

Scaling factors of these Student's t distribution(s).


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.std(name='std')` {#StudentTWithAbsDfSoftplusSigma.std}

Standard deviation.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.survival_function(value, name='survival_function')` {#StudentTWithAbsDfSoftplusSigma.survival_function}

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

##### Returns:

  Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.validate_args` {#StudentTWithAbsDfSoftplusSigma.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.StudentTWithAbsDfSoftplusSigma.variance(name='variance')` {#StudentTWithAbsDfSoftplusSigma.variance}

Variance.

Additional documentation from `StudentT`:

The variance for Student's T equals

```
df / (df - 2), when df > 2
infinity, when 1 < df <= 2
NaN, when df <= 1
```


