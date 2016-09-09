Chi2 with parameter transform `df = floor(abs(df))`.
- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.__init__(df, validate_args=False, allow_nan_stats=True, name='Chi2WithAbsDf')` {#Chi2WithAbsDf.__init__}




- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.allow_nan_stats` {#Chi2WithAbsDf.allow_nan_stats}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.alpha` {#Chi2WithAbsDf.alpha}

Shape parameter.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.batch_shape(name='batch_shape')` {#Chi2WithAbsDf.batch_shape}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.beta` {#Chi2WithAbsDf.beta}

Inverse scale parameter.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.cdf(value, name='cdf')` {#Chi2WithAbsDf.cdf}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.df` {#Chi2WithAbsDf.df}




- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.dtype` {#Chi2WithAbsDf.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.entropy(name='entropy')` {#Chi2WithAbsDf.entropy}

Shanon entropy in nats.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.event_shape(name='event_shape')` {#Chi2WithAbsDf.event_shape}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.get_batch_shape()` {#Chi2WithAbsDf.get_batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.get_event_shape()` {#Chi2WithAbsDf.get_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.is_continuous` {#Chi2WithAbsDf.is_continuous}




- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.is_reparameterized` {#Chi2WithAbsDf.is_reparameterized}




- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.log_cdf(value, name='log_cdf')` {#Chi2WithAbsDf.log_cdf}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.log_pdf(value, name='log_pdf')` {#Chi2WithAbsDf.log_pdf}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.log_pmf(value, name='log_pmf')` {#Chi2WithAbsDf.log_pmf}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.log_prob(value, name='log_prob')` {#Chi2WithAbsDf.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.log_survival_function(value, name='log_survival_function')` {#Chi2WithAbsDf.log_survival_function}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.mean(name='mean')` {#Chi2WithAbsDf.mean}

Mean.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.mode(name='mode')` {#Chi2WithAbsDf.mode}

Mode.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.name` {#Chi2WithAbsDf.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#Chi2WithAbsDf.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.param_static_shapes(cls, sample_shape)` {#Chi2WithAbsDf.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.parameters` {#Chi2WithAbsDf.parameters}

Dictionary of parameters used by this `Distribution`.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.pdf(value, name='pdf')` {#Chi2WithAbsDf.pdf}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.pmf(value, name='pmf')` {#Chi2WithAbsDf.pmf}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.prob(value, name='prob')` {#Chi2WithAbsDf.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.sample(sample_shape=(), seed=None, name='sample')` {#Chi2WithAbsDf.sample}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.sample_n(n, seed=None, name='sample_n')` {#Chi2WithAbsDf.sample_n}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.std(name='std')` {#Chi2WithAbsDf.std}

Standard deviation.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.survival_function(value, name='survival_function')` {#Chi2WithAbsDf.survival_function}

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

#### `tf.contrib.distributions.Chi2WithAbsDf.validate_args` {#Chi2WithAbsDf.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.Chi2WithAbsDf.variance(name='variance')` {#Chi2WithAbsDf.variance}

Variance.


