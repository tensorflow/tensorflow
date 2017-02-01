MultivariateNormalDiag with `diag_stddev = softplus(diag_stddev)`.
- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.__init__(mu, diag_stddev, validate_args=False, allow_nan_stats=True, name='MultivariateNormalDiagWithSoftplusStdDev')` {#MultivariateNormalDiagWithSoftplusStDev.__init__}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.allow_nan_stats` {#MultivariateNormalDiagWithSoftplusStDev.allow_nan_stats}

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

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.batch_shape` {#MultivariateNormalDiagWithSoftplusStDev.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.batch_shape_tensor(name='batch_shape_tensor')` {#MultivariateNormalDiagWithSoftplusStDev.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.cdf(value, name='cdf')` {#MultivariateNormalDiagWithSoftplusStDev.cdf}

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

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.copy(**override_parameters_kwargs)` {#MultivariateNormalDiagWithSoftplusStDev.copy}

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

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.covariance(name='covariance')` {#MultivariateNormalDiagWithSoftplusStDev.covariance}

Covariance.

Covariance is (possibly) defined only for non-scalar-event distributions.

For example, for a length-`k`, vector-valued distribution, it is calculated
as,

```none
Cov[i, j] = Covariance(X_i, X_j) = E[(X_i - E[X_i]) (X_j - E[X_j])]
```

where `Cov` is a (batch of) `k x k` matrix, `0 <= (i, j) < k`, and `E`
denotes expectation.

Alternatively, for non-vector, multivariate distributions (e.g.,
matrix-valued, Wishart), `Covariance` shall return a (batch of) matrices
under some vectorization of the events, i.e.,

```none
Cov[i, j] = Covariance(Vec(X)_i, Vec(X)_j) = [as above]
````

where `Cov` is a (batch of) `k' x k'` matrices,
`0 <= (i, j) < k' = reduce_prod(event_shape)`, and `Vec` is some function
mapping indices of this distribution's event dimensions to indices of a
length-`k'` vector.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`covariance`</b>: Floating-point `Tensor` with shape `[B1, ..., Bn, k', k']`
    where the first `n` dimensions are batch coordinates and
    `k' = reduce_prod(self.event_shape)`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.dtype` {#MultivariateNormalDiagWithSoftplusStDev.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.entropy(name='entropy')` {#MultivariateNormalDiagWithSoftplusStDev.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.event_shape` {#MultivariateNormalDiagWithSoftplusStDev.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.event_shape_tensor(name='event_shape_tensor')` {#MultivariateNormalDiagWithSoftplusStDev.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.is_continuous` {#MultivariateNormalDiagWithSoftplusStDev.is_continuous}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.is_scalar_batch(name='is_scalar_batch')` {#MultivariateNormalDiagWithSoftplusStDev.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `Boolean` `scalar` `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.is_scalar_event(name='is_scalar_event')` {#MultivariateNormalDiagWithSoftplusStDev.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `Boolean` `scalar` `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.log_cdf(value, name='log_cdf')` {#MultivariateNormalDiagWithSoftplusStDev.log_cdf}

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

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.log_prob(value, name='log_prob')` {#MultivariateNormalDiagWithSoftplusStDev.log_prob}

Log probability density/mass function (depending on `is_continuous`).


Additional documentation from `_MultivariateNormalOperatorPD`:

`x` is a batch vector with compatible shape if `x` is a `Tensor` whose
shape can be broadcast up to either:

```
self.batch_shape + self.event_shape
```

or

```
[M1,...,Mm] + self.batch_shape + self.event_shape
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.log_sigma_det(name='log_sigma_det')` {#MultivariateNormalDiagWithSoftplusStDev.log_sigma_det}

Log of determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.log_survival_function(value, name='log_survival_function')` {#MultivariateNormalDiagWithSoftplusStDev.log_survival_function}

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

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.mean(name='mean')` {#MultivariateNormalDiagWithSoftplusStDev.mean}

Mean.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.mode(name='mode')` {#MultivariateNormalDiagWithSoftplusStDev.mode}

Mode.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.mu` {#MultivariateNormalDiagWithSoftplusStDev.mu}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.name` {#MultivariateNormalDiagWithSoftplusStDev.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#MultivariateNormalDiagWithSoftplusStDev.param_shapes}

Shapes of parameters given the desired shape of a call to `sample()`.

This is a class method that describes what key/value arguments are required
to instantiate the given `Distribution` so that a particular shape is
returned for that instance's call to `sample()`.

Subclasses should override class method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.param_static_shapes(cls, sample_shape)` {#MultivariateNormalDiagWithSoftplusStDev.param_static_shapes}

param_shapes with static (i.e. `TensorShape`) shapes.

This is a class method that describes what key/value arguments are required
to instantiate the given `Distribution` so that a particular shape is
returned for that instance's call to `sample()`.  Assumes that
the sample's shape is known statically.

Subclasses should override class method `_param_shapes` to return
constant-valued tensors when constant values are fed.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.parameters` {#MultivariateNormalDiagWithSoftplusStDev.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.prob(value, name='prob')` {#MultivariateNormalDiagWithSoftplusStDev.prob}

Probability density/mass function (depending on `is_continuous`).


Additional documentation from `_MultivariateNormalOperatorPD`:

`x` is a batch vector with compatible shape if `x` is a `Tensor` whose
shape can be broadcast up to either:

```
self.batch_shape + self.event_shape
```

or

```
[M1,...,Mm] + self.batch_shape + self.event_shape
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.reparameterization_type` {#MultivariateNormalDiagWithSoftplusStDev.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.sample(sample_shape=(), seed=None, name='sample')` {#MultivariateNormalDiagWithSoftplusStDev.sample}

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

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.sigma` {#MultivariateNormalDiagWithSoftplusStDev.sigma}

Dense (batch) covariance matrix, if available.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.sigma_det(name='sigma_det')` {#MultivariateNormalDiagWithSoftplusStDev.sigma_det}

Determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.stddev(name='stddev')` {#MultivariateNormalDiagWithSoftplusStDev.stddev}

Standard deviation.

Standard deviation is defined as,

```none
stddev = E[(X - E[X])**2]**0.5
```

where `X` is the random variable associated with this distribution, `E`
denotes expectation, and `stddev.shape = batch_shape + event_shape`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`stddev`</b>: Floating-point `Tensor` with shape identical to
    `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.survival_function(value, name='survival_function')` {#MultivariateNormalDiagWithSoftplusStDev.survival_function}

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

  `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.validate_args` {#MultivariateNormalDiagWithSoftplusStDev.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiagWithSoftplusStDev.variance(name='variance')` {#MultivariateNormalDiagWithSoftplusStDev.variance}

Variance.

Variance is defined as,

```none
Var = E[(X - E[X])**2]
```

where `X` is the random variable associated with this distribution, `E`
denotes expectation, and `Var.shape = batch_shape + event_shape`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`variance`</b>: Floating-point `Tensor` with shape identical to
    `batch_shape + event_shape`, i.e., the same shape as `self.mean()`.


