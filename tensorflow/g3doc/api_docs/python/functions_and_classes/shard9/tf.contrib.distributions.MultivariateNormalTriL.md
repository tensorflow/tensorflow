The multivariate normal distribution on `R^k`.

The Multivariate Normal distribution is defined over `R^k` and parameterized
by a (batch of) length-`k` `loc` vector (aka "mu") and a (batch of) `k x k`
`scale` matrix; `covariance = scale @ scale.T` where `@` denotes
matrix-multiplication.

#### Mathematical Details

The probability density function (pdf) is,

```none
pdf(x; loc, scale) = exp(-0.5 ||y||**2) / Z,
y = inv(scale) @ (x - loc),
Z = (2 pi)**(0.5 k) |det(scale)|,
```

where:

* `loc` is a vector in `R^k`,
* `scale` is a linear operator in `R^{k x k}`, `cov = scale @ scale.T`,
* `Z` denotes the normalization constant, and,
* `||y||**2` denotes the squared Euclidean norm of `y`.

A (non-batch) `scale` matrix is:

```none
scale = scale_tril
```

where `scale_tril` is lower-triangular `k x k` matrix with non-zero diagonal,
i.e., `tf.diag_part(scale_tril) != 0`.

Additional leading dimensions (if any) will index batches.

The MultivariateNormal distribution is a member of the [location-scale
family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
constructed as,

```none
X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
Y = scale @ X + loc
```

Trainable (batch) lower-triangular matrices can be created with
`ds.matrix_diag_transform()` and/or `ds.fill_lower_triangular()`

#### Examples

```python
ds = tf.contrib.distributions

# Initialize a single 3-variate Gaussian.
mu = [1., 2, 3]
cov = [[ 0.36,  0.12,  0.06],
       [ 0.12,  0.29, -0.13],
       [ 0.06, -0.13,  0.26]]
scale = tf.cholesky(cov)
# ==> [[ 0.6,  0. ,  0. ],
#      [ 0.2,  0.5,  0. ],
#      [ 0.1, -0.3,  0.4]])
mvn = ds.MultivariateNormalTriL(
    loc=mu,
    scale_tril=scale)

mvn.mean().eval()
# ==> [1., 2, 3]

# Covariance agrees with cholesky(cov) parameterization.
mvn.covariance().eval()
# ==> [[ 0.36,  0.12,  0.06],
#      [ 0.12,  0.29, -0.13],
#      [ 0.06, -0.13,  0.26]]

# Compute the pdf of an observation in `R^3` ; return a scalar.
mvn.prob([-1., 0, 1]).eval()  # shape: []

# Initialize a 2-batch of 3-variate Gaussians.
mu = [[1., 2, 3],
      [11, 22, 33]]              # shape: [2, 3]
tril = ...  # shape: [2, 3, 3], lower triangular, non-zero diagonal.
mvn = ds.MultivariateNormalTriL(
    loc=mu,
    scale_tril=tril)

# Compute the pdf of two `R^3` observations; return a length-2 vector.
x = [[-0.9, 0, 0.1],
     [-10, 0, 9]]     # shape: [2, 3]
mvn.prob(x).eval()    # shape: [2]

```
- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.__init__(loc=None, scale_tril=None, validate_args=False, allow_nan_stats=True, name='MultivariateNormalTriL')` {#MultivariateNormalTriL.__init__}

Construct Multivariate Normal distribution on `R^k`.

The `batch_shape` is the broadcast shape between `loc` and `scale`
arguments.

The `event_shape` is given by the last dimension of `loc` or the last
dimension of the matrix implied by `scale`.

Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix is:

```none
scale = scale_tril
```

where `scale_tril` is lower-triangular `k x k` matrix with non-zero
diagonal, i.e., `tf.diag_part(scale_tril) != 0`.

Additional leading dimensions (if any) will index batches.

##### Args:


*  <b>`loc`</b>: Floating-point `Tensor`. If this is set to `None`, `loc` is
    implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
    `b >= 0` and `k` is the event size.
*  <b>`scale_tril`</b>: Floating-point, lower-triangular `Tensor` with non-zero
    diagonal elements. `scale_tril` has shape `[B1, ..., Bb, k, k]` where
    `b >= 0` and `k` is the event size.
*  <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
*  <b>`allow_nan_stats`</b>: Python `bool`, default `True`. When `True`,
    statistics (e.g., mean, mode, variance) use the value "`NaN`" to
    indicate the result is undefined. When `False`, an exception is raised
    if one or more of the statistic's batch members are undefined.
*  <b>`name`</b>: Python `str` name prefixed to Ops created by this class.

##### Raises:


*  <b>`ValueError`</b>: if neither `loc` nor `scale_tril` are specified.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.allow_nan_stats` {#MultivariateNormalTriL.allow_nan_stats}

Python `bool` describing behavior when a stat is undefined.

Stats return +/- infinity when it makes sense. E.g., the variance of a
Cauchy distribution is infinity. However, sometimes the statistic is
undefined, e.g., if a distribution's pdf does not achieve a maximum within
the support of the distribution, the mode is undefined. If the mean is
undefined, then by definition the variance is undefined. E.g. the mean for
Student's T for df = 1 is undefined (no clear way to say it is either + or -
infinity), so the variance = E[(X - mean)**2] is also undefined.

##### Returns:


*  <b>`allow_nan_stats`</b>: Python `bool`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.batch_shape` {#MultivariateNormalTriL.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.batch_shape_tensor(name='batch_shape_tensor')` {#MultivariateNormalTriL.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.bijector` {#MultivariateNormalTriL.bijector}

Function transforming x => y.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.cdf(value, name='cdf')` {#MultivariateNormalTriL.cdf}

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

#### `tf.contrib.distributions.MultivariateNormalTriL.copy(**override_parameters_kwargs)` {#MultivariateNormalTriL.copy}

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

#### `tf.contrib.distributions.MultivariateNormalTriL.covariance(name='covariance')` {#MultivariateNormalTriL.covariance}

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

#### `tf.contrib.distributions.MultivariateNormalTriL.det_covariance(name='det_covariance')` {#MultivariateNormalTriL.det_covariance}

Determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.distribution` {#MultivariateNormalTriL.distribution}

Base distribution, p(x).


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.dtype` {#MultivariateNormalTriL.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.entropy(name='entropy')` {#MultivariateNormalTriL.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.event_shape` {#MultivariateNormalTriL.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.event_shape_tensor(name='event_shape_tensor')` {#MultivariateNormalTriL.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.is_continuous` {#MultivariateNormalTriL.is_continuous}




- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.is_scalar_batch(name='is_scalar_batch')` {#MultivariateNormalTriL.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.is_scalar_event(name='is_scalar_event')` {#MultivariateNormalTriL.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.loc` {#MultivariateNormalTriL.loc}

The `loc` `Tensor` in `Y = scale @ X + loc`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.log_cdf(value, name='log_cdf')` {#MultivariateNormalTriL.log_cdf}

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

#### `tf.contrib.distributions.MultivariateNormalTriL.log_det_covariance(name='log_det_covariance')` {#MultivariateNormalTriL.log_det_covariance}

Log of determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.log_prob(value, name='log_prob')` {#MultivariateNormalTriL.log_prob}

Log probability density/mass function (depending on `is_continuous`).


Additional documentation from `MultivariateNormalLinearOperator`:

`value` is a batch vector with compatible shape if `value` is a `Tensor` whose
shape can be broadcast up to either:

```python
self.batch_shape + self.event_shape
```

or

```python
[M1, ..., Mm] + self.batch_shape + self.event_shape
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.log_survival_function(value, name='log_survival_function')` {#MultivariateNormalTriL.log_survival_function}

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

#### `tf.contrib.distributions.MultivariateNormalTriL.mean(name='mean')` {#MultivariateNormalTriL.mean}

Mean.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.mode(name='mode')` {#MultivariateNormalTriL.mode}

Mode.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.name` {#MultivariateNormalTriL.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#MultivariateNormalTriL.param_shapes}

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

#### `tf.contrib.distributions.MultivariateNormalTriL.param_static_shapes(cls, sample_shape)` {#MultivariateNormalTriL.param_static_shapes}

param_shapes with static (i.e. `TensorShape`) shapes.

This is a class method that describes what key/value arguments are required
to instantiate the given `Distribution` so that a particular shape is
returned for that instance's call to `sample()`. Assumes that the sample's
shape is known statically.

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

#### `tf.contrib.distributions.MultivariateNormalTriL.parameters` {#MultivariateNormalTriL.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.prob(value, name='prob')` {#MultivariateNormalTriL.prob}

Probability density/mass function (depending on `is_continuous`).


Additional documentation from `MultivariateNormalLinearOperator`:

`value` is a batch vector with compatible shape if `value` is a `Tensor` whose
shape can be broadcast up to either:

```python
self.batch_shape + self.event_shape
```

or

```python
[M1, ..., Mm] + self.batch_shape + self.event_shape
```

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.reparameterization_type` {#MultivariateNormalTriL.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.sample(sample_shape=(), seed=None, name='sample')` {#MultivariateNormalTriL.sample}

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

#### `tf.contrib.distributions.MultivariateNormalTriL.scale` {#MultivariateNormalTriL.scale}

The `scale` `LinearOperator` in `Y = scale @ X + loc`.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.stddev(name='stddev')` {#MultivariateNormalTriL.stddev}

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

#### `tf.contrib.distributions.MultivariateNormalTriL.survival_function(value, name='survival_function')` {#MultivariateNormalTriL.survival_function}

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

#### `tf.contrib.distributions.MultivariateNormalTriL.validate_args` {#MultivariateNormalTriL.validate_args}

Python `bool` indicating possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.MultivariateNormalTriL.variance(name='variance')` {#MultivariateNormalTriL.variance}

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


