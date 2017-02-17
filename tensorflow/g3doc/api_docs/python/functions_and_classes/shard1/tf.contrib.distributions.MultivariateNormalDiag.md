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
scale = diag(scale_diag + scale_identity_multiplier * ones(k))
```

where:

* `scale_diag.shape = [k]`, and,
* `scale_identity_multiplier.shape = []`.

Additional leading dimensions (if any) will index batches.

If both `scale_diag` and `scale_identity_multiplier` are `None`, then
`scale` is the Identity matrix.

The MultivariateNormal distribution is a member of the [location-scale
family](https://en.wikipedia.org/wiki/Location-scale_family), i.e., it can be
constructed as,

```none
X ~ MultivariateNormal(loc=0, scale=1)   # Identity scale, zero shift.
Y = scale @ X + loc
```

#### Examples

```python
ds = tf.contrib.distributions

# Initialize a single 2-variate Gaussian.
mvn = ds.MultivariateNormalDiag(
    loc=[1., -1],
    scale_diag=[1, 2.])

mvn.mean().eval()
# ==> [1., -1]

mvn.stddev().eval()
# ==> [1., 2]

# Evaluate this on an observation in `R^2`, returning a scalar.
mvn.prob([-1., 0]).eval()  # shape: []

# Initialize a 3-batch, 2-variate scaled-identity Gaussian.
mvn = ds.MultivariateNormalDiag(
    loc=[1., -1],
    scale_identity_multiplier=[1, 2., 3])

mvn.mean().eval()  # shape: [3, 2]
# ==> [[1., -1]
#      [1, -1],
#      [1, -1]]

mvn.stddev().eval()  # shape: [3, 2]
# ==> [[1., 1],
#      [2, 2],
#      [3, 3]]

# Evaluate this on an observation in `R^2`, returning a length-3 vector.
mvn.prob([-1., 0]).eval()  # shape: [3]

# Initialize a 2-batch of 3-variate Gaussians.
mvn = ds.MultivariateNormalDiag(
    loc=[[1., 2, 3],
         [11, 22, 33]]           # shape: [2, 3]
    scale_diag=[[1., 2, 3],
                [0.5, 1, 1.5]])  # shape: [2, 3]

# Evaluate this on a two observations, each in `R^3`, returning a length-2
# vector.
x = [[-1., 0, 1],
     [-11, 0, 11.]]   # shape: [2, 3].
mvn.prob(x).eval()    # shape: [2]
```
- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.__init__(loc=None, scale_diag=None, scale_identity_multiplier=None, validate_args=False, allow_nan_stats=True, name='MultivariateNormalDiag')` {#MultivariateNormalDiag.__init__}

Construct Multivariate Normal distribution on `R^k`.

The `batch_shape` is the broadcast shape between `loc` and `scale`
arguments.

The `event_shape` is given by the last dimension of `loc` or the last
dimension of the matrix implied by `scale`.

Recall that `covariance = scale @ scale.T`. A (non-batch) `scale` matrix is:

```none
scale = diag(scale_diag + scale_identity_multiplier * ones(k))
```

where:

* `scale_diag.shape = [k]`, and,
* `scale_identity_multiplier.shape = []`.

Additional leading dimensions (if any) will index batches.

If both `scale_diag` and `scale_identity_multiplier` are `None`, then
`scale` is the Identity matrix.

##### Args:


*  <b>`loc`</b>: Floating-point `Tensor`. If this is set to `None`, `loc` is
    implicitly `0`. When specified, may have shape `[B1, ..., Bb, k]` where
    `b >= 0` and `k` is the event size.
*  <b>`scale_diag`</b>: Non-zero, floating-point `Tensor` representing a diagonal
    matrix added to `scale`. May have shape `[B1, ..., Bb, k]`, `b >= 0`,
    and characterizes `b`-batches of `k x k` diagonal matrices added to
    `scale`. When both `scale_identity_multiplier` and `scale_diag` are
    `None` then `scale` is the `Identity`.
*  <b>`scale_identity_multiplier`</b>: Non-zero, floating-point `Tensor` representing
    a scaled-identity-matrix added to `scale`. May have shape
    `[B1, ..., Bb]`, `b >= 0`, and characterizes `b`-batches of scaled
    `k x k` identity matrices added to `scale`. When both
    `scale_identity_multiplier` and `scale_diag` are `None` then `scale` is
    the `Identity`.
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


*  <b>`ValueError`</b>: if at most `scale_identity_multiplier` is specified.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.allow_nan_stats` {#MultivariateNormalDiag.allow_nan_stats}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.batch_shape` {#MultivariateNormalDiag.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.batch_shape_tensor(name='batch_shape_tensor')` {#MultivariateNormalDiag.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.bijector` {#MultivariateNormalDiag.bijector}

Function transforming x => y.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.cdf(value, name='cdf')` {#MultivariateNormalDiag.cdf}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.copy(**override_parameters_kwargs)` {#MultivariateNormalDiag.copy}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.covariance(name='covariance')` {#MultivariateNormalDiag.covariance}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.det_covariance(name='det_covariance')` {#MultivariateNormalDiag.det_covariance}

Determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.distribution` {#MultivariateNormalDiag.distribution}

Base distribution, p(x).


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.dtype` {#MultivariateNormalDiag.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.entropy(name='entropy')` {#MultivariateNormalDiag.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.event_shape` {#MultivariateNormalDiag.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.event_shape_tensor(name='event_shape_tensor')` {#MultivariateNormalDiag.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.is_continuous` {#MultivariateNormalDiag.is_continuous}




- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.is_scalar_batch(name='is_scalar_batch')` {#MultivariateNormalDiag.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.is_scalar_event(name='is_scalar_event')` {#MultivariateNormalDiag.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.loc` {#MultivariateNormalDiag.loc}

The `loc` `Tensor` in `Y = scale @ X + loc`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.log_cdf(value, name='log_cdf')` {#MultivariateNormalDiag.log_cdf}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.log_det_covariance(name='log_det_covariance')` {#MultivariateNormalDiag.log_det_covariance}

Log of determinant of covariance matrix.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.log_prob(value, name='log_prob')` {#MultivariateNormalDiag.log_prob}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.log_survival_function(value, name='log_survival_function')` {#MultivariateNormalDiag.log_survival_function}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.mean(name='mean')` {#MultivariateNormalDiag.mean}

Mean.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.mode(name='mode')` {#MultivariateNormalDiag.mode}

Mode.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.name` {#MultivariateNormalDiag.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#MultivariateNormalDiag.param_shapes}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.param_static_shapes(cls, sample_shape)` {#MultivariateNormalDiag.param_static_shapes}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.parameters` {#MultivariateNormalDiag.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.prob(value, name='prob')` {#MultivariateNormalDiag.prob}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.reparameterization_type` {#MultivariateNormalDiag.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.sample(sample_shape=(), seed=None, name='sample')` {#MultivariateNormalDiag.sample}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.scale` {#MultivariateNormalDiag.scale}

The `scale` `LinearOperator` in `Y = scale @ X + loc`.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.stddev(name='stddev')` {#MultivariateNormalDiag.stddev}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.survival_function(value, name='survival_function')` {#MultivariateNormalDiag.survival_function}

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

#### `tf.contrib.distributions.MultivariateNormalDiag.validate_args` {#MultivariateNormalDiag.validate_args}

Python `bool` indicating possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.MultivariateNormalDiag.variance(name='variance')` {#MultivariateNormalDiag.variance}

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


