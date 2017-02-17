Distribution representing the quantization `Y = ceiling(X)`.

#### Definition in terms of sampling.

```
1. Draw X
2. Set Y <-- ceiling(X)
3. If Y < low, reset Y <-- low
4. If Y > high, reset Y <-- high
5. Return Y
```

#### Definition in terms of the probability mass function.

Given scalar random variable `X`, we define a discrete random variable `Y`
supported on the integers as follows:

```
P[Y = j] := P[X <= low],  if j == low,
         := P[X > high - 1],  j == high,
         := 0, if j < low or j > high,
         := P[j - 1 < X <= j],  all other j.
```

Conceptually, without cutoffs, the quantization process partitions the real
line `R` into half open intervals, and identifies an integer `j` with the
right endpoints:

```
R = ... (-2, -1](-1, 0](0, 1](1, 2](2, 3](3, 4] ...
j = ...      -1      0     1     2     3     4  ...
```

`P[Y = j]` is the mass of `X` within the `jth` interval.
If `low = 0`, and `high = 2`, then the intervals are redrawn
and `j` is re-assigned:

```
R = (-infty, 0](0, 1](1, infty)
j =          0     1     2
```

`P[Y = j]` is still the mass of `X` within the `jth` interval.

#### Caveats

Since evaluation of each `P[Y = j]` involves a cdf evaluation (rather than
a closed form function such as for a Poisson), computations such as mean and
entropy are better done with samples or approximations, and are not
implemented by this class.
- - -

#### `tf.contrib.distributions.QuantizedDistribution.__init__(distribution, low=None, high=None, validate_args=False, name='QuantizedDistribution')` {#QuantizedDistribution.__init__}

Construct a Quantized Distribution representing `Y = ceiling(X)`.

Some properties are inherited from the distribution defining `X`. Example:
`allow_nan_stats` is determined for this `QuantizedDistribution` by reading
the `distribution`.

##### Args:


*  <b>`distribution`</b>: The base distribution class to transform. Typically an
    instance of `Distribution`.
*  <b>`low`</b>: `Tensor` with same `dtype` as this distribution and shape
    able to be added to samples. Should be a whole number. Default `None`.
    If provided, base distribution's `prob` should be defined at
    `low`.
*  <b>`high`</b>: `Tensor` with same `dtype` as this distribution and shape
    able to be added to samples. Should be a whole number. Default `None`.
    If provided, base distribution's `prob` should be defined at
    `high - 1`.
    `high` must be strictly greater than `low`.
*  <b>`validate_args`</b>: Python `bool`, default `False`. When `True` distribution
    parameters are checked for validity despite possibly degrading runtime
    performance. When `False` invalid inputs may silently render incorrect
    outputs.
*  <b>`name`</b>: Python `str` name prefixed to Ops created by this class.

##### Raises:


*  <b>`TypeError`</b>: If `dist_cls` is not a subclass of
      `Distribution` or continuous.
*  <b>`NotImplementedError`</b>: If the base distribution does not implement `cdf`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.allow_nan_stats` {#QuantizedDistribution.allow_nan_stats}

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

#### `tf.contrib.distributions.QuantizedDistribution.batch_shape` {#QuantizedDistribution.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.batch_shape_tensor(name='batch_shape_tensor')` {#QuantizedDistribution.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.cdf(value, name='cdf')` {#QuantizedDistribution.cdf}

Cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
cdf(x) := P[X <= x]
```


Additional documentation from `QuantizedDistribution`:

For whole numbers `y`,

```
cdf(y) := P[Y <= y]
        = 1, if y >= high,
        = 0, if y < low,
        = P[X <= y], otherwise.
```

Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
This dictates that fractional `y` are first floored to a whole number, and
then above definition applies.

The base distribution's `cdf` method must be defined on `y - 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`cdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.copy(**override_parameters_kwargs)` {#QuantizedDistribution.copy}

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

#### `tf.contrib.distributions.QuantizedDistribution.covariance(name='covariance')` {#QuantizedDistribution.covariance}

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

#### `tf.contrib.distributions.QuantizedDistribution.distribution` {#QuantizedDistribution.distribution}

Base distribution, p(x).


- - -

#### `tf.contrib.distributions.QuantizedDistribution.dtype` {#QuantizedDistribution.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.entropy(name='entropy')` {#QuantizedDistribution.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.event_shape` {#QuantizedDistribution.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.event_shape_tensor(name='event_shape_tensor')` {#QuantizedDistribution.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.is_continuous` {#QuantizedDistribution.is_continuous}




- - -

#### `tf.contrib.distributions.QuantizedDistribution.is_scalar_batch(name='is_scalar_batch')` {#QuantizedDistribution.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.is_scalar_event(name='is_scalar_event')` {#QuantizedDistribution.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.log_cdf(value, name='log_cdf')` {#QuantizedDistribution.log_cdf}

Log cumulative distribution function.

Given random variable `X`, the cumulative distribution function `cdf` is:

```
log_cdf(x) := Log[ P[X <= x] ]
```

Often, a numerical approximation can be used for `log_cdf(x)` that yields
a more accurate answer than simply taking the logarithm of the `cdf` when
`x << -1`.


Additional documentation from `QuantizedDistribution`:

For whole numbers `y`,

```
cdf(y) := P[Y <= y]
        = 1, if y >= high,
        = 0, if y < low,
        = P[X <= y], otherwise.
```

Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
This dictates that fractional `y` are first floored to a whole number, and
then above definition applies.

The base distribution's `log_cdf` method must be defined on `y - 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`logcdf`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.log_prob(value, name='log_prob')` {#QuantizedDistribution.log_prob}

Log probability density/mass function (depending on `is_continuous`).


Additional documentation from `QuantizedDistribution`:

For whole numbers `y`,

```
P[Y = y] := P[X <= low],  if y == low,
         := P[X > high - 1],  y == high,
         := 0, if j < low or y > high,
         := P[y - 1 < X <= y],  all other y.
```


The base distribution's `log_cdf` method must be defined on `y - 1`. If the
base distribution has a `log_survival_function` method results will be more
accurate for large values of `y`, and in this case the `log_survival_function`
must also be defined on `y - 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.log_survival_function(value, name='log_survival_function')` {#QuantizedDistribution.log_survival_function}

Log survival function.

Given random variable `X`, the survival function is defined:

```
log_survival_function(x) = Log[ P[X > x] ]
                         = Log[ 1 - P[X <= x] ]
                         = Log[ 1 - cdf(x) ]
```

Typically, different numerical approximations can be used for the log
survival function, which are more accurate than `1 - cdf(x)` when `x >> 1`.


Additional documentation from `QuantizedDistribution`:

For whole numbers `y`,

```
survival_function(y) := P[Y > y]
                      = 0, if y >= high,
                      = 1, if y < low,
                      = P[X <= y], otherwise.
```

Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
This dictates that fractional `y` are first floored to a whole number, and
then above definition applies.

The base distribution's `log_cdf` method must be defined on `y - 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.mean(name='mean')` {#QuantizedDistribution.mean}

Mean.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.mode(name='mode')` {#QuantizedDistribution.mode}

Mode.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.name` {#QuantizedDistribution.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#QuantizedDistribution.param_shapes}

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

#### `tf.contrib.distributions.QuantizedDistribution.param_static_shapes(cls, sample_shape)` {#QuantizedDistribution.param_static_shapes}

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

#### `tf.contrib.distributions.QuantizedDistribution.parameters` {#QuantizedDistribution.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.prob(value, name='prob')` {#QuantizedDistribution.prob}

Probability density/mass function (depending on `is_continuous`).


Additional documentation from `QuantizedDistribution`:

For whole numbers `y`,

```
P[Y = y] := P[X <= low],  if y == low,
         := P[X > high - 1],  y == high,
         := 0, if j < low or y > high,
         := P[y - 1 < X <= y],  all other y.
```


The base distribution's `cdf` method must be defined on `y - 1`. If the
base distribution has a `survival_function` method, results will be more
accurate for large values of `y`, and in this case the `survival_function` must
also be defined on `y - 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.reparameterization_type` {#QuantizedDistribution.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.sample(sample_shape=(), seed=None, name='sample')` {#QuantizedDistribution.sample}

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

#### `tf.contrib.distributions.QuantizedDistribution.stddev(name='stddev')` {#QuantizedDistribution.stddev}

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

#### `tf.contrib.distributions.QuantizedDistribution.survival_function(value, name='survival_function')` {#QuantizedDistribution.survival_function}

Survival function.

Given random variable `X`, the survival function is defined:

```
survival_function(x) = P[X > x]
                     = 1 - P[X <= x]
                     = 1 - cdf(x).
```


Additional documentation from `QuantizedDistribution`:

For whole numbers `y`,

```
survival_function(y) := P[Y > y]
                      = 0, if y >= high,
                      = 1, if y < low,
                      = P[X <= y], otherwise.
```

Since `Y` only has mass at whole numbers, `P[Y <= y] = P[Y <= floor(y)]`.
This dictates that fractional `y` are first floored to a whole number, and
then above definition applies.

The base distribution's `cdf` method must be defined on `y - 1`.

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:

  `Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.validate_args` {#QuantizedDistribution.validate_args}

Python `bool` indicating possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.variance(name='variance')` {#QuantizedDistribution.variance}

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


