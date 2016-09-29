Distribution representing the quantization `Y = ceiling(X)`.

#### Definition in terms of sampling.

```
1. Draw X
2. Set Y <-- ceiling(X)
3. If Y < lower_cutoff, reset Y <-- lower_cutoff
4. If Y > upper_cutoff, reset Y <-- upper_cutoff
5. Return Y
```

#### Definition in terms of the probability mass function.

Given scalar random variable `X`, we define a discrete random variable `Y`
supported on the integers as follows:

```
P[Y = j] := P[X <= lower_cutoff],  if j == lower_cutoff,
         := P[X > upper_cutoff - 1],  j == upper_cutoff,
         := 0, if j < lower_cutoff or j > upper_cutoff,
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
If `lower_cutoff = 0`, and `upper_cutoff = 2`, then the intervals are redrawn
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

#### `tf.contrib.distributions.QuantizedDistribution.__init__(base_dist_cls, lower_cutoff=None, upper_cutoff=None, name='QuantizedDistribution', **base_dist_args)` {#QuantizedDistribution.__init__}

Construct a Quantized Distribution.

Some properties are inherited from the distribution defining `X`.
In particular, `validate_args` and `allow_nan_stats` are determined for this
`QuantizedDistribution` by reading the additional kwargs passed as
`base_dist_args`.

##### Args:


*  <b>`base_dist_cls`</b>: the base distribution class to transform. Must be a
      subclass of `Distribution` implementing `cdf`.
*  <b>`lower_cutoff`</b>: `Tensor` with same `dtype` as this distribution and shape
    able to be added to samples.  Should be a whole number.  Default `None`.
    If provided, base distribution's pdf/pmf should be defined at
    `lower_cutoff`.
*  <b>`upper_cutoff`</b>: `Tensor` with same `dtype` as this distribution and shape
    able to be added to samples.  Should be a whole number.  Default `None`.
    If provided, base distribution's pdf/pmf should be defined at
    `upper_cutoff - 1`.
    `upper_cutoff` must be strictly greater than `lower_cutoff`.
*  <b>`name`</b>: The name for the distribution.
*  <b>`**base_dist_args`</b>: kwargs to pass on to dist_cls on construction.
    These determine the shape and dtype of this distribution.

##### Raises:


*  <b>`TypeError`</b>: If `base_dist_cls` is not a subclass of
      `Distribution` or continuous.
*  <b>`NotImplementedError`</b>: If the base distribution does not implement `cdf`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.allow_nan_stats` {#QuantizedDistribution.allow_nan_stats}

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

#### `tf.contrib.distributions.QuantizedDistribution.base_distribution` {#QuantizedDistribution.base_distribution}

Base distribution, p(x).


- - -

#### `tf.contrib.distributions.QuantizedDistribution.batch_shape(name='batch_shape')` {#QuantizedDistribution.batch_shape}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

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
        = 1, if y >= upper_cutoff,
        = 0, if y < lower_cutoff,
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

#### `tf.contrib.distributions.QuantizedDistribution.dtype` {#QuantizedDistribution.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.entropy(name='entropy')` {#QuantizedDistribution.entropy}

Shanon entropy in nats.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.event_shape(name='event_shape')` {#QuantizedDistribution.event_shape}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.get_batch_shape()` {#QuantizedDistribution.get_batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.get_event_shape()` {#QuantizedDistribution.get_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.is_continuous` {#QuantizedDistribution.is_continuous}




- - -

#### `tf.contrib.distributions.QuantizedDistribution.is_reparameterized` {#QuantizedDistribution.is_reparameterized}




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
        = 1, if y >= upper_cutoff,
        = 0, if y < lower_cutoff,
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

#### `tf.contrib.distributions.QuantizedDistribution.log_pdf(value, name='log_pdf')` {#QuantizedDistribution.log_pdf}

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

#### `tf.contrib.distributions.QuantizedDistribution.log_pmf(value, name='log_pmf')` {#QuantizedDistribution.log_pmf}

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

#### `tf.contrib.distributions.QuantizedDistribution.log_prob(value, name='log_prob')` {#QuantizedDistribution.log_prob}

Log probability density/mass function (depending on `is_continuous`).


Additional documentation from `QuantizedDistribution`:

For whole numbers `y`,

```
P[Y = y] := P[X <= lower_cutoff],  if y == lower_cutoff,
         := P[X > upper_cutoff - 1],  y == upper_cutoff,
         := 0, if j < lower_cutoff or y > upper_cutoff,
         := P[y - 1 < X <= y],  all other y.
```


The base distribution's `log_cdf` method must be defined on `y - 1`.  If the
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
                      = 0, if y >= upper_cutoff,
                      = 1, if y < lower_cutoff,
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

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.param_static_shapes(cls, sample_shape)` {#QuantizedDistribution.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.parameters` {#QuantizedDistribution.parameters}

Dictionary of parameters used by this `Distribution`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.pdf(value, name='pdf')` {#QuantizedDistribution.pdf}

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

#### `tf.contrib.distributions.QuantizedDistribution.pmf(value, name='pmf')` {#QuantizedDistribution.pmf}

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

#### `tf.contrib.distributions.QuantizedDistribution.prob(value, name='prob')` {#QuantizedDistribution.prob}

Probability density/mass function (depending on `is_continuous`).


Additional documentation from `QuantizedDistribution`:

For whole numbers `y`,

```
P[Y = y] := P[X <= lower_cutoff],  if y == lower_cutoff,
         := P[X > upper_cutoff - 1],  y == upper_cutoff,
         := 0, if j < lower_cutoff or y > upper_cutoff,
         := P[y - 1 < X <= y],  all other y.
```


The base distribution's `cdf` method must be defined on `y - 1`.  If the
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

#### `tf.contrib.distributions.QuantizedDistribution.sample_n(n, seed=None, name='sample_n')` {#QuantizedDistribution.sample_n}

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

#### `tf.contrib.distributions.QuantizedDistribution.std(name='std')` {#QuantizedDistribution.std}

Standard deviation.


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
                      = 0, if y >= upper_cutoff,
                      = 1, if y < lower_cutoff,
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

  Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.validate_args` {#QuantizedDistribution.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.QuantizedDistribution.variance(name='variance')` {#QuantizedDistribution.variance}

Variance.


