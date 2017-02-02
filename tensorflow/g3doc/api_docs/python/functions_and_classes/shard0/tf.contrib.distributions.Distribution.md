A generic probability distribution base class.

`Distribution` is a base class for constructing and organizing properties
(e.g., mean, variance) of random variables (e.g, Bernoulli, Gaussian).

### Subclassing

Subclasses are expected to implement a leading-underscore version of the
same-named function.  The argument signature should be identical except for
the omission of `name="..."`.  For example, to enable `log_prob(value,
name="log_prob")` a subclass should implement `_log_prob(value)`.

Subclasses can append to public-level docstrings by providing
docstrings for their method specializations. For example:

```python
@distribution_util.AppendDocstring("Some other details.")
def _log_prob(self, value):
  ...
```

would add the string "Some other details." to the `log_prob` function
docstring.  This is implemented as a simple decorator to avoid python
linter complaining about missing Args/Returns/Raises sections in the
partial docstrings.

### Broadcasting, batching, and shapes

All distributions support batches of independent distributions of that type.
The batch shape is determined by broadcasting together the parameters.

The shape of arguments to `__init__`, `cdf`, `log_cdf`, `prob`, and
`log_prob` reflect this broadcasting, as does the return value of `sample` and
`sample_n`.

`sample_n_shape = (n,) + batch_shape + event_shape`, where `sample_n_shape` is
the shape of the `Tensor` returned from `sample_n`, `n` is the number of
samples, `batch_shape` defines how many independent distributions there are,
and `event_shape` defines the shape of samples from each of those independent
distributions. Samples are independent along the `batch_shape` dimensions, but
not necessarily so along the `event_shape` dimensions (depending on the
particulars of the underlying distribution).

Using the `Uniform` distribution as an example:

```python
minval = 3.0
maxval = [[4.0, 6.0],
          [10.0, 12.0]]

# Broadcasting:
# This instance represents 4 Uniform distributions. Each has a lower bound at
# 3.0 as the `minval` parameter was broadcasted to match `maxval`'s shape.
u = Uniform(minval, maxval)

# `event_shape` is `TensorShape([])`.
event_shape = u.event_shape
# `event_shape_t` is a `Tensor` which will evaluate to [].
event_shape_t = u.event_shape_tensor()

# Sampling returns a sample per distribution.  `samples` has shape
# (5, 2, 2), which is (n,) + batch_shape + event_shape, where n=5,
# batch_shape=(2, 2), and event_shape=().
samples = u.sample_n(5)

# The broadcasting holds across methods. Here we use `cdf` as an example. The
# same holds for `log_cdf` and the likelihood functions.

# `cum_prob` has shape (2, 2) as the `value` argument was broadcasted to the
# shape of the `Uniform` instance.
cum_prob_broadcast = u.cdf(4.0)

# `cum_prob`'s shape is (2, 2), one per distribution. No broadcasting
# occurred.
cum_prob_per_dist = u.cdf([[4.0, 5.0],
                           [6.0, 7.0]])

# INVALID as the `value` argument is not broadcastable to the distribution's
# shape.
cum_prob_invalid = u.cdf([4.0, 5.0, 6.0])
```

### Parameter values leading to undefined statistics or distributions.

Some distributions do not have well-defined statistics for all initialization
parameter values.  For example, the beta distribution is parameterized by
positive real numbers `a` and `b`, and does not have well-defined mode if
`a < 1` or `b < 1`.

The user is given the option of raising an exception or returning `NaN`.

```python
a = tf.exp(tf.matmul(logits, weights_a))
b = tf.exp(tf.matmul(logits, weights_b))

# Will raise exception if ANY batch member has a < 1 or b < 1.
dist = distributions.beta(a, b, allow_nan_stats=False)
mode = dist.mode().eval()

# Will return NaN for batch members with either a < 1 or b < 1.
dist = distributions.beta(a, b, allow_nan_stats=True)  # Default behavior
mode = dist.mode().eval()
```

In all cases, an exception is raised if *invalid* parameters are passed, e.g.

```python
# Will raise an exception if any Op is run.
negative_a = -1.0 * a  # beta distribution by definition has a > 0.
dist = distributions.beta(negative_a, b, allow_nan_stats=True)
dist.mean().eval()
```
- - -

#### `tf.contrib.distributions.Distribution.__init__(dtype, is_continuous, reparameterization_type, validate_args, allow_nan_stats, parameters=None, graph_parents=None, name=None)` {#Distribution.__init__}

Constructs the `Distribution`.

**This is a private method for subclass use.**

##### Args:


*  <b>`dtype`</b>: The type of the event samples. `None` implies no type-enforcement.
*  <b>`is_continuous`</b>: Python boolean. If `True` this
    `Distribution` is continuous over its supported domain.
*  <b>`reparameterization_type`</b>: Instance of `ReparameterizationType`.
    If `distributions.FULLY_REPARAMETERIZED`, this
    `Distribution` can be reparameterized in terms of some standard
    distribution with a function whose Jacobian is constant for the support
    of the standard distribution.  If `distributions.NOT_REPARAMETERIZED`,
    then no such reparameterization is available.
*  <b>`validate_args`</b>: Python boolean.  Whether to validate input with asserts.
    If `validate_args` is `False`, and the inputs are invalid,
    correct behavior is not guaranteed.
*  <b>`allow_nan_stats`</b>: Python boolean.  If `False`, raise an
    exception if a statistic (e.g., mean, mode) is undefined for any batch
    member. If True, batch members with valid parameters leading to
    undefined statistics will return `NaN` for this statistic.
*  <b>`parameters`</b>: Python dictionary of parameters used to instantiate this
    `Distribution`.
*  <b>`graph_parents`</b>: Python list of graph prerequisites of this `Distribution`.
*  <b>`name`</b>: A name for this distribution. Default: subclass name.

##### Raises:


*  <b>`ValueError`</b>: if any member of graph_parents is `None` or not a `Tensor`.


- - -

#### `tf.contrib.distributions.Distribution.allow_nan_stats` {#Distribution.allow_nan_stats}

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

#### `tf.contrib.distributions.Distribution.batch_shape` {#Distribution.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.Distribution.batch_shape_tensor(name='batch_shape_tensor')` {#Distribution.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.Distribution.cdf(value, name='cdf')` {#Distribution.cdf}

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

#### `tf.contrib.distributions.Distribution.copy(**override_parameters_kwargs)` {#Distribution.copy}

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

#### `tf.contrib.distributions.Distribution.covariance(name='covariance')` {#Distribution.covariance}

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

#### `tf.contrib.distributions.Distribution.dtype` {#Distribution.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.Distribution.entropy(name='entropy')` {#Distribution.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.Distribution.event_shape` {#Distribution.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.Distribution.event_shape_tensor(name='event_shape_tensor')` {#Distribution.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.Distribution.is_continuous` {#Distribution.is_continuous}




- - -

#### `tf.contrib.distributions.Distribution.is_scalar_batch(name='is_scalar_batch')` {#Distribution.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `Boolean` `scalar` `Tensor`.


- - -

#### `tf.contrib.distributions.Distribution.is_scalar_event(name='is_scalar_event')` {#Distribution.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `Boolean` `scalar` `Tensor`.


- - -

#### `tf.contrib.distributions.Distribution.log_cdf(value, name='log_cdf')` {#Distribution.log_cdf}

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

#### `tf.contrib.distributions.Distribution.log_prob(value, name='log_prob')` {#Distribution.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Distribution.log_survival_function(value, name='log_survival_function')` {#Distribution.log_survival_function}

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

#### `tf.contrib.distributions.Distribution.mean(name='mean')` {#Distribution.mean}

Mean.


- - -

#### `tf.contrib.distributions.Distribution.mode(name='mode')` {#Distribution.mode}

Mode.


- - -

#### `tf.contrib.distributions.Distribution.name` {#Distribution.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.Distribution.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#Distribution.param_shapes}

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

#### `tf.contrib.distributions.Distribution.param_static_shapes(cls, sample_shape)` {#Distribution.param_static_shapes}

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

#### `tf.contrib.distributions.Distribution.parameters` {#Distribution.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.Distribution.prob(value, name='prob')` {#Distribution.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Distribution.reparameterization_type` {#Distribution.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.Distribution.sample(sample_shape=(), seed=None, name='sample')` {#Distribution.sample}

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

#### `tf.contrib.distributions.Distribution.stddev(name='stddev')` {#Distribution.stddev}

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

#### `tf.contrib.distributions.Distribution.survival_function(value, name='survival_function')` {#Distribution.survival_function}

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

#### `tf.contrib.distributions.Distribution.validate_args` {#Distribution.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.Distribution.variance(name='variance')` {#Distribution.variance}

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


