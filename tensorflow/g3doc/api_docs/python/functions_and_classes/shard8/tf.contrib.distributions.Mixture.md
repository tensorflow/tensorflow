Mixture distribution.

The `Mixture` object implements batched mixture distributions.
The mixture model is defined by a `Categorical` distribution (the mixture)
and a python list of `Distribution` objects.

Methods supported include `log_prob`, `prob`, `mean`, `sample`, and
`entropy_lower_bound`.
- - -

#### `tf.contrib.distributions.Mixture.__init__(cat, components, validate_args=False, allow_nan_stats=True, name='Mixture')` {#Mixture.__init__}

Initialize a Mixture distribution.

A `Mixture` is defined by a `Categorical` (`cat`, representing the
mixture probabilities) and a list of `Distribution` objects
all having matching dtype, batch shape, event shape, and continuity
properties (the components).

The `num_classes` of `cat` must be possible to infer at graph construction
time and match `len(components)`.

##### Args:


*  <b>`cat`</b>: A `Categorical` distribution instance, representing the probabilities
      of `distributions`.
*  <b>`components`</b>: A list or tuple of `Distribution` instances.
    Each instance must have the same type, be defined on the same domain,
    and have matching `event_shape` and `batch_shape`.
*  <b>`validate_args`</b>: Python `bool`, default `False`. If `True`, raise a runtime
    error if batch or event ranks are inconsistent between cat and any of
    the distributions. This is only checked if the ranks cannot be
    determined statically at graph construction time.
*  <b>`allow_nan_stats`</b>: Boolean, default `True`. If `False`, raise an
   exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member. If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: A name for this distribution (optional).

##### Raises:


*  <b>`TypeError`</b>: If cat is not a `Categorical`, or `components` is not
    a list or tuple, or the elements of `components` are not
    instances of `Distribution`, or do not have matching `dtype`.
*  <b>`ValueError`</b>: If `components` is an empty list or tuple, or its
    elements do not have a statically known event rank.
    If `cat.num_classes` cannot be inferred at graph creation time,
    or the constant value of `cat.num_classes` is not equal to
    `len(components)`, or all `components` and `cat` do not have
    matching static batch shapes, or all components do not
    have matching static event shapes.


- - -

#### `tf.contrib.distributions.Mixture.allow_nan_stats` {#Mixture.allow_nan_stats}

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

#### `tf.contrib.distributions.Mixture.batch_shape` {#Mixture.batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

May be partially defined or unknown.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.Mixture.batch_shape_tensor(name='batch_shape_tensor')` {#Mixture.batch_shape_tensor}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The batch dimensions are indexes into independent, non-identical
parameterizations of this distribution.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`batch_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.Mixture.cat` {#Mixture.cat}




- - -

#### `tf.contrib.distributions.Mixture.cdf(value, name='cdf')` {#Mixture.cdf}

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

#### `tf.contrib.distributions.Mixture.components` {#Mixture.components}




- - -

#### `tf.contrib.distributions.Mixture.copy(**override_parameters_kwargs)` {#Mixture.copy}

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

#### `tf.contrib.distributions.Mixture.covariance(name='covariance')` {#Mixture.covariance}

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

#### `tf.contrib.distributions.Mixture.dtype` {#Mixture.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.Mixture.entropy(name='entropy')` {#Mixture.entropy}

Shannon entropy in nats.


- - -

#### `tf.contrib.distributions.Mixture.entropy_lower_bound(name='entropy_lower_bound')` {#Mixture.entropy_lower_bound}

A lower bound on the entropy of this mixture model.

The bound below is not always very tight, and its usefulness depends
on the mixture probabilities and the components in use.

A lower bound is useful for ELBO when the `Mixture` is the variational
distribution:

\\(
\log p(x) >= ELBO = \int q(z) \log p(x, z) dz + H[q]
\\)

where \\( p \\) is the prior distribution, \\( q \\) is the variational,
and \\( H[q] \\) is the entropy of \\( q \\). If there is a lower bound
\\( G[q] \\) such that \\( H[q] \geq G[q] \\) then it can be used in
place of \\( H[q] \\).

For a mixture of distributions \\( q(Z) = \sum_i c_i q_i(Z) \\) with
\\( \sum_i c_i = 1 \\), by the concavity of \\( f(x) = -x \log x \\), a
simple lower bound is:

\\(
\begin{align}
H[q] & = - \int q(z) \log q(z) dz \\\
   & = - \int (\sum_i c_i q_i(z)) \log(\sum_i c_i q_i(z)) dz \\\
   & \geq - \sum_i c_i \int q_i(z) \log q_i(z) dz \\\
   & = \sum_i c_i H[q_i]
\end{align}
\\)

This is the term we calculate below for \\( G[q] \\).

##### Args:


*  <b>`name`</b>: A name for this operation (optional).

##### Returns:

  A lower bound on the Mixture's entropy.


- - -

#### `tf.contrib.distributions.Mixture.event_shape` {#Mixture.event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

May be partially defined or unknown.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.Mixture.event_shape_tensor(name='event_shape_tensor')` {#Mixture.event_shape_tensor}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.Mixture.is_continuous` {#Mixture.is_continuous}




- - -

#### `tf.contrib.distributions.Mixture.is_scalar_batch(name='is_scalar_batch')` {#Mixture.is_scalar_batch}

Indicates that `batch_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_batch`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.Mixture.is_scalar_event(name='is_scalar_event')` {#Mixture.is_scalar_event}

Indicates that `event_shape == []`.

##### Args:


*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`is_scalar_event`</b>: `bool` scalar `Tensor`.


- - -

#### `tf.contrib.distributions.Mixture.log_cdf(value, name='log_cdf')` {#Mixture.log_cdf}

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

#### `tf.contrib.distributions.Mixture.log_prob(value, name='log_prob')` {#Mixture.log_prob}

Log probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`log_prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Mixture.log_survival_function(value, name='log_survival_function')` {#Mixture.log_survival_function}

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

#### `tf.contrib.distributions.Mixture.mean(name='mean')` {#Mixture.mean}

Mean.


- - -

#### `tf.contrib.distributions.Mixture.mode(name='mode')` {#Mixture.mode}

Mode.


- - -

#### `tf.contrib.distributions.Mixture.name` {#Mixture.name}

Name prepended to all ops created by this `Distribution`.


- - -

#### `tf.contrib.distributions.Mixture.num_components` {#Mixture.num_components}




- - -

#### `tf.contrib.distributions.Mixture.param_shapes(cls, sample_shape, name='DistributionParamShapes')` {#Mixture.param_shapes}

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

#### `tf.contrib.distributions.Mixture.param_static_shapes(cls, sample_shape)` {#Mixture.param_static_shapes}

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

#### `tf.contrib.distributions.Mixture.parameters` {#Mixture.parameters}

Dictionary of parameters used to instantiate this `Distribution`.


- - -

#### `tf.contrib.distributions.Mixture.prob(value, name='prob')` {#Mixture.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Mixture.reparameterization_type` {#Mixture.reparameterization_type}

Describes how samples from the distribution are reparameterized.

Currently this is one of the static instances
`distributions.FULLY_REPARAMETERIZED`
or `distributions.NOT_REPARAMETERIZED`.

##### Returns:

  An instance of `ReparameterizationType`.


- - -

#### `tf.contrib.distributions.Mixture.sample(sample_shape=(), seed=None, name='sample')` {#Mixture.sample}

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

#### `tf.contrib.distributions.Mixture.stddev(name='stddev')` {#Mixture.stddev}

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

#### `tf.contrib.distributions.Mixture.survival_function(value, name='survival_function')` {#Mixture.survival_function}

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

#### `tf.contrib.distributions.Mixture.validate_args` {#Mixture.validate_args}

Python `bool` indicating possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.Mixture.variance(name='variance')` {#Mixture.variance}

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


