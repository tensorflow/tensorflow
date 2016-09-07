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

The user does not pass the list of distributions directly, but rather a
list of `(constructor, batch_tensor_params_dict)` pairs,
called `components`. The list of distributions is created via:

```python
distributions = [
  c(**params_dict) for (c, params_dict) in zip(*components)
]
```

This form allows for certain types of batch-shape optimizations within
this class.

An example of `components`:

```python
components = [
  (tf.contrib.distributions.Normal, {"mu": 3.0, "sigma": 1.0}),
  (functools.partial(tf.contrib.distributions.Normal, validate_args=False),
   {"mu": 3.0, "sigma": 2.0}),
  (tf.contrib.distributions.Normal.from_params,
   {"mu": 1.0, "sigma": -1.0})
]
```

The `num_classes` of `cat` must be possible to infer at graph construction
time and match `len(distributions)`.

##### Args:


*  <b>`cat`</b>: A `Categorical` distribution instance, representing the probabilities
      of `distributions`.
*  <b>`components`</b>: A list or tuple of `(constructor, batch_tensor_params)`
    tuples.  The `constructor` must be a callable, and `batch_tensor_params`
    must be a dict mapping constructor kwargs to batchwise parameters.
    Each `Distribution` instance created by calling
    `constructor(**batch_tensor_params)` must have the same type, be defined
    on the same domain, and have matching `event_shape` and `batch_shape`.
*  <b>`validate_args`</b>: `Boolean`, default `False`.  If `True`, raise a runtime
    error if batch or event ranks are inconsistent between cat and any of
    the distributions.  This is only checked if the ranks cannot be
    determined statically at graph construction time.
*  <b>`allow_nan_stats`</b>: Boolean, default `True`.  If `False`, raise an
   exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: A name for this distribution (optional).

##### Raises:


*  <b>`TypeError`</b>: If cat is not a `Categorical`, or `components` is not
    a list or tuple, or the elements of `components` are not
    tuples of the form `(callable, dict)`, or the objects resulting
    from calling `callable(**dict)` are not instances of `Distribution`, or
    the resulting instances of `Distribution` do not have matching
    continuity properties, or do not have matching `dtype`.
*  <b>`ValueError`</b>: If `components` is an empty list or tuple, or the
    distributions created from `components` do have a statically known event
    rank.  If `cat.num_classes` cannot be inferred at graph creation time,
    or the constant value of `cat.num_classes` is not equal to
    `len(distributions)`, or all `distributions` and `cat` do not have
    matching static batch shapes, or all components' distributions do not
    have matching static event shapes.


- - -

#### `tf.contrib.distributions.Mixture.allow_nan_stats` {#Mixture.allow_nan_stats}

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

#### `tf.contrib.distributions.Mixture.batch_shape(name='batch_shape')` {#Mixture.batch_shape}

Shape of a single sample from a single event index as a 1-D `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

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

#### `tf.contrib.distributions.Mixture.distributions` {#Mixture.distributions}




- - -

#### `tf.contrib.distributions.Mixture.dtype` {#Mixture.dtype}

The `DType` of `Tensor`s handled by this `Distribution`.


- - -

#### `tf.contrib.distributions.Mixture.entropy(name='entropy')` {#Mixture.entropy}

Shanon entropy in nats.


- - -

#### `tf.contrib.distributions.Mixture.entropy_lower_bound(name='entropy_lower_bound')` {#Mixture.entropy_lower_bound}

A lower bound on the entropy of this mixture model.

The bound below is not always very tight, and its usefulness depends
on the mixture probabilities and the distributions in use.

A lower bound is useful for ELBO when the `Mixture` is the variational
distribution:

\\(
\log p(x) >= ELBO = \int q(z) \log p(x, z) dz + H[q]
\\)

where \\( p \\) is the prior disribution, \\( q \\) is the variational,
and \\( H[q] \\) is the entropy of \\( q \\).  If there is a lower bound
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

#### `tf.contrib.distributions.Mixture.event_shape(name='event_shape')` {#Mixture.event_shape}

Shape of a single sample from a single batch as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:


*  <b>`event_shape`</b>: `Tensor`.


- - -

#### `tf.contrib.distributions.Mixture.from_params(cls, make_safe=True, **kwargs)` {#Mixture.from_params}

Given (unconstrained) parameters, return an instantiated distribution.

Subclasses should implement a static method `_safe_transforms` that returns
a dict of parameter transforms, which will be used if `make_safe = True`.

Example usage:

```
# Let's say we want a sample of size (batch_size, 10)
shapes = MultiVariateNormalDiag.param_shapes([batch_size, 10])

# shapes has a Tensor shape for mu and sigma
# shapes == {
#   "mu": tf.constant([batch_size, 10]),
#   "sigma": tf.constant([batch_size, 10]),
# }

# Here we parameterize mu and sigma with the output of a linear
# layer. Note that sigma is unconstrained.
params = {}
for name, shape in shapes.items():
  params[name] = linear(x, shape[1])

# Note that you can forward other kwargs to the `Distribution`, like
# `allow_nan_stats` or `name`.
mvn = MultiVariateNormalDiag.from_params(**params, allow_nan_stats=True)
```

Distribution parameters may have constraints (e.g. `sigma` must be positive
for a `Normal` distribution) and the `from_params` method will apply default
parameter transforms. If a user wants to use their own transform, they can
apply it externally and set `make_safe=False`.

##### Args:


*  <b>`make_safe`</b>: Whether the `params` should be constrained. If True,
    `from_params` will apply default parameter transforms. If False, no
    parameter transforms will be applied.
*  <b>`**kwargs`</b>: dict of parameters for the distribution.

##### Returns:

  A distribution parameterized by possibly transformed parameters in
  `kwargs`.

##### Raises:


*  <b>`TypeError`</b>: if `make_safe` is `True` but `_safe_transforms` is not
    implemented directly for `cls`.


- - -

#### `tf.contrib.distributions.Mixture.get_batch_shape()` {#Mixture.get_batch_shape}

Shape of a single sample from a single event index as a `TensorShape`.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:


*  <b>`batch_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.Mixture.get_event_shape()` {#Mixture.get_event_shape}

Shape of a single sample from a single batch as a `TensorShape`.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:


*  <b>`event_shape`</b>: `TensorShape`, possibly unknown.


- - -

#### `tf.contrib.distributions.Mixture.is_continuous` {#Mixture.is_continuous}




- - -

#### `tf.contrib.distributions.Mixture.is_reparameterized` {#Mixture.is_reparameterized}




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

#### `tf.contrib.distributions.Mixture.log_pdf(value, name='log_pdf')` {#Mixture.log_pdf}

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

#### `tf.contrib.distributions.Mixture.log_pmf(value, name='log_pmf')` {#Mixture.log_pmf}

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

Subclasses should override static method `_param_shapes`.

##### Args:


*  <b>`sample_shape`</b>: `Tensor` or python list/tuple. Desired shape of a call to
    `sample()`.
*  <b>`name`</b>: name to prepend ops with.

##### Returns:

  `dict` of parameter name to `Tensor` shapes.


- - -

#### `tf.contrib.distributions.Mixture.param_static_shapes(cls, sample_shape)` {#Mixture.param_static_shapes}

param_shapes with static (i.e. TensorShape) shapes.

##### Args:


*  <b>`sample_shape`</b>: `TensorShape` or python list/tuple. Desired shape of a call
    to `sample()`.

##### Returns:

  `dict` of parameter name to `TensorShape`.

##### Raises:


*  <b>`ValueError`</b>: if `sample_shape` is a `TensorShape` and is not fully defined.


- - -

#### `tf.contrib.distributions.Mixture.parameters` {#Mixture.parameters}

Dictionary of parameters used by this `Distribution`.


- - -

#### `tf.contrib.distributions.Mixture.pdf(value, name='pdf')` {#Mixture.pdf}

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

#### `tf.contrib.distributions.Mixture.pmf(value, name='pmf')` {#Mixture.pmf}

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

#### `tf.contrib.distributions.Mixture.prob(value, name='prob')` {#Mixture.prob}

Probability density/mass function (depending on `is_continuous`).

##### Args:


*  <b>`value`</b>: `float` or `double` `Tensor`.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`prob`</b>: a `Tensor` of shape `sample_shape(x) + self.batch_shape` with
    values of type `self.dtype`.


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

#### `tf.contrib.distributions.Mixture.sample_n(n, seed=None, name='sample_n')` {#Mixture.sample_n}

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

#### `tf.contrib.distributions.Mixture.std(name='std')` {#Mixture.std}

Standard deviation.


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

  Tensor` of shape `sample_shape(x) + self.batch_shape` with values of type
    `self.dtype`.


- - -

#### `tf.contrib.distributions.Mixture.validate_args` {#Mixture.validate_args}

Python boolean indicated possibly expensive checks are enabled.


- - -

#### `tf.contrib.distributions.Mixture.variance(name='variance')` {#Mixture.variance}

Variance.


