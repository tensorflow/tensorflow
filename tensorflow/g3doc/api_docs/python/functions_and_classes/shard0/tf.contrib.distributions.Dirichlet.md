Dirichlet distribution.

This distribution is parameterized by a vector `alpha` of concentration
parameters for `k` classes.

#### Mathematical details

The Dirichlet is a distribution over the standard n-simplex, where the
standard n-simplex is defined by:
```{ (x_1, ..., x_n) in R^(n+1) | sum_j x_j = 1 and x_j >= 0 for all j }```.
The distribution has hyperparameters `alpha = (alpha_1,...,alpha_k)`,
and probability mass function (prob):

```prob(x) = 1 / Beta(alpha) * prod_j x_j^(alpha_j - 1)```

where `Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the multivariate
beta function.


This class provides methods to create indexed batches of Dirichlet
distributions.  If the provided `alpha` is rank 2 or higher, for
every fixed set of leading dimensions, the last dimension represents one
single Dirichlet distribution.  When calling distribution
functions (e.g. `dist.prob(x)`), `alpha` and `x` are broadcast to the
same shape (if possible).  In all cases, the last dimension of alpha/x
represents single Dirichlet distributions.

#### Examples

```python
alpha = [1, 2, 3]
dist = Dirichlet(alpha)
```

Creates a 3-class distribution, with the 3rd class is most likely to be drawn.
The distribution functions can be evaluated on x.

```python
# x same shape as alpha.
x = [.2, .3, .5]
dist.prob(x)  # Shape []

# alpha will be broadcast to [[1, 2, 3], [1, 2, 3]] to match x.
x = [[.1, .4, .5], [.2, .3, .5]]
dist.prob(x)  # Shape [2]

# alpha will be broadcast to shape [5, 7, 3] to match x.
x = [[...]]  # Shape [5, 7, 3]
dist.prob(x)  # Shape [5, 7]
```

Creates a 2-batch of 3-class distributions.

```python
alpha = [[1, 2, 3], [4, 5, 6]]  # Shape [2, 3]
dist = Dirichlet(alpha)

# x will be broadcast to [[2, 1, 0], [2, 1, 0]] to match alpha.
x = [.2, .3, .5]
dist.prob(x)  # Shape [2]
```
- - -

#### `tf.contrib.distributions.Dirichlet.__init__(alpha, validate_args=True, allow_nan_stats=False, name='Dirichlet')` {#Dirichlet.__init__}

Initialize a batch of Dirichlet distributions.

##### Args:


*  <b>`alpha`</b>: Positive floating point tensor with shape broadcastable to
    `[N1,..., Nm, k]` `m >= 0`.  Defines this as a batch of `N1 x ... x Nm`
     different `k` class Dirichlet distributions.
*  <b>`validate_args`</b>: Whether to assert valid values for parameters `alpha` and
    `x` in `prob` and `log_prob`.  If `False`, correct behavior is not
    guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to prefix Ops created by this distribution class.


*  <b>`Examples`</b>: 

```python
# Define 1-batch of 2-class Dirichlet distributions,
# also known as a Beta distribution.
dist = Dirichlet([1.1, 2.0])

# Define a 2-batch of 3-class distributions.
dist = Dirichlet([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```


- - -

#### `tf.contrib.distributions.Dirichlet.allow_nan_stats` {#Dirichlet.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Dirichlet.alpha` {#Dirichlet.alpha}

Shape parameter.


- - -

#### `tf.contrib.distributions.Dirichlet.batch_shape(name='batch_shape')` {#Dirichlet.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.Dirichlet.cdf(x, name='cdf')` {#Dirichlet.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.Dirichlet.dtype` {#Dirichlet.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.Dirichlet.entropy(name='entropy')` {#Dirichlet.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.Dirichlet.event_shape(name='event_shape')` {#Dirichlet.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.Dirichlet.get_batch_shape()` {#Dirichlet.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  batch shape


- - -

#### `tf.contrib.distributions.Dirichlet.get_event_shape()` {#Dirichlet.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  event shape


- - -

#### `tf.contrib.distributions.Dirichlet.is_continuous` {#Dirichlet.is_continuous}




- - -

#### `tf.contrib.distributions.Dirichlet.is_reparameterized` {#Dirichlet.is_reparameterized}




- - -

#### `tf.contrib.distributions.Dirichlet.log_cdf(x, name='log_cdf')` {#Dirichlet.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.Dirichlet.log_pdf(value, name='log_pdf')` {#Dirichlet.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Dirichlet.log_pmf(value, name='log_pmf')` {#Dirichlet.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Dirichlet.log_prob(x, name='log_prob')` {#Dirichlet.log_prob}

`Log(P[counts])`, computed for every batch member.

##### Args:


*  <b>`x`</b>: Non-negative tensor with dtype `dtype` and whose shape can
    be broadcast with `self.alpha`.  For fixed leading dimensions, the last
    dimension represents counts for the corresponding Dirichlet distribution
    in `self.alpha`. `x` is only legal if it sums up to one.
*  <b>`name`</b>: Name to give this Op, defaults to "log_prob".

##### Returns:

  Log probabilities for each record, shape `[N1,...,Nm]`.


- - -

#### `tf.contrib.distributions.Dirichlet.mean(name='mean')` {#Dirichlet.mean}

Mean of the distribution.


- - -

#### `tf.contrib.distributions.Dirichlet.mode(name='mode')` {#Dirichlet.mode}

Mode of the distribution.

Note that the mode for the Beta distribution is only defined
when `alpha > 1`. This returns the mode when `alpha > 1`,
and NaN otherwise. If `self.allow_nan_stats` is `False`, an exception
will be raised rather than returning `NaN`.

##### Args:


*  <b>`name`</b>: The name for this op.

##### Returns:

  Mode of the Dirichlet distribution.


- - -

#### `tf.contrib.distributions.Dirichlet.name` {#Dirichlet.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.Dirichlet.pdf(value, name='pdf')` {#Dirichlet.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Dirichlet.pmf(value, name='pmf')` {#Dirichlet.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Dirichlet.prob(x, name='prob')` {#Dirichlet.prob}

`P[x]`, computed for every batch member.

##### Args:


*  <b>`x`</b>: Non-negative tensor with dtype `dtype` and whose shape can
    be broadcast with `self.alpha`.  For fixed leading dimensions, the last
    dimension represents x for the corresponding Dirichlet distribution in
    `self.alpha` and `self.beta`. `x` is only legal if it sums up to one.
*  <b>`name`</b>: Name to give this Op, defaults to "prob".

##### Returns:

  Probabilities for each record, shape `[N1,...,Nm]`.


- - -

#### `tf.contrib.distributions.Dirichlet.sample(sample_shape=(), seed=None, name='sample')` {#Dirichlet.sample}

Generate samples of the specified shape for each batched distribution.

Note that a call to `sample()` without arguments will generate a single
sample per batched distribution.

##### Args:


*  <b>`sample_shape`</b>: `int32` `Tensor` or tuple or list. Shape of the generated
    samples.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of dtype `self.dtype` and shape
      `sample_shape + self.batch_shape + self.event_shape`.


- - -

#### `tf.contrib.distributions.Dirichlet.sample_n(n, seed=None, name='sample_n')` {#Dirichlet.sample_n}

Sample `n` observations from the distributions.

##### Args:


*  <b>`n`</b>: `Scalar`, type int32, the number of observations to sample.
*  <b>`seed`</b>: Python integer, the random seed.
*  <b>`name`</b>: The name to give this op.

##### Returns:


*  <b>`samples`</b>: `[n, ...]`, a `Tensor` of `n` samples for each
    of the distributions determined by broadcasting the hyperparameters.


- - -

#### `tf.contrib.distributions.Dirichlet.std(name='std')` {#Dirichlet.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.Dirichlet.validate_args` {#Dirichlet.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Dirichlet.variance(name='variance')` {#Dirichlet.variance}

Variance of the distribution.


