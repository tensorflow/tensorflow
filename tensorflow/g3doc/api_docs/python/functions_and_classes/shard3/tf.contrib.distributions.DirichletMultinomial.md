DirichletMultinomial mixture distribution.

This distribution is parameterized by a vector `alpha` of concentration
parameters for `k` classes and `n`, the counts per each class..

#### Mathematical details

The Dirichlet Multinomial is a distribution over k-class count data, meaning
for each k-tuple of non-negative integer `counts = [c_1,...,c_k]`, we have a
probability of these draws being made from the distribution.  The distribution
has hyperparameters `alpha = (alpha_1,...,alpha_k)`, and probability mass
function (pmf):

```pmf(counts) = N! / (n_1!...n_k!) * Beta(alpha + c) / Beta(alpha)```

where above `N = sum_j n_j`, `N!` is `N` factorial, and
`Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the multivariate beta
function.

This is a mixture distribution in that `M` samples can be produced by:
  1. Choose class probabilities `p = (p_1,...,p_k) ~ Dir(alpha)`
  2. Draw integers `m = (n_1,...,n_k) ~ Multinomial(N, p)`

This class provides methods to create indexed batches of Dirichlet
Multinomial distributions.  If the provided `alpha` is rank 2 or higher, for
every fixed set of leading dimensions, the last dimension represents one
single Dirichlet Multinomial distribution.  When calling distribution
functions (e.g. `dist.pmf(counts)`), `alpha` and `counts` are broadcast to the
same shape (if possible).  In all cases, the last dimension of alpha/counts
represents single Dirichlet Multinomial distributions.

#### Examples

```python
alpha = [1, 2, 3]
n = 2
dist = DirichletMultinomial(n, alpha)
```

Creates a 3-class distribution, with the 3rd class is most likely to be drawn.
The distribution functions can be evaluated on counts.

```python
# counts same shape as alpha.
counts = [0, 0, 2]
dist.pmf(counts)  # Shape []

# alpha will be broadcast to [[1, 2, 3], [1, 2, 3]] to match counts.
counts = [[1, 1, 0], [1, 0, 1]]
dist.pmf(counts)  # Shape [2]

# alpha will be broadcast to shape [5, 7, 3] to match counts.
counts = [[...]]  # Shape [5, 7, 3]
dist.pmf(counts)  # Shape [5, 7]
```

Creates a 2-batch of 3-class distributions.

```python
alpha = [[1, 2, 3], [4, 5, 6]]  # Shape [2, 3]
n = [3, 3]
dist = DirichletMultinomial(n, alpha)

# counts will be broadcast to [[2, 1, 0], [2, 1, 0]] to match alpha.
counts = [2, 1, 0]
dist.pmf(counts)  # Shape [2]
```
- - -

#### `tf.contrib.distributions.DirichletMultinomial.__init__(n, alpha, validate_args=True, allow_nan_stats=False, name='DirichletMultinomial')` {#DirichletMultinomial.__init__}

Initialize a batch of DirichletMultinomial distributions.

##### Args:


*  <b>`n`</b>: Non-negative floating point tensor, whose dtype is the same as
    `alpha`. The shape is broadcastable to `[N1,..., Nm]` with `m >= 0`.
    Defines this as a batch of `N1 x ... x Nm` different Dirichlet
    multinomial distributions. Its components should be equal to integer
    values.
*  <b>`alpha`</b>: Positive floating point tensor, whose dtype is the same as
    `n` with shape broadcastable to `[N1,..., Nm, k]` `m >= 0`.  Defines
    this as a batch of `N1 x ... x Nm` different `k` class Dirichlet
    multinomial distributions.
*  <b>`validate_args`</b>: Whether to assert valid values for parameters `alpha` and
    `n`, and `x` in `prob` and `log_prob`.  If `False`, correct behavior is
    not guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to prefix Ops created by this distribution class.


*  <b>`Examples`</b>: 

```python
# Define 1-batch of 2-class Dirichlet multinomial distribution,
# also known as a beta-binomial.
dist = DirichletMultinomial(2.0, [1.1, 2.0])

# Define a 2-batch of 3-class distributions.
dist = DirichletMultinomial([3., 4], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```


- - -

#### `tf.contrib.distributions.DirichletMultinomial.allow_nan_stats` {#DirichletMultinomial.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.alpha` {#DirichletMultinomial.alpha}

Parameter defining this distribution.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.batch_shape(name='batch_shape')` {#DirichletMultinomial.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.DirichletMultinomial.cdf(x, name='cdf')` {#DirichletMultinomial.cdf}




- - -

#### `tf.contrib.distributions.DirichletMultinomial.dtype` {#DirichletMultinomial.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.entropy(name='entropy')` {#DirichletMultinomial.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.event_shape(name='event_shape')` {#DirichletMultinomial.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.DirichletMultinomial.get_batch_shape()` {#DirichletMultinomial.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  batch shape


- - -

#### `tf.contrib.distributions.DirichletMultinomial.get_event_shape()` {#DirichletMultinomial.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  event shape


- - -

#### `tf.contrib.distributions.DirichletMultinomial.is_continuous` {#DirichletMultinomial.is_continuous}




- - -

#### `tf.contrib.distributions.DirichletMultinomial.is_reparameterized` {#DirichletMultinomial.is_reparameterized}




- - -

#### `tf.contrib.distributions.DirichletMultinomial.log_cdf(x, name='log_cdf')` {#DirichletMultinomial.log_cdf}




- - -

#### `tf.contrib.distributions.DirichletMultinomial.log_pdf(value, name='log_pdf')` {#DirichletMultinomial.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.log_pmf(value, name='log_pmf')` {#DirichletMultinomial.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.log_prob(counts, name='log_prob')` {#DirichletMultinomial.log_prob}

`Log(P[counts])`, computed for every batch member.

For each batch of counts `[n_1,...,n_k]`, `P[counts]` is the probability
that after sampling `n` draws from this Dirichlet Multinomial
distribution, the number of draws falling in class `j` is `n_j`.  Note that
different sequences of draws can result in the same counts, thus the
probability includes a combinatorial coefficient.

##### Args:


*  <b>`counts`</b>: Non-negative tensor with dtype `dtype` and whose shape can be
    broadcast with `self.alpha`.  For fixed leading dimensions, the last
    dimension represents counts for the corresponding Dirichlet Multinomial
    distribution in `self.alpha`. `counts` is only legal if it sums up to
    `n` and its components are equal to integer values.
*  <b>`name`</b>: Name to give this Op, defaults to "log_prob".

##### Returns:

  Log probabilities for each record, shape `[N1,...,Nn]`.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.mean(name='mean')` {#DirichletMultinomial.mean}

Class means for every batch member.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.mode(name='mode')` {#DirichletMultinomial.mode}

Mode of the distribution.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.n` {#DirichletMultinomial.n}

Parameter defining this distribution.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.name` {#DirichletMultinomial.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.pdf(value, name='pdf')` {#DirichletMultinomial.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.pmf(value, name='pmf')` {#DirichletMultinomial.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.prob(counts, name='prob')` {#DirichletMultinomial.prob}

`P[counts]`, computed for every batch member.

For each batch of counts `[c_1,...,c_k]`, `P[counts]` is the probability
that after sampling `sum_j c_j` draws from this Dirichlet Multinomial
distribution, the number of draws falling in class `j` is `c_j`.  Note that
different sequences of draws can result in the same counts, thus the
probability includes a combinatorial coefficient.

##### Args:


*  <b>`counts`</b>: Non-negative tensor with dtype `dtype` and whose shape can be
    broadcast with `self.alpha`.  For fixed leading dimensions, the last
    dimension represents counts for the corresponding Dirichlet Multinomial
    distribution in `self.alpha`. `counts` is only legal if it sums up to
    `n` and its components are equal to integer values.
*  <b>`name`</b>: Name to give this Op, defaults to "prob".

##### Returns:

  Probabilities for each record, shape `[N1,...,Nn]`.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.sample(sample_shape=(), seed=None, name='sample')` {#DirichletMultinomial.sample}

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

#### `tf.contrib.distributions.DirichletMultinomial.sample_n(n, seed=None, name='sample_n')` {#DirichletMultinomial.sample_n}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: scalar. Number of samples to draw from each distribution.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.std(name='std')` {#DirichletMultinomial.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.validate_args` {#DirichletMultinomial.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.variance(name='mean')` {#DirichletMultinomial.variance}

Class variances for every batch member.

The variance for each batch member is defined as the following:

```
Var(X_j) = n * alpha_j / alpha_0 * (1 - alpha_j / alpha_0) *
  (n + alpha_0) / (1 + alpha_0)
```

where `alpha_0 = sum_j alpha_j`.

The covariance between elements in a batch is defined as:

```
Cov(X_i, X_j) = -n * alpha_i * alpha_j / alpha_0 ** 2 *
  (n + alpha_0) / (1 + alpha_0)
```

##### Args:


*  <b>`name`</b>: The name for this op.

##### Returns:

  A `Tensor` representing the variances for each batch member.


