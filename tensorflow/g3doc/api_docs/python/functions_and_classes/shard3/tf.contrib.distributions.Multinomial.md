Multinomial distribution.

This distribution is parameterized by a vector `p` of probability
parameters for `k` classes and `n`, the counts per each class..

#### Mathematical details

The Multinomial is a distribution over k-class count data, meaning
for each k-tuple of non-negative integer `counts = [n_1,...,n_k]`, we have a
probability of these draws being made from the distribution.  The distribution
has hyperparameters `p = (p_1,...,p_k)`, and probability mass
function (pmf):

```pmf(counts) = n! / (n_1!...n_k!) * (p_1)^n_1*(p_2)^n_2*...(p_k)^n_k```

where above `n = sum_j n_j`, `n!` is `n` factorial.

#### Examples

Create a 3-class distribution, with the 3rd class is most likely to be drawn,
using logits..

```python
logits = [-50., -43, 0]
dist = Multinomial(n=4., logits=logits)
```

Create a 3-class distribution, with the 3rd class is most likely to be drawn.

```python
p = [.2, .3, .5]
dist = Multinomial(n=4., p=p)
```

The distribution functions can be evaluated on counts.

```python
# counts same shape as p.
counts = [1., 0, 3]
dist.prob(counts)  # Shape []

# p will be broadcast to [[.2, .3, .5], [.2, .3, .5]] to match counts.
counts = [[1., 2, 1], [2, 2, 0]]
dist.prob(counts)  # Shape [2]

# p will be broadcast to shape [5, 7, 3] to match counts.
counts = [[...]]  # Shape [5, 7, 3]
dist.prob(counts)  # Shape [5, 7]
```

Create a 2-batch of 3-class distributions.

```python
p = [[.1, .2, .7], [.3, .3, .4]]  # Shape [2, 3]
dist = Multinomial(n=[4., 5], p=p)

counts = [[2., 1, 1], [3, 1, 1]]
dist.prob(counts)  # Shape [2]
```
- - -

#### `tf.contrib.distributions.Multinomial.__init__(n, logits=None, p=None, validate_args=True, allow_nan_stats=False, name='Multinomial')` {#Multinomial.__init__}

Initialize a batch of Multinomial distributions.

##### Args:


*  <b>`n`</b>: Non-negative floating point tensor with shape broadcastable to
    `[N1,..., Nm]` with `m >= 0`. Defines this as a batch of
    `N1 x ... x Nm` different Multinomial distributions.  Its components
    should be equal to integer values.
*  <b>`logits`</b>: Floating point tensor representing the log-odds of a
    positive event with shape broadcastable to `[N1,..., Nm, k], m >= 0`,
    and the same dtype as `n`. Defines this as a batch of `N1 x ... x Nm`
    different `k` class Multinomial distributions.
*  <b>`p`</b>: Positive floating point tensor with shape broadcastable to
    `[N1,..., Nm, k]` `m >= 0` and same dtype as `n`.  Defines this as
    a batch of `N1 x ... x Nm` different `k` class Multinomial
    distributions. `p`'s components in the last portion of its shape should
    sum up to 1.
*  <b>`validate_args`</b>: Whether to assert valid values for parameters `n` and `p`,
    and `x` in `prob` and `log_prob`.  If `False`, correct behavior is not
    guaranteed.
*  <b>`allow_nan_stats`</b>: Boolean, default `False`.  If `False`, raise an
    exception if a statistic (e.g. mean/mode/etc...) is undefined for any
    batch member.  If `True`, batch members with valid parameters leading to
    undefined statistics will return NaN for this statistic.
*  <b>`name`</b>: The name to prefix Ops created by this distribution class.


*  <b>`Examples`</b>: 

```python
# Define 1-batch of 2-class multinomial distribution,
# also known as a Binomial distribution.
dist = Multinomial(n=2., p=[.1, .9])

# Define a 2-batch of 3-class distributions.
dist = Multinomial(n=[4., 5], p=[[.1, .3, .6], [.4, .05, .55]])
```


- - -

#### `tf.contrib.distributions.Multinomial.allow_nan_stats` {#Multinomial.allow_nan_stats}

Boolean describing behavior when a stat is undefined for batch member.


- - -

#### `tf.contrib.distributions.Multinomial.batch_shape(name='batch_shape')` {#Multinomial.batch_shape}

Batch dimensions of this instance as a 1-D int32 `Tensor`.

The product of the dimensions of the `batch_shape` is the number of
independent distributions of this kind the instance represents.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `batch_shape`


- - -

#### `tf.contrib.distributions.Multinomial.cdf(value, name='cdf')` {#Multinomial.cdf}

Cumulative distribution function.


- - -

#### `tf.contrib.distributions.Multinomial.dtype` {#Multinomial.dtype}

dtype of samples from this distribution.


- - -

#### `tf.contrib.distributions.Multinomial.entropy(name='entropy')` {#Multinomial.entropy}

Entropy of the distribution in nats.


- - -

#### `tf.contrib.distributions.Multinomial.event_shape(name='event_shape')` {#Multinomial.event_shape}

Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

##### Args:


*  <b>`name`</b>: name to give to the op

##### Returns:

  `Tensor` `event_shape`


- - -

#### `tf.contrib.distributions.Multinomial.get_batch_shape()` {#Multinomial.get_batch_shape}

`TensorShape` available at graph construction time.

Same meaning as `batch_shape`. May be only partially defined.

##### Returns:

  batch shape


- - -

#### `tf.contrib.distributions.Multinomial.get_event_shape()` {#Multinomial.get_event_shape}

`TensorShape` available at graph construction time.

Same meaning as `event_shape`. May be only partially defined.

##### Returns:

  event shape


- - -

#### `tf.contrib.distributions.Multinomial.is_continuous` {#Multinomial.is_continuous}




- - -

#### `tf.contrib.distributions.Multinomial.is_reparameterized` {#Multinomial.is_reparameterized}




- - -

#### `tf.contrib.distributions.Multinomial.log_cdf(value, name='log_cdf')` {#Multinomial.log_cdf}

Log CDF.


- - -

#### `tf.contrib.distributions.Multinomial.log_pdf(value, name='log_pdf')` {#Multinomial.log_pdf}

Log of the probability density function.


- - -

#### `tf.contrib.distributions.Multinomial.log_pmf(value, name='log_pmf')` {#Multinomial.log_pmf}

Log of the probability mass function.


- - -

#### `tf.contrib.distributions.Multinomial.log_prob(counts, name='log_prob')` {#Multinomial.log_prob}

`Log(P[counts])`, computed for every batch member.

For each batch of counts `[n_1,...,n_k]`, `P[counts]` is the probability
that after sampling `n` draws from this Multinomial distribution, the
number of draws falling in class `j` is `n_j`.  Note that different
sequences of draws can result in the same counts, thus the probability
includes a combinatorial coefficient.

##### Args:


*  <b>`counts`</b>: Non-negative tensor with dtype `dtype` and whose shape can
    be broadcast with `self.p` and `self.n`.  For fixed leading dimensions,
    the last dimension represents counts for the corresponding Multinomial
    distribution in `self.p`. `counts` is only legal if it sums up to `n`
    and its components are equal to integer values.
*  <b>`name`</b>: Name to give this Op, defaults to "log_prob".

##### Returns:

  Log probabilities for each record, shape `[N1,...,Nm]`.


- - -

#### `tf.contrib.distributions.Multinomial.logits` {#Multinomial.logits}

Log-odds.


- - -

#### `tf.contrib.distributions.Multinomial.mean(name='mean')` {#Multinomial.mean}

Mean of the distribution.


- - -

#### `tf.contrib.distributions.Multinomial.mode(name='mode')` {#Multinomial.mode}

Mode of the distribution.


- - -

#### `tf.contrib.distributions.Multinomial.n` {#Multinomial.n}

Number of trials.


- - -

#### `tf.contrib.distributions.Multinomial.name` {#Multinomial.name}

Name to prepend to all ops.


- - -

#### `tf.contrib.distributions.Multinomial.p` {#Multinomial.p}

Event probabilities.


- - -

#### `tf.contrib.distributions.Multinomial.pdf(value, name='pdf')` {#Multinomial.pdf}

The probability density function.


- - -

#### `tf.contrib.distributions.Multinomial.pmf(value, name='pmf')` {#Multinomial.pmf}

The probability mass function.


- - -

#### `tf.contrib.distributions.Multinomial.prob(counts, name='prob')` {#Multinomial.prob}

`P[counts]`, computed for every batch member.

For each batch of counts `[n_1,...,n_k]`, `P[counts]` is the probability
that after sampling `n` draws from this Multinomial distribution, the
number of draws falling in class `j` is `n_j`.  Note that different
sequences of draws can result in the same counts, thus the probability
includes a combinatorial coefficient.

##### Args:


*  <b>`counts`</b>: Non-negative tensor with dtype `dtype` and whose shape can
    be broadcast with `self.p` and `self.n`.  For fixed leading dimensions,
    the last dimension represents counts for the corresponding Multinomial
    distribution in `self.p`. `counts` is only legal if it sums up to `n`
    and its components are equal to integer values.
*  <b>`name`</b>: Name to give this Op, defaults to "prob".

##### Returns:

  Probabilities for each record, shape `[N1,...,Nm]`.


- - -

#### `tf.contrib.distributions.Multinomial.sample(sample_shape=(), seed=None, name='sample')` {#Multinomial.sample}

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

#### `tf.contrib.distributions.Multinomial.sample_n(n, seed=None, name='sample_n')` {#Multinomial.sample_n}

Generate `n` samples.

##### Args:


*  <b>`n`</b>: scalar. Number of samples to draw from each distribution.
*  <b>`seed`</b>: Python integer seed for RNG
*  <b>`name`</b>: name to give to the op.

##### Returns:


*  <b>`samples`</b>: a `Tensor` of shape `(n,) + self.batch_shape + self.event_shape`
      with values of type `self.dtype`.


- - -

#### `tf.contrib.distributions.Multinomial.std(name='std')` {#Multinomial.std}

Standard deviation of the distribution.


- - -

#### `tf.contrib.distributions.Multinomial.validate_args` {#Multinomial.validate_args}

Boolean describing behavior on invalid input.


- - -

#### `tf.contrib.distributions.Multinomial.variance(name='variance')` {#Multinomial.variance}

Variance of the distribution.


