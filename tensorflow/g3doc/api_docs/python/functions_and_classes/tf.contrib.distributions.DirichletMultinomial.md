DirichletMultinomial mixture distribution.

This distribution is parameterized by a vector `alpha` of concentration
parameters for `k` classes.

#### Mathematical details

The Dirichlet Multinomial is a distribution over k-class count data, meaning
for each k-tuple of non-negative integer `counts = [c_1,...,c_k]`, we have a
probability of these draws being made from the distribution.  The distribution
has hyperparameters `alpha = (alpha_1,...,alpha_k)`, and probability mass
function (pmf):

```pmf(counts) = C! / (c_1!...c_k!) * Beta(alpha + c) / Beta(alpha)```

where above `C = sum_j c_j`, `N!` is `N` factorial, and
`Beta(x) = prod_j Gamma(x_j) / Gamma(sum_j x_j)` is the multivariate beta
function.

This is a mixture distribution in that `N` samples can be produced by:
  1. Choose class probabilities `p = (p_1,...,p_k) ~ Dir(alpha)`
  2. Draw integers `m = (m_1,...,m_k) ~ Multinomial(p, N)`

This class provides methods to create indexed batches of Dirichlet
Multinomial distributions.  If the provided `alpha` is rank 2 or higher, for
every fixed set of leading dimensions, the last dimension represents one
single Dirichlet Multinomial distribution.  When calling distribution
functions (e.g. `dist.pdf(counts)`), `alpha` and `counts` are broadcast to the
same shape (if possible).  In all cases, the last dimension of alpha/counts
represents single Dirichlet Multinomial distributions.

#### Examples

```python
alpha = [1, 2, 3]
dist = DirichletMultinomial(alpha)
```

Creates a 3-class distribution, with the 3rd class is most likely to be drawn.
The distribution functions can be evaluated on counts.

```python
# counts same shape as alpha.
counts = [0, 2, 0]
dist.pdf(counts)  # Shape []

# alpha will be broadcast to [[1, 2, 3], [1, 2, 3]] to match counts.
counts = [[11, 22, 33], [44, 55, 66]]
dist.pdf(counts)  # Shape [2]

# alpha will be broadcast to shape [5, 7, 3] to match counts.
counts = [[...]]  # Shape [5, 7, 3]
dist.pdf(counts)  # Shape [5, 7]
```

Creates a 2-batch of 3-class distributions.

```python
alpha = [[1, 2, 3], [4, 5, 6]]  # Shape [2, 3]
dist = DirichletMultinomial(alpha)

# counts will be broadcast to [[11, 22, 33], [11, 22, 33]] to match alpha.
counts = [11, 22, 33]
dist.pdf(counts)  # Shape [2]
```
- - -

#### `tf.contrib.distributions.DirichletMultinomial.__init__(alpha)` {#DirichletMultinomial.__init__}

Initialize a batch of DirichletMultinomial distributions.

##### Args:


*  <b>`alpha`</b>: Shape `[N1,..., Nn, k]` positive `float` or `double` tensor with
    `n >= 0`.  Defines this as a batch of `N1 x ... x Nn` different `k`
    class Dirichlet multinomial distributions.


*  <b>`Examples`</b>: 

```python
# Define 1-batch of 2-class Dirichlet multinomial distribution,
# also known as a beta-binomial.
dist = DirichletMultinomial([1.1, 2.0])

# Define a 2-batch of 3-class distributions.
dist = DirichletMultinomial([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```


- - -

#### `tf.contrib.distributions.DirichletMultinomial.alpha` {#DirichletMultinomial.alpha}

Parameters defining this distribution.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.cdf(x)` {#DirichletMultinomial.cdf}




- - -

#### `tf.contrib.distributions.DirichletMultinomial.dtype` {#DirichletMultinomial.dtype}




- - -

#### `tf.contrib.distributions.DirichletMultinomial.log_cdf(x)` {#DirichletMultinomial.log_cdf}




- - -

#### `tf.contrib.distributions.DirichletMultinomial.log_pmf(counts, name=None)` {#DirichletMultinomial.log_pmf}

`Log(P[counts])`, computed for every batch member.

For each batch of counts `[c_1,...,c_k]`, `P[counts]` is the probability
that after sampling `sum_j c_j` draws from this Dirichlet Multinomial
distribution, the number of draws falling in class `j` is `c_j`.  Note that
different sequences of draws can result in the same counts, thus the
probability includes a combinatorial coefficient.

##### Args:


*  <b>`counts`</b>: Non-negative `float`, `double`, or `int` tensor whose shape can
    be broadcast with `self.alpha`.  For fixed leading dimensions, the last
    dimension represents counts for the corresponding Dirichlet Multinomial
    distribution in `self.alpha`.
*  <b>`name`</b>: Name to give this Op, defaults to "log_pmf".

##### Returns:

  Log probabilities for each record, shape `[N1,...,Nn]`.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.mean` {#DirichletMultinomial.mean}

Class means for every batch member.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.num_classes` {#DirichletMultinomial.num_classes}

Tensor providing number of classes in each batch member.


- - -

#### `tf.contrib.distributions.DirichletMultinomial.pmf(counts, name=None)` {#DirichletMultinomial.pmf}

`P[counts]`, computed for every batch member.

For each batch of counts `[c_1,...,c_k]`, `P[counts]` is the probability
that after sampling `sum_j c_j` draws from this Dirichlet Multinomial
distribution, the number of draws falling in class `j` is `c_j`.  Note that
different sequences of draws can result in the same counts, thus the
probability includes a combinatorial coefficient.

##### Args:


*  <b>`counts`</b>: Non-negative `float`, `double`, or `int` tensor whose shape can
    be broadcast with `self.alpha`.  For fixed leading dimensions, the last
    dimension represents counts for the corresponding Dirichlet Multinomial
    distribution in `self.alpha`.
*  <b>`name`</b>: Name to give this Op, defaults to "pmf".

##### Returns:

  Probabilities for each record, shape `[N1,...,Nn]`.


