### `tf.contrib.bayesflow.entropy.elbo_ratio(log_p, q, z=None, n=None, seed=None, form=None, name='elbo_ratio')` {#elbo_ratio}

Estimate of the ratio appearing in the `ELBO` and `KL` divergence.

With `p(z) := exp{log_p(z)}`, this `Op` returns an approximation of

```
E_q[ Log[p(Z) / q(Z)] ]
```

The term `E_q[ Log[p(Z)] ]` is always computed as a sample mean.
The term `E_q[ Log[q(z)] ]` can be computed with samples, or an exact formula
if `q.entropy()` is defined.  This is controlled with the kwarg `form`.

This log-ratio appears in different contexts:

#### `KL[q || p]`

If `log_p(z) = Log[p(z)]` for distribution `p`, this `Op` approximates
the negative Kullback-Leibler divergence.

```
elbo_ratio(log_p, q, n=100) = -1 * KL[q || p],
KL[q || p] = E[ Log[q(Z)] - Log[p(Z)] ]
```

Note that if `p` is a `Distribution`, then `distributions.kl(q, p)` may be
defined and available as an exact result.

#### ELBO

If `log_p(z) = Log[p(z, x)]` is the log joint of a distribution `p`, this is
the Evidence Lower BOund (ELBO):

```
ELBO ~= E[ Log[p(Z, x)] - Log[q(Z)] ]
      = Log[p(x)] - KL[q || p]
     <= Log[p(x)]
```

User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

##### Args:


*  <b>`log_p`</b>: Callable mapping samples from `q` to `Tensors` with
    shape broadcastable to `q.batch_shape`.
    For example, `log_p` works "just like" `q.log_prob`.
*  <b>`q`</b>: `tf.contrib.distributions.BaseDistribution`.
*  <b>`z`</b>: `Tensor` of samples from `q`, produced by `q.sample_n`.
*  <b>`n`</b>: Integer `Tensor`.  Number of samples to generate if `z` is not provided.
*  <b>`seed`</b>: Python integer to seed the random number generator.
*  <b>`form`</b>: Either `ELBOForms.analytic_entropy` (use formula for entropy of `q`)
    or `ELBOForms.sample` (sample estimate of entropy), or `ELBOForms.default`
    (attempt analytic entropy, fallback on sample).
    Default value is `ELBOForms.default`.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  Scalar `Tensor` holding sample mean KL divergence.  `shape` is the batch
    shape of `q`, and `dtype` is the same as `q`.

##### Raises:


*  <b>`ValueError`</b>: If `form` is not handled by this function.

