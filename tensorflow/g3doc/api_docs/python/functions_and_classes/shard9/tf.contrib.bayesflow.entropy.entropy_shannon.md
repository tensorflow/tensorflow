### `tf.contrib.bayesflow.entropy.entropy_shannon(p, z=None, n=None, seed=None, form=None, name='entropy_shannon')` {#entropy_shannon}

Monte Carlo or deterministic computation of Shannon's entropy.

Depending on the kwarg `form`, this `Op` returns either the analytic entropy
of the distribution `p`, or the sampled entropy:

```
-n^{-1} sum_{i=1}^n p.log_prob(z_i),  where z_i ~ p,
    \approx - E_p[ Log[p(Z)] ]
    = Entropy[p]
```

User supplies either `Tensor` of samples `z`, or number of samples to draw `n`

##### Args:


*  <b>`p`</b>: `tf.contrib.distributions.Distribution`
*  <b>`z`</b>: `Tensor` of samples from `p`, produced by `p.sample(n)` for some `n`.
*  <b>`n`</b>: Integer `Tensor`.  Number of samples to generate if `z` is not provided.
*  <b>`seed`</b>: Python integer to seed the random number generator.
*  <b>`form`</b>: Either `ELBOForms.analytic_entropy` (use formula for entropy of `q`)
    or `ELBOForms.sample` (sample estimate of entropy), or `ELBOForms.default`
    (attempt analytic entropy, fallback on sample).
    Default value is `ELBOForms.default`.
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:

  A `Tensor` with same `dtype` as `p`, and shape equal to `p.batch_shape`.

##### Raises:


*  <b>`ValueError`</b>: If `form` not handled by this function.
*  <b>`ValueError`</b>: If `form` is `ELBOForms.analytic_entropy` and `n` was provided.

