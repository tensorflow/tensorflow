### `tf.contrib.bayesflow.entropy.renyi_alpha(step, decay_time, alpha_min, alpha_max=0.99999, name='renyi_alpha')` {#renyi_alpha}

Exponentially decaying `Tensor` appropriate for Renyi ratios.

When minimizing the Renyi divergence for `0 <= alpha < 1` (or maximizing the
Renyi equivalent of elbo) in high dimensions, it is not uncommon to experience
`NaN` and `inf` values when `alpha` is far from `1`.

For that reason, it is often desirable to start the optimization with `alpha`
very close to 1, and reduce it to a final `alpha_min` according to some
schedule.  The user may even want to optimize using `elbo_ratio` for
some fixed time before switching to Renyi based methods.

This `Op` returns an `alpha` decaying exponentially with step:

```
s(step) = (exp{step / decay_time} - 1) / (e - 1)
t(s) = max(0, min(s, 1)),  (smooth growth from 0 to 1)
alpha(t) = (1 - t) alpha_min + t alpha_max
```

##### Args:


*  <b>`step`</b>: Non-negative scalar `Tensor`.  Typically the global step or an
    offset version thereof.
*  <b>`decay_time`</b>: Postive scalar `Tensor`.
*  <b>`alpha_min`</b>: `float` or `double` `Tensor`.
    The minimal, final value of `alpha`, achieved when `step >= decay_time`
*  <b>`alpha_max`</b>: `Tensor` of same `dtype` as `alpha_min`.
    The maximal, beginning value of `alpha`, achieved when `step == 0`
*  <b>`name`</b>: A name to give this `Op`.

##### Returns:


*  <b>`alpha`</b>: A `Tensor` of same `dtype` as `alpha_min`.

