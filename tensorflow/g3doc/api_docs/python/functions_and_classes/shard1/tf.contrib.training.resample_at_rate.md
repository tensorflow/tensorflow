### `tf.contrib.training.resample_at_rate(inputs, rates, scope=None, seed=None, back_prop=False)` {#resample_at_rate}

Given `inputs` tensors, stochastically resamples each at a given rate.

For example, if the inputs are `[[a1, a2], [b1, b2]]` and the rates
tensor contains `[3, 1]`, then the return value may look like `[[a1,
a2, a1, a1], [b1, b2, b1, b1]]`. However, many other outputs are
possible, since this is stochastic -- averaged over many repeated
calls, each set of inputs should appear in the output `rate` times
the number of invocations.

Uses Knuth's method to generate samples from the poisson
distribution (but instead of just incrementing a count, actually
emits the input); this is described at
https://en.wikipedia.org/wiki/Poisson_distribution in the section on
generating Poisson-distributed random variables.

Note that this method is not appropriate for large rate values: with
float16 it will stop performing correctly for rates above 9.17;
float32, 87; and float64, 708. (These are the base-e versions of the
minimum representable exponent for each type.)

##### Args:


*  <b>`inputs`</b>: A list of tensors, each of which has a shape of `[batch_size, ...]`
*  <b>`rates`</b>: A tensor of shape `[batch_size]` contiaining the resampling rates
         for each input.
*  <b>`scope`</b>: Scope for the op.
*  <b>`seed`</b>: Random seed to use.
*  <b>`back_prop`</b>: Whether to allow back-propagation through this op.

##### Returns:

  Selections from the input tensors.

