### `tf.contrib.bayesflow.stochastic_graph.surrogate_loss(sample_losses, stochastic_tensors=None, name='SurrogateLoss')` {#surrogate_loss}

Surrogate loss for stochastic graphs.

This function will call `loss_fn` on each `StochasticTensor`
upstream of `sample_losses`, passing the losses that it influenced.

Note that currently `surrogate_loss` does not work with `StochasticTensor`s
instantiated in `while_loop`s or other control structures.

##### Args:


*  <b>`sample_losses`</b>: a list or tuple of final losses. Each loss should be per
    example in the batch (and possibly per sample); that is, it should have
    dimensionality of 1 or greater. All losses should have the same shape.
*  <b>`stochastic_tensors`</b>: a list of `StochasticTensor`s to add loss terms for.
    If None, defaults to all `StochasticTensor`s in the graph upstream of
    the `Tensor`s in `sample_losses`.
*  <b>`name`</b>: the name with which to prepend created ops.

##### Returns:

  `Tensor` loss, which is the sum of `sample_losses` and the
  `loss_fn`s returned by the `StochasticTensor`s.

##### Raises:


*  <b>`TypeError`</b>: if `sample_losses` is not a list or tuple, or if its elements
    are not `Tensor`s.
*  <b>`ValueError`</b>: if any loss in `sample_losses` does not have dimensionality 1
    or greater.

