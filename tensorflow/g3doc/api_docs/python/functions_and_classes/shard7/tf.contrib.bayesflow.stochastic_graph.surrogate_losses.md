### `tf.contrib.bayesflow.stochastic_graph.surrogate_losses(sample_losses, name='SurrogateLosses')` {#surrogate_losses}

Compute surrogate losses for StochasticTensors in the graph.

This function will call `surrogate_loss` on each `StochasticTensor` in the
graph and pass the losses in `sample_losses` that that `StochasticTensor`
influenced.

Note that currently `surrogate_losses` does not work with `StochasticTensor`s
instantiated in `while_loop`s or other control structures.

##### Args:


*  <b>`sample_losses`</b>: a list or tuple of final losses. Each loss should be per
    example in the batch (and possibly per sample); that is, it should have
    dimensionality of 1 or greater. All losses should have the same shape.
*  <b>`name`</b>: the name with which to prepend created ops.

##### Returns:

  A list of surrogate losses.

##### Raises:


*  <b>`TypeError`</b>: if `sample_losses` is not a list or tuple, or if its elements
    are not `Tensor`s.
*  <b>`ValueError`</b>: if any loss in `sample_losses` does not have dimensionality 1
    or greater.

