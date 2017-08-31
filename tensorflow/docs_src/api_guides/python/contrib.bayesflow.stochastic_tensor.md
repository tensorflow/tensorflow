# BayesFlow Stochastic Tensors (contrib)
[TOC]

Classes and helper functions for creating Stochastic Tensors.

`StochasticTensor` objects wrap `Distribution` objects.  Their
values may be samples from the underlying distribution, or the distribution
mean (as governed by `value_type`).  These objects provide a `loss`
method for use when sampling from a non-reparameterized distribution.
The `loss`method is used in conjunction with `stochastic_graph.surrogate_loss`
to produce a single differentiable loss in stochastic graphs having
both continuous and discrete stochastic nodes.

## Stochastic Tensor Classes

*   @{tf.contrib.bayesflow.stochastic_tensor.BaseStochasticTensor}
*   @{tf.contrib.bayesflow.stochastic_tensor.StochasticTensor}

## Stochastic Tensor Value Types

*   @{tf.contrib.bayesflow.stochastic_tensor.MeanValue}
*   @{tf.contrib.bayesflow.stochastic_tensor.SampleValue}
*   @{tf.contrib.bayesflow.stochastic_tensor.value_type}
*   @{tf.contrib.bayesflow.stochastic_tensor.get_current_value_type}
