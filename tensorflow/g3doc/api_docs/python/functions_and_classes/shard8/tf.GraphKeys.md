Standard names to use for graph collections.

The standard library uses various well-known names to collect and
retrieve values associated with a graph. For example, the
`tf.Optimizer` subclasses default to optimizing the variables
collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
specified, but it is also possible to pass an explicit list of
variables.

The following standard keys are defined:

* `VARIABLES`: the `Variable` objects that comprise a model, and
  must be saved and restored together. See
  [`tf.all_variables()`](../../api_docs/python/state_ops.md#all_variables)
  for more details.
* `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
  be trained by an optimizer. See
  [`tf.trainable_variables()`](../../api_docs/python/state_ops.md#trainable_variables)
  for more details.
* `SUMMARIES`: the summary `Tensor` objects that have been created in the
  graph. See
  [`tf.merge_all_summaries()`](../../api_docs/python/train.md#merge_all_summaries)
  for more details.
* `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
  produce input for a computation. See
  [`tf.start_queue_runners()`](../../api_docs/python/train.md#start_queue_runners)
  for more details.
* `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
  keep moving averages.  See
  [`tf.moving_average_variables()`](../../api_docs/python/state_ops.md#moving_average_variables)
  for more details.
* `REGULARIZATION_LOSSES`: regularization losses collected during graph
  construction.
* `WEIGHTS`: weights inside neural network layers
* `BIASES`: biases inside neural network layers
* `ACTIVATIONS`: activations of neural network layers
