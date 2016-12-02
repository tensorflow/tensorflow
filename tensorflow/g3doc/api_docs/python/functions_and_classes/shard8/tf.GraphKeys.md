Standard names to use for graph collections.

The standard library uses various well-known names to collect and
retrieve values associated with a graph. For example, the
`tf.Optimizer` subclasses default to optimizing the variables
collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
specified, but it is also possible to pass an explicit list of
variables.

The following standard keys are defined:

* `GLOBAL_VARIABLES`: the default collection of `Variable` objects, shared
  across distributed environment (model variables are subset of these). See
  [`tf.global_variables()`](../../api_docs/python/state_ops.md#global_variables)
  for more details.
  Commonly, all `TRAINABLE_VARIABLES` variables will be in `MODEL_VARIABLES`,
  and all `MODEL_VARIABLES` variables will be in `GLOBAL_VARIABLES`.
* `LOCAL_VARIABLES`: the subset of `Variable` objects that are local to each
  machine. Usually used for temporarily variables, like counters.
  Note: use `tf.contrib.framework.local_variable` to add to this collection.
* `MODEL_VARIABLES`: the subset of `Variable` objects that are used in the
  model for inference (feed forward). Note: use
  `tf.contrib.framework.model_variable` to add to this collection.
* `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
  be trained by an optimizer. See
  [`tf.trainable_variables()`](../../api_docs/python/state_ops.md#trainable_variables)
  for more details.
* `SUMMARIES`: the summary `Tensor` objects that have been created in the
  graph. See
  [`tf.summary.merge_all()`](../../api_docs/python/summary.md#merge_all)
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
