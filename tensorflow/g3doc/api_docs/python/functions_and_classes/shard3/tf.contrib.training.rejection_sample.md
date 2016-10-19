### `tf.contrib.training.rejection_sample(tensors, accept_prob_fn, batch_size, queue_threads=1, enqueue_many=False, prebatch_capacity=16, prebatch_threads=1, runtime_checks=False, name=None)` {#rejection_sample}

Stochastically creates batches by rejection sampling.

Each list of non-batched tensors is evaluated by `accept_prob_fn`, to produce
a scalar tensor between 0 and 1. This tensor corresponds to the probability of
being accepted. When `batch_size` tensor groups have been accepted, the batch
queue will return a mini-batch.

##### Args:


*  <b>`tensors`</b>: List of tensors for data. All tensors are either one item or a
      batch, according to enqueue_many.
*  <b>`accept_prob_fn`</b>: A python lambda that takes a non-batch tensor from each
      item in `tensors`, and produces a scalar tensor.
*  <b>`batch_size`</b>: Size of batch to be returned.
*  <b>`queue_threads`</b>: The number of threads for the queue that will hold the final
    batch.
*  <b>`enqueue_many`</b>: Bool. If true, interpret input tensors as having a batch
      dimension.
*  <b>`prebatch_capacity`</b>: Capacity for the large queue that is used to convert
    batched tensors to single examples.
*  <b>`prebatch_threads`</b>: Number of threads for the large queue that is used to
    convert batched tensors to single examples.
*  <b>`runtime_checks`</b>: Bool. If true, insert runtime checks on the output of
      `accept_prob_fn`. Using `True` might have a performance impact.
*  <b>`name`</b>: Optional prefix for ops created by this function.

##### Raises:


*  <b>`ValueError`</b>: enqueue_many is True and labels doesn't have a batch
      dimension, or if enqueue_many is False and labels isn't a scalar.
*  <b>`ValueError`</b>: enqueue_many is True, and batch dimension on data and labels
      don't match.
*  <b>`ValueError`</b>: if a zero initial probability class has a nonzero target
      probability.

##### Returns:

  A list of tensors of the same length as `tensors`, with batch dimension
  `batch_size`.

##### Example:

  # Get tensor for a single data and label example.
  data, label = data_provider.Get(['data', 'label'])

  # Get stratified batch according to data tensor.
  accept_prob_fn = lambda x: (tf.tanh(x[0]) + 1) / 2
  data_batch = tf.contrib.training.rejection_sample(
      [data, label], accept_prob_fn, 16)

  # Run batch through network.
  ...

