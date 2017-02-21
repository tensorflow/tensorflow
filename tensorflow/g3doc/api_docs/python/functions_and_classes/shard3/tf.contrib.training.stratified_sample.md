### `tf.contrib.training.stratified_sample(tensors, labels, target_probs, batch_size, init_probs=None, enqueue_many=False, queue_capacity=16, threads_per_queue=1, name=None)` {#stratified_sample}

Stochastically creates batches based on per-class probabilities.

This method discards examples. Internally, it creates one queue to amortize
the cost of disk reads, and one queue to hold the properly-proportioned
batch.

##### Args:


*  <b>`tensors`</b>: List of tensors for data. All tensors are either one item or a
      batch, according to enqueue_many.
*  <b>`labels`</b>: Tensor for label of data. Label is a single integer or a batch,
      depending on enqueue_many. It is not a one-hot vector.
*  <b>`target_probs`</b>: Target class proportions in batch. An object whose type has a
      registered Tensor conversion function.
*  <b>`batch_size`</b>: Size of batch to be returned.
*  <b>`init_probs`</b>: Class proportions in the data. An object whose type has a
      registered Tensor conversion function, or `None` for estimating the
      initial distribution.
*  <b>`enqueue_many`</b>: Bool. If true, interpret input tensors as having a batch
      dimension.
*  <b>`queue_capacity`</b>: Capacity of the large queue that holds input examples.
*  <b>`threads_per_queue`</b>: Number of threads for the large queue that holds input
      examples and for the final queue with the proper class proportions.
*  <b>`name`</b>: Optional prefix for ops created by this function.

##### Raises:


*  <b>`ValueError`</b>: enqueue_many is True and labels doesn't have a batch
      dimension, or if enqueue_many is False and labels isn't a scalar.
*  <b>`ValueError`</b>: enqueue_many is True, and batch dimension on data and labels
      don't match.
*  <b>`ValueError`</b>: if probs don't sum to one.
*  <b>`ValueError`</b>: if a zero initial probability class has a nonzero target
      probability.
*  <b>`TFAssertion`</b>: if labels aren't integers in [0, num classes).

##### Returns:

  (data_batch, label_batch), where data_batch is a list of tensors of the same
      length as `tensors`

##### Example:

  # Get tensor for a single data and label example.
  data, label = data_provider.Get(['data', 'label'])

  # Get stratified batch according to per-class probabilities.
  target_probs = [...distribution you want...]
  [data_batch], labels = tf.contrib.training.stratified_sample(
      [data], label, target_probs)

  # Run batch through network.
  ...

