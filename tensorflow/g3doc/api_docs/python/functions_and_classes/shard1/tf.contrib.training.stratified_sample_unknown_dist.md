### `tf.contrib.training.stratified_sample_unknown_dist(tensors, labels, probs, batch_size, enqueue_many=False, queue_capacity=16, threads_per_queue=1, name=None)` {#stratified_sample_unknown_dist}

Stochastically creates batches based on per-class probabilities.

**NOTICE** This sampler can be significantly slower than `stratified_sample`
due to each thread discarding all examples not in its assigned class.

This uses a number of threads proportional to the number of classes. See
`stratified_sample` for an implementation that discards fewer examples and
uses a fixed number of threads. This function's only advantage over
`stratified_sample` is that the class data-distribution doesn't need to be
known ahead of time.

##### Args:


*  <b>`tensors`</b>: List of tensors for data. All tensors are either one item or a
      batch, according to enqueue_many.
*  <b>`labels`</b>: Tensor for label of data. Label is a single integer or a batch,
      depending on enqueue_many. It is not a one-hot vector.
*  <b>`probs`</b>: Target class probabilities. An object whose type has a registered
      Tensor conversion function.
*  <b>`batch_size`</b>: Size of batch to be returned.
*  <b>`enqueue_many`</b>: Bool. If true, interpret input tensors as having a batch
      dimension.
*  <b>`queue_capacity`</b>: Capacity of each per-class queue.
*  <b>`threads_per_queue`</b>: Number of threads for each per-class queue.
*  <b>`name`</b>: Optional prefix for ops created by this function.

##### Raises:


*  <b>`ValueError`</b>: enqueue_many is True and labels doesn't have a batch
      dimension, or if enqueue_many is False and labels isn't a scalar.
*  <b>`ValueError`</b>: enqueue_many is True, and batch dimension of data and labels
      don't match.
*  <b>`ValueError`</b>: if probs don't sum to one.
*  <b>`TFAssertion`</b>: if labels aren't integers in [0, num classes).

##### Returns:

  (data_batch, label_batch), where data_batch is a list of tensors of the same
      length as `tensors`

##### Example:

  # Get tensor for a single data and label example.
  data, label = data_provider.Get(['data', 'label'])

  # Get stratified batch according to per-class probabilities.
  init_probs = [1.0/NUM_CLASSES for _ in range(NUM_CLASSES)]
  [data_batch], labels = (
      tf.contrib.training.stratified_sample_unknown_dist(
          [data], label, init_probs, 16))

  # Run batch through network.
  ...

