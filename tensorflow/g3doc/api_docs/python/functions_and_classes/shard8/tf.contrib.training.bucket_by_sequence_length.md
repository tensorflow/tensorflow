### `tf.contrib.training.bucket_by_sequence_length(input_length, tensors, batch_size, bucket_boundaries, num_threads=1, capacity=32, shapes=None, dynamic_pad=False, allow_smaller_final_batch=False, keep_input=True, shared_name=None, name=None)` {#bucket_by_sequence_length}

Lazy bucketing of inputs according to their length.

This method calls `tf.contrib.training.bucket` under the hood, after first
subdividing the bucket boundaries into separate buckets and identifying which
bucket the given `input_length` belongs to.  See the documentation for
`which_bucket` for details of the other arguments.

##### Args:


*  <b>`input_length`</b>: `int32` scalar `Tensor`, the sequence length of tensors.
*  <b>`tensors`</b>: The list or dictionary of tensors, representing a single element,
    to bucket.  Nested lists are not supported.
*  <b>`batch_size`</b>: The new batch size pulled from the queue (all queues will have
    the same size).  If a list is passed in then each bucket will have a
    different batch_size.
    (python int, int32 scalar or iterable of integers of length num_buckets).
*  <b>`bucket_boundaries`</b>: int list, increasing non-negative numbers.
    The edges of the buckets to use when bucketing tensors.  Two extra buckets
    are created, one for `input_length < bucket_boundaries[0]` and
    one for `input_length >= bucket_boundaries[-1]`.
*  <b>`num_threads`</b>: An integer.  The number of threads enqueuing `tensors`.
*  <b>`capacity`</b>: An integer. The maximum number of minibatches in the top queue,
    and also the maximum number of elements within each bucket.
*  <b>`shapes`</b>: (Optional) The shapes for each example.  Defaults to the
    inferred shapes for `tensors`.
*  <b>`dynamic_pad`</b>: Boolean.  Allow variable dimensions in input shapes.
    The given dimensions are padded upon dequeue so that tensors within a
    batch have the same shapes.
*  <b>`allow_smaller_final_batch`</b>: (Optional) Boolean. If `True`, allow the final
    batches to be smaller if there are insufficient items left in the queues.
*  <b>`keep_input`</b>: A `bool` scalar Tensor.  If provided, this tensor controls
    whether the input is added to the queue or not.  If it evaluates `True`,
    then `tensors` are added to the bucket; otherwise they are dropped.  This
    tensor essentially acts as a filtering mechanism.
*  <b>`shared_name`</b>: (Optional). If set, the queues will be shared under the given
    name across multiple sessions.
*  <b>`name`</b>: (Optional) A name for the operations.

##### Returns:

  A tuple `(sequence_length, outputs)` where `sequence_length` is
  a 1-D `Tensor` of size `batch_size` and `outputs` is a list or dictionary
  of batched, bucketed, outputs corresponding to elements of `tensors`.

##### Raises:


*  <b>`TypeError`</b>: if `bucket_boundaries` is not a list of python integers.
*  <b>`ValueError`</b>: if `bucket_boundaries` is empty or contains non-increasing
    values or if batch_size is a list and it's length doesn't equal the number
    of buckets.

