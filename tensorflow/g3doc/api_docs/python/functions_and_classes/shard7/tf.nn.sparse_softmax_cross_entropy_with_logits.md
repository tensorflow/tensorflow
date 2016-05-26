### `tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels, name=None)` {#sparse_softmax_cross_entropy_with_logits}

Computes sparse softmax cross entropy between `logits` and `labels`.

Measures the probability error in discrete classification tasks in which the
classes are mutually exclusive (each entry is in exactly one class).  For
example, each CIFAR-10 image is labeled with one and only one label: an image
can be a dog or a truck, but not both.

**NOTE:**  For this operation, the probability of a given label is considered
exclusive.  That is, soft classes are not allowed, and the `labels` vector
must provide a single specific index for the true class for each row of
`logits` (each minibatch entry).  For soft softmax classification with
a probability distribution for each entry, see
`softmax_cross_entropy_with_logits`.

**WARNING:** This op expects unscaled logits, since it performs a softmax
on `logits` internally for efficiency.  Do not call this op with the
output of `softmax`, as it will produce incorrect results.

`logits` must have the shape `[batch_size, num_classes]`
and dtype `float32` or `float64`.

`labels` must have the shape `[batch_size]` and dtype `int32` or `int64`.

##### Args:


*  <b>`logits`</b>: Unscaled log probabilities.
*  <b>`labels`</b>: Each entry `labels[i]` must be an index in `[0, num_classes)`. Other
    values will result in a loss of 0, but incorrect gradient computations.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
  softmax cross entropy loss.

