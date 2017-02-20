### `tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, labels=None, logits=None, name=None)` {#sparse_softmax_cross_entropy_with_logits}

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

A common use case is to have logits of shape `[batch_size, num_classes]` and
labels of shape `[batch_size]`. But higher dimensions are supported.

**Note that to avoid confusion, it is required to pass only named arguments to
this function.**

##### Args:

  _sentinel: Used to prevent positional parameters. Internal, do not use.

*  <b>`labels`</b>: `Tensor` of shape `[d_0, d_1, ..., d_{r-1}]` (where `r` is rank of
    `labels` and result) and dtype `int32` or `int64`. Each entry in `labels`
    must be an index in `[0, num_classes)`. Other values will raise an
    exception when this op is run on CPU, and return `NaN` for corresponding
    loss and gradient rows on GPU.
*  <b>`logits`</b>: Unscaled log probabilities of shape
    `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float32` or `float64`.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of the same shape as `labels` and of the same type as `logits`
  with the softmax cross entropy loss.

##### Raises:


*  <b>`ValueError`</b>: If logits are scalars (need to have rank >= 1) or if the rank
    of the labels is not equal to the rank of the labels minus one.

