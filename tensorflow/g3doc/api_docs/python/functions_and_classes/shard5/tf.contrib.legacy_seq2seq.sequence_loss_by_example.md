### `tf.contrib.legacy_seq2seq.sequence_loss_by_example(logits, targets, weights, average_across_timesteps=True, softmax_loss_function=None, name=None)` {#sequence_loss_by_example}

Weighted cross-entropy loss for a sequence of logits (per example).

##### Args:


*  <b>`logits`</b>: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
*  <b>`targets`</b>: List of 1D batch-sized int32 Tensors of the same length as logits.
*  <b>`weights`</b>: List of 1D batch-sized float-Tensors of the same length as logits.
*  <b>`average_across_timesteps`</b>: If set, divide the returned cost by the total
    label weight.
*  <b>`softmax_loss_function`</b>: Function (labels-batch, inputs-batch) -> loss-batch
    to be used instead of the standard softmax (the default if this is None).
*  <b>`name`</b>: Optional name for this operation, default: "sequence_loss_by_example".

##### Returns:

  1D batch-sized float Tensor: The log-perplexity for each sequence.

##### Raises:


*  <b>`ValueError`</b>: If len(logits) is different from len(targets) or len(weights).

