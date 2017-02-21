### `tf.contrib.crf.crf_log_likelihood(inputs, tag_indices, sequence_lengths, transition_params=None)` {#crf_log_likelihood}

Computes the log-likelihood of tag sequences in a CRF.

##### Args:


*  <b>`inputs`</b>: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
      to use as input to the CRF layer.
*  <b>`tag_indices`</b>: A [batch_size, max_seq_len] matrix of tag indices for which we
      compute the log-likelihood.
*  <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.
*  <b>`transition_params`</b>: A [num_tags, num_tags] transition matrix, if available.

##### Returns:


*  <b>`log_likelihood`</b>: A scalar containing the log-likelihood of the given sequence
      of tag indices.
*  <b>`transition_params`</b>: A [num_tags, num_tags] transition matrix. This is either
      provided by the caller or created in this function.

