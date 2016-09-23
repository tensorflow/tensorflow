### `tf.contrib.crf.crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params)` {#crf_sequence_score}

Computes the unnormalized score for a tag sequence.

##### Args:


*  <b>`inputs`</b>: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
      to use as input to the CRF layer.
*  <b>`tag_indices`</b>: A [batch_size, max_seq_len] matrix of tag indices for which we
      compute the unnormalized score.
*  <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.
*  <b>`transition_params`</b>: A [num_tags, num_tags] transition matrix.

##### Returns:


*  <b>`sequence_scores`</b>: A [batch_size] vector of unnormalized sequence scores.

