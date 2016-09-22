### `tf.contrib.crf.viterbi_decode(score, transition_params)` {#viterbi_decode}

Decode the highest scoring sequence of tags outside of TensorFlow.

This should only be used at test time.

##### Args:


*  <b>`score`</b>: A [seq_len, num_tags] matrix of unary potentials.
*  <b>`transition_params`</b>: A [num_tags, num_tags] matrix of binary potentials.

##### Returns:


*  <b>`viterbi`</b>: A [seq_len] list of integers containing the highest scoring tag
      indicies.
*  <b>`viterbi_score`</b>: A float containing the score for the viterbi sequence.

