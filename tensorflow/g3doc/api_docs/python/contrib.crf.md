<!-- This file is machine generated: DO NOT EDIT! -->

# CRF (contrib)
[TOC]

Linear-chain CRF layer.

## This package provides functions for building a linear-chain CRF layer.

- - -

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


- - -

### `tf.contrib.crf.crf_log_norm(inputs, sequence_lengths, transition_params)` {#crf_log_norm}

Computes the normalization for a CRF.

##### Args:


*  <b>`inputs`</b>: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
      to use as input to the CRF layer.
*  <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.
*  <b>`transition_params`</b>: A [num_tags, num_tags] transition matrix.

##### Returns:


*  <b>`log_norm`</b>: A [batch_size] vector of normalizers for a CRF.


- - -

### `tf.contrib.crf.crf_log_likelihood(inputs, tag_indices, sequence_lengths, transition_params=None)` {#crf_log_likelihood}

Computes the log-likehood of tag sequences in a CRF.

##### Args:


*  <b>`inputs`</b>: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
      to use as input to the CRF layer.
*  <b>`tag_indices`</b>: A [batch_size, max_seq_len] matrix of tag indices for which we
      compute the log-likehood.
*  <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.
*  <b>`transition_params`</b>: A [num_tags, num_tags] transition matrix, if available.

##### Returns:


*  <b>`log_likelihood`</b>: A scalar containing the log-likelihood of the given sequence
      of tag indices.
*  <b>`transition_params`</b>: A [num_tags, num_tags] transition matrix. This is either
      provided by the caller or created in this function.


- - -

### `tf.contrib.crf.crf_unary_score(tag_indices, sequence_lengths, inputs)` {#crf_unary_score}

Computes the unary scores of tag sequences.

##### Args:


*  <b>`tag_indices`</b>: A [batch_size, max_seq_len] matrix of tag indices.
*  <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.
*  <b>`inputs`</b>: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.

##### Returns:


*  <b>`unary_scores`</b>: A [batch_size] vector of unary scores.


- - -

### `tf.contrib.crf.crf_binary_score(tag_indices, sequence_lengths, transition_params)` {#crf_binary_score}

Computes the binary scores of tag sequences.

##### Args:


*  <b>`tag_indices`</b>: A [batch_size, max_seq_len] matrix of tag indices.
*  <b>`sequence_lengths`</b>: A [batch_size] vector of true sequence lengths.
*  <b>`transition_params`</b>: A [num_tags, num_tags] matrix of binary potentials.

##### Returns:


*  <b>`binary_scores`</b>: A [batch_size] vector of binary scores.


- - -

### `class tf.contrib.crf.CrfForwardRnnCell` {#CrfForwardRnnCell}

Computes the alpha values in a linear-chain CRF.

See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
- - -

#### `tf.contrib.crf.CrfForwardRnnCell.__call__(inputs, state, scope=None)` {#CrfForwardRnnCell.__call__}

Build the CrfForwardRnnCell.

##### Args:


*  <b>`inputs`</b>: A [batch_size, num_tags] matrix of unary potentials.
*  <b>`state`</b>: A [batch_size, num_tags] matrix containing the previous alpha
      values.
*  <b>`scope`</b>: Unused variable scope of this cell.

##### Returns:

  new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
      values containing the new alpha values.


- - -

#### `tf.contrib.crf.CrfForwardRnnCell.__init__(transition_params)` {#CrfForwardRnnCell.__init__}

Initialize the CrfForwardRnnCell.

##### Args:


*  <b>`transition_params`</b>: A [num_tags, num_tags] matrix of binary potentials.
      This matrix is expanded into a [1, num_tags, num_tags] in preparation
      for the broadcast summation occurring within the cell.


- - -

#### `tf.contrib.crf.CrfForwardRnnCell.output_size` {#CrfForwardRnnCell.output_size}




- - -

#### `tf.contrib.crf.CrfForwardRnnCell.state_size` {#CrfForwardRnnCell.state_size}




- - -

#### `tf.contrib.crf.CrfForwardRnnCell.zero_state(batch_size, dtype)` {#CrfForwardRnnCell.zero_state}

Return zero-filled state tensor(s).

##### Args:


*  <b>`batch_size`</b>: int, float, or unit Tensor representing the batch size.
*  <b>`dtype`</b>: the data type to use for the state.

##### Returns:

  If `state_size` is an int or TensorShape, then the return value is a
  `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

  If `state_size` is a nested list or tuple, then the return value is
  a nested list or tuple (of the same structure) of `2-D` tensors with
the shapes `[batch_size x s]` for each s in `state_size`.



- - -

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


