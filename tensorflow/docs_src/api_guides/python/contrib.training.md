# Training (contrib)
[TOC]

Training and input utilities.

## Splitting sequence inputs into minibatches with state saving

Use `tf.contrib.training.SequenceQueueingStateSaver` or
its wrapper `tf.contrib.training.batch_sequences_with_states` if
you have input data with a dynamic primary time / frame count axis which
you'd like to convert into fixed size segments during minibatching, and would
like to store state in the forward direction across segments of an example.

*   `tf.contrib.training.batch_sequences_with_states`
*   `tf.contrib.training.NextQueuedSequenceBatch`
*   `tf.contrib.training.SequenceQueueingStateSaver`


## Online data resampling

To resample data with replacement on a per-example basis, use
`tf.contrib.training.rejection_sample` or
`tf.contrib.training.resample_at_rate`. For `rejection_sample`, provide
a boolean Tensor describing whether to accept or reject. Resulting batch sizes
are always the same. For `resample_at_rate`, provide the desired rate for each
example. Resulting batch sizes may vary. If you wish to specify relative
rates, rather than absolute ones, use `tf.contrib.training.weighted_resample`
(which also returns the actual resampling rate used for each output example).

Use `tf.contrib.training.stratified_sample` to resample without replacement
from the data to achieve a desired mix of class proportions that the Tensorflow
graph sees. For instance, if you have a binary classification dataset that is
99.9% class 1, a common approach is to resample from the data so that the data
is more balanced.

*   `tf.contrib.training.rejection_sample`
*   `tf.contrib.training.resample_at_rate`
*   `tf.contrib.training.stratified_sample`
*   `tf.contrib.training.weighted_resample`

## Bucketing

Use `tf.contrib.training.bucket` or
`tf.contrib.training.bucket_by_sequence_length` to stratify
minibatches into groups ("buckets").  Use `bucket_by_sequence_length`
with the argument `dynamic_pad=True` to receive minibatches of similarly
sized sequences for efficient training via `dynamic_rnn`.

*   `tf.contrib.training.bucket`
*   `tf.contrib.training.bucket_by_sequence_length`
