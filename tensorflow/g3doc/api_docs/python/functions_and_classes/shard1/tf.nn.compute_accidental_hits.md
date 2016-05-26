### `tf.nn.compute_accidental_hits(true_classes, sampled_candidates, num_true, seed=None, name=None)` {#compute_accidental_hits}

Compute the position ids in `sampled_candidates` matching `true_classes`.

In Candidate Sampling, this operation facilitates virtually removing
sampled classes which happen to match target classes.  This is done
in Sampled Softmax and Sampled Logistic.

See our [Candidate Sampling Algorithms
Reference](http://www.tensorflow.org/extras/candidate_sampling.pdf).

We presuppose that the `sampled_candidates` are unique.

We call it an 'accidental hit' when one of the target classes
matches one of the sampled classes.  This operation reports
accidental hits as triples `(index, id, weight)`, where `index`
represents the row number in `true_classes`, `id` represents the
position in `sampled_candidates`, and weight is `-FLOAT_MAX`.

The result of this op should be passed through a `sparse_to_dense`
operation, then added to the logits of the sampled classes. This
removes the contradictory effect of accidentally sampling the true
target classes as noise classes for the same example.

##### Args:


*  <b>`true_classes`</b>: A `Tensor` of type `int64` and shape `[batch_size,
    num_true]`. The target classes.
*  <b>`sampled_candidates`</b>: A tensor of type `int64` and shape `[num_sampled]`.
    The sampled_candidates output of CandidateSampler.
*  <b>`num_true`</b>: An `int`.  The number of target classes per training example.
*  <b>`seed`</b>: An `int`. An operation-specific seed. Default is 0.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:


*  <b>`indices`</b>: A `Tensor` of type `int32` and shape `[num_accidental_hits]`.
    Values indicate rows in `true_classes`.
*  <b>`ids`</b>: A `Tensor` of type `int64` and shape `[num_accidental_hits]`.
    Values indicate positions in `sampled_candidates`.
*  <b>`weights`</b>: A `Tensor` of type `float` and shape `[num_accidental_hits]`.
    Each value is `-FLOAT_MAX`.

