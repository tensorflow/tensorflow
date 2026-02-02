### `tf.nn.ctc_loss(inputs, labels, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True)` {#ctc_loss}

Computes the CTC (Connectionist Temporal Classification) Loss.

This op implements the CTC loss as presented in the article:

A. Graves, S. Fernandez, F. Gomez, J. Schmidhuber.
Connectionist Temporal Classification: Labelling Unsegmented Sequence Data
with Recurrent Neural Networks. ICML 2006, Pittsburgh, USA, pp. 369-376.

http://www.cs.toronto.edu/~graves/icml_2006.pdf

Input requirements:

```
sequence_length(b) <= time for all b

max(labels.indices(labels.indices[:, 1] == b, 2))
  <= sequence_length(b) for all b.
```

Regarding the arguments `preprocess_collapse_repeated` and
`ctc_merge_repeated`:

If `preprocess_collapse_repeated` is True, then a preprocessing step runs
before loss calculation, wherein repeated labels passed to the loss
are merged into single labels.  This is useful if the training labels come
from, e.g., forced alignments and therefore have unnecessary repetitions.

If `ctc_merge_repeated` is set False, then deep within the CTC calculation,
repeated non-blank labels will not be merged and are interpreted
as individual labels.  This is a simplified (non-standard) version of CTC.

Here is a table of the (roughly) expected first order behavior:

* `preprocess_collapse_repeated=False`, `ctc_merge_repeated=True`

  Classical CTC behavior: Outputs true repeated classes with blanks in
  between, and can also output repeated classes with no blanks in
  between that need to be collapsed by the decoder.

* `preprocess_collapse_repeated=True`, `ctc_merge_repeated=False`

  Never learns to output repeated classes, as they are collapsed
  in the input labels before training.

* `preprocess_collapse_repeated=False`, `ctc_merge_repeated=False`

  Outputs repeated classes with blanks in between, but generally does not
  require the decoder to collapse/merge repeated classes.

* `preprocess_collapse_repeated=True`, `ctc_merge_repeated=True`

  Untested.  Very likely will not learn to output repeated classes.

##### Args:


*  <b>`inputs`</b>: 3-D `float` `Tensor` sized
    `[max_time x batch_size x num_classes]`.  The logits.
*  <b>`labels`</b>: An `int32` `SparseTensor`.
    `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores
    the id for (batch b, time t).  See `core/ops/ctc_ops.cc` for more details.
*  <b>`sequence_length`</b>: 1-D `int32` vector, size `[batch_size]`.
    The sequence lengths.
*  <b>`preprocess_collapse_repeated`</b>: Boolean.  Default: False.
    If True, repeated labels are collapsed prior to the CTC calculation.
*  <b>`ctc_merge_repeated`</b>: Boolean.  Default: True.

##### Returns:

  A 1-D `float` `Tensor`, size `[batch]`, containing the negative log probabilities.

##### Raises:


*  <b>`TypeError`</b>: if labels is not a `SparseTensor`.

