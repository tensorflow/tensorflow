### `tf.nn.ctc_loss(inputs, labels, sequence_length, preprocess_collapse_repeated=False, ctc_merge_repeated=True)` {#ctc_loss}

Computes the CTC (Connectionist Temporal Classification) Loss.

See the article:

A. Graves, S. Fernandez, F. Gomez, J. Schmidhuber.
Connectionist Temporal Classification: Labelling Unsegmented Sequence Data
with Recurrent Neural Networks. ICML 2006, Pittsburgh, USA, pp. 369-376.

Input requirements:

```
sequence_length(b) <= time for all b

max(labels.indices(labels.indices[:, 1] == b, 2))
  <= sequence_length(b) for all b.
```

Regarding the arguments `preprocess_collapse_repeated` and
`ctc_merge_repeated`:

If `ctc_merge_repeated` is set False, then deep within the CTC calculation,
repeated non-blank labels will not be merged and are interpreted
as individual labels.  This is a simplified (non-standard) version of CTC.

Here is a table of the (roughly) expected first order behavior:

* `preprocess_collapse_repeated=False, ctc_merge_repeated=True`

  Classical CTC behavior: Outputs true repeated classes with nulls in
  between, and can also output repeated classes with no nulls in
  between that need to be collapsed by the decoder.

* `preprocess_collapse_repeated=True, ctc_merge_repeated=False`

  Never learns repeated class of the same class under any circumstances.

* `preprocess_collapse_repeated=False, ctc_merge_repeated=False`

  Outputs repeated classes with nulls in between, but generally does not
  require the decoder to collapse/merge repeated classes.

* `preprocess_collapse_repeated=True, ctc_merge_repeated=True`

  Untested.
```

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

  A 1-D `float` `Tensor`, size `[batch]`, containing logits.

##### Raises:


*  <b>`TypeError`</b>: if labels is not a `SparseTensor`.

