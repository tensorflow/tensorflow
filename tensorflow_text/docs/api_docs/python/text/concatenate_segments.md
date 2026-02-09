description: Concatenate input segments for a model's input sequence.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.concatenate_segments" />
<meta itemprop="path" content="Stable" />
</div>

# text.concatenate_segments

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/segment_combiner_ops.py">View
source</a>

Concatenate input segments for a model's input sequence.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.concatenate_segments(
    segments
)
</code></pre>

<!-- Placeholder for "Used in" -->

`concatenate_segments` combines the tokens of one or more input segments to a
single sequence of token values and generates matching segment ids.
`concatenate_segments` can follow a `Trimmer`, who limit segment lengths and
emit `RaggedTensor` outputs, and can be followed up by `ModelInputPacker`.

`concatenate_segments` first flattens and combines a list of one or more
segments (`RaggedTensor`s of n dimensions) together along the 1st axis, then
packages any special tokens into a final n dimensional `RaggedTensor`.

And finally `concatenate_segments` generates another `RaggedTensor` (with the
same rank as the final combined `RaggedTensor`) that contains a distinct int id
for each segment.

#### Example usage:

```
segment_a = [[1, 2],
             [3, 4,],
             [5, 6, 7, 8, 9]]

segment_b = [[10, 20,],
             [30, 40, 50, 60,],
             [70, 80]]
expected_combined, expected_ids = concatenate_segments([segment_a, segment_b])

# segment_a and segment_b have been concatenated as is.
expected_combined=[
 [1, 2, 10, 20],
 [3, 4, 30, 40, 50, 60],
 [5, 6, 7, 8, 9, 70, 80],
]

# ids describing which items belong to which segment.
expected_ids=[
 [0, 0, 1, 1],
 [0, 0, 1, 1, 1, 1],
 [0, 0, 0, 0, 0, 1, 1]]
```

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`segments`<a id="segments"></a>
</td>
<td>
A list of `RaggedTensor`s with the tokens of the input segments.
All elements must have the same dtype (int32 or int64), same rank, and
same dimension 0 (namely batch size). Slice `segments[i][j, ...]`
contains the tokens of the i-th input segment to the j-th example in the
batch.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a tuple of (combined_segments, segment_ids), where:
</td>
</tr>
<tr>
<td>
`combined_segments`<a id="combined_segments"></a>
</td>
<td>
A `RaggedTensor` with segments combined and special
tokens inserted.
</td>
</tr><tr>
<td>
`segment_ids`<a id="segment_ids"></a>
</td>
<td>
 A `RaggedTensor` w/ the same shape as `combined_segments`
and containing int ids for each item detailing the segment that they
correspond to.
</td>
</tr>
</table>
