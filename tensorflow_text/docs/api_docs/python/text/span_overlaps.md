description: Returns a boolean tensor indicating which source and target spans
overlap.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.span_overlaps" />
<meta itemprop="path" content="Stable" />
</div>

# text.span_overlaps

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/pointer_ops.py">View
source</a>

Returns a boolean tensor indicating which source and target spans overlap.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.span_overlaps(
    source_start,
    source_limit,
    target_start,
    target_limit,
    contains=False,
    contained_by=False,
    partial_overlap=False,
    name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

The source and target spans are specified using B+1 dimensional tensors,
with `B>=0` batch dimensions followed by a final dimension that lists the
span offsets for each span in the batch:

* The `i`th source span in batch `b1...bB` starts at
  `source_start[b1...bB, i]` (inclusive), and extends to just before
  `source_limit[b1...bB, i]` (exclusive).
* The `j`th target span in batch `b1...bB` starts at
  `target_start[b1...bB, j]` (inclusive), and extends to just before
  `target_limit[b1...bB, j]` (exclusive).

`result[b1...bB, i, j]` is true if the `i`th source span overlaps with the
`j`th target span in batch `b1...bB`, where a source span overlaps a target
span if any of the following are true:

  * The spans are identical.
  * `contains` is true, and the source span contains the target span.
  * `contained_by` is true, and the source span is contained by the target
    span.
  * `partial_overlap` is true, and there is a non-zero overlap between the
    source span and the target span.

#### Example:

Given the following source and target spans (with no batch dimensions):

```
  >>>  #         0    5    10   15   20   25   30   35   40
  >>>  #         |====|====|====|====|====|====|====|====|
  >>>  # Source: [-0-]     [-1-] [2] [-3-][-4-][-5-]
  >>>  # Target: [-0-][-1-]     [-2-] [3]   [-4-][-5-]
  >>>  #         |====|====|====|====|====|====|====|====|
  >>> source_start = [0, 10, 16, 20, 25, 30]
  >>> source_limit = [5, 15, 19, 25, 30, 35]
  >>> target_start = [0,  5, 15, 21, 27, 31]
  >>> target_limit = [5, 10, 20, 24, 32, 37]
```

`result[i, j]` will be true at the following locations:

```
* `[0, 0]` (always)
* `[2, 2]` (if contained_by=True or partial_overlaps=True)
* `[3, 3]` (if contains=True or partial_overlaps=True)
* `[4, 4]` (if partial_overlaps=True)
* `[5, 4]` (if partial_overlaps=True)
* `[5, 5]` (if partial_overlaps=True)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`source_start`<a id="source_start"></a>
</td>
<td>
A B+1 dimensional potentially ragged tensor with shape
`[D1...DB, source_size]`: the start offset of each source span.
</td>
</tr><tr>
<td>
`source_limit`<a id="source_limit"></a>
</td>
<td>
A B+1 dimensional potentially ragged tensor with shape
`[D1...DB, source_size]`: the limit offset of each source span.
</td>
</tr><tr>
<td>
`target_start`<a id="target_start"></a>
</td>
<td>
A B+1 dimensional potentially ragged tensor with shape
`[D1...DB, target_size]`: the start offset of each target span.
</td>
</tr><tr>
<td>
`target_limit`<a id="target_limit"></a>
</td>
<td>
A B+1 dimensional potentially ragged tensor with shape
`[D1...DB, target_size]`: the limit offset of each target span.
</td>
</tr><tr>
<td>
`contains`<a id="contains"></a>
</td>
<td>
If true, then a source span is considered to overlap a target span
when the source span contains the target span.
</td>
</tr><tr>
<td>
`contained_by`<a id="contained_by"></a>
</td>
<td>
If true, then a source span is considered to overlap a target
span when the source span is contained by the target span.
</td>
</tr><tr>
<td>
`partial_overlap`<a id="partial_overlap"></a>
</td>
<td>
If true, then a source span is considered to overlap a
target span when the source span partially overlaps the target span.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A B+2 dimensional potentially ragged boolean tensor with shape
`[D1...DB, source_size, target_size]`.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If the span tensors are incompatible.
</td>
</tr>
</table>
