description: Add padding to the beginning and end of data in a specific
dimension.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.pad_along_dimension" />
<meta itemprop="path" content="Stable" />
</div>

# text.pad_along_dimension

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/pad_along_dimension_op.py">View
source</a>

Add padding to the beginning and end of data in a specific dimension.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.pad_along_dimension(
    data, axis=-1, left_pad=None, right_pad=None, name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Returns a tensor constructed from `data`, where each row in dimension `axis`
is replaced by the concatenation of the left padding followed by the row
followed by the right padding.  I.e., if `L=left_pad.shape[0]` and
`R=right_pad.shape[0]`, then:

```python
result[i1...iaxis, 0:L] = left_pad
result[i1...iaxis, L:-R] = data[i0...iaxis]
result[i1...iaxis, -R:] = right_pad
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`<a id="data"></a>
</td>
<td>
`<dtype>[O1...ON, A, I1...IM]` A potentially ragged `K` dimensional
tensor with outer dimensions of size `O1...ON`; axis dimension of size
`A`; and inner dimensions of size `I1...IM`.  I.e. `K = N + 1 + M`, where
`N>=0` and `M>=0`.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
An integer constant specifying the axis along which padding is added.
Negative axis values from `-K` to `-1` are supported.
</td>
</tr><tr>
<td>
`left_pad`<a id="left_pad"></a>
</td>
<td>
`<dtype>[L, I1...IM]` An `M+1` dimensional tensor that should be
prepended to each row along dimension `axis`; or `None` if no padding
should be added to the left side.
</td>
</tr><tr>
<td>
`right_pad`<a id="right_pad"></a>
</td>
<td>
`<dtype>[R, I1...IM]` An `M+1` dimensional tensor that should be
appended to each row along dimension `axis`; or `None` if no padding
should be added to the right side.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
The name of this op (optional).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
`<dtype>[O1...ON, L + A + R, I1...IM]`
A potentially ragged `K` dimensional tensor with outer dimensions of size
`O1...ON`; padded axis dimension size `L+A+R`; and inner dimensions of
size `I1...IM`.  If `data` is a `RaggedTensor`, then the returned tensor
is a `RaggedTensor` with the same `ragged_rank`.
</td>
</tr>

</table>
