description: Maps the input post-normalized string offsets to pre-normalized offsets.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.find_source_offsets" />
<meta itemprop="path" content="Stable" />
</div>

# text.find_source_offsets

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/normalize_ops.py">View
source</a>

Maps the input post-normalized string offsets to pre-normalized offsets.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.find_source_offsets(
    offsets_map, input_offsets, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Returns the source (i.e. pre-normalized) string offsets mapped from the input
post-normalized string offsets using the input offsets_map, which is an output
from the `normalize_utf8_with_offsets_map` op. offsets_map can be indexed or
sliced along with the input_offsets.

#### Examples:

```
>>> # input: <string>[num_strings]
>>> post_normalized_str, offsets_map = normalize_utf8_with_offsets_map(
...     ["株式会社", "ＫＡＤＯＫＡＷＡ"])
>>> # input: <variant>[num_strings], <int64>[num_strings, num_offsets]
>>> find_source_offsets(offsets_map, [[0, 1, 2], [0, 1, 2]])
>>> # output: <int64>[num_strings, num_offsets]
<tf.Tensor: shape=(2, 3), dtype=int64, numpy=array([[0, 1, 2], [0, 3, 6]])>
>>> # Offsets map can be indexed.
>>> find_source_offsets(offsets_map[1], [[0, 1, 2]])
<tf.Tensor: shape=(1, 3), dtype=int64, numpy=array([[0, 3, 6]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`offsets_map`<a id="offsets_map"></a>
</td>
<td>
A `Tensor` or `RaggedTensor` of type `variant`, used to map the
post-normalized string offsets to pre-normalized string offsets.
offsets_map is an output from `normalize_utf8_with_offsets_map` function.
</td>
</tr><tr>
<td>
`input_offsets`<a id="input_offsets"></a>
</td>
<td>
A `Tensor` or `RaggedTensor` of type int64 representing the
the post-normalized string offsets,
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
The name for this op (optional).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`results`<a id="results"></a>
</td>
<td>
A `Tensor` or `RaggedTensor` of type int64, with pre-normalized
string offsets.
</td>
</tr>
</table>
