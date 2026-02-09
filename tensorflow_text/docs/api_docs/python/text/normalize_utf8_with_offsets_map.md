description: Normalizes each UTF-8 string in the input tensor using the specified rule.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.normalize_utf8_with_offsets_map" />
<meta itemprop="path" content="Stable" />
</div>

# text.normalize_utf8_with_offsets_map

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/normalize_ops.py">View
source</a>

Normalizes each UTF-8 string in the input tensor using the specified rule.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.normalize_utf8_with_offsets_map(
    input, normalization_form=&#x27;NFKC&#x27;, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Returns normalized strings and an offset map used by another operation to map
post-normalized string offsets to pre-normalized string offsets.

See http://unicode.org/reports/tr15/

#### Examples:

```
>>> # input: <string>[num_strings]
>>> normalize_utf8_with_offsets_map(["株式会社", "ＫＡＤＯＫＡＷＡ"])
>>> # output: <string>[num_strings], <variant>[num_strings]
NormalizeUTF8WithOffsetsMap(output=<tf.Tensor: shape=(2,), dtype=string,
numpy=
array([b'\xe6\xa0\xaa\xe5\xbc\x8f\xe4\xbc\x9a\xe7\xa4\xbe', b'KADOKAWA'],
      dtype=object)>, offsets_map=<tf.Tensor: shape=(2,), dtype=variant,
      numpy=<unprintable>>)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A `Tensor` or `RaggedTensor` of type string. (Must be UTF-8.)
</td>
</tr><tr>
<td>
`normalization_form`<a id="normalization_form"></a>
</td>
<td>
One of the following string values ('NFC', 'NFKC',
'NFD', 'NFKD'). Default is 'NFKC'. NOTE: `NFD` and `NFKD` for
`normalize_utf8_with_offsets_map` will not be available until the
tf.text release w/ ICU 69 (scheduled after 4/2021).
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
<tr class="alt">
<td colspan="2">
A tuple of (results, offsets_map) where:
</td>
</tr>
<tr>
<td>
`results`<a id="results"></a>
</td>
<td>
A `Tensor` or `RaggedTensor` of type string, with normalized
contents.
</td>
</tr><tr>
<td>
`offsets_map`<a id="offsets_map"></a>
</td>
<td>
A `Tensor` or `RaggedTensor` of type `variant`, used to map
the post-normalized string offsets to pre-normalized string offsets. It
has the same shape as the results tensor. offsets_map is an input to
`find_source_offsets` op.
</td>
</tr>
</table>
