description: Normalizes each UTF-8 string in the input tensor using the
specified rule.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.normalize_utf8" />
<meta itemprop="path" content="Stable" />
</div>

# text.normalize_utf8

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/normalize_ops.py">View
source</a>

Normalizes each UTF-8 string in the input tensor using the specified rule.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.normalize_utf8(
    input, normalization_form=&#x27;NFKC&#x27;, name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

See http://unicode.org/reports/tr15/

#### Examples:

```
>>> # input: <string>[num_strings]
>>> normalize_utf8(["株式会社", "ＫＡＤＯＫＡＷＡ"])
>>> # output: <string>[num_strings]
<tf.Tensor: shape=(2,), dtype=string, numpy=
array([b'\xe6\xa0\xaa\xe5\xbc\x8f\xe4\xbc\x9a\xe7\xa4\xbe', b'KADOKAWA'],
      dtype=object)>
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
'NFD', 'NFKD'). Default is 'NFKC'.
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
A `Tensor` or `RaggedTensor` of type string, with normalized contents.
</td>
</tr>

</table>
