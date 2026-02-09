description: Decode UTF8 tokens into code points and return their bits.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.utf8_binarize" />
<meta itemprop="path" content="Stable" />
</div>

# text.utf8_binarize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/utf8_binarize_op.py">View
source</a>

Decode UTF8 tokens into code points and return their bits.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.utf8_binarize(
    tokens, word_length=16, bits_per_char=24, replacement_char=65533, name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

See the [RetVec paper](https://arxiv.org/abs/2302.09207) for details.

#### Example:

```
>>> code_points = utf8_binarize("hello", word_length=3, bits_per_char=4)
>>> print(code_points.numpy())
[0. 0. 0. 1. 1. 0. 1. 0. 0. 0. 1. 1.]
```

The codepoints are encoded bitwise in the little-endian order. The inner
dimension of the output is always `word_length * bits_per_char`, because extra
characters are truncated / missing characters are padded, and `bits_per_char`
lowest bits of each codepoint is stored.

Decoding errors (which in applications are often replaced with the character
U+65533 "REPLACEMENT CHARACTER") are represented with `replacement_char`'s
`bits_per_char` lowest bits.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tokens`<a id="tokens"></a>
</td>
<td>
A `Tensor` of tokens (strings) with any shape.
</td>
</tr><tr>
<td>
`word_length`<a id="word_length"></a>
</td>
<td>
Number of Unicode characters to process per word (the rest are
silently ignored; the output is zero-padded).
</td>
</tr><tr>
<td>
`bits_per_char`<a id="bits_per_char"></a>
</td>
<td>
The number of lowest bits of the Unicode codepoint to encode.
</td>
</tr><tr>
<td>
`replacement_char`<a id="replacement_char"></a>
</td>
<td>
The Unicode codepoint to use on decoding errors.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
The op name (optional).
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of floating-point zero and one values corresponding to the bits
of the token characters' Unicode code points.
</td>
</tr>
<tr>
<td>
`Shape`<a id="Shape"></a>
</td>
<td>
`[<shape of`tokens`>, word_length * bits_per_char]`.
</td>
</tr>
</table>
