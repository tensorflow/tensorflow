description: Normalizes a tensor of UTF-8 strings.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.FastBertNormalizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="normalize"/>
<meta itemprop="property" content="normalize_with_offsets"/>
</div>

# text.FastBertNormalizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_bert_normalizer.py">View
source</a>

Normalizes a tensor of UTF-8 strings.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.FastBertNormalizer(
    lower_case_nfd_strip_accents=False, model_buffer=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`lower_case_nfd_strip_accents`<a id="lower_case_nfd_strip_accents"></a>
</td>
<td>
(optional). - If true, it first lowercases
the text, applies NFD normalization, strips accents characters, and then
replaces control characters with whitespaces. - If false, it only
replaces control characters with whitespaces.
</td>
</tr><tr>
<td>
`model_buffer`<a id="model_buffer"></a>
</td>
<td>
(optional) bytes object (or a uint8 tf.Tenosr) that contains
the fast bert normalizer model in flatbuffer format (see
fast_bert_normalizer_model.fbs). If not `None`, all other arguments are
ignored.
</td>
</tr>
</table>

## Methods

<h3 id="normalize"><code>normalize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_bert_normalizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>normalize(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 strings.

### Example:

```
>>> texts = [["They're", "the", "Greatest", "\xC0bc"]]
>>> normalizer = FastBertNormalizer(lower_case_nfd_strip_accents=True)
>>> normalizer.normalize(texts)
<tf.RaggedTensor [[b"they're", b'the', b'greatest', b'abc']]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.
</td>
</tr>

</table>

<h3 id="normalize_with_offsets"><code>normalize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_bert_normalizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>normalize_with_offsets(
    input
)
</code></pre>

Normalizes a tensor of UTF-8 strings and returns offsets map.

### Example:

```
>>> texts = ["They're", "the", "Greatest", "\xC0bc"]
>>> normalizer = FastBertNormalizer(lower_case_nfd_strip_accents=True)
>>> normalized_text, offsets = (
...   normalizer.normalize_with_offsets(texts))
>>> normalized_text
<tf.Tensor: shape=(4,), dtype=string, numpy=array([b"they're", b'the',
b'greatest', b'abc'], dtype=object)>
>>> offsets
<tf.RaggedTensor [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5,
6, 7, 8], [0, 2, 3, 4]]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
An N-dimensional `Tensor` or `RaggedTensor` of UTF-8 strings.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple `(normalized_texts, offsets)` where:
</td>
</tr>
<tr>
<td>
`normalized_texts`
</td>
<td>
is a `Tensor` or `RaggedTensor`.
</td>
</tr><tr>
<td>
`offsets`
</td>
<td>
is a `RaggedTensor` of the byte offsets from the output
to the input. For example, if the input is `input[i1...iN]` with `N`
strings, `offsets[i1...iN, k]` is the byte offset in `inputs[i1...iN]`
for the `kth` byte in `normalized_texts[i1...iN]`. Note that
`offsets[i1...iN, ...]` also covers the position following the last byte
in `normalized_texts[i1...iN]`, so that we know the byte offset position
in `input[i1...iN]` that corresponds to the end of
`normalized_texts[i1...iN]`.
</td>
</tr>
</table>
