description: Converts the token offsets and BOISE tags into span offsets and
span type.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.boise_tags_to_offsets" />
<meta itemprop="path" content="Stable" />
</div>

# text.boise_tags_to_offsets

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/boise_offset_converter.py">View
source</a>

Converts the token offsets and BOISE tags into span offsets and span type.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.boise_tags_to_offsets(
    token_begin_offsets, token_end_offsets, boise_tags
)
</code></pre>

<!-- Placeholder for "Used in" -->

In the BOISE scheme there is a set of 5 labels for each type: - (B)egin: meaning
the beginning of the span type. - (O)utside: meaning the token is outside of any
span type - (I)nside: the token is inside the span - (S)ingleton: the entire
span consists of this single token. - (E)nd: this token is the end of the span.

For example, given the following example string and entity:

content = "Who let the dogs out" entity = "dogs" tokens = ["Who", "let", "the",
"dogs", "out"] token_begin_offsets = [0, 4, 8, 12, 17] token_end_offsets = [3,
7, 11, 16, 20] span_begin_offsets = [12] span_end_offsets = [16] span_type =
["animal"]

BOISE tags are: ["O", "O", "O", "S-animal", "O"] | | | | | Who let the dogs out

When given the token begin/end offsets and BOISE tags for an input text
sequence, this function translates them into entity span begin/end offsets and
span types.

### Example:

```
>>> token_begin_offsets = tf.ragged.constant(
...   [[0, 4, 8, 12, 17], [0, 4, 8, 12]])
>>> token_end_offsets = tf.ragged.constant(
...   [[3, 7, 11, 16, 20], [3, 7, 11, 16]])
>>> boise_tags = tf.ragged.constant(
...   [['O', 'B-animal', 'I-animal', 'E-animal', 'O'],
...    ['O', 'O', 'O', 'S-loc']])
>>> (span_begin_offsets, span_end_offsets, span_type) = (
...   tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
...     boise_tags))
>>> span_begin_offsets
<tf.RaggedTensor [[4], [12]]>
>>> span_end_offsets
<tf.RaggedTensor [[16], [16]]>
>>> span_type
<tf.RaggedTensor [[b'animal'], [b'loc']]>
```

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`token_begin_offsets`<a id="token_begin_offsets"></a>
</td>
<td>
A `RaggedTensor` or `Tensor` of token begin byte
offsets of int32 or int64.
</td>
</tr><tr>
<td>
`token_end_offsets`<a id="token_end_offsets"></a>
</td>
<td>
A `RaggedTensor` or `Tensor` of token end byte offsets of
int32 or int64.
</td>
</tr><tr>
<td>
`boise_tags`<a id="boise_tags"></a>
</td>
<td>
A `RaggedTensor` of BOISE tag strings in the same dimension as
the token begin and end offsets.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple containing `span_begin_offsets`, `span_end_offsets` and `span_type`.
`span_begin_offsets` is a `RaggedTensor` or `Tensor` of span begin byte
   offsets of int32 or int64.
`span_end_offsets` is a `RaggedTensor` or `Tensor` of span end byte offsets
   of int32 or int64.
`span_type` is a `RaggedTensor` or `Tensor` of span type strings.
</td>
</tr>

</table>
