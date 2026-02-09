description: Converts the given tokens and spans in offsets format into BOISE
tags.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.offsets_to_boise_tags" />
<meta itemprop="path" content="Stable" />
</div>

# text.offsets_to_boise_tags

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/boise_offset_converter.py">View
source</a>

Converts the given tokens and spans in offsets format into BOISE tags.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.offsets_to_boise_tags(
    token_begin_offsets,
    token_end_offsets,
    span_begin_offsets,
    span_end_offsets,
    span_type,
    use_strict_boundary_mode=False
)
</code></pre>

<!-- Placeholder for "Used in" -->

In the BOISE scheme there is a set of 5 labels for each type: - (B)egin: meaning
the beginning of the span type. - (O)utside: meaning the token is outside of any
span type - (I)nside: the token is inside the span - (S)ingleton: the entire
span consists of this single token. - (E)nd: this token is the end of the span.

When given the span begin & end offsets along with a set of token begin & end
offsets, this function helps translate which each token into one of the 5
labels.

For example, given the following example string and entity:

content = "Who let the dogs out" entity = "dogs" tokens = ["Who", "let", "the",
"dogs", "out"] token_begin_offsets = [0, 4, 8, 12, 17] token_end_offsets = [3,
7, 11, 16, 20] span_begin_offsets = [12] span_end_offsets = [16] span_type =
["animal"]

Foo will produce the following labels: ["O", "O", "O", "S-animal", "O"] | | | |
| Who let the dogs out

Special Case 1: Loose or Strict Boundary Criteria: By default, loose boundary
criteria are used to decide token start and end, given a entity span. In the
above example, say if we have

span_begin_offsets = [13]; span_end_offsets = [16];

we still get ["O", "O", "O", "S-animal", "O"], even though the span begin offset
(13) is not exactly aligned with the token begin offset (12). Partial overlap
between a token and a BOISE tag still qualify the token to be labeled with this
tag.

You can choose to use strict boundary criteria by passing in
use_strict_boundary_mode = false argument, with which Foo will produce ["O",
"O", "O", "O", "O"] for the case described above.

Special Case 2: One Token Mapped to Multiple BOISE Tags: In cases where a token
is overlapped with multiple BOISE tags, we label the token with the last tag.
For example, given the following example inputs:

std::string content = "Getty Center"; std::vector<string> tokens = { "Getty
Center" }; std::vector<int> token_begin_offsets = { 0 }; std::vector<int>
token_end_offsets = { 12 }; std::vector<int> span_begin_offsets = { 0, 6 };
std::vector<int> span_end_offsets = { 5, 12 }; std::vector<string> span_type = {
"per", "loc" }

Foo will produce the following labels: ["B-loc"]

### Example:
```
>>> token_begin_offsets = tf.ragged.constant(
...   [[0, 4, 8, 12, 17], [0, 4, 8, 12]])
>>> token_end_offsets = tf.ragged.constant(
...   [[3, 7, 11, 16, 20], [3, 7, 11, 16]])
>>> span_begin_offsets = tf.ragged.constant([[4], [12]])
>>> span_end_offsets = tf.ragged.constant([[16], [16]])
>>> span_type = tf.ragged.constant([['animal'], ['loc']])
>>> boise_tags = tf_text.offsets_to_boise_tags(token_begin_offsets,
...   token_end_offsets, span_begin_offsets, span_end_offsets, span_type)
>>> boise_tags
<tf.RaggedTensor [[b'O', b'B-animal', b'I-animal', b'E-animal', b'O'],
[b'O', b'O', b'O', b'S-loc']]>
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
`span_begin_offsets`<a id="span_begin_offsets"></a>
</td>
<td>
A `RaggedTensor` or `Tensor` of span begin byte offsets
of int32 or int64.
</td>
</tr><tr>
<td>
`span_end_offsets`<a id="span_end_offsets"></a>
</td>
<td>
A `RaggedTensor` or `Tensor` of span end byte offsets of
int32 or int64.
</td>
</tr><tr>
<td>
`span_type`<a id="span_type"></a>
</td>
<td>
A `RaggedTensor` or `Tensor` of span type strings.
</td>
</tr><tr>
<td>
`use_strict_boundary_mode`<a id="use_strict_boundary_mode"></a>
</td>
<td>
A bool indicating whether to use the strict
boundary mode, which excludes a token from a span label when the token
begin/end byte range partially overlaps with the span range.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` of BOISE tag strings in the same dimension as the input
token begin and end offsets.
</td>
</tr>

</table>
