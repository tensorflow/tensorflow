description: Tokenizer used for BERT, a faster version with TFLite support.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.FastBertTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.FastBertTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_bert_tokenizer.py">View
source</a>

Tokenizer used for BERT, a faster version with TFLite support.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md), [`Detokenizer`](../text/Detokenizer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.FastBertTokenizer(
    vocab=None,
    suffix_indicator=&#x27;##&#x27;,
    max_bytes_per_word=100,
    token_out_type=dtypes.int64,
    unknown_token=&#x27;[UNK]&#x27;,
    no_pretokenization=False,
    support_detokenization=False,
    fast_wordpiece_model_buffer=None,
    lower_case_nfd_strip_accents=False,
    fast_bert_normalizer_model_buffer=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

This tokenizer applies an end-to-end, text string to wordpiece tokenization. It
is equivalent to `BertTokenizer` for most common scenarios while running faster
and supporting TFLite. It does not support certain special settings (see the
docs below).

See `WordpieceTokenizer` for details on the subword tokenization.

For an example of use, see
https://www.tensorflow.org/text/guide/bert_preprocessing_guide

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`vocab`<a id="vocab"></a>
</td>
<td>
(optional) The list of tokens in the vocabulary.
</td>
</tr><tr>
<td>
`suffix_indicator`<a id="suffix_indicator"></a>
</td>
<td>
(optional) The characters prepended to a wordpiece to
indicate that it is a suffix to another subword.
</td>
</tr><tr>
<td>
`max_bytes_per_word`<a id="max_bytes_per_word"></a>
</td>
<td>
(optional) Max size of input token.
</td>
</tr><tr>
<td>
`token_out_type`<a id="token_out_type"></a>
</td>
<td>
(optional) The type of the token to return. This can be
`tf.int64` or `tf.int32` IDs, or `tf.string` subwords.
</td>
</tr><tr>
<td>
`unknown_token`<a id="unknown_token"></a>
</td>
<td>
(optional) The string value to substitute for an unknown
token. It must be included in `vocab`.
</td>
</tr><tr>
<td>
`no_pretokenization`<a id="no_pretokenization"></a>
</td>
<td>
(optional) By default, the input is split on
whitespaces and punctuations before applying the Wordpiece tokenization.
When true, the input is assumed to be pretokenized already.
</td>
</tr><tr>
<td>
`support_detokenization`<a id="support_detokenization"></a>
</td>
<td>
(optional) Whether to make the tokenizer support
doing detokenization. Setting it to true expands the size of the model
flatbuffer. As a reference, when using 120k multilingual BERT WordPiece
vocab, the flatbuffer's size increases from ~5MB to ~6MB.
</td>
</tr><tr>
<td>
`fast_wordpiece_model_buffer`<a id="fast_wordpiece_model_buffer"></a>
</td>
<td>
(optional) Bytes object (or a uint8 tf.Tenosr)
that contains the wordpiece model in flatbuffer format (see
fast_wordpiece_tokenizer_model.fbs). If not `None`, all other arguments
related to FastWordPieceTokenizer (except `token_output_type`) are
ignored.
</td>
</tr><tr>
<td>
`lower_case_nfd_strip_accents`<a id="lower_case_nfd_strip_accents"></a>
</td>
<td>
(optional) .
- If true, it first lowercases the text, applies NFD normalization, strips
accents characters, and then replaces control characters with whitespaces.
- If false, it only replaces control characters with whitespaces.
</td>
</tr><tr>
<td>
`fast_bert_normalizer_model_buffer`<a id="fast_bert_normalizer_model_buffer"></a>
</td>
<td>
(optional) bytes object (or a uint8
tf.Tenosr) that contains the fast bert normalizer model in flatbuffer
format (see fast_bert_normalizer_model.fbs). If not `None`,
`lower_case_nfd_strip_accents` is ignored.
</td>
</tr>
</table>

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_bert_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>detokenize(
    token_ids
)
</code></pre>

Convert a `Tensor` or `RaggedTensor` of wordpiece IDs to string-words.

See
<a href="../text/WordpieceTokenizer.md#detokenize"><code>WordpieceTokenizer.detokenize</code></a>
for details.

Note:
<a href="../text/FastBertTokenizer.md#tokenize"><code>FastBertTokenizer.tokenize</code></a>/<a href="../text/FastBertTokenizer.md#detokenize"><code>FastBertTokenizer.detokenize</code></a>
does not round trip losslessly. The result of `detokenize` will not, in general,
have the same content or offsets as the input to `tokenize`. This is because the
"basic tokenization" step, that splits the strings into words before applying
the `WordpieceTokenizer`, includes irreversible steps like lower-casing and
splitting on punctuation. `WordpieceTokenizer` on the other hand **is**
reversible.

Note: This method assumes wordpiece IDs are dense on the interval `[0,
vocab_size)`.

#### Example:

```
>>> vocab = ['they', "##'", '##re', 'the', 'great', '##est', '[UNK]']
>>> tokenizer = FastBertTokenizer(vocab=vocab, support_detokenization=True)
>>> tokenizer.detokenize([[4, 5]])
<tf.Tensor: shape=(1,), dtype=string, numpy=array([b'greatest'],
dtype=object)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`token_ids`
</td>
<td>
A `RaggedTensor` or `Tensor` with an int dtype.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` with dtype `string` and the same rank as the input
`token_ids`.
</td>
</tr>

</table>

<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split(
    input
)
</code></pre>

Alias for
<a href="../text/Tokenizer.md#tokenize"><code>Tokenizer.tokenize</code></a>.

<h3 id="split_with_offsets"><code>split_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split_with_offsets(
    input
)
</code></pre>

Alias for
<a href="../text/TokenizerWithOffsets.md#tokenize_with_offsets"><code>TokenizerWithOffsets.tokenize_with_offsets</code></a>.

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_bert_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    text_input
)
</code></pre>

Tokenizes a tensor of string tokens into subword tokens for BERT.

#### Example:

```
>>> vocab = ['they', "##'", '##re', 'the', 'great', '##est', '[UNK]']
>>> tokenizer = FastBertTokenizer(vocab=vocab)
>>> text_inputs = tf.constant(['greatest'.encode('utf-8') ])
>>> tokenizer.tokenize(text_inputs)
<tf.RaggedTensor [[4, 5]]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`text_input`
</td>
<td>
input: A `Tensor` or `RaggedTensor` of untokenized UTF-8
strings.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` of tokens where `tokens[i1...iN, j]` is the string
contents (or ID in the vocab_lookup_table representing that string)
of the `jth` token in `input[i1...iN]`
</td>
</tr>

</table>

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_bert_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    text_input
)
</code></pre>

Tokenizes a tensor of string tokens into subword tokens for BERT.

#### Example:

```
>>> vocab = ['they', "##'", '##re', 'the', 'great', '##est', '[UNK]']
>>> tokenizer = FastBertTokenizer(vocab=vocab)
>>> text_inputs = tf.constant(['greatest'.encode('utf-8')])
>>> tokenizer.tokenize_with_offsets(text_inputs)
(<tf.RaggedTensor [[4, 5]]>,
 <tf.RaggedTensor [[0, 5]]>,
 <tf.RaggedTensor [[5, 8]]>)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`text_input`
</td>
<td>
input: A `Tensor` or `RaggedTensor` of untokenized UTF-8
strings.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of `RaggedTensor`s where the first element is the tokens where
`tokens[i1...iN, j]`, the second element is the starting offsets, the
third element is the end offset. (Please look at `tokenize` for details
on tokens.)
</td>
</tr>

</table>
