description: Tokenizes a tensor of UTF-8 string tokens into subword pieces.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.FastWordpieceTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.FastWordpieceTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_wordpiece_tokenizer.py">View
source</a>

Tokenizes a tensor of UTF-8 string tokens into subword pieces.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md), [`Detokenizer`](../text/Detokenizer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.FastWordpieceTokenizer(
    vocab=None,
    suffix_indicator=&#x27;##&#x27;,
    max_bytes_per_word=100,
    token_out_type=dtypes.int64,
    unknown_token=&#x27;[UNK]&#x27;,
    no_pretokenization=False,
    support_detokenization=False,
    model_buffer=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

It employs the linear (as opposed to quadratic) WordPiece algorithm (see the
[paper](http://go/arxiv/2012.15524)).

Differences compared to the classic
[WordpieceTokenizer](https://www.tensorflow.org/text/api_docs/python/text/WordpieceTokenizer)
are as follows (as of 11/2021):

*   `unknown_token` cannot be None or empty. That means if a word is too long or
    cannot be tokenized, FastWordpieceTokenizer always returns `unknown_token`.
    In constrast, the original
    [WordpieceTokenizer](https://www.tensorflow.org/text/api_docs/python/text/WordpieceTokenizer)
    would return the original word if `unknown_token` is empty or None.

*   `unknown_token` must be included in the vocabulary.

*   When `unknown_token` is returned, in tokenize_with_offsets(), the result
    end_offset is set to be the length of the original input word. In contrast,
    when `unknown_token` is returned by the original
    [WordpieceTokenizer](https://www.tensorflow.org/text/api_docs/python/text/WordpieceTokenizer),
    the end_offset is set to be the length of the `unknown_token` string.

*   `split_unknown_characters` is not supported.

*   `max_chars_per_token` is not used or needed.

*   By default the input is assumed to be general text (i.e., sentences), and
    FastWordpieceTokenizer first splits it on whitespaces and punctuations and
    then applies the Wordpiece tokenization (see the parameter
    `no_pretokenization`). If the input already contains single words only,
    please set `no_pretokenization=True` to be consistent with the classic
    [WordpieceTokenizer](https://www.tensorflow.org/text/api_docs/python/text/WordpieceTokenizer).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

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
`model_buffer`<a id="model_buffer"></a>
</td>
<td>
(optional) Bytes object (or a uint8 tf.Tenosr) that contains
the wordpiece model in flatbuffer format (see
fast_wordpiece_tokenizer_model.fbs). If not `None`, all other arguments
(except `token_output_type`) are ignored.
</td>
</tr>
</table>

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_wordpiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>detokenize(
    input
)
</code></pre>

Detokenizes a tensor of int64 or int32 subword ids into sentences.

Detokenize and tokenize an input string returns itself when the input string is
normalized and the tokenized wordpieces don't contain `<unk>`.

### Example:
```
>>> vocab = ["they", "##'", "##re", "the", "great", "##est", "[UNK]",
...          "'", "re", "ok"]
>>> tokenizer = FastWordpieceTokenizer(vocab, support_detokenization=True)
>>> ids = tf.ragged.constant([[0, 1, 2, 3, 4, 5], [9]])
>>> tokenizer.detokenize(ids)
<tf.Tensor: shape=(2,), dtype=string,
...         numpy=array([b"they're the greatest", b'ok'], dtype=object)>
>>> ragged_ids = tf.ragged.constant([[[0, 1, 2, 3, 4, 5], [9]], [[4, 5]]])
>>> tokenizer.detokenize(ragged_ids)
<tf.RaggedTensor [[b"they're the greatest", b'ok'], [b'greatest']]>
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
An N-dimensional `Tensor` or `RaggedTensor` of int64 or int32.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` of sentences that has N - 1 dimension when N > 1.
Otherwise, a string tensor.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_wordpiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 string tokens further into subword tokens.

### Example 1, single word tokenization:
```
>>> vocab = ["they", "##'", "##re", "the", "great", "##est", "[UNK]"]
>>> tokenizer = FastWordpieceTokenizer(vocab, token_out_type=tf.string,
...                                    no_pretokenization=True)
>>> tokens = [["they're", "the", "greatest"]]
>>> tokenizer.tokenize(tokens)
<tf.RaggedTensor [[[b'they', b"##'", b'##re'], [b'the'],
                   [b'great', b'##est']]]>
```

### Example 2, general text tokenization (pre-tokenization on
### punctuation and whitespace followed by WordPiece tokenization):
```
>>> vocab = ["they", "##'", "##re", "the", "great", "##est", "[UNK]",
...          "'", "re"]
>>> tokenizer = FastWordpieceTokenizer(vocab, token_out_type=tf.string)
>>> tokens = [["they're the greatest", "the greatest"]]
>>> tokenizer.tokenize(tokens)
<tf.RaggedTensor [[[b'they', b"'", b're', b'the', b'great', b'##est'],
                   [b'the', b'great', b'##est']]]>
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
A `RaggedTensor` of tokens where `tokens[i, j]` is the j-th token
(i.e., wordpiece) for `input[i]` (i.e., the i-th input word). This token
is either the actual token string content, or the corresponding integer
id, i.e., the index of that token string in the vocabulary.  This choice
is controlled by the `token_out_type` parameter passed to the initializer
method.
</td>
</tr>

</table>

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/fast_wordpiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 string tokens further into subword tokens.

### Example 1, single word tokenization:
```
>>> vocab = ["they", "##'", "##re", "the", "great", "##est", "[UNK]"]
>>> tokenizer = FastWordpieceTokenizer(vocab, token_out_type=tf.string,
...                                    no_pretokenization=True)
>>> tokens = [["they're", "the", "greatest"]]
>>> subtokens, starts, ends = tokenizer.tokenize_with_offsets(tokens)
>>> subtokens
<tf.RaggedTensor [[[b'they', b"##'", b'##re'], [b'the'],
                   [b'great', b'##est']]]>
>>> starts
<tf.RaggedTensor [[[0, 4, 5], [0], [0, 5]]]>
>>> ends
<tf.RaggedTensor [[[4, 5, 7], [3], [5, 8]]]>
```

### Example 2, general text tokenization (pre-tokenization on
### punctuation and whitespace followed by WordPiece tokenization):
```
>>> vocab = ["they", "##'", "##re", "the", "great", "##est", "[UNK]",
...          "'", "re"]
>>> tokenizer = FastWordpieceTokenizer(vocab, token_out_type=tf.string)
>>> tokens = [["they're the greatest", "the greatest"]]
>>> subtokens, starts, ends = tokenizer.tokenize_with_offsets(tokens)
>>> subtokens
<tf.RaggedTensor [[[b'they', b"'", b're', b'the', b'great', b'##est'],
                   [b'the', b'great', b'##est']]]>
>>> starts
<tf.RaggedTensor [[[0, 4, 5, 8, 12, 17], [0, 4, 9]]]>
>>> ends
<tf.RaggedTensor [[[4, 5, 7, 11, 17, 20], [3, 9, 12]]]>
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
A tuple `(tokens, start_offsets, end_offsets)` where:
</td>
</tr>
<tr>
<td>
`tokens`
</td>
<td>
is a `RaggedTensor`, where `tokens[i, j]` is the j-th token
    (i.e., wordpiece) for `input[i]` (i.e., the i-th input word). This
    token is either the actual token string content, or the corresponding
    integer id, i.e., the index of that token string in the vocabulary.
    This choice is controlled by the `token_out_type` parameter passed to
    the initializer method.
start_offsets[i1...iN, j]: is a `RaggedTensor` of the byte offsets
    for the inclusive start of the `jth` token in `input[i1...iN]`.
end_offsets[i1...iN, j]: is a `RaggedTensor` of the byte offsets for
    the exclusive end of the `jth` token in `input[i`...iN]` (exclusive,
    i.e., first byte after the end of the token).
</td>
</tr>
</table>
