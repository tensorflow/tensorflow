description: Tokenizes a tensor of UTF-8 string tokens into subword pieces.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.WordpieceTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
<meta itemprop="property" content="vocab_size"/>
</div>

# text.WordpieceTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordpiece_tokenizer.py">View
source</a>

Tokenizes a tensor of UTF-8 string tokens into subword pieces.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md), [`Detokenizer`](../text/Detokenizer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.WordpieceTokenizer(
    vocab_lookup_table,
    suffix_indicator=&#x27;##&#x27;,
    max_bytes_per_word=100,
    max_chars_per_token=None,
    token_out_type=dtypes.int64,
    unknown_token=&#x27;[UNK]&#x27;,
    split_unknown_characters=False
)
</code></pre>

<!-- Placeholder for "Used in" -->

Each UTF-8 string token in the input is split into its corresponding wordpieces,
drawing from the list in the file `vocab_lookup_table`.

Algorithm summary: For each token, the longest token prefix that is in the
vocabulary is split off. Any part of the token that remains is prefixed using
the `suffix_indicator`, and the process of removing the longest token prefix
continues. The `unknown_token` (UNK) is used when what remains of the token is
not in the vocabulary, or if the token is too long.

When `token_out_type` is tf.string, the output tensor contains strings in the
vocabulary (or UNK). When it is an integer type, the output tensor contains
indices into the vocabulary list (with UNK being after the last entry).

#### Example:

```
>>> import pathlib
>>> pathlib.Path('/tmp/tok_vocab.txt').write_text(
...   "they ##' ##re the great ##est".replace(' ', '\n'))
>>> tokenizer = WordpieceTokenizer('/tmp/tok_vocab.txt',
...   token_out_type=tf.string)
```

```
>>> tokenizer.tokenize(["they're", "the", "greatest"])
<tf.RaggedTensor [[b'they', b"##'", b'##re'], [b'the'], [b'great', b'##est']]>
```

```
>>> tokenizer.tokenize(["they", "are", "great"])
<tf.RaggedTensor [[b'they'], [b'[UNK]'], [b'great']]>
```

```
>>> int_tokenizer = WordpieceTokenizer('/tmp/tok_vocab.txt',
...   token_out_type=tf.int32)
```

```
>>> int_tokenizer.tokenize(["the", "greatest"])
<tf.RaggedTensor [[3], [4, 5]]>
```

```
>>> int_tokenizer.tokenize(["really", "the", "greatest"])
<tf.RaggedTensor [[6], [3], [4, 5]]>
```

Tensor or ragged tensor inputs result in ragged tensor outputs. Scalar inputs
(which are just a single token) result in tensor outputs.

```
>>> tokenizer.tokenize("they're")
<tf.Tensor: shape=(3,), dtype=string, numpy=array([b'they', b"##'", b'##re'],
dtype=object)>
>>> tokenizer.tokenize(["they're"])
<tf.RaggedTensor [[b'they', b"##'", b'##re']]>
>>> tokenizer.tokenize(tf.ragged.constant([["they're"]]))
<tf.RaggedTensor [[[b'they', b"##'", b'##re']]]>
```

Empty strings are tokenized into empty (ragged) tensors.

```
>>> tokenizer.tokenize([""])
<tf.RaggedTensor [[]]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`vocab_lookup_table`<a id="vocab_lookup_table"></a>
</td>
<td>
A lookup table implementing the LookupInterface
containing the vocabulary of subwords or a string which is the file path
to the vocab.txt file.
</td>
</tr><tr>
<td>
`suffix_indicator`<a id="suffix_indicator"></a>
</td>
<td>
(optional) The characters prepended to a wordpiece to
indicate that it is a suffix to another subword. Default is '##'.
</td>
</tr><tr>
<td>
`max_bytes_per_word`<a id="max_bytes_per_word"></a>
</td>
<td>
(optional) Max size of input token. Default is 100.
</td>
</tr><tr>
<td>
`max_chars_per_token`<a id="max_chars_per_token"></a>
</td>
<td>
(optional) Max size of subwords, excluding suffix
indicator. If known, providing this improves the efficiency of decoding
long words.
</td>
</tr><tr>
<td>
`token_out_type`<a id="token_out_type"></a>
</td>
<td>
(optional) The type of the token to return. This can be
`tf.int64` or `tf.int32` IDs, or `tf.string` subwords. The default is
`tf.int64`.
</td>
</tr><tr>
<td>
`unknown_token`<a id="unknown_token"></a>
</td>
<td>
(optional) The string value to substitute for an unknown
token. Default is "[UNK]". If set to `None`, no substitution occurs.
If `token_out_type` is `tf.int32`/`tf.int64`, the `vocab_lookup_table`
is used (after substitution) to convert the unknown token to an integer.
</td>
</tr><tr>
<td>
`split_unknown_characters`<a id="split_unknown_characters"></a>
</td>
<td>
(optional) Whether to split out single unknown
characters as subtokens. If False (default), words containing unknown
characters will be treated as single unknown tokens.
</td>
</tr>
</table>

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordpiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>detokenize(
    token_ids
)
</code></pre>

Convert a `Tensor` or `RaggedTensor` of wordpiece IDs to string-words.

```
>>> import pathlib
>>> pathlib.Path('/tmp/detok_vocab.txt').write_text(
...     'a b c ##a ##b ##c'.replace(' ', '\n'))
>>> wordpiece = WordpieceTokenizer('/tmp/detok_vocab.txt')
>>> token_ids = [[0, 4, 5, 2, 5, 5, 5]]
>>> wordpiece.detokenize(token_ids)
<tf.RaggedTensor [[b'abc', b'cccc']]>
```

The word pieces are joined along the innermost axis to make words. So the result
has the same rank as the input, but the innermost axis of the result indexes
words instead of word pieces.

The shape transformation is: `[..., wordpieces] => [..., words]`

When the input shape is `[..., words, wordpieces]` (like the output of
<a href="../text/WordpieceTokenizer.md#tokenize"><code>WordpieceTokenizer.tokenize</code></a>)
the result's shape is `[..., words, 1]`. The additional ragged axis can be
removed using `words.merge_dims(-2, -1)`.

Note: This method assumes wordpiece IDs are dense on the interval `[0,
vocab_size)`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`token_ids`
</td>
<td>
A `RaggedTensor` or `Tensor` with an int dtype. Must have
`ndims >= 2`
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `RaggedTensor` with dtype `string` and the rank as the input
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

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordpiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 string tokens further into subword tokens.

### Example:

```
>>> import pathlib
>>> pathlib.Path('/tmp/tok_vocab.txt').write_text(
...     "they ##' ##re the great ##est".replace(' ', '\n'))
>>> tokens = [["they're", 'the', 'greatest']]
>>> tokenizer = WordpieceTokenizer('/tmp/tok_vocab.txt',
...                                token_out_type=tf.string)
>>> tokenizer.tokenize(tokens)
<tf.RaggedTensor [[[b'they', b"##'", b'##re'], [b'the'],
                   [b'great', b'##est']]]>
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
A `RaggedTensor` of tokens where `tokens[i1...iN, j]` is the string
contents (or ID in the vocab_lookup_table representing that string)
of the `jth` token in `input[i1...iN]`
</td>
</tr>

</table>



<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordpiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 string tokens further into subword tokens.

### Example:

```
>>> import pathlib
>>> pathlib.Path('/tmp/tok_vocab.txt').write_text(
...     "they ##' ##re the great ##est".replace(' ', '\n'))
>>> tokens = [["they're", 'the', 'greatest']]
>>> tokenizer = WordpieceTokenizer('/tmp/tok_vocab.txt',
...                                token_out_type=tf.string)
>>> subtokens, starts, ends = tokenizer.tokenize_with_offsets(tokens)
>>> subtokens
<tf.RaggedTensor [[[b'they', b"##'", b'##re'], [b'the'],
                   [b'great', b'##est']]]>
>>> starts
<tf.RaggedTensor [[[0, 4, 5], [0], [0, 5]]]>
>>> ends
<tf.RaggedTensor [[[4, 5, 7], [3], [5, 8]]]>
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

tokens[i1...iN, j]: is a `RaggedTensor` of the string contents (or ID in the
vocab_lookup_table representing that string) of the `jth` token in
`input[i1...iN]`. start_offsets[i1...iN, j]: is a `RaggedTensor` of the byte
offsets for the inclusive start of the `jth` token in `input[i1...iN]`.
end_offsets[i1...iN, j]: is a `RaggedTensor` of the byte offsets for the
exclusive end of the `jth` token in `input[i`...iN]` (exclusive, i.e., first
byte after the end of the token). </td> </tr>

</table>

<h3 id="vocab_size"><code>vocab_size</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordpiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>vocab_size(
    name=None
)
</code></pre>

Returns the vocabulary size.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`name`
</td>
<td>
The name argument that is passed to the op function.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A scalar representing the vocabulary size.
</td>
</tr>

</table>
