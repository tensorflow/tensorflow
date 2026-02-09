description: Tokenizer used for BERT.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.BertTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.BertTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/bert_tokenizer.py">View
source</a>

Tokenizer used for BERT.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md), [`Detokenizer`](../text/Detokenizer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.BertTokenizer(
    vocab_lookup_table,
    suffix_indicator=&#x27;##&#x27;,
    max_bytes_per_word=100,
    max_chars_per_token=None,
    token_out_type=dtypes.int64,
    unknown_token=&#x27;[UNK]&#x27;,
    split_unknown_characters=False,
    lower_case=False,
    keep_whitespace=False,
    normalization_form=None,
    preserve_unused_token=False,
    basic_tokenizer_class=BasicTokenizer
)
</code></pre>

<!-- Placeholder for "Used in" -->

This tokenizer applies an end-to-end, text string to wordpiece tokenization. It
first applies basic tokenization, followed by wordpiece tokenization.

See `WordpieceTokenizer` for details on the subword tokenization.

For an example of use, see
https://www.tensorflow.org/text/guide/bert_preprocessing_guide

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

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
`tf.int64` IDs, or `tf.string` subwords. The default is `tf.int64`.
</td>
</tr><tr>
<td>
`unknown_token`<a id="unknown_token"></a>
</td>
<td>
(optional) The value to use when an unknown token is found.
Default is "[UNK]". If this is set to a string, and `token_out_type` is
`tf.int64`, the `vocab_lookup_table` is used to convert the
`unknown_token` to an integer. If this is set to `None`, out-of-vocabulary
tokens are left as is.
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
</tr><tr>
<td>
`lower_case`<a id="lower_case"></a>
</td>
<td>
bool - If true, a preprocessing step is added to lowercase the
text, apply NFD normalization, and strip accents characters.
</td>
</tr><tr>
<td>
`keep_whitespace`<a id="keep_whitespace"></a>
</td>
<td>
bool - If true, preserves whitespace characters instead of
stripping them away.
</td>
</tr><tr>
<td>
`normalization_form`<a id="normalization_form"></a>
</td>
<td>
If set to a valid value and lower_case=False, the input
text will be normalized to `normalization_form`. See normalize_utf8() op
for a list of valid values.
</td>
</tr><tr>
<td>
`preserve_unused_token`<a id="preserve_unused_token"></a>
</td>
<td>
If true, text in the regex format `\\[unused\\d+\\]`
will be treated as a token and thus remain preserved as is to be looked up
in the vocabulary.
</td>
</tr><tr>
<td>
`basic_tokenizer_class`<a id="basic_tokenizer_class"></a>
</td>
<td>
If set, the class to use instead of BasicTokenizer
</td>
</tr>
</table>

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/bert_tokenizer.py">View
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
<a href="../text/BertTokenizer.md#tokenize"><code>BertTokenizer.tokenize</code></a>/<a href="../text/BertTokenizer.md#detokenize"><code>BertTokenizer.detokenize</code></a>
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
>>> import pathlib
>>> pathlib.Path('/tmp/tok_vocab.txt').write_text(
...    "they ##' ##re the great ##est".replace(' ', '\n'))
>>> tokenizer = BertTokenizer(
...    vocab_lookup_table='/tmp/tok_vocab.txt')
>>> text_inputs = tf.constant(['greatest'.encode('utf-8')])
>>> tokenizer.detokenize([[4, 5]])
<tf.RaggedTensor [[b'greatest']]>
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

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/bert_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    text_input
)
</code></pre>

Tokenizes a tensor of string tokens into subword tokens for BERT.

#### Example:

```
>>> import pathlib
>>> pathlib.Path('/tmp/tok_vocab.txt').write_text(
...     "they ##' ##re the great ##est".replace(' ', '\n'))
>>> tokenizer = BertTokenizer(
...     vocab_lookup_table='/tmp/tok_vocab.txt')
>>> text_inputs = tf.constant(['greatest'.encode('utf-8') ])
>>> tokenizer.tokenize(text_inputs)
<tf.RaggedTensor [[[4, 5]]]>
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

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/bert_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    text_input
)
</code></pre>

Tokenizes a tensor of string tokens into subword tokens for BERT.

#### Example:

```
>>> import pathlib
>>> pathlib.Path('/tmp/tok_vocab.txt').write_text(
...     "they ##' ##re the great ##est".replace(' ', '\n'))
>>> tokenizer = BertTokenizer(
...     vocab_lookup_table='/tmp/tok_vocab.txt')
>>> text_inputs = tf.constant(['greatest'.encode('utf-8')])
>>> tokenizer.tokenize_with_offsets(text_inputs)
(<tf.RaggedTensor [[[4, 5]]]>,
 <tf.RaggedTensor [[[0, 5]]]>,
 <tf.RaggedTensor [[[5, 8]]]>)
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
