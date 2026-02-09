description: Tokenizes a tensor of UTF-8 strings.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.SentencepieceTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
<meta itemprop="property" content="id_to_string"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="string_to_id"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
<meta itemprop="property" content="vocab_size"/>
</div>

# text.SentencepieceTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

Tokenizes a tensor of UTF-8 strings.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md), [`Detokenizer`](../text/Detokenizer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.SentencepieceTokenizer(
    model=None,
    out_type=dtypes.int32,
    nbest_size=0,
    alpha=1.0,
    reverse=False,
    add_bos=False,
    add_eos=False,
    return_nbest=False,
    name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

SentencePiece is an unsupervised text tokenizer and detokenizer. It is used
mainly for Neural Network-based text generation systems where the vocabulary
size is predetermined prior to the neural model training. SentencePiece
implements subword units with the extension of direct training from raw
sentences.

Before using the tokenizer, you will need to train a vocabulary and build a
model configuration for it. Please visit the
[Sentencepiece repository](https://github.com/google/sentencepiece#train-sentencepiece-model)
for the most up-to-date instructions on this process.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model`<a id="model"></a>
</td>
<td>
The sentencepiece model serialized proto.
</td>
</tr><tr>
<td>
`out_type`<a id="out_type"></a>
</td>
<td>
output type. tf.int32 or tf.string (Default = tf.int32) Setting
tf.int32 directly encodes the string into an id sequence.
</td>
</tr><tr>
<td>
`nbest_size`<a id="nbest_size"></a>
</td>
<td>
A scalar for sampling.
* `nbest_size = {0,1}`: No sampling is performed. (default)
* `nbest_size > 1`: samples from the nbest_size results.
* `nbest_size < 0`: assuming that nbest_size is infinite and samples
    from the all hypothesis (lattice) using
    forward-filtering-and-backward-sampling algorithm.
</td>
</tr><tr>
<td>
`alpha`<a id="alpha"></a>
</td>
<td>
A scalar for a smoothing parameter. Inverse temperature for
probability rescaling.
</td>
</tr><tr>
<td>
`reverse`<a id="reverse"></a>
</td>
<td>
Reverses the tokenized sequence (Default = false)
</td>
</tr><tr>
<td>
`add_bos`<a id="add_bos"></a>
</td>
<td>
Add beginning of sentence token to the result (Default = false)
</td>
</tr><tr>
<td>
`add_eos`<a id="add_eos"></a>
</td>
<td>
Add end of sentence token to the result (Default = false). When
`reverse=True` beginning/end of sentence tokens are added after
reversing.
</td>
</tr><tr>
<td>
`return_nbest`<a id="return_nbest"></a>
</td>
<td>
If True requires that `nbest_size` is a scalar and `> 1`.
Returns the `nbest_size` best tokenizations for each sentence instead
of a single one. The returned tensor has shape
`[batch * nbest, (tokens)]`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
The name argument that is passed to the op function.
</td>
</tr>
</table>

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>detokenize(
    input, name=None
)
</code></pre>

Detokenizes tokens into preprocessed text.

This function accepts tokenized text, and reforms it back into sentences.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
A `RaggedTensor` or `Tensor` of UTF-8 string tokens with a rank of
at least 1.
</td>
</tr><tr>
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
A N-1 dimensional string Tensor or RaggedTensor of the detokenized text.
</td>
</tr>

</table>

<h3 id="id_to_string"><code>id_to_string</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>id_to_string(
    input, name=None
)
</code></pre>

Converts vocabulary id into a token.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
An arbitrary tensor of int32 representing the token IDs.
</td>
</tr><tr>
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
A tensor of string with the same shape as input.
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

<h3 id="string_to_id"><code>string_to_id</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>string_to_id(
    input, name=None
)
</code></pre>

Converts token into a vocabulary id.

This function is particularly helpful for determining the IDs for any special
tokens whose ID could not be determined through normal tokenization.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
An arbitrary tensor of string tokens.
</td>
</tr><tr>
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
A tensor of int32 representing the IDs with the same shape as input.
</td>
</tr>

</table>

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    input, name=None
)
</code></pre>

Tokenizes a tensor of UTF-8 strings.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
A `RaggedTensor` or `Tensor` of UTF-8 strings with any shape.
</td>
</tr><tr>
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
A `RaggedTensor` of tokenized text. The returned shape is the shape of the
input tensor with an added ragged dimension for tokens of each string.
</td>
</tr>

</table>

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    input, name=None
)
</code></pre>

Tokenizes a tensor of UTF-8 strings.

This function returns a tuple containing the tokens along with start and end
byte offsets that mark where in the original string each token was located.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
A `RaggedTensor` or `Tensor` of UTF-8 strings with any shape.
</td>
</tr><tr>
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
A tuple `(tokens, start_offsets, end_offsets)` where:
</td>
</tr>
<tr>
<td>
`tokens`
</td>
<td>
is an N+1-dimensional UTF-8 string or integer `Tensor` or
`RaggedTensor`.
</td>
</tr><tr>
<td>
`start_offsets`
</td>
<td>
is an N+1-dimensional integer `Tensor` or
`RaggedTensor` containing the starting indices of each token (byte
indices for input strings).
</td>
</tr><tr>
<td>
`end_offsets`
</td>
<td>
is an N+1-dimensional integer `Tensor` or
`RaggedTensor` containing the exclusive ending indices of each token
(byte indices for input strings).
</td>
</tr>
</table>

<h3 id="vocab_size"><code>vocab_size</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/sentencepiece_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>vocab_size(
    name=None
)
</code></pre>

Returns the vocabulary size.

The number of tokens from within the Sentencepiece vocabulary provided at the
time of initialization.

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
