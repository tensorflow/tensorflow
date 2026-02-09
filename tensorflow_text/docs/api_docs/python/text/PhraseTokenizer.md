description: Tokenizes a tensor of UTF-8 string tokens into phrases.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.PhraseTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="tokenize"/>
</div>

# text.PhraseTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/phrase_tokenizer.py">View
source</a>

Tokenizes a tensor of UTF-8 string tokens into phrases.

Inherits From: [`Tokenizer`](../text/Tokenizer.md),
[`Splitter`](../text/Splitter.md), [`Detokenizer`](../text/Detokenizer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.PhraseTokenizer(
    vocab=None,
    token_out_type=dtypes.int32,
    unknown_token=&#x27;&lt;UNK&gt;&#x27;,
    support_detokenization=True,
    prob=0,
    model_buffer=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

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
`support_detokenization`<a id="support_detokenization"></a>
</td>
<td>
(optional) Whether to make the tokenizer support
doing detokenization. Setting it to true expands the size of the model
flatbuffer.
</td>
</tr><tr>
<td>
`prob`<a id="prob"></a>
</td>
<td>
Probability of emitting a phrase when there is a match.
</td>
</tr><tr>
<td>
`model_buffer`<a id="model_buffer"></a>
</td>
<td>
(optional) Bytes object (or a uint8 tf.Tenosr) that contains
the phrase model in flatbuffer format (see phrase_tokenizer_model.fbs).
If not `None`, all other arguments (except `token_output_type`) are
ignored.
</td>
</tr>
</table>

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/phrase_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>detokenize(
    input_t
)
</code></pre>

Detokenizes a tensor of int64 or int32 phrase ids into sentences.

Detokenize and tokenize an input string returns itself when the input string is
normalized and the tokenized phrases don't contain `<unk>`.

### Example:
```
>>> vocab = ["I", "have", "a", "dream", "a dream", "I have a", "<UNK>"]
>>> tokenizer = PhraseTokenizer(vocab, support_detokenization=True)
>>> ids = tf.ragged.constant([[0, 1, 2], [5, 3]])
>>> tokenizer.detokenize(ids)
<tf.Tensor: shape=(2,), dtype=string,
...       numpy=array([b'I have a', b'I have a dream'], dtype=object)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_t`
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

<h3 id="tokenize"><code>tokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/phrase_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 string tokens further into phrase tokens.

### Example, single string tokenization:
```
>>> vocab = ["I", "have", "a", "dream", "a dream", "I have a", "<UNK>"]
>>> tokenizer = PhraseTokenizer(vocab, token_out_type=tf.string)
>>> tokens = [["I have a dream"]]
>>> phrases = tokenizer.tokenize(tokens)
>>> phrases
<tf.RaggedTensor [[[b'I have a', b'dream']]]>
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

<tr>
<td>
`tokens`
</td>
<td>
is a `RaggedTensor`, where `tokens[i, j]` is the j-th token
(i.e., phrase) for `input[i]` (i.e., the i-th input word). This
token is either the actual token string content, or the corresponding
integer id, i.e., the index of that token string in the vocabulary.
This choice is controlled by the `token_out_type` parameter passed to
the initializer method.
</td>
</tr>
</table>
