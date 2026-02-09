description: Base class for tokenizer implementations that return offsets.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.TokenizerWithOffsets" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.TokenizerWithOffsets

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

Base class for tokenizer implementations that return offsets.

Inherits From: [`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.TokenizerWithOffsets(
    name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

The offsets indicate which substring from the input string was used to generate
each token. E.g., if `input` is a single string, then each token `token[i]` was
generated from the substring `input[starts[i]:ends[i]]`.

Each TokenizerWithOffsets subclass must implement the `tokenize_with_offsets`
method, which returns a tuple containing both the pieces and the start and end
offsets where those pieces occurred in the input string. I.e., if `tokens,
starts, ends = tokenize_with_offsets(s)`, then each token `token[i]` corresponds
with `tf.strings.substr(s, starts[i], ends[i] - starts[i])`.

If the tokenizer encodes tokens as strings (and not token ids), then it will
usually be the case that these corresponding strings are equal; but that is not
technically required. For example, a tokenizer might choose to downcase strings

#### Example:

```
>>> class CharTokenizer(TokenizerWithOffsets):
...   def tokenize_with_offsets(self, input):
...     chars, starts = tf.strings.unicode_split_with_offsets(input, 'UTF-8')
...     lengths = tf.expand_dims(tf.strings.length(input), -1)
...     ends = tf.concat([starts[..., 1:], tf.cast(lengths, tf.int64)], -1)
...     return chars, starts, ends
...   def tokenize(self, input):
...     return self.tokenize_with_offsets(input)[0]
>>> pieces, starts, ends = CharTokenizer().split_with_offsets("a😊c")
>>> print(pieces.numpy(), starts.numpy(), ends.numpy())
[b'a' b'\xf0\x9f\x98\x8a' b'c'] [0 1 5] [1 5 6]
```

## Methods

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

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>tokenize(
    input
)
</code></pre>

Tokenizes the input tensor.

Splits each string in the input tensor into a sequence of tokens. Tokens
generally correspond to short substrings of the source string. Tokens can be
encoded using either strings or integer ids.

#### Example:

```
>>> print(tf_text.WhitespaceTokenizer().tokenize("small medium large"))
tf.Tensor([b'small' b'medium' b'large'], shape=(3,), dtype=string)
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
An N-dimensional UTF-8 string (or optionally integer) `Tensor` or
`RaggedTensor`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An N+1-dimensional UTF-8 string or integer `Tensor` or `RaggedTensor`.
For each string from the input tensor, the final, extra dimension contains
the tokens that string was split into.
</td>
</tr>

</table>

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>tokenize_with_offsets(
    input
)
</code></pre>

Tokenizes the input tensor and returns the result with byte-offsets.

The offsets indicate which substring from the input string was used to generate
each token. E.g., if `input` is a `tf.string` tensor, then each token `token[i]`
was generated from the substring `tf.substr(input, starts[i],
len=ends[i]-starts[i])`.

Note: Remember that the `tf.string` type is a byte-string. The returned indices
are in units of bytes, not characters like a Python `str`.

#### Example:

```
>>> splitter = tf_text.WhitespaceTokenizer()
>>> pieces, starts, ends = splitter.tokenize_with_offsets("a bb ccc")
>>> print(pieces.numpy(), starts.numpy(), ends.numpy())
[b'a' b'bb' b'ccc'] [0 2 5] [1 4 8]
>>> print(tf.strings.substr("a bb ccc", starts, ends-starts))
tf.Tensor([b'a' b'bb' b'ccc'], shape=(3,), dtype=string)
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
An N-dimensional UTF-8 string (or optionally integer) `Tensor` or
`RaggedTensor`.
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

*   `tokens` is an N+1-dimensional UTF-8 string or integer `Tensor` or
    `RaggedTensor`.
*   `start_offsets` is an N+1-dimensional integer `Tensor` or `RaggedTensor`
    containing the starting indices of each token (byte indices for input
    strings).
*   `end_offsets` is an N+1-dimensional integer `Tensor` or `RaggedTensor`
    containing the exclusive ending indices of each token (byte indices for
    input strings). </td> </tr>

</table>
