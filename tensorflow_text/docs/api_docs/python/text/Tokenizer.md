description: Base class for tokenizer implementations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.Tokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="tokenize"/>
</div>

# text.Tokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

Base class for tokenizer implementations.

Inherits From: [`Splitter`](../text/Splitter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.Tokenizer(
    name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

A Tokenizer is a <a href="../text/Splitter.md"><code>text.Splitter</code></a>
that splits strings into *tokens*. Tokens generally correspond to short
substrings of the source string. Tokens can be encoded using either strings or
integer ids (where integer ids could be created by hashing strings or by looking
them up in a fixed vocabulary table that maps strings to ids).

Each Tokenizer subclass must implement a `tokenize` method, which splits each
string in a Tensor into tokens. E.g.:

```
>>> class SimpleTokenizer(tf_text.Tokenizer):
...   def tokenize(self, input):
...     return tf.strings.split(input)
>>> print(SimpleTokenizer().tokenize(["hello world", "this is a test"]))
<tf.RaggedTensor [[b'hello', b'world'], [b'this', b'is', b'a', b'test']]>
```

By default, the `split` method simply delegates to `tokenize`.

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
