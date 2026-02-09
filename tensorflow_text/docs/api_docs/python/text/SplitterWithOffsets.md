description: An abstract base class for splitters that return offsets.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.SplitterWithOffsets" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
</div>

# text.SplitterWithOffsets

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/splitter.py">View
source</a>

An abstract base class for splitters that return offsets.

Inherits From: [`Splitter`](../text/Splitter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.SplitterWithOffsets(
    name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Each SplitterWithOffsets subclass must implement the `split_with_offsets`
method, which returns a tuple containing both the pieces and the offsets where
those pieces occurred in the input string. E.g.:

```
>>> class CharSplitter(SplitterWithOffsets):
...   def split_with_offsets(self, input):
...     chars, starts = tf.strings.unicode_split_with_offsets(input, 'UTF-8')
...     lengths = tf.expand_dims(tf.strings.length(input), -1)
...     ends = tf.concat([starts[..., 1:], tf.cast(lengths, tf.int64)], -1)
...     return chars, starts, ends
...   def split(self, input):
...     return self.split_with_offsets(input)[0]
>>> pieces, starts, ends = CharSplitter().split_with_offsets("a😊c")
>>> print(pieces.numpy(), starts.numpy(), ends.numpy())
[b'a' b'\xf0\x9f\x98\x8a' b'c'] [0 1 5] [1 5 6]
```

## Methods

<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/splitter.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>split(
    input
)
</code></pre>

Splits the input tensor into pieces.

Generally, the pieces returned by a splitter correspond to substrings of the
original string, and can be encoded using either strings or integer ids.

#### Example:

```
>>> print(tf_text.WhitespaceTokenizer().split("small medium large"))
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
the pieces that string was split into.
</td>
</tr>

</table>

<h3 id="split_with_offsets"><code>split_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/splitter.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>split_with_offsets(
    input
)
</code></pre>

Splits the input tensor, and returns the resulting pieces with offsets.

#### Example:

```
>>> splitter = tf_text.WhitespaceTokenizer()
>>> pieces, starts, ends = splitter.split_with_offsets("a bb ccc")
>>> print(pieces.numpy(), starts.numpy(), ends.numpy())
[b'a' b'bb' b'ccc'] [0 2 5] [1 4 8]
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
A tuple `(pieces, start_offsets, end_offsets)` where:

*   `pieces` is an N+1-dimensional UTF-8 string or integer `Tensor` or
    `RaggedTensor`.
*   `start_offsets` is an N+1-dimensional integer `Tensor` or `RaggedTensor`
    containing the starting indices of each piece (byte indices for input
    strings).
*   `end_offsets` is an N+1-dimensional integer `Tensor` or `RaggedTensor`
    containing the exclusive ending indices of each piece (byte indices for
    input strings). </td> </tr>

</table>
