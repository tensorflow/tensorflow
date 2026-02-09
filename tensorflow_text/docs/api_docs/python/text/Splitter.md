description: An abstract base class for splitting text.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.Splitter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
</div>

# text.Splitter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/splitter.py">View
source</a>

An abstract base class for splitting text.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.Splitter(
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

A Splitter is a module that splits strings into pieces. Generally, the pieces
returned by a splitter correspond to substrings of the original string, and can
be encoded using either strings or integer ids (where integer ids could be
created by hashing strings or by looking them up in a fixed vocabulary table
that maps strings to ids).

Each Splitter subclass must implement a `split` method, which subdivides each
string in an input Tensor into pieces. E.g.:

```
>>> class SimpleSplitter(tf_text.Splitter):
...   def split(self, input):
...     return tf.strings.split(input)
>>> print(SimpleSplitter().split(["hello world", "this is a test"]))
<tf.RaggedTensor [[b'hello', b'world'], [b'this', b'is', b'a', b'test']]>
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





