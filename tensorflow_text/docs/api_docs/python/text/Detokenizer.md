description: Base class for detokenizer implementations.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.Detokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
</div>

# text.Detokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

Base class for detokenizer implementations.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.Detokenizer(
    name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

A Detokenizer is a module that combines tokens to form strings. Generally,
subclasses of `Detokenizer` will also be subclasses of `Tokenizer`; and the
`detokenize` method will be the inverse of the `tokenize` method. I.e.,
`tokenizer.detokenize(tokenizer.tokenize(s)) == s`.

Each Detokenizer subclass must implement a `detokenize` method, which combines
tokens together to form strings. E.g.:

```
>>> class SimpleDetokenizer(tf_text.Detokenizer):
...   def detokenize(self, input):
...     return tf.strings.reduce_join(input, axis=-1, separator=" ")
>>> text = tf.ragged.constant([["hello", "world"], ["a", "b", "c"]])
>>> print(SimpleDetokenizer().detokenize(text))
tf.Tensor([b'hello world' b'a b c'], shape=(2,), dtype=string)
```

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/tokenization.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>detokenize(
    input
)
</code></pre>

Assembles the tokens in the input tensor into a string.

Generally, `detokenize` is the inverse of the `tokenize` method, and can be used
to reconstrct a string from a set of tokens. This is especially helpful in cases
where the tokens are integer ids, such as indexes into a vocabulary table -- in
that case, the tokenized encoding is not very human-readable (since it's just a
list of integers), so the `detokenize` method can be used to turn it back into
something that's more readable.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
An N-dimensional UTF-8 string or integer `Tensor` or
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
An (N-1)-dimensional UTF-8 string `Tensor` or `RaggedTensor`.
</td>
</tr>

</table>
