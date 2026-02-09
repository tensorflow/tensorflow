description: RegexSplitter splits text on the given regular expression.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.RegexSplitter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
</div>

# text.RegexSplitter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/regex_split_ops.py">View
source</a>

`RegexSplitter` splits text on the given regular expression.

Inherits From: [`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.RegexSplitter(
    split_regex=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The default is a newline character pattern. It can also return the beginning and
ending byte offsets as well.

By default, this splitter will break on newlines, ignoring any trailing ones.
```

> > > splitter = RegexSplitter() text_input=[ ... b"Hi there.\nWhat time is
> > > it?\nIt is gametime.", ... b"Who let the dogs out?\nWho?\nWho?\nWho?\n\n",
> > > ... ] splitter.split(text_input)
> > > <tf.RaggedTensor [[b'Hi there.', b'What time is it?', b'It is gametime.'], [b'Who let the dogs out?', b'Who?', b'Who?', b'Who?']]>
> > > ```

The splitter can be passed a custom split pattern, as well. The pattern can be
any string, but we're using a single character (tab) in this example. ```

> > > splitter = RegexSplitter(split_regex='\t') text_input=[ ... b"Hi
> > > there.\tWhat time is it?\tIt is gametime.", ... b"Who let the dogs
> > > out?\tWho?\tWho?\tWho?\t\t", ... ] splitter.split(text_input)
> > > <tf.RaggedTensor [[b'Hi there.', b'What time is it?', b'It is gametime.'], [b'Who let the dogs out?', b'Who?', b'Who?', b'Who?']]>
> > > ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`split_regex`<a id="split_regex"></a>
</td>
<td>
(optional) A string containing the regex pattern of a
delimiter to split on. Default is '\r?\n'.
</td>
</tr>
</table>

## Methods

<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/regex_split_ops.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
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

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/regex_split_ops.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
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





