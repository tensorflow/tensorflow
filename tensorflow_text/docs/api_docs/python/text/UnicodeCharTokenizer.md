description: Tokenizes a tensor of UTF-8 strings on Unicode character
boundaries.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.UnicodeCharTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="detokenize"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.UnicodeCharTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

Tokenizes a tensor of UTF-8 strings on Unicode character boundaries.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md), [`Detokenizer`](../text/Detokenizer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.UnicodeCharTokenizer()
</code></pre>

<!-- Placeholder for "Used in" -->

Resulting tokens are integers (unicode codepoints). Scalar input will produce a
`Tensor` output containing the codepoints. Tensor inputs will produce
`RaggedTensor` outputs.

#### Example:

```
>>> tokenizer = tf_text.UnicodeCharTokenizer()
>>> tokens = tokenizer.tokenize("abc")
>>> print(tokens)
tf.Tensor([97 98 99], shape=(3,), dtype=int32)
```

```
>>> tokens = tokenizer.tokenize(["abc", "de"])
>>> print(tokens)
<tf.RaggedTensor [[97, 98, 99], [100, 101]]>
```

Note: any remaining illegal and special UTF-8 characters (like BOM characters)
in the input string will not be treated specially by the tokenizer and show up
in the output tokens. These should be normalized out before or after
tokenization if they are unwanted in the application.

```
>>> t = ["abc" + chr(0xfffe) + chr(0x1fffe) ]
>>> tokens = tokenizer.tokenize(t)
>>> print(tokens.to_list())
[[97, 98, 99, 65534, 131070]]
```

Passing malformed UTF-8 will result in unpredictable behavior. Make sure inputs
conform to UTF-8.

## Methods

<h3 id="detokenize"><code>detokenize</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>detokenize(
    input, name=None
)
</code></pre>

Detokenizes input codepoints (integers) to UTF-8 strings.

#### Example:

```
>>> tokenizer = tf_text.UnicodeCharTokenizer()
>>> tokens = tokenizer.tokenize(["abc", "de"])
>>> s = tokenizer.detokenize(tokens)
>>> print(s)
tf.Tensor([b'abc' b'de'], shape=(2,), dtype=string)
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
A `RaggedTensor` or `Tensor` of codepoints (ints) with a rank of at
least 1.
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
A N-1 dimensional string tensor of the text corresponding to the UTF-8
codepoints in the input.
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

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 strings on Unicode character boundaries.

Input strings are split on character boundaries using
unicode_decode_with_offsets.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input`
</td>
<td>
A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.
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
input tensor with an added ragged dimension for tokens (characters) of
each string.
</td>
</tr>

</table>

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_char_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    input
)
</code></pre>

Tokenizes a tensor of UTF-8 strings to Unicode characters.

#### Example:

```
>>> tokenizer = tf_text.UnicodeCharTokenizer()
>>> tokens = tokenizer.tokenize_with_offsets("a"+chr(8364)+chr(10340))
>>> print(tokens[0])
tf.Tensor([   97  8364 10340], shape=(3,), dtype=int32)
>>> print(tokens[1])
tf.Tensor([0 1 4], shape=(3,), dtype=int64)
>>> print(tokens[2])
tf.Tensor([1 4 7], shape=(3,), dtype=int64)
```

The `start_offsets` and `end_offsets` are in byte indices of the original
string. When calling with multiple string inputs, the offset indices will be
relative to the individual source strings:

```
>>> toks = tokenizer.tokenize_with_offsets(["a"+chr(8364), "b"+chr(10300) ])
>>> print(toks[0])
<tf.RaggedTensor [[97, 8364], [98, 10300]]>
>>> print(toks[1])
<tf.RaggedTensor [[0, 1], [0, 1]]>
>>> print(toks[2])
<tf.RaggedTensor [[1, 4], [1, 4]]>
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
A `RaggedTensor`or `Tensor` of UTF-8 strings with any shape.
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

*   `tokens`: A `RaggedTensor` of code points (integer type).
*   `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
*   `end_offsets`: A `RaggedTensor` of the tokens' ending byte offset. </td>
    </tr>

</table>
