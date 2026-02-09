description: Tokenizes UTF-8 by splitting when there is a change in Unicode
script.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.UnicodeScriptTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.UnicodeScriptTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_script_tokenizer.py">View
source</a>

Tokenizes UTF-8 by splitting when there is a change in Unicode script.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.UnicodeScriptTokenizer(
    keep_whitespace=False
)
</code></pre>

<!-- Placeholder for "Used in" -->

By default, this tokenizer leaves out scripts matching the whitespace unicode
property (use the `keep_whitespace` argument to keep it), so in this case the
results are similar to the `WhitespaceTokenizer`. Any punctuation will get its
own token (since it is in a different script), and any script change in the
input string will be the location of a split.

#### Example:

```
>>> tokenizer = tf_text.UnicodeScriptTokenizer()
>>> tokens = tokenizer.tokenize(["xy.,z de", "fg?h", "abαβ"])
>>> print(tokens.to_list())
[[b'xy', b'.,', b'z', b'de'], [b'fg', b'?', b'h'],
 [b'ab', b'\xce\xb1\xce\xb2']]
```

```
>>> tokens = tokenizer.tokenize(u"累計7239人")
>>> print(tokens)
tf.Tensor([b'\xe7\xb4\xaf\xe8\xa8\x88' b'7239' b'\xe4\xba\xba'], shape=(3,),
          dtype=string)
```

Both the punctuation and the whitespace in the first string have been split, but
the punctuation run is present as a token while the whitespace isn't emitted (by
default). The third example shows the case of a script change without any
whitespace. This results in a split at that boundary point.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`keep_whitespace`<a id="keep_whitespace"></a>
</td>
<td>
A boolean that specifices whether to emit whitespace
tokens (default `False`).
</td>
</tr>
</table>

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

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_script_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    input
)
</code></pre>

Tokenizes UTF-8 by splitting when there is a change in Unicode script.

The strings are split when successive tokens change their Unicode script or
change being whitespace or not. The script codes used correspond to
International Components for Unicode (ICU) UScriptCode values. See:
http://icu-project.org/apiref/icu4c/uscript_8h.html

ICU-defined whitespace characters are dropped, unless the `keep_whitespace`
option was specified at construction time.

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
input tensor with an added ragged dimension for tokens of each string.
</td>
</tr>

</table>

<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/unicode_script_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    input
)
</code></pre>

Tokenizes UTF-8 by splitting when there is a change in Unicode script.

The strings are split when a change in the Unicode script is detected between
sequential tokens. The script codes used correspond to International Components
for Unicode (ICU) UScriptCode values. See:
http://icu-project.org/apiref/icu4c/uscript_8h.html

ICU defined whitespace characters are dropped, unless the keep_whitespace option
was specified at construction time.

#### Example:

```
>>> tokenizer = tf_text.UnicodeScriptTokenizer()
>>> tokens = tokenizer.tokenize_with_offsets(["xy.,z de", "abαβ"])
>>> print(tokens[0].to_list())
[[b'xy', b'.,', b'z', b'de'], [b'ab', b'\xce\xb1\xce\xb2']]
>>> print(tokens[1].to_list())
[[0, 2, 4, 6], [0, 2]]
>>> print(tokens[2].to_list())
[[2, 4, 5, 8], [2, 6]]
```

```
>>> tokens = tokenizer.tokenize_with_offsets(u"累計7239人")
>>> print(tokens[0])
tf.Tensor([b'\xe7\xb4\xaf\xe8\xa8\x88' b'7239' b'\xe4\xba\xba'],
    shape=(3,), dtype=string)
>>> print(tokens[1])
tf.Tensor([ 0  6 10], shape=(3,), dtype=int64)
>>> print(tokens[2])
tf.Tensor([ 6 10 13], shape=(3,), dtype=int64)
```

The start_offsets and end_offsets are in byte indices of the original string.
When calling with multiple string inputs, the offset indices will be relative to
the individual source strings.

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

*   `tokens`: A `RaggedTensor` of tokenized text.
*   `start_offsets`: A `RaggedTensor` of the tokens' starting byte offset.
*   `end_offsets`: A `RaggedTensor` of the tokens' ending byte offset. </td>
    </tr>

</table>
