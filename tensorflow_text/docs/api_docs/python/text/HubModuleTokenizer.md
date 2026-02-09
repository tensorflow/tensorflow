description: Tokenizer that uses a Hub module.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.HubModuleTokenizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
<meta itemprop="property" content="tokenize"/>
<meta itemprop="property" content="tokenize_with_offsets"/>
</div>

# text.HubModuleTokenizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/hub_module_tokenizer.py">View
source</a>

Tokenizer that uses a Hub module.

Inherits From: [`TokenizerWithOffsets`](../text/TokenizerWithOffsets.md),
[`Tokenizer`](../text/Tokenizer.md),
[`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.HubModuleTokenizer(
    hub_module_handle
)
</code></pre>



<!-- Placeholder for "Used in" -->

This class is just a wrapper around an internal HubModuleSplitter.  It offers
the same functionality, but with 'token'-based method names: e.g., one can use
tokenize() instead of the more general and less informatively named split().

#### Example:

```
>>> HUB_MODULE = "https://tfhub.dev/google/zh_segmentation/1"
>>> segmenter = HubModuleTokenizer(hub.resolve(HUB_MODULE))
>>> segmenter.tokenize(["新华社北京"])
<tf.RaggedTensor [[b'\xe6\x96\xb0\xe5\x8d\x8e\xe7\xa4\xbe',
                   b'\xe5\x8c\x97\xe4\xba\xac']]>
```

You can also use this tokenizer to return the split strings and their offsets:

```
>>> HUB_MODULE = "https://tfhub.dev/google/zh_segmentation/1"
>>> segmenter = HubModuleTokenizer(hub.resolve(HUB_MODULE))
>>> pieces, starts, ends = segmenter.tokenize_with_offsets(["新华社北京"])
>>> print("pieces: %s starts: %s ends: %s" % (pieces, starts, ends))
pieces: <tf.RaggedTensor [[b'\xe6\x96\xb0\xe5\x8d\x8e\xe7\xa4\xbe',
                           b'\xe5\x8c\x97\xe4\xba\xac']]>
starts: <tf.RaggedTensor [[0, 9]]>
ends: <tf.RaggedTensor [[9, 15]]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`hub_module_handle`<a id="hub_module_handle"></a>
</td>
<td>
A string handle accepted by hub.load().  Supported
cases include (1) a local path to a directory containing a module, and
(2) a handle to a module uploaded to e.g., https://tfhub.dev
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

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/hub_module_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize(
    input_strs
)
</code></pre>

Tokenizes a tensor of UTF-8 strings into words.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_strs`
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
<tr class="alt">
<td colspan="2">
A `RaggedTensor` of segmented text. The returned shape is the shape of the
input tensor with an added ragged dimension for tokens of each string.
</td>
</tr>

</table>



<h3 id="tokenize_with_offsets"><code>tokenize_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/hub_module_tokenizer.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tokenize_with_offsets(
    input_strs
)
</code></pre>

Tokenizes a tensor of UTF-8 strings into words with [start,end) offsets.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_strs`
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
<tr class="alt">
<td colspan="2">
A tuple `(tokens, start_offsets, end_offsets)` where:
* `tokens` is a `RaggedTensor` of strings where `tokens[i1...iN, j]` is
  the string content of the `j-th` token in `input_strs[i1...iN]`
* `start_offsets` is a `RaggedTensor` of int64s where
  `start_offsets[i1...iN, j]` is the byte offset for the start of the
  `j-th` token in `input_strs[i1...iN]`.
* `end_offsets` is a `RaggedTensor` of int64s where
  `end_offsets[i1...iN, j]` is the byte offset immediately after the
  end of the `j-th` token in `input_strs[i...iN]`.
</td>
</tr>

</table>





