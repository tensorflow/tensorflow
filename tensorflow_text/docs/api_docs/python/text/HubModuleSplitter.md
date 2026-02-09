description: Splitter that uses a Hub module.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.HubModuleSplitter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="split"/>
<meta itemprop="property" content="split_with_offsets"/>
</div>

# text.HubModuleSplitter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/hub_module_splitter.py">View
source</a>

Splitter that uses a Hub module.

Inherits From: [`SplitterWithOffsets`](../text/SplitterWithOffsets.md),
[`Splitter`](../text/Splitter.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.HubModuleSplitter(
    hub_module_handle
)
</code></pre>



<!-- Placeholder for "Used in" -->

The TensorFlow graph from the module performs the real work.  The Python code
from this class handles the details of interfacing with that module, as well
as the support for ragged tensors and high-rank tensors.

The Hub module should be supported by `hub.load()
<https://www.tensorflow.org/hub/api_docs/python/hub/load>`_ If a v1 module, it
should have a graph variant with an empty set of tags; we consider that graph
variant to be the module and ignore everything else. The module should have a
signature named `default` that takes a
<a href="../text.md"><code>text</code></a> input (a rank-1 tensor of strings to
split into pieces) and returns a dictionary of tensors, let's say `output_dict`,
such that:

* `output_dict['num_pieces']` is a rank-1 tensor of integers, where
num_pieces[i] is the number of pieces that text[i] was split into.

* `output_dict['pieces']` is a rank-1 tensor of strings containing all pieces
for text[0] (in order), followed by all pieces for text[1] (in order) and so
on.

* `output_dict['starts']` is a rank-1 tensor of integers with the byte offsets
where the pieces start (relative to the beginning of the corresponding input
string).

* `output_dict['end']` is a rank-1 tensor of integers with the byte offsets
right after the end of the tokens (relative to the beginning of the
corresponding input string).

The output dictionary may contain other tensors (e.g., for debugging) but this
class is not using them.

#### Example:

```
>>> HUB_MODULE = "https://tfhub.dev/google/zh_segmentation/1"
>>> segmenter = HubModuleSplitter(hub.resolve(HUB_MODULE))
>>> segmenter.split(["新华社北京"])
<tf.RaggedTensor [[b'\xe6\x96\xb0\xe5\x8d\x8e\xe7\xa4\xbe',
                   b'\xe5\x8c\x97\xe4\xba\xac']]>
```

You can also use this tokenizer to return the split strings and their offsets:

```
>>> HUB_MODULE = "https://tfhub.dev/google/zh_segmentation/1"
>>> segmenter = HubModuleSplitter(hub.resolve(HUB_MODULE))
>>> pieces, starts, ends = segmenter.split_with_offsets(["新华社北京"])
>>> print("pieces: %s starts: %s ends: %s" % (pieces, starts, ends))
pieces: <tf.RaggedTensor [[b'\xe6\x96\xb0\xe5\x8d\x8e\xe7\xa4\xbe',
                           b'\xe5\x8c\x97\xe4\xba\xac']]>
starts: <tf.RaggedTensor [[0, 9]]>
ends: <tf.RaggedTensor [[9, 15]]>
```

Currently, this class also supports an older API, which uses slightly
different key names for the output dictionary.  For new Hub modules, please
use the API described above.

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
(2) a handle to a module uploaded to e.g., https://tfhub.dev.  The
module should implement the signature described in the docstring for
this class.
</td>
</tr>
</table>

## Methods

<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/hub_module_splitter.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split(
    input_strs
)
</code></pre>

Splits a tensor of UTF-8 strings into pieces.


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
input tensor with an added ragged dimension for the pieces of each string.
</td>
</tr>

</table>



<h3 id="split_with_offsets"><code>split_with_offsets</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/hub_module_splitter.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split_with_offsets(
    input_strs
)
</code></pre>

Splits a tensor of UTF-8 strings into pieces with [start,end) offsets.


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
A tuple `(pieces, start_offsets, end_offsets)` where:
* `pieces` is a `RaggedTensor` of strings where `pieces[i1...iN, j]` is
  the string content of the `j-th` piece in `input_strs[i1...iN]`
* `start_offsets` is a `RaggedTensor` of int64s where
  `start_offsets[i1...iN, j]` is the byte offset for the start of the
  `j-th` piece in `input_strs[i1...iN]`.
* `end_offsets` is a `RaggedTensor` of int64s where
  `end_offsets[i1...iN, j]` is the byte offset immediately after the
  end of the `j-th` piece in `input_strs[i...iN]`.
</td>
</tr>

</table>





