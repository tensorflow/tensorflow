description: A Trimmer that truncates the longest segment.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.ShrinkLongestTrimmer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="generate_mask"/>
<meta itemprop="property" content="trim"/>
</div>

# text.ShrinkLongestTrimmer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/trimmer_ops.py">View
source</a>

A `Trimmer` that truncates the longest segment.

Inherits From: [`Trimmer`](../text/Trimmer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.ShrinkLongestTrimmer(
    max_seq_length, axis=-1
)
</code></pre>

<!-- Placeholder for "Used in" -->

A `Trimmer` that allocates a length budget to segments by shrinking whatever is
the longest segment at each round at the end, until the total length of segments
is no larger than the allocated budget. See `generate_mask()` for more details.

## Methods

<h3 id="generate_mask"><code>generate_mask</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/trimmer_ops.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>generate_mask(
    segments
)
</code></pre>

Calculates a truncation mask given a per-batch budget.

Calculate a truncation mask given a budget of the max number of items for each
batch row. The allocation of the budget is done using a 'shrink the largest
segment' algorithm. This algorithm identifies the currently longest segment (in
cases of tie, picking whichever segment occurs first) and reduces its length by
1 by dropping its last element, repeating until the total length of segments is
no larger than `_max_seq_length`.

For example if the budget is [7] and we have segments of size [3, 4, 4], the
truncate budget will be allocated as [2, 2, 3], going through truncation steps #
Truncate the second segment. [3, 3, 4] # Truncate the last segment. [3, 3, 3] #
Truncate the first segment. [2, 3, 3] # Truncate the second segment. [2, 2, 3]

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`segments`
</td>
<td>
A list of `RaggedTensor` each w/ a shape of [num_batch,
(num_items)].
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a list with len(segments) of `RaggedTensor`s, see superclass for details.
</td>
</tr>

</table>

<h3 id="trim"><code>trim</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/trimmer_ops.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>trim(
    segments
)
</code></pre>

Truncate the list of `segments`.

Truncate the list of `segments` using the truncation strategy defined by
`generate_mask`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`segments`
</td>
<td>
A list of `RaggedTensor`s w/ shape [num_batch, (num_items)].
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a list of `RaggedTensor`s with len(segments) number of items and where
each item has the same shape as its counterpart in `segments` and
with unwanted values dropped. The values are dropped according to the
`TruncationStrategy` defined.
</td>
</tr>

</table>
