description: A Trimmer that allocates a length budget to segments via round
robin.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.RoundRobinTrimmer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="generate_mask"/>
<meta itemprop="property" content="trim"/>
</div>

# text.RoundRobinTrimmer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/trimmer_ops.py">View
source</a>

A `Trimmer` that allocates a length budget to segments via round robin.

Inherits From: [`Trimmer`](../text/Trimmer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.RoundRobinTrimmer(
    max_seq_length, axis=-1
)
</code></pre>

<!-- Placeholder for "Used in" -->

A `Trimmer` that allocates a length budget to segments using a round robin
strategy, then drops elements outside of the segment's allocated budget. See
`generate_mask()` for more details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`max_seq_length`<a id="max_seq_length"></a>
</td>
<td>
a scalar `Tensor` int32 that describes the number max
number of elements allowed in a batch.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
Axis to apply trimming on.
</td>
</tr>
</table>

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
or all batch row. The allocation of the budget is done using a 'round robin'
algorithm. This algorithm allocates quota in each bucket, left-to-right
repeatedly until all the buckets are filled.

For example if the budget of [5] and we have segments of size [3, 4, 2], the
truncate budget will be allocated as [2, 2, 1].

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`segments`
</td>
<td>
A list of `RaggedTensor`s each with a shape of [num_batch,
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
A list with len(segments) of `RaggedTensor`s, see superclass for details.
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

Truncate the list of `segments` using the 'round-robin' strategy which allocates
quota in each bucket, left-to-right repeatedly until all buckets are filled.

For example if the budget of [5] and we have segments of size [3, 4, 2], the
truncate budget will be allocated as [2, 2, 1].

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
A list with len(segments) of `RaggedTensor`s, see superclass for details.
</td>
</tr>

</table>
