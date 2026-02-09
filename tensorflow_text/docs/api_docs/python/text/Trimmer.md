description: Truncates a list of segments using a pre-determined truncation
strategy.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.Trimmer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="generate_mask"/>
<meta itemprop="property" content="trim"/>
</div>

# text.Trimmer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/trimmer_ops.py">View
source</a>

Truncates a list of segments using a pre-determined truncation strategy.

<!-- Placeholder for "Used in" -->

## Methods

<h3 id="generate_mask"><code>generate_mask</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/trimmer_ops.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>generate_mask(
    segments
)
</code></pre>

Generates a boolean mask specifying which portions of `segments` to drop.

Users should be able to use the results of generate_mask() to drop items in
segments using `tf.ragged.boolean_mask(seg, mask)`.

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
a list with len(segments) number of items and where each item is a
`RaggedTensor` with the same shape as its counterpart in `segments` and
with a boolean dtype where each value is True if the corresponding
value in `segments` should be kept and False if it should be dropped
instead.
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
