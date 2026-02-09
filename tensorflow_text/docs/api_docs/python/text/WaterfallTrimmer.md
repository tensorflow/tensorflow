description: A Trimmer that allocates a length budget to segments in order.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.WaterfallTrimmer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="generate_mask"/>
<meta itemprop="property" content="trim"/>
</div>

# text.WaterfallTrimmer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/trimmer_ops.py">View
source</a>

A `Trimmer` that allocates a length budget to segments in order.

Inherits From: [`Trimmer`](../text/Trimmer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.WaterfallTrimmer(
    max_seq_length, axis=-1
)
</code></pre>



<!-- Placeholder for "Used in" -->

A `Trimmer` that allocates a length budget to segments in order. It selects
elements to drop, according to a max sequence length budget, and then applies
this mask to actually drop the elements. See `generate_mask()` for more details.

#### Example:

```
>>> a = tf.ragged.constant([['a', 'b', 'c'], [], ['d']])
>>> b = tf.ragged.constant([['1', '2', '3'], [], ['4', '5', '6', '7']])
>>> trimmer = tf_text.WaterfallTrimmer(4)
>>> trimmer.trim([a, b])
[<tf.RaggedTensor [[b'a', b'b', b'c'], [], [b'd']]>,
 <tf.RaggedTensor [[b'1'], [], [b'4', b'5', b'6']]>]
```

Here, for the first pair of elements, `['a', 'b', 'c']` and `['1', '2', '3']`,
the `'2'` and `'3'` are dropped to fit the sequence within the max sequence
length budget.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`max_seq_length`<a id="max_seq_length"></a>
</td>
<td>
a scalar `Tensor` or a 1D `Tensor` of type int32 that
describes the number max number of elements allowed in a batch. If a
scalar is provided, the value is broadcasted and applied to all values
across the batch.
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

Calculate a truncation mask given a budget of the max number of items for
each or all batch row. The allocation of the budget is done using a
'waterfall' algorithm. This algorithm allocates quota in a left-to-right
manner and fill up the buckets until we run out of budget.

For example if the budget of [5] and we have segments of size
[3, 4, 2], the truncate budget will be allocated as [3, 2, 0].

The budget can be a scalar, in which case the same budget is broadcasted
and applied to all batch rows. It can also be a 1D `Tensor` of size
`batch_size`, in which each batch row i will have a budget corresponding to
`per_batch_quota[i]`.

#### Example:

```
>>> a = tf.ragged.constant([['a', 'b', 'c'], [], ['d']])
>>> b = tf.ragged.constant([['1', '2', '3'], [], ['4', '5', '6', '7']])
>>> trimmer = tf_text.WaterfallTrimmer(4)
>>> trimmer.generate_mask([a, b])
[<tf.RaggedTensor [[True, True, True], [], [True]]>,
 <tf.RaggedTensor [[True, False, False], [], [True, True, True, False]]>]
```

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





