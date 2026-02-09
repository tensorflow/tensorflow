description: Returns a subgradient of the MaximumSpanningTree op.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.max_spanning_tree_gradient" />
<meta itemprop="path" content="Stable" />
</div>

# text.max_spanning_tree_gradient

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/mst_ops.py">View
source</a>

Returns a subgradient of the MaximumSpanningTree op.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.max_spanning_tree_gradient(
    mst_op, d_loss_d_max_scores, *_
)
</code></pre>

<!-- Placeholder for "Used in" -->

Note that MaximumSpanningTree is only differentiable w.r.t. its |scores| input
and its |max_scores| output.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mst_op`<a id="mst_op"></a>
</td>
<td>
The MaximumSpanningTree op being differentiated.
</td>
</tr><tr>
<td>
`d_loss_d_max_scores`<a id="d_loss_d_max_scores"></a>
</td>
<td>
[B] vector where entry b is the gradient of the network
loss w.r.t. entry b of the |max_scores| output of the |mst_op|.
</td>
</tr><tr>
<td>
`*_`<a id="*_"></a>
</td>
<td>
The gradients w.r.t. the other outputs; ignored.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
1. None, since the op is not differentiable w.r.t. its |num_nodes| input.
2. [B,M,M] tensor where entry b,t,s is a subgradient of the network loss
   w.r.t. entry b,t,s of the |scores| input, with the same dtype as
   |d_loss_d_max_scores|.
</td>
</tr>

</table>
