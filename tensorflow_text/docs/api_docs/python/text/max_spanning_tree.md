description: Finds the maximum directed spanning tree of a digraph.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.max_spanning_tree" />
<meta itemprop="path" content="Stable" />
</div>

# text.max_spanning_tree

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/gen_mst_ops.py">View
source</a>

Finds the maximum directed spanning tree of a digraph.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.max_spanning_tree(
    num_nodes, scores, forest=False, name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Given a batch of directed graphs with scored arcs and root selections, solves
for the maximum spanning tree of each digraph, where the score of a tree is
defined as the sum of the scores of the arcs and roots making up the tree.

Returns the score of the maximum spanning tree of each digraph, as well as the
arcs and roots in that tree. Each digraph in a batch may contain a different
number of nodes, so the sizes of the digraphs must be provided as an input.

Note that this operation is only differentiable w.r.t. its |scores| input and
its |max_scores| output.

The code here is intended for NLP applications, but attempts to remain agnostic
to particular NLP tasks (such as dependency parsing).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_nodes`<a id="num_nodes"></a>
</td>
<td>
A `Tensor` of type `int32`.
[B] vector where entry b is number of nodes in the b'th digraph.
</td>
</tr><tr>
<td>
`scores`<a id="scores"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `float32`, `float64`.
[B,M,M] tensor where entry b,t,s is the score of the arc from node s to
node t in the b'th directed graph if s!=t, or the score of selecting
node t as a root in the b'th digraph if s==t. This uniform tenosor
requires that M is >= num_nodes[b] for all b (ie. all graphs in the
batch), and ignores entries b,s,t where s or t is >= num_nodes[b].
Arcs or root selections with non-finite score are treated as
nonexistent.
</td>
</tr><tr>
<td>
`forest`<a id="forest"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
If true, solves for a maximum spanning forest instead of a maximum
spanning tree, where a spanning forest is a set of disjoint trees that
span the nodes of the digraph.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple of `Tensor` objects (max_scores, argmax_sources).
</td>
</tr>
<tr>
<td>
`max_scores`<a id="max_scores"></a>
</td>
<td>
A `Tensor`. Has the same type as `scores`. [B] vector where entry b is the score of the maximum spanning tree
of the b'th digraph.
</td>
</tr><tr>
<td>
`argmax_sources`<a id="argmax_sources"></a>
</td>
<td>
A `Tensor` of type `int32`. [B,M] matrix where entry b,t is the source of the arc inbound to
t in the maximum spanning tree of the b'th digraph, or t if t is
a root. Entries b,t where t is >= num_nodes[b] are set to -1.
Quickly finding the roots can be done as:
tf.equal(tf.map_fn(lambda x: tf.range(tf.size(x)),
argmax_sources), argmax_sources)
</td>
</tr>
</table>
