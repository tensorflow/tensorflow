description: Type of reduction to be done by the n-gram op.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.Reduction" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="MEAN"/>
<meta itemprop="property" content="STRING_JOIN"/>
<meta itemprop="property" content="SUM"/>
</div>

# text.Reduction

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/ngrams_op.py">View
source</a>

Type of reduction to be done by the n-gram op.

<!-- Placeholder for "Used in" -->

The supported reductions are as follows:

*   <a href="../text/Reduction.md#SUM"><code>Reduction.SUM</code></a>: Add
    values in the window.
*   <a href="../text/Reduction.md#MEAN"><code>Reduction.MEAN</code></a>: Average
    values in the window.
*   <a href="../text/Reduction.md#STRING_JOIN"><code>Reduction.STRING_JOIN</code></a>:
    Join strings in the window.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
MEAN<a id="MEAN"></a>
</td>
<td>
`<Reduction.MEAN: 2>`
</td>
</tr><tr>
<td>
STRING_JOIN<a id="STRING_JOIN"></a>
</td>
<td>
`<Reduction.STRING_JOIN: 3>`
</td>
</tr><tr>
<td>
SUM<a id="SUM"></a>
</td>
<td>
`<Reduction.SUM: 1>`
</td>
</tr>
</table>
