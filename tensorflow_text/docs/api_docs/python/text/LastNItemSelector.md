description: An ItemSelector that selects the last n items in the batch.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.LastNItemSelector" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_selectable"/>
<meta itemprop="property" content="get_selection_mask"/>
</div>

# text.LastNItemSelector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/item_selector_ops.py">View
source</a>

An `ItemSelector` that selects the last `n` items in the batch.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.LastNItemSelector(
    num_to_select, unselectable_ids=None
)
</code></pre>

<!-- Placeholder for "Used in" -->
<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_to_select`<a id="num_to_select"></a>
</td>
<td>
An int which is the leading number of items to select.
</td>
</tr><tr>
<td>
`unselectable_ids`<a id="unselectable_ids"></a>
</td>
<td>
(optional) A list of int ids that cannot be selected.
Default is empty list.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `unselectable_ids`<a id="unselectable_ids"></a> </td> <td>

</td>
</tr>
</table>

## Methods

<h3 id="get_selectable"><code>get_selectable</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/item_selector_ops.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_selectable(
    input_ids, axis
)
</code></pre>

See `get_selectable()` in superclass.

<h3 id="get_selection_mask"><code>get_selection_mask</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/item_selector_ops.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_selection_mask(
    input_ids, axis=1
)
</code></pre>

Returns a mask of items that have been selected.

The default implementation simply returns all items not excluded by
`get_selectable`.

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_ids`
</td>
<td>
A `RaggedTensor`.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
(optional) An int detailing the dimension to apply selection on.
Default is the 1st dimension.
</td>
</tr>
</table>

<!-- Tabular view -->

 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a `RaggedTensor` with shape `input_ids.shape[:axis]`. Its values are True
if the corresponding item (or broadcasted subitems) should be selected.
</td>
</tr>

</table>
