description: An ItemSelector implementation that randomly selects items in a batch.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.RandomItemSelector" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_selectable"/>
<meta itemprop="property" content="get_selection_mask"/>
</div>

# text.RandomItemSelector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/item_selector_ops.py">View
source</a>

An `ItemSelector` implementation that randomly selects items in a batch.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.RandomItemSelector(
    max_selections_per_batch,
    selection_rate,
    unselectable_ids=None,
    shuffle_fn=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

`RandomItemSelector` randomly selects items in a batch subject to
restrictions given (max_selections_per_batch, selection_rate and
unselectable_ids).

#### Example:

```
>>> vocab = ["[UNK]", "[MASK]", "[RANDOM]", "[CLS]", "[SEP]",
...          "abc", "def", "ghi"]
>>> # Note that commonly in masked language model work, there are
>>> # special tokens we don't want to mask, like CLS, SEP, and probably
>>> # any OOV (out-of-vocab) tokens here called UNK.
>>> # Note that if e.g. there are bucketed OOV tokens in the code,
>>> # that might be a use case for overriding `get_selectable()` to
>>> # exclude a range of IDs rather than enumerating them.
>>> tf.random.set_seed(1234)
>>> selector = tf_text.RandomItemSelector(
...     max_selections_per_batch=2,
...     selection_rate=0.2,
...     unselectable_ids=[0, 3, 4])  # indices of UNK, CLS, SEP
>>> selection = selector.get_selection_mask(
...     tf.ragged.constant([[3, 5, 7, 7], [4, 6, 7, 5]]), axis=1)
>>> print(selection)
<tf.RaggedTensor [[False, False, False, True], [False, False, True, False]]>
```

The selection has skipped the first elements (the CLS and SEP token codings) and
picked random elements from the other elements of the segments -- if run with a
different random seed the selections might be different.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`max_selections_per_batch`<a id="max_selections_per_batch"></a>
</td>
<td>
An int of the max number of items to mask out.
</td>
</tr><tr>
<td>
`selection_rate`<a id="selection_rate"></a>
</td>
<td>
The rate at which items are randomly selected.
</td>
</tr><tr>
<td>
`unselectable_ids`<a id="unselectable_ids"></a>
</td>
<td>
(optional) A list of python ints or 1D `Tensor` of ints
which are ids that will be not be masked.
</td>
</tr><tr>
<td>
`shuffle_fn`<a id="shuffle_fn"></a>
</td>
<td>
(optional) A function that shuffles a 1D `Tensor`. Default
uses `tf.random.shuffle`.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `max_selections_per_batch`<a id="max_selections_per_batch"></a> </td>
<td>

</td> </tr><tr> <td> `selection_rate`<a id="selection_rate"></a> </td> <td>

</td> </tr><tr> <td> `shuffle_fn`<a id="shuffle_fn"></a> </td> <td>

</td> </tr><tr> <td> `unselectable_ids`<a id="unselectable_ids"></a> </td> <td>

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

Return a boolean mask of items that can be chosen for selection.

The default implementation marks all items whose IDs are not in the
`unselectable_ids` list. This can be overridden if there is a need for a more
complex or algorithmic approach for selectability.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_ids`
</td>
<td>
a `RaggedTensor`.
</td>
</tr><tr>
<td>
`axis`
</td>
<td>
axis to apply selection on.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a `RaggedTensor` with dtype of bool and with shape
`input_ids.shape[:axis]`. Its values are True if the
corresponding item (or broadcasted subitems) should be considered for
masking. In the default implementation, all `input_ids` items that are not
listed in `unselectable_ids` (from the class arg) are considered
selectable.
</td>
</tr>

</table>



<h3 id="get_selection_mask"><code>get_selection_mask</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/item_selector_ops.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_selection_mask(
    input_ids, axis
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





