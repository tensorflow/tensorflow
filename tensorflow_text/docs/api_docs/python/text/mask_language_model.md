description: Applies dynamic language model masking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.mask_language_model" />
<meta itemprop="path" content="Stable" />
</div>

# text.mask_language_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/masking_ops.py">View
source</a>

Applies dynamic language model masking.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.mask_language_model(
    input_ids, item_selector, mask_values_chooser, axis=1
)
</code></pre>



<!-- Placeholder for "Used in" -->

`mask_language_model` implements the `Masked LM and Masking Procedure`
described in `BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding`  (https://arxiv.org/pdf/1810.04805.pdf).
`mask_language_model` uses an `ItemSelector` to select the items for masking,
and a `MaskValuesChooser` to assign the values to the selected items.
The purpose of this is to bias the representation towards the actual
observed item.

Masking is performed on items in an axis. A decision is taken independently at
random to mask with [MASK], mask with random tokens from the full vocab, or
not mask at all. Note that the masking decision is broadcasted to the
sub-dimensions.

For example, in a RaggedTensor of shape `[batch, (wordpieces)]` and if axis=1,
each wordpiece independently gets masked (or not).

With the following input:

```
[[b"Sp", b"##onge", b"bob", b"Sq", b"##uare", b"##pants" ],
[b"Bar", b"##ack", b"Ob", b"##ama"],
[b"Mar", b"##vel", b"A", b"##ven", b"##gers"]],
```

`mask_language_model` could end up masking individual wordpieces:

```
[[b"[MASK]", b"##onge", b"bob", b"Sq", b"[MASK]", b"##pants" ],
[b"Bar", b"##ack", b"[MASK]", b"##ama"],
[b"[MASK]", b"##vel", b"A", b"##ven", b"##gers"]]
```

Or with random token inserted:

```
[[b"[MASK]", b"##onge", b"bob", b"Sq", b"[MASK]", b"##pants" ],
[b"Bar", b"##ack", b"Sq", b"##ama"],   # random token inserted for 'Ob'
[b"Bar", b"##vel", b"A", b"##ven", b"##gers"]]  # random token inserted for
                                                # 'Mar'
```

In a RaggedTensor of shape `[batch, (words), (wordpieces)]`, whole words get
masked (or not). If a word gets masked, all its tokens are independently
either replaced by `[MASK]`, by random tokens, or no substitution occurs.
Note that any arbitrary spans that can be constructed by a `RaggedTensor` can
be masked in the same way.

For example, if we have an `RaggedTensor` with shape
`[batch, (token), (wordpieces)]`:

```
[[[b"Sp", "##onge"], [b"bob"], [b"Sq", b"##uare", b"##pants"]],
 [[b"Bar", "##ack"], [b"Ob", b"##ama"]],
 [[b"Mar", "##vel"], [b"A", b"##ven", b"##gers"]]]
```

`mask_language_model` could mask whole spans (items grouped together
by the same 1st dimension):

```
[[[b"[MASK]", "[MASK]"], [b"bob"], [b"Sq", b"##uare", b"##pants"]],
 [[b"Bar", "##ack"], [b"[MASK]", b"[MASK]"]],
 [[b"[MASK]", "[MASK]"], [b"A", b"##ven", b"##gers"]]]
```

or insert random items in spans:

```
 [[[b"Mar", "##ama"], [b"bob"], [b"Sq", b"##uare", b"##pants"]],
  [[b"Bar", "##ack"], [b"##onge", b"##gers"]],
  [[b"Ob", "Sp"], [b"A", b"##ven", b"##gers"]]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_ids`<a id="input_ids"></a>
</td>
<td>
A `RaggedTensor` of n dimensions (where n >= 2) on which
masking will be applied to items up to dimension 1.
</td>
</tr><tr>
<td>
`item_selector`<a id="item_selector"></a>
</td>
<td>
An instance of `ItemSelector` that is used for selecting
items to be masked.
</td>
</tr><tr>
<td>
`mask_values_chooser`<a id="mask_values_chooser"></a>
</td>
<td>
An instance of `MaskValuesChooser` which determines the
values assigned to the ids chosen for masking.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
the axis where items will be treated atomically for masking.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tuple of (masked_input_ids, masked_positions, masked_ids) where:
</td>
</tr>
<tr>
<td>
`masked_input_ids`<a id="masked_input_ids"></a>
</td>
<td>
A `RaggedTensor` in the same shape and dtype as
`input_ids`, but with items in `masked_positions` possibly replaced
with `mask_token`, random id, or no change.
</td>
</tr><tr>
<td>
`masked_positions`<a id="masked_positions"></a>
</td>
<td>
A `RaggedTensor` of ints with shape
[batch, (num_masked)] containing the positions of items selected for
masking.
</td>
</tr><tr>
<td>
`masked_ids`<a id="masked_ids"></a>
</td>
<td>
A `RaggedTensor` with shape [batch, (num_masked)] and same
type as `input_ids` containing the original values before masking
and thus used as labels for the task.
</td>
</tr>
</table>
