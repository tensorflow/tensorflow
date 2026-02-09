description: Assigns values to the items chosen for masking.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.MaskValuesChooser" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_mask_values"/>
</div>

# text.MaskValuesChooser

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/masking_ops.py">View
source</a>

Assigns values to the items chosen for masking.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.MaskValuesChooser(
    vocab_size, mask_token, mask_token_rate=0.8, random_token_rate=0.1
)
</code></pre>



<!-- Placeholder for "Used in" -->

`MaskValuesChooser` encapsulates the logic for deciding the value to assign
items that where chosen for masking. The following are the behavior in the
default implementation:

For `mask_token_rate` of the time, replace the item with the `[MASK]` token:

```
my dog is hairy -> my dog is [MASK]
```

For `random_token_rate` of the time, replace the item with a random word:

```
my dog is hairy -> my dog is apple
```

For `1 - mask_token_rate - random_token_rate` of the time, keep the item
unchanged:

```
my dog is hairy -> my dog is hairy.
```

The default behavior is consistent with the methodology specified in
`Masked LM and Masking Procedure` described in `BERT: Pre-training of Deep
Bidirectional Transformers for Language Understanding`
(https://arxiv.org/pdf/1810.04805.pdf).

Users may further customize this with behavior through subclassing and
overriding `get_mask_values()`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`vocab_size`<a id="vocab_size"></a>
</td>
<td>
size of vocabulary.
</td>
</tr><tr>
<td>
`mask_token`<a id="mask_token"></a>
</td>
<td>
The id of the mask token.
</td>
</tr><tr>
<td>
`mask_token_rate`<a id="mask_token_rate"></a>
</td>
<td>
(optional) A float between 0 and 1 which indicates how
often the `mask_token` is substituted for tokens selected for masking.
Default is 0.8, NOTE: `mask_token_rate` + `random_token_rate` <= 1.
</td>
</tr><tr>
<td>
`random_token_rate`<a id="random_token_rate"></a>
</td>
<td>
A float between 0 and 1 which indicates how often a
random token is substituted for tokens selected for masking. Default is
0.1. NOTE: `mask_token_rate` + `random_token_rate` <= 1.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr> <td> `mask_token`<a id="mask_token"></a> </td> <td>

</td> </tr><tr> <td> `random_token_rate`<a id="random_token_rate"></a> </td>
<td>

</td> </tr><tr> <td> `vocab_size`<a id="vocab_size"></a> </td> <td>

</td>
</tr>
</table>



## Methods

<h3 id="get_mask_values"><code>get_mask_values</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/masking_ops.py">View
source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_mask_values(
    masked_lm_ids
)
</code></pre>

Get the values used for masking, random injection or no-op.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`masked_lm_ids`
</td>
<td>
a `RaggedTensor` of n dimensions and dtype int32 or int64
whose values are the ids of items that have been selected for masking.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
a `RaggedTensor` of the same dtype and shape with `masked_lm_ids` whose
values contain either the mask token, randomly injected token or original
value.
</td>
</tr>

</table>





