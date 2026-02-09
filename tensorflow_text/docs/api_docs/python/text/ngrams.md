description: Create a tensor of n-grams based on the input data data.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.ngrams" />
<meta itemprop="path" content="Stable" />
</div>

# text.ngrams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/ngrams_op.py">View
source</a>

Create a tensor of n-grams based on the input data `data`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.ngrams(
    data,
    width,
    axis=-1,
    reduction_type=None,
    string_separator=&#x27; &#x27;,
    name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

Creates a tensor of n-grams based on `data`. The n-grams are of width `width`
and are created along axis `axis`; the n-grams are created by combining
windows of `width` adjacent elements from `data` using `reduction_type`. This
op is intended to cover basic use cases; more complex combinations can be
created using the sliding_window op.

```
>>> input_data = tf.ragged.constant([["e", "f", "g"], ["dd", "ee"]])
>>> ngrams(
...   input_data,
...   width=2,
...   axis=-1,
...   reduction_type=Reduction.STRING_JOIN,
...   string_separator="|")
<tf.RaggedTensor [[b'e|f', b'f|g'], [b'dd|ee']]>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr> <td> `data`<a id="data"></a> </td> <td> The data to reduce. </td> </tr><tr>
<td> `width`<a id="width"></a> </td> <td> The width of the ngram window. If
there is not sufficient data to fill out the ngram window, the resulting ngram
will be empty. </td> </tr><tr> <td> `axis`<a id="axis"></a> </td> <td> The axis
to create ngrams along. Note that for string join reductions, only axis '-1' is
supported; for other reductions, any positive or negative axis can be used.
Should be a constant. </td> </tr><tr> <td>
`reduction_type`<a id="reduction_type"></a> </td> <td> A member of the Reduction
enum. Should be a constant. Currently supports:

*   <a href="../text/Reduction.md#SUM"><code>Reduction.SUM</code></a>: Add
    values in the window.
*   <a href="../text/Reduction.md#MEAN"><code>Reduction.MEAN</code></a>: Average
    values in the window.
*   <a href="../text/Reduction.md#STRING_JOIN"><code>Reduction.STRING_JOIN</code></a>: Join strings in the window.
    Note that axis must be -1 here.
    </td>
    </tr><tr>
    <td>
    `string_separator`<a id="string_separator"></a>
    </td>
    <td>
    The separator string used for <a href="../text/Reduction.md#STRING_JOIN"><code>Reduction.STRING_JOIN</code></a>.
    Ignored otherwise. Must be a string constant, not a Tensor.
    </td>
    </tr><tr>
    <td>
    `name`<a id="name"></a>
    </td>
    <td>
    The op name.
    </td>
    </tr>
    </table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensor of ngrams. If the input is a tf.Tensor, the output will also
be a tf.Tensor; if the input is a tf.RaggedTensor, the output will be
a tf.RaggedTensor.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`InvalidArgumentError`<a id="InvalidArgumentError"></a>
</td>
<td>
if `reduction_type` is either None or not a Reduction,
or if `reduction_type` is STRING_JOIN and `axis` is not -1.
</td>
</tr>
</table>
