description: Determine wordshape features for each input string.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.wordshape" />
<meta itemprop="path" content="Stable" />
</div>

# text.wordshape

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/wordshape_ops.py">View
source</a>

Determine wordshape features for each input string.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.wordshape(
    input_tensor, pattern, name=None
)
</code></pre>

<!-- Placeholder for "Used in" -->

In this example, we test for title case (the first character is upper or title
case, and the remaining characters are lowercase). ```

> > > input = [ ... u"abc", u"ABc", u"ABC", u"Abc", u"aBcd",
> > > u"\u01c8bc".encode("utf-8") ... ] wordshape(input,
> > > WordShape.HAS_TITLE_CASE)
> > > <tf.Tensor: shape=(6,), dtype=bool, numpy=array([False, False, False, True, False, True])>
> > > ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_tensor`<a id="input_tensor"></a>
</td>
<td>
string `Tensor` with any shape.
</td>
</tr><tr>
<td>
`pattern`<a id="pattern"></a>
</td>
<td>
A `tftext.WordShape` or a list of WordShapes.
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
`<bool>[input_tensor.shape + pattern.shape]`: A tensor where
`result[i1...iN, j]` is true if `input_tensor[i1...iN]` has the wordshape
specified by `pattern[j]`.
</td>
</tr>

</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If `pattern` contains an unknown identifier.
</td>
</tr>
</table>
