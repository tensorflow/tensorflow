description: Gather slices with indices=-1 mapped to default.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.gather_with_default" />
<meta itemprop="path" content="Stable" />
</div>

# text.gather_with_default

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/ops/pointer_ops.py">View
source</a>

Gather slices with `indices=-1` mapped to `default`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.gather_with_default(
    params, indices, default, name=None, axis=0
)
</code></pre>

<!-- Placeholder for "Used in" -->

This operation is similar to `tf.gather()`, except that any value of `-1`
in `indices` will be mapped to `default`.  Example:

```
>>> gather_with_default(['a', 'b', 'c', 'd'], [2, 0, -1, 2, -1], '_')
<tf.Tensor: shape=(5,), dtype=string,
    numpy=array([b'c', b'a', b'_', b'c', b'_'], dtype=object)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`params`<a id="params"></a>
</td>
<td>
The `Tensor` from which to gather values.  Must be at least rank
`axis + 1`.
</td>
</tr><tr>
<td>
`indices`<a id="indices"></a>
</td>
<td>
The index `Tensor`.  Must have dtype `int32` or `int64`, and values
must be in the range `[-1, params.shape[axis])`.
</td>
</tr><tr>
<td>
`default`<a id="default"></a>
</td>
<td>
The value to use when `indices` is `-1`.  `default.shape` must
be equal to `params.shape[axis + 1:]`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
The axis in `params` to gather `indices` from.  Must be a scalar
`int32` or `int64`.  Supports negative indices.
</td>
</tr>
</table>

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` with the same type as `param`, and with shape
`params.shape[:axis] + indices.shape + params.shape[axis + 1:]`.
</td>
</tr>

</table>
