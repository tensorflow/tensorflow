description: Layer that makes padding and masking a Composite Tensors
effortless.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="text.keras.layers.ToDense" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# text.keras.layers.ToDense

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="https://github.com/tensorflow/text/tree/master/tensorflow_text/python/keras/layers/todense.py">View
source</a>

Layer that makes padding and masking a Composite Tensors effortless.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>text.keras.layers.ToDense(
    pad_value=0, mask=False, shape=None, **kwargs
)
</code></pre>

<!-- Placeholder for "Used in" -->

The layer takes a RaggedTensor or a SparseTensor and converts it to a uniform
tensor by right-padding it or filling in missing values.

#### Example:

```python
x = tf.keras.layers.Input(shape=(None, None), ragged=True)
y = tf_text.keras.layers.ToDense(mask=True)(x)
model = tf.keras.Model(x, y)

rt = tf.RaggedTensor.from_nested_row_splits(
  flat_values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
  nested_row_splits=([0, 1, 1, 5], [0, 3, 3, 5, 9, 10]))
model.predict(rt)

[[[10, 11, 12,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],
 [[ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0], [ 0,  0,  0,  0]],
 [[ 0,  0,  0,  0], [13, 14,  0,  0], [15, 16, 17, 18], [19,  0,  0,  0]]]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`pad_value`<a id="pad_value"></a>
</td>
<td>
A value used to pad and fill in the missing values. Should be a
meaningless value for the input data. Default is '0'.
</td>
</tr><tr>
<td>
`mask`<a id="mask"></a>
</td>
<td>
A Boolean value representing whether to mask the padded values. If
true, no any downstream Masking layer or Embedding layer with
mask_zero=True should be added. Default is 'False'.
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
If not `None`, the resulting dense tensor will be guaranteed to have
this shape. For RaggedTensor inputs, this is passed to `tf.RaggedTensor`'s
`to_tensor` method. For other tensor types, a `tf.ensure_shape` call is
added to assert that the output has this shape.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
kwargs of parent class.
</td>
</tr>
</table>

Input shape: Any Ragged or Sparse Tensor is accepted, but it requires the type
of input to be specified via the Input or InputLayer from the Keras API. Output
shape: The output is a uniform tensor having the same shape, in case of a ragged
input or the same dense shape, in case of a sparse input.
