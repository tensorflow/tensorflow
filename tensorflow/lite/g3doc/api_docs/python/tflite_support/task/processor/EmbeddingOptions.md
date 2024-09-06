page_type: reference
description: Options for embedding processor.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.processor.EmbeddingOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="l2_normalize"/>
<meta itemprop="property" content="quantize"/>
</div>

# tflite_support.task.processor.EmbeddingOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/embedding_options.proto">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Options for embedding processor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.processor.EmbeddingOptions(
    l2_normalize: Optional[bool] = None, quantize: Optional[bool] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`l2_normalize`<a id="l2_normalize"></a>
</td>
<td>
Whether to normalize the returned feature vector with L2 norm.
Use this option only if the model does not already contain a native
L2_NORMALIZATION TF Lite Op. In most cases, this is already the case and
L2 norm is thus achieved through TF Lite inference.
</td>
</tr><tr>
<td>
`quantize`<a id="quantize"></a>
</td>
<td>
Whether the returned embedding should be quantized to bytes via
scalar quantization. Embeddings are implicitly assumed to be unit-norm and
therefore any dimension is guaranteed to have a value in [-1.0, 1.0]. Use
the l2_normalize option if this is not the case.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/embedding_options.proto">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other: Any
) -> bool
</code></pre>

Checks if this object is equal to the given object.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
The object to be compared with.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
True if the objects are equal.
</td>
</tr>

</table>







<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
l2_normalize<a id="l2_normalize"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
quantize<a id="quantize"></a>
</td>
<td>
`None`
</td>
</tr>
</table>
