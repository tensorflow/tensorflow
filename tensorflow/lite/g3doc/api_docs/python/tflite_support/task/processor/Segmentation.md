page_type: reference
description: Represents one Segmentation object in the image segmenter's results.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.processor.Segmentation" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="category_mask"/>
<meta itemprop="property" content="confidence_masks"/>
</div>

# tflite_support.task.processor.Segmentation

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/segmentations.proto">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Represents one Segmentation object in the image segmenter's results.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.processor.Segmentation(
    height: int,
    width: int,
    colored_labels: List[<a href="../../../tflite_support/task/processor/ColoredLabel"><code>tflite_support.task.processor.ColoredLabel</code></a>],
    category_mask: Optional[np.ndarray] = None,
    confidence_masks: Optional[List[ConfidenceMask]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`height`<a id="height"></a>
</td>
<td>
The height of the mask. This is an intrinsic parameter of the model
being used, and does not depend on the input image dimensions.
</td>
</tr><tr>
<td>
`width`<a id="width"></a>
</td>
<td>
The width of the mask. This is an intrinsic parameter of the model
being used, and does not depend on the input image dimensions.
</td>
</tr><tr>
<td>
`colored_labels`<a id="colored_labels"></a>
</td>
<td>
A list of `ColoredLabel` objects.
</td>
</tr><tr>
<td>
`category_mask`<a id="category_mask"></a>
</td>
<td>
A NumPy 2D-array of the category mask.
</td>
</tr><tr>
<td>
`confidence_masks`<a id="confidence_masks"></a>
</td>
<td>
A list of `ConfidenceMask` objects.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/segmentations.proto">View source</a>

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
category_mask<a id="category_mask"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
confidence_masks<a id="confidence_masks"></a>
</td>
<td>
`None`
</td>
</tr>
</table>
