page_type: reference
description: Options for segmentation processor.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.processor.SegmentationOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="display_names_locale"/>
<meta itemprop="property" content="output_type"/>
</div>

# tflite_support.task.processor.SegmentationOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/segmentation_options.proto">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Options for segmentation processor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.processor.SegmentationOptions(
    display_names_locale: Optional[str] = None,
    output_type: Optional[<a href="../../../tflite_support/task/processor/OutputType"><code>tflite_support.task.processor.OutputType</code></a>] = <a href="../../../tflite_support/task/processor/OutputType#CATEGORY_MASK"><code>tflite_support.task.processor.OutputType.CATEGORY_MASK</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`display_names_locale`<a id="display_names_locale"></a>
</td>
<td>
The locale to use for display names specified through
the TFLite Model Metadata.
</td>
</tr><tr>
<td>
`output_type`<a id="output_type"></a>
</td>
<td>
The output mask type allows specifying the type of
post-processing to perform on the raw model results.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/segmentation_options.proto">View source</a>

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
display_names_locale<a id="display_names_locale"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
output_type<a id="output_type"></a>
</td>
<td>
`<OutputType.CATEGORY_MASK: 1>`
</td>
</tr>
</table>
