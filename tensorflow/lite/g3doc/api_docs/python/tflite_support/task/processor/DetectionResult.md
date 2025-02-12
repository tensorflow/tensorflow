page_type: reference
description: Represents the list of detected objects.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.processor.DetectionResult" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tflite_support.task.processor.DetectionResult

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/detections.proto">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Represents the list of detected objects.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.processor.DetectionResult(
    detections: List[<a href="../../../tflite_support/task/processor/Detection"><code>tflite_support.task.processor.Detection</code></a>]
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`detections`<a id="detections"></a>
</td>
<td>
A list of `Detection` objects.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/detections.proto">View source</a>

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
