page_type: reference
description: List of predicted classes (aka labels) for a given classifier head.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.processor.Classifications" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tflite_support.task.processor.Classifications

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/classifications.proto">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



List of predicted classes (aka labels) for a given classifier head.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.processor.Classifications(
    categories: List[<a href="../../../tflite_support/task/processor/Category"><code>tflite_support.task.processor.Category</code></a>],
    head_index: int,
    head_name: str
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`categories`<a id="categories"></a>
</td>
<td>
The array of predicted categories, usually sorted by descending
scores (e.g. from high to low probability).
</td>
</tr><tr>
<td>
`head_index`<a id="head_index"></a>
</td>
<td>
The index of the classifier head these categories refer to. This
is useful for multi-head models.
</td>
</tr><tr>
<td>
`head_name`<a id="head_name"></a>
</td>
<td>
The name of the classifier head, which is the corresponding
tensor metadata.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/classifications.proto">View source</a>

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
