page_type: reference
description: Options for Bert CLU Annotator processor.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.processor.BertCluAnnotationOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="categorical_slot_threshold"/>
<meta itemprop="property" content="domain_threshold"/>
<meta itemprop="property" content="intent_threshold"/>
<meta itemprop="property" content="max_history_turns"/>
<meta itemprop="property" content="mentioned_slot_threshold"/>
</div>

# tflite_support.task.processor.BertCluAnnotationOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/clu_annotation_options.proto">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Options for Bert CLU Annotator processor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.processor.BertCluAnnotationOptions(
    max_history_turns: Optional[int] = 5,
    domain_threshold: Optional[float] = 0.5,
    intent_threshold: Optional[float] = 0.5,
    categorical_slot_threshold: Optional[float] = 0.5,
    mentioned_slot_threshold: Optional[float] = 0.5
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`max_history_turns`<a id="max_history_turns"></a>
</td>
<td>
Max number of history turns to encode by the model.
</td>
</tr><tr>
<td>
`domain_threshold`<a id="domain_threshold"></a>
</td>
<td>
The threshold of domain prediction.
</td>
</tr><tr>
<td>
`intent_threshold`<a id="intent_threshold"></a>
</td>
<td>
The threshold of intent prediction.
</td>
</tr><tr>
<td>
`categorical_slot_threshold`<a id="categorical_slot_threshold"></a>
</td>
<td>
The threshold of categorical slot prediction.
</td>
</tr><tr>
<td>
`mentioned_slot_threshold`<a id="mentioned_slot_threshold"></a>
</td>
<td>
The threshold of mentioned slot
prediction.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/clu_annotation_options.proto">View source</a>

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
categorical_slot_threshold<a id="categorical_slot_threshold"></a>
</td>
<td>
`0.5`
</td>
</tr><tr>
<td>
domain_threshold<a id="domain_threshold"></a>
</td>
<td>
`0.5`
</td>
</tr><tr>
<td>
intent_threshold<a id="intent_threshold"></a>
</td>
<td>
`0.5`
</td>
</tr><tr>
<td>
max_history_turns<a id="max_history_turns"></a>
</td>
<td>
`5`
</td>
</tr><tr>
<td>
mentioned_slot_threshold<a id="mentioned_slot_threshold"></a>
</td>
<td>
`0.5`
</td>
</tr>
</table>
