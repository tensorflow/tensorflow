page_type: reference
description: The output of CLU.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.processor.CluResponse" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
</div>

# tflite_support.task.processor.CluResponse

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/clu.proto">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The output of CLU.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.processor.CluResponse(
    domains: List[<a href="../../../tflite_support/task/processor/Category"><code>tflite_support.task.processor.Category</code></a>],
    intents: List[<a href="../../../tflite_support/task/processor/Category"><code>tflite_support.task.processor.Category</code></a>],
    categorical_slots: List[<a href="../../../tflite_support/task/processor/CategoricalSlot"><code>tflite_support.task.processor.CategoricalSlot</code></a>],
    mentioned_slots: List[<a href="../../../tflite_support/task/processor/MentionedSlot"><code>tflite_support.task.processor.MentionedSlot</code></a>]
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`domains`<a id="domains"></a>
</td>
<td>
The list of predicted domains.
</td>
</tr><tr>
<td>
`intents`<a id="intents"></a>
</td>
<td>
The list of predicted intents.
</td>
</tr><tr>
<td>
`categorical_slots`<a id="categorical_slots"></a>
</td>
<td>
The list of predicted categorical slots.
</td>
</tr><tr>
<td>
`mentioned_slots`<a id="mentioned_slots"></a>
</td>
<td>
The list of predicted mentioned slots.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/clu.proto">View source</a>

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
