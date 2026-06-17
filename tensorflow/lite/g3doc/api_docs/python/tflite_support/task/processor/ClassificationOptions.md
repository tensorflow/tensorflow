page_type: reference
description: Options for classification processor.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.processor.ClassificationOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="category_name_allowlist"/>
<meta itemprop="property" content="category_name_denylist"/>
<meta itemprop="property" content="display_names_locale"/>
<meta itemprop="property" content="max_results"/>
<meta itemprop="property" content="score_threshold"/>
</div>

# tflite_support.task.processor.ClassificationOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/classification_options.proto">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Options for classification processor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.processor.ClassificationOptions(
    score_threshold: Optional[float] = None,
    category_name_allowlist: Optional[List[str]] = None,
    category_name_denylist: Optional[List[str]] = None,
    display_names_locale: Optional[str] = None,
    max_results: Optional[int] = None
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
`max_results`<a id="max_results"></a>
</td>
<td>
The maximum number of top-scored classification results to
return.
</td>
</tr><tr>
<td>
`score_threshold`<a id="score_threshold"></a>
</td>
<td>
Overrides the ones provided in the model metadata. Results
below this value are rejected.
</td>
</tr><tr>
<td>
`category_name_allowlist`<a id="category_name_allowlist"></a>
</td>
<td>
If non-empty, classifications whose class name is
not in this set will be filtered out. Duplicate or unknown class names are
ignored. Mutually exclusive with `category_name_denylist`.
</td>
</tr><tr>
<td>
`category_name_denylist`<a id="category_name_denylist"></a>
</td>
<td>
If non-empty, classifications whose class name is in
this set will be filtered out. Duplicate or unknown class names are
ignored. Mutually exclusive with `category_name_allowlist`.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/classification_options.proto">View source</a>

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
category_name_allowlist<a id="category_name_allowlist"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
category_name_denylist<a id="category_name_denylist"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
display_names_locale<a id="display_names_locale"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
max_results<a id="max_results"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
score_threshold<a id="score_threshold"></a>
</td>
<td>
`None`
</td>
</tr>
</table>
