page_type: reference
description: Options for search processor.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.processor.SearchOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="index_file_content"/>
<meta itemprop="property" content="index_file_name"/>
<meta itemprop="property" content="max_results"/>
</div>

# tflite_support.task.processor.SearchOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/search_options.proto">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Options for search processor.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.processor.SearchOptions(
    index_file_name: Optional[str] = None,
    index_file_content: Optional[bytes] = None,
    max_results: Optional[int] = 5
)
</code></pre>



<!-- Placeholder for "Used in" -->

The index file to search into. Mandatory only if the index is not attached
to the output tensor metadata as an AssociatedFile with type SCANN_INDEX_FILE.
The index file can be specified by one of the following two ways:

(1) file contents loaded in `index_file_content`.
(2) file path in `index_file_name`.

If more than one field of these fields is provided, they are used in this
precedence order.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`index_file_name`<a id="index_file_name"></a>
</td>
<td>
Path to the index.
</td>
</tr><tr>
<td>
`index_file_content`<a id="index_file_content"></a>
</td>
<td>
The index file contents as bytes.
</td>
</tr><tr>
<td>
`max_results`<a id="max_results"></a>
</td>
<td>
Maximum number of nearest neighbor results to return.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/processor/proto/search_options.proto">View source</a>

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
index_file_content<a id="index_file_content"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
index_file_name<a id="index_file_name"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
max_results<a id="max_results"></a>
</td>
<td>
`5`
</td>
</tr>
</table>
