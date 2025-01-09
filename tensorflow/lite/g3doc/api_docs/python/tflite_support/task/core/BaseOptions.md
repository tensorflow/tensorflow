page_type: reference
description: Base options for TensorFlow Lite Task Library's Python APIs.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.task.core.BaseOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="file_content"/>
<meta itemprop="property" content="file_name"/>
<meta itemprop="property" content="num_threads"/>
<meta itemprop="property" content="use_coral"/>
</div>

# tflite_support.task.core.BaseOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/core/base_options.py#L25-L84">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Base options for TensorFlow Lite Task Library's Python APIs.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.task.core.BaseOptions(
    file_name: Optional[str] = None,
    file_content: Optional[bytes] = None,
    num_threads: Optional[int] = -1,
    use_coral: Optional[bool] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Represents external files used by the Task APIs (e.g. TF Lite FlatBuffer or
plain-text labels file). The files can be specified by one of the following
two ways:

(1) file contents loaded in `file_content`.
(2) file path in `file_name`.

If more than one field of these fields is provided, they are used in this
precedence order.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`file_name`<a id="file_name"></a>
</td>
<td>
Path to the index.
</td>
</tr><tr>
<td>
`file_content`<a id="file_content"></a>
</td>
<td>
The index file contents as bytes.
</td>
</tr><tr>
<td>
`num_threads`<a id="num_threads"></a>
</td>
<td>
Number of thread, the default value is -1 which means
Interpreter will decide what is the most appropriate `num_threads`.
</td>
</tr><tr>
<td>
`use_coral`<a id="use_coral"></a>
</td>
<td>
If true, inference will be delegated to a connected Coral Edge
TPU device.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/python/task/core/base_options.py#L72-L84">View source</a>

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
file_content<a id="file_content"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
file_name<a id="file_name"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
num_threads<a id="num_threads"></a>
</td>
<td>
`-1`
</td>
</tr><tr>
<td>
use_coral<a id="use_coral"></a>
</td>
<td>
`None`
</td>
</tr>
</table>
