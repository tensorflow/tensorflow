page_type: reference
description: Gets the path to the specified file in the data dependencies.

<link rel="stylesheet" href="/site-assets/css/style.css">

<!-- DO NOT EDIT! Automatically generated file. -->


<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tflite_support.metadata.get_path_to_datafile" />
<meta itemprop="path" content="Stable" />
</div>

# tflite_support.metadata.get_path_to_datafile

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tflite-support/blob/v0.4.4/tensorflow_lite_support/metadata/python/metadata.py#L75-L91">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the path to the specified file in the data dependencies.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tflite_support.metadata.get_path_to_datafile(
    path
)
</code></pre>



<!-- Placeholder for "Used in" -->

The path is relative to the file calling the function.

It's a simple replacement of
"tensorflow.python.platform.resource_loader.get_path_to_datafile".

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`path`<a id="path"></a>
</td>
<td>
a string resource path relative to the calling file.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The path to the specified file present in the data attribute of py_test
or py_binary.
</td>
</tr>

</table>
